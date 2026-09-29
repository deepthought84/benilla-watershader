//! **The reflection march's hierarchical depth buffer, and the water's linear depth.**
//!
//! A pyramid of nearest-per-tile depths, rebuilt from the world view's depth every frame the
//! stylised water is on. `shaders/hiz_reduce.wgsl` carries the reasoning for the pyramid and for
//! why it reduces with `max` where Bevy's own depth pyramid reduces with `min`; this module is the
//! plumbing that builds it.
//!
//! Its seed, `shaders/hiz_seed.wgsl`, is one compute pass that reads the view's depth once and
//! writes mip 0 of both pyramids and, inside the water's screen rectangle
//! ([`super::world_view::WaterScreenRect`]), the **linear depth**: view Z in yards behind every
//! pixel, full resolution, 0 where nothing was drawn. That is what the water measures its column
//! against; it is published as `WaterPassImages::scene_depth`.
//!
//! The shape is `scene_color`'s, because the problem is the same one: an image the liquid material
//! samples, sized to the view, rebuilt when the view resizes, re-pointed into every material, and
//! filled by a render-graph node. The differences are that this one carries a mip chain and is
//! written by passes rather than copied into.
//!
//! ## From the VIEW's depth, after the opaque pass — not from the prepass
//!
//! It is built **after the opaque and transmissive passes and before the transparent pass**, from
//! the view's own depth attachment: the only window where the depth of the opaque world is complete
//! and the water has not yet been drawn. It is also the window [`super::scene_color`] copies the
//! colour in, so the depth the march walks and the colour it reads describe the same frame.
//!
//! The depth prepass is not the opaque world: models on the entity path opt out of it
//! (`WowModelExt::enable_prepass` — characters, creatures, fading and refused doodads), the WDL
//! horizon opts out, and the static world is in it only through a pass of its own that draws
//! nothing until all of its pipelines have compiled. A depth missing something the colour holds
//! lets a ray walk straight through it and read its colour off the "sky" behind it, and gives the
//! water's column the bed behind a hull instead of the hull. AMD FidelityFX SSSR builds from the
//! opaque depth for the same reason, and Bevy's own occlusion culling binds the view depth exactly
//! as this does.
//!
//! The view depth is only bindable once `liquid::depth` has added `TEXTURE_BINDING` to the world
//! camera's `depth_texture_usages`, which takes a frame to reach the texture; until then the
//! pyramid is not rebuilt, and it holds the last frame's depths or zeros, which the march reads as
//! sky.

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::FullscreenShader;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel,
};
use bevy::render::render_resource::binding_types::{
    texture_2d, texture_depth_2d, texture_depth_2d_multisampled, texture_storage_2d, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedComputePipelineId,
    CachedRenderPipelineId, ColorTargetState, ColorWrites, ComputePassDescriptor,
    ComputePipelineDescriptor, Extent3d, FragmentState, Operations, PipelineCache,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages,
    ShaderType, StorageTextureAccess, TextureAspect, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::view::{ExtractedView, Msaa, ViewDepthTexture};
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::shader::ShaderDefVal;

use super::world_view::{serves_world_view, Visit, WaterScreenRect, WaterWorldView};
use super::{WaterPassImages, WaterStyle};

/// The pyramid and the linear depth, and the sizes they were built for.
#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct WaterHiz {
    /// The NEAREST-depth pyramid — the one the traversal reads on every step.
    pub(crate) image: Handle<Image>,
    /// The FARTHEST-depth pyramid, same size — read only on the steps where the ray is behind a
    /// tile's nearest surface. See [`hiz_image`] for why it is a second image and not a channel.
    pub(crate) far: Handle<Image>,
    /// The linear depth: view Z in yards behind each view pixel, 0 = unknown. See [`lin_image`].
    pub(crate) lin: Handle<Image>,
    /// Its base size in physical pixels — a power of two on both axes.
    size: UVec2,
    /// How many levels it carries.
    mips: u32,
    /// The linear depth's size: the view's, in physical pixels.
    lin_size: UVec2,
    /// Only the stylised looks march, so only they need this.
    pub(crate) armed: bool,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct WaterHizLabel;

/// The largest power of two not greater than `v`, floored at one.
fn prev_pot(v: u32) -> u32 {
    if v <= 1 {
        1
    } else {
        1 << (31 - (v.max(1)).leading_zeros())
    }
}

/// The pyramid's base for a view of `px`: the previous power of two per axis, at least 8.
fn pyramid_size(px: UVec2) -> UVec2 {
    UVec2::new(prev_pot(px.x), prev_pot(px.y)).max(UVec2::splat(8))
}

/// How many levels a pyramid of this size carries, down to a single texel.
fn mip_count(size: UVec2) -> u32 {
    32 - size.x.max(size.y).max(1).leading_zeros()
}

/// One pyramid level chain, `R32Float`: full-precision float, because it holds raw reverse-Z NDC
/// depth and the traversal compares it against the ray's own — a normalised or half-precision
/// format would quantise exactly the near-plane precision reverse-Z exists to provide.
///
/// There are two of these, the NEAREST surface per tile ([`WaterHiz::image`]) and the FARTHEST
/// ([`WaterHiz::far`]). **Why the farthest at all:** the nearest alone can prove a tile EMPTY in
/// front of a ray, which is all FFX SSSR asks of it; it cannot prove the ray passed BEHIND a tile, so
/// a ray going under a tree crown — which the water sees and the camera does not — met the crown's
/// front leaves 24+ yards in front of it and was thrown away: the black fragments in reflected
/// canopies. With each depth sample given a thickness (McGuire & Mara, "Efficient GPU Screen-Space
/// Ray Tracing", JCGT 2014), a ray behind the slab of a tile's farthest surface has passed behind
/// everything in it; min-max hierarchies (Hofmann et al., "Hierarchical Multi-Layer Screen-Space
/// Ray Tracing", HPG 2017) skip such tiles whole.
///
/// **Why a second image and not a second channel.** It was first an `Rg32Float` channel, and the
/// water's pass got 0.9 ms slower at Mirror Lake (GPU journal: transparent 4.05 → 4.94 ms, the
/// pyramid build itself unchanged) — the march reads the pyramid on every step, a 64-bit texel
/// fetches at half rate on this GPU, and most steps cross empty space in front of everything and
/// never needed the farthest at all. Apart, the nearest is read as before and the farthest only on
/// the steps where the ray is behind a tile's nearest surface.
fn hiz_image(size: UVec2, mips: u32) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &0f32.to_ne_bytes(),
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    // Mip 0 is a storage write of the seed's compute pass; every level below it a render pass.
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::STORAGE_BINDING
        | TextureUsages::RENDER_ATTACHMENT;
    image.texture_descriptor.mip_level_count = mips;
    // **The view has to span the chain.** Bevy's default view is built from this descriptor, and
    // without saying so the shader sees one level — `textureNumLevels` reports it, the traversal
    // never ascends, and every read below level zero is out of range.
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        label: Some("hiz_all_mips"),
        dimension: Some(TextureViewDimension::D2),
        base_mip_level: 0,
        mip_level_count: Some(mips),
        ..default()
    });
    // Every texel is written by a pass before anything reads it, and `new_fill` would otherwise
    // upload a base level that is immediately overwritten.
    image.data = None;
    image
}

/// The water's linear depth, `R32Float` at the view's own size: view Z in yards along the view axis
/// behind every pixel, written by the seed inside the water's rectangle. Created without data, which
/// wgpu zero-initialises, and 0 reads as unknown. `COPY_SRC` for `WOW_DEPTH`, which reads it back.
fn lin_image(size: UVec2) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC;
    image
}

/// Keep the pyramid and the linear depth the size of the view, and armed only for the looks that
/// march.
fn drive_hiz(
    style: Res<WaterStyle>,
    mut hiz: ResMut<WaterHiz>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<&Camera, With<crate::view::WorldCamera>>,
) {
    hiz.armed = style.is_stylised();
    let Ok(camera) = cameras.single() else {
        return;
    };
    let Some(px) = camera.physical_target_size() else {
        return;
    };
    // **Previous power of two, deliberately.** Every level is then exactly half the one above, so
    // the traversal's cell arithmetic is a shift and a tile's bounds are exact. The cost is that
    // the pyramid is slightly smaller than the view, which is harmless: it is addressed by UV and
    // a conservative nearest-depth is still conservative when its tiles are a little larger.
    let want = pyramid_size(px);
    if want != hiz.size {
        let mips = mip_count(want);
        hiz.image = images.add(hiz_image(want, mips));
        hiz.far = images.add(hiz_image(want, mips));
        hiz.size = want;
        hiz.mips = mips;
    }
    let lin = px.max(UVec2::ONE);
    if lin != hiz.lin_size {
        hiz.lin = images.add(lin_image(lin));
        hiz.lin_size = lin;
    }
}

/// Re-point every liquid material and the published pass images at the pyramid and the linear
/// depth after a rebuild — change-gated on the handles, so it is a no-op on every frame that is not
/// a resize.
fn restamp_hiz(
    hiz: Res<WaterHiz>,
    mut materials: ResMut<Assets<benilla_assets::materials::LiquidMaterial>>,
    pass: Option<ResMut<WaterPassImages>>,
    mut stamped: Local<Option<(AssetId<Image>, AssetId<Image>)>>,
) {
    let now = (hiz.image.id(), hiz.lin.id());
    if *stamped == Some(now) {
        return;
    }
    *stamped = Some(now);
    for (_, material) in materials.iter_mut() {
        material.extension.hiz = hiz.image.clone();
        material.extension.hiz_far = hiz.far.clone();
        material.extension.scene_depth = hiz.lin.clone();
    }
    if let Some(mut pass) = pass {
        pass.hiz = hiz.image.clone();
        pass.hiz_far = hiz.far.clone();
        pass.scene_depth = hiz.lin.clone();
    }
}

/// The seed's uniform, `hiz_seed.wgsl`'s `SeedParams`.
#[derive(Clone, Copy, Default, ShaderType)]
struct SeedParams {
    /// The view depth's size, physical pixels.
    src_size: UVec2,
    /// The pyramid's mip-0 size.
    dst_size: UVec2,
    /// The linear-depth write rectangle: inclusive min, exclusive max.
    lin_min: UVec2,
    lin_max: UVec2,
    /// The first pyramid texel of the dispatch.
    tile_origin: UVec2,
    /// [`SEED_WRITE_HIZ`] | [`SEED_ORTHOGRAPHIC`].
    write_hiz: u32,
    /// Bevy's infinite reverse-Z near plane, yards: `clip_from_view.w_axis.z`.
    near: f32,
}

/// `SeedParams::write_hiz` bit 0: write mip 0 of both pyramids.
const SEED_WRITE_HIZ: u32 = 1;
/// `SeedParams::write_hiz` bit 1: an orthographic view, whose depth is not `near / z`; the linear
/// depth is written 0.
const SEED_ORTHOGRAPHIC: u32 = 2;

/// Render world: the seed's parameters for this frame's world view, one persistent uniform buffer.
#[derive(Resource, Default)]
struct HizSeedParams {
    uniform: UniformBuffer<SeedParams>,
    /// Whether this frame's parameters were written: the world view exists and the pass is armed.
    ready: bool,
}

/// The pipelines: the compute seed, one per sample count of the view's depth, then the render
/// passes that halve the pyramid repeatedly.
#[derive(Resource)]
struct HizPipelines {
    seed_shader: Handle<Shader>,
    seed_layout: BindGroupLayoutDescriptor,
    seed_ms_layout: BindGroupLayoutDescriptor,
    /// `(sample count, pipeline)`: 1 queued at startup, a multisampled count the first frame the
    /// world view carries it.
    seeds: Vec<(u32, CachedComputePipelineId)>,
    reduce_layout: BindGroupLayoutDescriptor,
    reduce: CachedRenderPipelineId,
}

impl HizPipelines {
    fn seed(&self, samples: u32) -> Option<CachedComputePipelineId> {
        self.seeds
            .iter()
            .find(|(n, _)| *n == samples)
            .map(|(_, id)| *id)
    }

    /// The seed for `samples` per pixel: `SEED_MS` with the count as a constant, so its sample loop
    /// unrolls.
    fn queue_seed(&mut self, cache: &PipelineCache, samples: u32) {
        let multisampled = samples > 1;
        let id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("hiz_seed".into()),
            layout: vec![if multisampled {
                self.seed_ms_layout.clone()
            } else {
                self.seed_layout.clone()
            }],
            shader: self.seed_shader.clone(),
            shader_defs: if multisampled {
                vec![
                    "SEED_MS".into(),
                    ShaderDefVal::UInt("SEED_SAMPLES".into(), samples),
                ]
            } else {
                vec![]
            },
            entry_point: Some("cs_seed".into()),
            ..default()
        });
        self.seeds.push((samples, id));
    }
}

fn init_hiz_pipelines(
    mut commands: Commands,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    // **Each variant gets only the source binding it declares, at its own index**: the plain seed
    // reads binding 0, the multisampled one binding 1. No sampler at all: every read is a
    // `textureLoad`, every write a `textureStore`.
    let seed_entries = |source| {
        BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                source,
                (
                    2,
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                ),
                (
                    3,
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                ),
                (
                    4,
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                ),
                (5, uniform_buffer::<SeedParams>(false)),
            ),
        )
    };
    let seed_layout =
        BindGroupLayoutDescriptor::new("hiz_seed_layout", &seed_entries((0, texture_depth_2d())));
    let seed_ms_layout = BindGroupLayoutDescriptor::new(
        "hiz_seed_ms_layout",
        &seed_entries((1, texture_depth_2d_multisampled())),
    );
    let reduce_layout = BindGroupLayoutDescriptor::new(
        "hiz_reduce_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (
                    1,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
                (
                    3,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
            ),
        ),
    );
    // Two targets, one per pyramid: location 0 the nearest, location 1 the farthest.
    let one = || {
        Some(ColorTargetState {
            format: TextureFormat::R32Float,
            blend: None,
            write_mask: ColorWrites::RED,
        })
    };
    let reduce = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("hiz_reduce".into()),
        layout: vec![reduce_layout.clone()],
        vertex: fullscreen_shader.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: asset_server.load("embedded://benilla_water/shaders/hiz_reduce.wgsl"),
            shader_defs: vec![],
            entry_point: Some("fs_reduce".into()),
            targets: vec![one(), one()],
        }),
        ..default()
    });
    let mut pipes = HizPipelines {
        seed_shader: asset_server.load("embedded://benilla_water/shaders/hiz_seed.wgsl"),
        seed_layout,
        seed_ms_layout,
        seeds: Vec::new(),
        reduce_layout,
        reduce,
    };
    // The single-sampled seed now; `gxMultisample` is latched at launch, so a multisampled view's
    // count is queued on its first frame, before any water is drawn.
    pipes.queue_seed(&pipeline_cache, 1);
    commands.insert_resource(pipes);
}

/// Render world, before the graph: queue the seed for the world view's sample count if it is new,
/// and write this frame's `SeedParams` — the view's size and near plane, the pyramid's size, the
/// water's rectangle.
fn prepare_hiz_seed(
    pipes: Option<ResMut<HizPipelines>>,
    cache: Res<PipelineCache>,
    hiz: Option<Res<WaterHiz>>,
    rect: Option<Res<WaterScreenRect>>,
    views: Query<(&ExtractedView, &ExtractedCamera, &Msaa), With<WaterWorldView>>,
    mut params: ResMut<HizSeedParams>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    params.ready = false;
    let (Some(mut pipes), Ok((view, camera, msaa))) = (pipes, views.single()) else {
        return;
    };
    if pipes.seed(msaa.samples()).is_none() {
        pipes.queue_seed(&cache, msaa.samples());
    }
    let (Some(src), true) = (camera.physical_target_size, hiz.is_some_and(|h| h.armed)) else {
        return;
    };
    // Nothing on screen is water: an empty rectangle, and the seed writes no linear depth.
    let (lin_min, lin_max) = match rect.as_deref() {
        Some(r) if r.visible => (r.min.min(src), r.max.min(src)),
        _ => (UVec2::ZERO, UVec2::ZERO),
    };
    let orthographic = view.clip_from_view.w_axis.w != 0.0;
    params.uniform.set(SeedParams {
        src_size: src,
        dst_size: pyramid_size(src),
        lin_min,
        lin_max,
        tile_origin: UVec2::ZERO,
        write_hiz: SEED_WRITE_HIZ | if orthographic { SEED_ORTHOGRAPHIC } else { 0 },
        near: view.clip_from_view.w_axis.z,
    });
    params.uniform.write_buffer(&device, &queue);
    params.ready = true;
}

#[derive(Default)]
struct WaterHizNode;

impl Node for WaterHizNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // **Only the view the water marches in.** Core3d runs for every 3D view: the mirrors and
        // the probe's cube faces too.
        if !serves_world_view(world, graph.view_entity(), Visit::Pyramid) {
            return Ok(());
        }
        let Some(hiz) = world.get_resource::<WaterHiz>() else {
            return Ok(());
        };
        if !hiz.armed {
            return Ok(());
        }
        let (Some(pipes), Some(cache), Some(params)) = (
            world.get_resource::<HizPipelines>(),
            world.get_resource::<PipelineCache>(),
            world.get_resource::<HizSeedParams>(),
        ) else {
            return Ok(());
        };
        let Some(params_binding) = params.uniform.binding().filter(|_| params.ready) else {
            return Ok(());
        };
        let Some(images) = world.get_resource::<RenderAssets<GpuImage>>() else {
            return Ok(());
        };
        let (Some(gpu), Some(gpu_far), Some(gpu_lin)) = (
            images.get(&hiz.image),
            images.get(&hiz.far),
            images.get(&hiz.lin),
        ) else {
            return Ok(());
        };
        // **The view's own depth, complete after the opaque pass** — see the module doc. Not
        // bindable yet on the frame the style arms: that frame is skipped.
        let Some(view_depth) = world
            .get::<ViewDepthTexture>(graph.view_entity())
            .filter(|d| d.texture.usage().contains(TextureUsages::TEXTURE_BINDING))
        else {
            return Ok(());
        };
        let samples = view_depth.texture.sample_count();
        let (Some(seed_pipeline), Some(reduce_pipeline)) = (
            pipes
                .seed(samples)
                .and_then(|id| cache.get_compute_pipeline(id)),
            cache.get_render_pipeline(pipes.reduce),
        ) else {
            return Ok(()); // still compiling
        };
        // **Sizes as the parameters say, or skip the frame**: a resize reaches the view's depth,
        // the parameters and the rebuilt images on different frames, and the seed's integer
        // footprint is exact only for the sizes it was given.
        let p = params.uniform.get();
        let depth_size = view_depth.texture.size();
        let src = UVec2::new(depth_size.width, depth_size.height);
        if src != p.src_size
            || UVec2::new(gpu.size.width, gpu.size.height) != p.dst_size
            || UVec2::new(gpu_lin.size.width, gpu_lin.size.height) != src
        {
            return Ok(());
        }
        let source_view = view_depth.texture.create_view(&TextureViewDescriptor {
            label: Some("hiz_view_depth"),
            aspect: TextureAspect::DepthOnly,
            ..default()
        });

        let view_of = |texture: &bevy::render::render_resource::Texture, level: u32| {
            texture.create_view(&TextureViewDescriptor {
                label: Some("hiz_level"),
                dimension: Some(TextureViewDimension::D2),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..default()
            })
        };
        let level_view = |level: u32| {
            (
                view_of(&gpu.texture, level),
                view_of(&gpu_far.texture, level),
            )
        };

        // Mip 0 of both pyramids and the linear depth, one compute pass over the pyramid's texels.
        let dst0 = level_view(0);
        let (layout, source) = if samples > 1 {
            (&pipes.seed_ms_layout, 1)
        } else {
            (&pipes.seed_layout, 0)
        };
        let seed_bind = render_context.render_device().create_bind_group(
            "hiz_seed_bind",
            &cache.get_bind_group_layout(layout),
            &BindGroupEntries::with_indices((
                (source, &source_view),
                (2, &dst0.0),
                (3, &dst0.1),
                (4, &gpu_lin.texture_view),
                (5, params_binding),
            )),
        );
        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("hiz_seed"),
                        timestamp_writes: None,
                    });
            pass.set_pipeline(seed_pipeline);
            pass.set_bind_group(0, &seed_bind, &[]);
            pass.dispatch_workgroups(p.dst_size.x.div_ceil(8), p.dst_size.y.div_ceil(8), 1);
        }

        let pass = |ctx: &mut RenderContext,
                    views: &(
            bevy::render::render_resource::TextureView,
            bevy::render::render_resource::TextureView,
        ),
                    pipeline: &bevy::render::render_resource::RenderPipeline,
                    bind: &bevy::render::render_resource::BindGroup| {
            let attach = |view| {
                Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations::default(),
                })
            };
            let mut p = ctx.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("hiz_level"),
                color_attachments: &[attach(&views.0), attach(&views.1)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            p.set_render_pipeline(pipeline);
            p.set_bind_group(0, bind, &[]);
            p.draw(0..3, 0..1);
        };

        // **Top down, and it cannot be reordered**: each level reads the one above it.
        for level in 1..hiz.mips {
            let src = level_view(level - 1);
            let dst = level_view(level);
            let bind = render_context.render_device().create_bind_group(
                "hiz_reduce_bind",
                &cache.get_bind_group_layout(&pipes.reduce_layout),
                &BindGroupEntries::with_indices(((1, &src.0), (3, &src.1))),
            );
            pass(render_context, &dst, reduce_pipeline, &bind);
        }
        Ok(())
    }
}

pub(super) fn register(app: &mut App) {
    let size = UVec2::splat(8);
    let mips = mip_count(size);
    let (image, far, lin) = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        (
            images.add(hiz_image(size, mips)),
            images.add(hiz_image(size, mips)),
            images.add(lin_image(size)),
        )
    };
    app.insert_resource(WaterHiz {
        image,
        far,
        lin,
        size,
        mips,
        lin_size: size,
        armed: false,
    })
    .add_plugins(ExtractResourcePlugin::<WaterHiz>::default())
    .add_systems(
        PostUpdate,
        (drive_hiz, restamp_hiz)
            .chain()
            .before(bevy::transform::TransformSystems::Propagate)
            .before(bevy::camera::CameraUpdateSystems)
            .after(super::probe::drive_probe),
    );

    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };

    render_app
        .init_resource::<HizSeedParams>()
        .add_systems(bevy::render::RenderStartup, init_hiz_pipelines)
        .add_systems(
            Render,
            prepare_hiz_seed.in_set(RenderSystems::PrepareResources),
        )
        .add_render_graph_node::<WaterHizNode>(Core3d, WaterHizLabel)
        // After the opaque world has written the view's depth, before the water is drawn — the
        // same window `scene_color` copies the colour in.
        .add_render_graph_edges(Core3d, (Node3d::MainTransmissivePass, WaterHizLabel))
        .add_render_graph_edges(Core3d, (WaterHizLabel, Node3d::MainTransparentPass));
}

#[cfg(test)]
mod tests {
    use std::cmp::min;

    use super::*;

    /// The pyramid's levels have to reach a single texel, or the traversal's top level still
    /// covers more than one tile and the march can walk off the end of the hierarchy.
    #[test]
    fn the_pyramid_reaches_one_texel() {
        for (size, want) in [
            (UVec2::new(1024, 512), 11),
            (UVec2::new(2048, 1024), 12),
            (UVec2::new(8, 8), 4),
        ] {
            assert_eq!(mip_count(size), want, "levels for {size:?}");
            // The last level is one texel on the longest axis.
            assert_eq!(size.x.max(size.y) >> (want - 1), 1);
        }
    }

    /// The base is a power of two on both axes, which is what makes a level exactly half the one
    /// above and the traversal's cell arithmetic a shift.
    #[test]
    fn the_base_is_a_power_of_two() {
        for v in [1u32, 2, 3, 7, 8, 9, 1600, 1920, 2560] {
            let p = prev_pot(v);
            assert!(p.is_power_of_two(), "{v} -> {p}");
            assert!(p <= v.max(1), "{v} -> {p} grew");
            assert!(
                p * 2 > v || v < 2,
                "{v} -> {p} is not the NEAREST power below"
            );
        }
    }

    // `hiz_seed.wgsl`'s integer expressions, per axis, written exactly as the shader writes them
    // (less its `u` suffixes): `src` view texels onto `dst` pyramid texels.

    /// The first view texel pyramid texel `d` covers.
    fn footprint_lo(d: u32, src: u32, dst: u32) -> u32 {
        (d * src) / dst
    }

    /// The last view texel pyramid texel `d` covers, inclusive.
    #[allow(clippy::manual_div_ceil)] // the shader's expression, as it is written there
    fn footprint_hi(d: u32, src: u32, dst: u32) -> u32 {
        min(((d + 1) * src + dst - 1) / dst, src) - 1
    }

    /// The pyramid texel view texel `x`'s centre falls in.
    fn owner(x: u32, src: u32, dst: u32) -> u32 {
        ((2 * x + 1) * dst) / (2 * src)
    }

    /// Every view size per axis from 8 to 4096, with the pyramid the node builds for it.
    fn sizes() -> impl Iterator<Item = (u32, u32)> {
        (8..=4096u32).map(|src| (src, pyramid_size(UVec2::splat(src)).x))
    }

    /// Each view pixel has exactly one owner, and lies in its owner's footprint: so the seed's
    /// invocation for that texel loads it and is the one that writes its linear depth. Ownership
    /// is a function, so no pixel is written twice; lying in the footprint means none is missed.
    #[test]
    fn every_view_pixel_has_one_owner_inside_its_footprint() {
        for (src, dst) in sizes() {
            let mut owned = vec![0u32; dst as usize];
            for x in 0..src {
                let d = owner(x, src, dst);
                assert!(
                    d < dst,
                    "src {src}: pixel {x} owned by {d}, past the pyramid"
                );
                let (lo, hi) = (footprint_lo(d, src, dst), footprint_hi(d, src, dst));
                assert!(
                    lo <= x && x <= hi,
                    "src {src}: pixel {x} outside its owner {d}'s footprint {lo}..={hi}"
                );
                owned[d as usize] += 1;
            }
            // Every texel owns what the ratio gives it: one or two pixels, never zero.
            assert!(
                owned.iter().all(|&n| (1..=2).contains(&n)),
                "src {src}: a pyramid texel owns {:?} pixels",
                owned
                    .iter()
                    .filter(|&&n| !(1..=2).contains(&n))
                    .collect::<Vec<_>>()
            );
        }
    }

    /// The pyramid is the previous power of two, so a footprint spans at most three view texels
    /// and the seed's clamped 3x3 loads cover it; the footprints tile the view without a gap.
    #[test]
    fn a_footprint_spans_at_most_three_texels() {
        for (src, dst) in sizes() {
            let mut next = 0;
            for d in 0..dst {
                let (lo, hi) = (footprint_lo(d, src, dst), footprint_hi(d, src, dst));
                assert!(hi - lo <= 2, "src {src} texel {d}: {lo}..={hi}");
                assert!(
                    lo <= next && hi < src,
                    "src {src} texel {d}: gap before {lo}"
                );
                next = next.max(hi + 1);
            }
            assert_eq!(
                next, src,
                "src {src}: the footprints stop short of the view"
            );
        }
    }

    /// The integer footprint is the one the fragment seed it replaced computed, `seed_rect`, in
    /// exact arithmetic: `lo = floor(d·src/dst)`, `hi = ceil((d+1)·src/dst) − 1`, clamped to the
    /// view. That seed took its texel size from the derivative of an interpolated UV, and a UV an
    /// ulp off flipped a boundary that falls exactly on a view texel edge; those texels are counted
    /// here, and they are where a pyramid built by the two seeds can differ.
    #[test]
    fn the_seed_footprint_is_the_fragment_seeds() {
        let mut on_edge = 0u64;
        let mut texels = 0u64;
        for (src, dst) in sizes() {
            for d in 0..dst {
                let (s, t, k) = (f64::from(src), f64::from(dst), f64::from(d));
                let lo = ((k * s / t).floor() as u32).min(src - 1);
                let hi = (((k + 1.0) * s / t).ceil() as u32).clamp(lo + 1, src) - 1;
                assert_eq!(
                    (footprint_lo(d, src, dst), footprint_hi(d, src, dst)),
                    (lo, hi),
                    "src {src} dst {dst} texel {d}"
                );
                texels += 1;
                if (d * src) % dst == 0 || ((d + 1) * src) % dst == 0 {
                    on_edge += 1;
                }
            }
        }
        eprintln!(
            "{on_edge} of {texels} pyramid texels have a footprint edge exactly on a view texel \
             edge, where the fragment seed's interpolated UV could round either way"
        );
    }

    /// The Rust expressions above are the shader's, character for character once WGSL's `u`
    /// suffixes are dropped: a change to one side fails here.
    #[test]
    fn the_seed_arithmetic_is_the_shaders() {
        let wgsl = include_str!("shaders/hiz_seed.wgsl");
        let mut bare = String::with_capacity(wgsl.len());
        let mut prev = ' ';
        for c in wgsl.chars() {
            if !(c == 'u' && prev.is_ascii_digit()) {
                bare.push(c);
            }
            prev = c;
        }
        let rust = include_str!("hiz.rs");
        for expr in [
            "(d * src) / dst",
            "min(((d + 1) * src + dst - 1) / dst, src) - 1",
            "((2 * x + 1) * dst) / (2 * src)",
        ] {
            assert!(bare.contains(expr), "hiz_seed.wgsl lost `{expr}`");
            // Once in its mirror function and once in this list.
            assert!(
                rust.matches(expr).count() >= 2,
                "the CPU mirror lost `{expr}`"
            );
        }
        // And the linearisation: `near / d`, 0 kept for the cleared far plane.
        assert!(bare.contains("select(0.0, p.near / d, d > 0.0"));
    }
}
