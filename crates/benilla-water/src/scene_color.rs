//! The **scene colour the stylised water reflects** — one copy of the main texture, taken after the
//! opaque world has drawn and before anything transparent does.
//!
//! ## Why a copy, and why our own
//!
//! Screen-space reflection needs two things: where the reflected ray lands (depth: the pyramid
//! [`super::hiz`] builds) and what colour is there. The second cannot be read
//! from the frame being drawn — the water draws in the transparent phase, where the main colour
//! target is bound as an attachment and sampling it is undefined. So a snapshot has to be taken
//! while it is still readable.
//!
//! **Bevy has exactly this texture and we cannot have it.** `ViewTransmissionTexture` is a copy of
//! the main texture made for specular transmission, and `mesh_view_bindings` even exposes it at
//! binding 24 — but `core_3d` only prepares it when the **`Transmissive3d` phase has items**, and
//! liquid is alpha-blended, so it lands in `Transparent3d` and that phase is empty. Making the
//! water transmissive to borrow the texture would change which phase it draws in, how it is sorted
//! and how it blends, which is a great deal of blast radius for a texture we can copy ourselves in
//! one draw.
//!
//! So this is the same shape [`super::reflect`] already uses: an image asset the material binds,
//! rebuilt when the view resizes, filled by a render-graph node. The difference is that the mirror
//! *renders* the world into its target and this only *blits* the frame into one: a fullscreen
//! triangle that loads each texel into a 10-bit target, over the water's rectangle alone when no
//! reader looks outside it ([`WaterSnapshotScope`]).
//!
//! ## Where it sits in the frame
//!
//! After [`Node3d::MainTransmissivePass`] and before [`Node3d::MainTransparentPass`]: late enough
//! that the opaque world, the alpha-masked lane and the sky are all in it, early enough that no
//! transparent draw has happened yet. That ordering is the whole contract — the snapshot must not
//! contain the water, or the water would reflect itself.
//!
//! What it therefore cannot show is any *other* transparent surface: a second sheet of water, a
//! particle, a ribbon. That is the same exclusion the opaque depth makes and for the same reason,
//! and it is the right one — a reflection of a puff of smoke that is itself being blended over the
//! water is not a thing to chase.

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::FullscreenShader;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel,
};
use bevy::render::render_resource::binding_types::texture_2d;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    CachedRenderPipelineId, ColorTargetState, ColorWrites, Extent3d, FragmentState, LoadOp,
    Operations, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipelineDescriptor, ShaderStages, StoreOp, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureViewId,
};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::GpuImage;
use bevy::render::view::ViewTarget;
use bevy::render::{Render, RenderApp, RenderSystems};

use super::hiz::WaterHizConsumers;
use super::world_view::{
    serves_world_view, water_on_screen, Visit, WaterScreenRect, WaterWorldView,
};
use super::WaterStyle;

/// How much smaller than the main view the snapshot is, per side.
///
/// **One, and it is not an oversight.** The mirror can afford half resolution because it is sampled
/// through a distorting normal at a fraction of its strength; a screen-space march cannot, because
/// it is looking for a *hit* — it walks the depth buffer pixel by pixel, and a snapshot at half the
/// main view's density halves the distance it can resolve before two steps land in one texel. If
/// this ever needs to come down it is the march that has to be told, not just the texture.
const SNAPSHOT_SCALE: u32 = 1;

/// The snapshot's format: 10 bits per colour channel. The opaque frame it copies is display-encoded
/// in [0, 1] except fullbright magma and slime, which clamp; half the bytes of the main texture's
/// `Rgba16Float`.
const SNAPSHOT_FORMAT: TextureFormat = TextureFormat::Rgb10a2Unorm;

/// The colour snapshot, and the size it was built for.
#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct WaterSceneColor {
    /// The image the liquid materials sample. Rebuilt on a view resize, exactly as the mirror's
    /// target is, and re-pointed by [`restamp_scene_color`].
    pub(crate) image: Handle<Image>,
    /// Its current size in physical pixels.
    size: UVec2,
    /// Whether the blit should run at all — the stylised look only, like everything else here.
    armed: bool,
}

/// The render-graph node's label.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct WaterSceneColorLabel;

/// Where the snapshot is drawn this frame. The pass clears the whole target to transparent black
/// first, so outside a scissored rectangle the texels are black, and the rectangle is enough only
/// while no reader looks outside it: the composite reads the texel under each water fragment and
/// the refracted bed one at most `WATER_RECT_MARGIN_PX - 2` pixels off it, both inside the
/// rectangle's margin; the reflection march and the floor occlusion read the occluder wherever on
/// screen it is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum WaterSnapshotScope {
    /// The water's rectangle, inclusive min, exclusive max, in the target's pixels.
    Rect { min: UVec2, max: UVec2 },
    /// The whole target.
    Full,
}

impl WaterSnapshotScope {
    /// The readers that read the snapshot outside the water's rectangle.
    const OUTSIDE_READERS: u32 = WaterHizConsumers::SSR | WaterHizConsumers::FLOOR_OCCLUSION;

    /// The scope for this frame's pyramid readers and water rectangle on a `size` target.
    pub(crate) fn of(consumers: WaterHizConsumers, rect: &WaterScreenRect, size: UVec2) -> Self {
        if consumers.0 & Self::OUTSIDE_READERS != 0 {
            return Self::Full;
        }
        let (min, max) = (rect.min.min(size), rect.max.min(size));
        if min == UVec2::ZERO && max == size {
            Self::Full
        } else {
            Self::Rect { min, max }
        }
    }

    /// The scissor rectangle `(x, y, width, height)`, `None` for the whole target or an empty one.
    fn scissor(self) -> Option<(u32, u32, u32, u32)> {
        match self {
            Self::Full => None,
            Self::Rect { min, max } => {
                let size = max.saturating_sub(min);
                Some((min.x, min.y, size.x, size.y))
            }
        }
    }
}

/// A 10-bit image the size of the view, drawn into by the snapshot's blit. `ImageSampler::Default`
/// (linear, clamp) is the one the material's `scene_samp` and the LUT's mip reads use.
fn snapshot_image(size: UVec2) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        SNAPSHOT_FORMAT,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
    image
}

/// Keep the snapshot the size of the view, and armed only for the stylised look.
fn drive_scene_color(
    style: Res<WaterStyle>,
    mut snap: ResMut<WaterSceneColor>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<&Camera, With<crate::view::WorldCamera>>,
) {
    // Both stylised lanes read the snapshot: it is what the march samples the reflected colour
    // from, and the SSR lane is nothing BUT the march.
    snap.armed = style.is_stylised();
    let Ok(camera) = cameras.single() else {
        return;
    };
    let want = camera
        .physical_target_size()
        .map_or(snap.size, |s| (s / SNAPSHOT_SCALE).max(UVec2::splat(64)));
    if want != snap.size {
        snap.image = images.add(snapshot_image(want));
        snap.size = want;
    }
}

/// Re-point every liquid material at the snapshot after a rebuild. Change-gated on the handle, so
/// it is a no-op on every frame that is not a resize.
fn restamp_scene_color(
    snap: Res<WaterSceneColor>,
    mut materials: ResMut<Assets<benilla_assets::materials::LiquidMaterial>>,
    mut stamped: Local<Option<AssetId<Image>>>,
) {
    if *stamped == Some(snap.image.id()) {
        return;
    }
    *stamped = Some(snap.image.id());
    for (_, material) in materials.iter_mut() {
        material.extension.scene_color = snap.image.clone();
    }
}

/// The blit's pipeline: the fullscreen triangle and a fragment that loads the main texture's texel.
#[derive(Resource)]
struct SnapshotPipeline {
    layout: BindGroupLayoutDescriptor,
    id: CachedRenderPipelineId,
}

fn init_snapshot_pipeline(
    mut commands: Commands,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "water_snapshot_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::FRAGMENT,
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    );
    let id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("water_snapshot".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen_shader.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: asset_server.load("embedded://benilla_water/shaders/scene_snapshot.wgsl"),
            shader_defs: vec![],
            entry_point: Some("fs_snapshot".into()),
            targets: vec![Some(ColorTargetState {
                format: SNAPSHOT_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        ..default()
    });
    commands.insert_resource(SnapshotPipeline { layout, id });
}

/// The blit's bind group over the world view's main texture, and the view it was built for.
#[derive(Resource, Default)]
struct SnapshotBindings {
    key: Option<TextureViewId>,
    group: Option<BindGroup>,
}

/// Render world: (re)build the blit's bind group when the world view's main texture is not the one
/// it was built for; a steady frame builds none.
fn prepare_snapshot_bindings(
    snap: Option<Res<WaterSceneColor>>,
    pipe: Option<Res<SnapshotPipeline>>,
    cache: Res<PipelineCache>,
    views: Query<&ViewTarget, With<WaterWorldView>>,
    device: Res<RenderDevice>,
    mut bindings: ResMut<SnapshotBindings>,
) {
    let armed = snap.is_some_and(|s| s.armed);
    let (true, Some(pipe), Ok(target)) = (armed, pipe, views.single()) else {
        // A kept group would hold its main texture alive after the view's cache has dropped it.
        *bindings = SnapshotBindings::default();
        return;
    };
    let main = target.main_texture_view();
    if bindings.key == Some(main.id()) {
        return;
    }
    bindings.group = Some(device.create_bind_group(
        "water_snapshot_bind",
        &cache.get_bind_group_layout(&pipe.layout),
        &BindGroupEntries::single(main),
    ));
    bindings.key = Some(main.id());
    debug!("scene colour: the blit's bind group for {:?}", main.id());
}

/// The blit itself: one fullscreen draw per frame water is on screen, scissored to the water's
/// rectangle when [`WaterSnapshotScope`] allows, and nothing else.
#[derive(Default)]
struct WaterSceneColorNode;

impl Node for WaterSceneColorNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // Only while the look is armed and water is on screen, and only for the world view:
        // Core3d runs for the mirrors and the probe's cube faces too.
        let Some(snap) = world.get_resource::<WaterSceneColor>() else {
            return Ok(());
        };
        if !(snap.armed
            && water_on_screen(world)
            && serves_world_view(world, graph.view_entity(), Visit::Snapshot))
        {
            return Ok(());
        }
        let Some(gpu) = world
            .get_resource::<RenderAssets<GpuImage>>()
            .and_then(|images| images.get(&snap.image))
        else {
            return Ok(()); // the image has not reached the render world yet
        };
        let Some(target) = world.get::<ViewTarget>(graph.view_entity()) else {
            return Ok(());
        };
        // A frame where the view has resized but the rebuilt image has not yet arrived: the blit
        // would load texels the snapshot does not have.
        let main = target.main_texture();
        if main.width() != gpu.size.width || main.height() != gpu.size.height {
            return Ok(());
        }
        let (Some(pipe), Some(cache)) = (
            world.get_resource::<SnapshotPipeline>(),
            world.get_resource::<PipelineCache>(),
        ) else {
            return Ok(());
        };
        let Some(pipeline) = cache.get_render_pipeline(pipe.id) else {
            return Ok(()); // still compiling
        };
        // The prepared group when it binds this frame's main texture, else one made here.
        let main_view = target.main_texture_view();
        let prepared = world
            .get_resource::<SnapshotBindings>()
            .filter(|b| b.key == Some(main_view.id()))
            .and_then(|b| b.group.clone());
        let group = prepared.unwrap_or_else(|| {
            render_context.render_device().create_bind_group(
                "water_snapshot_bind",
                &cache.get_bind_group_layout(&pipe.layout),
                &BindGroupEntries::single(main_view),
            )
        });
        let rect = world
            .get_resource::<WaterScreenRect>()
            .copied()
            .unwrap_or_default();
        let size = UVec2::new(gpu.size.width, gpu.size.height);
        // No word from the readers is read as an outside reader: the whole target.
        let scope = world
            .get_resource::<WaterHizConsumers>()
            .map_or(WaterSnapshotScope::Full, |c| {
                WaterSnapshotScope::of(*c, &rect, size)
            });
        let scissor = scope.scissor();
        if scissor.is_some_and(|(_, _, w, h)| w == 0 || h == 0) {
            return Ok(());
        }
        // `render/water_snapshot` on Vulkan and DX12: the perf journal's `gpu_water`.
        let diagnostics = render_context.diagnostic_recorder();
        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("water_snapshot"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &gpu.texture_view,
                depth_slice: None,
                resolve_target: None,
                // Cleared, so a read outside a scissored rectangle shows black, never an old frame.
                ops: Operations {
                    load: LoadOp::Clear(LinearRgba::NONE.into()),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        let timed = diagnostics.pass_span(&mut pass, "water_snapshot");
        if let Some((x, y, w, h)) = scissor {
            pass.set_scissor_rect(x, y, w, h);
        }
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
        timed.end(&mut pass);
        Ok(())
    }
}

pub(super) fn register(app: &mut App) {
    let image = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(snapshot_image(UVec2::new(64, 64)));
    app.insert_resource(WaterSceneColor {
        image,
        size: UVec2::new(64, 64),
        armed: false,
    })
    .add_plugins(ExtractResourcePlugin::<WaterSceneColor>::default())
    .add_systems(
        PostUpdate,
        (drive_scene_color, restamp_scene_color)
            .chain()
            .before(bevy::transform::TransformSystems::Propagate)
            .before(bevy::camera::CameraUpdateSystems)
            .after(super::probe::drive_probe),
    );

    // After the transmissive pass and before the transparent one — see the module doc. The snapshot
    // must hold the opaque world and NOT the water.
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<SnapshotBindings>()
        .add_systems(bevy::render::RenderStartup, init_snapshot_pipeline)
        .add_systems(
            Render,
            prepare_snapshot_bindings.in_set(RenderSystems::PrepareBindGroups),
        )
        .add_render_graph_node::<WaterSceneColorNode>(Core3d, WaterSceneColorLabel)
        .add_render_graph_edges(
            Core3d,
            (
                Node3d::MainTransmissivePass,
                WaterSceneColorLabel,
                Node3d::MainTransparentPass,
            ),
        );
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::image::ImageSampler;

    fn rect(min: (u32, u32), max: (u32, u32)) -> WaterScreenRect {
        WaterScreenRect {
            visible: true,
            min: UVec2::new(min.0, min.1),
            max: UVec2::new(max.0, max.1),
        }
    }

    #[test]
    fn the_snapshot_is_scissored_only_while_nothing_reads_outside_the_water() {
        let size = UVec2::new(1600, 900);
        let r = rect((100, 400), (900, 900));
        let scissored = WaterSnapshotScope::Rect {
            min: UVec2::new(100, 400),
            max: UVec2::new(900, 900),
        };
        assert_eq!(
            WaterSnapshotScope::of(WaterHizConsumers(0), &r, size),
            scissored
        );
        // The glint's and the body's visibility read depth, not colour.
        let depth_only = WaterHizConsumers::SUN_VISIBILITY | WaterHizConsumers::BODY_VISIBILITY;
        assert_eq!(
            WaterSnapshotScope::of(WaterHizConsumers(depth_only), &r, size),
            scissored
        );
        for readers in [
            WaterHizConsumers::SSR,
            WaterHizConsumers::FLOOR_OCCLUSION,
            WaterHizConsumers::SSR | WaterHizConsumers::FLOOR_OCCLUSION,
        ] {
            assert_eq!(
                WaterSnapshotScope::of(WaterHizConsumers(readers), &r, size),
                WaterSnapshotScope::Full
            );
        }
        assert_eq!(scissored.scissor(), Some((100, 400, 800, 500)));
        assert_eq!(WaterSnapshotScope::Full.scissor(), None);
    }

    #[test]
    fn a_rect_over_the_whole_target_draws_unscissored_and_never_past_it() {
        let size = UVec2::new(1600, 900);
        let whole = rect((0, 0), (1600, 900));
        assert_eq!(
            WaterSnapshotScope::of(WaterHizConsumers(0), &whole, size),
            WaterSnapshotScope::Full
        );
        // A rectangle extracted for a larger target than the snapshot's is clamped to it.
        let past = rect((1500, 800), (2000, 1200));
        let scope = WaterSnapshotScope::of(WaterHizConsumers(0), &past, size);
        assert_eq!(scope.scissor(), Some((1500, 800, 100, 100)));
        let beside = rect((1700, 0), (2000, 900));
        let scope = WaterSnapshotScope::of(WaterHizConsumers(0), &beside, size);
        assert_eq!(scope.scissor().map(|s| (s.2, s.3)), Some((0, 900)));
    }

    /// The image the blit draws and the material samples: 10 bits a channel (the composite's
    /// half-step bound, test 9 in `liquid_composite`), a render target, and the default linear
    /// sampler the sky LUT's mip reads rely on.
    #[test]
    fn the_snapshot_is_a_ten_bit_target_with_the_default_sampler() {
        let image = snapshot_image(UVec2::new(320, 180));
        let d = &image.texture_descriptor;
        assert_eq!(d.format, TextureFormat::Rgb10a2Unorm);
        assert_eq!(
            d.usage,
            TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT
        );
        assert_eq!(image.sampler, ImageSampler::Default);
        assert_eq!((d.size.width, d.size.height), (320, 180));
    }

    /// The functions of a WGSL source that use `scene_tex` for anything but its size, in order.
    fn snapshot_readers(wgsl: &str) -> Vec<&str> {
        let mut readers = Vec::new();
        let mut current = "";
        for line in wgsl.lines() {
            if let Some(rest) = line.strip_prefix("fn ") {
                current = rest.split('(').next().unwrap_or(rest);
            }
            let code = line.split("//").next().unwrap_or(line);
            if code.trim_start().starts_with('@') {
                continue; // the binding's declaration
            }
            let reads = code.match_indices("scene_tex").any(|(at, _)| {
                let before = &code[..at];
                let whole = !before.ends_with(|c: char| c.is_alphanumeric() || c == '_')
                    && !code[at + "scene_tex".len()..]
                        .starts_with(|c: char| c.is_alphanumeric() || c == '_');
                whole && !before.ends_with("textureDimensions(")
            });
            if reads && readers.last() != Some(&current) {
                readers.push(current);
            }
        }
        readers
    }

    /// Every snapshot read is one the scope accounts for: the composite's texel and the refracted
    /// bed's lie inside the rectangle's margin, the floor march's box is read only under
    /// `FLOOR_OCCLUSION` and the reflection march's only under `SSR`. A new reader either stays
    /// inside the margin or joins `OUTSIDE_READERS`, or a scissored frame hands it black.
    #[test]
    fn every_snapshot_reader_is_one_the_scope_accounts_for() {
        let liquid = include_str!("../../benilla-assets/src/shaders/liquid.wgsl");
        assert_eq!(
            snapshot_readers(liquid),
            ["snapshot_box", "ssr_read", "water_snapshot_px", "water_bed"]
        );
        let compute = include_str!("shaders/water_compute.wgsl");
        assert_eq!(snapshot_readers(compute), ["snapshot_box"]);
    }
}
