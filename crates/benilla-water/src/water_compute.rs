//! **The water's compute pass**: one compute pass on the world view, after the depth pyramids
//! ([`super::hiz`]) and the scene snapshot ([`super::scene_color`]), before the water draws.
//! Its shader is `shaders/water_compute.wgsl`.
//!
//! First the sky floor's cube ([`super::sky_lut`]), on every frame water on screen can show the sky
//! floor ([`super::floor_share`]), its occlusion on or off.
//!
//! Then `occl_plane`, the sky floor's occlusion for the water on an active mirror plane. The water's
//! fragment marches the depth pyramids per pixel to find what stands above it along the reflected
//! ray; on a mirror plane every pixel's unrippled point lies on the plane, so the same march runs
//! here once per quarter-resolution texel (an eighth on Low) over the water's rectangle, and the
//! fragment reads the result bilinearly ([`FloorOcc`]). The march's text is the floor march law,
//! byte-identical in both shaders (a test compares it).
//!
//! **Built once, bound every frame.** The pass's bind group lives in [`ComputeBindings`], keyed on
//! the id of every buffer and view it binds. The view uniform buffer is among them: Bevy
//! reallocates it when the number of views grows (the second mirror, the probe's cube faces, a
//! portrait), and a group holding the old one reads a buffer nothing writes any more. The world
//! view's uniform offset is passed on every dispatch.

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel,
};
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, texture_2d, texture_storage_2d, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BufferId,
    CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, Extent3d,
    PipelineCache, SamplerBindingType, SamplerId, ShaderStages, ShaderType, StorageTextureAccess,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureViewId,
    UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::view::{ViewUniform, ViewUniformOffset, ViewUniforms};
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::shader::ShaderDefVal;

use benilla_world::liquid::{WaterQuality, MIRROR2_LANES};

use super::floor_share::FloorOnScreen;
use super::hiz::{WaterHiz, WaterHizLabel};
use super::look::WaterLook;
use super::reflect::{WaterReflectBuffer, WaterReflectData};
use super::scene_color::{WaterSceneColor, WaterSceneColorLabel};
use super::world_view::{WaterScreenRect, WaterWorldView};
use super::{WaterPassImages, WaterStyle};

/// The view pixels per occlusion texel along each axis, by tier: 8 on Low (an eighth of the
/// resolution), 4 above it (a quarter).
fn occl_scale(tier: u8) -> u32 {
    if tier == 0 {
        8
    } else {
        4
    }
}

/// `a / b` per axis, rounded up.
fn ceil_div(a: UVec2, b: u32) -> UVec2 {
    UVec2::new(a.x.div_ceil(b), a.y.div_ceil(b))
}

/// The sky floor's occlusion over the mirror planes, and the size it was built for; extracted.
#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct FloorOcc {
    /// `Rgba8Unorm`: r the open sky v, g and b the body's sky and sun visibility, a 1 where the
    /// march reached a verdict. Bound at 120.
    pub(crate) occl: Handle<Image>,
    /// `Rgba16Float`: rgb the occluder's radiance times `1 − v`, linear; a `rho_c`. Bound at 121.
    pub(crate) col: Handle<Image>,
    /// Their size in texels.
    size: UVec2,
    /// The quality tier the pass compiles for, 0 Low, 1 Medium, 2 High.
    tier: u8,
    /// The stylised look with the floor occlusion on.
    armed: bool,
}

/// An occlusion image pair of `size` texels, filled with the neutral answer (the sky open, the
/// verdict unknown, no occluder), which is what the water reads before the first dispatch.
fn occl_images(size: UVec2) -> (Image, Image) {
    let make = |format: TextureFormat, texel: &[u8]| {
        let mut image = Image::new_fill(
            Extent3d {
                width: size.x.max(1),
                height: size.y.max(1),
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            texel,
            format,
            RenderAssetUsages::RENDER_WORLD,
        );
        image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_DST;
        image
    };
    (
        make(TextureFormat::Rgba8Unorm, &[255, 255, 255, 0]),
        make(TextureFormat::Rgba16Float, &[0; 8]),
    )
}

/// Keep the occlusion images at the view's size over the tier's scale, and armed only for the
/// stylised look with the floor occlusion on.
fn drive_floor_occ(
    style: Res<WaterStyle>,
    quality: Res<WaterQuality>,
    look: Res<WaterLook>,
    mut occ: ResMut<FloorOcc>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<&Camera, With<crate::view::WorldCamera>>,
) {
    occ.armed = style.is_stylised() && look.floor_occlusion;
    occ.tier = quality.tier();
    let Ok(camera) = cameras.single() else {
        return;
    };
    let Some(px) = camera.physical_target_size() else {
        return;
    };
    let want = ceil_div(px, occl_scale(occ.tier)).max(UVec2::ONE);
    if want != occ.size {
        let (occl, col) = occl_images(want);
        occ.occl = images.add(occl);
        occ.col = images.add(col);
        occ.size = want;
    }
}

/// Re-point every liquid material and the published pass images at the occlusion images after a
/// rebuild; change-gated on the handles, so a no-op on every frame that is not a resize.
fn restamp_floor_occ(
    occ: Res<FloorOcc>,
    mut materials: ResMut<Assets<benilla_assets::materials::LiquidMaterial>>,
    pass: Option<ResMut<WaterPassImages>>,
    mut stamped: Local<Option<(AssetId<Image>, AssetId<Image>)>>,
) {
    let now = (occ.occl.id(), occ.col.id());
    if *stamped == Some(now) {
        return;
    }
    *stamped = Some(now);
    for (_, material) in materials.iter_mut() {
        material.extension.water_occl = occ.occl.clone();
        material.extension.floor_occ = occ.col.clone();
    }
    if let Some(mut pass) = pass {
        pass.water_occl = occ.occl.clone();
        pass.floor_occ = occ.col.clone();
    }
}

/// `water_compute.wgsl`'s `OcclParams`.
#[derive(Clone, Copy, Default, ShaderType)]
struct OcclParams {
    /// The first texel of the dispatch and the end of its range (exclusive).
    tile_origin: UVec2,
    tile_end: UVec2,
    /// The lighting sun's travel direction; written 0 and read by no entry point.
    sun_travel: Vec4,
}

/// The quarter (or eighth) texels covering the view pixels `min..max` (exclusive) at `scale`
/// pixels a texel: the first texel and the end of the range. The water lies inside its boxes and
/// the rectangle's margin is several texels, so the fragment's bilinear read never reaches a texel
/// outside it.
fn occl_tiles(min: UVec2, max: UVec2, scale: u32, size: UVec2) -> (UVec2, UVec2) {
    ((min / scale).min(size), ceil_div(max, scale).min(size))
}

/// The reflection block's row of the mirror planes: `params`, then `mirror2`.
const PARAMS_ROW: u32 = 0;
const MIRROR2_ROW: u32 = (MIRROR2_LANES.start / 4) as u32;

/// The pass's pipelines: `occl_plane` per tier.
#[derive(Resource)]
struct ComputePipelines {
    layout: BindGroupLayoutDescriptor,
    occl: [CachedComputePipelineId; 3],
}

fn init_compute_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
) {
    let float = |filterable| texture_2d(TextureSampleType::Float { filterable });
    let layout = BindGroupLayoutDescriptor::new(
        "water_compute_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<ViewUniform>(true),
                float(false),
                float(false),
                float(false),
                float(true),
                sampler(SamplerBindingType::Filtering),
                storage_buffer_read_only_sized(false, None),
                texture_storage_2d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<OcclParams>(false),
            ),
        ),
    );
    let shader = asset_server.load("embedded://benilla_water/shaders/water_compute.wgsl");
    let occl = std::array::from_fn(|tier| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("water_occl_plane".into()),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            shader_defs: vec![
                benilla_assets::materials::LiquidKey::TIER_DEFS[tier].into(),
                ShaderDefVal::UInt("PARAMS_ROW".into(), PARAMS_ROW),
                ShaderDefVal::UInt("MIRROR2_ROW".into(), MIRROR2_ROW),
            ],
            entry_point: Some("occl_plane".into()),
            ..default()
        })
    });
    commands.insert_resource(ComputePipelines { layout, occl });
}

/// Render world: this frame's dispatches, written by [`prepare_water_compute`].
#[derive(Resource, Default)]
pub(crate) struct ComputeFrame {
    /// The sky floor's cube is rebuilt this frame: the stylised look, water on screen that can show
    /// the floor, occluded or not.
    pub(crate) lut: bool,
    params: UniformBuffer<OcclParams>,
    /// The occlusion dispatch runs this frame: armed, the floor on screen and occluded, a mirror
    /// plane active.
    occl: bool,
    /// Its workgroups.
    groups: UVec2,
    /// The tier's pipeline.
    tier: u8,
}

/// Whether the sky floor's cube and the plane occlusion run this frame: the cube while the stylised
/// look is armed and water on screen can show the floor, whatever its occlusion (every stylised
/// fragment reads the cube for the floor's share); the occlusion while that floor is occluded too,
/// the pass armed and a mirror plane active.
fn compute_gates(
    look_armed: bool,
    occl_armed: bool,
    water: bool,
    floor: FloorOnScreen,
    plane: bool,
) -> (bool, bool) {
    let lut = look_armed && water && floor.sky;
    (lut, occl_armed && water && floor.occluded && plane)
}

/// Render world, before the graph: which dispatches run this frame ([`compute_gates`]). The
/// occlusion runs over the rectangle's texels: off the planes the water marches per pixel and
/// never reads the images.
fn prepare_water_compute(
    hiz: Option<Res<WaterHiz>>,
    occ: Option<Res<FloorOcc>>,
    rect: Option<Res<WaterScreenRect>>,
    floor: Option<Res<FloorOnScreen>>,
    data: Option<Res<WaterReflectData>>,
    mut frame: ResMut<ComputeFrame>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    frame.occl = false;
    frame.lut = false;
    let (Some(occ), Some(rect), Some(data)) = (occ, rect, data) else {
        return;
    };
    let plane = |row: u32| data.0[row as usize * 4 + 1] > 0.0;
    let (lut, occl) = compute_gates(
        hiz.is_some_and(|h| h.armed),
        occ.armed,
        rect.visible,
        floor.map_or_else(FloorOnScreen::default, |f| *f),
        plane(PARAMS_ROW) || plane(MIRROR2_ROW),
    );
    frame.lut = lut;
    if !occl {
        return;
    }
    let (origin, end) = occl_tiles(rect.min, rect.max, occl_scale(occ.tier), occ.size);
    if origin.cmpge(end).any() {
        return;
    }
    frame.params.set(OcclParams {
        tile_origin: origin,
        tile_end: end,
        sun_travel: Vec4::ZERO,
    });
    frame.params.write_buffer(&device, &queue);
    frame.groups = ceil_div(end - origin, 8);
    frame.tier = occ.tier.min(2);
    frame.occl = true;
}

/// Every resource the pass's bind group binds, by id.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ComputeKey {
    view_uniforms: BufferId,
    hiz: TextureViewId,
    hiz_far: TextureViewId,
    lin: TextureViewId,
    snapshot: TextureViewId,
    sampler: SamplerId,
    lanes: BufferId,
    occl: TextureViewId,
    col: TextureViewId,
    params: BufferId,
}

/// Render world: the pass's bind group and what it was built for.
#[derive(Resource, Default)]
struct ComputeBindings {
    key: Option<ComputeKey>,
    group: Option<BindGroup>,
}

/// Render world, after the parameters: (re)build the bind group only where a buffer or view it
/// binds is not the one it was built for. `RUST_LOG=benilla_water::water_compute=debug` logs each
/// build.
#[allow(clippy::too_many_arguments)]
fn prepare_compute_bindings(
    frame: Res<ComputeFrame>,
    occ: Option<Res<FloorOcc>>,
    hiz: Option<Res<WaterHiz>>,
    snap: Option<Res<WaterSceneColor>>,
    lanes: Option<Res<WaterReflectBuffer>>,
    view_uniforms: Res<ViewUniforms>,
    pipes: Option<Res<ComputePipelines>>,
    cache: Res<PipelineCache>,
    images: Res<RenderAssets<GpuImage>>,
    device: Res<RenderDevice>,
    mut bindings: ResMut<ComputeBindings>,
) {
    if !frame.occl {
        return;
    }
    let (Some(occ), Some(hiz), Some(snap), Some(lanes), Some(pipes)) =
        (occ, hiz, snap, lanes, pipes)
    else {
        return;
    };
    let get = |h: &Handle<Image>| images.get(h);
    let (Some(near), Some(far), Some(lin), Some(snapshot), Some(occl), Some(col)) = (
        get(&hiz.image),
        get(&hiz.far),
        get(&hiz.lin),
        get(&snap.image),
        get(&occ.occl),
        get(&occ.col),
    ) else {
        return;
    };
    let (Some(view_buffer), Some(view_binding)) = (
        view_uniforms.uniforms.buffer(),
        view_uniforms.uniforms.binding(),
    ) else {
        return;
    };
    let (Some(params), Some(params_binding)) = (frame.params.buffer(), frame.params.binding())
    else {
        return;
    };
    let key = ComputeKey {
        view_uniforms: view_buffer.id(),
        hiz: near.texture_view.id(),
        hiz_far: far.texture_view.id(),
        lin: lin.texture_view.id(),
        snapshot: snapshot.texture_view.id(),
        sampler: snapshot.sampler.id(),
        lanes: lanes.0.id(),
        occl: occl.texture_view.id(),
        col: col.texture_view.id(),
        params: params.id(),
    };
    if bindings.key == Some(key) {
        return;
    }
    bindings.group = Some(device.create_bind_group(
        "water_compute_bind",
        &cache.get_bind_group_layout(&pipes.layout),
        &BindGroupEntries::sequential((
            view_binding,
            &near.texture_view,
            &far.texture_view,
            &lin.texture_view,
            &snapshot.texture_view,
            &snapshot.sampler,
            lanes.0.as_entire_binding(),
            &occl.texture_view,
            &col.texture_view,
            params_binding,
        )),
    ));
    debug!("water_compute: the bind group for {key:?}");
    bindings.key = Some(key);
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct WaterComputeLabel;

#[derive(Default)]
struct WaterComputeNode;

impl Node for WaterComputeNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // Only for the world view: Core3d runs for the mirrors and the probe's cube faces too.
        let view = graph.view_entity();
        if world.get::<WaterWorldView>(view).is_none() {
            return Ok(());
        }
        let (Some(frame), Some(pipes), Some(cache), Some(bindings)) = (
            world.get_resource::<ComputeFrame>(),
            world.get_resource::<ComputePipelines>(),
            world.get_resource::<PipelineCache>(),
            world.get_resource::<ComputeBindings>(),
        ) else {
            return Ok(());
        };
        if !(frame.lut || frame.occl) {
            return Ok(());
        }
        let occl = (
            world.get::<ViewUniformOffset>(view),
            bindings.group.as_ref(),
            cache.get_compute_pipeline(pipes.occl[usize::from(frame.tier)]),
        );
        // `render/water_compute` on Vulkan and DX12: the perf journal's `gpu_water`.
        let diagnostics = render_context.diagnostic_recorder();
        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("water_compute"),
                    timestamp_writes: None,
                });
        let timed = diagnostics.pass_span(&mut pass, "water_compute");
        // The cube first: nothing in this pass reads it, the water after it does.
        if frame.lut {
            super::sky_lut::dispatch_lut(&mut pass, world);
        }
        // Compiling, or the bind group not built yet: the images keep their last answer.
        if let (true, (Some(offset), Some(group), Some(pipeline))) = (frame.occl, occl) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[offset.offset]);
            pass.dispatch_workgroups(frame.groups.x, frame.groups.y, 1);
        }
        timed.end(&mut pass);
        Ok(())
    }
}

pub(super) fn register(app: &mut App) {
    let size = UVec2::ONE;
    let (occl, col) = {
        let (occl, col) = occl_images(size);
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        (images.add(occl), images.add(col))
    };
    app.insert_resource(FloorOcc {
        occl,
        col,
        size,
        tier: 1,
        armed: false,
    })
    .add_plugins(ExtractResourcePlugin::<FloorOcc>::default())
    .add_systems(
        PostUpdate,
        (drive_floor_occ, restamp_floor_occ)
            .chain()
            .before(bevy::transform::TransformSystems::Propagate)
            .before(bevy::camera::CameraUpdateSystems)
            .after(super::probe::drive_probe)
            // Both restamps write the published pass images; one declared order between them.
            .after(super::hiz::restamp_hiz),
    );

    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<ComputeFrame>()
        .init_resource::<ComputeBindings>()
        .add_systems(bevy::render::RenderStartup, init_compute_pipelines)
        .add_systems(
            Render,
            (
                prepare_water_compute.in_set(RenderSystems::PrepareResources),
                prepare_compute_bindings.in_set(RenderSystems::PrepareBindGroups),
            ),
        )
        .add_render_graph_node::<WaterComputeNode>(Core3d, WaterComputeLabel)
        // After both the pyramids and the snapshot it reads, before the water reads its images.
        .add_render_graph_edges(Core3d, (WaterHizLabel, WaterComputeLabel))
        .add_render_graph_edges(Core3d, (WaterSceneColorLabel, WaterComputeLabel))
        .add_render_graph_edges(Core3d, (WaterComputeLabel, Node3d::MainTransparentPass));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sky floor's cube is built wherever the floor shows, its occlusion on or off (a WMO
    /// interior pool's floor is shown and never occluded); the plane pass only where it is
    /// occluded on a mirror plane.
    #[test]
    fn the_sky_cube_is_built_wherever_the_floor_shows() {
        let floor = |sky, occluded| FloorOnScreen { sky, occluded };
        // (look armed, occlusion armed, water, floor, plane) -> (cube, plane pass)
        for (args, want) in [
            ((true, true, true, floor(true, true), true), (true, true)),
            ((true, false, true, floor(true, false), true), (true, false)),
            ((true, true, true, floor(true, false), true), (true, false)),
            ((true, true, true, floor(true, true), false), (true, false)),
            (
                (true, true, true, floor(false, false), true),
                (false, false),
            ),
            ((true, true, false, floor(true, true), true), (false, false)),
        ] {
            let (look, occl, water, f, plane) = args;
            assert_eq!(compute_gates(look, occl, water, f, plane), want, "{args:?}");
        }
    }

    /// The text between the floor-march-law markers of one shader.
    fn march_law(wgsl: &str) -> &str {
        let begin = wgsl
            .find("// ---- floor march law: begin ----")
            .expect("the floor march law's begin marker");
        let end = wgsl[begin..]
            .find("// ---- floor march law: end ----")
            .map(|e| begin + e)
            .expect("the floor march law's end marker");
        &wgsl[begin..end]
    }

    /// **The plane water's occlusion and the per-pixel one are one law.** The compute pass marches
    /// for the water on a mirror plane and the fragment for the water off it, and the two meet in
    /// a crossfade band: any difference between the copies is a seam there. The block is
    /// duplicated rather than imported (the material shader and the compute pass are composed
    /// apart), so this keeps them from drifting; the tiers' level and step budgets are part of it.
    #[test]
    fn the_floor_march_law_is_shared_verbatim() {
        let liquid = include_str!("../../benilla-assets/src/shaders/liquid.wgsl");
        let compute = include_str!("shaders/water_compute.wgsl");
        assert_eq!(
            march_law(liquid),
            march_law(compute),
            "liquid.wgsl and water_compute.wgsl carry different floor march laws — copy the \
             edited one across"
        );
        assert!(march_law(compute).contains("fn floor_march("));
        // One budget per tier, inside the shared text.
        assert!(march_law(compute).contains("const FLOOR_MARCH_STEPS: i32 = 8;"));
        // The law's colour decode is each file's own: the same gamma.
        for wgsl in [liquid, compute] {
            assert!(wgsl.contains("const WATER_GAMMA: f32 = 2.2;"));
        }
    }

    /// The floor march law's `law_cover`, as the shared text carries it.
    const LAW_COVER_WGSL: &str = "\
fn law_cover(start: f32, end: f32, depth: f32, ray_q: f32, frame_q: f32) -> f32 {
    if (end <= 0.0 || depth <= end) {
        return 0.0;
    }
    let c = 1.0 - exp(-(law_view_z(depth) - law_view_z(end)) / LAW_THICKNESS);
    if (depth < start) {
        return c;
    }
    if (ray_q <= 0.0 || frame_q <= 0.0) {
        return 0.0;
    }
    return c * (1.0 - saturate((law_view_z(ray_q) - law_view_z(frame_q)) / LAW_THICKNESS));
}";

    /// [`LAW_COVER_WGSL`] in Rust, with `law_view_z` = −near / depth and `LAW_THICKNESS` 6 yd.
    fn law_cover(start: f32, end: f32, depth: f32, ray_q: f32, frame_q: f32, near: f32) -> f32 {
        let view_z = |d: f32| -near / d;
        if end <= 0.0 || depth <= end {
            return 0.0;
        }
        let c = 1.0 - (-(view_z(depth) - view_z(end)) / 6.0).exp();
        if depth < start {
            return c;
        }
        if ray_q <= 0.0 || frame_q <= 0.0 {
            return 0.0;
        }
        c * (1.0 - ((view_z(ray_q) - view_z(frame_q)) / 6.0).clamp(0.0, 1.0))
    }

    /// **Geometry nearer the eye than the water covers its ray only where the ray is shut.** An
    /// unknown verdict reads the frame where the ray's path on screen ends. What the frame shows
    /// there beyond the water point covers the ray. What stands nearer the eye than the water (the
    /// avatar in third person, a trunk beside the camera, a deck overhead) is not on the ray: it
    /// covers it only where the ray's own point at its height is on or behind what the frame
    /// shows there, a deck that reaches out over the water, and not where that point floats in
    /// front of the far shore, as at the avatar's height.
    #[test]
    fn near_geometry_covers_a_ray_only_where_the_ray_is_shut() {
        let liquid = include_str!("../../benilla-assets/src/shaders/liquid.wgsl");
        assert!(
            march_law(liquid).contains(LAW_COVER_WGSL),
            "the law's law_cover is not the text this test mirrors"
        );
        assert!(march_law(liquid).contains("cover = law_cover(o.z, e.z, e_depth, ray_q, frame_q);"));
        // Reverse-Z depth is near / distance; near 0.1 yd.
        let near = 0.1;
        let at = |yd: f32| near / yd;
        let cover =
            |start, end, depth, ray_q, frame_q| law_cover(start, end, depth, ray_q, frame_q, near);
        // Water 40 yd out, its ray's path on screen ending at the far plane's shy point.
        let (water, end) = (at(40.0), at(40_000.0));
        // A canopy 60 yd out covers it, whatever the frame holds at the ray's point.
        assert!(cover(water, end, at(60.0), 0.0, 0.0) > 0.999);
        assert!(cover(water, end, at(41.0), 0.0, 0.0) > 0.999);
        // The avatar 4 yd from the eye, a trunk 12 yd: the ray's point at their height lies 70 yd
        // out, over the far shore 300 yd away. No cover; nor off the frame.
        for near_thing in [at(4.0), at(12.0)] {
            assert_eq!(cover(water, end, near_thing, at(70.0), at(300.0)), 0.0);
            assert_eq!(cover(water, end, near_thing, 0.0, 0.0), 0.0);
        }
        // A deck overhead, 3 yd from the eye: the ray's point at its height is on the deck 25 yd
        // out (the frame shows the deck there, half a yard behind it), or behind a beam in front
        // of it.
        assert!(cover(water, end, at(3.0), at(25.0), at(25.5)) > 0.9);
        assert!(cover(water, end, at(3.0), at(25.0), at(10.0)) > 0.999);
        // Softly by how far the point floats in front of what the frame shows: 3 yd, half.
        let c = cover(water, end, at(3.0), at(25.0), at(28.0));
        assert!((c - 0.5).abs() < 1e-3, "{c}");
        // Sky, and geometry behind the ray's end, do not cover it; nor does an end past the far
        // plane.
        assert_eq!(cover(water, end, 0.0, 0.0, 0.0), 0.0);
        assert_eq!(cover(water, at(50.0), at(55.0), 0.0, 0.0), 0.0);
        assert_eq!(cover(water, 0.0, at(60.0), 0.0, 0.0), 0.0);
        // Softly by the gap: 6 yd in front of the end is 1 − 1/e.
        let c = cover(at(10.0), at(30.0), at(24.0), 0.0, 0.0);
        assert!((c - (1.0 - (-1.0f32).exp())).abs() < 1e-4, "{c}");
    }

    /// The dispatch covers the texels under the rectangle's pixels, and no more than the image.
    #[test]
    fn the_occlusion_dispatch_covers_the_rectangle() {
        let size = UVec2::new(800, 450);
        assert_eq!(
            occl_tiles(UVec2::new(30, 100), UVec2::new(3200, 1800), 4, size),
            (UVec2::new(7, 25), UVec2::new(800, 450))
        );
        // A rectangle's last pixel 4k + 1 is in texel k, so the range ends at k + 1.
        assert_eq!(
            occl_tiles(UVec2::new(0, 0), UVec2::new(42, 9), 4, size),
            (UVec2::ZERO, UVec2::new(11, 3))
        );
        assert_eq!(
            occl_tiles(
                UVec2::new(16, 16),
                UVec2::new(64, 64),
                8,
                UVec2::new(400, 225)
            ),
            (UVec2::new(2, 2), UVec2::new(8, 8))
        );
    }

    /// The image's rows the shader reads the planes from are the block's.
    #[test]
    fn the_plane_rows_are_the_blocks() {
        assert_eq!(PARAMS_ROW, 0);
        assert_eq!(MIRROR2_ROW as usize * 4, MIRROR2_LANES.start);
    }
}
