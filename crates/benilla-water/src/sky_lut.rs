//! **The sky floor's cube**: the sky the stylised water reflects where no tier has a better
//! answer, as a 64² cube with a prefiltered mip chain (`shaders/sky_lut.wgsl`), bound to every
//! liquid material at 119 and read by `liquid.wgsl`'s `sky_floor` along the flat reflected ray,
//! blurred only by the roughness the surface no longer resolves and by no more than half the ray's
//! elevation.
//!
//! Mip 0 is the dome along each texel's direction by the sky pass's own law (the dome rows of the
//! reflection block, its single writer's), with the drawn cloud layer over it as the sky pass
//! blends it ([`SkyCloudLayer`]), decoded to linear light and unlit: the sky in the water is the sky
//! above it. Each mip below widens the one above by a Gaussian, so mip k holds the sky blurred to
//! `SKY_LUT_TEXEL0·2^k` radians.
//!
//! Rebuilt by [`super::water_compute`]'s pass on every frame the stylised water is on screen,
//! before the occlusion: the dome and the clouds change with the clock and the weather, and the
//! whole chain costs a few thousand texels.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, texture_2d, texture_cube, texture_storage_2d_array,
    uniform_buffer,
};
use bevy::render::render_resource::{
    AddressMode, BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    BufferId, CachedComputePipelineId, ComputePass, ComputePipelineDescriptor, Extent3d,
    FilterMode, PipelineCache, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
    StorageTextureAccess, TextureDimension, TextureFormat, TextureId, TextureSampleType,
    TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension, TextureViewId,
    UniformBuffer,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::shader::ShaderDefVal;

use benilla_world::clouds::SkyCloudLayer;
use benilla_world::liquid::DOME_LANES;

use super::reflect::WaterReflectBuffer;
use super::water_compute::ComputeFrame;

/// The cube's face size at mip 0, texels.
const LUT_SIZE: u32 = 64;
/// Its mips: 64² down to 2².
const LUT_MIPS: u32 = 6;

/// The sky floor's cube, made once at startup; extracted.
#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct SkyLut {
    pub(crate) image: Handle<Image>,
}

/// The cube: `Rgba16Float` linear light, 64² by six faces with six mips, written by the compute
/// pass and sampled as a cube. Created without data, which wgpu zero-initialises: black until the
/// first rebuild.
fn lut_image() -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: LUT_SIZE,
            height: LUT_SIZE,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.mip_level_count = LUT_MIPS;
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        label: Some("sky_lut_cube"),
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    image
}

/// The pipelines: the base mip from the dome and the clouds, and each mip from the one above.
#[derive(Resource)]
struct LutPipelines {
    base_layout: BindGroupLayoutDescriptor,
    prefilter_layout: BindGroupLayoutDescriptor,
    base: CachedComputePipelineId,
    prefilter: CachedComputePipelineId,
    /// Linear and clamped, for the cloud tile as the cloud dome samples it and for the mips.
    sampler: Sampler,
}

fn init_lut_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
) {
    let base_layout = BindGroupLayoutDescriptor::new(
        "sky_lut_base_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (0, storage_buffer_read_only_sized(false, None)),
                (1, texture_2d(TextureSampleType::Float { filterable: true })),
                (2, sampler(SamplerBindingType::Filtering)),
                (
                    3,
                    texture_storage_2d_array(
                        TextureFormat::Rgba16Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                ),
                (6, uniform_buffer::<Vec4>(false)),
            ),
        ),
    );
    let prefilter_layout = BindGroupLayoutDescriptor::new(
        "sky_lut_prefilter_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (
                    3,
                    texture_storage_2d_array(
                        TextureFormat::Rgba16Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                ),
                (
                    4,
                    texture_cube(TextureSampleType::Float { filterable: true }),
                ),
                (5, sampler(SamplerBindingType::Filtering)),
            ),
        ),
    );
    let shader = asset_server.load("embedded://benilla_water/shaders/sky_lut.wgsl");
    let defs = vec![ShaderDefVal::UInt(
        "DOME_ROW".into(),
        (DOME_LANES.start / 4) as u32,
    )];
    let pipeline = |layout: &BindGroupLayoutDescriptor, entry: &'static str| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("sky_{entry}").into()),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            shader_defs: defs.clone(),
            entry_point: Some(entry.into()),
            ..default()
        })
    };
    let base = pipeline(&base_layout, "lut_base");
    let prefilter = pipeline(&prefilter_layout, "lut_prefilter");
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("sky_lut_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });
    commands.insert_resource(LutPipelines {
        base_layout,
        prefilter_layout,
        base,
        prefilter,
        sampler,
    });
}

/// Render world: the cube's per-mip views and the bind groups that read and write them, built for
/// the textures and buffers they bind and kept until one changes.
#[derive(Resource, Default)]
struct LutBindings {
    /// The cube texture the views and the prefilter groups were made for.
    texture: Option<TextureId>,
    /// Per mip, its six faces as a storage array.
    storage: Vec<TextureView>,
    /// Per mip k ≥ 1, the group reading mip k − 1 as a cube and writing mip k.
    prefilter: Vec<BindGroup>,
    /// What the base group binds: the block, the cloud tile's view and the parameters.
    base_key: Option<(BufferId, TextureViewId, TextureViewId, BufferId)>,
    base: Option<BindGroup>,
    /// `x` the cloud layer drawn (1) or hidden (0).
    params: UniformBuffer<Vec4>,
    /// The chain runs this frame: the look armed, water on screen, every group built.
    ready: bool,
}

/// Render world, after the compute pass's gate ([`ComputeFrame::lut`]): write the cloud switch and
/// (re)build the views and groups only where a texture or buffer they bind changed.
#[allow(clippy::too_many_arguments)]
fn prepare_sky_lut(
    frame: Res<ComputeFrame>,
    lut: Option<Res<SkyLut>>,
    clouds: Option<Res<SkyCloudLayer>>,
    lanes: Option<Res<WaterReflectBuffer>>,
    pipes: Option<Res<LutPipelines>>,
    cache: Res<PipelineCache>,
    images: Res<RenderAssets<GpuImage>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut b: ResMut<LutBindings>,
) {
    b.ready = false;
    let (true, Some(lut), Some(clouds), Some(lanes), Some(pipes)) =
        (frame.lut, lut, clouds, lanes, pipes)
    else {
        return;
    };
    let (Some(cube), Some(cloud)) = (images.get(&lut.image), images.get(&clouds.image)) else {
        return;
    };
    if b.texture != Some(cube.texture.id()) {
        b.storage = (0..LUT_MIPS)
            .map(|mip| {
                cube.texture.create_view(&TextureViewDescriptor {
                    label: Some("sky_lut_mip"),
                    dimension: Some(TextureViewDimension::D2Array),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                    ..default()
                })
            })
            .collect();
        let layout = cache.get_bind_group_layout(&pipes.prefilter_layout);
        b.prefilter = (1..LUT_MIPS)
            .map(|mip| {
                let above = cube.texture.create_view(&TextureViewDescriptor {
                    label: Some("sky_lut_above"),
                    dimension: Some(TextureViewDimension::Cube),
                    base_mip_level: mip - 1,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(6),
                    ..default()
                });
                device.create_bind_group(
                    "sky_lut_prefilter_bind",
                    &layout,
                    &BindGroupEntries::with_indices((
                        (3, &b.storage[mip as usize]),
                        (4, &above),
                        (5, &pipes.sampler),
                    )),
                )
            })
            .collect();
        debug!("sky_lut: {} mip views and the prefilter groups", LUT_MIPS);
        b.texture = Some(cube.texture.id());
        b.base_key = None;
    }
    let visible = Vec4::new(f32::from(u8::from(clouds.visible)), 0.0, 0.0, 0.0);
    if b.params.get() != &visible || b.params.buffer().is_none() {
        b.params.set(visible);
        b.params.write_buffer(&device, &queue);
    }
    let (Some(params), Some(params_binding)) = (b.params.buffer(), b.params.binding()) else {
        return;
    };
    let key = (
        lanes.0.id(),
        cloud.texture_view.id(),
        b.storage[0].id(),
        params.id(),
    );
    if b.base_key != Some(key) {
        b.base = Some(device.create_bind_group(
            "sky_lut_base_bind",
            &cache.get_bind_group_layout(&pipes.base_layout),
            &BindGroupEntries::with_indices((
                (0, lanes.0.as_entire_binding()),
                (1, &cloud.texture_view),
                (2, &pipes.sampler),
                (3, &b.storage[0]),
                (6, params_binding),
            )),
        ));
        debug!("sky_lut: the base bind group for {key:?}");
        b.base_key = Some(key);
    }
    b.ready = true;
}

/// Record the chain into the water's compute pass: mip 0, then each mip from the one above.
pub(crate) fn dispatch_lut(pass: &mut ComputePass, world: &World) -> bool {
    let (Some(pipes), Some(cache), Some(b)) = (
        world.get_resource::<LutPipelines>(),
        world.get_resource::<PipelineCache>(),
        world.get_resource::<LutBindings>(),
    ) else {
        return false;
    };
    let (true, Some(base), Some(base_pipeline), Some(prefilter_pipeline)) = (
        b.ready,
        b.base.as_ref(),
        cache.get_compute_pipeline(pipes.base),
        cache.get_compute_pipeline(pipes.prefilter),
    ) else {
        return false;
    };
    pass.set_pipeline(base_pipeline);
    pass.set_bind_group(0, base, &[]);
    pass.dispatch_workgroups(LUT_SIZE.div_ceil(8), LUT_SIZE.div_ceil(8), 6);
    pass.set_pipeline(prefilter_pipeline);
    for (mip, group) in (1..LUT_MIPS).zip(&b.prefilter) {
        let size = LUT_SIZE >> mip;
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(size.div_ceil(8), size.div_ceil(8), 6);
    }
    true
}

pub(super) fn register(app: &mut App) {
    let image = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(lut_image());
    app.insert_resource(SkyLut { image }).add_plugins((
        ExtractResourcePlugin::<SkyLut>::default(),
        ExtractResourcePlugin::<SkyCloudLayer>::default(),
    ));
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<LutBindings>()
        .add_systems(bevy::render::RenderStartup, init_lut_pipelines)
        .add_systems(
            Render,
            prepare_sky_lut.in_set(RenderSystems::PrepareBindGroups),
        );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cube's mips reach 2², the prefilter's widest Gaussian (`σ_5 = 1 rad`).
    #[test]
    fn the_cube_reaches_two_texels() {
        assert_eq!(LUT_SIZE >> (LUT_MIPS - 1), 2);
    }

    /// The dome's rows the cube reads are the block's.
    #[test]
    fn the_dome_row_is_the_blocks() {
        assert_eq!(DOME_LANES.start % 4, 0);
        assert_eq!(DOME_LANES.len(), 28);
    }

    /// The cube decodes gamma as the water does.
    #[test]
    fn the_cube_and_the_water_share_one_gamma() {
        let gamma = "const WATER_GAMMA: f32 = 2.2;";
        assert!(include_str!("shaders/sky_lut.wgsl").contains(gamma));
        assert!(include_str!("../../benilla-assets/src/shaders/liquid.wgsl").contains(gamma));
    }
}
