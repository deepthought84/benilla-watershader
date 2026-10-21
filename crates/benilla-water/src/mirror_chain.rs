//! The planar mirrors' **roughness chains** for the preview's rough reflections (`rough`,
//! `benilla_assets::preview`): each capture downsampled at half its size through five levels, so
//! the water reads a prefiltered blur where its surface no longer resolves the ripple
//! (`liquid.wgsl`'s `planar_read_rough`). One compute pass on the world view, after the captures
//! (camera orders −1 and −2) and before the water draws; `shaders/mirror_chain.wgsl`.

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel,
};
use bevy::render::render_resource::binding_types::{sampler, texture_2d, texture_storage_2d};
use bevy::render::render_resource::{
    AddressMode, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, Extent3d,
    FilterMode, PipelineCache, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
    StorageTextureAccess, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureViewDescriptor,
};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::GpuImage;
use bevy::render::{RenderApp, RenderStartup};

use benilla_assets::preview as lever;
use benilla_world::liquid::MIRROR2_LANES;

use super::reflect::{WaterReflect, WaterReflectData};
use super::water_compute::WaterComputeLabel;
use super::world_view::WaterWorldView;
use super::WaterStyle;

/// The chain's levels at most.
const CHAIN_LEVELS: u32 = 5;

/// A chain for a capture of `cap` texels: half its size (at least one texel), up to
/// [`CHAIN_LEVELS`] levels, written by the pass and read by the water; no CPU data.
pub(crate) fn chain_image(cap: UVec2) -> Image {
    let size = (cap / 2).max(UVec2::ONE);
    let mut image = Image::new_fill(
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 8],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.mip_level_count = CHAIN_LEVELS.min(size.min_element().ilog2() + 1);
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
    image.data = None;
    image
}

/// This frame's chains: the captures, their chains and which mirror is live; extracted.
#[derive(Resource, Clone, Default, ExtractResource)]
struct MirrorChainFrame {
    cap: [Handle<Image>; 2],
    chain: [Handle<Image>; 2],
    on: [bool; 2],
}

/// Publish the chains for the frame: a mirror's chain is built while `rough` runs on the stylised
/// look and that mirror's strength lane is on.
fn drive_mirror_chain(
    style: Res<WaterStyle>,
    reflect: Option<Res<WaterReflect>>,
    data: Option<Res<WaterReflectData>>,
    mut frame: ResMut<MirrorChainFrame>,
) {
    let (Some(reflect), Some(data)) = (reflect, data) else {
        return;
    };
    let armed = lever::on(lever::ROUGH) && style.is_stylised();
    frame.cap = [reflect.image.clone(), reflect.image2.clone()];
    frame.chain = [reflect.chain.clone(), reflect.chain2.clone()];
    frame.on = [
        armed && data.0[1] > 0.0,
        armed && data.0[MIRROR2_LANES.start + 1] > 0.0,
    ];
}

/// The pass's pipeline, layout and sampler.
#[derive(Resource)]
struct ChainPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
    sampler: Sampler,
}

fn init_chain_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "water_mirror_chain_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let shader = asset_server.load(benilla_assets::dev_shaders::shader_path(
        "embedded://benilla_water/shaders/mirror_chain.wgsl",
    ));
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("water_mirror_chain".into()),
        layout: vec![layout.clone()],
        shader,
        entry_point: Some("mirror_chain".into()),
        ..default()
    });
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("water_mirror_chain_samp"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });
    commands.insert_resource(ChainPipeline {
        layout,
        pipeline,
        sampler,
    });
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct MirrorChainLabel;

#[derive(Default)]
struct MirrorChainNode;

impl Node for MirrorChainNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // Only for the world view: Core3d runs for the mirrors and the probe's faces too.
        if world.get::<WaterWorldView>(graph.view_entity()).is_none() {
            return Ok(());
        }
        let (Some(frame), Some(pipe), Some(cache), Some(images)) = (
            world.get_resource::<MirrorChainFrame>(),
            world.get_resource::<ChainPipeline>(),
            world.get_resource::<PipelineCache>(),
            world.get_resource::<RenderAssets<GpuImage>>(),
        ) else {
            return Ok(());
        };
        if !frame.on.iter().any(|on| *on) {
            return Ok(());
        }
        let Some(pipeline) = cache.get_compute_pipeline(pipe.pipeline) else {
            return Ok(());
        };
        let layout = cache.get_bind_group_layout(&pipe.layout);
        let device = render_context.render_device().clone();
        // Every level's views and bind group first: the pass borrows the encoder.
        let mut work = Vec::new();
        for i in 0..2 {
            if !frame.on[i] {
                continue;
            }
            let (Some(cap), Some(chain)) = (images.get(&frame.cap[i]), images.get(&frame.chain[i]))
            else {
                continue;
            };
            let want = (cap.size_2d() / 2).max(UVec2::ONE);
            if chain.size_2d() != want {
                continue; // a resize in flight: the water falls back to the sharp read
            }
            let levels = chain.texture.mip_level_count();
            for level in 0..levels {
                let view = |mip: u32| {
                    chain.texture.create_view(&TextureViewDescriptor {
                        label: Some("water_mirror_chain_level"),
                        base_mip_level: mip,
                        mip_level_count: Some(1),
                        ..default()
                    })
                };
                let src = if level == 0 {
                    cap.texture_view.clone()
                } else {
                    view(level - 1)
                };
                let dst = view(level);
                let group = device.create_bind_group(
                    "water_mirror_chain_bind",
                    &layout,
                    &BindGroupEntries::sequential((&src, &pipe.sampler, &dst)),
                );
                let size = (want >> level).max(UVec2::ONE);
                work.push((group, size.x.div_ceil(8), size.y.div_ceil(8)));
            }
        }
        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("water_mirror_chain"),
                    timestamp_writes: None,
                });
        pass.set_pipeline(pipeline);
        for (group, x, y) in &work {
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(*x, *y, 1);
        }
        Ok(())
    }
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<MirrorChainFrame>()
        .add_plugins(ExtractResourcePlugin::<MirrorChainFrame>::default());
    if lever::on(lever::ROUGH) {
        app.add_systems(
            PostUpdate,
            drive_mirror_chain.after(super::reflect::drive_reflection),
        );
    }
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .add_systems(RenderStartup, init_chain_pipeline)
        .add_render_graph_node::<MirrorChainNode>(Core3d, MirrorChainLabel)
        .add_render_graph_edges(Core3d, (WaterComputeLabel, MirrorChainLabel))
        .add_render_graph_edges(Core3d, (MirrorChainLabel, Node3d::MainTransparentPass));
}
