//! The Improved Water **preview**'s glow accumulator (FND-3, P1-37/38): every water pixel adds the
//! highlight excess its knee removed (and what its composite holds above white) to its
//! quarter-resolution texel (`liquid.wgsl`'s `water_glow_acc`, 16.16 fixed point), and FFXGlow's
//! water downsample puts it in the blur's alpha, which the combine adds as light
//! (`benilla_world::ffx_glow::FfxWaterGlow`). Cleared each frame before the water draws.

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel,
};
use bevy::render::render_resource::{Buffer, BufferDescriptor, BufferUsages};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::view::ViewTarget;
use bevy::render::RenderApp;

use benilla_assets::preview as lever;
use benilla_world::ffx_glow::FfxWaterGlow;

use super::{WaterPipelineCaps, WaterStyle};
use crate::world_view::WaterWorldView;

/// The quarter texels of the largest view served, 7680×4320; one u32 each.
const MAX_TEXELS: u64 = (7680 / 4) * (4320 / 4);

/// The accumulator and whether the glow runs this frame.
#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct WaterGlowAcc {
    pub(crate) buf: Buffer,
    armed: bool,
}

/// The accumulator, or 16 bytes the shader's length test writes nothing into while `glow` is off.
fn setup_glow_acc(mut commands: Commands, device: Res<RenderDevice>) {
    let buf = device.create_buffer(&BufferDescriptor {
        label: Some("water_glow_acc"),
        size: if lever::on(lever::GLOW) {
            MAX_TEXELS * 4
        } else {
            16
        },
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    commands.insert_resource(WaterGlowAcc { buf, armed: false });
}

/// Arm the glow while its pipeline variant is asked for on Improved Water, and hand FFXGlow the
/// accumulator with `$WOW_PREVIEW_GLOW_WEIGHT` (6) and `$WOW_PREVIEW_GLOW_CAP` (4.0).
fn arm_glow(
    caps: Res<WaterPipelineCaps>,
    style: Res<WaterStyle>,
    acc: Option<ResMut<WaterGlowAcc>>,
    ffx: Option<ResMut<FfxWaterGlow>>,
) {
    let (Some(mut acc), Some(mut ffx)) = (acc, ffx) else {
        return;
    };
    let armed = caps.glow && style.is_stylised();
    if acc.armed != armed {
        acc.armed = armed;
    }
    static GAINS: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();
    let (weight, cap) = *GAINS.get_or_init(|| {
        (
            lever::number("GLOW_WEIGHT", 6.0, 0.0, 32.0),
            lever::number("GLOW_CAP", 4.0, 0.0, 16.0),
        )
    });
    let live = ffx.buf.is_some();
    if armed != live {
        ffx.buf = armed.then(|| acc.buf.clone());
        ffx.weight = weight;
        ffx.cap = cap;
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct WaterGlowClearLabel;

/// Zero the world view's share of the accumulator before the water's transparent pass.
#[derive(Default)]
struct WaterGlowClearNode;

impl Node for WaterGlowClearNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(acc) = world.get_resource::<WaterGlowAcc>() else {
            return Ok(());
        };
        if !acc.armed || world.get::<WaterWorldView>(graph.view_entity()).is_none() {
            return Ok(());
        }
        let Some(target) = world.get::<ViewTarget>(graph.view_entity()) else {
            return Ok(());
        };
        let main = target.main_texture();
        let texels = u64::from((main.width() / 4).max(8)) * u64::from((main.height() / 4).max(8));
        render_context.command_encoder().clear_buffer(
            &acc.buf,
            0,
            Some(texels.min(MAX_TEXELS) * 4),
        );
        Ok(())
    }
}

pub(super) fn register(app: &mut App) {
    app.add_systems(
        Startup,
        setup_glow_acc
            .after(benilla_assets::AssetSet::Open)
            .before(super::WaterPassSet),
    );
    if !lever::on(lever::GLOW) {
        return;
    }
    app.add_plugins(ExtractResourcePlugin::<WaterGlowAcc>::default())
        .add_systems(PostUpdate, arm_glow);
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .add_render_graph_node::<WaterGlowClearNode>(Core3d, WaterGlowClearLabel)
        .add_render_graph_edges(
            Core3d,
            (
                Node3d::MainTransmissivePass,
                WaterGlowClearLabel,
                Node3d::MainTransparentPass,
            ),
        );
}
