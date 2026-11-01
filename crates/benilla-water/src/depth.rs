//! The world camera's **bindable view depth** while Improved Water is on.
//!
//! The water measures what is behind it against the **linear depth** [`super::hiz`] writes from the
//! view's own depth after the opaque pass (`WaterPassImages::scene_depth`, read by `liquid.wgsl`'s
//! `scene_behind`): everything the opaque pass drew, the static world, entity-path models and units
//! included. [`bindable_view_depth`] adds `TEXTURE_BINDING` to the world camera's
//! `depth_texture_usages`, which the pyramid's seed needs to read that depth.
//!
//! **No depth prepass.** Nothing reads one, and at render scale 2 the frame is faster without it:
//! the opaque pass's early-Z buys back less than the depth-only passes over the static world and the
//! entity path cost. Without one the world camera's view key is the same on both lanes, so a style
//! flip changes no world pipeline's view key, and the terrain's far-clip wall (`terrain.wgsl`
//! discards past it, where Bevy's stock prepass would have written depth) stands on both.

use bevy::prelude::*;
use bevy::render::render_resource::TextureUsages;

use super::WaterStyle;
use crate::view::WorldCamera;

/// Make the world camera's depth bindable while the stylised look is on: `liquid::hiz` seeds the
/// pyramid and the linear depth from it after the opaque pass. The same one flag Bevy's own
/// occlusion culling adds (`configure_occlusion_culling_view_targets`). Added once and never
/// removed: it costs nothing a look without the water's passes can notice, and taking it away
/// would re-create the texture for nothing on every flip. Written only when missing, so change
/// detection on the camera stays quiet.
fn bindable_view_depth(
    style: Res<WaterStyle>,
    mut cameras: Query<&mut Camera3d, With<WorldCamera>>,
) {
    if !style.is_stylised() {
        return;
    }
    for mut camera_3d in &mut cameras {
        let usages = TextureUsages::from(camera_3d.depth_texture_usages);
        if !usages.contains(TextureUsages::TEXTURE_BINDING) {
            camera_3d.depth_texture_usages = (usages | TextureUsages::TEXTURE_BINDING).into();
        }
    }
}

pub(super) fn register(app: &mut App) {
    // Every frame, but it is a query over one camera and a comparison; the write happens once.
    // After the frame's script calls, whose `waterStyle` callback writes the style.
    app.add_systems(
        Update,
        bindable_view_depth.after(benilla_world::schedule::WorldStage::Present),
    );
}
