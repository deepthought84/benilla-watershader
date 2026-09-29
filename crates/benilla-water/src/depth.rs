//! The world camera's **depth prepass** while Improved Water is on, and the bindable view depth.
//!
//! ## What reads what
//!
//! The water measures what is behind it against the **linear depth** [`super::hiz`] writes from the
//! view's own depth after the opaque pass (`WaterPassImages::scene_depth`, read by `liquid.wgsl`'s
//! `scene_behind`): everything the opaque pass drew, the static world, entity-path models and units
//! included. That view depth becomes bindable here: `maintain_depth_prepass` adds
//! `TEXTURE_BINDING` to the world camera's `depth_texture_usages`, which the pyramid's seed needs.
//!
//! Nothing samples the prepass. It fills the opaque world's depth ahead of the main pass while the
//! stylised look is on, and the reference lane pays nothing for it.
//!
//! ## What is in it, and what is not
//!
//! A prepass that writes depth for a fragment the main pass does not draw hides what is behind it:
//! the main pass then fails `GreaterEqual` against depth nothing drew.
//!
//! **The WDL horizon is not.** `wdl.wgsl`'s fragment cuts everything nearer than `farclip − 33`, so
//! the coarse hull draws only past the far-clip wall and the fine terrain owns the near field.
//! Bevy's stock prepass has no such cut: it would write the whole hull's depth, and the coarse
//! surface lies above the fine one wherever the ground is flat — the river valleys and lake beds —
//! so the terrain there would fail the test and the sky show through. `WdlExt::enable_prepass()`
//! keeps the hull out. `terrain.wgsl`'s position carries `@invariant`, the standard guard for a
//! material that draws in both passes.
//!
//! **Models are not** (`WowModelExt::enable_prepass`): their material rebuilds the vertex layout
//! around attributes Bevy does not know, and Bevy hands a material's `specialize` to the prepass
//! pipeline as well, which would fail validation the moment a skinned character is in view. The
//! static world is in it through its own pass (`static_gx`'s `StaticGxPrepassNode`, the same
//! geometry depth-only). Alpha-blended materials, the water among them, Bevy excludes by alpha
//! mode.

use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::prelude::*;
use bevy::render::render_resource::TextureUsages;

use super::WaterStyle;
use crate::view::WorldCamera;

/// Attach the depth prepass while the stylised look is on, and take it away again when it is not.
///
/// Change-gated on both sides: a steady state does nothing at all, and the insert/remove only fire
/// on the frame the answer actually differs from what the camera already carries.
fn maintain_depth_prepass(
    mut commands: Commands,
    style: Res<WaterStyle>,
    mut cameras: Query<(Entity, Has<DepthPrepass>, &mut Camera3d), With<WorldCamera>>,
) {
    let want = style.is_stylised();
    for (entity, has, mut camera_3d) in &mut cameras {
        // **The view's depth has to be bindable for the pyramid and the linear depth** —
        // `liquid::hiz` seeds both from it after the opaque pass, because the prepass is missing
        // everything on the entity path. The same one flag Bevy's own occlusion culling adds
        // (`configure_occlusion_culling_view_targets`). Added once and never removed: it costs
        // nothing a look without the march can notice, and taking it away would re-create the
        // texture for nothing on every flip. Written only when missing, so change detection on
        // the camera stays quiet.
        let usages = TextureUsages::from(camera_3d.depth_texture_usages);
        if want && !usages.contains(TextureUsages::TEXTURE_BINDING) {
            camera_3d.depth_texture_usages = (usages | TextureUsages::TEXTURE_BINDING).into();
        }
        if want == has {
            continue;
        }
        let mut cam = commands.entity(entity);
        if want {
            cam.insert(DepthPrepass);
        } else {
            cam.remove::<DepthPrepass>();
        }
    }
}

pub(super) fn register(app: &mut App) {
    // Every frame, but it is a query over one camera and a comparison — the work is the insert,
    // which happens on a style flip and on the first frame a camera exists.
    app.add_systems(Update, maintain_depth_prepass);
}
