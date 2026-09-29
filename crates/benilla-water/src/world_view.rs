//! **The one view the water's screen-space passes serve**, found by a marker.
//!
//! Core3d runs for every 3D view: the world camera, the two mirrors, the probe's cube faces and
//! the portrait booths. The depth pyramid and the colour snapshot describe the world camera's
//! frame alone, so that camera carries [`WaterWorldView`], extracted, and each node asks for it on
//! the view it runs for. A depth prepass or a texture of the right size is something the world
//! view happens to have, not what makes it the world view.
//!
//! [`WaterViewCheck`] counts each node's visits to the marked view; a frame in which the passes are
//! armed and a marked view rendered must count exactly one of each.

use std::sync::atomic::{AtomicU32, Ordering};

use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::view::ViewTarget;
use bevy::render::{Render, RenderApp, RenderSystems};

use crate::view::WorldCamera;

/// Marks the one view the water's screen-space passes serve: the `WorldCamera`. Extracted, so a
/// render-graph node asks `world.get::<WaterWorldView>(graph.view_entity())` instead of inferring
/// the view from a prepass or a texture size.
#[derive(Component, Clone, Copy, Default, ExtractComponent)]
pub(crate) struct WaterWorldView;

/// Stamp the marker on every `WorldCamera` that lacks it: three spawn sites, none of which should
/// know about water (the shape of `hooks::stamp_world_camera_layers`).
fn mark_world_view(
    mut commands: Commands,
    cams: Query<Entity, (With<WorldCamera>, Without<WaterWorldView>)>,
) {
    for cam in &cams {
        commands.entity(cam).insert(WaterWorldView);
    }
}

/// Which pass visited the world view.
#[derive(Clone, Copy)]
pub(crate) enum Visit {
    Pyramid,
    Snapshot,
}

/// Render world: each pass's visits to the marked view this frame, swapped to zero at cleanup.
#[derive(Resource, Default)]
pub(crate) struct WaterViewCheck {
    pyramid: AtomicU32,
    snapshot: AtomicU32,
}

/// Whether `view` is the world view; a visit to it is counted for `pass`. Every water node that
/// serves the world view opens with this.
pub(crate) fn serves_world_view(world: &World, view: Entity, pass: Visit) -> bool {
    if world.get::<WaterWorldView>(view).is_none() {
        return false;
    }
    if let Some(check) = world.get_resource::<WaterViewCheck>() {
        let counter = match pass {
            Visit::Pyramid => &check.pyramid,
            Visit::Snapshot => &check.snapshot,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }
    true
}

/// The verdict on one frame's visits: `None` when nothing was owed (passes unarmed, or no marked
/// view rendered), else whether each pass served the world view exactly once.
fn served_once(armed: bool, rendered: bool, visits: (u32, u32)) -> Option<bool> {
    (armed && rendered).then_some(visits == (1, 1))
}

/// A marked view that rendered this frame: an active camera (`ExtractedCamera`, removed from an
/// inactive one) with a target to draw into.
type RenderedWorldView = (
    With<WaterWorldView>,
    With<ExtractedCamera>,
    With<ViewTarget>,
);

/// Cleanup: while the passes are armed and a marked view rendered this frame, the pyramid and the
/// snapshot each served it exactly once. `RUST_LOG=benilla_water::world_view=trace` logs the
/// counts every frame.
fn check_world_view(
    check: Res<WaterViewCheck>,
    hiz: Option<Res<super::hiz::WaterHiz>>,
    views: Query<(), RenderedWorldView>,
) {
    let visits = (
        check.pyramid.swap(0, Ordering::Relaxed),
        check.snapshot.swap(0, Ordering::Relaxed),
    );
    let armed = hiz.is_some_and(|h| h.armed);
    let rendered = !views.is_empty();
    trace!(
        "world view visits: pyramid {} snapshot {} (armed {armed}, rendered {rendered})",
        visits.0,
        visits.1
    );
    if served_once(armed, rendered, visits) == Some(false) {
        debug_assert_eq!(
            visits,
            (1, 1),
            "water: the world view was served (pyramid, snapshot) {visits:?} times, not once each"
        );
        warn_once!(
            "water: the world view was served (pyramid, snapshot) {visits:?} times, not once each"
        );
    }
}

pub(super) fn register(app: &mut App) {
    app.add_plugins(ExtractComponentPlugin::<WaterWorldView>::default())
        .add_systems(Update, mark_world_view);
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<WaterViewCheck>()
        .add_systems(Render, check_world_view.in_set(RenderSystems::Cleanup));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Owed only while armed with a marked view rendered, and then exactly one visit each.
    #[test]
    fn the_world_view_is_served_once_by_each_pass() {
        assert_eq!(served_once(true, true, (1, 1)), Some(true));
        assert_eq!(served_once(true, true, (0, 1)), Some(false));
        assert_eq!(served_once(true, true, (1, 2)), Some(false));
        assert_eq!(served_once(false, true, (0, 0)), None);
        assert_eq!(served_once(true, false, (0, 0)), None);
    }
}
