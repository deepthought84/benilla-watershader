//! **The one view the water's screen-space passes serve**, found by a marker, and **where on it
//! water can be**.
//!
//! Core3d runs for every 3D view: the world camera, the two mirrors, the probe's cube faces and
//! the portrait booths. The depth pyramid and the colour snapshot describe the world camera's
//! frame alone, so that camera carries [`WaterWorldView`], extracted, and each node asks for it on
//! the view it runs for. A texture of the right size is something the world view happens to have,
//! not what makes it the world view.
//!
//! [`WaterViewCheck`] counts each node's visits to the marked view; a frame in which the passes are
//! armed, water is on screen and a marked view rendered must count exactly one of each, and any
//! other frame none.
//!
//! [`WaterScreenRect`] is the world view's rectangle around every visible water surface, grown by
//! a margin: what only the water reads (the linear depth) is written inside it alone, and a frame
//! with no water on screen runs none of the water's passes.

use std::sync::atomic::{AtomicU32, Ordering};

use benilla_assets::materials::LiquidMaterial;
use benilla_world::liquid::LiquidSurface;
use bevy::camera::primitives::Aabb;
use bevy::camera::visibility::VisibilitySystems;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
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

/// Whether any stylised water is on the world view's screen this frame, by the extracted
/// [`WaterScreenRect`]: the water's passes run only then.
pub(crate) fn water_on_screen(world: &World) -> bool {
    world
        .get_resource::<WaterScreenRect>()
        .is_some_and(|r| r.visible)
}

/// Whether `view` is the world view; a visit to it is counted for `pass`. Every water node that
/// serves the world view asks this once it is armed.
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

/// The verdict on one frame's visits: armed (the look on and water on screen) with a marked view
/// rendered, each pass served it exactly once; unarmed, neither ran; `None` when armed and no
/// marked view rendered, which owes nothing.
fn served_once(armed: bool, rendered: bool, visits: (u32, u32)) -> Option<bool> {
    if armed {
        rendered.then_some(visits == (1, 1))
    } else {
        Some(visits == (0, 0))
    }
}

/// A marked view that rendered this frame: an active camera (`ExtractedCamera`, removed from an
/// inactive one) with a target to draw into.
type RenderedWorldView = (
    With<WaterWorldView>,
    With<ExtractedCamera>,
    With<ViewTarget>,
);

/// Cleanup: while the passes are armed, water is on screen and a marked view rendered this frame,
/// the pyramid and the snapshot each served it exactly once; otherwise neither ran.
/// `RUST_LOG=benilla_water::world_view=trace` logs the counts every frame.
fn check_world_view(
    check: Res<WaterViewCheck>,
    hiz: Option<Res<super::hiz::WaterHiz>>,
    rect: Option<Res<WaterScreenRect>>,
    views: Query<(), RenderedWorldView>,
) {
    let visits = (
        check.pyramid.swap(0, Ordering::Relaxed),
        check.snapshot.swap(0, Ordering::Relaxed),
    );
    let armed = hiz.is_some_and(|h| h.armed) && rect.is_some_and(|r| r.visible);
    let rendered = !views.is_empty();
    trace!(
        "world view visits: pyramid {} snapshot {} (armed {armed}, rendered {rendered})",
        visits.0,
        visits.1
    );
    if served_once(armed, rendered, visits) == Some(false) {
        let owed = if armed { (1, 1) } else { (0, 0) };
        debug_assert_eq!(
            visits, owed,
            "water: the world view was served (pyramid, snapshot) {visits:?} times, not {owed:?}"
        );
        warn_once!(
            "water: the world view was served (pyramid, snapshot) {visits:?} times, not {owed:?}"
        );
    }
}

/// Where water can be on the world view's screen this frame: the projected boxes of every visible
/// liquid surface except magma and slime. Extracted; the seed writes the linear depth inside it.
#[derive(Resource, Clone, Copy, Default, PartialEq, Eq, ExtractResource)]
pub(crate) struct WaterScreenRect {
    /// Any non-fullbright liquid surface is visible to the world camera this frame.
    pub(crate) visible: bool,
    /// Inclusive min, exclusive max, physical pixels of the world camera's target, already grown
    /// by [`WATER_RECT_MARGIN_PX`] and clamped to the target.
    pub(crate) min: UVec2,
    pub(crate) max: UVec2,
}

/// Room for a refracted or contact read beside the surface (BODY-1 clamps its offset below this).
pub(crate) const WATER_RECT_MARGIN_PX: u32 = 32;

/// The screen rectangle covering every box, or `None` when none is on screen. A box with any
/// corner at or behind `w <= 0.5 * near` returns the whole target (conservative: the eye is at or
/// in it).
///
/// Pixels are the target's, y down: `x = (ndc.x·0.5 + 0.5)·W`, `y = (0.5 − ndc.y·0.5)·H`, min
/// floored, max ceiled, grown by `margin`, clamped to `[0, W]×[0, H]`.
pub(crate) fn project_rect(
    boxes: impl Iterator<Item = [Vec3; 8]>,
    clip_from_world: Mat4,
    near: f32,
    size: UVec2,
    margin: u32,
) -> Option<(UVec2, UVec2)> {
    let target = size.as_vec2();
    let mut lo = Vec2::INFINITY;
    let mut hi = Vec2::NEG_INFINITY;
    for corners in boxes {
        for corner in corners {
            let clip = clip_from_world * corner.extend(1.0);
            if clip.w <= 0.5 * near {
                return (size.x > 0 && size.y > 0).then_some((UVec2::ZERO, size));
            }
            let ndc = clip.xy() / clip.w;
            let px = Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * target;
            lo = lo.min(px);
            hi = hi.max(px);
        }
    }
    let grow = margin as f32;
    let min = (lo.floor() - grow).clamp(Vec2::ZERO, target);
    let max = (hi.ceil() + grow).clamp(Vec2::ZERO, target);
    // No box, or every box beside the target: nothing left after clamping.
    (min.x < max.x && min.y < max.y).then(|| (min.as_uvec2(), max.as_uvec2()))
}

/// A box's eight corners in world space.
fn world_corners(aabb: &Aabb, transform: &GlobalTransform) -> [Vec3; 8] {
    let (c, h) = (Vec3::from(aabb.center), Vec3::from(aabb.half_extents));
    std::array::from_fn(|i| {
        let s = Vec3::new(
            if i & 1 == 0 { -1.0 } else { 1.0 },
            if i & 2 == 0 { -1.0 } else { 1.0 },
            if i & 4 == 0 { -1.0 } else { 1.0 },
        );
        transform.transform_point(c + s * h)
    })
}

/// The world camera's water rectangle, after every `ViewVisibility` writer (the light checks run
/// before the newly-hidden marking, the last of them). Liquid rides
/// `WATER_RENDER_LAYER`, which only the world camera renders, so a surface's `ViewVisibility` is
/// that camera's verdict. A visible surface without a box (no frustum culling) takes the whole
/// target.
#[allow(clippy::type_complexity)]
fn water_screen_rect(
    mut rect: ResMut<WaterScreenRect>,
    cameras: Query<(&Camera, &GlobalTransform), With<WorldCamera>>,
    surfaces: Query<
        (
            &ViewVisibility,
            Option<&Aabb>,
            &GlobalTransform,
            &MeshMaterial3d<LiquidMaterial>,
        ),
        With<LiquidSurface>,
    >,
    materials: Res<Assets<LiquidMaterial>>,
) {
    let found = cameras.single().ok().and_then(|(camera, eye)| {
        let size = camera.physical_target_size()?;
        let clip_from_view = camera.clip_from_view();
        // Bevy's infinite reverse-Z keeps the near plane in `w_axis.z`, whatever built the matrix.
        let near = clip_from_view.w_axis.z;
        let clip_from_world = clip_from_view * eye.to_matrix().inverse();
        let water: Vec<_> = surfaces
            .iter()
            .filter(|(seen, ..)| seen.get())
            // Magma and slime are opaque and read no depth: `kind.x` is the fullbright lane.
            .filter(|(.., m)| {
                materials
                    .get(&m.0)
                    .is_some_and(|m| m.extension.kind.x <= 0.5)
            })
            .map(|(_, aabb, transform, _)| aabb.map(|b| world_corners(b, transform)))
            .collect();
        if water.iter().any(Option::is_none) {
            return Some((UVec2::ZERO, size));
        }
        project_rect(
            water.into_iter().flatten(),
            clip_from_world,
            near,
            size,
            WATER_RECT_MARGIN_PX,
        )
    });
    let next = match found {
        Some((min, max)) => WaterScreenRect {
            visible: true,
            min,
            max,
        },
        None => WaterScreenRect::default(),
    };
    trace!("water rect {:?}", (next.visible, next.min, next.max));
    rect.set_if_neq(next);
}

pub(super) fn register(app: &mut App) {
    app.add_plugins(ExtractComponentPlugin::<WaterWorldView>::default())
        .add_plugins(ExtractResourcePlugin::<WaterScreenRect>::default())
        .init_resource::<WaterScreenRect>()
        .add_systems(Update, mark_world_view)
        .add_systems(
            PostUpdate,
            water_screen_rect.after(VisibilitySystems::MarkNewlyHiddenEntitiesInvisible),
        );
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

    /// Armed with a marked view rendered, exactly one visit each; unarmed (the look off, or no
    /// water on screen), none; armed with no marked view rendered, nothing owed.
    #[test]
    fn the_world_view_is_served_once_by_each_pass() {
        assert_eq!(served_once(true, true, (1, 1)), Some(true));
        assert_eq!(served_once(true, true, (0, 1)), Some(false));
        assert_eq!(served_once(true, true, (1, 2)), Some(false));
        assert_eq!(served_once(false, true, (0, 0)), Some(true));
        assert_eq!(served_once(false, true, (1, 1)), Some(false));
        assert_eq!(served_once(false, false, (0, 1)), Some(false));
        assert_eq!(served_once(true, false, (0, 0)), None);
    }

    fn camera(size: UVec2) -> (Mat4, f32) {
        let near = 0.1;
        let proj = Mat4::perspective_infinite_reverse_rh(
            std::f32::consts::FRAC_PI_4,
            size.x as f32 / size.y as f32,
            near,
        );
        // At the origin, looking down -Z.
        (proj, near)
    }

    fn quad(center: Vec3, half: f32) -> [Vec3; 8] {
        let flat = |x: f32, z: f32| center + Vec3::new(x, 0.0, z);
        [
            flat(-half, -half),
            flat(half, -half),
            flat(-half, half),
            flat(half, half),
            flat(-half, -half),
            flat(half, -half),
            flat(-half, half),
            flat(half, half),
        ]
    }

    /// A flat square ahead and below maps to the rectangle its corners project to, grown by the
    /// margin: computed by hand from the projection.
    #[test]
    fn a_quad_ahead_maps_to_its_rectangle() {
        let size = UVec2::new(1600, 900);
        let (m, near) = camera(size);
        // A 2-yd square, 10 yd ahead, 1 yd below the eye: corners at z -9 and -11, x ±1.
        let r = project_rect(
            std::iter::once(quad(Vec3::new(0.0, -1.0, -10.0), 1.0)),
            m,
            near,
            size,
            0,
        )
        .unwrap();
        let f = 1.0 / (std::f32::consts::FRAC_PI_8).tan();
        let aspect = 1600.0 / 900.0;
        let px = |x: f32, y: f32, z: f32| {
            Vec2::new(
                (f / aspect * x / -z * 0.5 + 0.5) * 1600.0,
                (0.5 - f * y / -z * 0.5) * 900.0,
            )
        };
        // Nearest corners are the widest and the lowest; farthest the highest.
        let near_left = px(-1.0, -1.0, -9.0);
        let near_right = px(1.0, -1.0, -9.0);
        let far = px(0.0, -1.0, -11.0);
        assert_eq!(
            r.0,
            UVec2::new(near_left.x.floor() as u32, far.y.floor() as u32)
        );
        assert_eq!(
            r.1,
            UVec2::new(near_right.x.ceil() as u32, near_left.y.ceil() as u32)
        );
        let grown = project_rect(
            std::iter::once(quad(Vec3::new(0.0, -1.0, -10.0), 1.0)),
            m,
            near,
            size,
            32,
        )
        .unwrap();
        assert_eq!(grown.0, r.0 - UVec2::splat(32));
        assert_eq!(grown.1, r.1 + UVec2::splat(32));
    }

    /// A box with a corner at or behind half the near plane is the eye standing in it: the whole
    /// target.
    #[test]
    fn a_box_straddling_the_near_plane_is_the_whole_target() {
        let size = UVec2::new(1600, 900);
        let (m, near) = camera(size);
        let r = project_rect(
            std::iter::once(quad(Vec3::new(0.0, -1.0, 0.0), 5.0)),
            m,
            near,
            size,
            32,
        );
        assert_eq!(r, Some((UVec2::ZERO, size)));
    }

    /// A box wholly beside the frustum, and no box at all, are nothing on screen.
    #[test]
    fn a_box_beside_the_frustum_is_nothing() {
        let size = UVec2::new(1600, 900);
        let (m, near) = camera(size);
        let beside = quad(Vec3::new(200.0, -1.0, -10.0), 1.0);
        assert_eq!(
            project_rect(std::iter::once(beside), m, near, size, 32),
            None
        );
        assert_eq!(project_rect(std::iter::empty(), m, near, size, 32), None);
    }

    /// The margin grows the rectangle up to the target's edge and no further.
    #[test]
    fn the_margin_clamps_at_the_target_edge() {
        let size = UVec2::new(1600, 900);
        let (m, near) = camera(size);
        // Close and wide, 2 to 10 yd ahead: it runs off both sides and the bottom.
        let r = project_rect(
            std::iter::once(quad(Vec3::new(0.0, -1.0, -6.0), 4.0)),
            m,
            near,
            size,
            32,
        )
        .unwrap();
        assert_eq!((r.0.x, r.1.x, r.1.y), (0, 1600, 900));
        // Its far edge lies below the horizon (y 450), and the margin grows it upward.
        assert!(r.0.y > 450 && r.0.y < 900, "{r:?}");
    }
}
