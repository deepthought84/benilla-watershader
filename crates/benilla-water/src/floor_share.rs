//! **Whether any water on the screen can show the sky floor**, the reflection's answer where no
//! tier has a better one (`liquid.wgsl`'s `w_floor`): the floor's share of a pixel is
//! `(1 − w_p)(1 − w_m)`, the probe's and the mirrors' weights left over. Its sky cube
//! (`water_compute`'s first dispatch) and its occlusion (the depth pyramids, the plane pass, the
//! fragment's march) are worth building only where that share can be above zero.
//!
//! Each liquid surface is summarised once ([`SurfaceTiers`]) as patches about a liquid cell wide:
//! the triangles the mirrors serve whole (every vertex's mirror weight 1, the facet flat enough
//! for full trust) apart from the others, with the probe spots those read and whether any has
//! neither tier. Each frame ([`drive_floor_readers`]) a patch of a visible surface leaves the floor
//! no share when it lies outside the world camera's frustum or under the eye, when its mirror water
//! lies within an elected plane's full-trust tolerance, or when every spot it reads has a captured
//! cube at full reach. A surface whose whole mirror water and every spot are served that way is
//! passed without a look at its patches, so a sea of chunks costs one test each. Anything else, a
//! surface not summarised yet or a debug view, counts as floor on screen; so does, under the
//! grading, any visible surface with water below the eye, since every fragment the eye is above
//! marches then (`floor_occlusion_wanted`), whatever its tiers. The grade itself no longer reads
//! the verdict (it pivots on world sources only); the glints' open sky (`wg_glint_open`) does.
//!
//! The answer, [`FloorOnScreen`], has three parts. `sky` builds the cube: every stylised fragment
//! reads it for the floor's share, occluded or not, a WMO interior pool's too. `occluded`, the same
//! water outside WMO interiors with the occlusion on, arms the pyramids' reduce chain and the plane
//! pass, and the render world writes it into the look's `quality.y` lane
//! (`reflect::upload_reflect`): on a frame it is off the fragment does not march either, so the
//! floor is never occluded from a pyramid this frame did not build. `body`, on Medium and High
//! with a mirror plane active and water outside WMO interiors below the eye, arms the pyramids and
//! the plane pass for the body's sky and sun visibility, written into `quality.z` the same way:
//! off, the body reads its sky and sun open.

use benilla_assets::materials::{LiquidMaterial, ATTRIBUTE_WOW_PLANAR};
use benilla_formats::NO_SPOT;
use benilla_world::liquid::{
    LiquidSurface, WaterQuality, MIRROR2_LANES, PROBE_LANES, PROBE_SLOT_MAX,
};
use bevy::camera::primitives::{Aabb, Frustum};
use bevy::math::Affine3A;
use bevy::mesh::{Indices, VertexAttributeValues};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};

use super::hiz::WaterHizConsumers;
use super::look::WaterLook;
use super::reflect::{WaterReflectData, PLANE_TOLERANCE};
use super::world_view::WaterScreenRect;
use super::WaterStyle;
use crate::dev_state::DebugState;
use crate::view::WorldCamera;

/// `liquid.wgsl`'s `SLOPE_TRUSTED`: the facet tilt, radians, up to which a mirror's image is
/// trusted completely (a test compares them).
const SLOPE_TRUSTED: f32 = 0.03;

/// How far under [`SLOPE_TRUSTED`] a facet must stay to count as fully trusted here, radians: the
/// shader takes the tilt from screen-space derivatives of the interpolated position.
const SLOPE_MARGIN: f32 = 0.005;

/// A probe slot's reach, or the probe tier's gate, at least this counts as full: the floor's share
/// `(1 − w_p)·F` then stays under `liquid.wgsl`'s 1/1024 at any reflectance F up to 1.
const FULL: f32 = 1.0 - 1.0e-4;

/// How wide a patch of a surface's water is, in yards: about one liquid cell (4.17 yd).
const PATCH_YD: f32 = 4.0;

/// A patch of a surface's water, about one liquid cell: its world box and what it needs from the
/// reflection tiers for the sky floor to have no share of it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Patch {
    min: Vec3,
    max: Vec3,
    /// Every triangle in it is served by the mirrors alone; else by the probe `spots`.
    mirror: bool,
    /// The probe spots its triangles read, sorted, without repeats.
    spots: Vec<u16>,
    /// A triangle neither tier serves whole: a mirror weight below one or a tilt past full trust,
    /// and a probe spot missing where its share is needed.
    bare: bool,
}

/// What a liquid surface needs from the reflection tiers, patch by patch.
#[derive(Component, Clone, Debug, Default, PartialEq)]
pub(crate) struct SurfaceTiers {
    /// Its lowest water, world Y: an eye at or under it sees only the underside, which the floor
    /// occlusion leaves alone.
    low: f32,
    /// The lowest and highest mirror water, every spot and whether any patch is bare: the whole
    /// surface at once, which serves it without a look at its patches.
    mirror: Option<(f32, f32)>,
    spots: Vec<u16>,
    bare: bool,
    patches: Vec<Patch>,
    /// The placement it was summarised under.
    placed: GlobalTransform,
}

/// The default `ATTRIBUTE_WOW_PLANAR` of a mesh without it: the mirrors' water, no spots.
const NO_PLANAR: [f32; 4] = [1.0, NO_SPOT as f32, NO_SPOT as f32, 0.0];

/// Summarise a surface from its world positions, its `ATTRIBUTE_WOW_PLANAR` (mirror weight, the
/// two spots, the second's share; [`NO_PLANAR`] where absent) and its triangles, into patches by
/// each triangle's centre.
pub(crate) fn surface_tiers(
    positions: &[Vec3],
    planar: Option<&[[f32; 4]]>,
    triangles: impl Iterator<Item = [usize; 3]>,
) -> SurfaceTiers {
    let spot = |v: f32| v.round() as u16;
    let mut patches: Vec<((i32, i32, bool), Patch)> = Vec::new();
    for tri in triangles {
        let p = tri.map(|i| positions[i]);
        let a = tri.map(|i| planar.map_or(NO_PLANAR, |pl| pl[i]));
        let n = (p[1] - p[0]).cross(p[2] - p[0]);
        let len = n.length();
        let tilt = if len > 0.0 {
            (n.y.abs() / len).clamp(0.0, 1.0).acos()
        } else {
            0.0
        };
        let mirror = a.iter().all(|a| a[0] >= 1.0) && tilt <= SLOPE_TRUSTED - SLOPE_MARGIN;
        let centre = (p[0] + p[1] + p[2]) / 3.0;
        let key = (
            (centre.x / PATCH_YD).floor() as i32,
            (centre.z / PATCH_YD).floor() as i32,
            mirror,
        );
        let at = match patches.iter().position(|(k, _)| *k == key) {
            Some(at) => at,
            None => {
                patches.push((
                    key,
                    Patch {
                        min: Vec3::INFINITY,
                        max: Vec3::NEG_INFINITY,
                        mirror,
                        spots: Vec::new(),
                        bare: false,
                    },
                ));
                patches.len() - 1
            }
        };
        let patch = &mut patches[at].1;
        for v in p {
            patch.min = patch.min.min(v);
            patch.max = patch.max.max(v);
        }
        if mirror {
            continue;
        }
        // The probe's share: the first spot by `1 − share`, the second by `share`, so the first
        // must exist everywhere and the second wherever its share is above zero.
        for a in a {
            let (first, second, share) = (spot(a[1]), spot(a[2]), a[3]);
            if first == NO_SPOT || (share > 0.0 && second == NO_SPOT) {
                patch.bare = true;
            }
            for (s, needed) in [(first, true), (second, share > 0.0)] {
                if needed && s != NO_SPOT && !patch.spots.contains(&s) {
                    patch.spots.push(s);
                }
            }
        }
    }
    let mut out = SurfaceTiers {
        low: f32::INFINITY,
        ..default()
    };
    for (_, mut patch) in patches {
        patch.spots.sort_unstable();
        out.low = out.low.min(patch.min.y);
        if patch.mirror {
            let (lo, hi) = out.mirror.unwrap_or((f32::INFINITY, f32::NEG_INFINITY));
            out.mirror = Some((lo.min(patch.min.y), hi.max(patch.max.y)));
        }
        out.bare |= patch.bare;
        out.spots.extend(&patch.spots);
        out.patches.push(patch);
    }
    out.spots.sort_unstable();
    out.spots.dedup();
    out
}

/// [`surface_tiers`] of a liquid mesh placed by `transform`; `None` while the mesh lacks its
/// positions or indices.
fn mesh_tiers(mesh: &Mesh, transform: &GlobalTransform) -> Option<SurfaceTiers> {
    let Some(VertexAttributeValues::Float32x3(local)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return None;
    };
    let positions: Vec<Vec3> = local
        .iter()
        .map(|p| transform.transform_point(Vec3::from(*p)))
        .collect();
    let planar = match mesh.attribute(ATTRIBUTE_WOW_PLANAR) {
        Some(VertexAttributeValues::Float32x4(v)) => Some(v.as_slice()),
        _ => None,
    };
    let indices: Vec<usize> = match mesh.indices()? {
        Indices::U32(v) => v.iter().map(|&i| i as usize).collect(),
        Indices::U16(v) => v.iter().map(|&i| usize::from(i)).collect(),
    };
    let triangles = indices.as_chunks::<3>().0.iter().copied();
    Some(SurfaceTiers {
        placed: *transform,
        ..surface_tiers(&positions, planar, triangles)
    })
}

/// Summarise every liquid surface that has no summary yet, or whose placement moved, once its
/// mesh is in. After the water's rectangle, so after visibility and every `GlobalTransform`
/// writer it follows: a WMO pool's placement is its world one.
/// `RUST_LOG=benilla_water::floor_share=debug` logs each frame's count.
#[allow(clippy::type_complexity)]
fn summarise_surfaces(
    mut commands: Commands,
    surfaces: Query<
        (Entity, &Mesh3d, &GlobalTransform, Option<&SurfaceTiers>),
        (
            With<LiquidSurface>,
            Or<(Without<SurfaceTiers>, Changed<GlobalTransform>)>,
        ),
    >,
    meshes: Res<Assets<Mesh>>,
) {
    let mut n = 0;
    for (entity, mesh, transform, summary) in &surfaces {
        if summary.is_some_and(|s| s.placed == *transform) {
            continue;
        }
        if let Some(tiers) = meshes.get(&mesh.0).and_then(|m| mesh_tiers(m, transform)) {
            commands.entity(entity).insert(tiers);
            n += 1;
        }
    }
    if n > 0 {
        debug!("water: summarised {n} liquid surfaces");
    }
}

/// Whether any water on the world view's screen this frame can show the sky floor, under the
/// stylised look; extracted.
#[derive(Resource, Clone, Copy, Default, PartialEq, Eq, Debug, ExtractResource)]
pub(crate) struct FloorOnScreen {
    /// Some water's floor share can be above zero: every stylised fragment over such water reads
    /// the sky floor's cube, occluded or not, a WMO interior pool's included.
    pub(crate) sky: bool,
    /// And the floor is occluded there: the occlusion on, the water not a WMO interior pool's
    /// (`floor_occlusion_wanted`). It arms the pyramids' reduce chain, the plane pass and the
    /// fragment's march.
    pub(crate) occluded: bool,
    /// The body reads its sky and sun visibility from the plane pass's rays: the tier traces them
    /// (Medium, High), a mirror plane is active and water outside a WMO interior lies below the
    /// eye (`body_visibility`). It arms the pyramids' reduce chain and the plane pass.
    pub(crate) body: bool,
}

impl FloorOnScreen {
    /// From the look (`shown`: stylised, water on screen), the occlusion knob, whether a debug view
    /// wants the floor wherever water is, whether any visible water, and any visible water outside
    /// a WMO interior, leaves the floor a share or is graded from above, and whether the body's
    /// visibility rays have plane water to trace for ([`body_rays`]).
    fn decide(
        shown: bool,
        occlusion: bool,
        forced: bool,
        any: bool,
        exterior: bool,
        body: bool,
    ) -> Self {
        Self {
            sky: shown && (forced || any),
            occluded: shown && occlusion && (forced || exterior),
            body: shown && body,
        }
    }
}

/// Whether the body's visibility rays run: the tier traces them (Medium and High), a mirror plane
/// is active (they start from its points) and visible water outside a WMO interior lies below the
/// eye, or a debug view wants them wherever water is.
fn body_rays(tier: u8, plane: bool, forced: bool, below: bool) -> bool {
    tier >= 1 && plane && (forced || below)
}

/// The reflection tiers this frame, as the block's lanes hold them: the elected mirror planes,
/// and the probe spots whose cubes are in at full reach.
#[derive(Debug)]
struct Tiers {
    /// Each active mirror's plane height.
    planes: [Option<f32>; 2],
    /// Whether the probe tier is live (`probe_cfg.x`) with no slot forced on every fragment.
    probe_live: bool,
    /// The spots a full cube serves, sorted.
    full_spots: Vec<u16>,
}

impl Tiers {
    fn from_lanes(data: &[f32]) -> Self {
        let plane = |at: usize| (data[at + 1] >= 1.0).then_some(data[at]);
        let slot = |k: usize| &data[PROBE_LANES.start + 16 * k..][..16];
        // A spot is full when a slot holds it at full reach and no slot holds it at less: the
        // shader's strength sums the taps' reach times share, so a spot short of full reach, or
        // with no cube, leaves the floor its share.
        let mut full = Vec::new();
        let mut short = Vec::new();
        for k in 0..PROBE_SLOT_MAX {
            let s = slot(k);
            let (reach, mix, id) = (s[3], s[7], s[12]);
            if id < 0.0 || reach <= 0.0 {
                continue;
            }
            let id = id.round() as u16;
            if reach >= FULL && mix > 0.0 {
                full.push(id);
            } else {
                short.push(id);
            }
        }
        full.retain(|id| !short.contains(id));
        full.sort_unstable();
        full.dedup();
        Self {
            planes: [plane(0), plane(MIRROR2_LANES.start)],
            probe_live: data[32] >= FULL && data[36] < 0.0,
            full_spots: full,
        }
    }

    /// Whether an elected plane serves mirror water from `lo` to `hi` at full trust.
    fn mirrored(&self, lo: f32, hi: f32) -> bool {
        self.planes
            .iter()
            .flatten()
            .any(|p| lo >= p - PLANE_TOLERANCE && hi <= p + PLANE_TOLERANCE)
    }

    /// Whether full cubes serve every spot of `spots`, none of it bare.
    fn probed(&self, spots: &[u16], bare: bool) -> bool {
        !bare
            && (spots.is_empty()
                || (self.probe_live
                    && spots
                        .iter()
                        .all(|s| self.full_spots.binary_search(s).is_ok())))
    }

    /// Whether these tiers leave the floor no share of a patch seen from `eye_y`; `in_view` says
    /// whether a world box meets the world camera's frustum, asked only of a patch the tiers do
    /// not serve.
    fn serve(&self, patch: &Patch, eye_y: f32, in_view: &impl Fn(Vec3, Vec3) -> bool) -> bool {
        let served = if patch.mirror {
            self.mirrored(patch.min.y, patch.max.y)
        } else {
            self.probed(&patch.spots, patch.bare)
        };
        served || eye_y <= patch.min.y || !in_view(patch.min, patch.max)
    }

    /// The first patch of `surface` these tiers leave the floor a share of, seen from `eye_y`:
    /// none, without a look at the patches, where the tiers serve the whole surface.
    fn floor_patch<'a>(
        &self,
        surface: &'a SurfaceTiers,
        eye_y: f32,
        in_view: &impl Fn(Vec3, Vec3) -> bool,
    ) -> Option<&'a Patch> {
        let whole = eye_y <= surface.low
            || (surface.mirror.is_none_or(|(lo, hi)| self.mirrored(lo, hi))
                && self.probed(&surface.spots, surface.bare));
        if whole {
            return None;
        }
        surface
            .patches
            .iter()
            .find(|p| !self.serve(p, eye_y, in_view))
    }
}

/// A visible surface the tiers leave the floor a share of, and the patch (`None` while the surface
/// has no summary).
type FloorAt<'a, T> = Option<(T, Option<&'a Patch>)>;

/// Of the visible water `surfaces` (each with its summary, `None` while it has none, and whether
/// it is a WMO interior pool), the first with a patch that can show the sky floor, seen from
/// `eye_y` under `tiers`, and the first such outside a WMO interior, each with that patch. Under
/// the grading (`graded`) every fragment the eye is above marches the floor (the glints' gate
/// reads its verdict), so any surface with water below the eye counts, with no patch named.
fn floor_surfaces<'a, T: Copy>(
    tiers: &Tiers,
    eye_y: f32,
    in_view: &impl Fn(Vec3, Vec3) -> bool,
    graded: bool,
    surfaces: impl Iterator<Item = (T, Option<&'a SurfaceTiers>, bool)>,
) -> (FloorAt<'a, T>, FloorAt<'a, T>) {
    let mut any = None;
    for (t, summary, interior) in surfaces {
        let found = match summary {
            None => (t, None),
            Some(summary) => match tiers.floor_patch(summary, eye_y, in_view) {
                Some(patch) => (t, Some(patch)),
                None if graded && eye_y > summary.low => (t, None),
                None => continue,
            },
        };
        any = any.or(Some(found));
        if !interior {
            return (any, Some(found));
        }
    }
    (any, None)
}

/// Which readers want the depth pyramids: the floor occlusion while the floor is on screen, the
/// body's visibility rays while they run.
fn pyramid_readers(floor: bool, body: bool) -> WaterHizConsumers {
    let mut readers = 0;
    if floor {
        readers |= WaterHizConsumers::FLOOR_OCCLUSION;
    }
    if body {
        readers |= WaterHizConsumers::BODY_VISIBILITY;
    }
    WaterHizConsumers(readers)
}

/// Each frame, after the water's rectangle and the reflection block: whether any water on screen
/// can show the sky floor, whether it is occluded there and whether the body traces its visibility
/// ([`FloorOnScreen`]), and so which readers want the depth pyramids ([`WaterHizConsumers`]). A
/// debug view wants the floor wherever water is on screen; the grading, whose pivot reads the
/// floor's verdict and `rho_c` on every fragment the eye is above, and the body's rays, wherever
/// visible water lies below the eye. `RUST_LOG=benilla_water::floor_share=debug` logs each change.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn drive_floor_readers(
    style: Res<WaterStyle>,
    look: Res<WaterLook>,
    quality: Res<WaterQuality>,
    dev: Res<DebugState>,
    rect: Res<WaterScreenRect>,
    data: Res<WaterReflectData>,
    eye: Query<(&GlobalTransform, &Frustum), With<WorldCamera>>,
    surfaces: Query<
        (
            Entity,
            &ViewVisibility,
            Option<&SurfaceTiers>,
            &MeshMaterial3d<LiquidMaterial>,
        ),
        With<LiquidSurface>,
    >,
    materials: Res<Assets<LiquidMaterial>>,
    mut floor: ResMut<FloorOnScreen>,
    mut consumers: ResMut<WaterHizConsumers>,
) {
    let shown = style.is_stylised() && rect.visible;
    // A debug view wants the floor wherever water is; the grading wherever the eye is above it.
    let forced = dev.water.view != 0;
    let graded = look.grade_strength > 0.0;
    let tiers = Tiers::from_lanes(&data.0);
    let plane = data.0[1] > 0.0 || data.0[MIRROR2_LANES.start + 1] > 0.0;
    let (any, exterior, below) = (shown && !forced)
        .then(|| {
            let (eye, frustum) = eye.single().ok()?;
            let eye_y = eye.translation().y;
            let in_view = |min: Vec3, max: Vec3| {
                frustum.intersects_obb(
                    &Aabb::from_min_max(min, max),
                    &Affine3A::IDENTITY,
                    true,
                    true,
                )
            };
            // Magma and slime draw no reflection. A WMO interior pool reads the floor's cube
            // but its floor is never occluded (`floor_occlusion_wanted`).
            let water: Vec<_> = surfaces
                .iter()
                .filter_map(|(entity, seen, summary, material)| {
                    let m = materials.get(&material.0)?;
                    (seen.get() && m.extension.kind.x <= 0.5).then_some((
                        entity,
                        summary,
                        m.extension.path.x >= 1.5,
                    ))
                })
                .collect();
            let (any, exterior) =
                floor_surfaces(&tiers, eye_y, &in_view, graded, water.iter().copied());
            let below = water
                .iter()
                .any(|&(_, summary, interior)| !interior && summary.is_none_or(|s| eye_y > s.low));
            Some((any, exterior, below))
        })
        .flatten()
        .unwrap_or_default();
    let on = FloorOnScreen::decide(
        shown,
        look.floor_occlusion,
        forced,
        any.is_some(),
        exterior.is_some(),
        body_rays(quality.tier(), plane, forced, below),
    );
    if floor.set_if_neq(on) {
        match exterior.or(any) {
            Some((entity, patch)) => {
                debug!("water: the sky floor {on:?} at {entity} {patch:?}, under {tiers:?}")
            }
            None => debug!("water: the sky floor {on:?}"),
        }
    }
    consumers.set_if_neq(pyramid_readers(on.occluded, on.body));
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<FloorOnScreen>()
        .add_plugins(ExtractResourcePlugin::<FloorOnScreen>::default())
        .add_systems(
            PostUpdate,
            (
                summarise_surfaces.after(super::world_view::water_screen_rect),
                drive_floor_readers
                    .after(summarise_surfaces)
                    .after(super::reflect::drive_reflection),
            ),
        );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat square of two triangles at height `y`, every vertex carrying `planar`.
    fn square(y: f32, planar: [f32; 4]) -> (Vec<Vec3>, Vec<[f32; 4]>) {
        let p = vec![
            Vec3::new(0.0, y, 0.0),
            Vec3::new(4.0, y, 0.0),
            Vec3::new(0.0, y, 4.0),
            Vec3::new(4.0, y, 4.0),
        ];
        (p, vec![planar; 4])
    }

    const TRIS: [[usize; 3]; 2] = [[0, 2, 3], [0, 3, 1]];

    /// The one patch of a summary.
    fn one(t: &SurfaceTiers) -> &Patch {
        assert_eq!(t.patches.len(), 1, "{t:?}");
        &t.patches[0]
    }

    #[test]
    fn mirror_water_is_summarised_by_its_box() {
        let (p, pl) = square(10.0, [1.0, NO_SPOT as f32, NO_SPOT as f32, 0.0]);
        let t = surface_tiers(&p, Some(&pl), TRIS.into_iter());
        let patch = one(&t);
        assert!(patch.mirror && patch.spots.is_empty() && !patch.bare);
        assert_eq!(
            (patch.min, patch.max),
            (Vec3::new(0.0, 10.0, 0.0), Vec3::new(4.0, 10.0, 4.0))
        );
        assert_eq!((t.low, t.mirror, t.bare), (10.0, Some((10.0, 10.0)), false));
        // Without the attribute: the mirrors' water.
        assert_eq!(surface_tiers(&p, None, TRIS.into_iter()), t);
    }

    /// Triangles fall into patches by their centres, a cell wide, mirror water apart.
    #[test]
    fn water_is_cut_into_patches() {
        let mut p = Vec::new();
        let mut pl = Vec::new();
        for (x0, w) in [(0.0, 1.0), (4.0, 1.0), (8.0, 0.5)] {
            let (q, a) = square(1.0, [w, 7.0, NO_SPOT as f32, 0.0]);
            p.extend(q.iter().map(|v| *v + Vec3::new(x0, 0.0, 0.0)));
            pl.extend(a);
        }
        let tris = (0..3).flat_map(|k| TRIS.map(|t| t.map(|i| i + 4 * k)));
        let t = surface_tiers(&p, Some(&pl), tris);
        assert_eq!(t.patches.len(), 3, "{t:?}");
        assert_eq!(t.patches.iter().filter(|p| p.mirror).count(), 2);
        assert_eq!(
            (t.spots.clone(), t.mirror, t.low),
            (vec![7], Some((1.0, 1.0)), 1.0)
        );
        let probe = t.patches.iter().find(|p| !p.mirror).unwrap();
        assert_eq!(
            (probe.spots.clone(), probe.min.x, probe.max.x),
            (vec![7], 8.0, 12.0)
        );
    }

    #[test]
    fn probe_and_ramp_water_reads_its_spots() {
        let (p, pl) = square(3.0, [0.4, 7.0, 9.0, 0.25]);
        let t = surface_tiers(&p, Some(&pl), TRIS.into_iter());
        let patch = one(&t);
        assert_eq!(
            (patch.mirror, patch.spots.clone(), patch.bare),
            (false, vec![7, 9], false)
        );
        // A second spot with no share is not needed.
        let (_, pl) = square(3.0, [0.0, 7.0, 9.0, 0.0]);
        assert_eq!(
            one(&surface_tiers(&p, Some(&pl), TRIS.into_iter())).spots,
            vec![7]
        );
        // No spot: neither tier serves it whole.
        let (_, pl) = square(3.0, [0.5, NO_SPOT as f32, NO_SPOT as f32, 0.0]);
        assert!(one(&surface_tiers(&p, Some(&pl), TRIS.into_iter())).bare);
        // A share on a missing second spot.
        let (_, pl) = square(3.0, [0.0, 7.0, NO_SPOT as f32, 0.5]);
        assert!(one(&surface_tiers(&p, Some(&pl), TRIS.into_iter())).bare);
    }

    /// A tilted facet loses the mirror's full trust: it needs the probe, and with no spot it is
    /// bare.
    #[test]
    fn a_sloped_facet_is_not_the_mirrors_alone() {
        let (mut p, pl) = square(0.0, [1.0, NO_SPOT as f32, NO_SPOT as f32, 0.0]);
        // 0.2 yd over 4 yd: 0.05 rad, past full trust.
        p[1].y = 0.2;
        p[3].y = 0.2;
        let t = surface_tiers(&p, Some(&pl), TRIS.into_iter());
        assert!(one(&t).bare && !one(&t).mirror, "{t:?}");
        // 0.08 yd over 4 yd: 0.02 rad, trusted.
        p[1].y = 0.08;
        p[3].y = 0.08;
        let t = surface_tiers(&p, Some(&pl), TRIS.into_iter());
        assert!(one(&t).mirror && one(&t).max.y == 0.08);
    }

    /// The lanes as `drive_reflection` writes them: a plane at 10 in the first mirror, the probe
    /// live, slot 0 holding spot 7 at full reach and slot 1 spot 9 still fading in.
    fn lanes() -> Vec<f32> {
        let mut d = vec![0.0; MIRROR2_LANES.end + 4];
        d[0] = 10.0;
        d[1] = 1.0;
        d[32] = 1.0;
        d[36] = -1.0;
        for k in 0..PROBE_SLOT_MAX {
            d[PROBE_LANES.start + 16 * k + 12] = -1.0;
        }
        let slot = |d: &mut Vec<f32>, k: usize, reach: f32, id: f32| {
            let s = PROBE_LANES.start + 16 * k;
            d[s + 3] = reach;
            d[s + 7] = 1.0;
            d[s + 12] = id;
        };
        slot(&mut d, 0, 1.0, 7.0);
        slot(&mut d, 1, 0.5, 9.0);
        d
    }

    /// A patch at heights `lo..hi`.
    fn patch(lo: f32, hi: f32, mirror: bool, spots: Vec<u16>, bare: bool) -> Patch {
        Patch {
            min: Vec3::new(0.0, lo, 0.0),
            max: Vec3::new(4.0, hi, 4.0),
            mirror,
            spots,
            bare,
        }
    }

    #[test]
    fn the_tiers_serve_what_their_lanes_cover() {
        let tiers = Tiers::from_lanes(&lanes());
        assert_eq!(tiers.planes, [Some(10.0), None]);
        assert_eq!(tiers.full_spots, vec![7]);
        let seen = |_: Vec3, _: Vec3| true;
        let serve = |p: &Patch| tiers.serve(p, 30.0, &seen);
        assert!(serve(&patch(9.5, 10.5, true, vec![], false)));
        assert!(
            serve(&patch(0.5, 1.0, true, vec![], false)),
            "within the tolerance"
        );
        assert!(!serve(&patch(-5.0, -1.0, true, vec![], false)), "past it");
        assert!(serve(&patch(3.0, 3.0, false, vec![7], false)));
        assert!(
            !serve(&patch(3.0, 3.0, false, vec![7, 9], false)),
            "9 is fading in"
        );
        assert!(
            !serve(&patch(3.0, 3.0, false, vec![4], false)),
            "no cube for 4"
        );
        let bare = patch(3.0, 3.0, false, vec![], true);
        assert!(!serve(&bare));
        assert!(tiers.serve(&bare, 2.0, &seen), "seen from under it");
        assert!(
            tiers.serve(&bare, 30.0, &|_, _| false),
            "outside the frustum"
        );
        // The probe off: its spots go unserved; a second, fading slot of a full spot spoils it.
        let mut off = lanes();
        off[32] = 0.0;
        assert!(!Tiers::from_lanes(&off).serve(
            &patch(3.0, 3.0, false, vec![7], false),
            30.0,
            &seen
        ));
        let mut twin = lanes();
        let s = PROBE_LANES.start + 16 * 2;
        twin[s + 3] = 0.3;
        twin[s + 7] = 1.0;
        twin[s + 12] = 7.0;
        assert!(Tiers::from_lanes(&twin).full_spots.is_empty());
    }

    /// One patch the tiers leave the floor a share of is enough, and a surface not summarised
    /// yet counts as one.
    #[test]
    fn one_patch_with_floor_is_floor_on_screen() {
        let tiers = Tiers::from_lanes(&lanes());
        let summary = |patches: Vec<Patch>| {
            let mut s = SurfaceTiers {
                low: f32::INFINITY,
                ..default()
            };
            for p in &patches {
                s.low = s.low.min(p.min.y);
                s.bare |= p.bare;
                s.spots.extend(&p.spots);
                if p.mirror {
                    let (lo, hi) = s.mirror.unwrap_or((f32::INFINITY, f32::NEG_INFINITY));
                    s.mirror = Some((lo.min(p.min.y), hi.max(p.max.y)));
                }
            }
            s.patches = patches;
            s
        };
        let served = summary(vec![patch(10.0, 10.0, true, vec![], false)]);
        let bare = summary(vec![
            patch(10.0, 10.0, true, vec![], false),
            patch(3.0, 3.0, false, vec![], true),
        ]);
        let seen = |_: Vec3, _: Vec3| true;
        let first = |s: Vec<Option<&SurfaceTiers>>| {
            let s = s.into_iter().enumerate().map(|(i, s)| (i, s, false));
            let (any, exterior) = floor_surfaces(&tiers, 30.0, &seen, false, s);
            assert_eq!(any.map(|(i, _)| i), exterior.map(|(i, _)| i));
            any.map(|(i, p)| (i, p.is_some()))
        };
        assert_eq!(first(vec![Some(&served), Some(&served)]), None);
        assert_eq!(first(vec![Some(&served), Some(&bare)]), Some((1, true)));
        assert_eq!(first(vec![Some(&served), None]), Some((1, false)));
        assert_eq!(first(vec![]), None);
        // A patch with no cube, outside the frustum: its surface, served whole by nothing, is
        // served patch by patch.
        let probe = summary(vec![
            patch(10.0, 10.0, true, vec![], false),
            patch(3.0, 3.0, false, vec![4], false),
        ]);
        assert!(tiers.floor_patch(&probe, 30.0, &seen).is_some());
        let away = |min: Vec3, _: Vec3| min.y > 5.0;
        assert_eq!(tiers.floor_patch(&probe, 30.0, &away), None);
    }

    /// A WMO interior pool reads the floor's cube but is never occluded: an interior with floor
    /// is sky without occlusion, and the walk goes on to an exterior one.
    #[test]
    fn an_interior_pool_is_sky_without_occlusion() {
        let tiers = Tiers::from_lanes(&lanes());
        let bare = SurfaceTiers {
            low: 3.0,
            spots: vec![],
            bare: true,
            patches: vec![patch(3.0, 3.0, false, vec![], true)],
            ..default()
        };
        let served = SurfaceTiers {
            low: 10.0,
            mirror: Some((10.0, 10.0)),
            patches: vec![patch(10.0, 10.0, true, vec![], false)],
            ..default()
        };
        let seen = |_: Vec3, _: Vec3| true;
        let walk = |s: Vec<(&SurfaceTiers, bool)>| {
            let s = s.into_iter().enumerate().map(|(i, (t, x))| (i, Some(t), x));
            let (any, exterior) = floor_surfaces(&tiers, 30.0, &seen, false, s);
            (any.map(|(i, _)| i), exterior.map(|(i, _)| i))
        };
        assert_eq!(walk(vec![(&served, false), (&bare, true)]), (Some(1), None));
        assert_eq!(
            walk(vec![(&bare, true), (&served, false), (&bare, false)]),
            (Some(0), Some(2))
        );
        assert_eq!(walk(vec![(&bare, true), (&served, true)]), (Some(0), None));
        assert_eq!(walk(vec![(&served, true)]), (None, None));
        // An interior-only frame: the cube is built, nothing is occluded.
        let interior = FloorOnScreen::decide(true, true, false, true, false, false);
        assert_eq!(
            interior,
            FloorOnScreen {
                sky: true,
                occluded: false,
                body: false,
            }
        );
    }

    /// Under the grading every fragment the eye is above reads the floor's verdict, so a surface
    /// the tiers serve whole still counts while its water lies below the eye, and only then; an
    /// interior pool counts for the cube alone.
    #[test]
    fn the_grading_wants_the_floor_where_the_eye_is_above_water() {
        let tiers = Tiers::from_lanes(&lanes());
        let served = SurfaceTiers {
            low: 10.0,
            mirror: Some((10.0, 10.0)),
            patches: vec![patch(10.0, 10.0, true, vec![], false)],
            ..default()
        };
        let seen = |_: Vec3, _: Vec3| true;
        let walk = |eye_y: f32, graded: bool, s: Vec<(&SurfaceTiers, bool)>| {
            let s = s.into_iter().enumerate().map(|(i, (t, x))| (i, Some(t), x));
            let (any, exterior) = floor_surfaces(&tiers, eye_y, &seen, graded, s);
            (any.map(|(i, _)| i), exterior.map(|(i, _)| i))
        };
        assert_eq!(walk(30.0, false, vec![(&served, false)]), (None, None));
        assert_eq!(walk(30.0, true, vec![(&served, false)]), (Some(0), Some(0)));
        assert_eq!(
            walk(5.0, true, vec![(&served, false)]),
            (None, None),
            "under it"
        );
        assert_eq!(
            walk(10.0, true, vec![(&served, false)]),
            (None, None),
            "level with it"
        );
        assert_eq!(
            walk(30.0, true, vec![(&served, true), (&served, false)]),
            (Some(0), Some(1))
        );
        assert_eq!(walk(30.0, true, vec![(&served, true)]), (Some(0), None));
    }

    /// The cube is the floor's whatever the occlusion knob says; the occlusion needs the knob, and
    /// neither runs without the stylised look and water on screen; the body's rays need neither
    /// the floor nor the knob.
    #[test]
    fn the_sky_cube_does_not_wait_for_the_occlusion() {
        let on = |sky, occluded, body| FloorOnScreen {
            sky,
            occluded,
            body,
        };
        // (shown, occlusion, forced, any, exterior, body)
        for (args, want) in [
            (
                (true, true, false, true, true, false),
                on(true, true, false),
            ),
            (
                (true, false, false, true, true, false),
                on(true, false, false),
            ),
            (
                (true, false, true, false, false, false),
                on(true, false, false),
            ),
            (
                (true, true, true, false, false, false),
                on(true, true, false),
            ),
            (
                (true, true, false, false, false, false),
                FloorOnScreen::default(),
            ),
            (
                (false, true, true, true, true, true),
                FloorOnScreen::default(),
            ),
            (
                (true, false, false, false, false, true),
                on(false, false, true),
            ),
            ((true, true, false, true, true, true), on(true, true, true)),
        ] {
            let (shown, occlusion, forced, any, exterior, body) = args;
            assert_eq!(
                FloorOnScreen::decide(shown, occlusion, forced, any, exterior, body),
                want,
                "{args:?}"
            );
        }
    }

    /// The body's visibility rays run on Medium and High, from an active mirror plane's points,
    /// for water below the eye or a debug view; Low reads the sky and sun open.
    #[test]
    fn the_body_rays_run_from_medium_up_over_plane_water() {
        // (tier, plane, forced, below)
        for (args, want) in [
            ((1, true, false, true), true),
            ((2, true, false, true), true),
            ((0, true, false, true), false),
            ((1, false, false, true), false),
            ((1, true, false, false), false),
            ((2, true, true, false), true),
            ((0, true, true, true), false),
        ] {
            let (tier, plane, forced, below) = args;
            assert_eq!(body_rays(tier, plane, forced, below), want, "{args:?}");
        }
        // `below` is any shown exterior water whose surface lies under the eye; a surface with
        // no summary counts.
        let src = include_str!("floor_share.rs");
        let drive = &src[src.find("fn drive_floor_readers(").unwrap()..];
        let drive = &drive[..drive.find("\n}\n").unwrap()];
        assert!(drive.contains(
            ".any(|&(_, summary, interior)| !interior && summary.is_none_or(|s| eye_y > s.low));"
        ));
        assert!(drive.contains("body_rays(quality.tier(), plane, forced, below)"));
    }

    /// The sky floor's occlusion walks the depth pyramids, so it asks for them while the floor is
    /// on screen; so do the body's rays while they run.
    #[test]
    fn the_floor_on_screen_asks_for_the_pyramid() {
        use WaterHizConsumers as C;
        for (floor, body, want) in [
            (true, false, C::FLOOR_OCCLUSION),
            (false, false, 0),
            (false, true, C::BODY_VISIBILITY),
            (true, true, C::FLOOR_OCCLUSION | C::BODY_VISIBILITY),
        ] {
            assert_eq!(
                pyramid_readers(floor, body).0,
                want,
                "floor {floor}, body {body}"
            );
        }
    }

    /// The trust constant is the shader's.
    #[test]
    fn the_slope_trust_is_the_shaders() {
        let wgsl = include_str!("../../benilla-assets/src/shaders/liquid.wgsl");
        assert!(wgsl.contains(&format!("const SLOPE_TRUSTED: f32 = {SLOPE_TRUSTED};")));
    }
}
