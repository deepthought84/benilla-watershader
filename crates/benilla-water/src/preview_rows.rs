//! The Improved Water **preview**'s runtime rows beyond the clock and the weather
//! (`super::preview`): the swimmers' wakes and their rings (`preview.wake`), the view from below
//! the surface (`preview.under`, also the improved underwater combine's uniform,
//! `benilla_world::ffx_glow::FfxUnderwater`), and the falls' foot lines (`preview.falls`), found
//! among the loaded models. Everything is derived each frame from what is loaded and where the
//! eye is; no map data.

use bevy::camera::primitives::Aabb;
use bevy::ecs::entity::EntityHashMap;
use bevy::prelude::*;

use benilla_assets::coords::{bevy_to_wow, wow_to_bevy};
use benilla_assets::materials::WowModelMaterial;
use benilla_assets::preview as lever;
use benilla_world::ffx_glow::FfxUnderwater;
use benilla_world::liquid::{
    EyeLiquid, LiquidClaim, WaterPassImages, LOOK_BODY, PREVIEW_LANES, PV_FALLS, PV_UNDER, PV_WAKE,
    PV_WEATHER, PV_WIND,
};
use benilla_world::water_fx::params::wave_push;

use super::clock::{cycle, WaterClock};
use super::preview::PreviewLanes;
use super::reflect::WaterReflectData;
use super::{WaterChunkInfo, WaterIndex, WaterQuality, WaterStyle};
use crate::view::{Viewer, WorldCamera};
use crate::world_unit::{ViewerUnit, WorldUnit};

/// `wake-underwater.wgsl`'s feature bits, which the wake's `cfg.x` and the underside's
/// `surface.w` carry.
const WU_KELVIN: u32 = 1;
const WU_BOW: u32 = 2;
const WU_COLLAR: u32 = 4;
const WU_TRAIL: u32 = 8;
const WU_RINGS: u32 = 16;
const WU_WINDOW: u32 = 32;
const WU_TINT: u32 = 64;
const WU_CAUSTICS: u32 = 128;
const WU_SHAFTS: u32 = 256;
const WU_WATERLINE: u32 = 512;
const WU_BODY_CUT: u32 = 2048;

/// The module's bits from the preview's.
fn wu_bits() -> u32 {
    let map = [
        (lever::WAKE, WU_KELVIN | WU_BOW | WU_BODY_CUT),
        (lever::WAKE_FOAM, WU_COLLAR | WU_TRAIL),
        (lever::RINGS, WU_RINGS),
        (lever::WINDOW, WU_WINDOW),
        (lever::TINT, WU_TINT),
        (lever::UW_CAUSTICS, WU_CAUSTICS),
        (lever::SHAFTS, WU_SHAFTS),
        (lever::WATERLINE, WU_WATERLINE),
    ];
    map.iter()
        .filter(|(b, _)| lever::on(*b))
        .fold(0, |m, (_, w)| m | w)
}

/// A preview block row's index into [`PreviewLanes`].
fn at(lane: usize) -> usize {
    lane - PREVIEW_LANES.start
}

// ---------------------------------------------------------------------------------------- wake

/// The swimmer slots and the ring buffer (`WU_SWIMMERS`, `WU_RINGS_MAX`).
const SWIMMERS: usize = 4;
const RINGS: usize = 8;
/// The bow point ahead of the body's centre, yards (`ripple_sim`'s `PUSH_BOW_OFFSET`).
const BOW_OFFSET: f32 = 0.55;
/// The live field's share kept in a swimmer's wedge, and the body push kept (INT-4).
const FIELD_KEEP: f32 = 0.5;
const BODY_KEEP: f32 = 0.25;
/// Heading turn rate, rad/s; speed attack and release, seconds; strength in and out, seconds.
const TURN_RATE: f32 = 3.0;
const SPEED_ATTACK: f32 = 0.15;
const SPEED_RELEASE: f32 = 0.8;
const EASE_IN: f32 = 0.3;
const EASE_OUT: f32 = 0.6;
/// A stroke ring every this many yards swum.
const STROKE_YD: f32 = 2.4;
/// The swim speed the phantom swimmer crosses the view at, yd/s.
const SWIM_SPEED: f32 = 4.72;

/// One tracked swimmer.
struct WakeTrack {
    last: Vec3,
    heading: Vec2,
    speed: f32,
    strength: f32,
    in_water: bool,
    credit: f32,
    side: f32,
}

/// The wake's state across frames.
#[derive(Default)]
struct WakeState {
    tracks: EntityHashMap<WakeTrack>,
    /// `[cx, cz, age, strength, radius, decay, foam, 0]`; strength 0 is an empty slot.
    rings: [[f32; 8]; RINGS],
    next: usize,
}

impl WakeState {
    fn ring(&mut self, at: Vec2, radius: f32, decay: f32, strength: f32, foam: f32) {
        self.rings[self.next] = [at.x, at.y, 0.0, strength, radius, decay, foam, 0.0];
        self.next = (self.next + 1) % RINGS;
    }
}

/// One slot's rows: the bow point on its surface and its strength, then heading, speed and half
/// width.
type Slot = ([f32; 4], [f32; 4]);

/// The surface height under a Bevy point, and its depth below it (surface minus the point).
fn surface_under(
    index: &WaterIndex,
    water: &Query<&WaterChunkInfo>,
    p: Vec3,
) -> Option<(f32, f32)> {
    let w = bevy_to_wow(p);
    index
        .over(w[0], w[1])
        .iter()
        .find_map(|&e| water.get(e).ok()?.surface_z_at(w[0], w[1]))
        .map(|surface| (surface, surface - w[2]))
}

/// The swimmers' and rings' rows (`preview.wake`), the avatar first, then the wading units nearest
/// the camera; `$WOW_PREVIEW_SWIMMER=<yd>` adds a phantom swimmer that far ahead of the camera,
/// crossing the view at swim speed with its strokes behind it, for a still capture.
#[allow(clippy::too_many_arguments)]
fn drive_wake_lanes(
    time: Res<Time>,
    clock: Res<WaterClock>,
    style: Res<WaterStyle>,
    viewer: Res<Viewer>,
    cam: Query<&GlobalTransform, With<WorldCamera>>,
    units: Query<(Entity, &Transform, &WorldUnit), Without<ViewerUnit>>,
    water: Query<&WaterChunkInfo>,
    index: Res<WaterIndex>,
    mut st: Local<WakeState>,
    mut lanes: ResMut<PreviewLanes>,
) {
    let base = at(PV_WAKE);
    let rows = &mut lanes.0[base..base + 104];
    rows.fill(0.0);
    let bits = wu_bits() & (WU_KELVIN | WU_BOW | WU_COLLAR | WU_TRAIL | WU_RINGS | WU_BODY_CUT);
    if !style.is_stylised() || bits == 0 {
        return;
    }
    let dt = time.delta_secs().clamp(0.0, 0.25);
    let cam_at = cam.single().map(|t| t.translation()).ok();

    // The candidates: the avatar, then every wading unit, with their heights.
    let mut subjects: Vec<(Entity, Vec3, f32, f32)> = Vec::new();
    if let Some(body) = viewer.at {
        subjects.push((Entity::PLACEHOLDER, body, viewer.height, 1.0));
    }
    for (e, t, u) in &units {
        if u.wades {
            subjects.push((e, t.translation, u.height, u.scale));
        }
    }
    let mut live: Vec<(f32, Slot)> = Vec::new();
    let mut seen: Vec<Entity> = Vec::new();
    for (entity, p, height, scale) in subjects {
        seen.push(entity);
        let surface = surface_under(&index, &water, p);
        let reach = surface.and_then(|(_, depth)| wave_push(height, depth));
        let track = st.tracks.entry(entity).or_insert(WakeTrack {
            last: p,
            heading: Vec2::Y,
            speed: 0.0,
            strength: 0.0,
            in_water: reach.is_some(),
            credit: 0.0,
            side: 1.0,
        });
        let step = Vec2::new(p.x - track.last.x, p.z - track.last.z);
        track.last = p;
        let moved = step.length();
        if dt > 0.0 {
            // The heading turns toward the step at most `TURN_RATE`, and holds while stationary.
            if moved > 1e-3 {
                let want = step / moved;
                let angle = track.heading.angle_to(want);
                let turn = angle.clamp(-TURN_RATE * dt, TURN_RATE * dt);
                track.heading = Vec2::from_angle(turn)
                    .rotate(track.heading)
                    .normalize_or(want);
            }
            let v = (moved / dt).min(12.0);
            let tau = if v > track.speed {
                SPEED_ATTACK
            } else {
                SPEED_RELEASE
            };
            track.speed += (v - track.speed) * (1.0 - (-dt / tau).exp());
            let target = reach.unwrap_or(0.0);
            let ease = if target > track.strength {
                EASE_IN
            } else {
                EASE_OUT
            };
            track.strength += (target - track.strength) * (dt / ease).min(1.0);
        }
        let s = (0.16 * height * scale).clamp(0.15, 2.0) / 0.32;
        let hw = 0.32 * s;
        // Entry and exit: a crown ring where the body crossed the surface.
        let wet = reach.is_some();
        if wet != track.in_water && bits & WU_RINGS != 0 {
            let at = Vec2::new(p.x, p.z);
            st.ring(at, 0.35 * s, 3.0, 1.0, 1.0);
        }
        let track = st.tracks.get_mut(&entity).expect("inserted above");
        track.in_water = wet;
        // Strokes: one ring per `STROKE_YD` swum, on alternating sides behind the bow.
        let mut stroke = None;
        if let Some(r) = reach {
            track.credit += moved / STROKE_YD;
            if track.credit >= 1.0 {
                track.credit = (track.credit - 1.0).min(1.0);
                track.side = -track.side;
                let fwd = track.heading;
                let side = Vec2::new(fwd.y, -fwd.x);
                let bow = Vec2::new(p.x, p.z) + fwd * BOW_OFFSET;
                stroke = Some((
                    bow - fwd * 0.9 * s + side * (0.35 * s * track.side),
                    0.8 * r,
                ));
            }
        }
        let (heading, speed, strength) = (track.heading, track.speed, track.strength);
        if let (Some(at), true) = (stroke, bits & WU_RINGS != 0) {
            st.ring(at.0, 0.07 * s, 2.5, at.1, 0.7);
        }
        let Some((surface_wow, _)) = surface else {
            continue;
        };
        if strength <= 0.01 {
            continue;
        }
        let bow = Vec2::new(p.x, p.z) + heading * BOW_OFFSET;
        let d = cam_at.map_or(0.0, |c| c.distance(p));
        live.push((
            if entity == Entity::PLACEHOLDER {
                -1.0
            } else {
                d
            },
            (
                [bow.x, surface_wow, bow.y, strength],
                [heading.x, heading.y, speed, hw],
            ),
        ));
    }
    st.tracks.retain(|e, _| seen.contains(e));

    // The phantom swimmer, ahead of the camera on the water there.
    static PHANTOM: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    let phantom = *PHANTOM.get_or_init(|| lever::number("SWIMMER", 0.0, 0.0, 200.0));
    if phantom > 0.0 {
        if let Ok(t) = cam.single() {
            let fwd = Vec2::new(t.forward().x, t.forward().z).normalize_or(Vec2::Y);
            let right = Vec2::new(-fwd.y, fwd.x);
            let c = t.translation();
            let at2 = Vec2::new(c.x, c.z) + fwd * phantom;
            if let Some((surface_wow, _)) =
                surface_under(&index, &water, Vec3::new(at2.x, c.y + 50.0, at2.y))
            {
                let heading = -right;
                live.insert(
                    0,
                    (
                        -2.0,
                        (
                            [at2.x, surface_wow, at2.y, 1.0],
                            [heading.x, heading.y, SWIM_SPEED, 0.32],
                        ),
                    ),
                );
                if bits & WU_RINGS != 0 {
                    // Its last three strokes, aged by the swim speed.
                    let side = Vec2::new(heading.y, -heading.x);
                    for k in 1..=3 {
                        let back = STROKE_YD * k as f32 - 1.5;
                        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                        let p = at2 - heading * (0.9 + back) + side * (0.35 * sign);
                        let i = RINGS - k;
                        st.rings[i] = [p.x, p.y, back / SWIM_SPEED, 0.8, 0.07, 2.5, 0.7, 0.0];
                    }
                }
            }
        }
    }
    live.sort_by(|a, b| a.0.total_cmp(&b.0));
    live.truncate(SWIMMERS);

    // The rings age, and go after four decays.
    for r in st.rings.iter_mut() {
        if r[3] > 0.0 {
            r[2] += dt;
            if r[2] > 4.0 * r[5] {
                *r = [0.0; 8];
            }
        }
    }

    // The bounding circle round every live slot and ring.
    let centre = cam_at.map_or(Vec2::ZERO, |c| Vec2::new(c.x, c.z));
    let mut reach2: f32 = 0.0;
    let mut any = false;
    for (_, (a, _)) in &live {
        reach2 = reach2.max(Vec2::new(a[0], a[2]).distance(centre) + 26.0);
        any = true;
    }
    for r in st.rings.iter().filter(|r| r[3] > 0.0) {
        let spread = 0.5 + 2.5 * (r[2] + 0.12) * (10.7283 * r[4].max(0.02) * 0.25).sqrt();
        reach2 = reach2.max(Vec2::new(r[0], r[1]).distance(centre) + spread + 1.0);
        any = true;
    }
    let keep = if bits & WU_KELVIN != 0 {
        FIELD_KEEP
    } else {
        1.0
    };
    let wake_clock = cycle(clock.secs() / 64.0) * 64.0;
    rows[0..4].copy_from_slice(&[bits as f32, wake_clock, keep, BODY_KEEP]);
    rows[4..8].copy_from_slice(&[
        centre.x,
        centre.y,
        reach2 * reach2,
        if any { 1.0 } else { 0.0 },
    ]);
    for (i, (_, (a, b))) in live.iter().enumerate() {
        rows[8 + 8 * i..12 + 8 * i].copy_from_slice(a);
        rows[12 + 8 * i..16 + 8 * i].copy_from_slice(b);
    }
    for (i, r) in st.rings.iter().enumerate() {
        rows[40 + 8 * i..48 + 8 * i].copy_from_slice(r);
    }
}

// --------------------------------------------------------------------------------------- under

/// The body's absorption shape under water, a quarter of its strength: the zone's visibility below
/// sets the distance.
const UW_ABS_SCALE: f32 = 0.25;
/// The in-scatter's extinction: 90% at the authored fog end.
const UW_INSCATTER_K: f32 = 2.3;

/// The view from below (`preview.under`) and the improved underwater combine's copy of it.
#[allow(clippy::too_many_arguments)]
fn drive_under_lanes(
    clock: Res<WaterClock>,
    style: Res<WaterStyle>,
    quality: Res<WaterQuality>,
    eye: EyeLiquid,
    cam: Query<(&GlobalTransform, &Camera, &Projection), With<WorldCamera>>,
    lighting: Res<benilla_world::lighting::WowLighting>,
    dry: Option<Res<benilla_world::lighting::DrySky>>,
    data: Option<Res<WaterReflectData>>,
    pass: Option<Res<WaterPassImages>>,
    mut lanes: ResMut<PreviewLanes>,
    ffx: Option<ResMut<FfxUnderwater>>,
) {
    let dry = dry.map(|d| *d).unwrap_or_default();
    let mut u = [0.0f32; 64];
    let mut row = |r: usize, v: [f32; 4]| u[4 * r..4 * r + 4].copy_from_slice(&v);
    if let Ok((t, camera, Projection::Perspective(p))) = cam.single() {
        let pos = t.translation();
        let (fwd, right, up) = (t.forward(), t.right(), t.up());
        let tan_v = (p.fov * 0.5).tan();
        let tan_h = tan_v * p.aspect_ratio;
        let height = camera.physical_target_size().map_or(900.0, |s| s.y as f32);
        row(0, [pos.x, pos.y, pos.z, p.near]);
        row(1, [fwd.x, fwd.y, fwd.z, 2.0 * tan_v / height.max(1.0)]);
        row(2, [right.x * tan_h, right.y * tan_h, right.z * tan_h, 0.0]);
        row(3, [up.x * tan_v, up.y * tan_v, up.z * tan_v, 0.0]);
    }
    let sub = eye.submersion();
    let in_water = sub.is_water() && style.is_stylised();
    let bits = wu_bits();
    row(
        4,
        [
            eye.surface_z(),
            eye.depth(),
            if in_water && bits & WU_TINT != 0 {
                1.0
            } else {
                0.0
            },
            bits as f32,
        ],
    );
    let body = |k: usize| -> [f32; 4] {
        let i = LOOK_BODY + 4 * k;
        data.as_ref()
            .map_or([0.0; 4], |d| [d.0[i], d.0[i + 1], d.0[i + 2], d.0[i + 3]])
    };
    let sig = body(if sub == benilla_formats::Submersion::Ocean {
        1
    } else {
        0
    });
    row(
        5,
        [
            sig[0] * UW_ABS_SCALE,
            sig[1] * UW_ABS_SCALE,
            sig[2] * UW_ABS_SCALE,
            UW_INSCATTER_K / dry.wet_fog_end.max(8.0),
        ],
    );
    let c = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
    row(6, c(dry.wet_fog_color));
    row(7, c(dry.fog_color));
    row(
        8,
        [
            dry.wet_fog_start,
            dry.wet_fog_end,
            dry.fog_start,
            dry.fog_end,
        ],
    );
    row(9, c(dry.sky_zenith));
    row(10, c(dry.sky_horizon));
    row(11, c(dry.sun_color));
    // The lighting sun refracted into the water, pointing up toward it.
    let to_sun = -lighting.sun_dir.normalize_or(Vec3::NEG_Y);
    let xz = Vec2::new(to_sun.x, to_sun.z) / 1.333;
    let y = (1.0 - xz.length_squared()).max(0.0).sqrt();
    row(12, [xz.x, y, xz.y, 0.0]);
    let (e_sky, e_sun, e_sun_foam) = (body(2), body(3), body(9));
    let luma = |v: [f32; 4]| 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
    let share = luma(e_sun) / (luma(e_sun) + luma(e_sky)).max(1e-4);
    row(13, [e_sun_foam[0], e_sun_foam[1], e_sun_foam[2], share]);
    let tier = match *quality {
        WaterQuality::Low => 0.0,
        WaterQuality::Medium => 1.0,
        WaterQuality::High => 2.0,
    };
    // Round 3, the veil (`ffx_glow.wgsl`'s `fs_combine_veil`): above the water, with `mist` or
    // `falls`, the combine lays the mist over everything the frame shows near the mirror plane and
    // the falls' spray over their feet, from the depth. `z` and `w` tell the water its own mist and
    // its pools' spray stand down while the veil draws them.
    let reflect = |i: usize| data.as_ref().map_or(0.0, |d| d.0[i]);
    let veil_on = style.is_stylised()
        && !in_water
        && (lever::on(lever::MIST) || lever::on(lever::FALLS))
        && reflect(1) > 0.0;
    row(
        14,
        [
            cycle(clock.secs() / 256.0) * 256.0,
            tier,
            if veil_on && lever::on(lever::MIST) {
                1.0
            } else {
                0.0
            },
            if veil_on && lever::on(lever::FALLS) {
                1.0
            } else {
                0.0
            },
        ],
    );
    let base = at(PV_UNDER);
    lanes.0[base..base + 64].copy_from_slice(&u);
    let mut v = [0.0f32; 64];
    if veil_on {
        let pv = |i: usize| lanes.0[at(i)];
        let mut vrow = |r: usize, x: [f32; 4]| v[4 * r..4 * r + 4].copy_from_slice(&x);
        // The mirror plane's Y, the rain and the storm; the fog's end, the mist and the falls on;
        // the visible sun; the foam light, sky then sun.
        vrow(0, [1.0, reflect(0), pv(PV_WEATHER), pv(PV_WIND + 3)]);
        vrow(
            1,
            [
                dry.fog_end,
                if lever::on(lever::MIST) { 1.0 } else { 0.0 },
                if lever::on(lever::FALLS) { 1.0 } else { 0.0 },
                0.0,
            ],
        );
        vrow(2, [reflect(4), reflect(5), reflect(6), reflect(7)]);
        let (e_sky_foam, e_sun_foam) = (body(8), body(9));
        vrow(3, [e_sky_foam[0], e_sky_foam[1], e_sky_foam[2], 0.0]);
        vrow(4, [e_sun_foam[0], e_sun_foam[1], e_sun_foam[2], 0.0]);
        // The four nearest falls' rows as `preview.falls` holds them: the foot line's ends (XZ),
        // then the pool's surface, the drop and the strength.
        let falls = at(PV_FALLS);
        v[32..64].copy_from_slice(&lanes.0[falls..falls + 32]);
    }
    if let Some(mut ffx) = ffx {
        ffx.lanes[..64].copy_from_slice(&u);
        ffx.lanes[64..].copy_from_slice(&v);
        ffx.depth = if u[18] > 0.5 || veil_on {
            pass.map(|p| p.scene_depth.clone())
        } else {
            None
        };
    }
}

// --------------------------------------------------------------------------------------- falls

/// The falls the rows hold, and how far from the eye one may be, yards.
const FALLS_MAX: usize = 8;
const FALLS_REACH: f32 = 400.0;
/// A fall model's part: at least this tall, yards, and standing in water whose surface it crosses
/// at least this far above its bottom (or its bottom within the second of the surface).
const FALL_MIN_HEIGHT: f32 = 2.5;
const FALL_FOOT_SLACK: f32 = 2.0;
/// Parts nearer than this, yards, are one fall, while the joined line stays under
/// `FALL_LINE_MAX`, which no single part may exceed either.
const FALL_JOIN: f32 = 3.0;
const FALL_LINE_MAX: f32 = 60.0;
/// The gap rule's reach from the eye, yards, how far out past a surface's end it looks for the
/// lower one, yards, and how often (seconds) or after how far a move (yards) it rescans.
const FALL_GAP_REACH: f32 = 140.0;
const FALL_GAP_OUT: [f32; 3] = [2.0, 6.0, 12.0];
const FALL_GAP_RESCAN: (f32, f32) = (4.0, 20.0);

/// Join `fall` to a fall in `found` it overlaps or nearly does while the joined line stays under
/// [`FALL_LINE_MAX`], or add it.
fn join_fall(found: &mut Vec<Fall>, fall: Fall) {
    let near = |f: &Fall| {
        let seg = |p: Vec2| {
            let ab = f.b - f.a;
            let t = ((p - f.a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
            (f.a + ab * t).distance(p)
        };
        seg(fall.a).min(seg(fall.b)) < FALL_JOIN && (f.foot_y - fall.foot_y).abs() < 1.0
    };
    let joined = |f: &Fall| {
        [f.a, f.b, fall.a, fall.b]
            .iter()
            .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(l, h), p| {
                (l.min(*p), h.max(*p))
            })
    };
    if let Some(f) = found
        .iter_mut()
        .find(|f| near(f) && (joined(f).1 - joined(f).0).max_element() <= FALL_LINE_MAX)
    {
        let (pl, ph) = joined(f);
        let mz = (pl + ph) * 0.5;
        (f.a, f.b) = if ph.x - pl.x >= ph.y - pl.y {
            (Vec2::new(pl.x, mz.y), Vec2::new(ph.x, mz.y))
        } else {
            (Vec2::new(mz.x, pl.y), Vec2::new(mz.x, ph.y))
        };
        f.drop = f.drop.max(fall.drop);
    } else {
        found.push(fall);
    }
}

/// A thick fall part's foot (round 3): its box `lo`-`hi` (Bevy) sampled on an 8×8 grid at the pool,
/// a cell wet where `surface_y` finds the pool there and the ground (`ground_y`) lies under it; the
/// line through the wet cells that have a dry neighbour, along their spread (at most
/// [`FALL_LINE_MAX`]), at their mean surface. `None` with no such cells (no rock under the part, or
/// no pool), so the caller keeps its own rule.
fn thick_foot(
    lo: Vec3,
    hi: Vec3,
    surface_y: &impl Fn(f32, f32) -> Option<f32>,
    ground_y: &impl Fn(f32, f32) -> Option<f32>,
) -> Option<(Vec2, Vec2, f32)> {
    const N: usize = 8;
    let cell = |i: usize, j: usize| {
        Vec2::new(
            lo.x + (hi.x - lo.x) * (i as f32 + 0.5) / N as f32,
            lo.z + (hi.z - lo.z) * (j as f32 + 0.5) / N as f32,
        )
    };
    let mut wet = [[None::<f32>; N]; N];
    for (i, col) in wet.iter_mut().enumerate() {
        for (j, w) in col.iter_mut().enumerate() {
            let p = cell(i, j);
            *w = surface_y(p.x, p.y).filter(|&y| ground_y(p.x, p.y).is_none_or(|g| g < y - 0.25));
        }
    }
    // The pool is the lowest water under the part; a reservoir above it (a dam's top) is not.
    let low = wet
        .iter()
        .flatten()
        .flatten()
        .fold(f32::INFINITY, |a, &y| a.min(y));
    for w in wet.iter_mut().flatten() {
        if w.is_some_and(|y| y > low + 1.5) {
            *w = None;
        }
    }
    let mut pts: Vec<(Vec2, f32)> = Vec::new();
    for i in 0..N {
        for j in 0..N {
            let Some(y) = wet[i][j] else {
                continue;
            };
            let dry = |di: i32, dj: i32| {
                let (a, b) = (i as i32 + di, j as i32 + dj);
                (0..N as i32).contains(&a)
                    && (0..N as i32).contains(&b)
                    && wet[a as usize][b as usize].is_none()
            };
            if dry(1, 0) || dry(-1, 0) || dry(0, 1) || dry(0, -1) {
                pts.push((cell(i, j), y));
            }
        }
    }
    if pts.len() < 2 {
        return None;
    }
    let n = pts.len() as f32;
    let m = pts.iter().fold(Vec2::ZERO, |acc, (p, _)| acc + *p) / n;
    let foot_y = pts.iter().map(|(_, y)| *y).sum::<f32>() / n;
    let (mut sxx, mut sxz, mut szz) = (0.0f32, 0.0f32, 0.0f32);
    for (p, _) in &pts {
        let d = *p - m;
        sxx += d.x * d.x;
        sxz += d.x * d.y;
        szz += d.y * d.y;
    }
    // The principal axis of the boundary cells.
    let ang = 0.5 * (2.0 * sxz).atan2(sxx - szz);
    let u = Vec2::new(ang.cos(), ang.sin());
    let (mut t0, mut t1) = (f32::INFINITY, f32::NEG_INFINITY);
    for (p, _) in &pts {
        let t = (*p - m).dot(u);
        t0 = t0.min(t);
        t1 = t1.max(t);
    }
    let pad = 0.5 * ((hi.x - lo.x).max(hi.z - lo.z) / N as f32);
    let (t0, t1) = (t0 - pad, t1 + pad);
    let (t0, t1) = if t1 - t0 > FALL_LINE_MAX {
        let c = 0.5 * (t0 + t1);
        (c - 0.5 * FALL_LINE_MAX, c + 0.5 * FALL_LINE_MAX)
    } else {
        (t0, t1)
    };
    Some((m + u * t0, m + u * t1, foot_y))
}

/// The longest run of `f`'s foot line, sampled every yard, where the ground (`ground_y`, Bevy Y; no
/// answer counts as water) lies at least a quarter yard under the pool's surface, padded half a
/// yard; `None` under 1.5 yd.
fn trim_fall(f: &Fall, ground_y: &impl Fn(f32, f32) -> Option<f32>) -> Option<Fall> {
    let len = f.a.distance(f.b);
    let n = (len.ceil() as usize).clamp(1, 120);
    let wet = |i: usize| {
        let p = f.a.lerp(f.b, i as f32 / n as f32);
        ground_y(p.x, p.y).is_none_or(|g| g < f.foot_y - 0.25)
    };
    let (mut best, mut run_start) = (None::<(usize, usize)>, None::<usize>);
    for i in 0..=n + 1 {
        let w = i <= n && wet(i);
        match (w, run_start) {
            (true, None) => run_start = Some(i),
            (false, Some(s0)) => {
                let e = i - 1;
                if best.is_none_or(|(b0, b1)| e - s0 > b1 - b0) {
                    best = Some((s0, e));
                }
                run_start = None;
            }
            _ => {}
        }
    }
    let (s0, e) = best?;
    let step = len / n as f32;
    if (e - s0) as f32 * step < 1.5 && len >= 1.5 {
        return None;
    }
    let u = (f.b - f.a) / len.max(1e-4);
    let t0 = (s0 as f32 * step - 0.5).max(0.0);
    let t1 = (e as f32 * step + 0.5).min(len);
    Some(Fall {
        a: f.a + u * t0,
        b: f.a + u * t1,
        ..*f
    })
}

/// One fall, in Bevy world space: its foot line's ends (XZ), the pool's surface, the drop.
#[derive(Clone, Copy)]
struct Fall {
    a: Vec2,
    b: Vec2,
    foot_y: f32,
    drop: f32,
}

/// The falls' foot lines (`preview.falls`): the loaded model parts that draw a scrolling texture
/// (a texture transform moving mostly along V), stand at least [`FALL_MIN_HEIGHT`] tall and come
/// down into water, each a line along its longer horizontal side at the water's surface. Rescanned
/// twice a second; the nearest [`FALLS_MAX`] within [`FALLS_REACH`] are kept.
#[allow(clippy::too_many_arguments)]
fn drive_fall_lanes(
    time: Res<Time>,
    style: Res<WaterStyle>,
    cam: Query<&GlobalTransform, With<WorldCamera>>,
    parts: Query<(
        &GlobalTransform,
        &Aabb,
        &MeshMaterial3d<WowModelMaterial>,
        &ViewVisibility,
    )>,
    uv: Option<Res<benilla_world::doodad_anim::UvAnimMaterials>>,
    water: Query<&WaterChunkInfo>,
    index: Res<WaterIndex>,
    ground: (
        Option<Res<benilla_world::terrain_stream::TerrainStreamer>>,
        Option<Res<Assets<benilla_assets::AdtTile>>>,
    ),
    mut last: Local<f32>,
    mut found: Local<Vec<Fall>>,
    mut gaps: Local<(Vec3, f32, Vec<Fall>)>,
    mut lanes: ResMut<PreviewLanes>,
) {
    let base = at(PV_FALLS);
    if !style.is_stylised() || !lever::on(lever::FALLS) {
        lanes.0[base..base + 64].fill(0.0);
        return;
    }
    let Ok(eye) = cam.single().map(|t| t.translation()) else {
        return;
    };
    let now = time.elapsed_secs();
    // The terrain's height (Bevy Y) under a Bevy XZ point, `None` off the streamed set.
    let ground_y = |x: f32, z: f32| -> Option<f32> {
        let (Some(st), Some(adt)) = (ground.0.as_deref(), ground.1.as_deref()) else {
            return None;
        };
        benilla_world::terrain_stream::terrain_height_under(st, adt, Vec3::new(x, 0.0, z))
    };
    if (now - *last).abs() >= 0.5 {
        *last = now;
        found.clear();
        // `$WOW_PREVIEW_FALLS_LOG=1` (round 3): every scrolling part near the eye, its box and its
        // scroll, logged whenever their number changes, to see why a visible fall finds no line.
        static FALLS_LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let log_parts = *FALLS_LOG.get_or_init(|| lever::number("FALLS_LOG", 0.0, 0.0, 1.0) > 0.5);
        let mut logged: Vec<String> = Vec::new();
        if let Some(uv) = uv {
            for (t, aabb, mat, _seen) in &parts {
                let Some(entry) = uv.0.get(&mat.0.id()) else {
                    continue;
                };
                let Some([du, dv]) = entry.scroll() else {
                    continue;
                };
                if log_parts {
                    let (c, h) = (Vec3::from(aabb.center), Vec3::from(aabb.half_extents));
                    let p = t.transform_point(c);
                    let ext = (t.affine().matrix3.abs() * h) * 2.0;
                    if Vec2::new(p.x - eye.x, p.z - eye.z).length() < 150.0 {
                        logged.push(format!(
                            "({:.1},{:.1},{:.1}) size {:.1}x{:.1}x{:.1} scroll {du:.3},{dv:.3}",
                            p.x, p.y, p.z, ext.x, ext.y, ext.z
                        ));
                    }
                }
                if dv.abs() < 1e-3 || dv.abs() < 1.5 * du.abs() {
                    continue;
                }
                let (c, h) = (Vec3::from(aabb.center), Vec3::from(aabb.half_extents));
                let mut lo = Vec3::INFINITY;
                let mut hi = Vec3::NEG_INFINITY;
                for i in 0..8 {
                    let s = Vec3::new(
                        if i & 1 == 0 { -1.0 } else { 1.0 },
                        if i & 2 == 0 { -1.0 } else { 1.0 },
                        if i & 4 == 0 { -1.0 } else { 1.0 },
                    );
                    let p = t.transform_point(c + s * h);
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
                let mid = (lo + hi) * 0.5;
                if Vec2::new(mid.x - eye.x, mid.z - eye.z).length() > FALLS_REACH {
                    continue;
                }
                // A curtain: tall, thin across, and no wider than a fall is.
                let (long, short) = (
                    (hi.x - lo.x).max(hi.z - lo.z),
                    (hi.x - lo.x).min(hi.z - lo.z),
                );
                // Round 3: a very tall curtain (a dam's spillway, half again as tall as it is long)
                // may be longer than a fall line; its line is cut to one about its middle below,
                // and trimmed to the pool with every other.
                let towering = hi.y - lo.y >= 1.5 * long && long <= 2.5 * FALL_LINE_MAX;
                if hi.y - lo.y < FALL_MIN_HEIGHT
                    || short > (0.5 * (hi.y - lo.y)).max(4.0)
                    || (long > FALL_LINE_MAX && !towering)
                {
                    continue;
                }
                // The water at the part's bottom at (x, z): the lowest surface there, Bevy Y.
                let surface_y = |x: f32, z: f32| {
                    let w = bevy_to_wow(Vec3::new(x, lo.y, z));
                    let over = index.over(w[0], w[1]);
                    benilla_world::liquid::water_surface_at(
                        over.iter().filter_map(|&e| water.get(e).ok()),
                        [w[0], w[1], w[2] + FALL_FOOT_SLACK],
                        LiquidClaim::Unknown,
                    )
                    .map(|s| wow_to_bevy([w[0], w[1], s]).y)
                    .filter(|&y| lo.y <= y + FALL_FOOT_SLACK && hi.y >= y + FALL_MIN_HEIGHT)
                };
                // Round 3, a thick part (a cascade or a spillway, a box deep both ways, which is
                // every fall model at Elwynn and the Stonewrought Dam): its foot is where the pool
                // meets the rock under it. Its footprint, sampled on a grid at the pool, wet where
                // the pool lies over its bed; the line fitted to the wet cells beside dry ones,
                // along their spread. A thin curtain, or a part standing wholly in water, keeps
                // the line along its longer side at its middle. (Round 2's "longer side" put a
                // spillway's line along its flow, out in the pool.)
                let thick = short > (0.15 * (hi.y - lo.y)).max(6.0);
                let grid = if thick {
                    thick_foot(lo, hi, &surface_y, &ground_y)
                } else {
                    None
                };
                let (a, b, foot_y) = match grid {
                    Some(g) => g,
                    None => {
                        let along_x = hi.x - lo.x >= hi.z - lo.z;
                        let side = if along_x { mid.z } else { mid.x };
                        let Some(foot_y) = surface_y(mid.x, mid.z) else {
                            continue;
                        };
                        // The line along the part's longer side at its foot.
                        let (l0, l1) = if along_x { (lo.x, hi.x) } else { (lo.z, hi.z) };
                        let (l0, l1) = if l1 - l0 > FALL_LINE_MAX {
                            let m = 0.5 * (l0 + l1);
                            (m - 0.5 * FALL_LINE_MAX, m + 0.5 * FALL_LINE_MAX)
                        } else {
                            (l0, l1)
                        };
                        let pt = |t: f32| {
                            if along_x {
                                Vec2::new(t, side)
                            } else {
                                Vec2::new(side, t)
                            }
                        };
                        (pt(l0), pt(l1), foot_y)
                    }
                };
                {
                    let fall = Fall {
                        a,
                        b,
                        foot_y,
                        drop: hi.y - foot_y,
                    };
                    // Join a part to a fall it overlaps or nearly does.
                    let near = |f: &Fall| {
                        let seg = |p: Vec2| {
                            let ab = f.b - f.a;
                            let t =
                                ((p - f.a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
                            (f.a + ab * t).distance(p)
                        };
                        seg(fall.a).min(seg(fall.b)) < FALL_JOIN
                            && (f.foot_y - fall.foot_y).abs() < 1.0
                    };
                    let joined = |f: &Fall| {
                        [f.a, f.b, fall.a, fall.b]
                            .iter()
                            .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(l, h), p| {
                                (l.min(*p), h.max(*p))
                            })
                    };
                    let fits = |f: &Fall| {
                        let (pl, ph) = joined(f);
                        (ph - pl).max_element() <= FALL_LINE_MAX
                    };
                    if let Some(f) = found.iter_mut().find(|f| near(f) && fits(f)) {
                        let (pl, ph) = joined(f);
                        let mz = (pl + ph) * 0.5;
                        (f.a, f.b) = if ph.x - pl.x >= ph.y - pl.y {
                            (Vec2::new(pl.x, mz.y), Vec2::new(ph.x, mz.y))
                        } else {
                            (Vec2::new(mz.x, pl.y), Vec2::new(mz.x, ph.y))
                        };
                        f.drop = f.drop.max(fall.drop);
                    } else {
                        found.push(fall);
                    }
                }
            }
        }
        if log_parts {
            static LOGGED: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(usize::MAX);
            if LOGGED.swap(logged.len(), std::sync::atomic::Ordering::Relaxed) != logged.len() {
                info!(
                    "preview falls: {} scrolling parts near the eye: {}",
                    logged.len(),
                    logged.join("; ")
                );
            }
        }
        // The classification's gap rule (RIV-4): where a water surface ends and another lies at
        // least `FALL_MIN_HEIGHT` below a few yards further out, water falls over that edge; its
        // foot line runs along the edge on the lower surface. Rescanned every few seconds or
        // after a move, over the wet cells within `FALL_GAP_REACH` of the eye.
        if now - gaps.1 >= FALL_GAP_RESCAN.0 || gaps.0.distance(eye) >= FALL_GAP_RESCAN.1 {
            gaps.0 = eye;
            gaps.1 = now;
            gaps.2.clear();
            let ew = bevy_to_wow(eye);
            let surface = |x: f32, y: f32, z: f32| {
                benilla_world::liquid::water_surface_at(
                    index.over(x, y).iter().filter_map(|&e| water.get(e).ok()),
                    [x, y, z],
                    LiquidClaim::Unknown,
                )
            };
            let mut edges: Vec<Fall> = Vec::new();
            let r = FALL_GAP_REACH;
            for e in index.over_box([ew[0] - r, ew[1] - r], [ew[0] + r, ew[1] + r]) {
                let Ok(chunk) = water.get(e) else {
                    continue;
                };
                chunk.for_each_wet_cell(|c| {
                    let cx = 0.25 * (c[0][0] + c[1][0] + c[2][0] + c[3][0]);
                    let cy = 0.25 * (c[0][1] + c[1][1] + c[2][1] + c[3][1]);
                    for (a, b) in [(c[0], c[1]), (c[1], c[3]), (c[3], c[2]), (c[2], c[0])] {
                        let m = Vec2::new(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]));
                        if m.distance(Vec2::new(ew[0], ew[1])) > r {
                            continue;
                        }
                        let top = 0.5 * (a[2] + b[2]);
                        let d = (m - Vec2::new(cx, cy)).normalize_or_zero();
                        // The surface goes on past the edge at its own height: no end here.
                        let p1 = m + d;
                        if surface(p1.x, p1.y, top - 0.5).is_some_and(|z| z <= top + 0.5) {
                            continue;
                        }
                        // Round 3, the lip: water pours over an edge only where its bed lies under
                        // the surface at the edge and no ground rises just past it. An upper pool
                        // that ends against its bank (a whole cliff top at Elwynn) is not a fall.
                        let bed = |q: Vec2| {
                            let b = wow_to_bevy([q.x, q.y, 0.0]);
                            ground_y(b.x, b.z)
                        };
                        if bed(m - d * 0.6).is_some_and(|z| z > top - 0.25)
                            || bed(m + d).is_some_and(|z| z > top - 0.1)
                        {
                            continue;
                        }
                        for out in FALL_GAP_OUT {
                            let p = m + d * out;
                            let Some(low) = surface(p.x, p.y, top - 300.0) else {
                                continue;
                            };
                            let drop = top - low;
                            if !(FALL_MIN_HEIGHT..=150.0).contains(&drop) {
                                continue;
                            }
                            let ba = wow_to_bevy([a[0] + d.x * out, a[1] + d.y * out, low]);
                            let bb = wow_to_bevy([b[0] + d.x * out, b[1] + d.y * out, low]);
                            edges.push(Fall {
                                a: Vec2::new(ba.x, ba.z),
                                b: Vec2::new(bb.x, bb.z),
                                foot_y: ba.y,
                                drop,
                            });
                            break;
                        }
                    }
                });
            }
            for f in edges {
                join_fall(&mut gaps.2, f);
            }
        }
        for f in gaps.2.iter().copied() {
            join_fall(&mut found, f);
        }
        // Round 3: each line keeps its longest run where the pool has water over its bed; a line
        // crossing ground that stands above the pool (a bank, a cliff's foot) loses that span.
        let trimmed: Vec<Fall> = found
            .iter()
            .filter_map(|f| trim_fall(f, &ground_y))
            .collect();
        *found = trimmed;
        let e2 = Vec2::new(eye.x, eye.z);
        found.sort_by(|p, q| {
            let d = |f: &Fall| (f.a + f.b).distance(e2 * 2.0);
            d(p).total_cmp(&d(q))
        });
        found.truncate(FALLS_MAX);
        // Logged when the count changes, with the nearest line.
        static SEEN: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(usize::MAX);
        if SEEN.swap(found.len(), std::sync::atomic::Ordering::Relaxed) != found.len() {
            info!(
                "preview falls: {} foot lines{}",
                found.len(),
                found.iter().fold(String::new(), |acc, f| format!(
                    "{acc}; {:.1},{:.1} to {:.1},{:.1} at y {:.1}, drop {:.1}",
                    f.a.x, f.a.y, f.b.x, f.b.y, f.foot_y, f.drop
                ))
            );
        }
    }
    let rows = &mut lanes.0[base..base + 64];
    rows.fill(0.0);
    for (i, f) in found.iter().enumerate() {
        rows[8 * i..8 * i + 4].copy_from_slice(&[f.a.x, f.a.y, f.b.x, f.b.y]);
        rows[8 * i + 4..8 * i + 8].copy_from_slice(&[f.foot_y, f.drop, 0.0, 1.0]);
    }
}

pub(super) fn register(app: &mut App) {
    app.add_systems(
        PostUpdate,
        (drive_wake_lanes, drive_under_lanes, drive_fall_lanes)
            .before(super::reflect::drive_reflection),
    );
}
