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
    row(14, [cycle(clock.secs() / 256.0) * 256.0, tier, 0.0, 0.0]);
    let base = at(PV_UNDER);
    lanes.0[base..base + 64].copy_from_slice(&u);
    if let Some(mut ffx) = ffx {
        ffx.lanes = u;
        ffx.depth = if u[18] > 0.5 {
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
/// Parts nearer than this, yards, are one fall.
const FALL_JOIN: f32 = 3.0;

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
    mut last: Local<f32>,
    mut found: Local<Vec<Fall>>,
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
    if (now - *last).abs() >= 0.5 {
        *last = now;
        found.clear();
        if let Some(uv) = uv {
            for (t, aabb, mat, _seen) in &parts {
                let Some(entry) = uv.0.get(&mat.0.id()) else {
                    continue;
                };
                let Some([du, dv]) = entry.scroll() else {
                    continue;
                };
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
                if hi.y - lo.y < FALL_MIN_HEIGHT {
                    continue;
                }
                // The water under the part's middle: the lowest surface there, at its bottom.
                let w = bevy_to_wow(Vec3::new(mid.x, lo.y, mid.z));
                let over = index.over(w[0], w[1]);
                let Some(surface) = benilla_world::liquid::water_surface_at(
                    over.iter().filter_map(|&e| water.get(e).ok()),
                    [w[0], w[1], w[2] + FALL_FOOT_SLACK],
                    LiquidClaim::Unknown,
                ) else {
                    continue;
                };
                let foot_y = wow_to_bevy([w[0], w[1], surface]).y;
                if lo.y > foot_y + FALL_FOOT_SLACK || hi.y < foot_y + FALL_MIN_HEIGHT {
                    continue;
                }
                let (a, b) = if hi.x - lo.x >= hi.z - lo.z {
                    (Vec2::new(lo.x, mid.z), Vec2::new(hi.x, mid.z))
                } else {
                    (Vec2::new(mid.x, lo.z), Vec2::new(mid.x, hi.z))
                };
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
                        let t = ((p - f.a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
                        (f.a + ab * t).distance(p)
                    };
                    seg(fall.a).min(seg(fall.b)) < FALL_JOIN && (f.foot_y - fall.foot_y).abs() < 1.0
                };
                if let Some(f) = found.iter_mut().find(|f| near(f)) {
                    let pts = [f.a, f.b, fall.a, fall.b];
                    let (pl, ph) = pts
                        .iter()
                        .fold((Vec2::INFINITY, Vec2::NEG_INFINITY), |(l, h), p| {
                            (l.min(*p), h.max(*p))
                        });
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
                found.first().map_or(String::new(), |f| format!(
                    ", nearest {:.1},{:.1} to {:.1},{:.1} at y {:.1}, drop {:.1}",
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
