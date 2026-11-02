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
    EyeLiquid, LiquidClaim, WaterPassImages, LOOK_BODY, PREVIEW_LANES, PV_CLOCKS, PV_DRIFT,
    PV_FALLS, PV_GAINS, PV_UNDER, PV_WAKE, PV_WEATHER, PV_WIND,
};
use benilla_world::water_fx::params::wave_push;

use super::clock::{cycle, WaterClock};
use super::mcsh::{WaterShadow, WaterShadowWindow};
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
        // One caustic net, seen from above and from below.
        (lever::CAUSTICS, WU_CAUSTICS),
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
/// A swimmer's waves keep the speed they were made at: it follows the body's at once when the body
/// quickens, relaxes toward it over `WAVE_RELAX` seconds while the body holds between `WAVE_HOLD`
/// of it and it, and under that share holds while the waves fade out over `WAVE_FADE` seconds;
/// the pattern takes the body's speed again once they have gone. Without the hold the Kelvin
/// field, whose wavelength is 2πU²/g, was redrawn from the body's easing speed and shrank onto a
/// stopping swimmer as a lattice of fine cells. (A test against the previous frame's speed is no
/// hold: an exponential release loses two percent a frame and passes any per-frame tolerance.)
const WAVE_HOLD: f32 = 0.7;
const WAVE_RELAX: f32 = 3.0;
const WAVE_FADE: f32 = 0.5;
/// The phantom swimmer's height and how deep its body floats, yards: a human's, at the reference
/// gate's full strength.
const PHANTOM_HEIGHT: f32 = 2.03;
const PHANTOM_FLOAT: f32 = 1.52;

/// The phantom swimmer's key in the trackers (the avatar's is `Entity::PLACEHOLDER`).
pub(crate) const PHANTOM: Entity = match Entity::from_raw_u32(u32::MAX - 1) {
    Some(e) => e,
    None => unreachable!(),
};

/// The phantom swimmer's body this frame, for the wave simulation to push on as it pushes on the
/// avatar's: its position (Bevy) and height. `None` unless the phantom swims its line
/// (`$WOW_PREVIEW_SWIMMER_PASS`, captures only).
#[derive(Resource, Default)]
pub(crate) struct PhantomBody(pub(crate) Option<(Vec3, f32)>);

/// One tracked swimmer.
struct WakeTrack {
    last: Vec3,
    heading: Vec2,
    speed: f32,
    strength: f32,
    in_water: bool,
    credit: f32,
    side: f32,
    /// The speed its waves were made at and their share left ([`WAVE_HOLD`]).
    wave_u: f32,
    wave: f32,
}

/// The wake's state across frames.
#[derive(Default)]
struct WakeState {
    tracks: EntityHashMap<WakeTrack>,
    /// `[cx, cz, age, strength, radius, decay, foam, 0]`; strength 0 is an empty slot.
    rings: [[f32; 8]; RINGS],
    next: usize,
    /// The phantom's swim along its fixed line ([`PhantomSwim`]), once it has one.
    swim: Option<PhantomSwim>,
}

/// The phantom swimmer's straight line (`$WOW_PREVIEW_SWIMMER_PASS`, captures only): `anchor` the
/// point it passes at `pass` seconds of virtual time, `heading` its course, `t` the virtual time so
/// far, and `stop` the time it halts where it is (`$WOW_PREVIEW_SWIMMER_STOP`; none, it swims on),
/// so a burst shows the water settle after a swimmer.
struct PhantomSwim {
    anchor: Vec2,
    heading: Vec2,
    pass: f32,
    t: f32,
    stop: Option<f32>,
}

/// `$WOW_PREVIEW_SWIMMER_PASS=<s>`, read once: the phantom swims a fixed line through the world,
/// passing its anchor `<s>` seconds of virtual time after the first frame, instead of riding the
/// camera; `None` unless set, and unless this run is a capture (`$WOW_CAPTURE`), so play never
/// sees it. A golden scene ages 1000 steps of 1/60 s, so 16.67 puts it on the anchor at the shot.
fn phantom_pass() -> Option<f32> {
    static PASS: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *PASS.get_or_init(|| {
        let pass = lever::number("SWIMMER_PASS", -1.0, 0.0, 3600.0);
        (pass >= 0.0 && std::env::var_os("WOW_CAPTURE").is_some()).then_some(pass)
    })
}

/// `$WOW_PREVIEW_SWIMMER_STOP=<s>`, read once: the virtual time the phantom halts at.
fn phantom_stop() -> Option<f32> {
    static STOP: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *STOP.get_or_init(|| {
        let stop = lever::number("SWIMMER_STOP", -1.0, 0.0, 3600.0);
        (stop >= 0.0).then_some(stop)
    })
}

impl WakeState {
    fn ring(&mut self, at: Vec2, radius: f32, decay: f32, strength: f32, foam: f32) {
        self.rings[self.next] = [at.x, at.y, 0.0, strength, radius, decay, foam, 0.0];
        self.next = (self.next + 1) % RINGS;
    }
}

/// One slot's rows: the bow point on its surface and its strength; heading, speed and half width;
/// the waves' speed and share.
type Slot = ([f32; 4], [f32; 4], [f32; 4]);

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
    mut phantom_body: ResMut<PhantomBody>,
) {
    let base = at(PV_WAKE);
    let rows = &mut lanes.0[base..base + 120];
    rows.fill(0.0);
    phantom_body.0 = None;
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
    // The phantom swimmer on its fixed line (`$WOW_PREVIEW_SWIMMER_PASS`): a body like the
    // avatar's, tracked, stroking and stopping as one, and the wave field's to push on.
    static PHANTOM_YD: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    let phantom = *PHANTOM_YD.get_or_init(|| lever::number("SWIMMER", 0.0, 0.0, 200.0));
    let cam_frame = cam.single().ok().map(|t| {
        let fwd = Vec2::new(t.forward().x, t.forward().z).normalize_or(Vec2::Y);
        let right = Vec2::new(-fwd.y, fwd.x);
        let c = t.translation();
        (c, fwd, right)
    });
    if let (true, Some(pass), Some((c, fwd, right))) = (phantom > 0.0, phantom_pass(), cam_frame) {
        let ahead = Vec2::new(c.x, c.z) + fwd * phantom;
        let sw = st.swim.get_or_insert(PhantomSwim {
            anchor: ahead,
            heading: -right,
            pass,
            t: 0.0,
            stop: phantom_stop(),
        });
        // Laid from the view as the clock starts, after the scene's still settle.
        if sw.t == 0.0 {
            sw.anchor = ahead;
            sw.heading = -right;
        }
        sw.t += dt;
        let along = sw.stop.map_or(sw.t, |stop| sw.t.min(stop));
        let at2 = sw.anchor + sw.heading * (SWIM_SPEED * (along - sw.pass));
        if let Some((surface_wow, _)) =
            surface_under(&index, &water, Vec3::new(at2.x, c.y + 50.0, at2.y))
        {
            let body = Vec3::new(at2.x, surface_wow - PHANTOM_FLOAT, at2.y);
            subjects.push((PHANTOM, body, PHANTOM_HEIGHT, 1.0));
            phantom_body.0 = Some((body, PHANTOM_HEIGHT));
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
            wave_u: 0.0,
            wave: 0.0,
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
            // The waves' speed follows the body's up at once and down only slowly; when the body
            // slows past `WAVE_HOLD` of it the waves keep their speed and fade, and take the
            // body's again once gone.
            if track.speed >= track.wave_u {
                track.wave_u = track.speed;
            } else if track.speed >= WAVE_HOLD * track.wave_u {
                track.wave_u += (track.speed - track.wave_u) * (dt / WAVE_RELAX).min(1.0);
            }
            let want = if track.speed >= WAVE_HOLD * track.wave_u {
                1.0
            } else {
                0.0
            };
            let ease = if want > track.wave { EASE_IN } else { WAVE_FADE };
            track.wave += (want - track.wave) * (dt / ease).min(1.0);
            if track.wave < 0.02 {
                track.wave_u = track.speed;
            }
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
        let (wave_u, wave) = (track.wave_u, track.wave);
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
            } else if entity == PHANTOM {
                -2.0
            } else {
                d
            },
            (
                [bow.x, surface_wow, bow.y, strength],
                [heading.x, heading.y, speed, hw],
                [wave_u, wave, 0.0, 0.0],
            ),
        ));
    }
    st.tracks.retain(|e, _| seen.contains(e));

    // The still phantom (`$WOW_PREVIEW_SWIMMER` without a pass time): ahead of the camera on the
    // water there, under way at swim speed with its last three strokes behind it, for a still
    // capture of the wake.
    if let (true, None, Some((c, fwd, right))) = (phantom > 0.0, phantom_pass(), cam_frame) {
        let at2 = Vec2::new(c.x, c.z) + fwd * phantom;
        let heading = -right;
        if let Some((surface_wow, _)) =
            surface_under(&index, &water, Vec3::new(at2.x, c.y + 50.0, at2.y))
        {
            live.insert(
                0,
                (
                    -2.0,
                    (
                        [at2.x, surface_wow, at2.y, 1.0],
                        [heading.x, heading.y, SWIM_SPEED, 0.32],
                        [SWIM_SPEED, 1.0, 0.0, 0.0],
                    ),
                ),
            );
            if bits & WU_RINGS != 0 {
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
    for (_, (a, _, _)) in &live {
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
    for (i, (_, (a, b, c))) in live.iter().enumerate() {
        rows[8 + 12 * i..12 + 12 * i].copy_from_slice(a);
        rows[12 + 12 * i..16 + 12 * i].copy_from_slice(b);
        rows[16 + 12 * i..20 + 12 * i].copy_from_slice(c);
    }
    for (i, r) in st.rings.iter().enumerate() {
        rows[56 + 8 * i..64 + 8 * i].copy_from_slice(r);
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
    shadow: (Res<WaterShadow>, Res<WaterShadowWindow>),
    mut lanes: ResMut<PreviewLanes>,
    ffx: Option<ResMut<FfxUnderwater>>,
) {
    let dry = dry.map(|d| *d).unwrap_or_default();
    let mut u = [0.0f32; 64];
    // The caustic net from below rides the drift the water above lights its bed with: the 3.6-yd
    // ripple octave's (the wind's under `waves`, today's otherwise), in the camera rows' spare
    // lanes; its clock rides the refracted sun's row.
    let drift = if lever::on(lever::WAVES) {
        let d = at(PV_DRIFT) + 6;
        [lanes.0[d], lanes.0[d + 1]]
    } else {
        let c = clock.lanes();
        [c[6], c[7]]
    };
    let mut row = |r: usize, v: [f32; 4]| u[4 * r..4 * r + 4].copy_from_slice(&v);
    if let Ok((t, camera, Projection::Perspective(p))) = cam.single() {
        let pos = t.translation();
        let (fwd, right, up) = (t.forward(), t.right(), t.up());
        let tan_v = (p.fov * 0.5).tan();
        let tan_h = tan_v * p.aspect_ratio;
        let height = camera.physical_target_size().map_or(900.0, |s| s.y as f32);
        row(0, [pos.x, pos.y, pos.z, p.near]);
        row(1, [fwd.x, fwd.y, fwd.z, 2.0 * tan_v / height.max(1.0)]);
        row(
            2,
            [right.x * tan_h, right.y * tan_h, right.z * tan_h, drift[0]],
        );
        row(3, [up.x * tan_v, up.y * tan_v, up.z * tan_v, drift[1]]);
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
    let caustics = if lever::on(lever::CAUSTICS) {
        lanes.0[at(PV_GAINS)]
    } else {
        0.0
    };
    // The kind's turbidity and the net's gain ride the fog rows' spare lanes.
    let wet = dry.wet_fog_color;
    row(6, [wet[0], wet[1], wet[2], sig[3]]);
    let fog = dry.fog_color;
    row(7, [fog[0], fog[1], fog[2], caustics]);
    row(
        8,
        [
            dry.wet_fog_start,
            dry.wet_fog_end,
            dry.fog_start,
            dry.fog_end,
        ],
    );
    // The sun's direct share's inputs ride the sky and sun rows' spare lanes: the share the sun
    // keeps in shadow, the open sky's luminance and the sun's on the body (`nwg_sun_share`).
    let (e_sky, e_sun) = (body(2), body(3));
    let luma = |v: [f32; 4]| 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
    let (zen, hor, sun) = (dry.sky_zenith, dry.sky_horizon, dry.sun_color);
    row(9, [zen[0], zen[1], zen[2], e_sky[3]]);
    row(10, [hor[0], hor[1], hor[2], luma(e_sky)]);
    row(11, [sun[0], sun[1], sun[2], luma(e_sun)]);
    // The lighting sun refracted into the water, pointing up toward it.
    let to_sun = -lighting.sun_dir.normalize_or(Vec3::NEG_Y);
    let xz = Vec2::new(to_sun.x, to_sun.z) / 1.333;
    let y = (1.0 - xz.length_squared()).max(0.0).sqrt();
    // Its w: the caustic net's clock, the weather clock's phase (`preview.clocks.y`).
    row(12, [xz.x, y, xz.y, lanes.0[at(PV_CLOCKS) + 1]]);
    let e_sun_foam = body(9);
    let share = luma(e_sun) / (luma(e_sun) + luma(e_sky)).max(1e-4);
    row(13, [e_sun_foam[0], e_sun_foam[1], e_sun_foam[2], share]);
    let tier = match *quality {
        WaterQuality::Low => 0.0,
        WaterQuality::Medium => 1.0,
        WaterQuality::High => 2.0,
    };
    // Round 3, the veil (`ffx_glow.wgsl`'s `fs_combine_veil`): above the water, with `mist` or
    // `falls`, the combine lays the mist over everything the frame shows near the mirror plane and
    // the falls' spray over their feet, from the depth. The mist lies on the mirror plane, so it
    // needs one elected (its strength, `reflect(1)`); the spray needs only a fall among the four
    // nearest rows it copies (`preview.falls`, strength in lane 7 of each). `z` and `w` tell the
    // water its own mist and its pools' spray stand down while the veil draws them.
    let reflect = |i: usize| data.as_ref().map_or(0.0, |d| d.0[i]);
    let pv_falls = at(PV_FALLS);
    let mist_on = lever::on(lever::MIST) && reflect(1) > 0.0;
    let falls_on = lever::on(lever::FALLS) && (0..4).any(|i| lanes.0[pv_falls + 8 * i + 7] > 0.0);
    let veil_on = style.is_stylised() && !in_water && (mist_on || falls_on);
    // `$WOW_PREVIEW_FALLS_LOG=1`: the veil's arming, logged when it changes.
    static VEIL_LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *VEIL_LOG.get_or_init(|| lever::number("FALLS_LOG", 0.0, 0.0, 1.0) > 0.5) {
        static VEIL_SEEN: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
        let text = format!(
            "veil {veil_on}: mirror plane {:.1} strength {:.2}, in water {in_water}, mist {mist_on}, \
             falls {falls_on}",
            reflect(0),
            reflect(1)
        );
        if let Ok(mut seen) = VEIL_SEEN.lock() {
            if *seen != text {
                info!("preview {text}");
                *seen = text;
            }
        }
    }
    let flag = |b: bool| if b { 1.0 } else { 0.0 };
    row(
        14,
        [
            cycle(clock.secs() / 256.0) * 256.0,
            tier,
            flag(veil_on && mist_on),
            flag(veil_on && falls_on),
        ],
    );
    row(15, shadow.1.lanes());
    let base = at(PV_UNDER);
    lanes.0[base..base + 64].copy_from_slice(&u);
    let mut v = [0.0f32; 64];
    if veil_on {
        let pv = |i: usize| lanes.0[at(i)];
        let mut vrow = |r: usize, x: [f32; 4]| v[4 * r..4 * r + 4].copy_from_slice(&x);
        // The mirror plane's Y, the rain and the storm; the fog's end, the mist and the falls on;
        // the visible sun; the foam light, sky then sun.
        vrow(0, [1.0, reflect(0), pv(PV_WEATHER), pv(PV_WIND + 3)]);
        vrow(1, [dry.fog_end, flag(mist_on), flag(falls_on), 0.0]);
        vrow(2, [reflect(4), reflect(5), reflect(6), reflect(7)]);
        let (e_sky_foam, e_sun_foam) = (body(8), body(9));
        vrow(3, [e_sky_foam[0], e_sky_foam[1], e_sky_foam[2], 0.0]);
        vrow(4, [e_sun_foam[0], e_sun_foam[1], e_sun_foam[2], 0.0]);
        // The four nearest falls' rows as `preview.falls` holds them: the foot line's ends (XZ),
        // then the pool's surface, the drop and the strength.
        v[32..64].copy_from_slice(&lanes.0[pv_falls..pv_falls + 32]);
    }
    if let Some(mut ffx) = ffx {
        ffx.lanes[..64].copy_from_slice(&u);
        ffx.lanes[64..].copy_from_slice(&v);
        ffx.depth = if u[18] > 0.5 || veil_on {
            pass.map(|p| p.scene_depth.clone())
        } else {
            None
        };
        // The terrain shadow the caustic net from below is cast under.
        ffx.shadow = (u[18] > 0.5).then(|| shadow.0.image());
    }
}

// --------------------------------------------------------------------------------------- falls

/// The falls the rows hold, and how far from the eye one may be, yards.
const FALLS_MAX: usize = 8;
const FALLS_REACH: f32 = 400.0;
/// A fall is in view within the frame's half width and this margin, radians, of the view's bearing
/// (the spray reaching past the frame's edge), or this near, yards.
const FALL_VIEW_MARGIN: f32 = 0.25;
const FALL_VIEW_NEAR: f32 = 20.0;
/// A fall model's part: at least this tall, yards, and standing in water whose surface it crosses
/// at least this far above its bottom (or its bottom within the second of the surface).
const FALL_MIN_HEIGHT: f32 = 2.5;
const FALL_FOOT_SLACK: f32 = 2.0;
/// Foot points nearer than this, yards, at one pool are one fall, and a run of them splits where
/// it leaves a wider gap; no line is longer than `FALL_LINE_MAX`.
const FALL_JOIN: f32 = 4.0;
const FALL_LINE_MAX: f32 = 60.0;
/// The gap rule's reach from the eye, yards, how far out past a surface's end it looks for the
/// lower one, yards, and how often (seconds) or after how far a move (yards) it rescans; a part's
/// foot is rebuilt as often while the loaded water or terrain changes.
const FALL_GAP_REACH: f32 = 140.0;
const FALL_GAP_OUT: [f32; 3] = [2.0, 6.0, 12.0];
const FALL_GAP_RESCAN: (f32, f32) = (4.0, 20.0);
/// A face is steep when its normal's vertical part is at most this share of the normal (more than
/// about 45° from level), and a part is a curtain when its steep faces are at least
/// `FALL_STEEP_SHARE` of its area.
const FALL_STEEP_NY: f32 = 0.7;
const FALL_STEEP_SHARE: f32 = 0.5;
/// A pool above a curtain's lowest one (a ledge it feeds, or the basin it leaves) is a foot too only
/// with at least this share of the curtain above the lowest pool also above it: a lip, a dam's
/// reservoir, has next to none.
const FALL_ABOVE_SHARE: f32 = 0.2;
/// Per part: the steep faces sampled for its pools, the pools kept, the waterline's spacing, yards,
/// and the foot points kept per pool.
const FALL_POOL_SAMPLES: usize = 128;
const FALL_POOLS: usize = 2;
const FALL_STEP: f32 = 0.5;
const FALL_POINTS: usize = 128;
/// A curtain that comes down on the bank, its waterline inside ground standing at most
/// `FALL_BANK_RISE` yards over the pool, has its foot at the water in front of it, within
/// `FALL_BANK_REACH` yards along the face's normal (at most `FALL_BANK_TRIES` crossings tried per
/// pool); such a foot spreads `FALL_BANK_SPREAD` as far as one in the pool (`$WOW_PREVIEW_FALLS_BANK`).
const FALL_BANK_RISE: f32 = 4.0;
const FALL_BANK_REACH: f32 = 6.0;
const FALL_BANK_TRIES: usize = 48;
const FALL_BANK_SPREAD: f32 = 0.6;
/// How often, seconds, a part that found no foot looks again while the loaded world holds (it
/// looks again a few seconds after the world changes).
const FALL_RETRY: f32 = 16.0;
/// The parts whose feet one scan may build: a new stretch of falls (Elwynn's 44 parts) spreads
/// over a few scans instead of one long frame.
const FALL_REBUILDS: usize = 8;
/// A run of foot points longer than `FALL_BEND_MIN` yards splits in two where its thirds turn by
/// more than this, radians, so a foot curving round a rock is a few lines, not one chord.
const FALL_BEND: f32 = 0.45;
const FALL_BEND_MIN: f32 = 9.0;

/// One point of a foot: where a curtain meets its pool (Bevy XZ), the pool's surface (Bevy Y), the
/// drop over it, and how far its pool spreads against one where a curtain enters the water (1).
#[derive(Clone, Copy, Debug, PartialEq)]
struct FootPt {
    p: Vec2,
    y: f32,
    drop: f32,
    spread: f32,
}

/// What [`mesh_foot`] found besides the points: the curtain's bounds, and the counts
/// `$WOW_PREVIEW_FALLS_LOG` reports.
#[derive(Default)]
struct FootStats {
    steep: f32,
    top: f32,
    bottom: f32,
    pools: Vec<f32>,
    cut: usize,
    wet: usize,
    bank: usize,
    dry: usize,
    off: usize,
    rise: (f32, f32),
    /// The steep faces' bounds, when the part is a curtain.
    curtain: Option<(Vec3, Vec3)>,
}

/// A curtain's foot from its own triangles (Bevy world space): where its steep faces cross the pool
/// below them. A pool is the water (`surface_y(x, z, y)`, Bevy Y: the lowest surface over x, z that
/// answers at y) under the curtain's faces that its bottom reaches within [`FALL_FOOT_SLACK`], with
/// at least [`FALL_MIN_HEIGHT`] of curtain above it, the lowest such and those with
/// [`FALL_ABOVE_SHARE`] of the curtain above it; a curtain that ends over its pool is cut at its
/// bottom edge. The waterline, every [`FALL_STEP`] yards, is a foot where the pool lies there over
/// its bed (`ground_y`, the terrain; no answer counts as water); where it runs inside the bank, no
/// more than [`FALL_BANK_RISE`] under the ground, the foot is the pool in front of it, within
/// [`FALL_BANK_REACH`] along the face's normal, spreading `bank` as far. At most [`FALL_POOLS`] pools
/// and [`FALL_POINTS`] points per pool.
fn mesh_foot(
    tris: &[[Vec3; 3]],
    surface_y: &impl Fn(f32, f32, f32) -> Option<f32>,
    ground_y: &impl Fn(f32, f32) -> Option<f32>,
    bank: f32,
    stats: &mut FootStats,
) -> Vec<FootPt> {
    // The steep faces: (lowest y, highest y, triangle, area).
    let mut steep: Vec<(f32, f32, usize, f32)> = Vec::new();
    let (mut total, mut steep_area) = (0.0f32, 0.0f32);
    let (mut top, mut bottom) = (f32::NEG_INFINITY, f32::INFINITY);
    let (mut lo3, mut hi3) = (Vec3::INFINITY, Vec3::NEG_INFINITY);
    for (i, t) in tris.iter().enumerate() {
        let n = (t[1] - t[0]).cross(t[2] - t[0]);
        let a = n.length();
        if !a.is_finite() || a < 1e-4 {
            continue;
        }
        total += a;
        if n.y.abs() <= FALL_STEEP_NY * a {
            let lo = t[0].y.min(t[1].y).min(t[2].y);
            let hi = t[0].y.max(t[1].y).max(t[2].y);
            steep_area += a;
            top = top.max(hi);
            bottom = bottom.min(lo);
            lo3 = lo3.min(t[0].min(t[1]).min(t[2]));
            hi3 = hi3.max(t[0].max(t[1]).max(t[2]));
            steep.push((lo, hi, i, a));
        }
    }
    stats.steep = steep_area / total.max(1e-6);
    (stats.top, stats.bottom) = (top, bottom);
    if steep.is_empty() || steep_area < FALL_STEEP_SHARE * total || top - bottom < FALL_MIN_HEIGHT {
        return Vec::new();
    }
    stats.curtain = Some((lo3, hi3));
    // The steep area above `s`, each face's share by its height.
    let above = |s: f32| -> f32 {
        steep
            .iter()
            .map(|&(lo, hi, _, a)| a * ((hi - s) / (hi - lo).max(1e-3)).clamp(0.0, 1.0))
            .sum()
    };
    // The pools under the steep faces.
    let stride = steep.len().div_ceil(FALL_POOL_SAMPLES).max(1);
    let mut pools: Vec<f32> = Vec::new();
    for &(_, _, i, _) in steep.iter().step_by(stride) {
        let t = &tris[i];
        let c = (t[0] + t[1] + t[2]) / 3.0;
        let Some(s) = surface_y(c.x, c.z, c.y) else {
            continue;
        };
        if bottom <= s + FALL_FOOT_SLACK
            && top - s >= FALL_MIN_HEIGHT
            && !pools.iter().any(|&p| (p - s).abs() < 0.5)
        {
            pools.push(s);
        }
    }
    pools.sort_by(f32::total_cmp);
    let base = pools.first().map_or(0.0, |&s| above(s));
    pools.retain(|&s| above(s) >= FALL_ABOVE_SHARE * base);
    pools.truncate(FALL_POOLS);
    stats.pools.clone_from(&pools);
    stats.rise = (f32::INFINITY, f32::NEG_INFINITY);
    let mut out = Vec::new();
    for h in pools {
        let cut = if bottom < h { h } else { bottom + 0.3 };
        // The waterline: each steep face's segment across the cut, every half yard, once per
        // half-yard cell, with the face's horizontal normal.
        let mut seen: bevy::platform::collections::HashSet<(i32, i32)> = Default::default();
        let mut line: Vec<(Vec2, Vec2)> = Vec::new();
        for &(lo, hi, i, _) in &steep {
            if lo > cut || hi < cut {
                continue;
            }
            let t = &tris[i];
            let mut ends: [Vec2; 2] = [Vec2::ZERO; 2];
            let mut k = 0;
            for (p, q) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                let (dp, dq) = (p.y - cut, q.y - cut);
                if (dp < 0.0) == (dq < 0.0) || k == 2 {
                    continue;
                }
                let x = p + (q - p) * (dp / (dp - dq));
                ends[k] = Vec2::new(x.x, x.z);
                k += 1;
            }
            if k == 0 {
                continue;
            }
            let n3 = (t[1] - t[0]).cross(t[2] - t[0]);
            let n = Vec2::new(n3.x, n3.z).normalize_or_zero();
            let (a, b) = (ends[0], ends[k - 1]);
            let steps = ((a.distance(b) / FALL_STEP).ceil() as usize).clamp(1, 256);
            for j in 0..=steps {
                let x = a.lerp(b, j as f32 / steps as f32);
                if seen.insert((
                    (x.x / FALL_STEP).floor() as i32,
                    (x.y / FALL_STEP).floor() as i32,
                )) {
                    line.push((x, n));
                }
            }
        }
        stats.cut += line.len();
        let wet = |q: Vec2| {
            ground_y(q.x, q.y).is_none_or(|g| g < h - 0.25)
                && surface_y(q.x, q.y, h + 1.0).is_some_and(|s| (s - h).abs() < 0.75)
        };
        let drop = (top - h).min(150.0);
        let stride = line.len().div_ceil(FALL_POINTS).max(1);
        let mut banked = Vec::new();
        for &(p, n) in line.iter().step_by(stride) {
            match ground_y(p.x, p.y) {
                Some(g) if g >= h - 0.25 => {
                    stats.rise = (stats.rise.0.min(g - h), stats.rise.1.max(g - h));
                    if g - h <= FALL_BANK_RISE {
                        banked.push((p, n));
                    } else {
                        stats.dry += 1;
                    }
                }
                _ if wet(p) => {
                    stats.wet += 1;
                    out.push(FootPt {
                        p,
                        y: h,
                        drop,
                        spread: 1.0,
                    });
                }
                _ => stats.off += 1,
            }
        }
        // On the bank: the nearest water along the face's normal, either way.
        let stride = banked.len().div_ceil(FALL_BANK_TRIES).max(1);
        for &(p, n) in banked.iter().step_by(stride) {
            let found = (1..=(FALL_BANK_REACH / FALL_STEP) as usize)
                .flat_map(|k| [1.0, -1.0].map(|sgn| p + n * (sgn * k as f32 * FALL_STEP)))
                .find(|&q| wet(q));
            match found {
                Some(q) => {
                    stats.bank += 1;
                    out.push(FootPt {
                        p: q,
                        y: h,
                        drop,
                        spread: bank,
                    });
                }
                None => stats.dry += 1,
            }
        }
    }
    out
}

/// The root of `i`'s set in a union-find forest, halving the path on the way.
fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Per curtain (its steep faces' bounds, Bevy), the top of the cascade it belongs to: curtains whose
/// footprints come within [`FALL_JOIN`] of each other and whose heights meet within
/// [`FALL_FOOT_SLACK`] are one cascade (Elwynn's tiers, one stream stepping down ledges), whose
/// spray at the foot is that of the whole fall.
fn cascade_tops(curtains: &[(Vec3, Vec3)]) -> Vec<f32> {
    let n = curtains.len();
    let mut group: Vec<usize> = (0..n).collect();
    let meet = |(alo, ahi): (Vec3, Vec3), (blo, bhi): (Vec3, Vec3)| {
        alo.x <= bhi.x + FALL_JOIN
            && blo.x <= ahi.x + FALL_JOIN
            && alo.z <= bhi.z + FALL_JOIN
            && blo.z <= ahi.z + FALL_JOIN
            && alo.y <= bhi.y + FALL_FOOT_SLACK
            && blo.y <= ahi.y + FALL_FOOT_SLACK
    };
    for i in 0..n {
        for j in i + 1..n {
            if meet(curtains[i], curtains[j]) {
                let (a, b) = (root(&mut group, i), root(&mut group, j));
                if a != b {
                    group[a.max(b)] = a.min(b);
                }
            }
        }
    }
    let mut top = vec![f32::NEG_INFINITY; n];
    for (i, c) in curtains.iter().enumerate() {
        let r = root(&mut group, i);
        top[r] = top[r].max(c.1.y);
    }
    (0..n).map(|i| top[root(&mut group, i)]).collect()
}

/// The foot lines through `pts`: points within [`FALL_JOIN`] of one another at one pool (within a
/// yard) are one fall, whatever part or rule found them, fitted by [`fit_run`]. Each pool's points
/// are first thinned to one per half-yard cell, in a fixed order, so the lines do not depend on
/// the order the parts were found in.
fn foot_lines(mut pts: Vec<FootPt>) -> Vec<Fall> {
    pts.sort_by(|a, b| {
        a.y.total_cmp(&b.y)
            .then(a.p.x.total_cmp(&b.p.x))
            .then(a.p.y.total_cmp(&b.p.y))
    });
    let key = |f: &FootPt| {
        (
            (f.p.x * 2.0).floor() as i32,
            (f.p.y * 2.0).floor() as i32,
            f.y.round() as i32,
        )
    };
    let mut seen: bevy::platform::collections::HashSet<(i32, i32, i32)> = Default::default();
    pts.retain(|f| seen.insert(key(f)));
    let n = pts.len();
    let cell = |p: Vec2| {
        (
            (p.x / FALL_JOIN).floor() as i32,
            (p.y / FALL_JOIN).floor() as i32,
        )
    };
    let mut grid: bevy::platform::collections::HashMap<(i32, i32), Vec<usize>> = Default::default();
    for (i, f) in pts.iter().enumerate() {
        grid.entry(cell(f.p)).or_default().push(i);
    }
    let mut parent: Vec<usize> = (0..n).collect();
    for i in 0..n {
        let (cx, cz) = cell(pts[i].p);
        for dx in -1..=1 {
            for dz in -1..=1 {
                let Some(near) = grid.get(&(cx + dx, cz + dz)) else {
                    continue;
                };
                for &j in near {
                    if j > i
                        && pts[i].p.distance(pts[j].p) < FALL_JOIN
                        && (pts[i].y - pts[j].y).abs() < 1.0
                    {
                        let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                        if a != b {
                            parent[a.max(b)] = a.min(b);
                        }
                    }
                }
            }
        }
    }
    let mut group_of: Vec<Option<usize>> = vec![None; n];
    let mut groups: Vec<Vec<FootPt>> = Vec::new();
    for (i, &f) in pts.iter().enumerate() {
        let r = root(&mut parent, i);
        let g = *group_of[r].get_or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(f);
    }
    let mut out = Vec::new();
    for mut g in groups {
        fit_run(&mut g, &mut out, 0);
    }
    out
}

/// Lines along one fall's foot points, appended to `out`: along the points' principal axis, at
/// their mean offset across it and padded half a yard, split where they leave a gap wider than
/// [`FALL_JOIN`], into even pieces no longer than [`FALL_LINE_MAX`], and in two (each refitted)
/// where they bend by more than [`FALL_BEND`]. A run under 1.5 yd long, padding included, is no
/// line. Each line takes its run's mean pool and spread and its greatest drop.
fn fit_run(pts: &mut [FootPt], out: &mut Vec<Fall>, depth: u32) {
    if pts.len() < 2 {
        return;
    }
    let n = pts.len() as f32;
    let m = pts.iter().map(|f| f.p).sum::<Vec2>() / n;
    let (mut sxx, mut sxz, mut szz) = (0.0f32, 0.0f32, 0.0f32);
    for f in pts.iter() {
        let d = f.p - m;
        sxx += d.x * d.x;
        sxz += d.x * d.y;
        szz += d.y * d.y;
    }
    let ang = 0.5 * (2.0 * sxz).atan2(sxx - szz);
    let u = Vec2::new(ang.cos(), ang.sin());
    let nrm = u.perp();
    let t = |f: &FootPt| (f.p - m).dot(u);
    pts.sort_by(|a, b| t(a).total_cmp(&t(b)));
    let mut start = 0;
    for i in 1..=pts.len() {
        if i < pts.len() && t(&pts[i]) - t(&pts[i - 1]) <= FALL_JOIN {
            continue;
        }
        let run = &mut pts[start..i];
        start = i;
        let (t0, t1) = (t(&run[0]), t(&run[run.len() - 1]));
        let len = t1 - t0;
        if len + 1.0 < 1.5 {
            continue;
        }
        if depth < 6 && run.len() >= 4 && len > FALL_LINE_MAX {
            let k = (len / FALL_LINE_MAX).ceil();
            let mut s = 0;
            for j in 1..=k as usize {
                let end = t0 + len * j as f32 / k;
                let e = if j == k as usize {
                    run.len()
                } else {
                    s + run[s..].partition_point(|f| t(f) <= end)
                };
                fit_run(&mut run[s..e], out, depth + 1);
                s = e;
            }
            continue;
        }
        if depth < 6 && run.len() >= 6 && len > FALL_BEND_MIN {
            let third = |k: usize| {
                let (s, e) = (run.len() * k / 3, run.len() * (k + 1) / 3);
                run[s..e].iter().map(|f| f.p).sum::<Vec2>() / (e - s) as f32
            };
            let (p0, p1, p2) = (third(0), third(1), third(2));
            let turn = (p1 - p0)
                .normalize_or_zero()
                .dot((p2 - p1).normalize_or_zero());
            if turn < FALL_BEND.cos() {
                let (l, r) = run.split_at_mut(run.len() / 2);
                fit_run(l, out, depth + 1);
                fit_run(r, out, depth + 1);
                continue;
            }
        }
        let k = run.len() as f32;
        let c = run.iter().map(|f| (f.p - m).dot(nrm)).sum::<f32>() / k;
        out.push(Fall {
            a: m + u * (t0 - 0.5) + nrm * c,
            b: m + u * (t1 + 0.5) + nrm * c,
            foot_y: run.iter().map(|f| f.y).sum::<f32>() / k,
            drop: run.iter().fold(0.0f32, |d, f| d.max(f.drop)),
            spread: run.iter().map(|f| f.spread).sum::<f32>() / k,
        });
    }
}

/// One fall, in Bevy world space: its foot line's ends (XZ), the pool's surface, the drop, and how
/// far its pool and spray spread against a fall's into open water (1).
#[derive(Clone, Copy)]
struct Fall {
    a: Vec2,
    b: Vec2,
    foot_y: f32,
    drop: f32,
    spread: f32,
}

/// A fall part's foot points as [`mesh_foot`] last found them, with the pose and the loaded world
/// (water surfaces, terrain tiles) they were found for and when.
struct PartFoot {
    pose: bevy::math::Affine3A,
    world: (usize, usize),
    built: f32,
    curtain: Option<(Vec3, Vec3)>,
    pts: Vec<FootPt>,
}

/// A model part the falls scan reads: its pose, bound, material (for its texture's scroll) and
/// resident triangles.
type FallPart = (
    Entity,
    &'static GlobalTransform,
    &'static Aabb,
    &'static MeshMaterial3d<WowModelMaterial>,
    Option<&'static benilla_world::interact::PickMesh>,
);

/// The falls' foot lines (`preview.falls`): among the loaded model parts that draw a scrolling
/// texture (a texture transform moving mostly along V) and stand at least [`FALL_MIN_HEIGHT`]
/// tall, where each curtain's own steep triangles ([`mesh_foot`]) cross the water below them,
/// found once per part and kept while its pose and the loaded world hold (built again a few
/// seconds after the world changes, or every [`FALL_RETRY`] while it finds none), each foot's
/// drop its cascade's
/// ([`cascade_tops`]), and the gap rule's edges; every point fitted into lines by [`foot_lines`].
/// Rescanned twice a second; of those within [`FALLS_REACH`],
/// the [`FALLS_MAX`] nearest are kept, those in view first.
#[allow(clippy::too_many_arguments)]
fn drive_fall_lanes(
    time: Res<Time>,
    style: Res<WaterStyle>,
    cam: Query<(&GlobalTransform, Option<&Projection>), With<WorldCamera>>,
    parts: Query<FallPart>,
    uv: Option<Res<benilla_world::doodad_anim::UvAnimMaterials>>,
    water: Query<&WaterChunkInfo>,
    index: Res<WaterIndex>,
    ground: (
        Option<Res<benilla_world::terrain_stream::TerrainStreamer>>,
        Option<Res<Assets<benilla_assets::AdtTile>>>,
    ),
    mut last: Local<f32>,
    mut found: Local<Vec<Fall>>,
    mut feet: Local<EntityHashMap<PartFoot>>,
    mut gaps: Local<(Vec3, f32, Vec<FootPt>)>,
    mut lanes: ResMut<PreviewLanes>,
) {
    let base = at(PV_FALLS);
    if !style.is_stylised() || !lever::on(lever::FALLS) {
        lanes.0[base..base + 64].fill(0.0);
        return;
    }
    let Ok((eye, fwd, half)) = cam.single().map(|(t, p)| {
        // The frame's horizontal half angle.
        let half = match p {
            Some(Projection::Perspective(p)) => ((p.fov * 0.5).tan() * p.aspect_ratio).atan(),
            _ => 0.65,
        };
        (t.translation(), t.forward(), half)
    }) else {
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
    // The lowest water surface (Bevy Y) over a Bevy XZ point that answers at Bevy height `y`.
    let surface_y = |x: f32, z: f32, y: f32| {
        let w = bevy_to_wow(Vec3::new(x, y, z));
        benilla_world::liquid::water_surface_at(
            index
                .over(w[0], w[1])
                .iter()
                .filter_map(|&e| water.get(e).ok()),
            [w[0], w[1], w[2]],
            LiquidClaim::Unknown,
        )
        .map(|s| wow_to_bevy([w[0], w[1], s]).y)
    };
    if (now - *last).abs() >= 0.5 {
        *last = now;
        // `$WOW_PREVIEW_FALLS_LOG=1` (round 3): every scrolling part near the eye, its box and its
        // scroll, logged whenever their number changes, and each fall part's foot as it is built.
        static FALLS_LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let log_parts = *FALLS_LOG.get_or_init(|| lever::number("FALLS_LOG", 0.0, 0.0, 1.0) > 0.5);
        static FALLS_BANK: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
        let bank =
            *FALLS_BANK.get_or_init(|| lever::number("FALLS_BANK", FALL_BANK_SPREAD, 0.0, 1.0));
        let mut logged: Vec<String> = Vec::new();
        let world = (
            water.iter().count(),
            ground.1.as_deref().map_or(0, |a| a.len()),
        );
        let mut pts: Vec<FootPt> = Vec::new();
        let mut seen: bevy::ecs::entity::EntityHashSet = Default::default();
        let mut curtains: Vec<Entity> = Vec::new();
        let mut rebuilt = 0;
        let clock = std::time::Instant::now();
        if let Some(uv) = uv {
            for (entity, t, aabb, mat, pick) in &parts {
                let Some(entry) = uv.0.get(&mat.0.id()) else {
                    continue;
                };
                let Some([du, dv]) = entry.scroll() else {
                    continue;
                };
                let (c, h) = (Vec3::from(aabb.center), Vec3::from(aabb.half_extents));
                let near_log = log_parts && {
                    let p = t.transform_point(c);
                    Vec2::new(p.x - eye.x, p.z - eye.z).length() < 150.0
                };
                if near_log {
                    let p = t.transform_point(c);
                    let ext = (t.affine().matrix3.abs() * h) * 2.0;
                    logged.push(format!(
                        "({:.1},{:.1},{:.1}) size {:.1}x{:.1}x{:.1} scroll {du:.3},{dv:.3}",
                        p.x, p.y, p.z, ext.x, ext.y, ext.z
                    ));
                }
                if dv.abs() < 1e-3 || dv.abs() < 1.5 * du.abs() {
                    continue;
                }
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
                if hi.y - lo.y < FALL_MIN_HEIGHT
                    || Vec2::new(mid.x - eye.x, mid.z - eye.z).length() > FALLS_REACH
                {
                    continue;
                }
                // A card turns to the camera: not a curtain.
                if pick.is_some_and(|m| m.0.billboard.is_some()) {
                    continue;
                }
                seen.insert(entity);
                let pose = t.affine();
                let stale = feet.get(&entity).is_none_or(|f| {
                    f.pose != pose
                        || (now - f.built >= FALL_GAP_RESCAN.0 && f.world != world)
                        || (now - f.built >= FALL_RETRY && f.pts.is_empty())
                });
                // At most `FALL_REBUILDS` parts a scan; the rest keep what they had until the next.
                if stale && rebuilt < FALL_REBUILDS {
                    rebuilt += 1;
                    // The curtain's triangles in the world, as the picker reads them; a part
                    // without resident geometry stands in as a sheet along its box's longer side.
                    let tris: Vec<[Vec3; 3]> = match pick {
                        Some(m) => {
                            let geo = &m.0;
                            let wp: Vec<Vec3> = geo
                                .positions
                                .iter()
                                .map(|p| t.transform_point(wow_to_bevy(*p)))
                                .collect();
                            geo.indices
                                .as_chunks::<3>()
                                .0
                                .iter()
                                .filter_map(|i| {
                                    Some([
                                        *wp.get(i[0] as usize)?,
                                        *wp.get(i[1] as usize)?,
                                        *wp.get(i[2] as usize)?,
                                    ])
                                })
                                .collect()
                        }
                        None => {
                            let (a, b) = if hi.x - lo.x >= hi.z - lo.z {
                                (Vec3::new(lo.x, 0.0, mid.z), Vec3::new(hi.x, 0.0, mid.z))
                            } else {
                                (Vec3::new(mid.x, 0.0, lo.z), Vec3::new(mid.x, 0.0, hi.z))
                            };
                            let (y0, y1) = (Vec3::Y * lo.y, Vec3::Y * hi.y);
                            vec![[a + y0, b + y0, b + y1], [a + y0, b + y1, a + y1]]
                        }
                    };
                    let mut stats = FootStats::default();
                    let foot = mesh_foot(&tris, &surface_y, &ground_y, bank, &mut stats);
                    let changed = feet.get(&entity).is_none_or(|f| f.pts.len() != foot.len());
                    if near_log && changed {
                        info!(
                            "preview falls: part {entity} at ({:.1},{:.1},{:.1}): {} tris, steep \
                             {:.2}, y {:.1} to {:.1}, pools {:?}, waterline {}: in the pool {}, \
                             on the bank {} (ground {:.1} to {:.1} over the pool), dry {}, no \
                             pool {}; kept {}",
                            mid.x,
                            mid.y,
                            mid.z,
                            tris.len(),
                            stats.steep,
                            stats.bottom,
                            stats.top,
                            stats.pools,
                            stats.cut,
                            stats.wet,
                            stats.bank,
                            stats.rise.0,
                            stats.rise.1,
                            stats.dry,
                            stats.off,
                            foot.len()
                        );
                    }
                    feet.insert(
                        entity,
                        PartFoot {
                            pose,
                            world,
                            built: now,
                            curtain: stats.curtain,
                            pts: foot,
                        },
                    );
                }
                curtains.push(entity);
            }
        }
        feet.retain(|e, _| seen.contains(e));
        if log_parts && rebuilt > 0 {
            info!(
                "preview falls: {rebuilt} parts' feet built in {:.2} ms",
                clock.elapsed().as_secs_f64() * 1e3
            );
        }
        // Each foot's drop is its cascade's.
        let found_feet: Vec<&PartFoot> = curtains
            .iter()
            .filter_map(|e| feet.get(e))
            .filter(|f| f.curtain.is_some())
            .collect();
        let bounds: Vec<(Vec3, Vec3)> = found_feet.iter().filter_map(|f| f.curtain).collect();
        for (f, top) in found_feet.iter().zip(cascade_tops(&bounds)) {
            pts.extend(f.pts.iter().map(|p| FootPt {
                drop: p.drop.max((top - p.y).min(150.0)),
                ..*p
            }));
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
        // foot runs along the edge on the lower surface, a point a yard, where the lower pool
        // lies over its bed. Rescanned every few seconds or after a move, over the wet cells
        // within `FALL_GAP_REACH` of the eye.
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
            let mut edges: Vec<FootPt> = Vec::new();
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
                            let steps = (ba.distance(bb).ceil() as usize).clamp(1, 16);
                            for i in 0..=steps {
                                let q = ba.lerp(bb, i as f32 / steps as f32);
                                if ground_y(q.x, q.z).is_none_or(|g| g < q.y - 0.25) {
                                    edges.push(FootPt {
                                        p: Vec2::new(q.x, q.z),
                                        y: q.y,
                                        drop,
                                        spread: 1.0,
                                    });
                                }
                            }
                            break;
                        }
                    }
                });
            }
            gaps.2 = edges;
        }
        pts.extend_from_slice(&gaps.2);
        *found = foot_lines(pts);
        let all = found.len();
        // The falls in view first (an end or the middle within the frame's half width and
        // `FALL_VIEW_MARGIN` of the view's bearing, or within `FALL_VIEW_NEAR`), then the rest,
        // each nearest first: the veil draws the spray of the first four only.
        let e2 = Vec2::new(eye.x, eye.z);
        let f2 = Vec2::new(fwd.x, fwd.z).normalize_or_zero();
        let cone = (half + FALL_VIEW_MARGIN).min(1.4).cos();
        let seen_at = |p: Vec2| {
            let d = p - e2;
            d.length() < FALL_VIEW_NEAR || d.normalize_or_zero().dot(f2) > cone
        };
        let key = |f: &Fall| {
            let shown = seen_at(f.a) || seen_at(f.b) || seen_at((f.a + f.b) * 0.5);
            (!shown, (f.a + f.b).distance(e2 * 2.0))
        };
        found.sort_by(|p, q| {
            let (kp, kq) = (key(p), key(q));
            kp.0.cmp(&kq.0).then(kp.1.total_cmp(&kq.1))
        });
        found.truncate(FALLS_MAX);
        // Logged when the lines change, in the rows' order.
        let text = found.iter().fold(String::new(), |acc, f| {
            format!(
                "{acc}; {:.1},{:.1} to {:.1},{:.1} at y {:.1}, drop {:.1}, spread {:.2}",
                f.a.x, f.a.y, f.b.x, f.b.y, f.foot_y, f.drop, f.spread
            )
        });
        static SEEN: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());
        if let Ok(mut seen) = SEEN.lock() {
            if *seen != text {
                info!("preview falls: {} foot lines of {all}{text}", found.len());
                *seen = text;
            }
        }
    }
    let rows = &mut lanes.0[base..base + 64];
    rows.fill(0.0);
    for (i, f) in found.iter().enumerate() {
        rows[8 * i..8 * i + 4].copy_from_slice(&[f.a.x, f.a.y, f.b.x, f.b.y]);
        rows[8 * i + 4..8 * i + 8].copy_from_slice(&[f.foot_y, f.drop, f.spread, 1.0]);
    }
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<PhantomBody>();
    app.add_systems(
        PostUpdate,
        (
            drive_wake_lanes,
            drive_fall_lanes,
            // The veil copies the falls' rows this frame's scan wrote.
            // The caustic net's clock and drift are the preview rows' of this frame.
            drive_under_lanes,
        )
            // One writer of the preview's lanes at a time, after the mask and clocks; the caustic
            // net reads this frame's water shadow and lands after the wave field's rows.
            .chain()
            .after(super::preview::drive_preview)
            .after(super::mcsh::drive_water_shadow)
            .after(super::ripple_sim::drive_ripple_sim)
            .before(super::reflect::drive_reflection),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sheet from `a` to `b` (its bottom edge, Bevy) rising `rise` (Bevy Y) and leaning `lean`
    /// (XZ) to its top edge, in one-yard cells of two triangles.
    fn sheet(a: Vec3, b: Vec3, rise: f32, lean: Vec2) -> Vec<[Vec3; 3]> {
        let n = (a.distance(b).ceil() as usize).max(1);
        let m = (rise.ceil() as usize).max(1);
        let up = Vec3::new(lean.x, rise, lean.y);
        let at = |i: usize, j: usize| a.lerp(b, i as f32 / n as f32) + up * (j as f32 / m as f32);
        (0..n)
            .flat_map(|i| (0..m).map(move |j| (i, j)))
            .flat_map(|(i, j)| {
                [
                    [at(i, j), at(i + 1, j), at(i + 1, j + 1)],
                    [at(i, j), at(i + 1, j + 1), at(i, j + 1)],
                ]
            })
            .collect()
    }

    fn pool(y: f32) -> impl Fn(f32, f32, f32) -> Option<f32> {
        move |_, _, _| Some(y)
    }

    fn open(_: f32, _: f32) -> Option<f32> {
        Some(-10.0)
    }

    fn lines(tris: &[[Vec3; 3]], surface: &impl Fn(f32, f32, f32) -> Option<f32>) -> Vec<Fall> {
        foot_lines(mesh_foot(
            tris,
            surface,
            &open,
            0.6,
            &mut FootStats::default(),
        ))
    }

    /// A vertical curtain into a flat pool: one line along it at the pool.
    #[test]
    fn a_curtain_meets_its_pool_along_its_waterline() {
        let tris = sheet(
            Vec3::new(-10.0, -3.0, 5.0),
            Vec3::new(10.0, -3.0, 5.0),
            23.0,
            Vec2::ZERO,
        );
        let got = lines(&tris, &pool(0.0));
        assert_eq!(got.len(), 1);
        let f = got[0];
        assert!((f.a.y - 5.0).abs() < 0.1 && (f.b.y - 5.0).abs() < 0.1);
        assert!((f.a.x.min(f.b.x) + 10.5).abs() < 0.6 && (f.a.x.max(f.b.x) - 10.5).abs() < 0.6);
        assert_eq!(f.foot_y, 0.0);
        assert!((f.drop - 20.0).abs() < 0.01);
    }

    /// A 60° chute: its line runs across it where it enters the water, not along its flow.
    #[test]
    fn a_chute_meets_its_pool_across_its_flow() {
        let run = 24.0 / 60f32.to_radians().tan();
        // The bottom edge at y -4, the top edge 24 yd up and `run` back toward -z.
        let tris = sheet(
            Vec3::new(-5.0, -4.0, run),
            Vec3::new(5.0, -4.0, run),
            24.0,
            Vec2::new(0.0, -run),
        );
        let got = lines(&tris, &pool(0.0));
        assert_eq!(got.len(), 1);
        let f = got[0];
        let z = run * 20.0 / 24.0;
        assert!(
            (f.a.y - z).abs() < 0.3 && (f.b.y - z).abs() < 0.3,
            "{:?} {:?}",
            f.a,
            f.b
        );
        assert!((f.a.x - f.b.x).abs() > 9.0);
    }

    /// A waterline inside ground well over the pool (a ledge) is no foot.
    #[test]
    fn a_waterline_in_high_ground_is_dropped() {
        let tris = sheet(
            Vec3::new(-10.0, -3.0, 5.0),
            Vec3::new(10.0, -3.0, 5.0),
            23.0,
            Vec2::ZERO,
        );
        let ledge = |_: f32, _: f32| Some(6.0);
        let mut stats = FootStats::default();
        assert!(mesh_foot(&tris, &pool(0.0), &ledge, 0.6, &mut stats).is_empty());
        assert!(stats.dry > 0);
    }

    /// A curtain coming down on a low bank has its foot at the water's edge in front of it,
    /// spreading less.
    #[test]
    fn a_curtain_on_the_bank_has_its_foot_at_the_water() {
        let tris = sheet(
            Vec3::new(-10.0, -3.0, 5.0),
            Vec3::new(10.0, -3.0, 5.0),
            23.0,
            Vec2::ZERO,
        );
        // The bank stands a yard over the pool up to z = 7, the pool's bed beyond.
        let bank = |_: f32, z: f32| Some(if z < 7.0 { 1.0 } else { -4.0 });
        let got = foot_lines(mesh_foot(
            &tris,
            &pool(0.0),
            &bank,
            0.6,
            &mut FootStats::default(),
        ));
        assert_eq!(got.len(), 1);
        let f = got[0];
        assert!(
            (f.a.y - 7.0).abs() < 0.6 && (f.b.y - 7.0).abs() < 0.6,
            "{:?} {:?}",
            f.a,
            f.b
        );
        assert!((f.spread - 0.6).abs() < 1e-4);
    }

    /// A curtain ending a yard over its pool is cut at its bottom edge; five yards over, it lands
    /// in no pool.
    #[test]
    fn a_curtain_ending_over_its_pool() {
        let near = sheet(
            Vec3::new(-10.0, 1.0, 5.0),
            Vec3::new(10.0, 1.0, 5.0),
            19.0,
            Vec2::ZERO,
        );
        let got = lines(&near, &pool(0.0));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].foot_y, 0.0);
        let high = sheet(
            Vec3::new(-10.0, 5.0, 5.0),
            Vec3::new(10.0, 5.0, 5.0),
            15.0,
            Vec2::ZERO,
        );
        assert!(lines(&high, &pool(0.0)).is_empty());
    }

    /// A spillway leaving its reservoir: the reservoir under its top is a lip, not a foot.
    #[test]
    fn a_reservoir_under_the_top_is_no_foot() {
        let run = 19.0;
        let tris = sheet(
            Vec3::new(-5.0, -3.0, run),
            Vec3::new(5.0, -3.0, run),
            33.0,
            Vec2::new(0.0, -run),
        );
        let surface = |_: f32, z: f32, _: f32| Some(if z < 2.0 { 25.0 } else { 0.0 });
        let got = lines(&tris, &surface);
        assert!(!got.is_empty());
        assert!(got.iter().all(|f| f.foot_y == 0.0));
    }

    /// The two layers of a part, a third of a yard apart, are one fall.
    #[test]
    fn layers_join_into_one_line() {
        let mut tris = sheet(
            Vec3::new(-10.0, -3.0, 5.0),
            Vec3::new(10.0, -3.0, 5.0),
            23.0,
            Vec2::ZERO,
        );
        tris.extend(sheet(
            Vec3::new(-10.0, -3.0, 5.3),
            Vec3::new(10.0, -3.0, 5.3),
            23.0,
            Vec2::ZERO,
        ));
        assert_eq!(lines(&tris, &pool(0.0)).len(), 1);
    }

    /// A 130-yd waterline is cut into even lines no longer than the cap, covering it.
    #[test]
    fn a_long_waterline_splits() {
        let tris = sheet(
            Vec3::new(0.0, -3.0, 0.0),
            Vec3::new(130.0, -3.0, 0.0),
            23.0,
            Vec2::ZERO,
        );
        let got = lines(&tris, &pool(0.0));
        assert!(got.len() >= 3, "{}", got.len());
        assert!(got.iter().all(|f| f.a.distance(f.b) <= FALL_LINE_MAX + 1.0));
        let cover: f32 = got.iter().map(|f| f.a.distance(f.b)).sum();
        assert!(cover > 125.0, "{cover}");
    }

    /// A foot turning a corner is two lines along its arms, not one chord across it.
    #[test]
    fn a_bent_foot_is_two_lines() {
        let pts: Vec<FootPt> = (0..=40)
            .map(|i| {
                let s = i as f32 * 0.5;
                FootPt {
                    p: Vec2::new(s, 0.0),
                    y: 0.0,
                    drop: 10.0,
                    spread: 1.0,
                }
            })
            .chain((1..=40).map(|i| FootPt {
                p: Vec2::new(20.0, i as f32 * 0.5),
                y: 0.0,
                drop: 10.0,
                spread: 1.0,
            }))
            .collect();
        let got = foot_lines(pts.clone());
        assert!(got.len() >= 2, "{}", got.len());
        for f in &got {
            let mid = (f.a + f.b) * 0.5;
            let near = pts
                .iter()
                .map(|q| q.p.distance(mid))
                .fold(f32::MAX, f32::min);
            assert!(
                near < 2.0,
                "a line's middle {mid} lies {near} yd off the foot"
            );
        }
    }

    /// Stacked tiers are one cascade with the top tier's height; a curtain off to the side is not.
    #[test]
    fn stacked_tiers_are_one_cascade() {
        let low = (Vec3::new(0.0, -5.0, 0.0), Vec3::new(10.0, 20.0, 2.0));
        let high = (Vec3::new(1.0, 18.0, -3.0), Vec3::new(9.0, 60.0, 1.0));
        let apart = (Vec3::new(40.0, -5.0, 0.0), Vec3::new(50.0, 30.0, 2.0));
        assert_eq!(cascade_tops(&[low, high, apart]), vec![60.0, 60.0, 30.0]);
    }

    /// Points at two pools, or further apart than the join, are separate falls; the order they
    /// come in does not change the lines.
    #[test]
    fn falls_part_by_pool_and_gap_and_ignore_order() {
        let row = |x0: f32, y: f32| {
            (0..20).map(move |i| FootPt {
                p: Vec2::new(x0 + i as f32 * 0.5, 0.0),
                y,
                drop: 5.0,
                spread: 1.0,
            })
        };
        let pts: Vec<FootPt> = row(0.0, 0.0)
            .chain(row(20.0, 0.0))
            .chain(row(0.0, 4.0))
            .collect();
        let a = foot_lines(pts.clone());
        assert_eq!(a.len(), 3);
        let mut rev = pts;
        rev.reverse();
        let b = foot_lines(rev);
        let key = |v: &[Fall]| {
            let mut k: Vec<[i32; 5]> = v
                .iter()
                .map(|f| {
                    let (l, h) = (f.a.min(f.b), f.a.max(f.b));
                    [l.x, l.y, h.x, h.y, f.foot_y].map(|x| (x * 100.0).round() as i32)
                })
                .collect();
            k.sort();
            k
        };
        assert_eq!(key(&a), key(&b));
    }
}
