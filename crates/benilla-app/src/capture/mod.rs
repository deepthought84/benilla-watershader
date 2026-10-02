//! Deterministic capture mode: with `$WOW_CAPTURE=<scenario>` the app boots server-less, pins the
//! game clock and the camera to a named viewpoint, waits for the rendered image to stop changing,
//! writes one PNG of the primary window to `$WOW_CAPTURE_OUT` and exits. `scripts/visual.sh`
//! drives it and `benilla-visual` diffs the shots against baselines.
//!
//! Captures are rendered game art: they live under the gitignored `target/visual/`.
//!
//! Two mechanisms make a capture reproducible:
//! 1. The shutter waits for the image itself to stop changing ([`STABLE_FRAMES`], [`FrameWatch`]),
//!    compared byte for byte off the framebuffer, so streaming, pipeline warm-up and late loads
//!    are all waited out.
//! 2. The game clock is frozen ([`CAPTURE_FRAME_DT`], [`hold_clock`]) while the scene builds, then
//!    released for exactly [`age_frames`] fixed steps, so the shot's sim age is the same on any
//!    machine.
//!
//! Neither fixes the residual flake: a few scenarios land in one of two stable states, because
//! draw order among coplanar equal-depth batches follows spawn order, which follows thread-pool
//! load completion. It is a renderer defect the harness reports; waiting longer cannot gate it
//! away. `creature-indoor-front` is one: two runs of one build differ at MAE 4.518, confined to
//! the subject's brightness, so a 4.518 there is the flake, not the change under test. Each golden
//! water scene's own two-state patches are in the first comment of its region file,
//! `scripts/water_regions/<scene>.toml`.
//!
//! ## Bursts
//! `$WOW_CAPTURE_BURST=<n>` writes `n` shots after the age instead of one, `<stem>-000.png`,
//! `<stem>-001.png`, … beside `$WOW_CAPTURE_OUT`, spaced `$WOW_CAPTURE_BURST_EVERY` fixed steps
//! apart ([`BURST_EVERY`]). The clock is held while each shot is saved and released for exactly
//! that many steps between two, so shot `i` stands `i·every` steps after shot 0 on any machine:
//! at [`age_frames`] `+ i·every` steps on a scene, and `i·every` steps past the fixture's own age on
//! a fixture viewer (`fxview`, `waterfx`). The log gives each shot's clock (a fixture's own age),
//! and the capture fails unless every shot's file is written: a run first removes the files an
//! earlier capture to the same path left ([`clear_old_shots`]). A single shot has no time in it; a
//! rate is measured across a burst.
//!
//! A scene's burst writes `<stem>.poses.csv` beside its shots, each shot's camera eye and look-at
//! in raw WoW coords. `$WOW_CAPTURE_PAN=<yaw>[,<pitch>]`, or a scene's own pan
//! ([`scenarios::scene_pan`]), sweeps the view that many degrees either side of the scene's over
//! the burst ([`pan_pose`]), the eye fixed.
//!
//! ## Running one capture by hand
//! Run through Cargo, never a bare `target/debug/benilla`, which may be stale:
//! ```text
//! WOW_CAPTURE=ui-unitframes \
//!     WOW_CAPTURE_OUT=/tmp/shot.png cargo run -q -p benilla
//! ```
//! `WOW_DATA` is needed only for an install the client does not find on its own
//! (`benilla_formats::wow_data`). A `ui-*` scenario opts the player UI in by declaring a `ui:`
//! fixture ([`scenarios::ui_opted_in`]); `WOW_CAPTURE_UI=1` paints the UI over a world scenario,
//! whose baselines are UI-free by default. `WOW_CAPTURE=list` prints the scenario names.

use std::path::Path;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot, ScreenshotCaptured};
use bevy::time::TimeUpdateStrategy;

use benilla_assets::coords::wow_to_bevy;

use crate::perf::PerfHud;
use benilla_world::dev_state::DebugState;
use benilla_world::schedule::WorldStage;
use benilla_world::terrain_stream::WorldLoadProgress;
use benilla_world::view::WorldCamera;

mod create_stage;
mod depth_probe;
mod fixtures;
mod glue_press;
mod live_shot;
mod phase_probe;
mod pick_probe;
mod probe_auction;
mod probe_bank;
mod probe_bg;
mod probe_bg_queue;
mod probe_binder;
mod probe_book;
mod probe_castcancel;
mod probe_charcreate;
mod probe_charter;
mod probe_chest;
mod probe_clam;
mod probe_crossing;
pub(crate) mod probe_env;
mod probe_gm_ticket;
mod probe_goquest;
mod probe_guard_poi;
mod probe_mail;
mod probe_melee;
mod probe_model_camera;
mod probe_partner;
mod probe_rig;
mod probe_service;
mod probe_stone;
mod probe_taxi;
mod probe_vendor_swap;
mod probes;
mod realm_list;
mod scenarios;
use crate::run_mode::CaptureMode;
pub(crate) use depth_probe::DepthProbePlugin;
use fixtures::seed_ui_fixture;
pub(crate) use live_shot::LiveShotPlugin;
pub(crate) use phase_probe::PhaseProbePlugin;
pub(crate) use pick_probe::PickProbePlugin;
pub(crate) use probe_auction::ProbeAuctionPlugin;
pub(crate) use probe_bank::ProbeBankPlugin;
pub(crate) use probe_bg::ProbeBgPlugin;
pub(crate) use probe_bg_queue::ProbeBgQueuePlugin;
pub(crate) use probe_binder::ProbeBinderPlugin;
pub(crate) use probe_book::ProbeBookPlugin;
pub(crate) use probe_castcancel::ProbeCastCancelPlugin;
pub(crate) use probe_charcreate::ProbeCharCreatePlugin;
pub(crate) use probe_charter::ProbeCharterPlugin;
pub(crate) use probe_chest::ProbeChestPlugin;
pub(crate) use probe_clam::ProbeClamPlugin;
pub(crate) use probe_crossing::ProbeCrossingPlugin;
pub(crate) use probe_gm_ticket::ProbeGmTicketPlugin;
pub(crate) use probe_goquest::ProbeGoQuestPlugin;
pub(crate) use probe_guard_poi::ProbeGuardPoiPlugin;
pub(crate) use probe_mail::ProbeMailPlugin;
pub(crate) use probe_melee::ProbeMeleePlugin;
pub(crate) use probe_model_camera::ProbeModelCameraPlugin;
pub(crate) use probe_partner::ProbePartnerPlugin;
pub(crate) use probe_rig::ProbeRigPlugin;
pub(crate) use probe_service::ProbeServicePlugin;
pub(crate) use probe_stone::ProbeStonePlugin;
pub(crate) use probe_taxi::ProbeTaxiPlugin;
pub(crate) use probe_vendor_swap::ProbeVendorSwapPlugin;
pub(crate) use probes::{
    fx_draw_census_plugin, DressCensusPlugin, EntityCensusPlugin, GroundCensusPlugin,
    JitterMeterPlugin, LiftCensusPlugin, LiveFpsPlugin, NodeProbePlugin, ParticleCensusPlugin,
    ProbeChatPlugin, ProbeClickPlugin, ProbeClock, ProbeDragPlugin, ProbeExitPlugin,
    ProbeFocusPlugin, ProbeGlueLuaPlugin, ProbeHoverPlugin, ProbeKeyPlugin, ProbeLuaPlugin,
    ProbeResizePlugin, RevealAuditPlugin, SchedCensusPlugin, StallPlugin, TrailCensusPlugin,
    UnitVisualsPlugin,
};
pub(crate) use scenarios::ui_opted_in;
use scenarios::{
    Scenario, SubjectKind, TrainerList, UiFixture, GLUE_SCENARIOS, GROUND_EYE, SCENARIOS,
};

pub(crate) mod fxview;
// The scripted aim, swim-pitch and camera-rig drivers order themselves before `player::control`:
// an instrument may name the gameplay system it drives, gameplay never names the instrument.
mod probe_cam;
mod probe_look;
mod probe_pitch;
pub(crate) mod waterfx;

pub(crate) use probe_cam::ProbeCamPlugin;
pub(crate) use probe_look::ProbeLookPlugin;
pub(crate) use probe_pitch::ProbePitchPlugin;

/// The screen a capture starts on, the dev arm of [`crate::run_mode::start_state`]: a glue capture
/// boots onto the login screen, from which it presses its own screen up, any other capture
/// straight in-world, no capture the login screen.
///
/// One of three readers of `$WOW_CAPTURE`, with `run_mode` and `dev_state::deterministic_run`,
/// one per layer and sharing no symbol: keep them in step.
pub(crate) fn start_state() -> crate::client_state::ClientState {
    if glue_capture() || !crate::run_mode::scenario_active() {
        crate::client_state::ClientState::Login
    } else {
        crate::client_state::ClientState::InWorld
    }
}

/// `$WOW_CAPTURE` names a glue scenario; read before the plugins build, since it decides the start
/// screen.
fn glue_capture() -> bool {
    std::env::var("WOW_CAPTURE").is_ok_and(|name| GLUE_SCENARIOS.iter().any(|g| g.name == name))
}

/// The `fxview` effect viewer: spawn one effect or missile model through the game's own
/// `attach_effect_visuals`, let it run `WOW_FX_AGE` seconds and shoot it from a chosen angle. Not a
/// golden scenario: its output depends on the knobs.
///
/// ```text
/// WOW_CAPTURE=fxview WOW_FX_MODEL='Spells\DemonArmor_Impact_Head.mdx' \
///   WOW_FX_AGE=1.2 WOW_FX_AZ=60 WOW_FX_EL=15 WOW_CAPTURE_OUT=/tmp/fx.png cargo run -q -p benilla
/// ```
///
/// Three lanes: the effect pool (`WOW_FX_MODEL`), a unit (`WOW_FX_DISPLAY`, the component set a
/// streamed creature gets) and a GameObject (`WOW_FX_GO`, whose sequence `crate::go_anim`'s state
/// machine picks, `0x5f3cb0`). A missing `GAMEOBJECT_STATE` reads as 0, ACTIVE, which on a model
/// with no `Opened` sequence resolves elsewhere.
///
/// Knobs, defaults in parentheses: `WOW_FX_AGE` s after attach (1.0), `WOW_FX_AZ`/`WOW_FX_EL`
/// orbit degrees (0/10), `WOW_FX_DIST` yd (5), `WOW_FX_FLY` yd/s along the facing (0),
/// `WOW_FX_YAW` degrees (0), `WOW_FX_TURN` deg/s after spawn (0), `WOW_FX_GROUND=1` seats it on
/// the terrain for ground decals, `WOW_FX_HOLD=1` keeps it past one sequence pass,
/// `WOW_FX_MINUTE` the clock (720).
#[derive(Resource)]
pub(crate) struct FxViewRequest {
    pub(crate) model_path: String,
    /// `WOW_FX_DISPLAY`: spawn the subject as a unit with this `CreatureDisplayInfo` id, the A/B
    /// that says whether a defect is the model's or the unit path's; `WOW_FX_MODEL` is optional.
    pub(crate) display: Option<u32>,
    /// `WOW_FX_GO`: spawn the subject as a GameObject with this `GameObjectDisplayInfo` id, the
    /// lane a placed trap renders through.
    pub(crate) go: Option<u32>,
    /// `WOW_FX_GO_STATE`: the descriptor's `GAMEOBJECT_STATE`, default 1 READY, the state vmangos
    /// creates a spell-cast trap in (`SpellEffects.cpp:5655`); 0 shows what an omitted field reads.
    pub(crate) go_state: u32,
    /// `WOW_FX_GO_TYPE`: the `GAMEOBJECT_TYPE_ID`, default 6 TRAP; decides whether
    /// [`crate::go_anim::go_animates`] runs the state machine at all.
    pub(crate) go_type: u32,
    /// `WOW_FX_SCALE`: the unit lane's wire scale.
    pub(crate) scale: f32,
    pub(crate) age: f32,
    pub(crate) az_deg: f32,
    pub(crate) el_deg: f32,
    pub(crate) dist: f32,
    pub(crate) fly: f32,
    pub(crate) yaw_deg: f32,
    pub(crate) turn: f32,
    pub(crate) ground: bool,
    pub(crate) hold: bool,
    /// `WOW_FX_AT=x,y,z`: a raw WoW point in place of [`FXVIEW_POS`], `WOW_MAP` the map. Ground
    /// decals depend on the surface: footprints need `TerrainType.Flags & 1` (snow, sand).
    pub(crate) at: Option<[f32; 3]>,
    /// `WOW_FX_UP`: yards above the resolved seat, for a model authored below its anchor (Arcane
    /// Intellect's star cluster starts 1.6 yd under it) whose opening the terrain would swallow.
    pub(crate) up: f32,
}

/// The fixture's live state, written by `fxview::drive_fx_view` and `drive_capture`.
#[derive(Resource, Default)]
pub(crate) struct FxViewState {
    /// Set once the scene has settled; the fixture spawns only then, so its age is the one asked.
    pub(crate) armed: bool,
    pub(crate) root: Option<Entity>,
    /// `time.elapsed_secs()` when the visuals attached: the age clock's zero.
    pub(crate) attached_at: Option<f32>,
    /// The root was reaped after one sequence pass, as the game reaps a discrete kit;
    /// `WOW_FX_HOLD=1` disables the reap.
    pub(crate) expired: bool,
}

/// Where the fxview effect spawns: mid-air over the Northshire slope, inside the ground
/// scenario's streamed tiles.
pub(crate) const FXVIEW_POS: [f32; 3] = [-8960.0, -145.0, 90.0];

/// How far the `vista` fixture seats its eye above the given position (yd): a standing human's
/// camera pivot, so a `.go xyz` off the debug panel frames what a player there saw.
const VISTA_EYE_HEIGHT: f32 = 2.0;

/// Print the baseline scenario names, one per line, for `WOW_CAPTURE=list`; `scripts/visual.sh`
/// reads its sweep from this. On-demand fixtures are not listed, except that
/// `WOW_CAPTURE_LIST=water` prints the golden water scenes `scripts/water-shots.sh` sweeps.
pub(crate) fn print_scenario_names() {
    if std::env::var("WOW_CAPTURE_LIST").as_deref() == Ok("water") {
        for name in scenarios::water_scene_names() {
            println!("{name}");
        }
        return;
    }
    for s in SCENARIOS {
        println!("{}", s.name);
    }
}

/// Consecutive byte-identical framebuffer readbacks that mean the scene is built: anything still
/// arriving changes the image or does not affect the shot. `$WOW_CAPTURE_STABLE` overrides it.
const STABLE_FRAMES: u32 = 30;

/// The golden water scenes' settle ([`scenarios::water_scene_names`]): they stream in slowly, and
/// [`STABLE_FRAMES`] pass on the one-colour clear frame before the world arrives.
const WATER_STABLE_FRAMES: u32 = 300;

/// The stable run `scene` waits for: `$WOW_CAPTURE_STABLE`, else its own settle.
fn stable_frames(scene: &str) -> u32 {
    std::env::var("WOW_CAPTURE_STABLE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            if scenarios::water_scene_names().any(|n| n == scene) {
                WATER_STABLE_FRAMES
            } else {
                STABLE_FRAMES
            }
        })
}

/// Held frames a scene may take to build before its stable run, so a scene that never settles (a
/// live UI animation) shoots anyway, with a warning that the shot is not reproducible.
const BUILD_CAP_FRAMES: u32 = 1800;

/// The cap on the held build: [`BUILD_CAP_FRAMES`] plus the stable run the shutter waits for, so a
/// longer `$WOW_CAPTURE_STABLE` does not eat into the frames the scene has to build, plus
/// `flat_credit`, the stable runs a golden water scene spent on its clear colour
/// ([`CaptureCtx::flat_credit`]), which are no part of the build.
fn build_cap_frames(scene: &str, flat_credit: u32) -> u32 {
    BUILD_CAP_FRAMES + stable_frames(scene) + flat_credit
}

/// Wall-clock ceiling on a capture run, past which it exits with an error. The other bounds count
/// frames, and macOS can stop granting drawables to a window it is not compositing (about 1 s per
/// `-[CAMetalLayer nextDrawable]`), which stretches the frame caps to half an hour.
const DEADLINE_SECS: u64 = 300;

/// The effective ceiling: `$WOW_CAPTURE_DEADLINE=<secs>`, `0` to disable.
fn capture_deadline() -> Option<Duration> {
    let secs = std::env::var("WOW_CAPTURE_DEADLINE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEADLINE_SECS);
    (secs > 0).then(|| Duration::from_secs(secs))
}

/// Steps of [`CAPTURE_FRAME_DT`] the sims run once the scene is built: the shot's sim age. 150 is
/// 2.5 s, past the 2 s spawn fade (`model_fade::APPEAR_FADE_SECS`) and a flame pool's particle
/// lifetime. `$WOW_CAPTURE_AGE` overrides it for a scene that fills slowly (snow wants ~660).
const AGE_FRAMES: u32 = 150;

/// The golden water scenes' age ([`scenarios::water_scene_names`]): the water probe takes its
/// first cubes 2 s into the age and a slot at a time after that, so a 2.5 s shot races them; at
/// 1000 steps (16.7 s) every cube is in and the shot repeats byte for byte outside the scene's
/// two-state patches (its region file's first comment, `scripts/water_regions/<scene>.toml`).
const WATER_AGE_FRAMES: u32 = 1000;

/// The age `scene` runs: `$WOW_CAPTURE_AGE`, else its own.
fn age_frames(scene: &str) -> u32 {
    std::env::var("WOW_CAPTURE_AGE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            if scenarios::water_scene_names().any(|n| n == scene) {
                WATER_AGE_FRAMES
            } else {
                AGE_FRAMES
            }
        })
}

/// The frozen clock's step, 1/60 s per capture frame, so everything integrated in seconds reaches
/// the shutter in the same state every run. On a live clock two runs of one build differ by up to
/// MAE 0.009, as much as a real render change.
const CAPTURE_FRAME_DT: Duration = Duration::from_nanos(16_666_667);

/// Fixed clock steps between two burst shots, 1/30 s: `$WOW_CAPTURE_BURST_EVERY` overrides it.
const BURST_EVERY: u32 = 2;

/// `$WOW_CAPTURE_BURST`: shots written after the age, 0 for the one shot at `$WOW_CAPTURE_OUT`.
fn burst_shots() -> u32 {
    std::env::var("WOW_CAPTURE_BURST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// `$WOW_CAPTURE_BURST_EVERY`, at least one step, since the clock runs at least one between shots.
fn burst_every() -> u32 {
    std::env::var("WOW_CAPTURE_BURST_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(BURST_EVERY)
        .max(1)
}

/// A burst's pan, `[yaw, pitch]` degrees either side of the view: `$WOW_CAPTURE_PAN=<yaw>[,<pitch>]`,
/// else the scene's own ([`scenarios::scene_pan`]).
fn burst_pan(scene: &str) -> Option<[f32; 2]> {
    std::env::var("WOW_CAPTURE_PAN")
        .ok()
        .and_then(|v| {
            let c: Vec<f32> = v.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            match c.as_slice() {
                [yaw] => Some([*yaw, 0.0]),
                [yaw, pitch] => Some([*yaw, *pitch]),
                _ => None,
            }
        })
        .or_else(|| scenarios::scene_pan(scene))
}

/// Burst shot `shot` of `shots` on a pan: the look-at point turned about the fixed eye, its
/// heading and elevation swept linearly from `−pan` to `+pan` degrees over the burst, at the same
/// distance. Raw WoW coords (Z up, headings counter-clockwise from +X). Without a pan, or for a
/// single shot, the scene's own `look`.
fn pan_pose(
    eye: [f32; 3],
    look: [f32; 3],
    pan: Option<[f32; 2]>,
    shot: u32,
    shots: u32,
) -> [f32; 3] {
    let Some([yaw, pitch]) = pan.filter(|_| shots > 1) else {
        return look;
    };
    let s = 2.0 * shot as f32 / (shots - 1) as f32 - 1.0;
    let d = Vec3::from(look) - Vec3::from(eye);
    let len = d.length();
    let heading = d.y.atan2(d.x) + (s * yaw).to_radians();
    let elev = (d.z / len).clamp(-1.0, 1.0).asin() + (s * pitch).to_radians();
    let dir = Vec3::new(
        elev.cos() * heading.cos(),
        elev.cos() * heading.sin(),
        elev.sin(),
    );
    (Vec3::from(eye) + dir * len).to_array()
}

/// `<stem>.poses.csv` beside a burst's shots: `shot,ex,ey,ez,lx,ly,lz`, each shot's camera eye and
/// look-at in raw WoW coords, which the pan rows of `scripts/water_metrics.py` reproject through.
fn write_poses(ctx: &CaptureCtx, scenario: &Scenario) -> std::io::Result<()> {
    let mut csv = String::from("shot,ex,ey,ez,lx,ly,lz\n");
    let e = scenario.eye;
    for shot in 0..ctx.burst {
        let l = pan_pose(e, scenario.look, ctx.pan, shot, ctx.burst);
        csv.push_str(&format!(
            "{shot},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}\n",
            e[0], e[1], e[2], l[0], l[1], l[2]
        ));
    }
    std::fs::write(Path::new(&ctx.out).with_extension("poses.csv"), csv)
}

/// Whether `name` is a burst shot of `stem`: `<stem>-NNN.<ext>`, three digits or more.
fn is_burst_shot(name: &str, stem: &str, ext: &str) -> bool {
    name.strip_prefix(stem)
        .and_then(|r| r.strip_prefix('-'))
        .and_then(|r| r.strip_suffix(ext))
        .and_then(|r| r.strip_suffix('.'))
        .is_some_and(|n| n.len() >= 3 && n.bytes().all(|b| b.is_ascii_digit()))
}

/// Remove what an earlier capture to `out` left, so the exit's check that every shot was written
/// and a reader of the series see this run's files alone: the shot at `out`, or for a burst every
/// `<stem>-NNN` shot beside it, however long that burst was, and its poses. A still leaves a burst
/// of the same stem in place, and a burst a still.
fn clear_old_shots(out: &str, burst: u32) -> std::io::Result<()> {
    let gone = |p: &Path| match std::fs::remove_file(p) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    };
    let path = Path::new(out);
    if burst == 0 {
        return gone(path);
    }
    gone(&path.with_extension("poses.csv"))?;
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let ext = path
        .extension()
        .unwrap_or(std::ffi::OsStr::new("png"))
        .to_string_lossy();
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if entry
            .file_name()
            .to_str()
            .is_some_and(|n| is_burst_shot(n, &stem, &ext))
        {
            gone(&entry.path())?;
        }
    }
    Ok(())
}

/// A shot's clock for the log: a fixture's age, the virtual clock past the moment its visuals
/// attached, which is the same on every run; else the virtual clock, which the frozen build makes
/// the scene's age.
fn shot_clock(game_time: &Time, fx: Option<&FxViewState>) -> f64 {
    let now = game_time.elapsed_secs_f64();
    fx.and_then(|s| s.attached_at)
        .map_or(now, |t0| now - f64::from(t0))
}

/// Burst shot `shot`'s file: `<stem>-NNN.<ext>` beside `out`.
fn burst_path(out: &str, shot: u32) -> String {
    let path = Path::new(out);
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let ext = path
        .extension()
        .unwrap_or(std::ffi::OsStr::new("png"))
        .to_string_lossy();
    path.with_file_name(format!("{stem}-{shot:03}.{ext}"))
        .to_string_lossy()
        .into_owned()
}

/// Cap on frames waiting for the screenshot save, so a capture never hangs.
const SAVE_TIMEOUT_FRAMES: u32 = 120;
/// A couple of grace frames after the save completes before exiting, so the file is flushed.
const EXIT_GRACE_FRAMES: u32 = 3;

/// Frames the perf probe discards after uncapping vsync (present-mode switch + pipeline settle).
const PROBE_WARMUP_FRAMES: u32 = 60;

/// Marks a stability readback, so [`Phase::Saving`]'s wait cannot mistake it for the shot.
#[derive(Component)]
struct StabilityShot;

/// The image-stability tracker. Compares whole bytes, not a hash: a `memcmp` is free next to the
/// readback and needs no collision argument.
#[derive(Resource, Default)]
struct FrameWatch {
    prev: Option<Vec<u8>>,
    stable: u32,
    /// A readback is outstanding; only one at a time, so `stable` counts distinct frames.
    in_flight: bool,
}

/// Observer for a stability readback: compare against the previous frame, count, and free the slot.
fn watch_frame(shot: On<ScreenshotCaptured>, mut watch: ResMut<FrameWatch>) {
    watch.in_flight = false;
    let Some(bytes) = shot.image.data.as_ref() else {
        return; // no payload (a zero-sized target): no evidence
    };
    watch.stable = if watch.prev.as_deref() == Some(bytes.as_slice()) {
        watch.stable + 1
    } else {
        0
    };
    watch.prev = Some(bytes.clone());
}

/// Capture lifecycle. Advances one step per frame in [`drive_capture`].
#[derive(Clone, Copy)]
enum Phase {
    /// Clock held; waiting for the image to stop changing ([`STABLE_FRAMES`]).
    Building(u32),
    /// Clock released: running exactly [`age_frames`] steps of [`CAPTURE_FRAME_DT`].
    Aging(u32),
    /// Fixture viewers only: fixture armed, waiting for it to attach and reach its age.
    FxAging,
    /// Perf probe (`$WOW_FPS_PROBE`): vsync off, discarding warm-up frames.
    ProbeWarmup(u32),
    /// Perf probe: sampling frame times, then print and exit.
    Probing(u32),
    /// Screenshot requested; waiting for the async save's `Capturing` marker to appear and clear.
    Saving { frames: u32, seen: bool },
    /// Burst only: clock released between two shots, running [`CaptureCtx::burst_every`] steps.
    Stepping(u32),
    /// Save done; a few grace frames, then `AppExit`.
    Done(u32),
}

/// A UI capture's stance bar, once: the demo's stances among the known spells, which the bar's
/// getters read its list off (`crate::ui_shapeshift`), each entry off the VM copy the demo feed
/// seeds while no player is held. The spellbook fixture's book is its own known spells, so it
/// lists none.
fn seed_demo_forms(
    ctx: Res<CaptureCtx>,
    actions: Option<ResMut<crate::ui_action::PlayerActions>>,
    mut done: Local<bool>,
) {
    let Some(fixture) = ctx.scenario.and_then(|s| s.ui) else {
        return;
    };
    let Some(mut actions) = actions.filter(|_| !*done) else {
        return;
    };
    *done = true;
    if fixture != UiFixture::SpellBook {
        actions.spells.extend(crate::ui_script::DEMO_STANCES);
        // The stored form list, as their learns leave it: the stances are in bar order already.
        actions.forms.extend(crate::ui_script::DEMO_STANCES);
    }
}

#[derive(Resource)]
struct CaptureCtx {
    /// The in-world viewpoint; `None` for a glue-screen capture, which has no world.
    scenario: Option<Scenario>,
    /// The scenario's name, for the output path and the probe line.
    name: &'static str,
    out: String,
    /// [`burst_shots`]: shots after the age, 0 for the one shot at `out`.
    burst: u32,
    /// [`burst_every`]: fixed clock steps between two burst shots.
    burst_every: u32,
    /// [`burst_pan`] on a burst, else `None`.
    pan: Option<[f32; 2]>,
    /// A golden water scene's settle met a one-colour frame and built on (logged once).
    waited_flat: bool,
    /// The frames of the stable runs a golden water scene spent on its clear colour, which extend
    /// its build cap ([`build_cap_frames`]), up to [`BUILD_CAP_FRAMES`] of them.
    flat_credit: u32,
    /// The burst shot being taken, `0..burst`.
    shot: u32,
    phase: Phase,
    /// The UI fixture is seeded ([`seed_ui_fixture`]).
    ui_seeded: bool,
    /// `$WOW_FPS_PROBE`: sample this many frames, vsync off, and print frame-time stats and scene
    /// counts instead of screenshotting; 0 is a normal capture.
    probe_frames: u32,
    /// The frozen clock ([`CAPTURE_FRAME_DT`]) is in force. A screenshot run keeps it to the end; a
    /// perf probe builds and ages frozen like a capture, so it measures a scene of the same age on
    /// any machine, then drops it because its measurement is the real frame cost.
    frozen_clock: bool,
    /// `$WOW_RESIZE` is applied (once, at first settle).
    resized: bool,
    /// [`quality_switches`]: the tiers still to switch to, one per settle, in order.
    quality_switches: Vec<benilla_world::liquid::WaterQuality>,
    probe_samples: Vec<f32>,
    /// The `$WOW_GPU_MS=1` meter's GPU milliseconds over the window; empty when it is off.
    gpu_samples: Vec<f32>,
    /// Process CPU seconds at the first sample, the baseline for the probe line's `cpu_ms`.
    probe_cpu_start: Option<f64>,
    /// Wall-clock start of the run ([`capture_deadline`]). A real [`Instant`]: under the frozen
    /// clock `Time<Real>` is itself manual and would only count frames.
    started: Instant,
    deadline: Option<Duration>,
    /// Frames [`drive_capture`] has run, the denominator of the deadline message's rate.
    frames: u32,
    /// The deadline already fired; `AppExit` is written and no phase advances again.
    bailed: bool,
}

impl CaptureCtx {
    /// Where the shot being taken goes: `out`, or the burst shot's [`burst_path`].
    fn shot_path(&self) -> String {
        if self.burst == 0 {
            self.out.clone()
        } else {
            burst_path(&self.out, self.shot)
        }
    }

    /// Every file the capture writes, in shot order: `out`, or each burst shot's [`burst_path`].
    fn shot_paths(&self) -> Vec<String> {
        if self.burst == 0 {
            vec![self.out.clone()]
        } else {
            (0..self.burst).map(|i| burst_path(&self.out, i)).collect()
        }
    }
}

/// `$WOW_RESIZE=WxH`: resize the window (logical px) once the image first settles, then settle
/// again before shooting.
fn resize_request() -> Option<(u32, u32)> {
    let v = std::env::var("WOW_RESIZE").ok()?;
    let (w, h) = v.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

/// `$WOW_WATER_QUALITY_SWITCH=<tier>[,<tier>…]`: each time the image settles, set `waterQuality`
/// to the next tier (a number or a name, [`benilla_world::liquid::WaterQuality::parse`]) as
/// `/console waterQuality` would, then settle again; the shot is taken at the last. The boot tier
/// is `$WOW_WATER_QUALITY`'s.
fn quality_switches() -> Vec<benilla_world::liquid::WaterQuality> {
    std::env::var("WOW_WATER_QUALITY_SWITCH")
        .map(|v| {
            v.split(',')
                .filter_map(benilla_world::liquid::WaterQuality::parse)
                .collect()
        })
        .unwrap_or_default()
}

/// The present mode a perf probe uncaps to, shared with the live probe.
///
/// `AutoNoVsync` by default: on macOS/Metal explicit `Immediate` rails near 16.6 ms and takes
/// 1.0-1.5 s drawable stalls, while `AutoNoVsync` uncaps when the window server grants it; when
/// it does not, `cpu_ms` on the probe line is the metric. `WOW_PROBE_UNCAP=immediate` selects
/// `Immediate`. `WOW_PROBE_UNCAP=vsync` keeps vsync for a sitting whose legs never rail; the vsync
/// wait costs process CPU unevenly, so read it only against a same-mode baseline.
pub(crate) fn probe_uncap_mode() -> bevy::window::PresentMode {
    match std::env::var("WOW_PROBE_UNCAP").as_deref() {
        Ok("immediate") => bevy::window::PresentMode::Immediate,
        Ok("vsync") => bevy::window::PresentMode::AutoVsync,
        _ => bevy::window::PresentMode::AutoNoVsync,
    }
}

pub(crate) struct CapturePlugin;

impl Plugin for CapturePlugin {
    fn build(&self, app: &mut App) {
        // The `waterfx` dummy moves before the foam emitter reads its motion.
        app.add_systems(
            Update,
            (waterfx::spawn, waterfx::drive)
                .chain()
                .before(benilla_world::water_fx::WaterFoamSet)
                .in_set(benilla_world::schedule::WorldStage::Present),
        );
        // The `fxview` driver creates the subject's display-cache entry, so it runs before the
        // frame's display build, after the net stage.
        app.add_systems(
            Update,
            fxview::drive_fx_view
                .in_set(crate::entities::EntityVisualsSet)
                .before(crate::entities::DisplayBuildSet)
                .after(benilla_world::schedule::WorldStage::Net),
        );
        let name = std::env::var("WOW_CAPTURE").unwrap_or_default();
        // A glue capture has no map, viewpoint or fixture; only the shutter is shared.
        let glue = GLUE_SCENARIOS.iter().find(|g| g.name == name).copied();
        // A pick in the environment wins over the scenario's, so one scenario photographs every
        // race's stage.
        if let Some(pick) = glue.and_then(|g| g.pick) {
            app.insert_resource(create_stage::CreatePick::resolve(pick));
        }
        // `fxview`, `waterfx`, `vista` and `name-close` are built from knobs, not the tables, and
        // are never in the golden sweep.
        let scenario: Option<Scenario> = if glue.is_some() {
            None
        } else {
            Some(if name == "fxview" {
                let at = std::env::var("WOW_FX_AT").ok().and_then(|v| {
                    let c: Vec<f32> = v.split(',').filter_map(|p| p.trim().parse().ok()).collect();
                    (c.len() == 3).then(|| [c[0], c[1], c[2]])
                });
                let display: Option<u32> = std::env::var("WOW_FX_DISPLAY")
                    .ok()
                    .and_then(|v| v.trim().parse().ok());
                let go: Option<u32> = std::env::var("WOW_FX_GO")
                    .ok()
                    .and_then(|v| v.trim().parse().ok());
                let model_path = match (std::env::var("WOW_FX_MODEL"), display.or(go)) {
                    (Ok(p), _) => p,
                    (Err(_), Some(_)) => String::new(), // the id lanes name their own model
                    (Err(_), None) => {
                        eprintln!(
                            "WOW_CAPTURE=fxview needs WOW_FX_MODEL=<internal .mdx/.m2 path>, \
                         WOW_FX_DISPLAY=<CreatureDisplayInfo id> or \
                         WOW_FX_GO=<GameObjectDisplayInfo id>"
                        );
                        std::process::exit(2);
                    }
                };
                let knob = |k: &str, d: f32| {
                    std::env::var(k)
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(d)
                };
                app.insert_resource(FxViewRequest {
                    model_path,
                    display,
                    go,
                    go_state: knob("WOW_FX_GO_STATE", 1.0) as u32,
                    go_type: knob("WOW_FX_GO_TYPE", 6.0) as u32,
                    scale: knob("WOW_FX_SCALE", 1.0),
                    age: knob("WOW_FX_AGE", 1.0),
                    az_deg: knob("WOW_FX_AZ", 0.0),
                    el_deg: knob("WOW_FX_EL", 10.0),
                    dist: knob("WOW_FX_DIST", 5.0),
                    fly: knob("WOW_FX_FLY", 0.0),
                    yaw_deg: knob("WOW_FX_YAW", 0.0),
                    turn: knob("WOW_FX_TURN", 0.0),
                    ground: knob("WOW_FX_GROUND", 0.0) > 0.5,
                    hold: knob("WOW_FX_HOLD", 0.0) > 0.5,
                    up: knob("WOW_FX_UP", 0.0),
                    at,
                })
                .init_resource::<FxViewState>();
                Scenario {
                    name: "fxview",
                    map: None,
                    eye: GROUND_EYE, // overridden per frame by the orbit in `pin_scene`
                    look: at.unwrap_or(FXVIEW_POS),
                    minute: knob("WOW_FX_MINUTE", 720.0) as u32,
                    ui: None,
                }
            } else if name == "waterfx" {
                // The water-foam viewer: a wading unit over a synthetic lattice, or in real
                // liquid with `WOW_WFX_AT`, on a fixed orbit. Knobs: `WOW_WFX_MODE`
                // (ring|wake|turn), `_SPEED` yd/s, `_HEAD` deg, `_AGE` s, `_DEPTH` yd, and the
                // camera's `_AZ`/`_EL`/`_DIST`.
                let knob = |k: &str, d: f32| {
                    std::env::var(k)
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(d)
                };
                let mode = match std::env::var("WOW_WFX_MODE").as_deref() {
                    Ok("wake") => waterfx::WfxMode::Wake,
                    Ok("turn") => waterfx::WfxMode::Turn,
                    _ => waterfx::WfxMode::Ring,
                };
                // Rig centre, raw WoW coords: a synthetic surface over the Northshire ground by
                // default; `WOW_WFX_AT` puts it in the real liquid there, `z` its surface.
                let at = std::env::var("WOW_WFX_AT").ok().and_then(|v| {
                    let c: Vec<f32> = v.split(',').filter_map(|p| p.trim().parse().ok()).collect();
                    (c.len() == 3).then(|| [c[0], c[1], c[2]])
                });
                let live = at.is_some();
                let center = at.unwrap_or([-8961.0_f32, -145.0, 95.0]);
                let az = knob("WOW_WFX_AZ", 180.0).to_radians();
                let el = knob("WOW_WFX_EL", 35.0).to_radians();
                let dist = knob("WOW_WFX_DIST", 14.0);
                let eye = [
                    center[0] + dist * el.cos() * az.cos(),
                    center[1] + dist * el.cos() * az.sin(),
                    center[2] + dist * el.sin(),
                ];
                app.insert_resource(waterfx::WaterFxView {
                    mode,
                    speed: knob("WOW_WFX_SPEED", 4.0),
                    heading: knob("WOW_WFX_HEAD", 0.0).to_radians(),
                    age: knob("WOW_WFX_AGE", 1.5),
                    center,
                    depth: knob("WOW_WFX_DEPTH", 0.5),
                    live,
                })
                .init_resource::<FxViewState>();
                Scenario {
                    name: "waterfx",
                    // `WOW_MAP` picks the map for a live rig.
                    map: None,
                    eye,
                    look: center,
                    minute: 720,
                    ui: None,
                }
            } else if name == "vista" {
                // The arbitrary-viewpoint instrument: any position, heading and clock, e.g.
                //
                //   WOW_CAPTURE=vista WOW_VISTA_AT=-5841.9,-3802.4,-59.7 WOW_VISTA_FACE=24 \
                //     WOW_VISTA_MIN=1052 WOW_FARCLIP=320 cargo run -q -p benilla
                //
                // `WOW_VISTA_AT` is the player's feet (the eye sits `VISTA_EYE_HEIGHT` above),
                // `WOW_VISTA_FACE` the debug panel's facing in degrees (0 = +X, counter-clockwise),
                // `WOW_VISTA_PITCH` degrees up (0), `WOW_VISTA_MIN` the game minute (720).
                // `WOW_FARCLIP` matches a player's far-clip slider.
                let knob = |k: &str, d: f32| {
                    std::env::var(k)
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(d)
                };
                let Some(at) = std::env::var("WOW_VISTA_AT").ok().and_then(|v| {
                    let c: Vec<f32> = v.split(',').filter_map(|p| p.trim().parse().ok()).collect();
                    (c.len() == 3).then(|| [c[0], c[1], c[2]])
                }) else {
                    eprintln!("WOW_CAPTURE=vista needs WOW_VISTA_AT=<x,y,z> (raw WoW coords)");
                    std::process::exit(2);
                };
                let face = knob("WOW_VISTA_FACE", 0.0).to_radians();
                let pitch = knob("WOW_VISTA_PITCH", 0.0).to_radians();
                let eye = [at[0], at[1], at[2] + VISTA_EYE_HEIGHT];
                // A look point far enough out that the framing is the heading, not the distance.
                let d = 500.0_f32;
                Scenario {
                    name: "vista",
                    // `WOW_MAP` picks the map (a `Map.dbc` id).
                    map: None,
                    eye,
                    look: [
                        eye[0] + d * pitch.cos() * face.cos(),
                        eye[1] + d * pitch.cos() * face.sin(),
                        eye[2] + d * pitch.sin(),
                    ],
                    minute: knob("WOW_VISTA_MIN", 720.0) as u32,
                    ui: None,
                }
            } else if name == "name-close" {
                // The magnified overhead-name instrument ([`scenarios::NAME_CLOSE_AT`]).
                let knob = |k: &str, d: f32| {
                    std::env::var(k)
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(d)
                };
                let at = scenarios::NAME_CLOSE_AT;
                // The camera orbits and aims at the name, `WOW_NAME_H` above the unit's feet.
                let name_at = [at[0], at[1], at[2] + knob("WOW_NAME_H", 1.4)];
                let (dist, az, el) = (
                    knob("WOW_NAME_DIST", 4.0),
                    knob("WOW_NAME_AZ", 124.0).to_radians(),
                    knob("WOW_NAME_EL", 8.0).to_radians(),
                );
                Scenario {
                    name: "name-close",
                    map: Some(scenarios::MAP_AZEROTH),
                    eye: [
                        name_at[0] + dist * el.cos() * az.cos(),
                        name_at[1] + dist * el.cos() * az.sin(),
                        name_at[2] + dist * el.sin(),
                    ],
                    look: name_at,
                    minute: 720,
                    ui: Some(UiFixture::NameWater),
                }
            } else if let Some(&s) = SCENARIOS
                .iter()
                .chain(scenarios::ON_DEMAND.iter())
                .find(|s| s.name == name)
            {
                s
            } else {
                let glue_known: Vec<_> = GLUE_SCENARIOS.iter().map(|g| g.name).collect();
                let known: Vec<_> = SCENARIOS
                    .iter()
                    .chain(scenarios::ON_DEMAND.iter())
                    .map(|s| s.name)
                    .collect();
                eprintln!(
                "WOW_CAPTURE={name:?} is not a known scenario; choose one of: {known:?}, {glue_known:?} (or fxview, waterfx, name-close)"
            );
                std::process::exit(2);
            })
        };
        // Seed the map before `world_map::load_world_map` reads `WOW_MAP` at `Startup`, the one
        // place `CurrentMap` is set server-less; a `map: None` instrument leaves the knob alone.
        if let Some(m) = scenario.as_ref().and_then(|s| s.map) {
            std::env::set_var("WOW_MAP", m.to_string());
        }
        // A scene's forced weather rides `WOW_WEATHER`, which the weather tick reads once; a
        // script already in the environment wins.
        if let Some(w) = scenario
            .as_ref()
            .and_then(|s| scenarios::scene_weather(s.name))
        {
            if std::env::var_os("WOW_WEATHER").is_none() {
                std::env::set_var("WOW_WEATHER", w);
            }
        }
        let shot_name = scenario.map(|s| s.name).or(glue.map(|g| g.name)).unwrap();
        let out = std::env::var("WOW_CAPTURE_OUT")
            .unwrap_or_else(|_| format!("target/visual/{shot_name}.png"));
        if let Some(parent) = Path::new(&out).parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!(
                    "capture: cannot create output dir {}: {e}",
                    parent.display()
                );
                std::process::exit(2);
            }
        }
        let probe_frames = std::env::var("WOW_FPS_PROBE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        // A perf probe writes no image, so it leaves the files at `out` alone.
        if probe_frames == 0 {
            if let Err(e) = clear_old_shots(&out, burst_shots()) {
                eprintln!("capture: cannot remove an earlier capture's shots at {out}: {e}");
                std::process::exit(2);
            }
        }
        // The frozen clock for every run; a probe switches back to `Automatic` before its first
        // sample, or it would measure a constant 16.67 ms ([`CaptureCtx::frozen_clock`]).
        app.insert_resource(TimeUpdateStrategy::ManualDuration(CAPTURE_FRAME_DT))
            .add_systems(Startup, hold_clock);
        app.insert_resource(CaptureMode)
            .init_resource::<FrameWatch>()
            .insert_resource(CaptureCtx {
                scenario,
                name: shot_name,
                out,
                burst: burst_shots(),
                burst_every: burst_every(),
                pan: burst_pan(shot_name).filter(|_| burst_shots() > 1),
                waited_flat: false,
                flat_credit: 0,
                shot: 0,
                phase: Phase::Building(0),
                ui_seeded: false,
                probe_frames,
                frozen_clock: true,
                resized: false,
                quality_switches: quality_switches(),
                probe_samples: Vec::new(),
                gpu_samples: Vec::new(),
                probe_cpu_start: None,
                started: Instant::now(),
                deadline: capture_deadline(),
                frames: 0,
                bailed: false,
            })
            .add_systems(Update, pin_scene.in_set(WorldStage::Present))
            // Before `UnitFeed`: the seed stands in for wire data live play delivers on earlier
            // frames, so the same frame's feeds and the one-shot `MERCHANT_SHOW` paint must see it.
            .add_systems(
                Update,
                (seed_demo_forms, seed_ui_fixture)
                    .chain()
                    .before(crate::ui_unit::UnitFeed),
            )
            .add_systems(
                Update,
                (
                    accept_agreements,
                    realm_list::seed_realm_list,
                    create_stage::seed_create_stage,
                )
                    .chain(),
            )
            .add_systems(Last, (drive_capture, log_sky).chain());
    }
}

/// A glue capture photographs the account form, not a fresh state folder's user agreements: each
/// one the login screen holds up is accepted as a player accepts it.
fn accept_agreements(ctx: Res<CaptureCtx>, stock: crate::ui_script::glue::StockGlueProbe) {
    if ctx.scenario.is_some() {
        return;
    }
    if let Some(vm) = Some(stock.vm()).filter(|vm| vm.frame_visible("TOSFrame")) {
        glue_press::press_accept_agreements(vm);
    }
}

/// Log the sky once the image settles: the scene's minute, the view's heading, and the visible
/// sun's and moon's elevations and bearings off that heading, so a shot that needs a body above the
/// horizon and ahead of the view is checked from its log. Headings are WoW's, counter-clockwise
/// from north (+X), so a positive bearing off the view is to its left. The water's look and its
/// quality tier follow on their own line, so a shot's log says which tier drew it.
fn log_sky(
    ctx: Res<CaptureCtx>,
    lighting: Option<Res<benilla_world::lighting::WowLighting>>,
    cam: Query<&Transform, With<WorldCamera>>,
    water: (
        Option<Res<benilla_world::liquid::WaterStyle>>,
        Option<Res<benilla_world::liquid::WaterQuality>>,
    ),
    mut logged: Local<bool>,
) {
    if *logged || matches!(ctx.phase, Phase::Building(_)) {
        return;
    }
    *logged = true;
    if let (Some(style), Some(quality)) = water {
        info!(
            "capture: {} water: {}, waterQuality {} ({:?})",
            ctx.name,
            if style.is_stylised() {
                "Improved"
            } else {
                "Reference"
            },
            quality.cvar(),
            *quality,
        );
    }
    let (Some(scenario), Some(light), Some(cam)) = (ctx.scenario, lighting, cam.iter().next())
    else {
        return;
    };
    let elevation = |to_body: Vec3| to_body.normalize_or_zero().y.asin().to_degrees();
    let heading = |dir: Vec3| {
        let [x, y, _] = benilla_assets::coords::bevy_to_wow(dir);
        y.atan2(x).to_degrees().rem_euclid(360.0)
    };
    let view = heading(*cam.forward());
    let off_view = |to_body: Vec3| (heading(to_body) - view + 540.0).rem_euclid(360.0) - 180.0;
    info!(
        "capture: {} at minute {}, view heading {:.1}°: sun elevation {:+.1}°, {:+.1}° off the \
         view; moon elevation {:+.1}°, {:+.1}° off the view",
        ctx.name,
        scenario.minute,
        view,
        elevation(light.celestial_dir),
        off_view(light.celestial_dir),
        elevation(light.moon_dir_white),
        off_view(light.moon_dir_white),
    );
}

/// Each frame, force the capture conditions: pinned time of day, no perf HUD, the fixed camera.
/// Runs in `WorldStage::Present`, after terrain streaming reads the camera.
fn pin_scene(
    ctx: Res<CaptureCtx>,
    mut debug: ResMut<DebugState>,
    mut perf: ResMut<PerfHud>,
    fx_req: Option<Res<FxViewRequest>>,
    fx_state: Option<Res<FxViewState>>,
    mut player: ResMut<crate::player::Player>,
    roots: Query<&Transform, Without<WorldCamera>>,
    mut cam: Query<&mut Transform, With<WorldCamera>>,
) {
    perf.visible = false;
    // A glue screen has no world, clock or camera to pin.
    let Some(scenario) = ctx.scenario else {
        return;
    };
    debug.lighting.follow_server_time = false;
    debug.lighting.manual_minute = scenario.minute;

    // `WOW_MM_PROBE=x,y,z` (raw WoW coords) places an active player there, so the interior
    // minimap renders server-less, with the camera above so the WMO streams in around it.
    let probe = std::env::var("WOW_MM_PROBE").ok().and_then(|s| {
        let v: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (v.len() == 3).then(|| [v[0], v[1], v[2]])
    });
    if let Some(p) = probe {
        player.pos = wow_to_bevy(p);
        player.active = true;
        player.detached = false;
        // Off to the side: straight down degenerates `looking_at` with Y up.
        let eye = wow_to_bevy([p[0] - 25.0, p[1] - 25.0, p[2] + 40.0]);
        for mut t in &mut cam {
            *t = Transform::from_translation(eye).looking_at(wow_to_bevy(p), Vec3::Y);
        }
        return;
    }

    let (eye, look) = match (&fx_req, &fx_state) {
        // fxview: orbit the fixture's live root (a missile moves), aimed one yard up, where the
        // effect models author their bodies.
        (Some(req), Some(state)) => {
            let root_pos = state
                .root
                .and_then(|r| roots.get(r).ok())
                .map(|t| t.translation)
                .unwrap_or_else(|| wow_to_bevy(FXVIEW_POS));
            let look = root_pos + Vec3::Y;
            let orbit = Quat::from_rotation_y(req.az_deg.to_radians())
                * Quat::from_rotation_x(-req.el_deg.to_radians());
            (look + orbit * (Vec3::Z * req.dist), look)
        }
        _ => (
            wow_to_bevy(scenario.eye),
            wow_to_bevy(pan_pose(
                scenario.eye,
                scenario.look,
                ctx.pan,
                ctx.shot,
                ctx.burst,
            )),
        ),
    };
    for mut t in &mut cam {
        *t = Transform::from_translation(eye).looking_at(look, Vec3::Y);
    }
}

/// Hold the game clock at zero until the image stops changing; [`drive_capture`] releases it.
///
/// Tiles spawn under a wall-clock budget, so without the hold an emitter would age from whichever
/// frame its tile arrived on; held, every emitter has run exactly [`age_frames`] steps at the
/// shutter. Streaming and the save are frame-driven and run on regardless.
fn hold_clock(mut clock: ResMut<Time<Virtual>>) {
    clock.pause();
}

/// The scene-population queries the `FPS_PROBE` line prints, bundled because `drive_capture` sits
/// at Bevy's 16-parameter ceiling.
#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct ProbeCensus<'w, 's> {
    particles: Query<'w, 's, &'static benilla_world::particles::ParticleEmitter>,
    parts: Query<'w, 's, &'static ViewVisibility, With<benilla_world::model_render::ModelPart>>,
    entities: Query<'w, 's, ()>,
}

/// Drive the capture lifecycle: settle, age, screenshot or probe, exit.
fn drive_capture(
    mut ctx: ResMut<CaptureCtx>,
    mut watch: ResMut<FrameWatch>,
    // Excludes stability readbacks, or one in flight could satisfy the real shot's wait.
    capturing: Query<(), (With<Capturing>, Without<StabilityShot>)>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
    // `ResMut` to re-anchor the clock at the probe's release (`Phase::Aging`).
    mut time: ResMut<Time<bevy::time::Real>>,
    // `Cvars` for `$WOW_WATER_QUALITY_SWITCH`.
    (mut windows, mut cvars): (
        Query<&mut Window, With<bevy::window::PrimaryWindow>>,
        Option<ResMut<crate::cvars::Cvars>>,
    ),
    census: ProbeCensus,
    fx_req: Option<Res<FxViewRequest>>,
    wfx_req: Option<Res<waterfx::WaterFxView>>,
    mut fx_state: Option<ResMut<FxViewState>>,
    // The harness's one virtual-clock read, allowlisted in
    // `probes::probe_schedules_read_the_wall_clock`: the fixture's age is measured on the clock the
    // effect animates on. Everything schedule-shaped reads [`probes::ProbeClock`].
    game_time: Res<Time>,
    mut clock: ResMut<Time<Virtual>>,
    // Switched to `Automatic` at the probe's release (`Phase::Aging`).
    mut time_strategy: ResMut<TimeUpdateStrategy>,
    // `None` on a glue-screen capture, which builds no composite lane.
    backdrop: Option<Res<crate::world_backdrop::WorldBackdrop>>,
    // `$WOW_GPU_MS=1`'s whole-frame GPU clock — `None` when the meter is off, because nothing
    // registers then. See [`CaptureCtx::gpu_samples`] for why a vista probe needs it at all.
    gpu: Option<Res<crate::perf::GpuMsShared>>,
) {
    // The wall-clock ceiling covers every phase: a fixture may never attach, a readback may never
    // land ([`capture_deadline`]).
    if ctx.bailed {
        return;
    }
    ctx.frames += 1;
    if let Some(limit) = ctx.deadline {
        let elapsed = ctx.started.elapsed();
        if elapsed > limit {
            ctx.bailed = true;
            error!(
                "capture: DEADLINE — {} ran {:.0}s without finishing ({} frames, {:.1} fps). \
                 The harness bounds itself in frames, so this is what a machine that stopped \
                 granting frames looks like; at 1 fps the frame caps are half an hour. No \
                 image written. ($WOW_CAPTURE_DEADLINE=<secs>, 0 disables.)",
                ctx.name,
                elapsed.as_secs_f32(),
                ctx.frames,
                ctx.frames as f32 / elapsed.as_secs_f32(),
            );
            exit.write(AppExit::error());
            return;
        }
    }
    let fixture_age = fx_req
        .as_ref()
        .map(|r| r.age)
        .or(wfx_req.as_ref().map(|r| r.age));
    ctx.phase = match ctx.phase {
        Phase::Building(n) => {
            // One readback at a time, requested only here, marked so `Saving` cannot mistake it.
            if !watch.in_flight {
                watch.in_flight = true;
                commands
                    .spawn((Screenshot::primary_window(), StabilityShot))
                    .observe(watch_frame);
            }
            let stable = stable_frames(ctx.name);
            let capped = n + 1 >= build_cap_frames(ctx.name, ctx.flat_credit);
            if capped {
                warn!(
                    "capture: the image never settled in {} frames ({} stable); the shot is not \
                     reproducible",
                    n + 1,
                    watch.stable,
                );
            }
            // Stability does not prove content: a lane that renders nothing is stable too, and so
            // is the clear colour a world scene shows before it streams in. Color channels only,
            // since alpha is opaque even on a black frame.
            if watch.stable >= stable || capped {
                if let Some((first, rest)) = watch
                    .prev
                    .as_deref()
                    .and_then(|px| px.as_chunks::<4>().0.split_first())
                {
                    let flat = rest.iter().all(|p| p[..3] == first[..3]);
                    // A golden water scene builds on until its world is in: its map can show
                    // the clear colour for longer than the whole settle (the Kalimdor scenes do).
                    let waits = scenarios::water_scene_names().any(|s| s == ctx.name);
                    if flat && waits && !capped {
                        if !ctx.waited_flat {
                            ctx.waited_flat = true;
                            info!(
                                "capture: {} stable frames of one colour {:?}; the world is not \
                                 in yet, building on",
                                watch.stable,
                                &first[..3],
                            );
                        }
                        // The clear colour is no part of the build: its run extends the cap.
                        ctx.flat_credit = (ctx.flat_credit + watch.stable).min(BUILD_CAP_FRAMES);
                        watch.stable = 0;
                    } else if flat && fixture_age.is_none() {
                        // A fixture viewer settles before it arms its subject, so its settle frame
                        // may be the clear colour and its shot is not.
                        error!(
                            "capture: settled image is FLAT — every pixel is {:?} in the \
                             readback's channel order; the scene rendered nothing",
                            &first[..3],
                        );
                        if waits {
                            error!(
                                "capture: FAILED — {} never left its clear colour; exiting \
                                 nonzero",
                                ctx.name
                            );
                            ctx.bailed = true;
                            exit.write(AppExit::error());
                            return;
                        }
                    }
                }
            }
            if watch.stable < stable && !capped {
                Phase::Building(n + 1)
            } else if let Some((rw, rh)) = resize_request().filter(|_| !ctx.resized) {
                // `$WOW_RESIZE`: resize after the UI was built and measured at the boot size, the
                // fullscreen-toggle flow a fresh boot at the target size cannot exercise.
                ctx.resized = true;
                if let Ok(mut w) = windows.single_mut() {
                    w.resolution.set(rw as f32, rh as f32);
                }
                watch.stable = 0;
                info!("capture: resized to {rw}x{rh}, re-settling");
                Phase::Building(0)
            } else if !ctx.quality_switches.is_empty() {
                // `$WOW_WATER_QUALITY_SWITCH`: through the CVar, so the switch takes the player's
                // path from the observer to the materials' pipeline key.
                let tier = ctx.quality_switches.remove(0);
                match cvars.as_deref_mut() {
                    Some(cvars) => {
                        cvars.set("waterQuality", tier.cvar());
                    }
                    None => warn!("capture: no CVar registry, so waterQuality cannot switch"),
                }
                watch.stable = 0;
                info!(
                    "capture: waterQuality {} ({tier:?}), re-settling",
                    tier.cvar()
                );
                Phase::Building(0)
            } else if fixture_age.is_some() {
                if let Some(state) = fx_state.as_deref_mut() {
                    state.armed = true; // the fixture spawns now, its age clock clean
                }
                Phase::FxAging
            } else {
                // Clock released: the sims run exactly `age_frames()` steps, probe or capture.
                info!(
                    "capture: image settled after {} frames, aging {}",
                    n + 1,
                    age_frames(ctx.name)
                );
                Phase::Aging(0)
            }
        }
        Phase::FxAging => {
            let aged = match (fixture_age, &fx_state) {
                (Some(age), Some(state)) => state
                    .attached_at
                    .is_some_and(|t0| game_time.elapsed_secs() - t0 >= age),
                _ => true,
            };
            if aged {
                let path = ctx.shot_path();
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(path.clone()));
                if ctx.burst > 0 {
                    info!(
                        "capture: effect aged, burst of {} shots {} steps apart; shot 0 at {:.6} \
                         s, writing {path}",
                        ctx.burst,
                        ctx.burst_every,
                        shot_clock(&game_time, fx_state.as_deref()),
                    );
                } else {
                    info!("capture: effect aged, writing {path}");
                }
                Phase::Saving {
                    frames: 0,
                    seen: false,
                }
            } else {
                Phase::FxAging
            }
        }
        Phase::Aging(n) => {
            // No restart on late arrivals: the image already stopped changing before release.
            if n + 1 < age_frames(ctx.name) {
                Phase::Aging(n + 1)
            } else if ctx.probe_frames > 0 {
                // Hand the clock back to real time before the first sample, or `Probing` would
                // read the manual [`CAPTURE_FRAME_DT`] and report a flawless 16.67 ms.
                ctx.frozen_clock = false;
                *time_strategy = TimeUpdateStrategy::Automatic;
                // Re-anchor `Time<Real>` to now: the manual strategy left `last_update` behind by
                // the frozen phase's wall time (~8 s), which the first automatic tick would bill
                // as one hitch and a quarter-second sim step. Called from `Last`, so the delta it
                // writes is overwritten by the next `time_system` before anything reads it.
                time.update_with_instant(Instant::now());
                // Uncap presentation to measure frame cost; `$WOW_PROBE_VSYNC=1` keeps vsync and
                // measures the present ceiling itself.
                let keep_vsync = std::env::var("WOW_PROBE_VSYNC").as_deref() == Ok("1");
                if !keep_vsync {
                    if let Ok(mut w) = windows.single_mut() {
                        w.present_mode = probe_uncap_mode();
                    }
                }
                // An instrumented run opens unfocused (`benilla_world::bgwin`), and bevy's default
                // `WinitSettings::game()` updates an unfocused window at 60 Hz whatever the present
                // mode: the probe runs the loop continuously, so the frame rate is the frame cost.
                commands.insert_resource(bevy::winit::WinitSettings::continuous());
                info!(
                    "probe: scene aged {} frames; vsync {}, warming {PROBE_WARMUP_FRAMES} frames",
                    age_frames(ctx.name),
                    if keep_vsync { "KEPT ON" } else { "off" }
                );
                Phase::ProbeWarmup(0)
            } else {
                let path = ctx.shot_path();
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(path.clone()));
                if ctx.burst > 0 {
                    if let Some(scenario) = ctx.scenario {
                        if let Err(e) = write_poses(&ctx, &scenario) {
                            error!(
                                "capture: cannot write the burst's poses beside {}: {e}",
                                ctx.out
                            );
                        }
                    }
                    let pan = ctx.pan.map_or(String::new(), |[yaw, pitch]| {
                        format!(", panning ±{yaw}° heading and ±{pitch}° elevation")
                    });
                    info!(
                        "capture: scene aged, burst of {} shots {} steps apart{pan}; shot 0 at \
                         {:.6} s, writing {path}",
                        ctx.burst,
                        ctx.burst_every,
                        game_time.elapsed_secs_f64(),
                    );
                } else {
                    info!("capture: scene aged, writing {path}");
                }
                Phase::Saving {
                    frames: 0,
                    seen: false,
                }
            }
        }
        Phase::Stepping(n) => {
            if n + 1 < ctx.burst_every {
                Phase::Stepping(n + 1)
            } else {
                let path = ctx.shot_path();
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(path.clone()));
                info!(
                    "capture: burst shot {} at {:.6} s, writing {path}",
                    ctx.shot,
                    shot_clock(&game_time, fx_state.as_deref()),
                );
                Phase::Saving {
                    frames: 0,
                    seen: false,
                }
            }
        }
        Phase::ProbeWarmup(n) => {
            if n + 1 >= PROBE_WARMUP_FRAMES {
                ctx.probe_cpu_start = crate::perf::process_cpu_secs();
                Phase::Probing(0)
            } else {
                Phase::ProbeWarmup(n + 1)
            }
        }
        Phase::Probing(n) => {
            let ms = time.delta_secs() * 1000.0;
            ctx.probe_samples.push(ms);
            // The meter reads zero until its first delta lands; a zero is "not yet" and is dropped.
            if let Some(gpu) = gpu.as_ref() {
                let ns = gpu.0.load(std::sync::atomic::Ordering::Relaxed);
                if ns > 0 {
                    ctx.gpu_samples.push(ns as f32 / 1.0e6);
                }
            }
            // `==`, not `>=`: `AppExit` takes a frame or two to drain, and `>=` would print twice.
            if n + 1 == ctx.probe_frames {
                let mut v = ctx.probe_samples.clone();
                v.sort_by(f32::total_cmp);
                let at = |q: f32| v[(((v.len() - 1) as f32) * q).round() as usize];
                let mean = v.iter().sum::<f32>() / v.len() as f32;
                let (emitters, active, live) = census
                    .particles
                    .iter()
                    .fold((0usize, 0usize, 0usize), |(e, a, l), p| {
                        (e + 1, a + usize::from(p.live() > 0), l + p.live())
                    });
                let px = windows
                    .single()
                    .map(|w| (w.physical_width(), w.physical_height()))
                    .unwrap_or((0, 0));
                // The world's render size, which need not match the window `px`: a supersample
                // would otherwise be priced as a native frame.
                let world_px = backdrop.map_or(String::new(), |b| {
                    let s = b.render_size();
                    format!(" world_px={}x{}", s.x, s.y)
                });
                // Submeshes, how many survived the cull, and the entity count.
                let (submeshes, drawn) = census.parts.iter().fold((0usize, 0usize), |(n, d), v| {
                    (n + 1, d + usize::from(v.get()))
                });
                let entity_count = census.entities.iter().len();
                // CPU per frame across every thread, the load-robust metric.
                let cpu = match (ctx.probe_cpu_start, crate::perf::process_cpu_secs()) {
                    (Some(t0), Some(t1)) => {
                        let per_frame_ms = (t1 - t0) * 1000.0 / v.len() as f64;
                        format!(
                            " cpu_ms={per_frame_ms:.2} cpu_pct={:.0}",
                            per_frame_ms / mean as f64 * 100.0
                        )
                    }
                    _ => String::new(),
                };
                // The present mode measured under, so a silently railed uncap is diagnosable.
                let present = windows
                    .single()
                    .map(|w| format!(" present={:?}", w.present_mode))
                    .unwrap_or_default();
                // The GPU meter's percentiles over the window (`$WOW_GPU_MS=1`), empty when it is off.
                let gpu_line = if ctx.gpu_samples.is_empty() {
                    String::new()
                } else {
                    let mut g = ctx.gpu_samples.clone();
                    g.sort_by(f32::total_cmp);
                    let gat = |q: f32| g[(((g.len() - 1) as f32) * q).round() as usize];
                    let gmean = g.iter().sum::<f32>() / g.len() as f32;
                    format!(
                        " gpu_n={} gpu_mean={gmean:.2} gpu_p50={:.2} gpu_p95={:.2} gpu_p99={:.2} gpu_max={:.2}",
                        g.len(),
                        gat(0.50),
                        gat(0.95),
                        gat(0.99),
                        g[g.len() - 1],
                    )
                };
                // One greppable line on stdout, clear of log filtering.
                println!(
                    "FPS_PROBE scenario={} frames={} mean_ms={mean:.2} p50_ms={:.2} p95_ms={:.2} p99_ms={:.2} max_ms={:.2} fps={:.1} emitters={emitters} active={active} particles={live} submeshes={submeshes} drawn={drawn} entities={entity_count} px={}x{}{world_px}{cpu}{present}{gpu_line}",
                    ctx.name,
                    v.len(),
                    at(0.50),
                    at(0.95),
                    at(0.99),
                    v[v.len() - 1],
                    1000.0 / mean,
                    px.0,
                    px.1,
                );
                exit.write(AppExit::Success);
            }
            Phase::Probing(n + 1)
        }
        Phase::Saving { frames, seen } => {
            let busy = !capturing.is_empty();
            let seen = seen || busy;
            if (seen && !busy) || frames + 1 >= SAVE_TIMEOUT_FRAMES {
                if ctx.shot + 1 < ctx.burst {
                    ctx.shot += 1;
                    Phase::Stepping(0)
                } else {
                    Phase::Done(0)
                }
            } else {
                Phase::Saving {
                    frames: frames + 1,
                    seen,
                }
            }
        }
        Phase::Done(n) => {
            if n + 1 >= EXIT_GRACE_FRAMES {
                // Exit on what is on disk: a save that timed out without a file is a failed
                // capture, so `scripts/visual.sh`'s `set -e` stops the sweep at it. A burst needs
                // every shot, since a gap would shift every rate read across the series.
                let missing: Vec<String> = ctx
                    .shot_paths()
                    .into_iter()
                    .filter(|p| !Path::new(p).is_file())
                    .collect();
                if missing.is_empty() {
                    info!("capture: saved {}, exiting", ctx.shot_path());
                    exit.write(AppExit::Success);
                } else {
                    error!(
                        "capture: FAILED — no file at {} after the save window; exiting nonzero",
                        missing.join(", ")
                    );
                    exit.write(AppExit::error());
                }
                Phase::Done(n)
            } else {
                Phase::Done(n + 1)
            }
        }
    };
    // The clock runs only once the image has stopped changing ([`hold_clock`]): held while
    // building and while saving, running while aging and between burst shots.
    if ctx.frozen_clock {
        let held = match ctx.phase {
            Phase::Building(_) => true,
            // Pipelined rendering may serve the screenshot a frame or two after the request; a
            // running clock would move every particle pool by a step in that gap.
            Phase::Saving { .. } | Phase::Done(_) => true,
            _ => false,
        };
        match (held, clock.is_paused()) {
            (false, true) => clock.unpause(),
            (true, false) => clock.pause(),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
    use bevy::prelude::*;

    /// A capture run's systems initialize in the client's own world, so a parameter that overlaps
    /// the VM turn's call host fails here and not at the first capture's schedule build.
    #[test]
    fn every_capture_system_initializes_in_the_client() {
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _c = EnvGuard::set("WOW_CAPTURE", "ui-trainer");
        let mut app = crate::game_plugins::schedule_tests::headless_client_with(|app| {
            app.add_plugins(super::CapturePlugin);
        });
        let world = app.world_mut();
        let labels: Vec<_> = world
            .resource::<bevy::ecs::schedule::Schedules>()
            .iter()
            .map(|(_, s)| s.label())
            .collect();
        let mut seen = false;
        for label in labels {
            world.schedule_scope(label, |world, schedule| {
                schedule.graph_mut().systems.initialize(world);
                seen |=
                    schedule.graph().systems.iter().any(|(_, system, _)| {
                        system.name().to_string().contains("seed_ui_fixture")
                    });
            });
        }
        assert!(
            seen,
            "the census reads the capture seed among the client's systems"
        );
    }
    /// The combo-points fixture's rogue banks its points on the wolf it selects, so the getter
    /// the dots read answers four through the host, as in game (`0x51a1f3`-`0x51a20d`).
    #[test]
    fn the_combo_fixture_banks_its_points_on_its_target() {
        use bevy::ecs::system::RunSystemOnce;
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _c = EnvGuard::set("WOW_CAPTURE", "ui-combopoints");
        let mut app = crate::game_plugins::schedule_tests::headless_client_with(|app| {
            app.add_plugins(super::CapturePlugin);
        });
        let world = app.world_mut();
        {
            let mut progress = world.resource_mut::<super::WorldLoadProgress>();
            progress.ready = 1;
            progress.total = 1;
        }
        world.insert_non_send_resource(benilla_ui::script::UiScript::new().unwrap());
        world.init_resource::<crate::player::Player>();
        world.init_resource::<crate::loading_screen::LoadingScreen>();
        world.init_resource::<crate::ui_quest::QuestGiver>();
        world.init_resource::<crate::ui_loot::LootState>();
        world.init_resource::<crate::net::HomeBind>();
        world
            .run_system_once(super::fixtures::seed_ui_fixture)
            .unwrap();
        world.flush();
        let read = crate::script_calls::with_hosted_vm(world, |s| {
            s.expect("a VM")
                .eval::<(i64, Option<i64>)>("return GetComboPoints(), UnitExists('target')")
                .unwrap()
        });
        assert_eq!(read, (4, Some(1)));
    }
    /// The character fixture's doll, by number: once its seed's spawns land, the worn slots'
    /// texture getters answer through the host, and the window opened over them shows those
    /// textures on the doll's buttons and the buffed stamina green.
    #[test]
    fn the_char_fixture_fills_the_doll() {
        use bevy::ecs::system::RunSystemOnce;
        let _data = benilla_formats::wow_data_or_skip!();
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _c = EnvGuard::set("WOW_CAPTURE", "ui-char");
        let mut app = crate::game_plugins::schedule_tests::headless_client_with(|app| {
            app.add_plugins(super::CapturePlugin);
        });
        let world = app.world_mut();
        {
            let mut progress = world.resource_mut::<super::WorldLoadProgress>();
            progress.ready = 1;
            progress.total = 1;
        }
        let script = benilla_ui::script::UiScript::new().unwrap();
        for f in crate::ui_script::test_ui::CHARACTER_UI {
            crate::ui_script::test_ui::load_ui_strict(&script, f);
        }
        world.insert_non_send_resource(script);
        world.init_resource::<crate::player::Player>();
        world.init_resource::<crate::loading_screen::LoadingScreen>();
        world.init_resource::<crate::ui_quest::QuestGiver>();
        world.init_resource::<crate::ui_loot::LootState>();
        world.init_resource::<crate::net::HomeBind>();
        // The level line's race and class names.
        let tables = crate::ui_unit::TestNames::english();
        world.insert_resource(crate::chr_classes::ChrRaceTable(tables.races));
        world.insert_resource(crate::chr_classes::ChrClassTable(tables.classes));
        // The fixture's displays (sword, shield, hearthstone), each with its own icon.
        let icon = |name: &str| benilla_formats::ItemDisplay {
            icon: Some(name.into()),
            ..Default::default()
        };
        world.insert_resource(crate::entities::ItemDisplays::icons_for_tests(
            benilla_formats::ItemDisplayCatalog::from_displays(std::collections::HashMap::from([
                (1542, icon("SWORD")),
                (18730, icon("SHIELD")),
                (6418, icon("STONE")),
            ])),
        ));
        world
            .run_system_once(super::fixtures::seed_ui_fixture)
            .unwrap();
        world.flush();
        // Chest, main hand, ranged and ammo: the getters, then the buttons the window drew.
        let read = crate::script_calls::with_hosted_vm(world, |s| {
            s.expect("a VM")
                .eval::<Vec<String>>(
                    r#"local t = {}
                    for i, slot in ipairs({5, 16, 18, 0}) do
                        t[i] = GetInventoryItemTexture("player", slot) or "none"
                    end
                    for i, name in ipairs({"Chest", "MainHand", "Ranged", "Ammo"}) do
                        local tex = getglobal("Character" .. name .. "SlotIconTexture")
                        t[4 + i] = tex:GetTexture() or "none"
                    end
                    t[9] = CharacterStatFrame3StatText:GetText() or "none"
                    t[10] = CharacterFrame:IsVisible() and "shown" or "hidden"
                    return t"#,
                )
                .unwrap()
        });
        let errors = world
            .non_send_resource::<benilla_ui::script::UiScript>()
            .errors();
        assert!(errors.is_empty(), "the open runs clean: {errors:?}");
        assert_eq!(read[9], "shown");
        let icons = ["SHIELD", "SWORD", "STONE", "STONE"];
        let shown = |t: &str, icon: &str| t.to_ascii_uppercase().ends_with(icon);
        for (i, icon) in icons.iter().enumerate() {
            assert!(shown(&read[i], icon), "the getter, slot {i}: {read:?}");
            assert!(shown(&read[4 + i], icon), "the button, slot {i}: {read:?}");
        }
        assert!(
            read[8].starts_with("|cff20ff20"),
            "the stamina line: {read:?}"
        );
    }

    /// The vendor and trainer fixtures open through the lists' own opens, so each queues its
    /// window's `MERCHANT_SHOW` or `TRAINER_SHOW` as a list does.
    #[test]
    fn the_window_fixtures_queue_their_show() {
        use bevy::ecs::system::RunSystemOnce;
        for scenario in ["ui-merchant", "ui-trainer"] {
            let _l = ENV_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _c = EnvGuard::set("WOW_CAPTURE", scenario);
            let mut app = crate::game_plugins::schedule_tests::headless_client_with(|app| {
                app.add_plugins(super::CapturePlugin);
            });
            let world = app.world_mut();
            {
                let mut progress = world.resource_mut::<super::WorldLoadProgress>();
                progress.ready = 1;
                progress.total = 1;
            }
            world.insert_non_send_resource(benilla_ui::script::UiScript::new().unwrap());
            world.init_resource::<crate::player::Player>();
            world.init_resource::<crate::loading_screen::LoadingScreen>();
            world.init_resource::<crate::ui_bank::BankOpen>();
            world.init_resource::<crate::ui_quest::QuestGiver>();
            world.init_resource::<crate::ui_loot::LootState>();
            world.init_resource::<crate::net::HomeBind>();
            world
                .run_system_once(super::fixtures::seed_ui_fixture)
                .unwrap();
            world.flush();
            let shows = world
                .resource::<crate::ui_merchant::MerchantLane>()
                .stamped()
                .len()
                + world
                    .resource::<crate::ui_trainer::TrainerLane>()
                    .stamped()
                    .len();
            assert_eq!(shows, 1, "{scenario}");
        }
    }
    /// A golden water scene settles on its own stable run and runs its own age; every other scene
    /// takes the defaults.
    #[test]
    fn a_golden_water_scene_carries_its_own_settle_and_age() {
        use super::*;
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _s = EnvGuard::unset("WOW_CAPTURE_STABLE");
        let _a = EnvGuard::unset("WOW_CAPTURE_AGE");
        for name in scenarios::water_scene_names() {
            assert_eq!(stable_frames(name), WATER_STABLE_FRAMES, "{name}");
            assert_eq!(age_frames(name), WATER_AGE_FRAMES, "{name}");
        }
        assert_eq!(stable_frames("water-noon"), STABLE_FRAMES);
        assert_eq!(age_frames("water-noon"), AGE_FRAMES);
        let _s = EnvGuard::set("WOW_CAPTURE_STABLE", "45");
        let _a = EnvGuard::set("WOW_CAPTURE_AGE", "150");
        assert_eq!(
            stable_frames("water-loch-modan"),
            45,
            "the env var still wins"
        );
        assert_eq!(
            age_frames("water-loch-modan"),
            150,
            "the env var still wins"
        );
    }

    /// A pan turns the look-at point about the eye from −pan to +pan over the burst, at the
    /// scene's distance; the middle shot of an odd burst is the scene's own view.
    #[test]
    fn a_pan_sweeps_the_view_either_side_of_the_scene() {
        let eye = [10.0, 20.0, 5.0];
        let look = [110.0, 20.0, 5.0]; // heading 0, level, 100 yd
        let heading = |l: [f32; 3]| (l[1] - eye[1]).atan2(l[0] - eye[0]).to_degrees();
        let elev = |l: [f32; 3]| {
            let d = Vec3::from(l) - Vec3::from(eye);
            (d.z / d.length()).asin().to_degrees()
        };
        let dist = |l: [f32; 3]| (Vec3::from(l) - Vec3::from(eye)).length();
        assert_eq!(pan_pose(eye, look, None, 3, 9), look, "no pan");
        assert_eq!(
            pan_pose(eye, look, Some([20.0, 10.0]), 0, 1),
            look,
            "a still"
        );
        let pan = Some([20.0, 10.0]);
        let first = pan_pose(eye, look, pan, 0, 9);
        let mid = pan_pose(eye, look, pan, 4, 9);
        let last = pan_pose(eye, look, pan, 8, 9);
        assert!((heading(first) + 20.0).abs() < 1e-3, "{}", heading(first));
        assert!((elev(first) + 10.0).abs() < 1e-3, "{}", elev(first));
        assert!((heading(last) - 20.0).abs() < 1e-3, "{}", heading(last));
        assert!((elev(last) - 10.0).abs() < 1e-3, "{}", elev(last));
        assert!((Vec3::from(mid) - Vec3::from(look)).length() < 1e-3);
        for l in [first, mid, last] {
            assert!((dist(l) - 100.0).abs() < 1e-2, "{}", dist(l));
        }
        // A quarter of the way along: a quarter of the sweep, −20° to −10°.
        let q = pan_pose(eye, look, pan, 2, 9);
        assert!((heading(q) + 10.0).abs() < 1e-3, "{}", heading(q));
    }

    /// A run removes the files an earlier capture to its path left: a still its own shot, a burst
    /// every numbered shot of its stem, a longer burst's included, and its poses. Neither touches
    /// the other's files nor another stem's.
    #[test]
    fn a_capture_starts_without_an_earlier_runs_shots() {
        let dir =
            std::env::temp_dir().join(format!("benilla-capture-clear-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let files = [
            "s.png",
            "s-000.png",
            "s-059.png",
            "s-1200.png",
            "s.poses.csv",
            "s-ref.png",
            "s@fresnel-000.png",
            "s-00.png",
            "s-000.txt",
            "t-000.png",
        ];
        let fill = || {
            for f in files {
                std::fs::write(dir.join(f), b"x").unwrap();
            }
        };
        let left = || {
            let mut v: Vec<String> = std::fs::read_dir(&dir)
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            v.sort();
            v
        };
        let out = dir.join("s.png").to_string_lossy().into_owned();
        fill();
        clear_old_shots(&out, 3).unwrap();
        assert_eq!(
            left(),
            [
                "s-00.png",
                "s-000.txt",
                "s-ref.png",
                "s.png",
                "s@fresnel-000.png",
                "t-000.png"
            ]
        );
        fill();
        clear_old_shots(&out, 0).unwrap();
        assert!(!dir.join("s.png").exists());
        assert_eq!(
            left().len(),
            files.len() - 1,
            "a still removes its shot alone"
        );
        clear_old_shots(&out, 0).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The clear colour's stable runs extend a golden water scene's build cap, not its build.
    #[test]
    fn the_clear_colours_wait_does_not_count_against_the_build() {
        use crate::local_state::test_env::{EnvGuard, ENV_LOCK};
        let _l = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _s = EnvGuard::unset("WOW_CAPTURE_STABLE");
        let base = build_cap_frames("water-loch-modan", 0);
        assert_eq!(base, BUILD_CAP_FRAMES + WATER_STABLE_FRAMES);
        assert_eq!(build_cap_frames("water-loch-modan", 600), base + 600);
    }

    /// Burst shots sit beside the single shot's path, numbered from 000, keeping its extension.
    #[test]
    fn burst_shots_are_numbered_beside_the_output() {
        assert_eq!(
            super::burst_path("/data/scratch/golden/water-canal.png", 0),
            "/data/scratch/golden/water-canal-000.png"
        );
        assert_eq!(super::burst_path("shot.png", 7), "shot-007.png");
        assert_eq!(
            super::burst_path("target/visual/a.b.png", 12),
            "target/visual/a.b-012.png"
        );
    }
}
