//! The Improved Water **preview**'s rows of the reflection block (`liquid.wgsl`'s
//! `WaterPreview`, [`PREVIEW_LANES`]): the feature bits and every runtime input the preview's
//! features read, derived each frame from the water clock and the zone weather, never from map
//! data. [`super::reflect::drive_reflection`] copies them whole; with the preview off they stay
//! zero and no pipeline compiles a reader.

use bevy::prelude::*;

use benilla_assets::preview as lever;
use benilla_world::liquid::{
    PREVIEW_LANES, PV_CAP, PV_CLOCKS, PV_DRIFT, PV_GAINS, PV_GUST, PV_LOOKS, PV_LOOKS2, PV_MASK,
    PV_SWELL, PV_TRAINS, PV_WEATHER, PV_WIND,
};
use benilla_world::weather::{storm_blend, WeatherKind, WeatherState};

use super::clock::{cycle, WaterClock};
use super::surface2;

/// The preview block's floats.
const PREVIEW_FLOATS: usize = PREVIEW_LANES.end - PREVIEW_LANES.start;

/// The preview's rows, written by [`drive_preview`], copied by `drive_reflection`.
#[derive(Resource)]
pub(crate) struct PreviewLanes(pub(crate) [f32; PREVIEW_FLOATS]);

impl Default for PreviewLanes {
    fn default() -> Self {
        Self([0.0; PREVIEW_FLOATS])
    }
}

/// The prevailing wind's to-direction in world XZ (`liquid.wgsl`'s `WG_WIND0`): a world constant,
/// not map data. The swell, the gust lattice and the octaves' drifts are built on it.
pub(crate) const WIND_PREVAILING: Vec2 = Vec2::new(0.6, 0.8);

/// The five ripple octaves' drifts under `waves`, map tiles a second: half the phase speed of each
/// octave's feature wavelength, downwind and fanned (waves-glint's `RIPPLE_DRIFT_WG`).
const RIPPLE_DRIFT_WG: [[f64; 2]; 5] = [
    [-0.00781, -0.01452],
    [-0.02712, 0.00037],
    [-0.04924, -0.00885],
    [-0.03401, 0.08501],
    [-0.06354, 0.13237],
];
/// The swell trains' periods, 2π/√(g·k), g = 10.73 yd/s², λ = 23, 13.7, 8.3 yd.
const SWELL_PERIOD_S: [f64; 3] = [3.6699, 2.8324, 2.2046];
/// The gust field at 4.5 yd/s and the slick field at 0.35 yd/s downwind, over their tiles.
const GUST_RATE: f64 = -0.053_571;
const SLICK_RATE: f64 = -0.001_346_2;
/// The capillary octave's drift in its tiles a second.
const CAP_DRIFT: [f64; 2] = [0.58175, -0.24960];
/// The preview clocks' periods, seconds: the river's, the weather's, the wake's.
const CLOCK_PERIOD_S: [f64; 3] = [16.0, 60.0, 64.0];
/// The wind speed that makes a full sea, yd/s.
const SEA_FULL_WIND: f32 = 12.0;

/// The wind at `secs` of water time under the storm blend `storm`: a slow periodic veer about the
/// prevailing direction and a 2-6 yd/s base, stronger in a storm. Each sine's argument is a phase
/// reduced in f64, so nothing pops at the hour.
fn wind_at(secs: f64, storm: f32) -> (Vec2, f32) {
    let s = |period: f64, phase: f32| (std::f32::consts::TAU * cycle(secs / period) + phase).sin();
    let veer = 0.25 * s(1307.0, 0.0) + 0.12 * s(389.0, 2.1);
    let base = 4.0 + 1.3 * s(613.0, 1.3) + 0.7 * s(241.0, 0.4);
    (
        Vec2::from_angle(veer).rotate(WIND_PREVAILING),
        base * (1.0 + 1.5 * storm),
    )
}

/// Fill the rows from the lever, the clock and the weather.
pub(super) fn drive_preview(
    clock: Res<WaterClock>,
    weather: Option<Res<WeatherState>>,
    glint: Res<benilla_world::sun::SunGlintVisibility>,
    mut lanes: ResMut<PreviewLanes>,
    dev: Res<crate::dev_state::DebugState>,
) {
    let secs = clock.secs();
    let at = |lane: usize| lane - PREVIEW_LANES.start;
    let mut row = |lane: usize, v: [f32; 4]| lanes.0[at(lane)..at(lane) + 4].copy_from_slice(&v);
    let mask = lever::mask();
    // The sub-levers, read once: the rough cone's share, a forced sea state (-1: derived) and the
    // rivers' slow tier's gain.
    static SUB: std::sync::OnceLock<(f32, f32, f32)> = std::sync::OnceLock::new();
    let (cone, sea_forced, river_slow) = *SUB.get_or_init(|| {
        (
            lever::number("ROUGH_CONE", 0.2, 0.0, 2.0),
            lever::number("SEA", -1.0, 0.0, 1.0),
            lever::number("RIVER_SLOW", 1.0, 0.0, 1.0),
        )
    });
    // The gains (lights, sparkles under the owner's gloss, river foam; the caustics' follows the
    // debug panel below), the river mist's, and a forced rain and snow (-1: the zone weather's).
    static GAINS: std::sync::OnceLock<([f32; 4], f32, f32, f32)> = std::sync::OnceLock::new();
    let (mut gains, mist_gain, rain_forced, snow_forced) = *GAINS.get_or_init(|| {
        (
            [
                1.0,
                lever::number("LIGHTS_GAIN", 1.0, 0.0, 4.0),
                lever::number("SPARKLES_GAIN", 1.0, 0.0, 4.0)
                    * crate::look::gloss_sparkle(crate::look::gloss()),
                lever::number("RIVER_FOAM_GAIN", 1.0, 0.0, 4.0),
            ],
            lever::number("RIVER_MIST_GAIN", 1.0, 0.0, 4.0),
            lever::number("FORCE_RAIN", -1.0, 0.0, 1.0),
            lever::number("FORCE_SNOW", -1.0, 0.0, 1.0),
        )
    });
    let cone = if lever::on(lever::ROUGH) { cone } else { 0.0 };
    // The slow tier's lane holds 1 − gain, so an unwritten (zero) lane runs it; `w` says whether
    // the dispersive surface draws under `waves`.
    let surface2 = lever::on(lever::WAVES) && dev.water.surface2;
    row(
        PV_MASK,
        [
            mask as f32,
            cone,
            1.0 - river_slow,
            f32::from(u8::from(surface2)),
        ],
    );
    // The caustic net's gain, live from the debug panel ($WOW_PREVIEW_CAUSTICS_GAIN at boot): the
    // one lane the net reads above water and the underwater rows copy below it.
    gains[0] = dev.water.caustics.clamp(0.0, 3.0);
    row(PV_GAINS, gains);
    // The owner's look switches, live from the debug panel, each lane 1 where its switch is off.
    let off = |on: bool| f32::from(u8::from(!on));
    row(
        PV_LOOKS,
        [
            off(dev.water.soft_shore),
            off(dev.water.lively),
            off(dev.water.surface_height),
            off(dev.water.reflect_waves),
        ],
    );
    row(
        PV_LOOKS2,
        [
            off(dev.water.sky_waves),
            off(dev.water.crest_shade),
            0.0,
            0.0,
        ],
    );
    row(
        PV_CLOCKS,
        [
            cycle(secs / CLOCK_PERIOD_S[0]),
            cycle(secs / CLOCK_PERIOD_S[1]),
            cycle(secs / CLOCK_PERIOD_S[2]),
            0.0,
        ],
    );
    let (sky, kind, density) = weather.map_or((0.0, WeatherKind::Fine, 0.0), |w| {
        (w.sky_density, w.effect_kind, w.effect_density)
    });
    let storm = storm_blend(sky);
    let (dir, speed) = wind_at(secs, storm);
    row(PV_WIND, [dir.x, dir.y, speed, storm]);
    let rain = if rain_forced >= 0.0 {
        rain_forced
    } else if kind == WeatherKind::Rain {
        density
    } else {
        0.0
    };
    let snow = if snow_forced >= 0.0 {
        snow_forced
    } else if kind == WeatherKind::Snow {
        density
    } else {
        0.0
    };
    let sea = if sea_forced >= 0.0 {
        sea_forced
    } else {
        (speed / SEA_FULL_WIND).max(storm).clamp(0.0, 1.0)
    };
    row(PV_WEATHER, [rain, snow, sea, mist_gain]);
    row(
        PV_SWELL,
        [
            cycle(secs / SWELL_PERIOD_S[0]),
            cycle(secs / SWELL_PERIOD_S[1]),
            cycle(secs / SWELL_PERIOD_S[2]),
            glint.0,
        ],
    );
    row(
        PV_GUST,
        [cycle(GUST_RATE * secs), 0.0, cycle(SLICK_RATE * secs), 0.0],
    );
    // `z`: the band the dispersive surface thins from at Medium, 1-based (`WOW_PREVIEW_SURFACE2_THIN`;
    // 0 the shader's default).
    static THIN: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    let thin = *THIN.get_or_init(|| lever::number("SURFACE2_THIN", 0.0, 0.0, 10.0).round());
    row(
        PV_CAP,
        [
            cycle(CAP_DRIFT[0] * secs),
            cycle(CAP_DRIFT[1] * secs),
            thin,
            0.0,
        ],
    );
    let d = |i: usize| {
        [
            cycle(RIPPLE_DRIFT_WG[i][0] * secs),
            cycle(RIPPLE_DRIFT_WG[i][1] * secs),
        ]
    };
    let (d0, d1, d2, d3, d4) = (d(0), d(1), d(2), d(3), d(4));
    row(PV_DRIFT, [d0[0], d0[1], d1[0], d1[1]]);
    row(PV_DRIFT + 4, [d2[0], d2[1], d3[0], d3[1]]);
    row(PV_DRIFT + 8, [d4[0], d4[1], 0.0, 0.0]);
    // The dispersive surface's trains, under the same wind.
    if surface2 {
        lanes.0[at(PV_TRAINS)..at(PV_TRAINS) + surface2::FLOATS]
            .copy_from_slice(&surface2::rows(secs, dir, speed));
    }
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<PreviewLanes>();
    if !lever::any() {
        return;
    }
    // Before the block's writer copies the rows; after the clock has stepped (`PreUpdate`).
    app.add_systems(
        PostUpdate,
        drive_preview.before(super::reflect::drive_reflection),
    );
    super::preview_rows::register(app);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader's prevailing wind is this one.
    #[test]
    fn the_prevailing_wind_is_the_shaders() {
        let wgsl = include_str!("../../benilla-assets/src/shaders/liquid.wgsl");
        assert!(wgsl.contains("const WG_WIND0: vec2<f32> = vec2<f32>(0.6, 0.8);"));
        assert_eq!(WIND_PREVAILING, Vec2::new(0.6, 0.8));
    }

    /// The wind stays a unit direction within its speed range, storm or not.
    #[test]
    fn the_wind_stays_in_range() {
        for secs in [0.0, 17.0, 3600.0, 1e6] {
            for storm in [0.0, 1.0] {
                let (dir, speed) = wind_at(secs, storm);
                assert!((dir.length() - 1.0).abs() < 1e-5);
                assert!((1.9..=15.1).contains(&speed), "{speed}");
            }
        }
    }
}
