//! The dispersive surface's **wave trains** (`liquid.wgsl`'s `water_surface_s2`; WAV-1, WAV-3,
//! WAV-5): a fixed set of directional sinusoids, each of its own wavelength travelling along its
//! own direction at the deep-water phase speed √(g/k), spread about the wind. Their amplitudes
//! follow a Phillips spectrum under the live wind with a Pierson–Moskowitz cutoff, and a swell
//! rides the longest bands. The CPU reduces each train's phase by its own period on the water
//! clock, so nothing pops at the hour, and publishes the set as the preview block's `trains` rows
//! ([`PV_TRAINS`]), two per train; the shader sums them per pixel and filters each by the pixel's
//! footprint, so the set is built from a seed and no map or texture enters.

use bevy::math::DVec2;
use bevy::prelude::*;

use benilla_assets::preview as lever;
use benilla_world::liquid::{S2_BANDS, S2_TRAINS};

use super::clock::cycle;
use super::preview::WIND_PREVAILING;

/// Gravity in yards a second squared (9.81 m/s²).
const G: f64 = 10.73;

/// The slope variance one octave of the wind sea carries where the spectrum is saturated. Phillips'
/// α = 0.0081 gives (α/2)·ln 2 = 0.0028; the comic look reads the waves at about half as much
/// again, and the kind gains (`WG_KIND_*`) take the seas and ponds up and down from here.
const OCTAVE_VAR: f64 = 0.0042;

/// The Pierson–Moskowitz cutoff: a train of wavenumber `k` keeps `exp(−(k_c / k)²)` of the
/// saturated spectrum, `k_c = PM_KC · g / U²`, so the longest waves a wind raises grow with its
/// speed squared (a 4 m/s breeze peaks near 13 yd, 8 m/s near 45 yd).
const PM_KC: f64 = 0.86;

/// One band of the spectrum: the longest wavelength in it (yards; it spans one octave down), its
/// trains, their half-spread about the prevailing wind (radians), the directional sharpness `p`
/// the live wind weighs them by (`cos^(2p)(Δ/2)`), the swell's rms slope over the band, Gerstner's
/// Q on its orbital displacement and the Stokes second harmonic's share of its slope.
struct Band {
    lambda: f64,
    trains: usize,
    spread: f64,
    sharp: f64,
    swell: f64,
    q: f64,
    stokes: f64,
}

/// Nine octaves from 48 yd down to 0.19 yd: the three longest carry the swell, the long ones are
/// narrow about the wind and steep-crested, the short ones spread round the compass.
const BANDS: [Band; 9] = [
    Band {
        lambda: 48.0,
        trains: 3,
        spread: 0.50,
        sharp: 4.0,
        swell: 0.050,
        q: 0.8,
        stokes: 0.22,
    },
    Band {
        lambda: 24.0,
        trains: 3,
        spread: 0.55,
        sharp: 4.0,
        swell: 0.045,
        q: 0.8,
        stokes: 0.22,
    },
    Band {
        lambda: 12.0,
        trains: 4,
        spread: 0.60,
        sharp: 3.0,
        swell: 0.035,
        q: 0.8,
        stokes: 0.20,
    },
    Band {
        lambda: 6.0,
        trains: 5,
        spread: 0.70,
        sharp: 3.0,
        swell: 0.0,
        q: 0.7,
        stokes: 0.18,
    },
    Band {
        lambda: 3.0,
        trains: 5,
        spread: 0.85,
        sharp: 2.0,
        swell: 0.0,
        q: 0.6,
        stokes: 0.15,
    },
    Band {
        lambda: 1.5,
        trains: 5,
        spread: 1.00,
        sharp: 2.0,
        swell: 0.0,
        q: 0.5,
        stokes: 0.12,
    },
    Band {
        lambda: 0.75,
        trains: 5,
        spread: 1.20,
        sharp: 1.0,
        swell: 0.0,
        q: 0.0,
        stokes: 0.08,
    },
    Band {
        lambda: 0.375,
        trains: 5,
        spread: 1.40,
        sharp: 1.0,
        swell: 0.0,
        q: 0.0,
        stokes: 0.0,
    },
    Band {
        lambda: 0.19,
        trains: 5,
        spread: 1.57,
        sharp: 0.5,
        swell: 0.0,
        q: 0.0,
        stokes: 0.0,
    },
];

/// The floats a train takes in the block (three `vec4` rows), where the bands' rows start, the
/// floats a band takes (three rows), and the whole set's.
const TRAIN_FLOATS: usize = 12;
const BAND_FLOATS_AT: usize = TRAIN_FLOATS * S2_TRAINS;
const BAND_FLOATS: usize = 12;
pub(crate) const FLOATS: usize = BAND_FLOATS_AT + BAND_FLOATS * S2_BANDS;

/// One train, fixed for the run.
struct Train {
    /// Its band.
    band: usize,
    /// Its wavenumber, radians a yard.
    k: f64,
    /// Its to-direction, unit, world XZ.
    dir: DVec2,
    /// Its turn off the prevailing wind, radians.
    delta: f64,
    /// Its own phase at the clock's zero, cycles.
    phase0: f64,
}

/// A hash in [0, 1) of `i` (splitmix64's finaliser).
fn hash(i: u64) -> f64 {
    let mut z = i
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678_9ABC_DEF1);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// The trains, built once: in each band the directions are stratified over the spread and
/// jittered, the wavelengths stratified in log over the octave and jittered, and the two shuffled
/// against each other, so neither a lattice nor a fan shows.
fn trains() -> &'static [Train] {
    static TRAINS: std::sync::OnceLock<Vec<Train>> = std::sync::OnceLock::new();
    TRAINS.get_or_init(|| {
        let a0 = f64::from(WIND_PREVAILING.y).atan2(f64::from(WIND_PREVAILING.x));
        let mut out = Vec::with_capacity(S2_TRAINS);
        for (b, band) in BANDS.iter().enumerate() {
            let n = band.trains;
            let seed = (b as u64) << 8;
            // The wavelength slots, in a hashed order.
            let mut slots: Vec<usize> = (0..n).collect();
            slots.sort_by(|&x, &y| {
                hash(seed + 64 + x as u64)
                    .partial_cmp(&hash(seed + 64 + y as u64))
                    .unwrap()
            });
            for (j, &slot) in slots.iter().enumerate() {
                let cell = 2.0 * band.spread / n as f64;
                let delta =
                    -band.spread + cell * (j as f64 + 0.5 + 0.8 * (hash(seed + j as u64) - 0.5));
                let u = (slot as f64 + 0.5 + 0.8 * (hash(seed + 32 + j as u64) - 0.5)) / n as f64;
                let lambda = band.lambda * 2f64.powf(-u);
                let a = a0 + delta;
                out.push(Train {
                    band: b,
                    k: std::f64::consts::TAU / lambda,
                    dir: DVec2::new(a.cos(), a.sin()),
                    delta,
                    phase0: hash(seed + 96 + j as u64),
                });
            }
        }
        assert_eq!(out.len(), S2_TRAINS, "the bands' trains are S2_TRAINS");
        out
    })
}

/// `cos^(2p)(Δ/2)`: the share of a band's energy a train `delta` radians off the wind takes.
fn spread_weight(delta: f64, sharp: f64) -> f64 {
    (0.5 * delta).cos().abs().powf(2.0 * sharp)
}

/// The trains' and bands' rows at `secs` of water time under the live wind (`dir` unit, `speed`
/// yd/s). Per train, three rows: (k·d / 2π in cycles a yard, the phase in cycles, the wind sea's
/// slope amplitude a·k), (the swell's slope amplitude, Q·λ/2π of the orbital displacement, the
/// Stokes share, –), (the direction, 1/λ², 1 + St²). Per band, three rows: its slope variance
/// sums at unit gains over its trains, with the Stokes share in them, (Σ wind², Σ 2·wind·swell,
/// Σ swell², the kind gain's share — positive the long gain's, negative the capillary gain's,
/// zero the short one's), (its longest wavelength squared, its first train, its trains, –), and
/// its direction tensor (xx, zz, xz, –) of trace 1, by which a band the pixel cannot resolve at all
/// joins the roughness without a visit to its trains.
pub(crate) fn rows(secs: f64, dir: Vec2, speed: f32) -> [f32; FLOATS] {
    static LEVERS: std::sync::OnceLock<(f64, f64)> = std::sync::OnceLock::new();
    let (gain, swell_gain) = *LEVERS.get_or_init(|| {
        (
            f64::from(lever::number("SURFACE2_GAIN", 1.0, 0.0, 4.0)),
            f64::from(lever::number("SURFACE2_SWELL", 1.0, 0.0, 4.0)),
        )
    });
    let trains = trains();
    let wind = DVec2::new(f64::from(dir.x), f64::from(dir.y)).normalize_or(DVec2::new(0.6, 0.8));
    let u = f64::from(speed).max(0.5);
    let kc = PM_KC * G / (u * u);
    // Each band's weights under the live wind and about the prevailing one, to normalise by.
    let mut sum_live = [0.0f64; BANDS.len()];
    let mut sum_fixed = [0.0f64; BANDS.len()];
    for t in trains {
        let band = &BANDS[t.band];
        let live = wind.y.atan2(wind.x) - t.dir.y.atan2(t.dir.x);
        sum_live[t.band] += spread_weight(live, band.sharp);
        sum_fixed[t.band] += spread_weight(t.delta, band.sharp);
    }
    let mut out = [0.0f32; FLOATS];
    // Per band: the variance sums, the share's mean, the tensor, the first train.
    let mut sums = [[0.0f64; 3]; BANDS.len()];
    let mut share = [0.0f64; BANDS.len()];
    let mut tensor = [[0.0f64; 3]; BANDS.len()];
    let mut first = [usize::MAX; BANDS.len()];
    for (i, t) in trains.iter().enumerate() {
        let band = &BANDS[t.band];
        let lambda = std::f64::consts::TAU / t.k;
        let period = std::f64::consts::TAU / (G * t.k).sqrt();
        let phase = cycle(secs / period + t.phase0);
        let live = wind.y.atan2(wind.x) - t.dir.y.atan2(t.dir.x);
        let w_live = spread_weight(live, band.sharp) / sum_live[t.band].max(1e-9);
        let w_fixed = spread_weight(t.delta, band.sharp) / sum_fixed[t.band].max(1e-9);
        let cut = (-(kc / t.k).powi(2)).exp();
        let amp_wind = (2.0 * OCTAVE_VAR * cut * w_live).sqrt() * gain;
        let amp_swell = band.swell * (2.0 * w_fixed).sqrt() * swell_gain * gain;
        let long = smoothstep(1.5, 8.0, lambda);
        let cap = 1.0 - smoothstep(0.2, 0.6, lambda);
        let st2 = 1.0 + band.stokes * band.stokes;
        let kd = t.dir * (t.k / std::f64::consts::TAU);
        let row = &mut out[TRAIN_FLOATS * i..TRAIN_FLOATS * (i + 1)];
        row[0] = kd.x as f32;
        row[1] = kd.y as f32;
        row[2] = phase;
        row[3] = amp_wind as f32;
        row[4] = amp_swell as f32;
        row[5] = (band.q * lambda / std::f64::consts::TAU) as f32;
        row[6] = band.stokes as f32;
        row[8] = t.dir.x as f32;
        row[9] = t.dir.y as f32;
        row[10] = (1.0 / (lambda * lambda)) as f32;
        row[11] = st2 as f32;
        let b = t.band;
        sums[b][0] += amp_wind * amp_wind * st2;
        sums[b][1] += 2.0 * amp_wind * amp_swell * st2;
        sums[b][2] += amp_swell * amp_swell * st2;
        share[b] += (long - cap) / band.trains as f64;
        let w = (amp_wind + amp_swell).powi(2);
        tensor[b][0] += w * t.dir.x * t.dir.x;
        tensor[b][1] += w * t.dir.y * t.dir.y;
        tensor[b][2] += w * t.dir.x * t.dir.y;
        first[b] = first[b].min(i);
    }
    for (b, band) in BANDS.iter().enumerate() {
        let row =
            &mut out[BAND_FLOATS_AT + BAND_FLOATS * b..BAND_FLOATS_AT + BAND_FLOATS * (b + 1)];
        row[0] = sums[b][0] as f32;
        row[1] = sums[b][1] as f32;
        row[2] = sums[b][2] as f32;
        row[3] = share[b] as f32;
        row[4] = (band.lambda * band.lambda) as f32;
        row[5] = first[b] as f32;
        row[6] = band.trains as f32;
        let tr = (tensor[b][0] + tensor[b][1]).max(1e-12);
        row[8] = (tensor[b][0] / tr) as f32;
        row[9] = (tensor[b][1] / tr) as f32;
        row[10] = (tensor[b][2] / tr) as f32;
    }
    out
}

/// Hermite 0..1 over `lo..hi`.
fn smoothstep(lo: f64, hi: f64, x: f64) -> f64 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader sums as many trains as the block holds.
    #[test]
    fn the_shader_sums_the_blocks_trains() {
        let wgsl = include_str!("../../benilla-assets/src/shaders/liquid.wgsl");
        assert!(wgsl.contains(&format!("const S2_TRAINS: i32 = {S2_TRAINS};")));
        assert!(wgsl.contains(&format!("const S2_BANDS: i32 = {S2_BANDS};")));
        assert!(wgsl.contains(&format!("trains: array<vec4<f32>, {}>,", 3 * S2_TRAINS)));
        assert!(wgsl.contains(&format!("bands: array<vec4<f32>, {}>,", 3 * S2_BANDS)));
        assert_eq!(BANDS.len(), S2_BANDS);
    }

    /// Every train is a unit direction of a wavelength within its band, and the set spans the
    /// octaves without a gap.
    #[test]
    fn the_trains_fill_their_bands() {
        for t in trains() {
            let band = &BANDS[t.band];
            let lambda = std::f64::consts::TAU / t.k;
            assert!((t.dir.length() - 1.0).abs() < 1e-9);
            assert!(
                lambda <= band.lambda && lambda > band.lambda / 2.0,
                "{lambda} in {}",
                band.lambda
            );
            assert!(t.delta.abs() <= band.spread);
        }
    }

    /// Across Bevy's 3600 s wrap, and a week in, every train's phase steps by its own rate times
    /// the time step: deep-water dispersion, √(g·k)/2π cycles a second, on the f64 clock.
    #[test]
    fn every_train_steps_at_its_own_speed_across_the_hour() {
        let dir = Vec2::new(0.6, 0.8);
        for (before, after) in [(3599.9, 3600.1), (604_799.9, 604_800.1)] {
            let (a, b) = (rows(before, dir, 4.0), rows(after, dir, 4.0));
            for (i, t) in trains().iter().enumerate() {
                let rate = (G * t.k).sqrt() / std::f64::consts::TAU;
                let want = rate * (after - before);
                let want = want - want.round();
                let got = f64::from(b[TRAIN_FLOATS * i + 2]) - f64::from(a[TRAIN_FLOATS * i + 2]);
                let got = got - got.round();
                assert!(
                    (got - want).abs() < 2e-5,
                    "train {i}: stepped {got}, not {want}"
                );
            }
        }
    }

    /// A band's wind-sea trains together carry the octave's variance under the cutoff, whatever
    /// the wind's turn; a stronger wind brings in the long bands.
    #[test]
    fn the_spectrum_is_phillips_under_a_pm_cutoff() {
        let var_of = |rows: &[f32; FLOATS], band: usize| -> f64 {
            trains()
                .iter()
                .enumerate()
                .filter(|(_, t)| t.band == band)
                .map(|(i, _)| 0.5 * f64::from(rows[TRAIN_FLOATS * i + 3]).powi(2))
                .sum()
        };
        let breeze = rows(10.0, Vec2::new(0.6, 0.8), 4.4);
        let veered = rows(10.0, Vec2::new(0.8, 0.6), 4.4);
        let gale = rows(10.0, Vec2::new(0.6, 0.8), 12.0);
        // The 0.75 yd band is saturated at any wind (the cutoff takes 1.3 % of it at 4.4 yd/s).
        assert!((var_of(&breeze, 6) - OCTAVE_VAR).abs() < 0.03 * OCTAVE_VAR);
        assert!((var_of(&veered, 6) - OCTAVE_VAR).abs() < 0.03 * OCTAVE_VAR);
        // The 48 yd band needs a gale.
        assert!(var_of(&breeze, 0) < 0.02 * OCTAVE_VAR);
        assert!(var_of(&gale, 0) > 0.3 * OCTAVE_VAR);
    }
}
