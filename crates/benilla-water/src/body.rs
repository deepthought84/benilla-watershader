//! **The body's optics and light** for the stylised lane, from the zone's own `Light.dbc` water
//! swatch: how fast each channel is absorbed and how murky the water is ([`kind_optics`], from the
//! clear-noon swatch, so absorption does not follow the clock), and the light the body scatters
//! back ([`body_light`]: the dome's irradiance and the lighting sun, at the terrain's own open-sky
//! luminance). Pure functions; [`body_lanes`] lays them out as the look block's body rows, which
//! `look` writes.

use benilla_formats::LiquidKind;
use benilla_world::lighting::WowLighting;
use bevy::math::{Vec3, Vec4};

/// The body rows' floats: ten `vec4`s, `liquid.wgsl`'s `WaterBody`.
pub const BODY_FLOATS: usize = 40;

/// The water's display gamma, `liquid.wgsl`'s `WATER_GAMMA`: a pure power law.
const WATER_GAMMA: f32 = 2.2;

/// Rec. 709 luminance weights.
const LUMA_709: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Hemispherical mean Fresnel reflectance of water (n = 1.333) for Lambertian sky light.
pub const F_DIFFUSE: f32 = 0.0664;

/// Pure water's absorption per yard: Pope and Fry (1997), the band means over 600-680, 500-570
/// and 440-490 nm, (0.34, 0.049, 0.0105) per metre.
pub const SIGMA_PURE_WATER: [f32; 3] = [0.311, 0.0448, 0.0096];

/// A clear lake's extinction per yard, (0.45, 0.085, 0.065) per metre: pure water plus clear-water
/// scattering. Only its shape is used.
pub const SIGMA_CLEAR_LAKE: [f32; 3] = [0.4115, 0.0777, 0.0594];

/// The steepest spectrum water has, Pope and Fry's red over blue (0.34/0.0105 = 32.4).
pub const SHAPE_RATIO_MAX: f32 = 32.0;

/// The clear lake's grey scattering per yard, 0.26 per metre: the turbidity of the modal swatch.
pub const TURB_REF: f32 = 0.2377;

/// The shallow alpha nearly every `LightParams` row carries (199 of 204 on maps 0 and 1), river
/// then ocean; the deep alpha is 1 everywhere.
pub const MODAL_SHALLOW_ALPHA: [f32; 2] = [0.5, 0.75];

/// The dome law's stops weighed by their share of a horizontal surface's irradiance (the cosine
/// integral over the rings at 90, 16.8, 9.8, 3.7, 1.8 and 0 degrees): sky0 to sky4, then fog.
pub const DOME_IRRADIANCE_W: [f32; 6] =
    [0.391711, 0.554255, 0.039301, 0.012340, 0.002064, 0.000329];

/// `terrainp_s.bls`' shadow factor in gamma, `0.3·S + 0.7` at S = 0: what MCSH shadow leaves of
/// the terrain's light.
const TERRAIN_SHADOW_G: f32 = 0.7;

/// `ocean0_s.bls`' lift, the reference's `(secondary + 0.25)·detail.a`.
pub const SHEEN_LIFT: f32 = 0.25;

/// The owner's knobs for the body: the defaults the look ships with.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyLook {
    /// How far the absorption's shape leans from the swatch's toward the clear lake's (0 to 1).
    pub k_phys: f32,
    /// The turbidity's scale, river then ocean.
    pub turb_scale: [f32; 2],
    /// The swatch ramp's rate against the turbidity.
    pub ramp_rate: f32,
    /// The three bands' push and edge.
    pub band_push: f32,
    pub band_edge: f32,
    /// The share of the illuminant's chroma the body keeps by day.
    pub illum_chroma: f32,
    /// The lift's gain against the reference's [`SHEEN_LIFT`].
    pub lift_k: f32,
    /// The refraction's strength (1 the physical deflection) and its largest offset, as a share
    /// of the view height.
    pub refract: f32,
    pub refract_max_frac: f32,
    /// Whether the probe's baked fog is re-applied at the mirror's distance.
    pub probe_refog: bool,
    /// Where the deep stop sits on the zone's line from swatch row 0 to row 63 (the mid stop at
    /// half of it); the gamma body's `BODY_DEPTH_MAX` in `liquid.wgsl` is the same cap.
    ///
    /// Under 1 on purpose. The 1.12 water palettes were authored for a surface with no reflection
    /// and no glitter on it, and several zones' deep row is nearly black (Stranglethorn's, at Booty
    /// Bay). Lit and fogged, that row reads as tar rather than as deep water. Capping the ramp keeps
    /// deep water inside the zone's own pair of colours, one step back up its own ramp, and never a
    /// wash of grey from outside the palette.
    pub deep_ceiling: f32,
}

impl Default for BodyLook {
    fn default() -> Self {
        Self {
            k_phys: 0.25,
            turb_scale: [1.0; 2],
            ramp_rate: 0.615,
            band_push: 0.5,
            band_edge: 0.22,
            illum_chroma: 0.25,
            lift_k: 1.0,
            refract: 1.0,
            refract_max_frac: 0.01,
            probe_refog: true,
            deep_ceiling: 0.72,
        }
    }
}

/// A kind's optics: absorption per yard per channel and grey turbidity per yard.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KindOptics {
    pub sigma_abs: [f32; 3],
    pub sigma_turb: f32,
}

/// The body's illuminant, linear: the sky's and the sun's parts with the body's chroma (`e_*`)
/// and with the light's own (`e_*_raw`, for foam), at the terrain's open-sky luminance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyLight {
    pub e_sky: [f32; 3],
    pub e_sun: [f32; 3],
    pub e_sky_raw: [f32; 3],
    pub e_sun_raw: [f32; 3],
    /// The sun's share kept in MCSH shadow, so shadowed water darkens as the terrain beside it.
    pub s0: f32,
    /// The scale from the dome-and-sun irradiance to the terrain's luminance.
    pub k_e: f32,
    /// The sun's share of the luminance.
    pub sun_share: f32,
    /// 1 with the celestial sun below the horizon, 0 with it up.
    pub night: f32,
}

/// The water's decode, `liquid.wgsl`'s `water_dec`.
pub fn dec(g: f32) -> f32 {
    g.max(0.0).powf(WATER_GAMMA)
}

fn dec3(c: [f32; 3]) -> [f32; 3] {
    c.map(dec)
}

/// Rec. 709 luminance.
pub fn luma(c: [f32; 3]) -> f32 {
    c[0] * LUMA_709[0] + c[1] * LUMA_709[1] + c[2] * LUMA_709[2]
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `exp(mix(ln a, ln b, t))` per channel.
fn geometric_mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    std::array::from_fn(|c| (a[c].ln() * (1.0 - t) + b[c].ln() * t).exp())
}

/// The reference's swatch row `i` in bytes/255, `liquid.wgsl`'s `swatch_row`: the integer ramp
/// `c0 + floor(i·(c1 − c0)/64)`, and on the ocean the last row's value × 0.9.
pub fn swatch_row(shallow: [f32; 3], deep: [f32; 3], i: u32, ocean: bool) -> [f32; 3] {
    std::array::from_fn(|c| {
        let (c0, c1) = ((shallow[c] * 255.0).round(), (deep[c] * 255.0).round());
        let row = c0 + (i as f32 * (c1 - c0) / 64.0).floor();
        if ocean && i >= 63 {
            (row * 0.9).floor() / 255.0
        } else {
            row / 255.0
        }
    })
}

/// The deep albedo stop in gamma: `ceiling` of the way from the swatch's first row to its last
/// (without the ocean's ×0.9 tail), on the zone's own line.
pub fn deep_stop(shallow: [f32; 3], deep: [f32; 3], ocean: bool, ceiling: f32) -> [f32; 3] {
    let (r0, r63) = (
        swatch_row(shallow, deep, 0, ocean),
        swatch_row(shallow, deep, 63, false),
    );
    std::array::from_fn(|c| r0[c] + (r63[c] - r0[c]) * ceiling)
}

/// The absorption's shape from the deepest row `d` (linear): its single-scattering inversion
/// `1/d̂`, `d̂` the row over its peak channel floored at `1/`[`SHAPE_RATIO_MAX`], blended toward
/// the clear lake's as the row fades to black between bytes 1 and 6 (an unkeyed sphere leaves a
/// black row).
fn swatch_shape(d: [f32; 3]) -> [f32; 3] {
    let peak = d[0].max(d[1]).max(d[2]);
    let floor = dec(1.0 / 255.0);
    let m = peak.max(floor);
    let shape = d.map(|x| 1.0 / (x / m).max(1.0 / SHAPE_RATIO_MAX));
    geometric_mix(
        SIGMA_CLEAR_LAKE,
        shape,
        smoothstep(floor, dec(6.0 / 255.0), peak),
    )
}

/// `shape` scaled so no channel absorbs less than pure water and one absorbs exactly that.
fn anchored(shape: [f32; 3]) -> [f32; 3] {
    let s = (0..3)
        .map(|c| SIGMA_PURE_WATER[c] / shape[c])
        .fold(0.0, f32::max);
    shape.map(|x| x * s)
}

/// The absorption per yard of water whose deepest row is `d` (linear), its shape leaning
/// `k_phys` toward the clear lake's.
fn absorption(d: [f32; 3], k_phys: f32) -> [f32; 3] {
    anchored(geometric_mix(swatch_shape(d), SIGMA_CLEAR_LAKE, k_phys))
}

/// The clear lake's absorption luminance, the turbidity's hue reference: 0.1113 per yard.
pub fn sbar_clear() -> f32 {
    luma(anchored(SIGMA_CLEAR_LAKE))
}

/// A kind's optics from its swatch (`WowLighting::water_colors_noon`): the absorption from the
/// deepest row the reference draws, and the turbidity from the clear lake's, scaled by how much
/// less transparent the shallow alpha is than its kind's modal one and by how much more the
/// swatch's hue absorbs than the clear lake.
pub fn kind_optics(
    shallow: [f32; 3],
    deep: [f32; 3],
    shallow_alpha: f32,
    ocean: bool,
    look: &BodyLook,
) -> KindOptics {
    let sigma_abs = absorption(dec3(swatch_row(shallow, deep, 63, ocean)), look.k_phys);
    let kind = usize::from(ocean);
    let f_alpha = (1.0 - shallow_alpha.clamp(0.0, 254.0 / 255.0)).ln()
        / (1.0 - MODAL_SHALLOW_ALPHA[kind]).ln();
    let f_hue = luma(sigma_abs) / sbar_clear();
    KindOptics {
        sigma_abs,
        sigma_turb: TURB_REF * look.turb_scale[kind] * f_alpha * f_hue,
    }
}

/// The dome's irradiance on a level surface over π, linear: the cosine-weighted mean of its stops
/// (`dome` as `benilla_world::sky::dome_uniforms` lays it out). The dawn and dusk warp only moves
/// light between the mid rings, so it is left out.
pub fn sky_irradiance(dome: &[Vec4; 7]) -> [f32; 3] {
    let mut e = [0.0; 3];
    for (row, w) in dome.iter().zip(DOME_IRRADIANCE_W) {
        for (c, v) in e.iter_mut().zip(row.to_array()) {
            *c += w * dec(v);
        }
    }
    e
}

/// `e` with its luminance kept and `chroma` of its chroma.
fn bound_chroma(e: [f32; 3], chroma: f32) -> [f32; 3] {
    let y = luma(e);
    e.map(|x| y * (1.0 + chroma * (x / y.max(1e-4) - 1.0)))
}

/// The body's illuminant: the dome's irradiance and the lighting sun's, each through the
/// surface's Fresnel, scaled to the luminance of the terrain's own open-sky light
/// (`clamp(ambient + diffuse·μ_s)`, decoded); the body keeps `illum_chroma` of their chroma by day
/// and all of it with the celestial sun below the horizon.
pub fn body_light(
    ambient: [f32; 3],
    diffuse: [f32; 3],
    sun_dir: Vec3,
    celestial_dir: Vec3,
    dome: &[Vec4; 7],
    look: &BodyLook,
) -> BodyLight {
    let mu_s = (-sun_dir.normalize_or_zero().y).max(0.0);
    let f_in = 0.02 + 0.98 * (1.0 - mu_s).powi(5);
    let e_sky = sky_irradiance(dome).map(|x| x * (1.0 - F_DIFFUSE));
    let e_sun = dec3(diffuse).map(|x| x * mu_s * (1.0 - f_in));
    let lit_open: [f32; 3] =
        std::array::from_fn(|c| (ambient[c] + diffuse[c] * mu_s).clamp(0.0, 1.0));
    let total: [f32; 3] = std::array::from_fn(|c| e_sky[c] + e_sun[c]);
    let k_e = (luma(dec3(lit_open)) / luma(total).max(1e-4)).clamp(0.25, 16.0);
    let night = 1.0 - smoothstep(-0.10, 0.05, celestial_dir.normalize_or_zero().y);
    let chroma = look.illum_chroma + (1.0 - look.illum_chroma) * night;
    let (e_sky_raw, e_sun_raw) = (e_sky.map(|x| x * k_e), e_sun.map(|x| x * k_e));
    let (e_sky, e_sun) = (
        bound_chroma(e_sky_raw, chroma),
        bound_chroma(e_sun_raw, chroma),
    );
    let sun_share = luma(e_sun) / (luma(e_sky) + luma(e_sun)).max(1e-4);
    let s0 = ((dec(TERRAIN_SHADOW_G) - (1.0 - sun_share)) / sun_share.max(1e-4)).max(0.0);
    BodyLight {
        e_sky,
        e_sun,
        e_sky_raw,
        e_sun_raw,
        s0,
        k_e,
        sun_share,
        night,
    }
}

/// The lift `liquid.wgsl`'s `body_lift` adds to an albedo `a_g` (gamma): `lift` scaled by the
/// albedo's hue, its peak channel never above `lift`.
pub fn body_lift(a_g: [f32; 3], lift: f32) -> [f32; 3] {
    let peak = a_g[0].max(a_g[1]).max(a_g[2]);
    a_g.map(|x| lift * x / peak.max(lift.max(1e-4)))
}

/// The body rows (`WaterBody`): each kind's optics from the clear-noon swatch, the illuminant, the
/// look's knobs and the terrain shadow's window `shadow` (centre x, z, half extent, valid).
pub fn body_lanes(
    light: &WowLighting,
    dome: &[Vec4; 7],
    look: &BodyLook,
    shadow: [f32; 4],
) -> [f32; BODY_FLOATS] {
    let optics = |kind: LiquidKind| {
        let (shallow, deep, alpha, _) = light.water_colors_noon(kind);
        let o = kind_optics(shallow, deep, alpha, kind == LiquidKind::Ocean, look);
        let [r, g, b] = o.sigma_abs;
        [r, g, b, o.sigma_turb]
    };
    let e = body_light(
        light.ambient,
        light.diffuse,
        light.sun_dir,
        light.celestial_dir,
        dome,
        look,
    );
    let rgb = |c: [f32; 3], w: f32| [c[0], c[1], c[2], w];
    let rows: [[f32; 4]; 10] = [
        optics(LiquidKind::Still),
        optics(LiquidKind::Ocean),
        rgb(e.e_sky, e.s0),
        rgb(e.e_sun, e.k_e),
        [
            look.ramp_rate,
            look.band_push,
            look.band_edge,
            look.lift_k * SHEEN_LIFT,
        ],
        [
            look.refract,
            look.refract_max_frac,
            f32::from(u8::from(look.probe_refog)),
            0.0,
        ],
        shadow,
        [e.sun_share, look.deep_ceiling, e.night, 0.0],
        rgb(e.e_sky_raw, 0.0),
        rgb(e.e_sun_raw, 0.0),
    ];
    let mut lanes = [0.0; BODY_FLOATS];
    lanes.copy_from_slice(rows.as_flattened());
    lanes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A noon-like day: a blue dome, a warm sun 30° up, the celestial sun near the zenith.
    fn day() -> ([f32; 3], [f32; 3], Vec3, [Vec4; 7]) {
        let g = |r: f32, g: f32, b: f32| Vec4::new(r / 255.0, g / 255.0, b / 255.0, 0.0);
        let dome = [
            g(42.0, 101.0, 172.0),
            g(94.0, 147.0, 203.0),
            g(130.0, 174.0, 219.0),
            g(160.0, 196.0, 225.0),
            g(178.0, 207.0, 229.0),
            g(186.0, 211.0, 230.0),
            Vec4::ZERO,
        ];
        let el = 30f32.to_radians();
        (
            [0.40, 0.40, 0.42],
            [0.80, 0.70, 0.55],
            Vec3::new(el.cos(), -el.sin(), 0.0),
            dome,
        )
    }

    /// The day's light with the celestial sun along `celestial`.
    fn light_toward(celestial: Vec3, look: &BodyLook) -> BodyLight {
        let (ambient, diffuse, sun, dome) = day();
        body_light(ambient, diffuse, sun, celestial, &dome, look)
    }

    fn noon_light() -> BodyLight {
        light_toward(Vec3::new(0.0, 1.0, 0.1), &BodyLook::default())
    }

    /// HSV hue in degrees.
    fn hue(c: [f32; 3]) -> f32 {
        let (max, min) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
        let d = max - min;
        let h = if max == c[0] {
            ((c[1] - c[2]) / d).rem_euclid(6.0)
        } else if max == c[1] {
            (c[2] - c[0]) / d + 2.0
        } else {
            (c[0] - c[1]) / d + 4.0
        };
        60.0 * h
    }

    fn chroma(c: [f32; 3]) -> f32 {
        c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
    }

    /// The weights are the cosine integral of the dome law's rings, piecewise linear in
    /// elevation, and they sum to 1; the rings are `dome_color`'s.
    #[test]
    fn the_dome_weights_are_the_dome_laws_irradiance() {
        let sum: f32 = DOME_IRRADIANCE_W.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "{sum}");
        // fog, sky4, sky3, sky2, sky1, sky0 at their elevations in degrees.
        let rings = [0.0f64, 1.8, 3.7, 9.8, 16.8, 90.0];
        let mut w = [0.0f64; 6];
        const N: usize = 20_000;
        for i in 0..5 {
            let (a, b) = (rings[i], rings[i + 1]);
            for k in 0..N {
                let e0 = a + (b - a) * k as f64 / N as f64;
                let e1 = a + (b - a) * (k + 1) as f64 / N as f64;
                let u = (0.5 * (e0 + e1) - a) / (b - a);
                let dw = e1.to_radians().sin().powi(2) - e0.to_radians().sin().powi(2);
                w[i] += (1.0 - u) * dw;
                w[i + 1] += u * dw;
            }
        }
        // The constant runs sky0 to sky4, then fog: the rings reversed.
        for (stop, want) in w.iter().rev().enumerate() {
            let got = DOME_IRRADIANCE_W[stop];
            assert!(
                (got - *want as f32).abs() < 1e-5,
                "stop {stop}: {got} against {want}"
            );
        }
        let law = include_str!("../../benilla-assets/src/shaders/liquid.wgsl");
        let law = &law[law.find("fn dome_color(").expect("the dome law")..];
        for ring in [
            "elev <= 0.0",
            "elev < 1.8",
            "elev < 3.7",
            "elev < 9.8",
            "elev < 16.8",
            "(90.0 - 16.8)",
        ] {
            assert!(law.contains(ring), "the dome law lost `{ring}`");
        }
    }

    /// A row's absorption is anchored on pure water: no channel absorbs less, one exactly as much.
    #[test]
    fn the_absorption_is_anchored_on_pure_water() {
        let look = BodyLook::default();
        // Bytes from a fixed LCG, black and grey rows besides.
        let mut seed = 0x2545_f491_u32;
        let mut byte = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 24) as f32 / 255.0
        };
        let mut rows = vec![[0.0; 3], [0.02; 3], [0.10, 0.25, 0.30], [0.35, 0.33, 0.10]];
        rows.extend((0..200).map(|_| [byte(), byte(), byte()]));
        for deep in rows {
            for ocean in [false, true] {
                let o = kind_optics([0.3, 0.4, 0.35], deep, 0.5, ocean, &look);
                let ratio = (0..3)
                    .map(|c| o.sigma_abs[c] / SIGMA_PURE_WATER[c])
                    .fold(f32::MAX, f32::min);
                assert!((ratio - 1.0).abs() < 1e-5, "{deep:?}: {:?}", o.sigma_abs);
            }
        }
    }

    /// A blue deep row absorbs red most; a yellow one, blue.
    #[test]
    fn blue_rows_absorb_red_and_yellow_rows_blue() {
        let look = BodyLook::default();
        let blue = kind_optics([0.3, 0.4, 0.4], [0.08, 0.22, 0.30], 0.5, false, &look).sigma_abs;
        assert!(blue[0] > blue[1] && blue[0] > blue[2], "{blue:?}");
        let yellow = kind_optics([0.4, 0.4, 0.2], [0.30, 0.28, 0.05], 0.5, false, &look).sigma_abs;
        assert!(yellow[2] > yellow[0] && yellow[2] > yellow[1], "{yellow:?}");
    }

    /// A black deep row takes the clear lake's shape.
    #[test]
    fn a_black_row_takes_the_clear_lakes_shape() {
        let o = kind_optics([0.2, 0.3, 0.3], [0.0; 3], 0.5, false, &BodyLook::default());
        let want = anchored(SIGMA_CLEAR_LAKE);
        for (got, want) in o.sigma_abs.iter().zip(want) {
            assert!((got / want - 1.0).abs() < 1e-5, "{:?}", o.sigma_abs);
        }
    }

    /// The modal alpha leaves the turbidity at the reference times the hue factor, on both kinds.
    #[test]
    fn the_modal_alpha_scales_the_turbidity_by_one() {
        let look = BodyLook::default();
        for (ocean, alpha) in [(false, 0.5), (true, 0.75)] {
            let o = kind_optics([0.3, 0.4, 0.4], [0.1, 0.2, 0.25], alpha, ocean, &look);
            let want = TURB_REF * luma(o.sigma_abs) / sbar_clear();
            assert!((o.sigma_turb / want - 1.0).abs() < 1e-5, "{o:?}");
        }
        // A 0.85 river is murkier: ln(0.15)/ln(0.5) = 2.74.
        let modal = kind_optics([0.3, 0.4, 0.4], [0.1, 0.2, 0.25], 0.5, false, &look);
        let murky = kind_optics([0.3, 0.4, 0.4], [0.1, 0.2, 0.25], 0.85, false, &look);
        assert!((murky.sigma_turb / modal.sigma_turb - 2.737).abs() < 1e-3);
    }

    /// The clear lake's absorption luminance, the turbidity's hue reference.
    #[test]
    fn the_clear_lakes_absorption_luminance_is_pinned() {
        assert!((sbar_clear() - 0.1113).abs() < 1e-3, "{}", sbar_clear());
    }

    /// The body's light has the terrain's open-sky luminance, whatever its chroma.
    #[test]
    fn the_body_light_has_the_terrains_luminance() {
        let (ambient, diffuse, sun, _) = day();
        let mu = -sun.normalize().y;
        let lit: [f32; 3] = std::array::from_fn(|c| (ambient[c] + diffuse[c] * mu).clamp(0.0, 1.0));
        let e = noon_light();
        assert!(e.k_e > 0.25 && e.k_e < 16.0, "k_E clamped: {}", e.k_e);
        let open: [f32; 3] = std::array::from_fn(|c| e.e_sky[c] + e.e_sun[c]);
        assert!(
            (luma(open) - luma(dec3(lit))).abs() < 1e-5,
            "{open:?} against {lit:?}"
        );
    }

    /// The foam rows keep the light's full chroma at the body's luminance.
    #[test]
    fn the_foam_rows_carry_the_full_chroma_at_the_same_luminance() {
        let e = noon_light();
        for (body, raw) in [(e.e_sky, e.e_sky_raw), (e.e_sun, e.e_sun_raw)] {
            assert!((luma(body) - luma(raw)).abs() < 1e-5, "{body:?} {raw:?}");
            assert!(chroma(raw) > chroma(body), "{body:?} {raw:?}");
        }
        // The raw rows are the light itself, rescaled.
        let (ambient, diffuse, sun, dome) = day();
        let unscaled = body_light(ambient, diffuse, sun, Vec3::Y, &dome, &BodyLook::default());
        let sky = sky_irradiance(&dome);
        for (got, sky) in unscaled.e_sky_raw.iter().zip(sky) {
            assert!((got - sky * (1.0 - F_DIFFUSE) * unscaled.k_e).abs() < 1e-6);
        }
    }

    /// With the celestial sun below the horizon the body keeps the light's whole chroma; at noon,
    /// a quarter.
    #[test]
    fn night_light_keeps_its_blue() {
        let look = BodyLook::default();
        let night = light_toward(Vec3::new(0.95, -0.3, 0.0), &look);
        assert_eq!(night.night, 1.0);
        for c in 0..3 {
            assert!(
                (night.e_sky[c] - night.e_sky_raw[c]).abs() < 1e-6,
                "{night:?}"
            );
            assert!(
                (night.e_sun[c] - night.e_sun_raw[c]).abs() < 1e-6,
                "{night:?}"
            );
        }
        let noon = noon_light();
        assert_eq!(noon.night, 0.0);
        let y = luma(noon.e_sky_raw);
        for c in 0..3 {
            let want = y + 0.25 * (noon.e_sky_raw[c] - y);
            assert!((noon.e_sky[c] - want).abs() < 1e-6, "{noon:?}");
        }
    }

    /// No chroma kept, the body's light is grey.
    #[test]
    fn no_chroma_is_neutral() {
        let look = BodyLook {
            illum_chroma: 0.0,
            ..BodyLook::default()
        };
        let e = light_toward(Vec3::Y, &look);
        for c in [e.e_sky, e.e_sun] {
            assert!(chroma(c) < 1e-6, "{c:?}");
        }
    }

    /// In MCSH shadow the body keeps what the terrain keeps, 0.7 in gamma: the sun's floor `s0`
    /// makes `Y(e_sky + s0·e_sun)` 0.456 of the open light, or 0 where the sky alone is more.
    #[test]
    fn shadowed_water_darkens_as_the_terrain_beside_it() {
        let law = dec(0.7);
        assert!((law - 0.456).abs() < 1e-3, "{law}");
        let (ambient, _, sun, dome) = day();
        // A night-like dim dome under a bright sun, the day, and a sky-lit dusk.
        let dim = dome.map(|r| r * 0.15);
        let mut branches = [0; 2];
        for (diffuse, dome) in [
            ([1.0, 1.0, 1.0], &dim),
            ([0.80, 0.70, 0.55], &dome),
            ([0.2, 0.2, 0.25], &dome),
        ] {
            let e = body_light(ambient, diffuse, sun, Vec3::Y, dome, &BodyLook::default());
            let open = luma(e.e_sky) + luma(e.e_sun);
            let shade = luma(e.e_sky) + e.s0 * luma(e.e_sun);
            if 1.0 - e.sun_share < law {
                assert!((shade / open - law).abs() < 1e-4, "{e:?}");
                branches[0] += 1;
            } else {
                assert_eq!(e.s0, 0.0, "{e:?}");
                branches[1] += 1;
            }
            assert!((0.0..=1.0).contains(&e.s0), "{e:?}");
        }
        assert!(
            branches.iter().all(|&n| n > 0),
            "both sides of the law: {branches:?}"
        );
    }

    /// A zeroed light, a dark sky with no sun and a sunless dome are all finite: every division
    /// is guarded.
    #[test]
    fn a_zeroed_light_is_finite() {
        let look = BodyLook::default();
        let cases = [
            body_light(
                [0.0; 3],
                [0.0; 3],
                Vec3::ZERO,
                Vec3::ZERO,
                &[Vec4::ZERO; 7],
                &look,
            ),
            body_light(
                [0.0; 3],
                [0.0; 3],
                Vec3::NEG_Y,
                Vec3::Y,
                &[Vec4::ZERO; 7],
                &look,
            ),
            body_light([0.1; 3], [0.0; 3], Vec3::X, Vec3::Y, &day().3, &look),
        ];
        for e in cases {
            let all = [e.e_sky, e.e_sun, e.e_sky_raw, e.e_sun_raw].concat();
            assert!(
                all.iter()
                    .chain(&[e.s0, e.k_e, e.sun_share])
                    .all(|x| x.is_finite()),
                "{e:?}"
            );
        }
        assert_eq!(cases[2].sun_share, 0.0);
        let mut light = WowLighting::default();
        light.sun_dir = Vec3::ZERO;
        let lanes = body_lanes(&light, &[Vec4::ZERO; 7], &look, [0.0; 4]);
        assert!(lanes.iter().all(|x| x.is_finite()), "{lanes:?}");
    }

    /// The black-row branch is a blend: sweeping the deepest row's peak from black to byte 8 in
    /// 2000 steps moves no absorption channel by more than 1% a step.
    #[test]
    fn a_black_deep_row_fades_to_the_clear_lake_shape_without_a_step() {
        let top = dec(8.0 / 255.0);
        for tint in [[1.0, 0.5, 0.2], [0.1, 0.4, 1.0], [0.3, 1.0, 0.6]] {
            let at = |i: usize| {
                let m = top * i as f32 / 2000.0;
                absorption(tint.map(|t| t * m), 0.25)
            };
            let mut last = at(0);
            for i in 1..=2000 {
                let s = at(i);
                for c in 0..3 {
                    let step = (s[c] / last[c] - 1.0).abs();
                    assert!(step <= 0.01, "{tint:?} at step {i}, channel {c}: {step}");
                }
                last = s;
            }
        }
    }

    /// The lift adds at most `lift` to any channel and keeps the row's hue.
    #[test]
    fn the_lift_never_adds_more_than_lift_to_a_channel() {
        let mut rows = vec![[0.05, 0.15, 0.30], [0.0; 3], [1.0, 1.0, 1.0]];
        for r in 0..6 {
            for g in 0..6 {
                for b in 0..6 {
                    rows.push([r as f32 / 5.0, g as f32 / 5.0, b as f32 / 5.0]);
                }
            }
        }
        for row in rows {
            for k in 0..=25 {
                let lift = k as f32 / 100.0;
                let l = body_lift(row, lift);
                let peak = l[0].max(l[1]).max(l[2]);
                assert!(peak <= lift + 1e-6, "{row:?} at {lift}: {l:?}");
                if chroma(row) > 0.02 && lift > 0.0 {
                    assert!((hue(l) - hue(row)).abs() < 1e-3, "{row:?} at {lift}: {l:?}");
                }
            }
        }
    }

    /// The shader keeps these lines of the body: `body_lift`'s return as [`body_lift`] has it, the
    /// stops read from rows 0 and 63 (no ocean tail, as [`deep_stop`]) at the ceiling lane and half
    /// of it, and `t_bed` and `w_in` from one `t_turb`. A text pin: the path length, `x`, the bands,
    /// the ramp and the stops' argument order are not checked here.
    #[test]
    fn the_shader_keeps_the_body_lift_stop_and_transport_lines() {
        let wgsl = include_str!("../../benilla-assets/src/shaders/liquid.wgsl");
        for line in [
            "return lift * a_g / max(max(a_g.r, max(a_g.g, a_g.b)), max(lift, 1e-4));",
            "let r0 = swatch_row(body_shallow, body_deep, 0.0, ocean).rgb;",
            "let r63 = swatch_row(body_shallow, body_deep, 63.0, false).rgb;",
            "let dc = water_reflect.look.body.spare.y;",
            "mix(r0, r63, 0.5 * dc),",
            "mix(r0, r63, dc),",
            "o.t_bed = select(vec3<f32>(0.0), exp(-bi.sig_abs * o.l_opt) * t_turb, o.known);",
            "o.w_in = select(1.0, 1.0 - t_turb, o.known);",
        ] {
            assert!(wgsl.contains(line), "liquid.wgsl lost `{line}`");
        }
    }

    /// The deep stop at the ceiling is today's `mix(shallow, deep, BODY_DEPTH_MAX)` within one row
    /// step of the byte ramp, `ceil(|c1 − c0|/64)`.
    #[test]
    fn the_deep_ceiling_is_body_depth_max() {
        let wgsl = include_str!("../../benilla-assets/src/shaders/liquid.wgsl");
        let ceiling = BodyLook::default().deep_ceiling;
        assert!(wgsl.contains(&format!("const BODY_DEPTH_MAX: f32 = {ceiling};")));
        let swatches = [
            ([90, 140, 140], [45, 35, 36]),
            ([60, 88, 89], [10, 25, 40]),
            ([255, 0, 128], [0, 255, 3]),
            ([30, 30, 30], [31, 29, 30]),
        ];
        for (s, d) in swatches {
            let (s, d) = (
                s.map(|b: i32| b as f32 / 255.0),
                d.map(|b: i32| b as f32 / 255.0),
            );
            for ocean in [false, true] {
                let stop = deep_stop(s, d, ocean, ceiling);
                for c in 0..3 {
                    let today = s[c] + (d[c] - s[c]) * ceiling;
                    let step = ((d[c] - s[c]).abs() * 255.0 / 64.0).ceil() / 255.0;
                    assert!((stop[c] - today).abs() <= step + 1e-6, "{s:?} {d:?} {c}");
                }
            }
        }
    }

    /// The rows sit where `WaterBody` declares them.
    #[test]
    fn the_body_rows_are_the_shaders() {
        let wgsl = include_str!("../../benilla-assets/src/shaders/liquid.wgsl");
        let body = &wgsl[wgsl.find("struct WaterBody {").expect("WaterBody")..];
        let body = &body[..body.find("};").expect("its end")];
        let members: Vec<&str> = body
            .lines()
            .map(str::trim)
            .filter_map(|l| l.strip_suffix(": vec4<f32>,"))
            .collect();
        assert_eq!(
            members,
            [
                "sig_river",
                "sig_ocean",
                "e_sky",
                "e_sun",
                "look",
                "refr",
                "shadow",
                "spare",
                "e_sky_raw",
                "e_sun_raw"
            ]
        );
        assert_eq!(members.len() * 4, BODY_FLOATS);
        let (ambient, diffuse, sun, dome) = day();
        let mut light = WowLighting::default();
        light.ambient = ambient;
        light.diffuse = diffuse;
        light.sun_dir = sun;
        light.celestial_dir = Vec3::new(0.0, 1.0, 0.1);
        let lanes = body_lanes(&light, &dome, &BodyLook::default(), [1.0, 2.0, 500.0, 1.0]);
        let row = |i: usize| -> [f32; 4] { lanes[4 * i..4 * i + 4].try_into().unwrap() };
        assert_eq!(row(4), [0.615, 0.5, 0.22, 0.25]);
        assert_eq!(row(5), [1.0, 0.01, 1.0, 0.0]);
        assert_eq!(row(6), [1.0, 2.0, 500.0, 1.0]);
        assert_eq!(row(7)[1], 0.72);
        let e = noon_light();
        assert_eq!(row(2), [e.e_sky[0], e.e_sky[1], e.e_sky[2], e.s0]);
        assert_eq!(row(9)[..3], e.e_sun_raw);
    }
}
