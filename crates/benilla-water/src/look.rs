//! **The stylised look's rows** of the reflection block (`liquid.wgsl`'s `WaterReflect::look`, a
//! `WaterLook`): the owner's and the developer's knobs, written whole by `drive_reflection` every
//! frame, so every row is valid on the frames the mirror is off.
//!
//! Each row has one writer. [`look_lanes`] writes every row but `water_rect` and `quality.y`, which
//! it leaves 0: the water's screen rectangle, and whether its floor is occluded this frame
//! (`floor_share`), are known only after visibility, which runs after the mirror's system, so the
//! render world writes them itself (`reflect::upload_reflect`). The body's rows are
//! `body::body_lanes`, from the resolved light and the dome.

use std::sync::OnceLock;

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::render::camera::MipBias;

use crate::body::{body_lanes, BodyLook, BODY_FLOATS};
use crate::lighting::WowLighting;
use crate::mcsh::WaterShadowWindow;
use crate::view::WorldCamera;
use benilla_world::liquid::{
    RippleAniso, RippleStats, WaterQuality, LOOK_BODY, LOOK_COMPOSITE, LOOK_DETAIL, LOOK_GRADE,
    LOOK_LANES, LOOK_QUALITY, LOOK_RIPPLE_LOSS, LOOK_RIPPLE_MEAN, LOOK_SHORE, LOOK_SHORE_KIND,
    LOOK_SURF, RIPPLE_LOSS_ROWS, RIPPLE_SIZE,
};

/// The calm surface's GGX roughness α0: √0.003, Cox and Munk's (1954) clean-surface slope
/// variance.
const ALPHA_CALM: f32 = 0.0548;

/// Where a ripple octave's features fade from resolved slope into roughness, in pixels of
/// footprint along the major axis: below 2 (Nyquist) none resolve, from 4 (a 2x guard for the box
/// mip's leakage) all do.
const FEATURE_FADE_PX: (f32, f32) = (2.0, 4.0);

/// The glints' weight in linear light: the gamma look's `SPEC_WEIGHT` 0.22 raised to 2.2, the same
/// peak over black.
const GLINT_WEIGHT_LIN: f32 = 0.035_754_43;

/// The reflectance's ceiling at grazing angles: the gamma look's `mix(0.02, 0.45, fresnel)` peak.
const GRAZING_LIMIT: f32 = 0.45;

/// The reflection grading's strength `s`: near reflections (a shore's huts, a pier) soften to about
/// the gamma look's contrast, open-sky ones keep the linear composite's physical strength.
const GRADE_STRENGTH: f32 = 0.8;

/// The ripple map's anisotropy clamp on the High tier; Medium and Low take none.
const HIGH_RIPPLE_ANISO: u16 = 4;

/// The ripple octaves `liquid.wgsl`'s `water_surface` sums: each tiles 12·scale yards of the map
/// and weighs its slope by weight.
const RIPPLE_OCTAVES: [(f32, f32); 5] = [
    (9.2, 0.85),
    (3.4, 1.0),
    (1.0, 1.0),
    (0.30, 0.70),
    (0.12, 0.20),
];

/// The loss table's footprint LOD at row 0, at one texel a yard (`liquid.wgsl`'s
/// `RIPPLE_LOSS_FROM`): the finest octave reads 7.47 LODs above it, so from here its mip has lost
/// nothing. The [`RIPPLE_LOSS_ROWS`] reach LOD 7, past where the coarsest (1.21 above) has lost
/// all it can.
const LOSS_FROM: f32 = -8.0;

/// The loss table's rows to each LOD (`RIPPLE_LOSS_PER_LOD`): a linear read between them stays
/// within about 2 codes of the ROUGHNESS view of the octave-by-octave sum, where the feature fade
/// bends it.
const LOSS_PER_LOD: f32 = 8.0;

/// [`FEATURE_FADE_PX`], unless `$WOW_WATER_FADE=<start>,<end>` (read once, pixels, start below
/// end) sets the fade, to measure it.
fn feature_fade() -> (f32, f32) {
    static FORCED: OnceLock<Option<(f32, f32)>> = OnceLock::new();
    FORCED
        .get_or_init(|| {
            let v = std::env::var("WOW_WATER_FADE").ok()?;
            let (a, b) = v.split_once(',')?;
            let (a, b) = (a.trim().parse::<f32>().ok()?, b.trim().parse::<f32>().ok()?);
            (0.0 < a && a < b).then_some((a, b))
        })
        .unwrap_or(FEATURE_FADE_PX)
}

/// Whether the sky floor is occluded: on, unless `$WOW_WATER_FLOOR_OCC=0` (read once) turns it
/// off, for an A/B.
fn floor_occlusion() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    !*OFF.get_or_init(|| std::env::var("WOW_WATER_FLOOR_OCC").is_ok_and(|v| v.trim() == "0"))
}

/// Whether the stylised pipelines take the linear composite (`WATER_COMPOSITE`): on, unless
/// `$WOW_WATER_COMPOSITE` (read once) is an off word, [`lever_off`], which asks for the gamma look,
/// for an A/B.
pub(crate) fn composite_variant() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    !*OFF.get_or_init(|| std::env::var("WOW_WATER_COMPOSITE").is_ok_and(|v| lever_off(&v)))
}

/// Whether an on/off lever's value turns it off: `0`, `false`, `off` or `no`, in any case.
fn lever_off(v: &str) -> bool {
    let v = v.trim();
    ["0", "false", "off", "no"]
        .iter()
        .any(|w| v.eq_ignore_ascii_case(w))
}

/// The glints' linear weight: `SPEC_WEIGHT` 0.22 raised to the gamma, so the peak over black is the
/// gamma look's, unless `$WOW_WATER_GLINT=<weight>` (read once, above 0) sets it, to fit it.
fn glint_weight_lin() -> f32 {
    static FORCED: OnceLock<Option<f32>> = OnceLock::new();
    FORCED
        .get_or_init(|| {
            let w = std::env::var("WOW_WATER_GLINT")
                .ok()?
                .trim()
                .parse::<f32>()
                .ok()?;
            (w > 0.0 && w.is_finite()).then_some(w)
        })
        .unwrap_or(GLINT_WEIGHT_LIN)
}

/// A unit-interval lever read once from `$<var>`, for an A/B of an owner's knob.
fn unit_lever(var: &str) -> Option<f32> {
    let v = std::env::var(var).ok()?.trim().parse::<f32>().ok()?;
    (0.0..=1.0).contains(&v).then_some(v)
}

/// The grading's strength, [`GRADE_STRENGTH`], unless `$WOW_WATER_GRADE=<s>` (read once, finite,
/// at least 0) sets it, to measure it. Only the linear composite grades: under the gamma look
/// (`$WOW_WATER_COMPOSITE=0`) it is 0 and the lever is ignored, so it cannot force the floor's
/// readers on there.
fn grade_strength() -> f32 {
    static FORCED: OnceLock<Option<f32>> = OnceLock::new();
    let forced = *FORCED.get_or_init(|| {
        let s = std::env::var("WOW_WATER_GRADE")
            .ok()?
            .trim()
            .parse::<f32>()
            .ok()?;
        if !composite_variant() && s != 0.0 {
            warn!("water: WOW_WATER_GRADE={s} ignored: the gamma look has no grading");
        }
        (s >= 0.0 && s.is_finite()).then_some(s)
    });
    if !composite_variant() {
        return 0.0;
    }
    forced.unwrap_or(GRADE_STRENGTH)
}

/// The grazing limit, 0.45, unless `$WOW_WATER_GRAZE=<0-1>` (read once; 1 for none) sets it.
fn grazing_limit() -> f32 {
    static FORCED: OnceLock<Option<f32>> = OnceLock::new();
    FORCED
        .get_or_init(|| unit_lever("WOW_WATER_GRAZE"))
        .unwrap_or(GRAZING_LIMIT)
}

/// The body's knobs at their defaults, unless `$WOW_WATER_DEEP_CEILING=<0-1>` sets where the deep
/// stop sits on the zone's line (1 the swatch's last row), `$WOW_WATER_BAND_PUSH=<0-1>` how far
/// the three bands pull the ramp (0 unbanded) or `$WOW_WATER_TURB=<0-4>` the turbidity's scale on
/// both kinds; each read once, to measure it.
fn body_look() -> BodyLook {
    static FORCED: OnceLock<(Option<f32>, Option<f32>, Option<f32>)> = OnceLock::new();
    let (ceiling, push, turb) = *FORCED.get_or_init(|| {
        let turb = std::env::var("WOW_WATER_TURB")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|v| (0.0..=4.0).contains(v));
        (
            unit_lever("WOW_WATER_DEEP_CEILING"),
            unit_lever("WOW_WATER_BAND_PUSH"),
            turb,
        )
    });
    let look = BodyLook::default();
    BodyLook {
        deep_ceiling: ceiling.unwrap_or(look.deep_ceiling),
        band_push: push.unwrap_or(look.band_push),
        turb_scale: turb.map_or(look.turb_scale, |t| look.turb_scale.map(|s| s * t)),
        ..look
    }
}

/// The ripple map's anisotropy clamp for quality tier `tier` (0 Low, 1 Medium, 2 High).
pub(crate) fn tier_ripple_aniso(tier: u8) -> u16 {
    if tier >= 2 {
        HIGH_RIPPLE_ANISO
    } else {
        1
    }
}

/// [`tier_ripple_aniso`], unless `$WOW_WATER_AF=<1-16>` (read once) sets the clamp for every tier,
/// to measure its cost.
pub(crate) fn ripple_aniso(tier: u8) -> u16 {
    static FORCED: OnceLock<Option<u16>> = OnceLock::new();
    FORCED
        .get_or_init(|| {
            std::env::var("WOW_WATER_AF")
                .ok()
                .and_then(|v| v.trim().parse::<u16>().ok())
                .map(|n| n.clamp(1, 16))
        })
        .unwrap_or_else(|| tier_ripple_aniso(tier))
}

/// The owner's and the developer's knobs for the look rows, at their defaults.
#[derive(Resource, Clone, Debug, PartialEq)]
pub(crate) struct WaterLook {
    /// The linear composite's reflectance ceiling at grazing angles, 1 for none: 0.45, the
    /// ceiling of the gamma look's fixed `mix(0.02, 0.45, fresnel)`.
    pub(crate) grazing_limit: f32,
    /// The glint lobes' weight in linear light: the gamma weight 0.22 raised to 2.2, so the peak
    /// over black is unchanged.
    pub(crate) glint_weight_lin: f32,
    /// A glow excess forced onto every water pixel, for testing the glow path; 0.
    pub(crate) glow_test: f32,
    /// The reflection grading's strength `s`, [`GRADE_STRENGTH`]; 0 under the gamma look. The
    /// linear composite reads it.
    pub(crate) grade_strength: f32,
    /// The grading's near share `C_NEAR`, 0.2.
    pub(crate) grade_c_near: f32,
    /// The grading's far share `C_FAR`, 0.3.
    pub(crate) grade_c_far: f32,
    /// Whether the sky floor is occluded by what the frame shows above the water.
    pub(crate) floor_occlusion: bool,
    /// The authored sheet's contrast as painted detail; 0, off.
    pub(crate) painted_detail: f32,
    /// The body's optics and light.
    pub(crate) body: BodyLook,
    /// The shore's wet line, wet darkening, contact foam and meniscus gains.
    pub(crate) k_line: f32,
    pub(crate) f_wet: f32,
    pub(crate) k_contact: f32,
    pub(crate) k_meniscus: f32,
    /// The wet line's multiplier on still water, rapids and ocean.
    pub(crate) kind_line: [f32; 3],
    /// Where a ripple octave's features fade from resolved slope into roughness, in pixels
    /// ([`FEATURE_FADE_PX`]).
    pub(crate) feature_fade: (f32, f32),
}

impl Default for WaterLook {
    fn default() -> Self {
        Self {
            grazing_limit: grazing_limit(),
            glint_weight_lin: glint_weight_lin(),
            glow_test: 0.0,
            grade_strength: grade_strength(),
            grade_c_near: 0.2,
            grade_c_far: 0.3,
            floor_occlusion: floor_occlusion(),
            painted_detail: 0.0,
            body: body_look(),
            k_line: 0.35,
            f_wet: 0.61,
            k_contact: 1.0,
            k_meniscus: 1.0,
            kind_line: [1.0; 3],
            feature_fade: feature_fade(),
        }
    }
}

/// The look block's floats, [`LOOK_LANES`] long.
pub(crate) const LOOK_FLOATS: usize = LOOK_LANES.end - LOOK_LANES.start;

/// What the look rows copy from the engine rather than from a knob.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct LookInputs<'a> {
    /// The quality tier (`WaterQuality::tier`), 0 Low, 1 Medium, 2 High.
    pub(crate) tier: u8,
    /// The ripple map's slope statistics. Zero without a map, which never draws: no bias, and no
    /// feature wavelength, so every octave would fade out and the water would draw flat.
    pub(crate) ripple: RippleStats,
    /// The ripple sampler's anisotropy clamp ([`RippleAniso`]); 0 reads as 1, none.
    pub(crate) aniso: u16,
    /// The ripple's loss table ([`ripple_loss`]); rows it does not fill stay 0, nothing lost.
    pub(crate) loss: &'a [[f32; 4]],
    /// The body rows ([`body_lanes`]); `None` leaves them 0.
    pub(crate) body: Option<&'a [f32; BODY_FLOATS]>,
}

/// The ripple's loss table and the inputs it was tabulated from, so a frame rebuilds it only when
/// the map's statistics, the world camera's mip bias or the feature fade change.
#[derive(Default)]
pub(crate) struct LossTable {
    from: Option<(RippleStats, f32, (f32, f32))>,
    rows: Vec<[f32; 4]>,
}

impl LossTable {
    fn rows(&mut self, ripple: &RippleStats, bias: f32, fade: (f32, f32)) -> &[[f32; 4]] {
        let from = Some((*ripple, bias, fade));
        if self.from != from {
            self.rows = ripple_loss(ripple, bias, fade);
            self.from = from;
        }
        &self.rows
    }
}

/// What `drive_reflection` writes the look block from: the knobs, the quality tier, the ripple
/// map's statistics, when the map exists, its sampler's anisotropy clamp, the world camera's
/// texture LOD bias, which every octave's LOD carries, and the terrain shadow's window.
#[derive(SystemParam)]
pub(crate) struct LookSources<'w, 's> {
    look: Res<'w, WaterLook>,
    quality: Res<'w, WaterQuality>,
    ripple: Option<Res<'w, RippleStats>>,
    aniso: Res<'w, RippleAniso>,
    bias: Query<'w, 's, &'static MipBias, With<WorldCamera>>,
    shadow: Res<'w, WaterShadowWindow>,
    loss: Local<'s, LossTable>,
}

impl LookSources<'_, '_> {
    /// This frame's look block ([`look_lanes`]) under `light`, whose dome rows are `dome`
    /// (`sky::dome_uniforms`).
    pub(crate) fn lanes(&mut self, light: &WowLighting, dome: &[Vec4; 7]) -> [f32; LOOK_FLOATS] {
        let ripple = self.ripple.as_deref().copied().unwrap_or_default();
        let bias = self.bias.iter().next().map_or(0.0, |b| b.0);
        let body = body_lanes(light, dome, &self.look.body, self.shadow.lanes());
        let inputs = LookInputs {
            tier: self.quality.tier(),
            ripple,
            aniso: self.aniso.0,
            loss: self.loss.rows(&ripple, bias, self.look.feature_fade),
            body: Some(&body),
        };
        look_lanes(&self.look, &inputs)
    }
}

/// The ripple's lost slope variance, tabulated over the pixel's footprint for `liquid.wgsl`'s
/// `ripple_loss_at`: row i holds [`ripple_loss_at`] at footprint LOD `LOSS_FROM + i / LOSS_PER_LOD`
/// (one texel a yard, the world camera's mip bias `bias` in it), then at the next row's, so the
/// shader interpolates with one read. `fade` is the feature fade's start and end in pixels.
pub(crate) fn ripple_loss(stats: &RippleStats, bias: f32, fade: (f32, f32)) -> Vec<[f32; 4]> {
    let at = |row: usize| ripple_loss_at(stats, bias, fade, LOSS_FROM + row as f32 / LOSS_PER_LOD);
    (0..RIPPLE_LOSS_ROWS)
        .map(|i| {
            let ([all, mips], [next_all, next_mips]) = (at(i), at(i + 1));
            [all, mips, next_all, next_mips]
        })
        .collect()
}

/// The slope variance per axis the ripple octaves lose, w²-weighted, at footprint LOD `lod` (log2
/// yards plus the mip bias `bias`): every loss, then the mips' share. Each octave reads the map at
/// LOD `lod + log2(its texels a yard)`, where its mip has lost the level's variance against level
/// 0 (`stats.levels`, interpolated between levels and averaged over the two axes: the map is
/// isotropic), and its feature fade, smoothstep(`fade`) of the map's feature wavelength in pixels,
/// drops `1 − keep²` of what the level still resolves.
fn ripple_loss_at(stats: &RippleStats, bias: f32, fade: (f32, f32), lod: f32) -> [f32; 2] {
    let last = (stats.levels.len() - 1) as f32;
    let level = |lam: f32| {
        let l = lam.clamp(0.0, last);
        let i = l.floor() as usize;
        let j = (i + 1).min(stats.levels.len() - 1);
        let mix = |c: usize| {
            stats.levels[i][c] + (stats.levels[j][c] - stats.levels[i][c]) * (l - i as f32)
        };
        (0.5 * (mix(0) + mix(1)), mix(3))
    };
    let smoothstep = |x: f32| {
        let t = ((x - fade.0) / (fade.1 - fade.0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    // The footprint's major extent in yards.
    let major = (lod - bias).exp2();
    let (mut all, mut mips) = (0.0, 0.0);
    for (scale, weight) in RIPPLE_OCTAVES {
        let texels_per_yd = RIPPLE_SIZE as f32 / (12.0 * scale);
        let (lost, resolved) = level(lod + texels_per_yd.log2());
        let keep = smoothstep(stats.mean[3] / (major * texels_per_yd));
        let w2 = weight * weight;
        mips += w2 * lost;
        all += w2 * (lost + resolved * (1.0 - keep * keep));
    }
    [all, mips]
}

/// The look block as `WaterLook` lays it out. `water_rect` and `quality.y` are left 0 for the render
/// world.
pub(crate) fn look_lanes(look: &WaterLook, inputs: &LookInputs) -> [f32; LOOK_FLOATS] {
    let mut lanes = [0.0; LOOK_FLOATS];
    let mut row = |at: usize, v: [f32; 4]| {
        lanes[at - LOOK_LANES.start..][..4].copy_from_slice(&v);
    };
    // The tier is a copy for the debug views: the shader branches on its def.
    row(LOOK_QUALITY, [f32::from(inputs.tier), 0.0, 0.0, 0.0]);
    row(
        LOOK_COMPOSITE,
        [
            look.grazing_limit,
            look.glint_weight_lin,
            look.glow_test,
            0.0,
        ],
    );
    row(
        LOOK_GRADE,
        [
            look.grade_strength,
            look.grade_c_far,
            look.grade_c_near,
            0.0,
        ],
    );
    // `y` is the ripple sampler's anisotropy clamp, so the shader's minor-axis LOD is the probes'.
    row(
        LOOK_SURF,
        [
            ALPHA_CALM,
            f32::from(inputs.aniso.max(1)),
            look.feature_fade.0,
            look.feature_fade.1,
        ],
    );
    row(LOOK_DETAIL, [look.painted_detail, 0.0, 0.0, 0.0]);
    for (i, loss) in inputs.loss.iter().take(RIPPLE_LOSS_ROWS).enumerate() {
        row(LOOK_RIPPLE_LOSS + 4 * i, *loss);
    }
    row(LOOK_RIPPLE_MEAN, inputs.ripple.mean);
    row(
        LOOK_SHORE,
        [look.k_line, look.f_wet, look.k_contact, look.k_meniscus],
    );
    let [still, rapids, ocean] = look.kind_line;
    row(LOOK_SHORE_KIND, [still, rapids, ocean, 0.0]);
    if let Some(body) = inputs.body {
        lanes[LOOK_BODY - LOOK_LANES.start..][..BODY_FLOATS].copy_from_slice(body);
    }
    lanes
}

#[cfg(test)]
mod tests {
    use super::*;
    use benilla_world::liquid::LOOK_WATER_RECT;

    /// The defaults land in their rows; the render world's row, and the body's without its
    /// lanes, stay 0.
    #[test]
    fn the_look_rows_carry_their_defaults() {
        let inputs = LookInputs {
            tier: 1,
            ..LookInputs::default()
        };
        // The composite's strength comes from the constant, not `grade_strength()`, so
        // `$WOW_WATER_COMPOSITE` and `$WOW_WATER_GRADE` do not reach the test.
        let look = WaterLook {
            grade_strength: super::GRADE_STRENGTH,
            ..WaterLook::default()
        };
        let lanes = look_lanes(&look, &inputs);
        let row =
            |at: usize| -> [f32; 4] { lanes[at - LOOK_LANES.start..][..4].try_into().unwrap() };
        assert_eq!(row(LOOK_QUALITY), [1.0, 0.0, 0.0, 0.0]);
        let high = look_lanes(
            &WaterLook::default(),
            &LookInputs {
                tier: 2,
                ..LookInputs::default()
            },
        );
        assert_eq!(
            high[LOOK_QUALITY - LOOK_LANES.start],
            2.0,
            "the tier's copy"
        );
        let [graze, glint, glow, _] = row(LOOK_COMPOSITE);
        assert_eq!((graze, glow), (0.45, 0.0));
        assert!((glint - 0.0358).abs() < 1e-4, "{glint}");
        assert!(
            (super::GLINT_WEIGHT_LIN - 0.22f32.powf(2.2)).abs() < 1e-8,
            "the gamma weight 0.22 raised to the gamma"
        );
        assert_eq!(row(LOOK_GRADE), [0.8, 0.3, 0.2, 0.0]);
        assert!(lever_off(" Off ") && lever_off("0") && lever_off("FALSE") && !lever_off("1"));
        assert_eq!(row(LOOK_SHORE), [0.35, 0.61, 1.0, 1.0]);
        assert_eq!(row(LOOK_SHORE_KIND), [1.0, 1.0, 1.0, 0.0]);
        assert_eq!(row(LOOK_SURF), [0.0548, 1.0, 2.0, 4.0]);
        assert_eq!(row(LOOK_WATER_RECT), [0.0; 4], "the render world's row");
        let written = [
            LOOK_QUALITY,
            LOOK_COMPOSITE,
            LOOK_SURF,
            LOOK_GRADE,
            LOOK_DETAIL,
            LOOK_SHORE,
            LOOK_SHORE_KIND,
        ];
        for (i, v) in lanes.iter().enumerate() {
            let at = LOOK_LANES.start + i;
            if !written.iter().any(|&r| (r..r + 4).contains(&at)) {
                assert_eq!(*v, 0.0, "lane {at} has no writer and must stay 0");
            }
        }
    }

    /// The loss table lands in its rows and the map's mean in its own row; without a map both
    /// are 0, which the shader reads as nothing lost.
    #[test]
    fn the_ripple_rows_carry_the_loss_and_the_mean() {
        let ripple = RippleStats {
            mean: [-0.004, -0.003, 6.5e-4, 17.9],
            ..RippleStats::default()
        };
        let loss: Vec<[f32; 4]> = (0..RIPPLE_LOSS_ROWS)
            .map(|i| [i as f32, 1.0, i as f32 + 1.0, 2.0])
            .collect();
        let lanes = look_lanes(
            &WaterLook::default(),
            &LookInputs {
                tier: 1,
                ripple,
                loss: &loss,
                ..LookInputs::default()
            },
        );
        let row =
            |at: usize| -> [f32; 4] { lanes[at - LOOK_LANES.start..][..4].try_into().unwrap() };
        for (i, l) in loss.iter().enumerate() {
            assert_eq!(row(LOOK_RIPPLE_LOSS + 4 * i), *l, "row {i}");
        }
        assert_eq!(row(LOOK_RIPPLE_MEAN), ripple.mean);
        let bare = look_lanes(&WaterLook::default(), &LookInputs::default());
        let rows = LOOK_RIPPLE_LOSS - LOOK_LANES.start..LOOK_RIPPLE_MEAN + 4 - LOOK_LANES.start;
        assert!(bare[rows].iter().all(|&v| v == 0.0));
    }

    /// A map like the shipped one (var0 6.536e-4, feature wavelength 17.9 texels): each level
    /// loses the share of var0 the shipped chain loses, and resolves the rest.
    fn shipped_like() -> RippleStats {
        const LOST: [f32; 9] = [
            0.0, 0.0298, 0.1321, 0.3673, 0.6156, 0.8291, 0.9612, 1.0, 1.0,
        ];
        let var0 = 6.536e-4;
        RippleStats {
            levels: LOST.map(|f| [f * var0, f * var0, 0.0, (1.0 - f) * var0]),
            mean: [-0.0039, -0.0039, var0, 17.9],
        }
    }

    /// The table is the octave-by-octave law, read linearly: each row starts where the last one
    /// ended; nothing is lost below row 0 and everything past the last row; the mips lose no more
    /// than the mips and the fade together; and the mips' share is the same LOD for LOD at any
    /// mip bias, which moves only the fade (it is in pixels).
    #[test]
    fn the_loss_table_tabulates_the_octaves() {
        let stats = shipped_like();
        let w2: f32 = RIPPLE_OCTAVES.iter().map(|(_, w)| w * w).sum();
        let full = w2 * stats.mean[2];
        for bias in [0.0, -1.0] {
            let rows = ripple_loss(&stats, bias, FEATURE_FADE_PX);
            assert_eq!(rows.len(), RIPPLE_LOSS_ROWS);
            for pair in rows.windows(2) {
                assert_eq!(pair[0][2..], pair[1][..2], "{pair:?}");
            }
            assert_eq!(rows[0][..2], [0.0, 0.0], "bias {bias}: row 0 loses nothing");
            let [.., all, mips] = rows[RIPPLE_LOSS_ROWS - 1];
            assert!(
                (all / full - 1.0).abs() < 1e-5,
                "bias {bias}: {all} of {full}"
            );
            assert!(
                (mips / full - 1.0).abs() < 1e-5,
                "bias {bias}: {mips} of {full}"
            );
            for (i, [all, mips, ..]) in rows.iter().enumerate() {
                assert!(*mips <= *all + 1e-9, "bias {bias}, row {i}: {mips} > {all}");
            }
        }
        let (sharp, biased) = (
            ripple_loss(&stats, 0.0, FEATURE_FADE_PX),
            ripple_loss(&stats, -1.0, FEATURE_FADE_PX),
        );
        for (a, b) in sharp.iter().zip(&biased) {
            assert_eq!(a[1], b[1], "the mips' share is the LOD's alone");
        }
        assert_ne!(sharp, biased, "the fade is in pixels, which the bias moves");
        // One point by hand: at LOD 0 the coarsest octave reads its level 1.21 and keeps all of
        // its features (17.9 texels over 2.32 texels a pixel), the finest reads 7.47 and keeps none.
        let [all, mips] = ripple_loss_at(&stats, 0.0, FEATURE_FADE_PX, 0.0);
        let lost = |lam: f32| {
            let (i, t) = (lam.floor() as usize, lam.fract());
            let f = |i: usize| stats.levels[i.min(8)][0];
            f(i) + (f(i + 1) - f(i)) * t
        };
        let mut by_hand = 0.0;
        for (scale, weight) in RIPPLE_OCTAVES {
            by_hand += weight * weight * lost((256.0 / (12.0 * scale)).log2());
        }
        assert!(
            (mips / by_hand - 1.0).abs() < 1e-5,
            "{mips} against {by_hand}"
        );
        assert!(all > mips, "the finer octaves' features fade");
    }

    /// The octaves the table sums are the shader's: their tiles and weights in `water_surface`,
    /// their Σ w² (`OCTAVE_W2_SUM`), and the table's shape (`RIPPLE_LOSS_*`).
    #[test]
    fn the_loss_tables_octaves_are_the_shaders() {
        let wgsl = include_str!("../../benilla-assets/src/shaders/liquid.wgsl");
        let body = &wgsl[wgsl.find("fn water_surface(").unwrap()..];
        let line = |prefix: &str| {
            body.lines()
                .map(str::trim)
                .find(|l| l.starts_with(prefix))
                .unwrap_or_else(|| panic!("water_surface has no `{prefix}`"))
        };
        let literals = |l: &str| -> Vec<f32> {
            l.split(|c: char| " ,();*/=".contains(c))
                .filter(|t| t.contains('.'))
                .filter_map(|t| t.parse().ok())
                .collect()
        };
        for (k, (scale, weight)) in RIPPLE_OCTAVES.iter().enumerate() {
            // `let fK = 1.0 / (12.0 * scale);`, or `1.0 / 12.0` at scale 1.
            let tile: f32 = literals(line(&format!("let f{k} = ")))[1..]
                .iter()
                .product();
            assert!((tile - 12.0 * scale).abs() < 1e-5, "octave {k}: {tile} yd");
            assert_eq!(
                literals(line(&format!("let o{k} = "))),
                [*weight],
                "octave {k}'s weight"
            );
        }
        let w2: f32 = RIPPLE_OCTAVES.iter().map(|(_, w)| w * w).sum();
        assert!(
            wgsl.contains(&format!("const OCTAVE_W2_SUM: f32 = {w2:.4};")),
            "{w2}"
        );
        for decl in [
            format!("const RIPPLE_LOSS_ROWS: u32 = {RIPPLE_LOSS_ROWS}u;"),
            format!("const RIPPLE_LOSS_FROM: f32 = {LOSS_FROM:.1};"),
            format!("const RIPPLE_LOSS_PER_LOD: f32 = {LOSS_PER_LOD:.1};"),
        ] {
            assert!(wgsl.contains(&decl), "liquid.wgsl lacks `{decl}`");
        }
    }

    /// The body's ten rows land in `WaterLook::body`, between the ripple's mean and the shore.
    #[test]
    fn the_body_rows_land_in_the_look_block() {
        let body: [f32; BODY_FLOATS] = std::array::from_fn(|i| i as f32 + 1.0);
        let lanes = look_lanes(
            &WaterLook::default(),
            &LookInputs {
                body: Some(&body),
                ..LookInputs::default()
            },
        );
        let at = LOOK_BODY - LOOK_LANES.start;
        assert_eq!(lanes[at..at + BODY_FLOATS], body);
        assert_eq!(LOOK_BODY, benilla_world::liquid::LOOK_RIPPLE_MEAN + 4);
        assert_eq!(LOOK_BODY + BODY_FLOATS, LOOK_SHORE);
        assert_eq!(WaterLook::default().body, BodyLook::default());
    }

    /// The ripple sampler's clamp lands in `surf.y`, the shader's N; High takes 4, the others none.
    #[test]
    fn the_surf_row_carries_the_ripple_anisotropy() {
        let surf_y = |aniso| {
            look_lanes(
                &WaterLook::default(),
                &LookInputs {
                    aniso,
                    ..LookInputs::default()
                },
            )[LOOK_SURF + 1 - LOOK_LANES.start]
        };
        assert_eq!(surf_y(0), 1.0, "no clamp reads as none");
        assert_eq!(surf_y(4), 4.0);
        assert_eq!(
            [0, 1, 2, 3].map(tier_ripple_aniso),
            [1, 1, 4, 4],
            "Low, Medium, High, and a stale tier past High"
        );
    }
}
