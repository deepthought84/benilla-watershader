//! **The stylised look's rows** of the reflection block (`liquid.wgsl`'s `WaterReflect::look`, a
//! `WaterLook`): the owner's and the developer's knobs, written whole by `drive_reflection` every
//! frame, so every row is valid on the frames the mirror is off.
//!
//! Each row has one writer. [`look_lanes`] writes every row but `water_rect`, which leaves it 0: the
//! water's screen rectangle is known only after visibility, which runs after the mirror's system,
//! so the render world writes that row itself (`reflect::upload_reflect`). A row whose knob does not
//! exist yet stays 0.

use std::sync::OnceLock;

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;

use benilla_world::liquid::{
    RippleAniso, RippleStats, WaterQuality, LOOK_COMPOSITE, LOOK_DETAIL, LOOK_GRADE, LOOK_LANES,
    LOOK_QUALITY, LOOK_RIPPLE_MEAN, LOOK_RIPPLE_STATS, LOOK_SHORE, LOOK_SHORE_KIND, LOOK_SURF,
};

/// The calm surface's GGX roughness α0: √0.003, Cox and Munk's (1954) clean-surface slope
/// variance.
const ALPHA_CALM: f32 = 0.0548;

/// Where a ripple octave's features fade from resolved slope into roughness, in pixels of
/// footprint along the major axis: below 2 (Nyquist) none resolve, from 4 (a 2x guard for the box
/// mip's leakage) all do.
const FEATURE_FADE_PX: (f32, f32) = (2.0, 4.0);

/// The ripple map's anisotropy clamp on the High tier; Medium and Low take none.
const HIGH_RIPPLE_ANISO: u16 = 4;

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
    /// The reflection weight's transitional ceiling at grazing angles, 1 for none: 0.45, today's
    /// `mix(0.02, 0.45, …)` ceiling.
    pub(crate) grazing_limit: f32,
    /// The glint lobes' weight in linear light: the gamma weight 0.22 raised to 2.2, so the peak
    /// over black is unchanged.
    pub(crate) glint_weight_lin: f32,
    /// A glow excess forced onto every water pixel, for testing the glow path; 0.
    pub(crate) glow_test: f32,
    /// The reflection grading's strength `s`; 0, no grading.
    pub(crate) grade_strength: f32,
    /// The grading's near share `C_NEAR`, 0.2.
    pub(crate) grade_c_near: f32,
    /// The grading's far share `C_FAR`, 0.3.
    pub(crate) grade_c_far: f32,
    /// Whether the sky floor is occluded by what the frame shows above the water.
    pub(crate) floor_occlusion: bool,
    /// The authored sheet's contrast as painted detail; 0, off.
    pub(crate) painted_detail: f32,
    /// The shore's wet line, wet darkening, contact foam and meniscus gains.
    pub(crate) k_line: f32,
    pub(crate) f_wet: f32,
    pub(crate) k_contact: f32,
    pub(crate) k_meniscus: f32,
    /// The wet line's multiplier on still water, rapids and ocean.
    pub(crate) kind_line: [f32; 3],
}

impl Default for WaterLook {
    fn default() -> Self {
        Self {
            grazing_limit: 0.45,
            glint_weight_lin: 0.22f32.powf(2.2),
            glow_test: 0.0,
            grade_strength: 0.0,
            grade_c_near: 0.2,
            grade_c_far: 0.3,
            floor_occlusion: true,
            painted_detail: 0.0,
            k_line: 0.35,
            f_wet: 0.61,
            k_contact: 1.0,
            k_meniscus: 1.0,
            kind_line: [1.0; 3],
        }
    }
}

/// The look block's floats, [`LOOK_LANES`] long.
pub(crate) const LOOK_FLOATS: usize = LOOK_LANES.end - LOOK_LANES.start;

/// What the look rows copy from the engine rather than from a knob.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct LookInputs {
    /// The quality tier (`WaterQuality::tier`), 0 Low, 1 Medium, 2 High.
    pub(crate) tier: u8,
    /// The ripple map's slope statistics; zero without a map, which reads as nothing lost and no
    /// bias, the calm roughness everywhere.
    pub(crate) ripple: RippleStats,
    /// The ripple sampler's anisotropy clamp ([`RippleAniso`]); 0 reads as 1, none.
    pub(crate) aniso: u16,
}

/// What `drive_reflection` writes the look block from: the knobs, the quality tier, the ripple
/// map's statistics, when the map exists, and its sampler's anisotropy clamp.
#[derive(SystemParam)]
pub(crate) struct LookSources<'w> {
    look: Res<'w, WaterLook>,
    quality: Res<'w, WaterQuality>,
    ripple: Option<Res<'w, RippleStats>>,
    aniso: Res<'w, RippleAniso>,
}

impl LookSources<'_> {
    /// This frame's look block ([`look_lanes`]).
    pub(crate) fn lanes(&self) -> [f32; LOOK_FLOATS] {
        let inputs = LookInputs {
            tier: self.quality.tier(),
            ripple: self.ripple.as_deref().copied().unwrap_or_default(),
            aniso: self.aniso.0,
        };
        look_lanes(&self.look, &inputs)
    }
}

/// The look block as `WaterLook` lays it out. `water_rect` is left 0 for the render world, and
/// the rows no knob fills yet (the body) stay 0.
pub(crate) fn look_lanes(look: &WaterLook, inputs: &LookInputs) -> [f32; LOOK_FLOATS] {
    let mut lanes = [0.0; LOOK_FLOATS];
    let mut row = |at: usize, v: [f32; 4]| {
        lanes[at - LOOK_LANES.start..][..4].copy_from_slice(&v);
    };
    // The tier is a copy for the debug views: the shader branches on its def.
    row(
        LOOK_QUALITY,
        [
            f32::from(inputs.tier),
            f32::from(look.floor_occlusion),
            0.0,
            0.0,
        ],
    );
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
            FEATURE_FADE_PX.0,
            FEATURE_FADE_PX.1,
        ],
    );
    row(LOOK_DETAIL, [look.painted_detail, 0.0, 0.0, 0.0]);
    for (i, level) in inputs.ripple.levels.iter().enumerate() {
        row(LOOK_RIPPLE_STATS + 4 * i, *level);
    }
    row(LOOK_RIPPLE_MEAN, inputs.ripple.mean);
    row(
        LOOK_SHORE,
        [look.k_line, look.f_wet, look.k_contact, look.k_meniscus],
    );
    let [still, rapids, ocean] = look.kind_line;
    row(LOOK_SHORE_KIND, [still, rapids, ocean, 0.0]);
    lanes
}

#[cfg(test)]
mod tests {
    use super::*;
    use benilla_world::liquid::LOOK_WATER_RECT;

    /// The defaults land in their rows; the render world's row and the unfilled rows stay 0.
    #[test]
    fn the_look_rows_carry_their_defaults() {
        let inputs = LookInputs {
            tier: 1,
            ..LookInputs::default()
        };
        let lanes = look_lanes(&WaterLook::default(), &inputs);
        let row =
            |at: usize| -> [f32; 4] { lanes[at - LOOK_LANES.start..][..4].try_into().unwrap() };
        assert_eq!(row(LOOK_QUALITY), [1.0, 1.0, 0.0, 0.0]);
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
        assert_eq!(row(LOOK_GRADE), [0.0, 0.3, 0.2, 0.0]);
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

    /// The ripple map's statistics land in the nine level rows and the mean row, in order; without
    /// a map they are 0, which the shader reads as nothing lost.
    #[test]
    fn the_ripple_rows_carry_the_maps_statistics() {
        let mut ripple = RippleStats::default();
        for (i, level) in ripple.levels.iter_mut().enumerate() {
            *level = [i as f32, 10.0 + i as f32, 20.0 + i as f32, 30.0 + i as f32];
        }
        ripple.mean = [-0.004, -0.003, 6.5e-4, 17.9];
        let lanes = look_lanes(
            &WaterLook::default(),
            &LookInputs {
                tier: 1,
                ripple,
                ..LookInputs::default()
            },
        );
        let row =
            |at: usize| -> [f32; 4] { lanes[at - LOOK_LANES.start..][..4].try_into().unwrap() };
        for (i, level) in ripple.levels.iter().enumerate() {
            assert_eq!(row(LOOK_RIPPLE_STATS + 4 * i), *level, "level {i}");
        }
        assert_eq!(row(LOOK_RIPPLE_MEAN), ripple.mean);
        let bare = look_lanes(&WaterLook::default(), &LookInputs::default());
        let rows = LOOK_RIPPLE_STATS - LOOK_LANES.start..LOOK_RIPPLE_MEAN + 4 - LOOK_LANES.start;
        assert!(bare[rows].iter().all(|&v| v == 0.0));
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
