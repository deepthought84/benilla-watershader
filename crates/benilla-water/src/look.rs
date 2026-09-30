//! **The stylised look's rows** of the reflection block (`liquid.wgsl`'s `WaterReflect::look`, a
//! `WaterLook`): the owner's and the developer's knobs, written whole by `drive_reflection` every
//! frame, so every row is valid on the frames the mirror is off.
//!
//! Each row has one writer. [`look_lanes`] writes every row but `water_rect`, which leaves it 0: the
//! water's screen rectangle is known only after visibility, which runs after the mirror's system,
//! so the render world writes that row itself (`reflect::upload_reflect`). A row whose knob does not
//! exist yet stays 0.

use bevy::prelude::*;

use benilla_world::liquid::{
    LOOK_COMPOSITE, LOOK_DETAIL, LOOK_GRADE, LOOK_LANES, LOOK_QUALITY, LOOK_SHORE, LOOK_SHORE_KIND,
};

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
}

/// The look block as `WaterLook` lays it out. `water_rect` is left 0 for the render world, and
/// the rows no knob fills yet (`surf`, the ripple's statistics, the body) stay 0.
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
    row(LOOK_DETAIL, [look.painted_detail, 0.0, 0.0, 0.0]);
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
        let lanes = look_lanes(&WaterLook::default(), &LookInputs { tier: 1 });
        let row =
            |at: usize| -> [f32; 4] { lanes[at - LOOK_LANES.start..][..4].try_into().unwrap() };
        assert_eq!(row(LOOK_QUALITY), [1.0, 1.0, 0.0, 0.0]);
        let high = look_lanes(&WaterLook::default(), &LookInputs { tier: 2 });
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
        assert_eq!(row(LOOK_WATER_RECT), [0.0; 4], "the render world's row");
        let written = [
            LOOK_QUALITY,
            LOOK_COMPOSITE,
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
}
