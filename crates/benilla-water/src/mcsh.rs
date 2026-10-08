//! **The lighting sun's terrain shadow around the camera** (the ADT chunks' MCSH bits), for the
//! body's light: the window of the world the shadow map covers, which the body rows carry.

use bevy::prelude::*;

/// The window the body's shadow map covers: its centre in Bevy x and z, its half extent in yards,
/// and whether a map is bound. The default binds none, and the body reads every surface as lit.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct WaterShadowWindow {
    centre: Vec2,
    half_extent: f32,
    valid: bool,
}

impl WaterShadowWindow {
    /// The body's `shadow` row: centre x, centre z, half extent, valid (1) or not (0).
    pub(crate) fn lanes(&self) -> [f32; 4] {
        [
            self.centre.x,
            self.centre.y,
            self.half_extent,
            f32::from(u8::from(self.valid)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With no map the row says so: every lane 0.
    #[test]
    fn no_map_is_an_invalid_window() {
        assert_eq!(WaterShadowWindow::default().lanes(), [0.0; 4]);
    }
}
