//! The dev state: the per-subsystem toggles the render and gameplay systems read, at
//! player-faithful defaults. The debug panel only edits it, so it stays in a player build.

use bevy::prelude::*;

/// Whether this is a deterministic capture run (`$WOW_CAPTURE`): wall-time variation freezes.
pub fn deterministic_run() -> bool {
    std::env::var("WOW_CAPTURE").is_ok()
}

/// Frames each still-frame input has read as changed since startup: `[camera transform,
/// DebugState, ViewDistance, ExteriorWindows, CameraInteriorClaim, any WmoPortalInstance, any
/// InheritedVisibility]`. The still-frame skips (visibility walk, billboards, emitter gates,
/// doodad hosts) engage only while all are still; `FPS_PROBE`'s `noisy=` prints them per window.
pub static STILL_INPUTS_CHANGED: [std::sync::atomic::AtomicU32; 7] = [
    std::sync::atomic::AtomicU32::new(0),
    std::sync::atomic::AtomicU32::new(0),
    std::sync::atomic::AtomicU32::new(0),
    std::sync::atomic::AtomicU32::new(0),
    std::sync::atomic::AtomicU32::new(0),
    std::sync::atomic::AtomicU32::new(0),
    std::sync::atomic::AtomicU32::new(0),
];

/// `Last`: count this frame's changed still-frame inputs into [`STILL_INPUTS_CHANGED`].
pub(crate) fn count_still_inputs(
    cam: Query<Ref<GlobalTransform>, With<crate::view::WorldCamera>>,
    debug: Res<DebugState>,
    view: Res<crate::view::ViewDistance>,
    windows: Res<crate::wmo_portal::ExteriorWindows>,
    claim: Res<crate::wmo_portal::CameraInteriorClaim>,
    portals: Query<(), Changed<crate::wmo_portal::WmoPortalInstance>>,
    flips: Query<(), Changed<InheritedVisibility>>,
) {
    let changed = [
        cam.iter().any(|t| t.is_changed()),
        debug.is_changed(),
        view.is_changed(),
        windows.is_changed(),
        claim.is_changed(),
        !portals.is_empty(),
        !flips.is_empty(),
    ];
    for (slot, c) in STILL_INPUTS_CHANGED.iter().zip(changed) {
        if c {
            slot.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Root debug state, one section per subsystem.
#[derive(Resource)]
pub struct DebugState {
    /// Panel visible; toggled with the dev chord + `D`. `$WOW_PANEL=1` starts it open, so a
    /// headless capture can shoot it.
    pub open: bool,
    pub models: ModelDebug,
    pub lighting: LightingDebug,
    pub sound: SoundDebug,
    pub weather: WeatherDebug,
    pub water: WaterDebug,
}

/// Water-instrument state. Both overlays are **off by default**: they answer a question about the
/// renderer rather than about the world, and a marker floating over a lake is exactly the kind of
/// thing that quietly ends up in a screenshot.
///
/// `$WOW_PROBE_MARKS=1` arms both at boot, which is how a headless capture gets them into frame —
/// the same reason `$WOW_PANEL` exists. Without it the overlays could only ever be checked by hand
/// in the director's own window, and an overlay that draws in the wrong place is precisely what a
/// capture catches for free.
pub struct WaterDebug {
    /// Draw the reflection probes on the minimap — green where the water is reading the probe,
    /// grey where the slot is placed but has no cube yet.
    pub probe_minimap: bool,
    /// Draw the same probes in the world, as gizmo rings at the capture points.
    pub probe_world: bool,
    /// Turn the reflection probe off: no capture, and the water falls back to the other tiers.
    pub probe_off: bool,
    /// Add the screen-space reflection march over Improved Water; off by default, `$WOW_SSR=1`
    /// arms it at boot.
    pub ssr: bool,
    /// The Improved Water debug view, an id into [`WATER_VIEWS`]; 0 (off) by default,
    /// `$WOW_WATER_VIEW=<name|id>` picks one at boot. Any other value compiles the stylised
    /// pipelines' `WATER_VIEW` variant.
    pub view: u8,
}

impl Default for WaterDebug {
    fn default() -> Self {
        let armed = std::env::var_os("WOW_PROBE_MARKS").is_some();
        Self {
            probe_minimap: armed,
            probe_world: armed,
            probe_off: false,
            ssr: std::env::var("WOW_SSR").as_deref() == Ok("1"),
            view: water_view_from_env(),
        }
    }
}

/// The Improved Water debug views by id (`WaterDebug::view`, `liquid.wgsl`'s `WV_*` consts); 0 is
/// off. A view paints one quantity over the water; `flat` is a mode instead, the whole look on the
/// unrippled normal, and `composite_identity` and `discard` change the linear composite's layer.
/// Those two and the body's `transmittance`, `in_scatter` and `albedo` are drawn only with the
/// linear composite; the gamma look (`$WOW_WATER_COMPOSITE=0`) draws as usual under them.
/// `fresnel` shows the gamma look's reflection weight, or with the composite its reflectance. The
/// grading's `grade`, `refl_ungraded` and `refl_graded` are the composite's too.
pub const WATER_VIEWS: [&str; 25] = [
    "off",
    "mirror1",
    "mirror2",
    "probe",
    "tiers",
    "flat",
    "flow",
    "foam",
    "thickness",
    "fresnel",
    "roughness",
    "roughness_aniso",
    "slope_full",
    "slope_lo",
    "floor_occ",
    "sky_floor",
    "sigma_shape",
    "composite_identity",
    "discard",
    "transmittance",
    "in_scatter",
    "albedo",
    "grade",
    "refl_ungraded",
    "refl_graded",
];

/// A debug view named by id or by name (any case), as `$WOW_WATER_VIEW` takes it.
pub fn parse_water_view(v: &str) -> Option<u8> {
    let v = v.trim();
    if let Ok(id) = v.parse::<u8>() {
        return (usize::from(id) < WATER_VIEWS.len()).then_some(id);
    }
    WATER_VIEWS
        .iter()
        .position(|n| n.eq_ignore_ascii_case(v))
        .and_then(|i| u8::try_from(i).ok())
}

/// `$WOW_WATER_VIEW=<name|id>`, read once with the debug state; an unknown value warns and stays
/// off.
fn water_view_from_env() -> u8 {
    let Ok(v) = std::env::var("WOW_WATER_VIEW") else {
        return 0;
    };
    parse_water_view(&v).unwrap_or_else(|| {
        warn!("WOW_WATER_VIEW={v}: no such view (one of {WATER_VIEWS:?}, or its id)");
        0
    })
}

impl Default for DebugState {
    fn default() -> Self {
        Self {
            open: std::env::var_os("WOW_PANEL").is_some(),
            models: default(),
            lighting: default(),
            sound: default(),
            weather: default(),
            water: default(),
        }
    }
}

/// Weather override through the wire's own `WeatherState::apply` path.
#[derive(Default)]
pub struct WeatherDebug {
    /// Override armed: the scrub below replaces the wire, whose weather is consumed and ignored.
    pub force: bool,
    /// One-shot: set by the panel when the scrub changes, taken by `weather_tick`.
    pub dirty: bool,
    /// Wire weather type (0 fine / 1 rain / 2 snow / 3 sand).
    pub kind: u32,
    /// Wire grade (0..1).
    pub grade: f32,
    /// Apply instantly (the wire's `instant` flag) instead of the ramp.
    pub instant: bool,
}

/// The sound kit probe, played through the real kit path (`sound::kit`); the sound settings are
/// `sound::SoundConfig`.
pub struct SoundDebug {
    /// A `SoundEntries` kit id or name for the "Play kit" probe.
    pub kit_query: String,
    /// One-shot: play `kit_query` through the kit player.
    pub play_kit: bool,
    /// Copies of `kit_query` fired in one frame, the overlap case (many attackers, a group buff):
    /// five of kit 3116 (`HolyProtection`) is a group Fortitude on a full party.
    pub play_copies: u32,
}

impl Default for SoundDebug {
    fn default() -> Self {
        Self {
            // A UI kit with several variations, to exercise the depleting weighted pick.
            kit_query: "igMiniMapZoomIn".into(),
            play_kit: false,
            play_copies: 1,
        }
    }
}

/// Scene lighting controls: `Light.dbc` is sampled at the server game clock unless scrubbed.
pub struct LightingDebug {
    /// Follow the live server game clock; when `false`, `manual_minute` sets the time.
    pub follow_server_time: bool,
    /// Minute of the game day (`0..1440`) while not following the server.
    pub manual_minute: u32,
    /// Disable the `Light.dbc` distance fog; the sky dome's horizon band is not fog and stays.
    pub disable_fog: bool,
    /// Hide the gradient sky dome.
    pub disable_sky_dome: bool,
}

impl Default for LightingDebug {
    fn default() -> Self {
        // `WOW_CLOCK=<minute 0..1439>` arms the manual scrub, to match a reference shot's hour.
        let clock = std::env::var("WOW_CLOCK")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .map(|m| m % 1440);
        Self {
            follow_server_time: clock.is_none(),
            manual_minute: clock.unwrap_or(720), // noon, only used when not following the server
            disable_fog: false,
            disable_sky_dome: false,
        }
    }
}

/// World-model render toggles: visibility per model kind and per blend layer.
pub struct ModelDebug {
    /// Visible flags indexed by [`crate::model_render::kind_index`].
    pub kind_visible: [bool; 4],
    /// Visible flags indexed by [`crate::model_render::blend_index`].
    pub blend_visible: [bool; 5],
    /// WMO portal culling: on is the reference's per-group PVS, off draws every group.
    pub portal_cull: bool,
}

impl Default for ModelDebug {
    fn default() -> Self {
        Self {
            kind_visible: [true; 4],
            blend_visible: [true; 5],
            // `WOW_NOPORTALCULL=1` starts the cull off, for a headless A/B of one viewpoint.
            portal_cull: std::env::var("WOW_NOPORTALCULL").is_err(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_water_view, WATER_VIEWS};

    /// A view is named by id or by name in any case; anything else is no view.
    #[test]
    fn a_water_view_parses_by_name_or_id() {
        assert_eq!(parse_water_view("thickness"), Some(8));
        assert_eq!(parse_water_view(" TIERS "), Some(4));
        assert_eq!(parse_water_view("9"), Some(9));
        assert_eq!(parse_water_view("0"), Some(0));
        assert_eq!(parse_water_view("Roughness_Aniso"), Some(11));
        assert_eq!(parse_water_view("floor_occ"), Some(14));
        assert_eq!(parse_water_view(&WATER_VIEWS.len().to_string()), None);
        assert_eq!(parse_water_view("glass"), None);
    }

    /// The names are the shader's ids: `liquid.wgsl` declares `WV_<NAME>` with each one's index,
    /// so the panel and `$WOW_WATER_VIEW` paint the view they name.
    #[test]
    fn the_water_views_are_the_shaders() {
        let wgsl = include_str!("../../benilla-assets/src/shaders/liquid.wgsl");
        for (id, name) in WATER_VIEWS.iter().enumerate().skip(1) {
            let decl = format!("const WV_{}: u32 = {id}u;", name.to_ascii_uppercase());
            assert!(wgsl.contains(&decl), "liquid.wgsl lost `{decl}`");
        }
        assert_eq!(
            wgsl.matches("const WV_").count(),
            WATER_VIEWS.len() - 1,
            "liquid.wgsl declares a view WATER_VIEWS does not name"
        );
    }
}
