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
    /// Draw the Improved Water surface as the dispersive wave field (on) or the scrolled ripple
    /// layers (off, the default); `$WOW_PREVIEW_SURFACE2=1` turns the wave field on at boot.
    pub surface2: bool,
    /// The Improved Water's transparency, 0 to 2 (the panel's "water transparency" slider, read
    /// each frame): one value for every source of the clear water's transparency,
    /// `benilla_water::body`'s clarity (`0.25·2^t`) and the authored bed floor's share (0 at 0, 0.6
    /// at 1, 1 at 2). 0 is the first Improved body, 2 the very clear look of 3 Nov, 1 the default.
    /// `$WOW_WATER_TRANSPARENCY=<0-2>` sets it at boot, or else `$WOW_WATER_CLARITY=<c>` as
    /// `log2(c / 0.25)` (clamped to 0-2: clarity 0.5 is 1, 1 is 2).
    pub transparency: f32,
    /// How fast the Improved Water closes with depth, 0.25 to 4, 1 the default (the panel's "depth
    /// falloff", read each frame): the absorption and the colour ramp's pace scale together, as if
    /// the water were this many times as deep; < 1 a bed stays visible deeper.
    /// `$WOW_WATER_DEPTH_FALLOFF` sets it at boot.
    pub depth_falloff: f32,
    /// The Improved Water body colour's saturation about its luminance, 0 (grey) to 2 (rich), 1 the
    /// default (the panel's "water saturation"; `$WOW_WATER_SATURATION`). The bed and the
    /// reflection keep theirs.
    pub saturation: f32,
    /// The body colour's brightness, hue kept, 0.5 to 1.5, 1 the default (the panel's "water
    /// brightness"; `$WOW_WATER_BRIGHTNESS`).
    pub brightness: f32,
    /// The Improved Water caustic net's gain, 0 (off) to 3, 1 the default (the panel's "caustics
    /// intensity", read each frame; the one net above and below water).
    /// `$WOW_PREVIEW_CAUSTICS_GAIN` sets it at boot.
    pub caustics: f32,
    /// Fade the Improved Water in over its first third of a yard of column to the bed (the panel's
    /// "soft shoreline", read each frame): the waterline's coverage, wet bed and wet line follow a
    /// soft edge instead of the hard cut where the surface meets the ground. On by default;
    /// `$WOW_WATER_SOFT_SHORE=0` turns it off at boot.
    pub soft_shore: bool,
    /// Take the Improved Water's reflectance per pixel on the resolved wave normal, averaged over
    /// the slope variance the surface no longer resolves, under the gloss's level (the panel's
    /// "lively reflection", read each frame): the reflection varies wave to wave instead of lying
    /// as one coat. On by default; `$WOW_WATER_LIVELY=0` turns it off at boot.
    pub lively: bool,
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
            surface2: matches!(
                std::env::var("WOW_PREVIEW_SURFACE2")
                    .as_deref()
                    .map(str::trim),
                Ok("1") | Ok("on")
            ),
            transparency: water_transparency_from_env(),
            depth_falloff: env_ranged("WOW_WATER_DEPTH_FALLOFF", 0.25, 4.0),
            saturation: env_ranged("WOW_WATER_SATURATION", 0.0, 2.0),
            brightness: env_ranged("WOW_WATER_BRIGHTNESS", 0.5, 1.5),
            caustics: env_ranged("WOW_PREVIEW_CAUSTICS_GAIN", 0.0, 3.0),
            soft_shore: env_switch("WOW_WATER_SOFT_SHORE"),
            lively: env_switch("WOW_WATER_LIVELY"),
            view: water_view_from_env(),
        }
    }
}

/// The transparency's default ([`WaterDebug::transparency`]), `benilla_water::body`'s
/// `TRANSPARENCY_DEFAULT`.
pub const WATER_TRANSPARENCY_DEFAULT: f32 = 1.0;

/// A water knob at boot: `$var` clamped to `lo..=hi`, or 1.
fn env_ranged(var: &str, lo: f32, hi: f32) -> f32 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .map_or(1.0, |v| v.clamp(lo, hi))
}

/// A water switch at boot, on unless `$var` is `0` or `off`.
fn env_switch(var: &str) -> bool {
    !matches!(
        std::env::var(var).as_deref().map(str::trim),
        Ok("0") | Ok("off")
    )
}

/// [`WaterDebug::transparency`] at boot: `$WOW_WATER_TRANSPARENCY`, or `$WOW_WATER_CLARITY` mapped
/// to it, or the default.
fn water_transparency_from_env() -> f32 {
    let num = |var: &str| {
        std::env::var(var)
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite())
    };
    num("WOW_WATER_TRANSPARENCY")
        .or_else(|| {
            num("WOW_WATER_CLARITY")
                .filter(|c| *c > 0.0)
                .map(|c| (c / 0.25).log2())
        })
        .map_or(WATER_TRANSPARENCY_DEFAULT, |t| t.clamp(0.0, 2.0))
}

/// The Improved Water debug views by id (`WaterDebug::view`, `liquid.wgsl`'s `WV_*` consts); 0 is
/// off. A view paints one quantity over the water; `flat` is a mode instead, the whole look on the
/// unrippled normal, and `composite_identity` and `discard` change the linear composite's layer.
/// The body's `transmittance`, `in_scatter` and `albedo` paint the composite's body terms;
/// `fresnel` its reflectance; the grading's `grade`, `refl_ungraded` and `refl_graded` its
/// reflection; `refracted_bed` the bed the composite passes and `refract_reject` the refracted
/// lookup's weight, offset and leak (Medium and High refract; Low shows the unrefracted bed);
/// `mcsh` the lighting sun's terrain shadow at the surface, white lit; `sky_vis` and `sun_vis` the
/// open share of the sky and of the lighting sun the body is lit through (1 off the mirror planes
/// and on Low); `foam_light` foam's illuminant left of the screen's centre against the body's right
/// of it; `fog` the surface's fog, the share that stays in red and the fog colour's green and blue;
/// `body_split` the pixel's split, the reflectance in red, the in-scatter's weight in green and the
/// bed's luma transmittance in blue (the reflection's, the body's and the bed's shares are `r`,
/// `(1 − r)·g` and `(1 − r)·b`).
pub const WATER_VIEWS: [&str; 33] = [
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
    "refracted_bed",
    "refract_reject",
    "mcsh",
    "sky_vis",
    "sun_vis",
    "foam_light",
    "fog",
    "body_split",
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
