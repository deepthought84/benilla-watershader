//! The per-frame time-of-day resolve: `Light.dbc` sampled at the camera eye for the game clock
//! (the server's, a debug scrub, or the noon fallback) into [`super::WowLighting`], with the scene
//! fog, the camera-in-WMO interior-fog crossfade and the startup seed.

use bevy::prelude::*;

use crate::dev_state::DebugState;
use crate::terrain_stream::SPAWN_XY;
use crate::view::WorldCamera;
use crate::world_map::CurrentMap;
use benilla_assets::coords::bevy_to_wow;
use benilla_assets::{LockRecover, WorldAssets};
use benilla_formats::{Atmosphere, LightCatalog};

use super::{daynight, quantize_glow, ClockSource, GameClock, LightSampler, WowLighting};

/// The scene-fog stage (`0x6cee30`): the pushed pair `end = min(zone end, farclip)` and
/// `start = frac × end`, unclamped. The storm's negative fraction (Elwynn −0.5) floors the near
/// field at `1 − 1/(1−frac)`, about 33% fog at the eye: the constant storm veil.
fn scene_fog(fog_end_raw: f32, fog_start_frac: f32, farclip: f32) -> (f32, f32) {
    let end = fog_end_raw.min(farclip);
    (fog_start_frac * end, end)
}

/// The camera-in-WMO fog crossfade rate, `[0x8115b0] = 0.25`/s: 4 s in and 4 s out
/// (`0x6cefcb`–`0x6cf051`).
const WMO_FOG_RAMP_PER_SEC: f32 = 0.25;

/// The camera-in-WMO interior crossfade, the reference's `[0xce9bdc]`, its fog target staged at
/// `0x6cef43`–`0x6cef62`: in a WMO interior the scene fog, storm veil included, fades toward the
/// building's MFOG fog over 4 s, and the same `t` is the WMO skybox's slot alpha (`crate::skybox`
/// reads [`Self::t`]). The staged fog latches while the camera leaves, so the fade-out starts from
/// the room's fog. A fog-record switch inside a building snaps the staged target; the reference
/// re-arms a lerp there (`0x6cef92`) whose semantics are untraced.
#[derive(Resource, Default)]
pub struct WmoCrossfade {
    t: f32,
    staged: Option<crate::wmo_portal::WmoFogTarget>,
}

impl WmoCrossfade {
    /// The crossfade weight this frame, `[0, 1]`.
    pub fn t(&self) -> f32 {
        self.t
    }
    /// Advances the ramp by `dt` toward `target` and returns the committed `(color, start, end)`,
    /// staged as `end = min(end, farclip)` and `start = end × start_scalar` (`0x6cef56`).
    fn blend(
        &mut self,
        target: Option<crate::wmo_portal::WmoFogTarget>,
        scene_color: [f32; 3],
        scene_start: f32,
        scene_end: f32,
        farclip: f32,
        dt: f32,
    ) -> ([f32; 3], f32, f32) {
        if target.is_some() {
            self.staged = target;
        }
        let dir = if target.is_some() { 1.0 } else { -1.0 };
        self.t = (self.t + dir * WMO_FOG_RAMP_PER_SEC * dt).clamp(0.0, 1.0);
        let Some(s) = self.staged else {
            return (scene_color, scene_start, scene_end);
        };
        if self.t <= 0.0 {
            self.staged = None;
            return (scene_color, scene_start, scene_end);
        }
        let end = s.end.min(farclip);
        let start = end * s.start_scalar;
        let k = self.t;
        let mut color = scene_color;
        for (c, w) in color.iter_mut().zip(s.color) {
            *c += (w - *c) * k;
        }
        (
            color,
            scene_start + (start - scene_start) * k,
            scene_end + (end - scene_end) * k,
        )
    }
}

/// Loads `Light.dbc` into the [`LightSampler`] and seeds [`WowLighting`] with spawn-zone noon for
/// frame 0; without client data it stays default.
pub(super) fn setup_lighting(
    mut commands: Commands,
    world_assets: Option<ResMut<WorldAssets>>,
    view: Res<crate::view::ViewDistance>,
) {
    let Some(world_assets) = world_assets else {
        return;
    };
    let mut chain = world_assets.chain.lock_recover();
    let catalog = match LightCatalog::load(&mut chain) {
        Ok(cat) => Some(cat),
        Err(e) => {
            warn!("Light.dbc unavailable, using neutral daytime lighting: {e:#}");
            None
        }
    };
    // Noon (half-minute 1440), clear and dry, on the area-light blend, not the single-sphere
    // `pick_light` of `sample_noon`.
    let atmo = catalog
        .as_ref()
        .map(|c| {
            c.sample_blended(
                0,
                [SPAWN_XY.0, SPAWN_XY.1, 83.5],
                1440,
                false,
                benilla_formats::Submersion::Dry,
                false,
            )
        })
        .unwrap_or(Atmosphere::DEFAULT);
    if let Some(cat) = catalog {
        commands.insert_resource(LightSampler(cat));
    }
    info!(
        "Elwynn noon lighting from Light.dbc: sun diffuse {:?}, ambient {:?}",
        atmo.sun_diffuse, atmo.ambient
    );
    let (fog_start, fog_end) = scene_fog(atmo.fog_end, atmo.fog_start_frac, view.farclip);
    commands.insert_resource(WowLighting {
        ambient: atmo.ambient,
        diffuse: atmo.sun_diffuse,
        spec: atmo.sun_color,
        sun_dir: daynight::sun_direction(720.0),
        celestial_dir: daynight::celestial_sun_direction(720.0),
        fog_color: atmo.fog_color,
        fog_start,
        fog_end,
        // The camera starts outdoors, so the interior fog equals the scene's.
        wmo_fog_color: atmo.fog_color,
        wmo_fog_start: fog_start,
        wmo_fog_end: fog_end,
        sky: atmo.sky,
        water_river: atmo.water_river,
        water_ocean: atmo.water_ocean,
        water_river_alpha: atmo.water_river_alpha,
        water_ocean_alpha: atmo.water_ocean_alpha,
        // The seed is the clear, dry noon already.
        water_noon: water_swatches(&atmo),
        glow: quantize_glow(atmo.glow),
        sky_warp: daynight::sky_warp(720.0, atmo.highlight_sky),
        sun_disc_scale: daynight::sun_disc_scale(720.0),
        sun_flare_dn: daynight::sun_flare_dn(720.0),
        moon_dir_white: daynight::moon_direction(720.0),
        moon_disc_scale: daynight::moon_disc_scale(720.0),
        moon_flare_dn: daynight::moon_flare_dn(720.0),
        moon_dir_02: daynight::moon02_state(0.5).0, // day 0, noon
        moon02_disc_scale: daynight::moon02_state(0.5).1,
        star_alpha: daynight::star_alpha(720.0),
        sidn_night: daynight::sidn_night_fraction(720.0),
        celestial_tint: atmo.sun_color,
        cloud_density: atmo.cloud_density,
        cloud_colors: atmo.cloud_colors,
        storm_bcc: 0.0,
        cloud_glow_dir: daynight::celestial_sun_direction(720.0), // noon: the sun
        cloud_glow_track: daynight::cloud_glow_track(720.0),
    });
}

/// The water swatch `(shallow, deep, shallow alpha, deep alpha)` of an atmosphere, river then
/// ocean, as [`WowLighting::water_colors`] reads it.
fn water_swatches(atmo: &Atmosphere) -> [([f32; 3], [f32; 3], f32, f32); 2] {
    let swatch = |rgb: [[f32; 3]; 2], a: [f32; 2]| (rgb[0], rgb[1], a[0], a[1]);
    [
        swatch(atmo.water_river, atmo.water_river_alpha),
        swatch(atmo.water_ocean, atmo.water_ocean_alpha),
    ]
}

/// How far the eye moves before the noon swatch is sampled again, in yards.
const NOON_RESAMPLE_YD: f32 = 1.0;

/// The noon swatch of [`WowLighting::water_colors_noon`] and the map and eye it was sampled at.
pub(super) struct NoonSwatch {
    map: u32,
    at: [f32; 3],
    rows: [([f32; 3], [f32; 3], f32, f32); 2],
}

impl NoonSwatch {
    /// The clear, dry noon swatch at `wow_pos` on `map` (the area blend `setup_lighting` seeds
    /// with), reused while the map is the same and the eye within [`NOON_RESAMPLE_YD`] of it.
    fn rows(
        cache: &mut Option<Self>,
        catalog: Option<&LightCatalog>,
        map: u32,
        wow_pos: [f32; 3],
    ) -> [([f32; 3], [f32; 3], f32, f32); 2] {
        let near = |c: &Self| {
            let d: f32 = (0..3).map(|i| (c.at[i] - wow_pos[i]).powi(2)).sum();
            c.map == map && d < NOON_RESAMPLE_YD * NOON_RESAMPLE_YD
        };
        if let Some(c) = cache.as_ref().filter(|c| near(c)) {
            return c.rows;
        }
        let atmo = catalog.map_or(Atmosphere::DEFAULT, |c| {
            c.sample_blended(
                map,
                wow_pos,
                1440,
                false,
                benilla_formats::Submersion::Dry,
                false,
            )
        });
        let rows = water_swatches(&atmo);
        *cache = Some(Self {
            map,
            at: wow_pos,
            rows,
        });
        rows
    }
}

/// The reference's HSV value-only scale (`0x6d1620`: RGB to HSV `0x7bbc80`, V × f, HSV to RGB
/// `0x7bbd60`), used by the ocean depth ramp. With H and S fixed it scales every channel by `f`,
/// so it is exactly a plain multiply. Deviation: no 8-bit repack (`×255 + 512 >> 14`), because
/// this light lane is float end to end.
fn hsv_value_scale(c: [f32; 3], f: f32) -> [f32; 3] {
    [c[0] * f, c[1] * f, c[2] * f]
}

/// The dry atmosphere while the eye is under water, for the Improved Water preview's view from
/// below (Snell's window, the waterline): the zone's dry sky stops, fog and sun colour, which the
/// underwater `LightParams` slot replaces in [`WowLighting`]; a copy of the resolved ones while dry
/// or with the preview off. The `wet_*` fog is the one the frame is drawn with.
#[derive(Resource, Clone, Copy, Default, PartialEq)]
pub struct DrySky {
    pub sky_zenith: [f32; 3],
    pub sky_horizon: [f32; 3],
    pub fog_color: [f32; 3],
    pub fog_start: f32,
    pub fog_end: f32,
    pub sun_color: [f32; 3],
    pub wet_fog_color: [f32; 3],
    pub wet_fog_start: f32,
    pub wet_fog_end: f32,
}

/// [`DrySky`]: a second `Light.dbc` blend on the dry slot (with the storm lerp) while submerged.
#[allow(clippy::too_many_arguments)]
pub(super) fn update_dry_sky(
    sampler: Option<Res<LightSampler>>,
    current_map: Option<Res<CurrentMap>>,
    cam: Query<&GlobalTransform, With<WorldCamera>>,
    clock: Res<GameClock>,
    eye_liquid: crate::liquid::EyeLiquid,
    viewer: Res<crate::view::Viewer>,
    weather: Option<Res<crate::weather::WeatherState>>,
    view: Res<crate::view::ViewDistance>,
    lighting: Res<WowLighting>,
    mut dry: ResMut<DrySky>,
) {
    let mut next = DrySky {
        sky_zenith: lighting.sky[0],
        sky_horizon: lighting.sky[4],
        fog_color: lighting.fog_color,
        fog_start: lighting.fog_start,
        fog_end: lighting.fog_end,
        sun_color: lighting.spec,
        wet_fog_color: lighting.fog_color,
        wet_fog_start: lighting.fog_start,
        wet_fog_end: lighting.fog_end,
    };
    if eye_liquid.submersion().any() && benilla_assets::preview::any() {
        if let (Some(s), Ok(t)) = (sampler.as_ref(), cam.single()) {
            let wow_pos = bevy_to_wow(t.translation());
            let map = current_map.as_ref().map_or(0, |m| m.0);
            let storm = weather
                .as_ref()
                .map_or(0.0, |w| crate::weather::storm_blend(w.sky_density));
            let dry_slot = benilla_formats::Submersion::Dry;
            let at = clock.minute * 2;
            let clear =
                s.0.sample_blended(map, wow_pos, at, false, dry_slot, viewer.ghost);
            let atmo = if storm > 0.0 {
                let stormy =
                    s.0.sample_blended(map, wow_pos, at, true, dry_slot, viewer.ghost);
                clear.lerp(&stormy, storm)
            } else {
                clear
            };
            let (fog_start, fog_end) = scene_fog(atmo.fog_end, atmo.fog_start_frac, view.farclip);
            next.sky_zenith = atmo.sky[0];
            next.sky_horizon = atmo.sky[4];
            next.fog_color = atmo.fog_color;
            next.fog_start = fog_start;
            next.fog_end = fog_end;
            next.sun_color = atmo.sun_color;
        }
    }
    dry.set_if_neq(next);
}

pub(super) fn update_time_lighting(
    world_time: Res<super::WorldTime>,
    sampler: Option<Res<LightSampler>>,
    debug: Res<DebugState>,
    cam: Query<&GlobalTransform, With<WorldCamera>>,
    current_map: Option<Res<CurrentMap>>,
    eye_liquid: crate::liquid::EyeLiquid,
    viewer: Res<crate::view::Viewer>,
    weather: Option<Res<crate::weather::WeatherState>>,
    view: Res<crate::view::ViewDistance>,
    time: Res<Time>,
    wmo_fog: Res<crate::wmo_portal::CameraWmoFog>,
    mut wmo_ramp: ResMut<WmoCrossfade>,
    mut clock: ResMut<GameClock>,
    mut lighting: ResMut<WowLighting>,
    (mut last_dump, mut last_fog_dump): (Local<Option<u32>>, Local<Option<u32>>),
    mut noon: Local<Option<NoonSwatch>>,
) {
    // `minute` drives the `Light.dbc` sample, the clock and the log; the fractional `minute_f`
    // drives the celestial positions and curves, which would otherwise step once per game minute.
    let (minute, minute_f, source) = if debug.lighting.follow_server_time {
        if world_time.live {
            (world_time.minute, world_time.minute_f, ClockSource::Server)
        } else {
            (
                world_time.minute,
                world_time.minute_f,
                ClockSource::Fallback,
            )
        }
    } else {
        let m = debug.lighting.manual_minute.min(1439);
        (m, m as f32, ClockSource::Manual)
    };
    // moon02's phase clock: the continuous day count, or under the manual scrub the day fraction.
    let day_f: f64 = match debug.lighting.follow_server_time {
        true => world_time.day,
        false => debug.lighting.manual_minute.min(1439) as f64 / 1440.0,
    };
    clock.minute = minute;
    clock.source = source;

    // The area-light blend samples the camera eye, not the character, every frame, as the
    // reference does (`0x6d2d00` reads `DNState+0x18`, stamped from the camera eye at `0x482ea0`).
    // No temporal smoothing, only the blend's spatial falloff; the spawn point stands in until the
    // camera exists.
    let wow_pos = match cam.single() {
        Ok(t) => bevy_to_wow(t.translation()),
        Err(_) => [SPAWN_XY.0, SPAWN_XY.1, 83.5],
    };
    // The map must be the current continent: `Light.dbc` spheres are filtered by continent, and
    // the wrong map falls back to the other continent's global light.
    let map = current_map.as_ref().map_or(0, |m| m.0);
    // Submerged, the underwater `LightParams` (slot 1) replaces the atmosphere for every consumer;
    // the reference switches the active param and draws no overlay.
    let submerged = eye_liquid.submersion();
    // A ghost's atmosphere is `LightParams` slot 4, the death profile (`[0xce9bb0]`), at once.
    let ghost = viewer.ghost;
    // The storm blend: `bcc = min(1, density·4)` (`0x6d4500`) lerps the storm `LightParams` over
    // the clear one, every band at once; a zone without one falls back to its clear param.
    let storm = weather
        .as_ref()
        .map_or(0.0, |w| crate::weather::storm_blend(w.sky_density));
    let atmo = sampler
        .as_ref()
        .map(|s| {
            let clear =
                s.0.sample_blended(map, wow_pos, minute * 2, false, submerged, ghost);
            if storm > 0.0 {
                let stormy =
                    s.0.sample_blended(map, wow_pos, minute * 2, true, submerged, ghost);
                clear.lerp(&stormy, storm)
            } else {
                clear
            }
        })
        .unwrap_or(Atmosphere::DEFAULT);
    let sun_dir = daynight::sun_direction(minute_f);
    let celestial_dir = daynight::celestial_sun_direction(minute_f);

    // Elwynn commits 125/500 clear and −139/278 at full storm (`0x6cee30`): the negative start is
    // the near veil, and the short storm end whites out the middle distance.
    let (fog_start, fog_end) = scene_fog(atmo.fog_end, atmo.fog_start_frac, view.farclip);
    // The interior crossfade is its own triple; the scene fog stays untouched. Submerged, the
    // underwater param owns the fog and the ramp is bypassed, as the reference snaps it for magma
    // and slime inside an MFOG interior (`0x6cef6f`); the reference's MFOG underwater block
    // (`+0x24`, flags 0x100/0x10) is not applied.
    let (wmo_fog_color, wmo_fog_start, wmo_fog_end) = if submerged.any() {
        (atmo.fog_color, fog_start, fog_end)
    } else {
        wmo_ramp.blend(
            wmo_fog.0,
            atmo.fog_color,
            fog_start,
            fog_end,
            view.farclip,
            time.delta_secs(),
        )
    };
    let moon02 = daynight::moon02_state(day_f);
    // The ocean depth ramp, ocean only: ambient (`DNState+0x178`) × `fac1` down to 0.5 (`0x6d28e0`)
    // and diffuse (`+0x174`) × `fac2` down to 0.75 (`0x6d28fe`) over the first 30 yd. The block's
    // fog-colour commit is dead, overwritten from the bands.
    let (fac1, fac2) = submerged
        .ocean_depth_factors(eye_liquid.eye_z())
        .unwrap_or((1.0, 1.0));
    let resolved = WowLighting {
        ambient: hsv_value_scale(atmo.ambient, fac1),
        diffuse: hsv_value_scale(atmo.sun_diffuse, fac2),
        spec: atmo.sun_color, // row 9, the specular colour
        sun_dir,
        celestial_dir,
        fog_color: atmo.fog_color,
        fog_start,
        fog_end,
        wmo_fog_color,
        wmo_fog_start,
        wmo_fog_end,
        sky: atmo.sky,
        // Area-blended like every band (`0x6d30e0` merges all 18 colour rows): a single-sphere pick
        // snaps the tint where a sphere's weight reaches zero.
        water_river: atmo.water_river,
        water_ocean: atmo.water_ocean,
        water_river_alpha: atmo.water_river_alpha,
        water_ocean_alpha: atmo.water_ocean_alpha,
        water_noon: NoonSwatch::rows(&mut noon, sampler.as_ref().map(|s| &s.0), map, wow_pos),
        glow: quantize_glow(atmo.glow),
        sky_warp: daynight::sky_warp(minute_f, atmo.highlight_sky),
        sun_disc_scale: daynight::sun_disc_scale(minute_f),
        sun_flare_dn: daynight::sun_flare_dn(minute_f),
        moon_dir_white: daynight::moon_direction(minute_f),
        moon_disc_scale: daynight::moon_disc_scale(minute_f),
        moon_flare_dn: daynight::moon_flare_dn(minute_f),
        moon_dir_02: moon02.0,
        moon02_disc_scale: moon02.1,
        star_alpha: daynight::star_alpha(minute_f),
        sidn_night: daynight::sidn_night_fraction(minute_f),
        celestial_tint: atmo.sun_color,
        cloud_density: atmo.cloud_density,
        cloud_colors: atmo.cloud_colors,
        // The one `bcc` that also weighted the storm blend above (`0x6d4500`).
        storm_bcc: storm,
        // The sun by day, the white moon by night (`0x6cfb00`).
        cloud_glow_dir: if daynight::cloud_glow_is_sun(minute_f) {
            celestial_dir
        } else {
            daynight::moon_direction(minute_f)
        },
        cloud_glow_track: daynight::cloud_glow_track(minute_f),
    };
    // Assigned only on a change: the deref marks the resource changed, and readers gate on that.
    if *lighting != resolved {
        *lighting = resolved;
    }
    // `WOW_FOG_DUMP=1` prints the committed scene and interior fog and the ramp `t` once a second;
    // `=frame` prints every frame, which a target flipping frame to frame needs.
    static FOG_DUMP: std::sync::OnceLock<Option<std::ffi::OsString>> = std::sync::OnceLock::new();
    if let Some(mode) = FOG_DUMP.get_or_init(|| std::env::var_os("WOW_FOG_DUMP")) {
        let sec = (mode.as_os_str() != "frame").then(|| time.elapsed_secs() as u32);
        if sec.is_none() || *last_fog_dump != sec {
            *last_fog_dump = sec;
            let b = |c: [f32; 3]| c.map(|v| (v * 255.0).round() as i32);
            eprintln!(
                "[fog] t {:.2} target {:?} scene {:?} {:.0}/{:.0} interior {:?} {:.0}/{:.0}",
                wmo_ramp.t,
                wmo_fog.0,
                b(resolved.fog_color),
                resolved.fog_start,
                resolved.fog_end,
                b(resolved.wmo_fog_color),
                resolved.wmo_fog_start,
                resolved.wmo_fog_end,
            );
        }
    }
    // `WOW_LIGHT_DUMP=1` prints the resolved light once per game minute, led by the map and eye
    // position it sampled, which `benilla-extract lightblend <map> <x> <y> <z> <minute>` replays
    // offline; `=frame` prints every frame, for anything positional.
    static LIGHT_DUMP: std::sync::OnceLock<Option<std::ffi::OsString>> = std::sync::OnceLock::new();
    if let Some(mode) = LIGHT_DUMP.get_or_init(|| std::env::var_os("WOW_LIGHT_DUMP")) {
        let key = (mode.as_os_str() != "frame").then_some(minute);
        if key.is_some() && *last_dump == key {
            return;
        }
        *last_dump = key;
        let b = |c: [f32; 3]| [c[0] * 255.0, c[1] * 255.0, c[2] * 255.0].map(|v| v.round() as i32);
        eprintln!(
            // The tail prints the ocean ramp's factors and the eye depth they came from.
            "[light] map {map} at ({:.1}, {:.1}, {:.1}) minute {minute} ({source:?}) ambient {:?} diffuse {:?} spec {:?} sun_dir {:.3} fog {:?} start/end {:.0}/{:.0} submersion {submerged:?} eye_z {:.1} ocean_fac {:.4}/{:.4}",
            wow_pos[0],
            wow_pos[1],
            wow_pos[2],
            b(resolved.ambient),
            b(resolved.diffuse),
            b(resolved.spec),
            resolved.sun_dir,
            b(resolved.fog_color),
            resolved.fog_start,
            resolved.fog_end,
            eye_liquid.eye_z(),
            fac1,
            fac2,
        );
    }
}

/// The clear colour is the row-7 fog colour, so a fully fogged texel at the far plane meets the
/// void behind it without a seam.
pub(super) fn apply_sky_backdrop(
    lighting: Option<Res<WowLighting>>,
    mut clear: ResMut<ClearColor>,
    mut last: Local<Option<[f32; 3]>>,
) {
    let Some(l) = lighting else {
        return;
    };
    if *last == Some(l.fog_color) {
        return;
    }
    *last = Some(l.fog_color);
    // The buffer holds gamma bytes, so the clear writes the DBC value raw (`linear_rgb` converts
    // nothing); the frame's one decode is the FFXGlow combine.
    clear.0 = Color::linear_rgb(l.fog_color[0], l.fog_color[1], l.fog_color[2]);
}

#[cfg(test)]
mod ocean_ramp_tests {
    use super::hsv_value_scale;

    /// A textbook `RGB → HSV → RGB` round trip, the reference's detour written out.
    fn round_trip_v_scale(c: [f32; 3], f: f32) -> [f32; 3] {
        let (r, g, b) = (c[0], c[1], c[2]);
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let v = max;
        let sat = if max == 0.0 { 0.0 } else { (max - min) / max };
        let hue = if max == min {
            0.0
        } else if max == r {
            60.0 * (((g - b) / (max - min)) % 6.0)
        } else if max == g {
            60.0 * ((b - r) / (max - min) + 2.0)
        } else {
            60.0 * ((r - g) / (max - min) + 4.0)
        };
        // V *= f, H and S untouched, as `0x6d1620` does.
        let v = v * f;
        // HSV -> RGB
        let cc = v * sat;
        let x = cc * (1.0 - (((hue / 60.0) % 2.0) - 1.0).abs());
        let m = v - cc;
        let (r1, g1, b1) = match (hue / 60.0).floor() as i32 {
            0 => (cc, x, 0.0),
            1 => (x, cc, 0.0),
            2 => (0.0, cc, x),
            3 => (0.0, x, cc),
            4 => (x, 0.0, cc),
            _ => (cc, 0.0, x),
        };
        [r1 + m, g1 + m, b1 + m]
    }

    /// The value-only scale equals a plain multiply, greys and black (hue undefined) included.
    #[test]
    fn the_hsv_detour_is_a_plain_multiply() {
        let colours = [
            [0.8, 0.4, 0.1],
            [0.1, 0.1, 0.1],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.2, 0.9, 0.55],
            [0.0, 0.3, 0.7],
        ];
        for c in colours {
            for f in [1.0f32, 0.875, 0.75, 0.5] {
                let want = round_trip_v_scale(c, f);
                let got = hsv_value_scale(c, f);
                for i in 0..3 {
                    assert!(
                        (want[i] - got[i]).abs() < 1e-6,
                        "{c:?} × {f}: HSV round trip {want:?} vs multiply {got:?}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wmo_portal::WmoFogTarget;

    /// The 4 s crossfade (`[0x8115b0]` = 0.25/s): the staged fog latches on exit and releases at 0.
    #[test]
    fn wmo_fog_ramp_fades_in_and_out_over_four_seconds() {
        let mut ramp = WmoCrossfade::default();
        let target = WmoFogTarget {
            color: [1.0, 0.5, 0.0],
            end: 194.4,
            start_scalar: 0.25,
        };
        let scene = ([0.2, 0.2, 0.2], -139.0, 278.0);
        // 2 s in: the end is midway between the storm's 278 and the room's 194.4.
        let (_, _, end) = ramp.blend(Some(target), scene.0, scene.1, scene.2, 1100.0, 2.0);
        assert!((ramp.t - 0.5).abs() < 1e-6);
        assert!((end - (278.0 + (194.4 - 278.0) * 0.5)).abs() < 1e-3);
        // 4 s in: the room's fog alone, start = end × scalar.
        let (color, start, end) = ramp.blend(Some(target), scene.0, scene.1, scene.2, 1100.0, 2.0);
        assert!((end - 194.4).abs() < 1e-3);
        assert!((start - 194.4 * 0.25).abs() < 1e-3);
        assert!((color[0] - 1.0).abs() < 1e-6);
        // 2 s out with no target: the latched room fog still blends at half weight.
        let (_, _, end) = ramp.blend(None, scene.0, scene.1, scene.2, 1100.0, 2.0);
        assert!((ramp.t - 0.5).abs() < 1e-6);
        assert!((end - (278.0 + (194.4 - 278.0) * 0.5)).abs() < 1e-3);
        // 4 s out: the scene fog verbatim.
        let (color, start, end) = ramp.blend(None, scene.0, scene.1, scene.2, 1100.0, 2.0);
        assert_eq!((color, start, end), scene);
        let (c2, s2, e2) = ramp.blend(None, scene.0, scene.1, scene.2, 1100.0, 2.0);
        assert_eq!((c2, s2, e2), scene);
        assert!(ramp.staged.is_none(), "latch releases at t = 0");
    }

    /// The resolve at `minute` on the manual clock, clear and dry at the spawn point (no camera);
    /// the resolved light.
    fn resolve_at(app: &mut App, minute: u32) -> WowLighting {
        use bevy::ecs::system::RunSystemOnce;
        {
            let mut debug = app.world_mut().resource_mut::<DebugState>();
            debug.lighting.follow_server_time = false;
            debug.lighting.manual_minute = minute;
        }
        app.world_mut()
            .run_system_once(update_time_lighting)
            .expect("the resolve runs");
        *app.world().resource::<WowLighting>()
    }

    /// The noon swatch is the swatch the resolve paints at noon, clear and dry, and it stays the
    /// noon one at dusk, where the current swatch moves.
    #[test]
    fn the_noon_swatch_is_the_swatch_at_noon() {
        use benilla_formats::LiquidKind;
        let data = benilla_formats::wow_data_or_skip!();
        let mut chain = benilla_formats::open_chain(&data).expect("open the install");
        let catalog = LightCatalog::load(&mut chain).expect("load Light/LightParams/*Band");
        let mut app = App::new();
        app.insert_resource(LightSampler(catalog))
            .init_resource::<super::super::WorldTime>()
            .insert_resource(DebugState::default())
            .init_resource::<crate::view::Viewer>()
            .init_resource::<crate::view::ViewDistance>()
            .init_resource::<Time>()
            .init_resource::<crate::wmo_portal::CameraWmoFog>()
            .init_resource::<WmoCrossfade>()
            .init_resource::<GameClock>()
            .init_resource::<WowLighting>();
        let noon = resolve_at(&mut app, 720);
        for kind in [LiquidKind::Still, LiquidKind::Ocean] {
            assert_eq!(
                noon.water_colors_noon(kind),
                noon.water_colors(kind),
                "{kind:?}"
            );
        }
        let dusk = resolve_at(&mut app, 1180);
        let moved = [LiquidKind::Still, LiquidKind::Ocean]
            .iter()
            .any(|&k| dusk.water_colors(k) != noon.water_colors(k));
        assert!(
            moved,
            "the swatch at 19:40 is the noon one: the control cannot fail"
        );
        for kind in [LiquidKind::Still, LiquidKind::Ocean] {
            assert_eq!(
                dusk.water_colors_noon(kind),
                noon.water_colors(kind),
                "{kind:?}"
            );
        }
    }

    /// The noon sample is reused within a yard of where it was taken on the same map, and taken
    /// again past it or on another map: a sampler-less call returns the default swatch only when
    /// it samples.
    #[test]
    fn the_noon_swatch_is_resampled_past_a_yard_or_on_another_map() {
        let data = benilla_formats::wow_data_or_skip!();
        let mut chain = benilla_formats::open_chain(&data).expect("open the install");
        let catalog = LightCatalog::load(&mut chain).expect("load Light/LightParams/*Band");
        let at = [SPAWN_XY.0, SPAWN_XY.1, 83.5];
        let fallback = water_swatches(&Atmosphere::DEFAULT);
        let mut cache = None;
        let sampled = NoonSwatch::rows(&mut cache, Some(&catalog), 0, at);
        assert_ne!(sampled, fallback, "the control cannot fail");
        let near = [at[0] + 0.6, at[1] - 0.6, at[2]];
        assert_eq!(NoonSwatch::rows(&mut cache, None, 0, near), sampled);
        assert_eq!(NoonSwatch::rows(&mut cache, None, 1, at), fallback);
        NoonSwatch::rows(&mut cache, Some(&catalog), 0, at);
        let far = [at[0] + 0.8, at[1] - 0.8, at[2]];
        assert_eq!(NoonSwatch::rows(&mut cache, None, 0, far), fallback);
    }

    /// A record end past the farclip commits the farclip (`0x6cef43` against `[0xce9b94]`), and
    /// the start scales off the clamped end.
    #[test]
    fn wmo_fog_staging_clamps_end_to_farclip() {
        let mut ramp = WmoCrossfade::default();
        let target = WmoFogTarget {
            color: [1.0; 3],
            end: 444.4,
            start_scalar: 0.25,
        };
        let (_, start, end) = ramp.blend(Some(target), [0.0; 3], 0.0, 200.0, 300.0, 4.0);
        assert!((end - 300.0).abs() < 1e-4);
        assert!((start - 75.0).abs() < 1e-4);
    }
}
