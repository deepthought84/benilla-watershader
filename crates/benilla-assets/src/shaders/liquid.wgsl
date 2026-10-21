// Liquid shader, one arm per reference liquid renderer:
//   ADT MCLQ river/ocean (`0x6851b0`/`0x685010`): `ocean0_s.bls`, the depth swatch on stage 0 and
//     the animated sheet on stage 1; the only arm with a depth ramp.
//   WMO MLIQ water (`0x6b62e0` category 0), split on `MOGP.flags & 0x48`: exterior `0x6b6630`
//     binds `MapObjExtWater0.bls`, interior `0x6b6420` is fixed-function and unlit.
//   Magma/slime (`0x6b68f0` WMO, `0x68dca0` ADT): the sheet is the opaque body.
//
// The ADT combine, `Shaders\Pixel\ocean0_s.bls` (0.25 is the program's own `PARAM`):
//   rgb   = primary·colorTex.rgb + detail.rgb + (secondary + 0.25)·detail.a
//   alpha = colorTex.a
// colorTex is the depth swatch off the zone's `Light.dbc` water bands (IntBand 16/17 river/lake,
// 14/15 ocean), rebuilt every frame (`0x680b90`, refill `0x58acd0`); detail is the `lake_a` or
// `ocean_h` frame, near-black RGB and the ripple in alpha, which its authored mips fade with
// distance, so the sampler's mips and anisotropy matter; primary is the lit default white vertex
// (`glColorMaterial`); secondary is the sun sheen.
// The ADT water alpha is the swatch's own: the `0xc7fbc0` LUT texture binds only behind
// `[0xc800ec]` (`0x685244`-`0x685257`), which never holds one (its one store, `0x68c7f8`, is 0).
// Deviation: this is the `specular`/`pixelShaders` = 1 leg the reference install runs; both CVars
// (`0x6886a0`/`0x688712`) default to 0, where water has no program, no specular, a plain ADD
// combine and no blend. An active ARB program bypasses the texture environment.
//
// Culling is off for every kind: all four reference liquid passes disable GL_CULL_FACE. Water
// blends with depth write off; magma and slime are opaque and write depth. Output is raw gamma.

#import bevy_pbr::{
    mesh_functions,
    forward_io::Vertex,
    view_transformations::{
        position_world_to_clip, position_world_to_ndc, direction_world_to_clip, ndc_to_uv,
        frag_coord_to_uv,
        depth_ndc_to_view_z, perspective_camera_near,
    },
    mesh_view_bindings::{view, globals},
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var frames: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var frames_samp: sampler;
// The STYLISED look's ripple map (see `stylised_water` at the bottom of this file, and
// `benilla_world::liquid::ripple`): R/G = a tiling slope field, B = the height it came from.
// Bound always, sampled only under `w.path.y`.
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var ripples: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var ripples_samp: sampler;
// The stylised look's PLANAR REFLECTION: the world as a camera mirrored through the water plane
// saw it (`benilla_world::liquid::reflect`), with **alpha as coverage**.
// The opaque scene in 10 bits, snapshotted before anything transparent drew
// (`benilla_water::scene_color`): the colour behind the water (`water_snapshot_px`), the
// refracted bed (`water_bed`), the floor march's occluder (`snapshot_box`) and the reflection's
// hit (`ssr_read`).
@group(#{MATERIAL_BIND_GROUP}) @binding(111) var scene_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(112) var scene_samp: sampler;
// Must equal `liquid::probe::PROBE_SLOT_MAX`: it is the length of the slot array in the reflection
// buffer, and a mismatch silently reads the wrong rows.
const PROBE_SLOT_MAX: i32 = 16;

// The most probes that may be blended into one fragment. Three is the recommendation and four is the
// ceiling the fixed-size arrays above are built for; the taps actually used come from
// More than this does not make the reflection more correct — averaging cubes
// whose errors point in different directions ghosts rather than converges, which is why Unity caps
// its own blend at two — it makes the HANDOVER smoother, which is the thing worth buying here.
const PROBE_TAPS_MAX: i32 = 3;

@group(#{MATERIAL_BIND_GROUP}) @binding(113) var probe_tex: texture_cube_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(114) var probe_samp: sampler;

// The hierarchical depth the march traverses — see `liquid::hiz` and `shaders/hiz_reduce.wgsl`.
// Each texel is the NEAREST surface in its tile, so a ray in front of it has provably missed
// everything the tile holds.
@group(#{MATERIAL_BIND_GROUP}) @binding(115) var hiz_tex: texture_2d<f32>;
// The same pyramid's FARTHEST-depth twin — what lets a ray pass behind a tile. Read only on the steps
// where the ray is behind a tile's nearest surface; see `liquid::hiz` for why it is a separate image.
@group(#{MATERIAL_BIND_GROUP}) @binding(117) var hiz_far_tex: texture_2d<f32>;
// The sky floor's occlusion over the mirror planes (`benilla_water::water_compute`), a quarter of the
// view's resolution: r the open sky, g and b the body's sky and sun visibility, a 1 where the march
// reached a verdict; then the occluder's radiance times (1 − r) in linear light, and rho_c in a.
// The sky floor (`benilla_water::sky_lut`): the dome and its clouds, linear and unlit, a 64² cube
// whose mip k is blurred to 2^k of its level-0 texel.
@group(#{MATERIAL_BIND_GROUP}) @binding(119) var sky_lut: texture_cube<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(120) var water_occl: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(121) var floor_occ: texture_2d<f32>;
// The lighting sun's terrain shadow around the camera (`benilla_water::mcsh`): the ADT chunks' MCSH
// bytes, 1 shadowed, in a 2048² torus over 32×32 chunks; one lit texel while it is not built.
@group(#{MATERIAL_BIND_GROUP}) @binding(122) var water_shadow: texture_2d<f32>;
#ifdef WATER_SCENE_DEPTH
// The opaque world's linear depth behind each pixel (`liquid::hiz`): view Z in yards, written after
// the opaque pass inside the water's screen rectangle, 0 where nothing was drawn. A texel outside
// the rectangle holds what an earlier frame wrote there, or 0 where none has since the image was
// made (at boot or a resize).
@group(#{MATERIAL_BIND_GROUP}) @binding(116) var scene_depth_tex: texture_2d<f32>;
#endif


@group(#{MATERIAL_BIND_GROUP}) @binding(105) var reflection_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var reflection_samp: sampler;
// The second planar mirror's capture, for a second water height in view; `water_reflect.mirror2`.
@group(#{MATERIAL_BIND_GROUP}) @binding(110) var reflection_tex2: texture_2d<f32>;

// The live wave field (`benilla_world::liquid::ripple_sim`) — a window of simulated water carried
// along with the viewer. R/G = surface slope, B = the foam the disturbance has whipped up.
@group(#{MATERIAL_BIND_GROUP}) @binding(108) var wake_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(109) var wake_samp: sampler;

// One cubemap reflection probe: where its cube was taken from, its share of the blend, and the
// fixed spot it holds. The parallax correction marches the cube's own stored distances.
struct ProbeSlot {
    // xyz = the capture point in world yards; w = 1 when this slot holds a cube FOR ITS CURRENT
    // CELL. Zero while it is being filled, and zero the moment it is reassigned — a cube of
    // somewhere else is still a plausible reflection, which is exactly why the check has to be
    // structural rather than something the eye is trusted to catch.
    at: vec4<f32>,
    // w = this slot's share of the blend (`probe::ProbeSlot::mix`); xyz spare.
    box_min: vec4<f32>,
    // Spare.
    box_max: vec4<f32>,
    // x = the fixed probe spot this slot holds (`benilla_formats::PlanarMap`), -1 for none; yzw
    // spare.
    bind: vec4<f32>,
};

// ---- dome law: begin (shared verbatim by `sky.wgsl`, `liquid.wgsl` and `sky_lut.wgsl`) ------
//
// The whole gradient as ONE function of the colours and a direction, so the water can ask what the
// sky looks like along a reflected ray without reading the colour buffer. Kept textually identical
// in every copy — `sky.rs`'s `the_dome_law_is_shared_verbatim` fails on any drift.

struct DomeColors {
    sky0: vec4<f32>, // zenith (90°)
    sky1: vec4<f32>, // 16.8°
    sky2: vec4<f32>, // 9.8°
    sky3: vec4<f32>, // 3.7°
    sky4: vec4<f32>, // 1.8°
    fog: vec4<f32>,  // horizon (0°) and below: LightIntBand row 7
    warp: vec4<f32>, // x = dawn/dusk warp strength S (0 = off), y = sun azimuth (rad), zw reserved
};

// Glow `g` for a sun-relative azimuth phase, sampled linearly like `0x6d0f50` from the reference's
// six-keyframe wrap-around table at `0xce9af8` (written by `0x6ce210`); 0.125 is the sun bearing.
fn azimuth_glow(phase: f32) -> f32 {
    let p = fract(phase);
    if (p < 0.125) { return mix(0.0, 1.0, (p + 0.125) / 0.25); } // wrap 0.875(g0)→1.125(g1)
    else if (p < 0.375) { return mix(1.0, 0.0, (p - 0.125) / 0.25); }
    else if (p < 0.5) { return mix(0.0, -0.5, (p - 0.375) / 0.125); }
    else if (p < 0.625) { return mix(-0.5, -0.7, (p - 0.5) / 0.125); }
    else if (p < 0.75) { return mix(-0.7, -0.5, (p - 0.625) / 0.125); }
    else if (p < 0.875) { return mix(-0.5, 0.0, (p - 0.75) / 0.125); }
    else { return mix(0.0, 1.0, (p - 0.875) / 0.25); } // wrap 0.875(g0)→1.125(g1)
}

// One mid-ring's warped colour for glow `g` (`0x6d0f50`): `S^2` is the prepass S nested in the
// per-segment S, and 0.7 is the constant at `0x7ffd7c`.
fn warp_one(c: DomeColors, base: vec3<f32>, g: f32, s: f32) -> vec3<f32> {
    let s2 = s * s;
    if (g >= 0.0) {
        return mix(base, c.sky1.rgb, (1.0 - g) * s2);
    }
    return mix(mix(base, c.sky1.rgb, s), c.sky0.rgb, 0.7 * (-g) * s2);
}

// The dome's colour along `dir` (unit, world space) from the camera.
fn dome_color(c: DomeColors, dir: vec3<f32>) -> vec3<f32> {
    let elev = degrees(asin(clamp(dir.y, -1.0, 1.0))); // −90..90, 0 = horizon

    // Dawn/dusk warp (`0x6d0f50`): only the four mid rings, never the apex or the fog rim, and
    // never brighter. The reference bakes it per vertex at 24 azimuth segments and interpolates,
    // hence the lerp of two segments; `+ 0.125` puts the sun bearing at glow phase 0.125.
    var s1 = c.sky1.rgb;
    var s2c = c.sky2.rgb;
    var s3 = c.sky3.rgb;
    var s4 = c.sky4.rgb;
    let warp_s = c.warp.x;
    if (warp_s > 0.0) {
        let az = fract((atan2(dir.z, dir.x) - c.warp.y) / 6.2831853 + 0.125);
        let seg = az * 24.0; // 24 azimuth segments, like the dome vertices
        let s0 = floor(seg);
        let f = seg - s0;
        let g0 = azimuth_glow(s0 / 24.0);
        let g1 = azimuth_glow((s0 + 1.0) / 24.0);
        s1 = mix(warp_one(c, c.sky1.rgb, g0, warp_s), warp_one(c, c.sky1.rgb, g1, warp_s), f);
        s2c = mix(warp_one(c, c.sky2.rgb, g0, warp_s), warp_one(c, c.sky2.rgb, g1, warp_s), f);
        s3 = mix(warp_one(c, c.sky3.rgb, g0, warp_s), warp_one(c, c.sky3.rgb, g1, warp_s), f);
        s4 = mix(warp_one(c, c.sky4.rgb, g0, warp_s), warp_one(c, c.sky4.rgb, g1, warp_s), f);
    }

    // Elevation gradient, linear between rings like the reference's Gouraud-shaded dome.
    var col: vec3<f32>;
    if (elev <= 0.0) {
        col = c.fog.rgb; // horizon and below = fog colour (row 7), unwarped
    } else if (elev < 1.8) {
        col = mix(c.fog.rgb, s4, elev / 1.8);
    } else if (elev < 3.7) {
        col = mix(s4, s3, (elev - 1.8) / (3.7 - 1.8));
    } else if (elev < 9.8) {
        col = mix(s3, s2c, (elev - 3.7) / (9.8 - 3.7));
    } else if (elev < 16.8) {
        col = mix(s2c, s1, (elev - 9.8) / (16.8 - 9.8));
    } else {
        col = mix(s1, c.sky0.rgb, (elev - 16.8) / (90.0 - 16.8)); // warped ring1 to the raw apex
    }
    return col;
}

// ---- dome law: end ---------------------------------------------------------------------------------


// The body's rows of the look block (`WaterLook::body`).
struct WaterBody {
    // rgb the absorption per yard, w the turbidity: river and lake, then the ocean.
    sig_river: vec4<f32>,
    sig_ocean: vec4<f32>,
    // rgb the sky's and the sun's irradiance, linear, chroma-bounded; e_sky.w the shadow floor's
    // share, e_sun.w the irradiance scale (a debug copy).
    e_sky: vec4<f32>,
    e_sun: vec4<f32>,
    // The ramp rate, the band push, the band edge, the lift.
    look: vec4<f32>,
    // The refraction's strength, its largest offset as a share of the view height, the probe's
    // re-fog and the floor occluder's (1 on).
    refr: vec4<f32>,
    // The terrain shadow's window: its centre's x and z and its half extent in yards, valid (1).
    shadow: vec4<f32>,
    // x the sun's share (debug), y the deep stop's ceiling, z the night weight (debug), w spare.
    spare: vec4<f32>,
    // rgb the sky's and the sun's irradiance in the terrain light's colour, for foam and the
    // underside: open, they add up to the terrain's open-sky light; w spare.
    e_sky_foam: vec4<f32>,
    e_sun_foam: vec4<f32>,
};

// The Improved Water preview's rows (`benilla_water`'s `preview`), the last member of
// `WaterReflect`: zero where the preview, or the feature a row feeds, is off. Declared for every
// liquid pipeline, since the buffer has this shape for all of them; read under `WATER_PREVIEW`.
struct WaterPreview {
    // x the feature bits (`benilla_assets::preview`, exact in an f32), y the rough reflection's
    // cone share, z the sea state's override (0: derived from the wind), w spare.
    mask: vec4<f32>,
    // x caustics, y lights, z sparkles, w river foam.
    gains: vec4<f32>,
    // Phases in [0, 1): x the river's, y the weather's, z the wake's; w spare.
    clocks: vec4<f32>,
    // xy the wind's to-direction in world XZ (unit), z its speed in yd/s with the storm in it, w
    // the storm blend.
    wind: vec4<f32>,
    // x rain, y snow, z the sea state 0..1, w the river mist's gain.
    weather: vec4<f32>,
    // xyz the three swell trains' phases as fractions of their periods, w the glint's own sun
    // reach without the flare's envelope (0: `sun.w`).
    swell: vec4<f32>,
    // xy the gust field's offset and zw the slick field's, in their tiles.
    gust: vec4<f32>,
    // xy the capillary octave's drift in its tiles, zw spare.
    cap: vec4<f32>,
    // The five ripple octaves' downwind drifts under `waves`: (d0, d1), (d2, d3), (d4, spare).
    drift: array<vec4<f32>, 3>,
    // Eight fall lines, two rows each.
    falls: array<vec4<f32>, 16>,
    // The swimmers' and rings' rows.
    wake: array<vec4<f32>, 26>,
    // The view from below the surface.
    under: array<vec4<f32>, 16>,
    // Room for shader-only iteration.
    spare: array<vec4<f32>, 8>,
};

// The stylised look's rows (`benilla_water`'s `look`), the last member of `WaterReflect` but the
// preview's: one writer per row, and a row no writer fills reads 0.
struct WaterLook {
    // x the quality tier 0/1/2 (a copy: the code branches on the tier's defs), y the sky floor's
    // occlusion this frame (1 on, with water on screen the floor can serve), z the body's
    // visibility rays traced this frame (1 on Medium and High with plane water on screen); y and
    // z written by the render world; w spare.
    quality: vec4<f32>,
    // x the reflection's grazing ceiling (1 = none), y the glint weight in linear light, z a glow
    // excess forced onto every water pixel, w spare.
    composite: vec4<f32>,
    // The water's screen rectangle: min x, min y, max x, max y in physical pixels of the world
    // target, written by the render world.
    water_rect: vec4<f32>,
    // x the calm surface's roughness α0, y the ripple's anisotropy clamp, z and w where the
    // ripple's feature fade starts and ends, in pixels.
    surf: vec4<f32>,
    // x the reflection grading's strength, y its far share, z its near share, w spare.
    grade: vec4<f32>,
    // x the painted detail's contrast, yzw spare.
    detail: vec4<f32>,
    // The slope variance the ripple octaves lose per axis over the pixel's footprint
    // (`ripple_loss_at`): row i holds every loss and the mips' share at row i's footprint, then at
    // row i + 1's.
    ripple_loss: array<vec4<f32>, 120>,
    // The ripple's mean slope x and z (the 8-bit truncation's bias), its per-axis slope variance,
    // its feature wavelength in texels.
    ripple_mean: vec4<f32>,
    body: WaterBody,
    // The wet line, the wet darkening, the contact foam and the meniscus gains.
    shore: vec4<f32>,
    // The wet line on still water, rapids and ocean; w the shore lap's phase.
    shore_kind: vec4<f32>,
};

struct WaterReflect {
    // x = the mirror plane's world Y; y = strength (0 = no reflection this frame — the look is off,
    // no water is near, or the eye is under the surface); z = UV distortion; w = the plane
    // tolerance a surface must be within to take the image.
    params: vec4<f32>,
    // The sun the PLAYER CAN SEE, which is not `wow_light.light_sun`: xyz = the celestial to-sun
    // direction, w = how much of it gets through (horizon x clouds x terrain — the lens flare's own
    // envelope, republished as `benilla_world::sun::SunVisibility`). Written every frame, including
    // the frames the mirror above is off.
    sun: vec4<f32>,
    // The sky dome's own gradient stops, zenith and horizon (`WowLighting.sky` rows 0 and 4) — the
    // colours the dome overhead is actually drawn with. The underside's reflection reads them
    // ([`underside_sky`]). `w` unused on both.
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    // The wave simulation's window: xy = its lower corner in world XZ yards, z = one over its side
    // length, w = strength — 0 on the reference lane, where the client's own painted splash decals
    // are the wake instead of this.
    sim: vec4<f32>,
    // The white moon, the same pair as `sun`: xyz = the to-moon direction, w = how much of it gets
    // through. Water reflects moonlight as readily as sunlight and the first pass had it reflecting
    // none.
    moon: vec4<f32>,
    /// `x` = the mirror capture's blur radius in capture texels (0 = off —
    /// see [`sample_mirror`]); `y` = the debug view's id (`WaterDebug::view`, read only by the
    /// `WATER_VIEW` variant, [`water_view_id`]); `z` = the march's OWN strength — 0 unless the
    /// debug toggle asks for it, and pointedly independent of `params.y`, which is the mirror's;
    /// `w` spare.
    ///
    /// `x` once carried a march kill switch and was dead weight even then: the
    /// gate the shader actually reads is `z`. The row is full at four lanes and the params buffer
    /// is sized to it, so a dead lane is the cheapest place for the blur to live.
    flags: vec4<f32>,
    // The cubemap probe (`benilla_water`'s `probe::probe_lanes`): xyz spare, the capture points
    // are per-slot in `probes`; w = draw the probe tier's reflection alone.
    probe_at: vec4<f32>,
    // x = strength (0 until every face has been captured, and on the reference lane); y/z spare;
    // w = the probe debug view.
    probe_cfg: vec4<f32>,
    // `probe_box_min`: x = the slot every fragment is pinned to (-1: none), y = the angle a mip-0
    // texel subtends, z = the blur floor, w = the surface ripple scale. `probe_box_max`: x spare,
    // y = the tap count, z = the deepest mip, w spare.
    probe_box_min: vec4<f32>,
    probe_box_max: vec4<f32>,
    // x = how much of the water's ripple the CUBE lookup sees, which nothing here reads; y = the
    // slot the lat-long debug unwraps; z = march the cube's stored distances (2) or run the fixed
    // point (1); w = the fixed point's steps.
    probe_extra: vec4<f32>,
    // One entry per probe slot — see `liquid::probe::probe_slot_lanes`. Fixed at PROBE_SLOT_MAX
    // rather than at the live count, because a WGSL array length is a compile-time constant; the
    // unused tail carries `at.w = 0` and is skipped.
    probes: array<ProbeSlot, PROBE_SLOT_MAX>,
    // The sky dome's own colours (`sky::dome_uniforms`), for the march's sky hits: where the depth
    // it walks holds nothing, the answer is the sky along the reflected ray by the dome's own law —
    // never the colour buffer, which may hold something drawn there that the depth does not.
    // `warp.z` = read the colour buffer for the march's sky hits instead of the dome law (off).
    dome: DomeColors,
    // The march's own switches. `x` = rays may pass BEHIND a tile once they are behind the
    // thickness slab of its farthest surface (see
    // [`ssr_trace`]); `y` = the ripple's shift of the ray origin; `z` = the probe's softening; `w`
    // spare.
    march: vec4<f32>,
    // The second planar mirror: the same four as `params` (plane, strength, distortion,
    // tolerance), zero while it is off.
    mirror2: vec4<f32>,
    // The stylised surface's clock (`benilla_water`'s `clock`): every phase reduced by its own
    // period on the CPU, so none jumps when `globals.time` wraps at the hour. The map offsets, in
    // tiles: the five ripple octaves longest first, then the shore bubbles' warp and two octaves.
    drift: array<vec2<f32>, 8>,
    // x = the flow restart phase, y = the shore swell's, as fractions of their periods; zw spare.
    cycle: vec4<f32>,
    // The stylised look's rows (`benilla_water`'s `look`).
    look: WaterLook,
    // The Improved Water preview's rows (`benilla_water`'s `preview`).
    preview: WaterPreview,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(107) var<storage, read> water_reflect: WaterReflect;

struct LiquidParams {
    // x = fullbright (magma/slime); y = ocean swatch; z = interior fog; w = sun-sheen shininess.
    kind: vec4<f32>,
    // x = renderer (`LiquidPath`): 0 = ADT MCLQ, 1 = WMO exterior, 2 = WMO interior; yzw reserved.
    path: vec4<f32>,
    // x = the frame sheet's mean alpha (the stylised lane's lift); y = frame count; z = scroll flag
    // (WMO magma/slime only, liquid nibbles 6/7); w = clock enable (0 on a deterministic run).
    anim: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<uniform> w: LiquidParams;

// The prefix of the shared global light (`lighting::global_light`); it must match row for row.
struct WowLight {
    light_ambient: vec4<f32>,      // 0  rgb = ambient; w = Mod2x scale
    light_diffuse: vec4<f32>,      // 1  rgb = sun diffuse; w = clamp flag
    light_sun: vec4<f32>,          // 2  xyz = sun travel direction (to-light = −xyz)
    light_spec: vec4<f32>,         // 3  rgb = row-9 specular colour; w = terrain shininess, unread
    fog_color: vec4<f32>,          // 4  rgb = scene fog (block 1, gamma 0..1); w = enable (>0.5)
    fog_params: vec4<f32>,         // 5  x = start yd; y = end yd; w = the farclip wall
    _sh: array<vec4<f32>, 6>,      // 6-11  model SH coefficients, unread here
    _sh_c16: vec4<f32>,            // 12
    water_river: array<vec4<f32>, 2>, // 13-14 river/lake shallow, deep (IntBand 16/17); w = alpha
    water_ocean: array<vec4<f32>, 2>, // 15-16 ocean shallow, deep (IntBand 14/15); w = alpha
    _grade: vec4<f32>,             // 17
    wmo_fog_color: vec4<f32>,      // 18 rgb = interior fog (block 2); w = enable
    wmo_fog_params: vec4<f32>,     // 19 x = start yd; y = end yd
};
@group(#{MATERIAL_BIND_GROUP}) @binding(90) var<storage, read> wow_light: WowLight;

struct LiquidVsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) depth: f32,
    // The sun sheen, per vertex and interpolated, as the reference's fixed-function stage does.
    @location(4) secondary_vtx: vec3<f32>,
    // A WMO interior pool's `MOMT.diffColor`, as its reference vertex carries; white elsewhere.
    @location(5) vcolor: vec4<f32>,
    // Distance from this vertex to the waterline in yards (UV1.y), for the shore foam.
    @location(6) shore: f32,
    // Offset to the nearest point on the waterline, world XZ yards; its length is taken per fragment.
    @location(7) shore_offset: vec2<f32>,
    // `MeshTag` bit 30: the room's per-frame interior-fog gate, the reference's `[0xca7f00]`.
    @location(8) @interpolate(flat) room_fog: u32,
    // The heightfield's smooth normal, world space; zero on a mesh without it.
    @location(9) surface_normal: vec3<f32>,
    // The two reflection tiers' split (`ATTRIBUTE_WOW_PLANAR`): x the mirrors' weight, y z the two
    // probe spots read, w the second's share.
    @location(10) planar: vec4<f32>,
    // The current (`ATTRIBUTE_WOW_FLOW`), world XZ yards a second; zero on still water.
    @location(11) flow: vec2<f32>,
}

// Sun sheen (`secondary`): the Blinn highlight `light_spec.rgb · (N·H)^shininess`.
fn sun_sheen(world_normal: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let n = normalize(world_normal);
    let to_light = -normalize(wow_light.light_sun.xyz);
    // Local viewer, per vertex: the reference sets `GL_LIGHT_MODEL_LOCAL_VIEWER = 1` at `0x59cf89`.
    let to_view = normalize(view.world_position.xyz - world_pos);
    let half_v = normalize(to_light + to_view);
    let ndoth = max(dot(n, half_v), 0.0);
    // No `N·L > 0` specular gate: the sun stays between +20° and +37° (`DayNight::SetDirection`),
    // so N·L on the flat up normal is always > 0. Shininess is water's own (6.0, `[0x8102e8]`) and
    // material specular is white (`SetRenderState(3, 0xffffffff)`). The reference's specular light
    // colour (`CGLight+0x48` to `glLightfv(GL_SPECULAR)`) is not pinned; row 9 stands in for it.
    return wow_light.light_spec.rgb * pow(ndoth, max(w.kind.w, 1.0));
}

// The authored sheet's clock (flip, scroll) and the magma grade's; the stylised water runs on
// `water_reflect.drift` and `cycle` instead.
fn anim_time() -> f32 {
    return w.anim.w * globals.time;
}

// The 24 fps flip, 30 frames over 1.25 s (`0x68aac0`), floored to a whole frame.
fn frame_layer() -> i32 {
    return i32(floor(anim_time() * 24.0) % max(w.anim.y, 1.0));
}

fn apply_scroll(uv: vec2<f32>) -> vec2<f32> {
    // Magma/slime scroll (liquid nibbles 6/7): the reference's stage-0 texture matrix (`0x6b68f0`,
    // pushed at `0x6b6ae3`) translates v by `fmod(t, 10) · 0.1` (rate `[0x801620]`, period
    // `[0x80e5a0]`), continuously; its phase is machine uptime, so only rate and period match.
    // REPEAT wrapping hides the reset.
    return vec2<f32>(uv.x, uv.y + w.anim.z * fract(anim_time() / 10.0));
}

// Planar eye-Z GL_LINEAR fog in gamma space, as terrain.wgsl; GL_FOG defaults on (`0x593bf0`) and
// no liquid pass turns it off. Block 1 (`+0x70/74/78`) is the scene fog (`0x66ff20`); block 2
// (`+0x80/84/88`) is the interior haze, eased toward the MFOG or zone target over about 4 s
// (`0x6cf054`). Only the WMO geometry pass (`0x6b51d9`/`0x6b51ea`) and the WMO liquid pass
// (`0x6b6323` to `0x6b6342`) submit block 2, under the room gate `[0xca7f00]`: `w.kind.z` is its
// static half (`MOGI & 0x48`), `room_fog` the per-frame half.
fn apply_fog(rgb: vec3<f32>, world_pos: vec3<f32>, room_fog: u32) -> vec3<f32> {
    var fog_color = wow_light.fog_color;
    var fog_span = wow_light.fog_params.xy;
    if (w.kind.z > 0.5 && room_fog != 0u) {
        fog_color = wow_light.wmo_fog_color;
        fog_span = wow_light.wmo_fog_params.xy;
    }
    if (fog_color.w <= 0.5) {
        return rgb;
    }
    let eye_z = -(view.view_from_world * vec4<f32>(world_pos, 1.0)).z;
    let denom = max(fog_span.y - fog_span.x, 0.001);
    let factor = clamp((fog_span.y - eye_z) / denom, 0.0, 1.0);
    return mix(fog_color.xyz, rgb, factor);
}

// ── The stylised look (`waterStyle` = 1) ────────────────────────────────────────────────────────
//
// **Not the reference, and it does not pretend to be.** Everything above this line reproduces what
// 1.12 draws; this is an alternative the player picks in Options → Graphics → Water Style, ported
// from the `water-test` sandbox's "Basic" preset. It replaces the *treatment* of the surface, not
// the world's own water data: the zone's `Light.dbc` swatch still supplies both colours, the same
// depth `V` still drives colour and opacity, the same lighting, the same fog — every input the
// world already decides is kept, and only what is done with them changes.
//
// What it does that `ocean0_s.bls` does not:
//
//   * **An animated surface normal.** Three layers of the generated slope map at unrelated scales
//     and drift directions. Three rather than two matters: two scrolling copies of one texture beat
//     against each other at a visible period, and a third at an unrelated scale hides it.
//   * **A real specular off that normal.** The reference's `secondary` is a Blinn highlight on a
//     FLAT plane, so it is a broad wash that the ripple *alpha* modulates. Here the ripples are in
//     the normal itself, so the same highlight breaks into a glitter path — which is what actually
//     reads as water in motion.
//   * **A Fresnel sky mix.** Water reflects almost nothing looking straight down and almost
//     everything at a glancing angle, and that ramp is most of why a lake reads as a lake. There is
//     no environment map on this path, so the sky is stood in for by the scene fog colour — which
//     is the horizon's colour, tracks the zone and the clock, and is already on this buffer.
//   * **Shore foam.** The depth ramp the reference uses for colour doubles as a shoreline: a band
//     that concentrates at the waterline, broken up at two scales by the map's height channel so it
//     reads as surf gathering rather than as an outline drawn around the water.
//
// The wavelengths are in YARDS of world space (the sandbox's metres, taken across at face value —
// the two units differ by less than a ripple).

/// How far out from the waterline the solid part of the line reaches, in yards. Thin on purpose:
/// this is a line drawn where the water meets the land, not a surf zone.
const FOAM_LINE_YARDS: f32 = 0.15;

/// How white the line gets at its strongest. Under 1 so the water's own colour still shows through
/// even at the very edge — the line is meant to be read as foam lying ON water, and a fully opaque
/// white one reads as a stroke drawn around the lake in a paint program.
const FOAM_LINE_LEVEL: f32 = 0.50;

/// How much the line's own width wavers along the shore, as a fraction of that width. Small — the
/// line wants to read as an even edge, and the broken-up look belongs to the bubbles behind it, not
/// to the line itself.
const FOAM_EDGE_WARP: f32 = 0.22;

/// How far past the line the bubbles carry before there is only water, in yards.
const FOAM_BUBBLE_YARDS: f32 = 0.62;

/// **The swell** — how far the waterline runs up and back down the shore, in yards; one breath of it
/// takes the clock's `SHORE_SWELL_S` (`water_reflect.cycle.y`).
///
/// The band was fixed in world space, which is the one thing a shoreline never is: the water's edge
/// is the most obviously *still* part of a frame that is otherwise moving. Advancing and retreating
/// the whole band along its own gradient costs nothing — it is the same distance field, read at a
/// moving threshold — and it is what makes a coast read as tidal rather than painted on.
///
/// Deliberately small against the band's own width: this is a swell lapping, not a tide coming in.
const TAU: f32 = 6.2831855;
const SHORE_RUNUP_YARDS: f32 = 0.16;

/// How far apart two stretches of coast are before they breathe out of step, in yards. Without this
/// every shoreline on the map advances and retreats in unison, which reads as the whole world
/// pulsing rather than as water; a slow phase drift along the coast makes each bay its own.
const SHORE_SWELL_SPREAD: f32 = 70.0;

/// How bright the bubbles are against the solid line.
const FOAM_BUBBLE_LEVEL: f32 = 0.30;

/// Edge length of the generated ripple map, in texels — `benilla_world::liquid::ripple`'s `SIZE`.
/// Needed here to turn its stored central-difference slope into a real `d/d(uv)`; see
/// [`foam_noise`].
const RIPPLE_TEXELS: f32 = 256.0;

/// How far from the waterline the shore band is evaluated at all, in yards.
///
/// **The band was drawn on every water fragment in the world.** Out in open ocean, four hundred
/// yards from any shore, `line` and `bubbles` both resolve to zero — and the surface still paid two
/// texture taps for the edge, three more inside [`foam_noise`], and two `fwidth` chains to arrive
/// at nothing. Five of the roughly thirteen taps a stylised water fragment makes, spent on the
/// overwhelming majority of them for no pixel.
///
/// The reach is the band's own outer extent and is derived from the terms that set it rather than
/// picked, so it cannot drift out of step with them: the line at its highest swell and widest warp,
/// plus the bubble taper behind it, plus a quarter yard for the anti-aliasing that softens the far
/// end. Past it `1 - out` is identically zero and so is the line's smoothstep.
const FOAM_REACH: f32 = (FOAM_LINE_YARDS + SHORE_RUNUP_YARDS) * (1.0 + FOAM_EDGE_WARP)
    + FOAM_BUBBLE_YARDS
    + 0.25;

/// The foam's own colour — spray, so near-white with the faintest cool cast.
const FOAM_COLOR: vec3<f32> = vec3<f32>(0.88, 0.94, 0.96);

/// How far past the trusted band the correction is allowed to keep working, as a multiple of it.
///
/// The reprojection does not stop being right at any particular distance — it degrades, because the
/// lifted sample is placed at the water's own depth rather than at the depth of what is being
/// reflected, and that approximation loosens as the lift grows. So the reflection is carried at
/// full weight across the trusted band and then faded out over another band this much wider, which
/// is long enough that no edge of it is visible as an edge.
const PLANE_FADE: f32 = 2.5;

/// The surface tilt, in radians, up to which the planar capture is trusted completely, and the
/// tilt at which it is trusted not at all.
///
/// **A capture is rendered through a HORIZONTAL mirror.** Its plane error can be corrected — that
/// is the reprojection — but its *orientation* cannot: a surface tilted by `t` reflects its rays
/// `2t` away from where the capture looked, and no amount of moving the sample around an image
/// recovers a direction the image never contained. Elwynn's stream falls about 3.04 yd across a
/// single chunk, roughly 5.2 deg, so its reflected ray is out by better than ten.
///
/// So tilt is the second half of the trust: past a few degrees the planar image is a confidently
/// wrong picture, and the sky mix it fades into is an honest one. The trusted band is generous
/// enough that an ordinary river keeps its reflection, and the limit is where a chute or a
/// cataract stops pretending.
/// **Tightened once the march landed**, and the reason is the march. These were 0.10 and 0.32,
/// chosen when the capture was the only tier there was: fading a sloped surface out then did not
/// hand it to something better, it simply took its reflection away, so the band had to be generous
/// enough to let an ordinary river keep one. Elwynn's stream is 5.22 deg (3.044 yd of relief across
/// one 33.33 yd chunk, measured), which the old band trusted **completely** — at a 10.4 deg ray
/// error.
///
/// Screen-space reflection is exact at any orientation and, measured, is
/// confident over nearly all of that same stream. So the capture no longer has to cover for it, and
/// can be honest about where a horizontal mirror stops describing the surface: 5.22 deg now keeps
/// about 0.71 of it and the rest comes from the tiers that are right.
const SLOPE_TRUSTED: f32 = 0.03; // ~1.7 deg
const SLOPE_LIMIT: f32 = 0.20; // ~11.5 deg

/// How much the sample is *additionally* smeared as trust falls away.
///
/// The reference's own reflections are faint and broken up, and that is precisely why its
/// single-plane errors never read as errors. A badly-fitted surface here does the same: it dims
/// (the trust weight) and it blurs (this), so what survives is a suggestion of a reflection rather
/// than a sharp claim about geometry the capture got wrong.
const DISTORT_LIFT: f32 = 3.0;

/// The **geometric** tilt of the water under this fragment, in radians.
///
/// Taken from screen-space derivatives of the world position rather than from the mesh, because
/// the liquid mesh carries no slope at all — `surface.rs` writes `[0, 1, 0]` into every vertex
/// normal, so `world_normal` is a constant and says nothing about the heightfield. The cross of
/// the two derivatives is the true facet normal and costs nothing extra here.
///
/// Called in uniform control flow, before any branch: derivatives past a per-fragment `if` are
/// undefined, which is the same rule the shore band is written around.
fn surface_tilt(world_pos: vec3<f32>) -> f32 {
    let g = cross(dpdx(world_pos), dpdy(world_pos));
    let len = length(g);
    if (len < 1e-12) {
        return 0.0;
    }
    // Only the magnitude of the tilt matters, so the facet's winding is irrelevant.
    return acos(clamp(abs(g.y / len), 0.0, 1.0));
}

/// The **geometric** normal of the water facet under this fragment — the heightfield's own
/// orientation, with none of the ripple in it.
///
/// Same derivative cross as [`surface_tilt`] and the same uniform-control-flow rule: call it before
/// any per-fragment branch. The winding IS forced here, because this one is a direction rather than
/// a magnitude and a liquid surface is always seen from above.
fn surface_normal(world_pos: vec3<f32>) -> vec3<f32> {
    let g = cross(dpdx(world_pos), dpdy(world_pos));
    let len = length(g);
    if (len < 1e-12) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    let n = g / len;
    return select(-n, n, n.y >= 0.0);
}

/// The mirror capture, read through a **4-tap box blur** — the reference pipeline's own shape.
///
/// Cataclysm's Ultra water does not sample its mirrored capture directly. It renders the mirror
/// into a downscaled target, runs a 4-tap box blur into a second target, and the water shader
/// samples *that* (`/data/scratch/done/water-ultra/UltraWater_Cata_4.3.4_Spec.md` §5.2). This pass
/// had the downscale and not the blur, and the difference is most of why the reflection read as
/// hard-edged debris rather than as a reflection: a capture is a sharp image of the world, and the
/// ripple offset below *displaces* the sample rather than softening it, so every artefact in the
/// capture arrives at full contrast with a crisp edge on it. Blurring is the step that makes a
/// wrong sample read as water instead of as a mistake — and it is also why the reference's own
/// reflections are faint and broken up, which the shore-band note above says in the other
/// direction.
///
/// Four bilinear taps on the diagonals, which is the whole kernel: at `radius` = 1 each tap already
/// averages the 2x2 it lands between, so the four together cover 3x3 with tent weights for the cost
/// of four fetches.
///
/// **The average is premultiplied, and it has to be.** The capture clears to *transparent black*
/// (alpha is coverage — see `benilla_world::liquid::reflect`), so a straight RGBA mean drags colour
/// toward black wherever a tap lands off the drawn geometry, and every silhouette in the reflection
/// would gain a dark fringe exactly where the blur was supposed to soften it. Dividing the summed
/// colour by the summed coverage is the mean of what was actually *drawn*, and leaves the coverage
/// itself to fade the tier out the way it already does.
///
/// `textureSampleLevel` rather than `textureSample`: this is called under a per-fragment branch,
/// where implicit derivatives are undefined, and the capture is a single-mip image so level 0 is
/// the only level there is.
fn sample_mirror(uv: vec2<f32>, radius: f32) -> vec4<f32> {
    if (radius <= 0.0) {
        return textureSampleLevel(reflection_tex, reflection_samp, uv, 0.0);
    }
    let o = radius / vec2<f32>(textureDimensions(reflection_tex));
    var acc = textureSampleLevel(reflection_tex, reflection_samp, uv + vec2<f32>(-o.x, -o.y), 0.0);
    acc += textureSampleLevel(reflection_tex, reflection_samp, uv + vec2<f32>(o.x, -o.y), 0.0);
    acc += textureSampleLevel(reflection_tex, reflection_samp, uv + vec2<f32>(-o.x, o.y), 0.0);
    acc += textureSampleLevel(reflection_tex, reflection_samp, uv + vec2<f32>(o.x, o.y), 0.0);
    acc *= 0.25;
    return vec4<f32>(acc.rgb / max(acc.a, 1.0e-4), acc.a);
}

/// [`sample_mirror`] on the second mirror's capture.
fn sample_mirror2(uv: vec2<f32>, radius: f32) -> vec4<f32> {
    if (radius <= 0.0) {
        return textureSampleLevel(reflection_tex2, reflection_samp, uv, 0.0);
    }
    let o = radius / vec2<f32>(textureDimensions(reflection_tex2));
    var acc = textureSampleLevel(reflection_tex2, reflection_samp, uv + vec2<f32>(-o.x, -o.y), 0.0);
    acc += textureSampleLevel(reflection_tex2, reflection_samp, uv + vec2<f32>(o.x, -o.y), 0.0);
    acc += textureSampleLevel(reflection_tex2, reflection_samp, uv + vec2<f32>(-o.x, o.y), 0.0);
    acc += textureSampleLevel(reflection_tex2, reflection_samp, uv + vec2<f32>(o.x, o.y), 0.0);
    acc *= 0.25;
    return vec4<f32>(acc.rgb / max(acc.a, 1.0e-4), acc.a);
}

/// How much of the planar reflection a surface `err` yards off the capture plane keeps.
///
/// **This replaced a hard cutoff, and the reason is the shot that broke it.** A capture serves one
/// plane exactly and its neighbours approximately (see the reprojection in `stylised_water`), and
/// the first cut simply refused any surface more than `trusted` yards away. But a frame regularly
/// holds several water bodies at once — a roadside pool at 10.07 and the sea at 0.00 is a real
/// Stranglethorn view — and under a threshold exactly one of them could reflect while the other
/// went flat. Which one flipped as the camera moved, because the capture plane is chosen from what
/// covers the screen.
///
/// A falloff serves them all instead: the body the capture was built for is exact, the others are
/// approximate and a little weaker, and nothing pops when the choice changes hands.
fn plane_trust(err: f32, trusted: f32, tilt: f32) -> f32 {
    let by_height = 1.0 - smoothstep(trusted, trusted * (1.0 + PLANE_FADE), err);
    let by_slope = 1.0 - smoothstep(SLOPE_TRUSTED, SLOPE_LIMIT, tilt);
    return by_height * by_slope;
}

/// How hard the ambient ripples tilt the surface ([`water_surface`]): enough to swing the reflected
/// sun off the highlight and back within one wave, which breaks its column into a mottle.
const NORMAL_STRENGTH: f32 = 1.35;

/// The glints' lobe exponent, the water's roughness to the sun: broad enough that the highlight
/// spreads into the ripples as a sheen instead of a small hard disc sitting on the surface, which
/// reads as oil. The lobe's weight is the look's `composite.y` lane.
const SPEC_POWER: f32 = 24.0;

/// The moon's own glitter, relative to the sun's. Moonlight IS sunlight at about a millionth the
/// intensity, but it lands on a scene lit at a millionth too — what actually differs on screen is
/// that the eye is scotopic, so the path reads dimmer, cooler and softer-edged than the sun's, and
/// that is what these three do.
const MOON_SPEC_WEIGHT: f32 = 0.55;
const MOON_SPEC_POWER: f32 = 34.0;
const MOON_SPEC_COLOR: vec3<f32> = vec3<f32>(0.62, 0.70, 0.86);

/// How hard the ripples are carved into the water's own body, as a ± fraction of its colour.
///
/// The body is otherwise a constant, and a constant is exactly what you see when you look STEEPLY
/// down at water close to you: Fresnel goes to nothing at that angle, so the sky mix and the
/// mirrored image both fall away and the swatch is all that is left — a flat sheet of paint under a
/// surface that is visibly moving everywhere else in the frame. That is the near half of "not
/// enough activity in the water"; the far half is the long swell and the sky gradient.
///
/// The tilt is measured along the **sun's compass bearing**, which is fixed all day, rather than
/// along the sun vector itself: at noon the sun is nearly overhead, every facet faces it about
/// equally, and a term built on the full vector would go flat at exactly midday. Taken this way the
/// shading is zero on flat water — it darkens one face of each wave and lightens the other by the
/// same amount, so it carves the surface without changing the zone's colour.
const BODY_RIPPLE_SHADE: f32 = 0.20;

/// How far in from the wave window's edge the simulation is faded out, as a fraction of its side.
/// It matches the absorbing band on the CPU (`ripple_sim::EDGE_ABSORB`), so a wave is already down
/// to nothing by the time this has finished hiding it — the two together are what keep the window
/// from having a visible square edge in the water.
/// How near the waterline the exact per-fragment distance is used instead of the interpolated
/// scalar, in yards, and how far the two may disagree before the scalar is trusted instead. The
/// reach is comfortably past everything the band draws; the trust bound is what keeps a channel's
/// midline ridge from producing a false line. See the use.
/// Over how many yards of water column the surface fades out where something passes through it.
///
/// Without it every rock, pier and hull meets the water on a hard aliased line, because the water's
/// alpha knows nothing about what is behind it; with it the surface closes around the intersection
/// the way real water does. It is the cheapest thing depth buys and probably the most visible.
///
/// **Short on purpose.** This is a fade against an INTERSECTION, not a model of shallow water. At
/// half a yard it stopped being one: a knee-deep lagoon went transparent across its whole floor,
/// because a shallow bar seen at a grazing angle covers a great deal of screen, and the result was
/// a flat pale plateau rather than water. A hand's breadth closes the seam at a rock and leaves
/// water that is merely shallow looking like water.
const SOFT_EDGE_YARDS: f32 = 0.22;

const SHORE_EXACT_REACH: f32 = 3.0;
const SHORE_EXACT_TRUST: f32 = 0.75;

const WAKE_EDGE_FADE: f32 = 0.08;

/// How white a simulated wake's crest goes. Below the shoreline foam's own level: a wake is
/// aerated water, a breaking waterline is foam sitting on top of it.
const WAKE_FOAM_LEVEL: f32 = 0.14;

/// The wake slope at which foam starts, and how fast it arrives after that.
///
/// The floor is high and the level low because the reference is much subtler than it first seems:
/// in the 1.14.2 footage a swimmer's trail is almost entirely a TONAL streak — the water simply a
/// shade different along the track — with white only in a fleck at the body itself. Most of what
/// reads as a wake there is not foam at all, which is why the body-depth term below carries the
/// trail and this only tips the crest nearest the swimmer. Derived from the SLOPE
/// rather than carried in its own channel: a surface only aerates where it is steep, the slope is
/// the quantity with the range to say so (the curvature this first used peaks twenty times smaller
/// than the height and vanished into the bottom of a byte), and taking it from the same two
/// channels that bend the normal means the white always lands on the face the light does.
const WAKE_FOAM_FLOOR: f32 = 0.80;
const WAKE_FOAM_GAIN: f32 = 2.6;

/// How far the wake's own height moves the body along the depth ramp. A trough is less water
/// between the eye and the bed and a crest is more, and reading it that way is what puts a wake on
/// screen when you are looking STEEPLY DOWN at it — the angle at which Fresnel is nothing, the sky
/// mix and the reflection are gone, and the body colour is the only term left.
const WAKE_BODY_DEPTH: f32 = 0.22;

// **What the linear march left behind, because it was expensive to learn.**
//
// This tier used a fixed-cadence march with geometric steps, a bisection and a thickness test, and
// all of it is gone — `ssr_trace` walks a hierarchy now. Three findings from it are not:
//
// * **The sky is a HIT, not a miss.** Reflecting off a surface below the eye sends the ray up and
//   forward, at sky that is on the screen, and the prepass draws no dome so the march read "no
//   depth" and gave up at exactly the pixels whose answer was in the snapshot. Treating it as a hit
//   took `water-noon` from 31.6% to 46.4% of water pixels answered. The hierarchy gets the same
//   thing for free: a cell reducing to zero is a tile of pure sky.
// * **The thickness test has to be asked at the crossing, not at the coarse step**, or it measures
//   the march's overshoot instead of the geometry and rejects solid hillsides in bands. That fix
//   took the same shot from 13.9% to 31.6%. The traversal honours it by construction: the test
//   runs once, at the level-zero texel the walk ended in.
// * **A finer march buys nothing.** 40 steps growing 1.12x against 28 growing 1.22 moved the hit
//   rate by 0.2 points. Sampling more often is not the answer to being blind between samples;
//   not being blind is.

/// The stylised lava look, in three numbers: how dark the crust goes, how hot the molten goes, and
/// how large the plates are in yards.
///
/// **Lava is the one liquid with no reflection to give it structure**, and on the faithful lane it
/// needs none — the reference draws magma as its animated sheet, fogged, and that sheet is the whole
/// surface. Ported unchanged onto the stylised lane it is the only liquid there that reads as a
/// scrolling texture rather than as a material, because every other kind got a normal, a Fresnel mix
/// and a reflection and magma got nothing.
///
/// What it wants instead of a reflection is **temperature**. Real lava is a dark basalt skin broken
/// into slow plates with molten seams between them, and the plates are what make it read as a crust
/// with something underneath rather than as a moving picture. So: a slow two-octave field off the
/// ripple map picks the plates, the sheet's own luminance says which parts are already hot, and the
/// surface is graded between a darkened crust and a molten seam that is pushed ABOVE 1.0 so the
/// glow pass blooms it. Nothing here invents colour — both ends are the zone's own animated texture
/// scaled — which is the same rule the water's tiers follow: change the treatment, never the
/// palette.
///
/// [`LAVA_GLOW`] above 1 is the point and not an accident. The pipeline is HDR with `ffx_glow`
/// downstream, so a seam at 2.6 blooms and a crust at 0.45 does not, and the difference between
/// them is most of what makes the plates read as solid.
const LAVA_CRUST: f32 = 0.45;
const LAVA_GLOW: f32 = 2.6;
/// Plate size in yards, and the finer break laid across it.
const LAVA_PLATE_YD: f32 = 19.0;
const LAVA_BREAK_YD: f32 = 5.5;

/// Foam noise that does not repeat.
///
/// The ripple map is one 256-texel tile and the shader reads it in world yards, so a single tap of
/// it at the bubbles' own scale repeats about every ten inches — close enough together to read as a
/// printed pattern stamped along the shore rather than as foam, which is what the report meant by
/// "the noise is repeating". Three things break it, and it takes all three:
///
/// * **Two taps at scales with no common multiple**, so neither one's period is the pair's.
/// * **On lattices rotated against each other**, so their seams cross instead of stacking.
/// * **Domain-warped** by a third, much coarser tap: the lookups are displaced by a field that
///   itself varies over tens of yards, so what repeats is only the field being used to displace
///   itself. This is the one that does most of the work — a warp of a third of a tile moves the
///   repeat further than the eye can carry it.
///
/// ## Explicit gradients, and why the value brings its own footprint back
///
/// The band this feeds is drawn only within [`FOAM_REACH`] of the waterline, which is per-fragment
/// control flow — and there an implicit-LOD `textureSample` is undefined and naga rejects it, as
/// does any derivative builtin. So the caller takes `d(world xz)/d(pixel)` in uniform flow and
/// hands it down, and every tap here is a `textureSampleGrad`.
///
/// `fw` comes back for the same reason: the threshold that cuts these flecks softens itself by the
/// noise's own screen footprint, and `fwidth` of the result is no longer available to measure it
/// with. It does not have to be. **The map's R/G is the gradient of its B** — that is what it holds
/// and what `ripple_octave` reads it as (`benilla_world::liquid::ripple`) — so the fetches that
/// give the value give its slope, and the footprint is that slope against the pixel's own step.
/// The stored slope is a central difference over two texels of a [`RIPPLE_TEXELS`]-wide map, so
/// `d(b)/d(uv)` is the decoded value times half the map's width.
///
/// The warp's contribution to the gradients is dropped: it varies over tens of yards, where the
/// terms it displaces vary over fractions of one.
struct FoamNoise {
    /// The field, 0..1.
    v: f32,
    /// How much `v` moves across one pixel — `fwidth(v)`, computed rather than sampled.
    fw: f32,
}

fn foam_noise(p: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> FoamNoise {
    // cos/sin of ~33.9 degrees — a turn that is not a neat fraction of one.
    let rot = vec2<f32>(0.8305, 0.5570);
    let r = mat2x2<f32>(rot.x, -rot.y, rot.y, rot.x);
    let warp = textureSampleGrad(
        ripples,
        ripples_samp,
        p * 0.031 + water_reflect.drift[5],
        ddx * 0.031,
        ddy * 0.031,
    ).rg * 2.0 - 1.0;
    let a_dx = ddx * 0.83;
    let a_dy = ddy * 0.83;
    let a = textureSampleGrad(
        ripples,
        ripples_samp,
        p * 0.83 + warp * 0.35 + water_reflect.drift[6],
        a_dx,
        a_dy,
    );
    let b_dx = (r * ddx) * 1.97;
    let b_dy = (r * ddy) * 1.97;
    let b = textureSampleGrad(
        ripples,
        ripples_samp,
        (r * p) * 1.97 + warp * 0.20 + water_reflect.drift[7],
        b_dx,
        b_dy,
    );
    // d(b)/d(uv), out of the same texels — see the header.
    let ga = (a.rg * 2.0 - 1.0) * (RIPPLE_TEXELS * 0.5);
    let gb = (b.rg * 2.0 - 1.0) * (RIPPLE_TEXELS * 0.5);
    let fw_a = abs(dot(ga, a_dx)) + abs(dot(ga, a_dy));
    let fw_b = abs(dot(gb, b_dx)) + abs(dot(gb, b_dy));
    var out: FoamNoise;
    out.v = a.b * 0.55 + b.b * 0.45;
    out.fw = fw_a * 0.55 + fw_b * 0.45;
    return out;
}

// ---- lava ------------------------------------------------------------------------------------
//
// **Outside the stylised region below, and that placement is load-bearing.** Magma and slime draw
// on BOTH lanes through one pipeline (`LiquidKey::FULLBRIGHT`), which carries none of the stylised
// defs, so a lava helper inside the region is missing exactly where magma needs the shader to
// compile — and a liquid shader that fails to compile does not fall back, it drops the draw: every
// magma surface in the zone stops being drawn, and the only sign is one `failed to process shader`
// line in the log.
/// One slow scalar octave of the ripple map, read in world yards.
///
/// `textureSampleLevel` rather than `textureSample`: this is called under the kind branch, and an
/// implicit-LOD fetch there needs a uniformity proof naga will not give it. Level 0 is the honest
/// level anyway — the features are tens of yards across, so there is nothing for a mip to prefilter.
fn lava_field(world_xz: vec2<f32>, wavelength_yd: f32, drift: vec2<f32>, t: f32) -> f32 {
    let uv = world_xz / wavelength_yd + drift * t;
    return textureSampleLevel(ripples, ripples_samp, uv, 0.0).b;
}

/// Grade the kind's animated sheet into crust and molten seam — see [`LAVA_CRUST`].
///
/// Takes the sheet's colour as it comes and only scales it, so a zone that authored dull red magma
/// keeps dull red magma and one that authored orange keeps orange. The drifts are deliberately an
/// order below the water's: lava moves, but it moves like something with a skin on it.
fn stylised_lava(sheet: vec3<f32>, world_xz: vec2<f32>, t: f32) -> vec3<f32> {
    let plates = lava_field(world_xz, LAVA_PLATE_YD, vec2<f32>(0.0041, 0.0024), t);
    let breaks = lava_field(world_xz, LAVA_BREAK_YD, vec2<f32>(-0.0063, 0.0038), t);
    // Thin crust = hot seam. The two octaves are weighted so the plates decide and the breaks only
    // roughen their edges; an even mix reads as noise rather than as plates.
    let crust = saturate(plates * 0.74 + breaks * 0.26);
    let seam = 1.0 - smoothstep(0.34, 0.62, crust);
    // The sheet is already painted with hot and cool regions; take them rather than fight them, so
    // the seams land where the artist put the bright parts and the plates where they did not.
    let lum = dot(sheet, vec3<f32>(0.299, 0.587, 0.114));
    let molten = saturate(max(seam, smoothstep(0.35, 0.85, lum)));
    return sheet * mix(LAVA_CRUST, LAVA_GLOW, molten);
}

#ifdef WATER_STYLISED
#ifdef WATER_PREVIEW
// ---- the Improved Water preview: its feature bits and shared helpers ---------------------------
//
// One def (`WATER_PREVIEW`) on every stylised pipeline while any feature runs; each feature then
// branches on its bit of `preview.mask.x` (`benilla_assets::preview`), a storage read, so the
// branch is uniform and costs no recompile.
const PV_WAVES: u32 = 1u;
const PV_GLINT: u32 = 2u;
const PV_ROUGH: u32 = 4u;
const PV_ROUGH_SKY: u32 = 8u;
const PV_SHORE: u32 = 16u;
const PV_SURF: u32 = 32u;

/// The preview's feature bits this run.
fn pv_bits() -> u32 {
    return u32(max(water_reflect.preview.mask.x, 0.0));
}

/// Whether any of `bits` runs.
fn pv_on(bits: u32) -> bool {
    return (pv_bits() & bits) != 0u;
}

/// Ripple octave `i`'s downwind drift under `waves` (`preview.drift`), in its tiles.
fn pv_drift(i: i32) -> vec2<f32> {
    let r = water_reflect.preview.drift[i / 2];
    return select(r.xy, r.zw, (i & 1) == 1);
}

/// Ripple octave `i`'s drift as the surface reads it: downwind under `waves`, today's otherwise.
fn pv_octave_drift(i: i32) -> vec2<f32> {
    return select(water_reflect.drift[i], pv_drift(i), pv_on(PV_WAVES));
}

/// The sea state, 0..1: the wind's (`preview.weather.z`), or the override the CPU folded in.
fn pv_sea() -> f32 {
    return saturate(water_reflect.preview.weather.z);
}

/// How much of the sun the glint reaches: its own gate where one runs (`preview.swell.w`), the
/// lens flare's (`sun.w`) otherwise.
fn pv_glint_reach() -> f32 {
    let r = water_reflect.preview.swell.w;
    return select(water_reflect.sun.w, r, r > 0.0);
}

/// The rough reflections' share of the physical lobe, 0 with `rough` off.
fn pv_rough_gain() -> f32 {
    return select(0.0, max(water_reflect.preview.mask.y, 0.0), pv_on(PV_ROUGH));
}
#endif // WATER_PREVIEW
// ── The stylised region ────────────────────────────────────────────────────────────────────────
//
// Everything from here to `stylised_water`'s end compiles only into the stylised water pipelines
// (`LiquidKey::composite`, the `WATER_STYLISED` def): the reference and magma pipelines never see it,
// so nothing in it can drop their draws by naming a symbol their defs leave out.

/// A screen-space reflection hit: what was there, and how much to believe it.
struct SsrHit {
    rgb: vec3<f32>,
    conf: f32,
}

#ifdef WATER_SCENE_DEPTH
// ---- what is behind the water -------------------------------------------------------------------

/// The opaque world's view Z behind pixel `px` in yards, from the linear depth; 0 where it is not
/// known (nothing drawn, or outside the image). Valid inside the water's screen rectangle: a pixel
/// outside it reads what an earlier frame wrote there, or 0 where none has since the image was made
/// (at boot or a resize).
fn scene_view_z(px: vec2<i32>) -> f32 {
    let dims = vec2<i32>(textureDimensions(scene_depth_tex));
    if (any(px < vec2<i32>(0)) || any(px >= dims)) {
        return 0.0;
    }
    return textureLoad(scene_depth_tex, px, 0).r;
}

/// The opaque surface behind a water fragment.
struct SceneBehind {
    // False on sky, where nothing was drawn, and on a frame whose depth is not this view's size.
    known: bool,
    // The opaque surface behind this water pixel, world yards.
    bed: vec3<f32>,
    // The view-Z gap from the water to the bed, yards.
    gap_z: f32,
    // The water's height above the bed, yards.
    vertical: f32,
    // The view ray's length through the water, surface to bed, yards.
    path: f32,
};

/// What is behind the water at `world_pos`, drawn at `frag_xy`: the bed on the pixel's own ray,
/// found from the linear depth.
fn scene_behind(world_pos: vec3<f32>, frag_xy: vec2<f32>) -> SceneBehind {
    var s = SceneBehind(false, world_pos, 0.0, 0.0, 0.0);
    let dims = vec2<i32>(textureDimensions(scene_depth_tex));
    if (any(dims != vec2<i32>(view.viewport.zw))) {
        return s;
    }
    let z_s = scene_view_z(vec2<i32>(frag_xy));
    // Nothing drawn behind the pixel is unknown, not infinitely deep: an endless column would run
    // the absorption to the deep row and paint a flat slab.
    if (z_s <= 0.0) {
        return s;
    }
    let eye = view.world_position.xyz;
    let z_w = -(view.view_from_world * vec4<f32>(world_pos, 1.0)).z;
    // Both points lie on this pixel's ray from the eye, and view Z scales linearly along it.
    let k = z_s / max(z_w, 1e-4);
    s.known = true;
    s.bed = eye + (world_pos - eye) * k;
    s.gap_z = max(z_s - z_w, 0.0);
    s.vertical = world_pos.y - s.bed.y;
    s.path = length(world_pos - eye) * max(k - 1.0, 0.0);
    return s;
}

/// The world point at view Z `z` yards on the ray through `uv`, the view's [0, 1] coordinate: the
/// linear depth holds view Z, not the distance along the ray.
fn scene_world_pos(uv: vec2<f32>, z: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let p_view = vec3<f32>(
        ndc.x * z / view.clip_from_view[0][0],
        ndc.y * z / view.clip_from_view[1][1],
        -z,
    );
    return (view.world_from_view * vec4<f32>(p_view, 1.0)).xyz;
}
#endif

// ---- colour transfer ---------------------------------------------------------------------------

/// The water's display gamma: a pure power law, so `enc(a·b) = enc(a)·enc(b)`. The composite's CPU
/// mirror (`liquid_composite`) holds the same value, and a test compares them.
const WATER_GAMMA: f32 = 2.2;
/// Rec. 709 luminance.
const LUMA_709: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

/// Gamma to linear light.
fn water_dec(g: vec3<f32>) -> vec3<f32> {
    return pow(max(g, vec3<f32>(0.0)), vec3<f32>(WATER_GAMMA));
}

/// Linear light to gamma.
fn water_enc(l: vec3<f32>) -> vec3<f32> {
    return pow(max(l, vec3<f32>(0.0)), vec3<f32>(1.0 / WATER_GAMMA));
}

fn luma709(c: vec3<f32>) -> f32 {
    return dot(c, LUMA_709);
}

#ifdef WATER_SCENE_DEPTH
// ---- floor march law: begin ----
//
// The sky floor's occlusion march, byte-identical in `liquid.wgsl` and `water_compute.wgsl` (a
// `benilla_water` test compares the text between these markers). It reads only `view`, the two
// depth pyramids, the scene snapshot and `water_dec`, which both files declare alike, and nothing
// of the screen-space reflection march it is ported from.

/// How thick every surface in the depth buffer is taken to be, in yards: McGuire & Mara's
/// `zThickness`, the screen-space march's value.
const LAW_THICKNESS: f32 = 6.0;
/// How far past a cell wall a ray is placed when it leaves the cell, in level-0 texels: FFX SSSR's
/// `uv_offset`.
const LAW_CROSS: f32 = 0.005;
/// How far short of the far plane the end of a ray's path on screen is read, as a share of the
/// way there: just inside it, where the depth still places the ray.
const LAW_FAR_SHY: f32 = 0.999;
/// The march's hard step bound: the largest tier's budget.
const LAW_MAX_STEPS: i32 = 12;

// The march's lowest level and step budget per tier, one budget for the fragment's march and the
// compute pass's, so the two give one verdict where they crossfade. Low and Medium walk from the
// coarser level 3, whose cells are twice as wide, in 10 steps: from there they reach what High's
// 12 steps from level 2 reach, a pier's deck seen in its reflection among it, so the grading
// pivots on the same thing at every tier. Each step costs about 0.035 ms over a full screen of
// floor-served water at rs2.
#ifdef WATER_TIER_HIGH
const FLOOR_MARCH_LEVEL: i32 = 2;
const FLOOR_MARCH_STEPS: i32 = 12;
#else
#ifdef WATER_TIER_LOW
const FLOOR_MARCH_LEVEL: i32 = 3;
const FLOOR_MARCH_STEPS: i32 = 10;
#else
const FLOOR_MARCH_LEVEL: i32 = 3;
const FLOOR_MARCH_STEPS: i32 = 10;
#endif
#endif

/// Bevy's infinite reverse-Z near plane in yards (`perspective_camera_near`).
fn law_near() -> f32 {
    return view.clip_from_view[3][2];
}

/// A world point in the march's space: `xy` the screen UV (y down), `z` reverse-Z device depth.
fn law_world_to_screen(p: vec3<f32>) -> vec3<f32> {
    let c = view.clip_from_world * vec4<f32>(p, 1.0);
    let ndc = c.xyz / c.w;
    return vec3<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

/// The view Z (negative, yards) of a reverse-Z device depth (`depth_ndc_to_view_z`).
fn law_view_z(ndc_z: f32) -> f32 {
    return -law_near() / ndc_z;
}

fn law_uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

fn law_ndc_to_world(ndc: vec3<f32>) -> vec3<f32> {
    let w = view.world_from_clip * vec4<f32>(ndc, 1.0);
    return w.xyz / w.w;
}

/// Is a ray at device depth `ray` behind the slab of the surface at `depth`, the point
/// [`LAW_THICKNESS`] yards behind it? Depth is `near / distance`, so the slab's back sits at
/// `near·d / (near + T·d)`, asked without a divide. Sky (depth 0) is never passed behind.
fn law_behind_slab(ray: f32, depth: f32, near: f32) -> bool {
    return ray * (near + LAW_THICKNESS * depth) < near * depth;
}

/// How much the geometry at the end of a ray's path on screen covers the ray (device depths: the
/// ray starts at `start` and its path ends at `end`; the frame shows `depth` there): geometry in
/// front of the end, softly by how far in front ([`LAW_THICKNESS`] for about two thirds). Geometry
/// farther than the water point stands on the ray's side of it. Geometry nearer the eye than the
/// water (the avatar, a trunk beside the camera, a deck overhead) covers it only where the ray's
/// own point at that geometry's height (`ray_q`, 0 off the frame) is shut in the frame there
/// (`frame_q`): on or behind what the frame shows, a deck reaching out over the water, and open
/// once it floats [`LAW_THICKNESS`] in front of it, as at the avatar's height over far water. 0
/// for sky and for an end past the far plane.
fn law_cover(start: f32, end: f32, depth: f32, ray_q: f32, frame_q: f32) -> f32 {
    if (end <= 0.0 || depth <= end) {
        return 0.0;
    }
    let c = 1.0 - exp(-(law_view_z(depth) - law_view_z(end)) / LAW_THICKNESS);
    if (depth < start) {
        return c;
    }
    if (ray_q <= 0.0 || frame_q <= 0.0) {
        return 0.0;
    }
    return c * (1.0 - saturate((law_view_z(ray_q) - law_view_z(frame_q)) / LAW_THICKNESS));
}

/// The occluder as the frame shows it, over the walked level's cell: four bilinear taps at
/// ±2^(level − 1) pixels, averaged, gamma.
fn snapshot_box(uv: vec2<f32>, level: i32) -> vec3<f32> {
    let o = exp2(f32(level) - 1.0) / vec2<f32>(textureDimensions(scene_tex));
    return 0.25 * (textureSampleLevel(scene_tex, scene_samp, uv + vec2<f32>(-o.x, -o.y), 0.0).rgb
        + textureSampleLevel(scene_tex, scene_samp, uv + vec2<f32>(o.x, -o.y), 0.0).rgb
        + textureSampleLevel(scene_tex, scene_samp, uv + vec2<f32>(-o.x, o.y), 0.0).rgb
        + textureSampleLevel(scene_tex, scene_samp, uv + vec2<f32>(o.x, o.y), 0.0).rgb);
}

/// The march's verdict: the screen cannot say, the ray reached the sky, or it met something.
const FLOOR_UNKNOWN: u32 = 0u;
const FLOOR_ESCAPED: u32 = 1u;
const FLOOR_HIT: u32 = 2u;

/// Where a floor march ended: its verdict, the screen UV and device depth it reached, how far
/// along the ray that point is from the origin, yards, and for an unknown verdict how much the
/// frame at the ray's end covers it (0 none).
struct FloorMarch {
    cover: f32,
    state: u32,
    uv: vec2<f32>,
    z: f32,
    travelled: f32,
};

/// Walk the ray `origin + t·dir` through the depth pyramids from level `min_level` up, at most
/// `max_steps` cells, and say whether the sky above the water is open along it.
///
/// The screen-space march's traversal (FFX SSSR's `FFX_SSSR_HierarchicalRaymarch` with the
/// farthest-depth pass-behind, always on), coarsened and turned from a colour into a verdict:
/// - the walk never descends below `min_level`, and falling below it is a hit;
/// - a tile of pure sky is not a hit: the ray crosses it and climbs, remembering it was over sky;
/// - leaving the frame through its top over sky, or passing the far plane, has escaped; leaving
///   it anywhere else, or running out of steps, is unknown: the screen cannot say;
/// - a hit is validated against the slab of the `min_level` texel it stopped in;
/// - an unknown verdict reads the frame at the end of the ray's path on screen: where its
///   projection meets the far plane inside the frame (the pixel the eye sees along the reflected
///   direction), else where it leaves the frame through the top. Geometry drawn there above the
///   water and in front of the ray covers it ([`law_cover`]; nearer the eye than the water only
///   where the ray's own point at its height is shut in the frame): the canopy, deck or far shore
///   the walk ran out of steps before reaching. The end moves with the pixel, not with the cells
///   the walk stopped in.
///
/// `travelled` is the distance from the origin to the point the verdict was reached at: the hit,
/// the covering geometry, where the ray left the frame, or where the budget ran out. The reflected
/// thing is at least that far behind the water.
fn floor_march(origin: vec3<f32>, dir: vec3<f32>, min_level: i32, max_steps: i32) -> FloorMarch {
    // The direction as the difference of two projected points; a ray heading back toward the eye
    // is cut to half the distance to the eye plane, so its second point stays in front of it.
    let clip_o = view.clip_from_world * vec4<f32>(origin, 1.0);
    let clip_d = view.clip_from_world * vec4<f32>(dir, 0.0);
    var reach = clip_o.w;
    if (clip_d.w < 0.0) {
        reach = min(reach, 0.5 * clip_o.w / -clip_d.w);
    }
    let o = law_world_to_screen(origin);
    let d = law_world_to_screen(origin + dir * reach) - o;
    let big = 3.4e38;
    let inv_d = vec3<f32>(
        select(big, 1.0 / d.x, d.x != 0.0),
        select(big, 1.0 / d.y, d.y != 0.0),
        select(big, 1.0 / d.z, d.z != 0.0),
    );

    let top = i32(textureNumLevels(hiz_tex)) - 1;
    let lowest = clamp(min_level, 0, top);
    let base_res = vec2<f32>(textureDimensions(hiz_tex, 0));
    let low_res = vec2<f32>(textureDimensions(hiz_tex, lowest));
    let cross = select(vec2<f32>(LAW_CROSS), vec2<f32>(-LAW_CROSS), d.xy < vec2<f32>(0.0)) / base_res;
    let wall = select(vec2<f32>(1.0), vec2<f32>(0.0), d.xy < vec2<f32>(0.0));

    // Step off the origin's own cell at the lowest level first, so the walk does not open by
    // meeting it.
    var t: f32;
    {
        let planes = (floor(o.xy * low_res) + wall) / low_res + cross;
        let tw = (planes - o.xy) * inv_d.xy;
        t = min(tw.x, tw.y);
    }
    var p = o + t * d;
    let near_plane = law_near();
    // Where the ray leaves the frame: the point an off-frame verdict is reached at, exactly rather
    // than at the cell wall the walk had reached beyond it.
    let t_out = min(
        select((1.0 - o.x) * inv_d.x, -o.x * inv_d.x, d.x < 0.0),
        select((1.0 - o.y) * inv_d.y, -o.y * inv_d.y, d.y < 0.0),
    );

    var level = lowest;
    var last_sky = false;
    var state = FLOOR_UNKNOWN;
    var decided = false;
    let steps = min(max_steps, LAW_MAX_STEPS);
    for (var i = 0; i < steps && level >= lowest; i = i + 1) {
        if (any(p.xy < vec2<f32>(0.0)) || any(p.xy >= vec2<f32>(1.0))) {
            // Off the frame: through its top over sky the sky is open; elsewhere nobody knows.
            state = select(FLOOR_UNKNOWN, FLOOR_ESCAPED, p.y < 0.0 && last_sky);
            p = o + min(t, t_out) * d;
            decided = true;
            break;
        }
        if (p.z <= 0.0) {
            state = FLOOR_ESCAPED; // past the far plane
            decided = true;
            break;
        }
        let res = vec2<f32>(textureDimensions(hiz_tex, level));
        let cell = floor(p.xy * res);
        let surface = textureLoad(hiz_tex, vec2<i32>(cell), level).r;
        let planes = vec3<f32>((cell + wall) / res + cross, surface);
        var tp = (planes - o) * inv_d;
        tp.z = select(big, tp.z, d.z < 0.0);
        let t_wall = min(tp.x, tp.y);
        // A tile of pure sky (the pyramid holds the nearest depth, and sky is 0) holds nothing to
        // meet: cross it and climb.
        let sky = surface <= 0.0;
        last_sky = sky;
        let t_min = min(t_wall, tp.z);
        // Larger depth is nearer under reverse-Z: the ray is in front of everything in the tile.
        let in_front = surface < p.z;
        // Or behind everything in it: behind the slab of the tile's farthest surface for its whole
        // way across, read only for a ray already past the nearest's slab. At the lowest level a
        // texel may hold two layers, and a ray between them walks on.
        var behind = false;
        if (!in_front && !sky) {
            let p_exit_z = o.z + t_wall * d.z;
            let seg_near = max(p.z, p_exit_z);
            let seg_far = min(p.z, p_exit_z);
            if (law_behind_slab(seg_near, surface, near_plane)) {
                let farthest = textureLoad(hiz_far_tex, vec2<i32>(cell), level).r;
                let behind_all = law_behind_slab(seg_near, farthest, near_plane);
                let between = level == lowest && seg_far > farthest;
                behind = behind_all || between;
            }
        }
        let skipped = sky || (t_min != tp.z && in_front) || behind;
        t = select(select(select(t, t_wall, behind), t_min, in_front), t_wall, sky);
        p = o + t * d;
        level = select(level - 1, min(level + 1, top), skipped);
    }
    if (!decided && level < lowest) {
        // Fell below the lowest level: a hit, if the ray stopped inside the slab of the surface
        // its texel holds (the nearest, or past that slab the farthest).
        if (all(p.xy >= vec2<f32>(0.0)) && all(p.xy < vec2<f32>(1.0))) {
            let texel = vec2<i32>(floor(p.xy * low_res));
            let near0 = textureLoad(hiz_tex, texel, lowest).r;
            var gap = law_view_z(near0) - law_view_z(p.z);
            if (gap > LAW_THICKNESS) {
                let far0 = textureLoad(hiz_far_tex, texel, lowest).r;
                gap = law_view_z(far0) - law_view_z(p.z);
            }
            state = select(FLOOR_UNKNOWN, FLOOR_HIT, abs(gap) <= LAW_THICKNESS);
        }
    }
    var cover = 0.0;
    if (state == FLOOR_UNKNOWN) {
        let t_far = select(big, -o.z * inv_d.z, d.z < 0.0);
        let to_far = t_far < t_out;
        if (to_far || (d.y < 0.0 && -o.y * inv_d.y <= t_out)) {
            let e = o + select(t_out, LAW_FAR_SHY * t_far, to_far) * d;
            let e_uv = clamp(e.xy, vec2<f32>(0.0), vec2<f32>(1.0) - 0.5 / base_res);
            let e_depth = textureLoad(hiz_tex, vec2<i32>(e_uv * base_res), 0).r;
            if (e.z > 0.0 && e_depth > e.z) {
                let at_e = law_ndc_to_world(vec3<f32>(law_uv_to_ndc(e_uv), e_depth));
                if (at_e.y > origin.y) {
                    // The ray's point at the geometry's height, and the frame there.
                    var ray_q = 0.0;
                    var frame_q = 0.0;
                    if (e_depth >= o.z && dir.y > 0.0) {
                        let q = law_world_to_screen(origin + dir * ((at_e.y - origin.y) / dir.y));
                        if (all(q.xy >= vec2<f32>(0.0)) && all(q.xy < vec2<f32>(1.0))) {
                            ray_q = q.z;
                            frame_q = textureLoad(hiz_tex, vec2<i32>(q.xy * base_res), 0).r;
                        }
                    }
                    cover = law_cover(o.z, e.z, e_depth, ray_q, frame_q);
                    if (cover > 0.0) {
                        p = vec3<f32>(e_uv, e_depth);
                    }
                }
            }
        }
    }
    // Past the far plane the depth cannot place the point: as far as it reaches.
    let at = law_ndc_to_world(vec3<f32>(law_uv_to_ndc(p.xy), max(p.z, 1e-6)));
    return FloorMarch(cover, state, p.xy, p.z, distance(origin, at));
}

/// The sky floor's occlusion at one point: `rgb` the occluder's radiance times `1 − v` (linear),
/// `v` how much of the sky is open, `rho_c` how far behind the water the reflected thing lies
/// ([`floor_rho_c`]: 0 at a contact, toward 1 far off), `known` 1 where the march reached a
/// verdict.
struct FloorOcc {
    rgb: vec3<f32>,
    v: f32,
    rho_c: f32,
    known: f32,
};

/// The reach, in yards behind the water, at which a reflected thing counts as half far
/// ([`floor_rho_c`]): a grazing ray climbs 8-26 yd to a pier's leg or a deck's underside, and the
/// far shore lies hundreds of yards off.
const FLOOR_CONTACT_REACH: f32 = 64.0;

/// How far behind the water the reflected thing lies, as the grading reads it: `D²/(D² + R²)` at
/// the distance `D` from the reflecting point and [`FLOOR_CONTACT_REACH`] `R`, whatever the eye's
/// distance, so a post or a deck the ray meets within 25 yd reads under 0.14 and the far shore
/// 300 yd off 0.96.
fn floor_rho_c(travelled: f32) -> f32 {
    let d2 = travelled * travelled;
    return d2 / (d2 + FLOOR_CONTACT_REACH * FLOOR_CONTACT_REACH);
}

/// The march's verdict as the floor takes it. `rho_c` ([`floor_rho_c`]) is taken at the distance
/// `D` the verdict was reached: a hit at `D`, an escape or an unknown past it, all say the
/// reflected thing is at least `D` behind the water, so it is continuous from a contact to the far
/// field. The occluder's colour is read at the hit shifted by `tap_shift`, the ripple's slide of
/// the lookup point on screen. An unknown verdict the frame's end covers is that share occluded
/// and known, its colour the frame's there, read at full resolution.
fn floor_from_march(m: FloorMarch, min_level: i32, tap_shift: vec2<f32>) -> FloorOcc {
    let rho = floor_rho_c(m.travelled);
    if (m.state == FLOOR_HIT) {
        return FloorOcc(water_dec(snapshot_box(m.uv + tap_shift, min_level)), 0.0, rho, 1.0);
    }
    if (m.state == FLOOR_ESCAPED) {
        return FloorOcc(vec3<f32>(0.0), 1.0, rho, 1.0);
    }
    if (m.cover > 0.0) {
        let c = m.cover;
        return FloorOcc(water_dec(snapshot_box(m.uv + tap_shift, 0)) * c, 1.0 - c, rho, c);
    }
    return FloorOcc(vec3<f32>(0.0), 1.0, rho, 0.0);
}
// ---- floor march law: end ----
#endif

/// The positive `t` where the ray `o + t*d` (with `o` measured from the capture point) is `dist`
/// away from that point — one step of the depth correction, in closed form.
///
/// `|o + t*d|^2 = dist^2` with `|d| = 1` is `t^2 + 2t(o.d) + |o|^2 - dist^2 = 0`. A negative
/// discriminant means the ray never reaches that distance, which happens at silhouettes where the
/// sampled distance belongs to something the ray passes beside; the nearest approach is the least
/// wrong answer available and keeps the iteration from producing a NaN.
fn probe_t_for(o: vec3<f32>, d: vec3<f32>, dist: f32) -> f32 {
    let b = dot(o, d);
    let disc = b * b - dot(o, o) + dist * dist;
    if (disc < 0.0) {
        return max(-b, 0.0);
    }
    return max(-b + sqrt(disc), 0.0);
}

/// Correct the reflected direction against the cube's own stored DISTANCES rather than against a
/// proxy shape.
///
/// **This is what removes the tuned box.** A box proxy asserts that the whole world stands on one
/// rectangle, so it can be right for the far treeline (which is what a grazing ray hits) or for the
/// cottage ten yards past the bank, but not both — and whichever distance the constant is set to is
/// some other lake's error. The capture already knows where everything actually is: `probe_face.wgsl`
/// writes the distance from the capture point into the cube's alpha, so the geometry is per-texel
/// and per-direction instead of one number fitted at a development site.
///
/// The iteration is the obvious fixed point. Guess a direction, ask the cube how far away the
/// geometry is that way, move along the reflected ray to exactly that distance, and re-derive the
/// direction from the capture point. Each round makes the direction and the distance more nearly
/// agree; four is past the point where the picture changes at these face sizes.
fn probe_dir_depth(p: vec3<f32>, d: vec3<f32>, c: vec3<f32>, idx: i32, steps: i32) -> vec3<f32> {
    let o = p - c;
    var dir = d;
    for (var k = 0; k < steps; k = k + 1) {
        let dist = textureSampleLevel(probe_tex, probe_samp, dir, idx, 0.0).a;
        // Nothing drawn this way — sky, or past the far clip. Sky IS at infinity, and the
        // uncorrected direction is exactly the right answer for it.
        if (dist > 9000.0) {
            return d;
        }
        dir = normalize(o + probe_t_for(o, d, dist) * d);
    }
    return dir;
}

/// Steps of [`probe_dir_march`]'s search, the nearest and furthest it looks along the reflected
/// ray (yards), and its bisection steps.
const PROBE_MARCH_STEPS: i32 = 16;
const PROBE_MARCH_NEAR: f32 = 0.5;
const PROBE_MARCH_FAR: f32 = 400.0;
const PROBE_MARCH_REFINE: i32 = 4;

/// Correct the reflected direction by **marching the reflected ray** against the cube's stored
/// distances: step outward from the water point, and at each point ask the cube how far the
/// geometry is in that point's direction from the capture point; the first point at or just past
/// that distance is the hit, refined by bisection. Sky is the answer only when the whole ray passes
/// nothing.
///
/// The fixed point [`probe_dir_depth`] starts from the uncorrected direction and gives up if that
/// sees sky — and from the capture point, the direction to a crown's reflection often passes above
/// the crown, so tree tops read as sky and reflected trees came out short.
///
/// A point further than `dt` behind the stored surface is the ray passing behind something the
/// probe sees in front of it, not a hit, and the march goes on.
fn probe_dir_march(p: vec3<f32>, d: vec3<f32>, c: vec3<f32>, idx: i32) -> vec3<f32> {
    let o = p - c;
    let g = pow(PROBE_MARCH_FAR / PROBE_MARCH_NEAR, 1.0 / f32(PROBE_MARCH_STEPS - 1));
    var t_prev = 0.0;
    var t = PROBE_MARCH_NEAR;
    for (var k = 0; k < PROBE_MARCH_STEPS; k = k + 1) {
        let q = o + t * d;
        let dist = textureSampleLevel(probe_tex, probe_samp, q, idx, 0.0).a;
        let over = length(q) - dist;
        if (dist < 9000.0 && over >= 0.0 && over <= max(t - t_prev, 1.0)) {
            var lo = t_prev;
            var hi = t;
            for (var j = 0; j < PROBE_MARCH_REFINE; j = j + 1) {
                let mid = 0.5 * (lo + hi);
                let qm = o + mid * d;
                if (length(qm) >= textureSampleLevel(probe_tex, probe_samp, qm, idx, 0.0).a) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            return normalize(o + hi * d);
        }
        t_prev = t;
        t = t * g;
    }
    return d;
}

/// **How blurred this probe should be read, derived rather than dialled.**
///
/// The probe no longer tries to put reflections in the right place — a long measurement pass
/// established that a parallax-corrected cube's drift away from its capture point is inherent, and
/// that a sharper march, a bigger face and a world-space landing consensus each moved it barely or
/// not at all. What is left is to stop the error being READABLE, which is what the prefilter chain
/// in `probe_filter.wgsl` is for and what this indexes.
///
/// The amount of blur is not a taste setting. Standing `d_probe` from the capture point, geometry
/// `d_geom` away reflects displaced by roughly `d_probe / d_geom` radians — small angles, and the
/// ratio is the whole of it. A mip-0 texel subtends `(pi/2) / face` radians, and each level down
/// doubles that. So the level whose texel is the size of the error is
/// `log2((d_probe / d_geom) / texel0)`, and at that level the mistake is smaller than the thing it
/// is written on, which is the definition of not being able to see it.
///
/// That derivation is why there is no tuned constant here and why it needs none at the next lake:
/// both distances are measured per fragment, and the texel angle is a property of the cube. The
/// FLOOR is the one authored number, and it is a look decision rather than a correction — a probe
/// read perfectly sharp invites the comparison with the planar mirror that it loses, so the tier is
/// never allowed to be a mirror even standing on top of a probe.
fn probe_lod(d_probe: f32, d_geom: f32) -> f32 {
    let texel0 = water_reflect.probe_box_min.y;
    let lo = water_reflect.probe_box_min.z;
    let hi = water_reflect.probe_box_max.z;
    if (texel0 <= 0.0) {
        return lo;
    }
    // `d_geom` is `PROBE_NO_HIT` for sky, which is correct and needs no special case: sky at ten
    // thousand yards has no parallax to hide and the ratio drives the level to the floor by itself.
    let err = d_probe / max(d_geom, 1.0);
    return clamp(lo + log2(max(err / texel0, 1.0)), lo, hi);
}

/// One probe's contribution: correct the reflected direction against THAT probe's stored distances,
/// then read its own slice of the cube array. `idx` is the slot, which is also the array index —
/// the capture writes six layers starting at `slot * 6`, and a cube-array lookup addresses them as
/// one cube.
fn sample_probe(sl: ProbeSlot, idx: i32, p: vec3<f32>, r: vec3<f32>, steps: i32) -> vec3<f32> {
    // `probe_extra.z` picks the correction against the cube's stored distances: the march by
    // default, the fixed point otherwise.
    var dir: vec3<f32>;
    if (water_reflect.probe_extra.z > 1.5) {
        dir = probe_dir_march(p, r, sl.at.xyz, idx);
    } else {
        dir = probe_dir_depth(p, r, sl.at.xyz, idx, steps);
    }
    // Level 0 for the DISTANCE and a chosen level for the colour. The chain's alpha carries the
    // centre tap's distance rather than a blurred one, but reading it at level 0 keeps the geometry
    // exact regardless of how soft the picture is being read.
    let d_geom = textureSampleLevel(probe_tex, probe_samp, dir, idx, 0.0).a;
    let lod = probe_lod(distance(p, sl.at.xyz), d_geom);
    // The fog the face camera laid on this direction and the fog the mirror would, taken before
    // the colour's reads so only the two shares are held across them.
    let shares = probe_fog_shares(dir, d_geom, sl.at.xyz, p);
    let centre = textureSampleLevel(probe_tex, probe_samp, dir, idx, lod).rgb;
    // A small softening (`march.z`, in cube texels): four more reads a fixed
    // angle around the direction, averaged with it, so the capture's texels do not read as pixels
    // on the water. Either way the colour leaves fogged as the mirror fogs ([`probe_refog`]).
    let spread = water_reflect.march.z * water_reflect.probe_box_min.y;
    if (spread <= 0.0) {
        return probe_refog(centre, shares);
    }
    let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(dir.y) > 0.9);
    let t1 = normalize(cross(dir, helper)) * spread;
    let t2 = cross(dir, t1);
    var sum = centre;
    sum = sum + textureSampleLevel(probe_tex, probe_samp, dir + t1, idx, lod).rgb;
    sum = sum + textureSampleLevel(probe_tex, probe_samp, dir - t1, idx, lod).rgb;
    sum = sum + textureSampleLevel(probe_tex, probe_samp, dir + t2, idx, lod).rgb;
    sum = sum + textureSampleLevel(probe_tex, probe_samp, dir - t2, idx, lod).rgb;
    return probe_refog(sum * 0.2, shares);
}

#ifdef WATER_SSR
// ---- the screen-space march: the `WATER_SSR` debug variant (`$WOW_SSR=1`) ----------------------
//
// Nothing outside this block names anything in it, so the shipped pipelines compile none of it.

/// How far the march's read is blurred, in main-view texels: four bilinear taps on the diagonals
/// this far out cover about the 6 texels the planar capture's kernel does ([`sample_mirror`], one
/// capture texel at [`REFLECT_DOWNSCALE`](benilla_world::liquid) 2, tented over 3x3), so the march's
/// hard-edged hits read with the planar tier's grain. Only the colour is filtered: the hit is found
/// on the depth pyramid, which this does not touch.
const SSR_READ_BLUR: f32 = 2.0;

/// How far from the edge of the frame the march's confidence starts falling, in UV.
///
/// **The one artefact every screen-space reflection has**, and the only honest handling of it: the
/// ray can only find what the camera drew, so a reflection whose source is off-screen has to stop
/// existing — and it must stop *gradually*, or the boundary is a hard line across the water that
/// moves whenever the camera does. Where this fades out, the planar capture and the sky mix
/// underneath it are still there to take over.
const SSR_EDGE_FADE: f32 = 0.14;

/// Read the scene snapshot where the ray landed, and score how much to believe it: the UV is
/// clamped inside the frame, which the snapshot does not reach past, and the confidence falls off
/// toward the frame's border ([`ssr_edge_conf`]). The march's opaque hits read through it; a sky hit
/// is the dome along the ray by the dome's own law with the same edge confidence, and reads the
/// snapshot here only under the debug switch `dome.warp.z`. The ripple moves the ray's origin, not
/// this read (`water_reflect.march.y`).
fn ssr_read(uv: vec2<f32>) -> SsrHit {
    var out: SsrHit;
    let read_uv = clamp(uv, vec2<f32>(0.002), vec2<f32>(0.998));
    // Four bilinear taps on the diagonals — [`sample_mirror`]'s kernel, and deliberately the same
    // one. No premultiplied divide here: the snapshot is a copy of an opaque frame, so every tap
    // has something in it and there is no coverage to weight by. See [`SSR_READ_BLUR`].
    let o = SSR_READ_BLUR / vec2<f32>(textureDimensions(scene_tex));
    var acc = textureSampleLevel(scene_tex, scene_samp, read_uv + vec2<f32>(-o.x, -o.y), 0.0).rgb;
    acc += textureSampleLevel(scene_tex, scene_samp, read_uv + vec2<f32>(o.x, -o.y), 0.0).rgb;
    acc += textureSampleLevel(scene_tex, scene_samp, read_uv + vec2<f32>(-o.x, o.y), 0.0).rgb;
    acc += textureSampleLevel(scene_tex, scene_samp, read_uv + vec2<f32>(o.x, o.y), 0.0).rgb;
    out.rgb = acc * 0.25;
    out.conf = ssr_edge_conf(read_uv);
    return out;
}

/// How much a march that ended at `uv` is believed on account of where on the screen that is: full
/// in the frame, fading to nothing over [`SSR_EDGE_FADE`] at its border, where the tier hands over to
/// whatever is under it.
fn ssr_edge_conf(uv: vec2<f32>) -> f32 {
    let read_uv = clamp(uv, vec2<f32>(0.002), vec2<f32>(0.998));
    let edge = min(
        min(read_uv.x, 1.0 - read_uv.x),
        min(read_uv.y, 1.0 - read_uv.y),
    );
    return smoothstep(0.0, SSR_EDGE_FADE, edge);
}

/// How many cells the traversal may visit before giving up — AMD's FidelityFX SSSR sample runs its
/// `FFX_SSSR_HierarchicalRaymarch` with 128, and this is that figure.
const HIZ_MAX_STEPS: i32 = 128;

/// How far past a cell wall the ray is placed when it leaves the cell, as a fraction of a
/// level-zero texel — FFX SSSR's `uv_offset`, and its value.
///
/// **In SCREEN units, and that is the fix.** The first version of this traversal parametrised the
/// ray in world yards and stepped `1e-4` of a yard past each wall. For a ray running toward the
/// horizon a ten-thousandth of a yard is a vanishing fraction of a texel, so the step often failed
/// to leave the cell at all and the ray sat on one wall until its budget ran out: at the
/// Stranglethorn camera 23% of water fragments exhausted it, 58% of the far field, and every one of
/// them had stalled. Measured in texels the overshoot is the same size everywhere on the screen.
const HIZ_CROSS: f32 = 0.005;

/// How thick every surface in the depth buffer is taken to be, in yards — McGuire & Mara's
/// `zThickness` ("Efficient GPU Screen-Space Ray Tracing", JCGT 2014): each depth sample is a slab
/// from its surface to this far behind it, and a ray hits only inside a slab.
///
/// A depth buffer records front faces and nothing behind them, so some thickness has to be
/// assumed. Too little and a ray that meets a surface at a grazing angle slips past it between
/// samples; too much and a ray that passed behind something thin — a trunk, a branch — is taken to
/// have hit it, and paints it where it does not reach. The old march's constant, kept.
const SSR_THICKNESS: f32 = 6.0;

/// Is a ray at reverse-Z depth `ray` BEHIND the slab of the surface at `depth` — farther than the
/// point [`SSR_THICKNESS`] yards behind that surface?
///
/// Bevy's view is an infinite reverse-Z perspective, where depth is `near / distance`, so the back
/// of the slab sits at `near·d / (near + T·d)` and the test is `ray < near·d / (near + T·d)`. Every
/// quantity is positive, so it is asked as `ray·(near + T·d) < near·d` instead — two fused
/// multiply-adds and no divide. That form is not cosmetic: the compiler hoists this arithmetic out
/// of the branch it sits in and runs it on every step of the march, taken or not, and with the
/// divide there (a reciprocal) merely compiling the pass-behind in cost 0.26 ms of the water's pass
/// at Mirror Lake with the branch never taken. Depth zero (sky) is never passed behind: the right
/// side is zero.
fn behind_slab(ray: f32, depth: f32, near: f32) -> bool {
    return ray * (near + SSR_THICKNESS * depth) < near * depth;
}

/// Project a world point to the traversal's space: `xy` the screen UV, `z` reverse-Z device depth.
fn hiz_screen(p: vec3<f32>) -> vec3<f32> {
    let ndc = position_world_to_ndc(p);
    return vec3<f32>(ndc_to_uv(ndc.xy), ndc.z);
}

/// Walk the reflected ray through `liquid::hiz`'s pyramid and read the scene snapshot where it
/// lands.
///
/// **This is AMD FidelityFX SSSR's `FFX_SSSR_HierarchicalRaymarch`**, ported rather than derived
/// (GPUOpen-Effects/FidelityFX-SSSR, `ffx_sssr.h`, built with `FFX_SSSR_INVERTED_DEPTH_RANGE`
/// because Bevy is reverse-Z). Its three load-bearing choices, each of which the first version of
/// this function got differently and paid for:
///
/// * **The ray lives in screen space** — UV and device depth. A perspective projection maps lines
///   to lines, so a world ray is a straight line in that space too, and every cell wall and every
///   depth plane is an axis-aligned plane the ray meets at a closed-form parameter.
/// * **Leaving a cell overshoots by a fraction of a texel** ([`HIZ_CROSS`]), so it always leaves.
/// * **A ray that is still in front of the tile's nearest surface but would pass it inside the
///   tile advances to that depth before descending**, rather than descending where it stands.
///
/// The pyramid holds the NEAREST depth per tile, so the rule is: in front of the tile's nearest
/// surface and leaving through a wall, the whole tile is provably empty along the ray — skip it and
/// go up a level; otherwise go down one. Falling below level zero is a hit.
///
/// **This is the half of the water's reflection that a plane cannot do.** The direction traced is
/// `reflect(-V, N)` on the fragment's GEOMETRIC normal ([`surface_normal`]) — the heightfield's own
/// facet, so a sloped stream is traced at its real orientation and needs no mirror built for it.
///
/// The fine ripple is deliberately NOT in that direction: it slides the ray's ORIGIN across the
/// surface instead, so neighbouring pixels keep one direction and the march stays coherent while
/// every sample is still a true reflection of a real point (`water_reflect.march.y`). What it cannot
/// do is see off the screen, which is why it returns a confidence rather than just a colour: where
/// the ray leaves the frame the planar capture and the sky mix behind it take over. Under
/// [`WaterStyle::StylisedSsr`] there is no capture behind it and the sky mix alone catches what it
/// misses, which is the trade that lane is offered for.
///
/// The water itself is not in the pyramid (`liquid::hiz` seeds it after the opaque pass, before
/// anything transparent draws), so there is no self-intersection to bias against — the first step
/// off the surface is already reading opaque geometry only.
fn ssr_trace(origin: vec3<f32>, dir: vec3<f32>) -> SsrHit {
    var out: SsrHit;
    out.rgb = vec3<f32>(0.0);
    out.conf = 0.0;

    // The direction as the difference of two projected points. The second point has to be in
    // front of the eye for the difference to mean anything; `clip.w` is view depth, so a ray
    // heading back toward the camera is cut to half the distance to the eye plane.
    let clip_o = position_world_to_clip(origin);
    let clip_d = direction_world_to_clip(dir);
    var reach = clip_o.w;
    if (clip_d.w < 0.0) {
        reach = min(reach, 0.5 * clip_o.w / -clip_d.w);
    }
    let o = hiz_screen(origin);
    let d = hiz_screen(origin + dir * reach) - o;
    let big = 3.4e38;
    let inv_d = vec3<f32>(
        select(big, 1.0 / d.x, d.x != 0.0),
        select(big, 1.0 / d.y, d.y != 0.0),
        select(big, 1.0 / d.z, d.z != 0.0),
    );

    let top = i32(textureNumLevels(hiz_tex)) - 1;
    let base_res = vec2<f32>(textureDimensions(hiz_tex, 0));
    let cross = select(vec2<f32>(HIZ_CROSS), vec2<f32>(-HIZ_CROSS), d.xy < vec2<f32>(0.0)) / base_res;
    let wall = select(vec2<f32>(1.0), vec2<f32>(0.0), d.xy < vec2<f32>(0.0));

    // Step off the origin's own cell first, so the walk does not open by intersecting it.
    var level = 0;
    var t: f32;
    {
        let planes = (floor(o.xy * base_res) + wall) / base_res + cross;
        let tw = (planes - o.xy) * inv_d.xy;
        t = min(tw.x, tw.y);
    }
    var p = o + t * d;

    // **Loop invariants, read once.** Both come from buffers — the lever from the water's storage
    // buffer, the near plane from the view uniform — and a load the compiler does not hoist is a
    // memory round trip on every step of a loop whose whole cost is its latency chain. Read inside
    // the loop, merely compiling the pass-behind in (branch never taken) cost 0.28 ms of the
    // water's pass at Mirror Lake against 0.09 for the pass-behind work itself.
    let pass_behind = water_reflect.march.x > 0.5;
    let near_plane = perspective_camera_near();

    var i = 0;
    for (; i < HIZ_MAX_STEPS && level >= 0; i = i + 1) {
        if (any(p.xy < vec2<f32>(0.0)) || any(p.xy >= vec2<f32>(1.0))) {
            return out; // off the frame: the screen cannot say
        }
        if (p.z <= 0.0) {
            return out; // past the far plane
        }
        let res = vec2<f32>(textureDimensions(hiz_tex, level));
        let cell = floor(p.xy * res);
        // The nearest surface anywhere in the tile. The farthest is fetched below, only if needed.
        let surface = textureLoad(hiz_tex, vec2<i32>(cell), level).r;

        // **A tile of pure sky, at any level.** The pyramid reduces with `max` and sky is the far
        // plane, so a cell reporting zero has nothing in it anywhere — the ray is over open sky.
        // That is a HIT: reflecting a surface below the eye sends the ray up and forward, at sky
        // that is on the screen, and treating it as a miss was most of why this mode fell back to
        // a flat colour over open water. Getting it from the hierarchy costs nothing extra.
        //
        // **The colour is the dome's along the reflected ray, not the colour buffer's here.** It
        // used to be `ssr_read(p.xy)`, which trusts that a pixel with no depth shows sky — and the
        // colour buffer can hold things the depth does not: anything drawn without writing it.
        // Every such pixel then became a sky hit that read the object's colour, the same few pixels
        // for every water row whose ray climbed that column, which is how a palm missing from the
        // depth came out as a stalk (Stefan's Stranglethorn report). The dome is a pure function
        // of direction, so the sky can be asked for instead of read: an object without depth then
        // reads as the sky behind it rather than being smeared down the water.
        //
        // It is also the more correct sky. Seen in the water, the sky is what lies along the
        // reflected ray — its own vanishing point — not whatever the screen shows where the ray
        // happened to leave the geometry. Reading at the entry point quantised that to the tile
        // the walk was on, which is what drew the blocky light rectangles in sky reflections.
        //
        // The edge fade stays keyed to where the ray left the frame's geometry, so the hand-over
        // to the tier underneath at the screen border is exactly what it was.
        if (surface <= 0.0) {
            if (water_reflect.dome.warp.z > 0.5) {
                return ssr_read(p.xy);
            }
            var sky_hit: SsrHit;
            sky_hit.rgb = dome_color(water_reflect.dome, normalize(dir));
            sky_hit.conf = ssr_edge_conf(p.xy);
            return sky_hit;
        }

        // FFX_SSSR_AdvanceRay. The cell's two exit walls and its nearest-depth plane; the depth
        // plane only counts for a ray moving away from the eye, which under reverse-Z is `d.z < 0`.
        let planes = vec3<f32>((cell + wall) / res + cross, surface);
        var tp = (planes - o) * inv_d;
        tp.z = select(big, tp.z, d.z < 0.0);
        let t_wall = min(tp.x, tp.y);
        let t_min = min(t_wall, tp.z);
        // Larger depth is nearer under reverse-Z: the ray is in front of everything in the tile.
        let in_front = surface < p.z;
        // **Or behind everything in it** — the half FFX SSSR does not have, and the reason there is
        // a farthest-depth pyramid beside the nearest. Every surface is a slab [`SSR_THICKNESS`] deep, so a
        // ray that stays behind the back of the FARTHEST surface's slab for its whole way across
        // the tile has passed behind all of them, and the tile can be skipped whole exactly as an
        // empty one in front is. Its nearest point in the tile is whichever end is nearer.
        //
        // Without it the ray could only descend onto the first surface it went behind, and a ray
        // going under a tree crown — which the water sees and the camera does not — met the
        // crown's front leaves 24 yards and more in front of it and had to be thrown away: the
        // black fragments in reflected canopies (Stefan's Stranglethorn report; 74% of those
        // rejects were that far behind). Now it walks on behind the crown to whatever it really
        // meets. What it cannot know is anything hidden behind the crown that the screen never
        // held; there it finds the next thing that IS on screen, which is the known cost of the
        // method and why Bevy's own marcher keeps it off for colour.
        //
        // **Only a ray that is NOT in front needs the farthest surface**, so only then is it read:
        // most steps cross empty space in front of everything, and reading both pyramids on every
        // step cost the water's pass nearly a millisecond at Mirror Lake as one 64-bit texel.
        var behind = false;
        // **The lever first, on its own.** `pass_behind` is the same for every lane, so this is a
        // scalar branch the compiler does not speculate past. Folded into one condition with the
        // per-lane `in_front`, it hoisted the whole test onto every step instead, and merely having
        // it compiled in with the lever OFF cost 0.24 ms of the water's pass at Mirror Lake; apart,
        // off measures level with the shader that never had it.
        if (pass_behind) {
        if (!in_front) {
            let p_exit_z = o.z + t_wall * d.z;
            let seg_near = max(p.z, p_exit_z);
            let seg_far = min(p.z, p_exit_z);
            // **Only then, and only if the ray is past the NEAREST surface's slab too.** A ray
            // still within it cannot be behind the farthest's (that slab is never nearer) nor
            // between the two, so the fetch below would be for nothing — and that is the common
            // case: a ray closing in on a hit. Measured at Mirror Lake before this gate, the pass
            // cost 0.44 ms of the water's pass over the lever off.
            if (behind_slab(seg_near, surface, near_plane)) {
                let farthest = textureLoad(hiz_far_tex, vec2<i32>(cell), level).r;
                let behind_all = behind_slab(seg_near, farthest, near_plane);
                // **At the finest level a texel is two layers, and the ray may be between them.**
                // A level-zero texel covers one and a half to three and a half screen pixels, so
                // at the porous edge of a crown most of them hold a leaf AND whatever shows through
                // beside it. A ray behind the leaf's slab but still in front of the far layer over
                // its whole way across is in the empty space between the two and walks on; without
                // this, one mixed texel on the path stopped the ray and it was rejected — measured:
                // with only the whole-tile skip the canopy's rejects barely moved (14.9% to 13.6%).
                // A single-layer texel is untouched (its two slabs are one).
                let between = level == 0 && seg_far > farthest;
                behind = behind_all || between;
            }
        }
        }
        let skipped = (t_min != tp.z && in_front) || behind;
        // **One position update per step.** The march loop is where this shader spends its time,
        // and its cost is instructions per step — written as two branches each recomputing the
        // position, the compiler ran both and merged them with selects.
        let t_next = select(select(t, t_wall, behind), t_min, in_front);
        t = t_next;
        p = o + t * d;
        level = select(level - 1, min(level + 1, top), skipped);
    }
    if (level >= 0) {
        return out; // budget
    }

    // FFX_SSSR_ValidateHit, the parts that apply. Off the frame is no hit; neither is a ray that
    // never really left its own texel.
    if (any(p.xy < vec2<f32>(0.0)) || any(p.xy >= vec2<f32>(1.0))) {
        return out;
    }
    // Against the texel's NEAREST surface when rays cannot pass behind — the test as it was. When
    // they can, a finest texel may hold two layers (a leaf and the bank behind it), the walk only
    // stops inside one of their slabs, and the gap is measured to the layer it stopped in.
    let texel0 = vec2<i32>(floor(p.xy * base_res));
    let near0 = textureLoad(hiz_tex, texel0, 0).r;
    var gap = depth_ndc_to_view_z(near0) - depth_ndc_to_view_z(p.z);
    if (pass_behind && gap > SSR_THICKNESS) {
        // Past the nearest layer's slab: the walk stopped in the farthest's.
        let far0 = textureLoad(hiz_far_tex, texel0, 0).r;
        gap = depth_ndc_to_view_z(far0) - depth_ndc_to_view_z(p.z);
    }
    if (abs(gap) > SSR_THICKNESS) {
        return out;
    }
    return ssr_read(p.xy);
}
#endif // WATER_SSR

/// One planar mirror's read for this fragment: its reprojected colour, and in `a` its weight —
/// strength times coverage times [`plane_trust`] — zero where it does not serve this water.
fn planar_read(
    params: vec4<f32>,
    second: bool,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    frag_coord: vec2<f32>,
    tilt: f32,
) -> vec4<f32> {
    let plane_error = abs(world_pos.y - params.x);
    let trust = plane_trust(plane_error, params.w, tilt);
    if (params.y <= 0.0 || trust <= 0.0 || view.world_position.y <= world_pos.y) {
        return vec4<f32>(0.0);
    }
    {
        // ---- the plane-error reprojection ----
        //
        // **A capture is exact only for fragments lying ON the plane it was made for**, and almost
        // no fragment does: a liquid grid is a heightfield, so an Elwynn stream slopes away from
        // its own capture plane along its whole length, and the pond above it is a second plane
        // entirely. Read at the fragment's own screen UV, the image those fragments get is mirrored
        // about the wrong height — and because the error varies with the height, it *slides* along
        // the surface as the water drops, which is what reads as a smeared or swimming reflection
        // on a stream and as a plainly wrong one on a second body of water.
        //
        // The correction is exact and costs one projection. Let `p` be the capture plane and `h`
        // this fragment's surface height. Mirroring a point about `h` and then back about `p`
        // translates it by `2(p − h)` in Y — so the texel holding what this fragment should reflect
        // is the one the capture drew for the world point `world_pos` lifted by that much, and its
        // screen UV is where the MAIN camera projects the lifted point. At `h == p` the lift is zero
        // and this reduces to `frag_coord_to_uv(frag_coord)` exactly, which is what it replaces.
        //
        // The one approximation: the lifted point is placed at the water's own depth rather than at
        // the depth of whatever is being reflected. That makes the correction exact for reflections
        // of things at the waterline — the bank, the trees on it, a hull alongside, which is what
        // the eye judges a reflection by — and an overcorrection for distant mountains, where the
        // true offset tends to zero and the residual is a fraction of a pixel through a wobble.
        // **The ripple moves the point the mirror is read FOR, not the UV it is read AT.** The
        // capture is a reflected image, so displacing its UV is far safer here than it was in the
        // march — but it is still shifting a picture rather than reflecting a different place, and
        // at a grazing angle it shifts it far enough to show the same content twice. Reprojecting a
        // rippled position instead asks the capture what is mirrored a few inches away, which is a
        // question with a true answer. Horizontal only: the wave slides the surface, the plane the
        // mirror was built for does not move, and `lift` below depends on a height this must not
        // change.
        let mirror_pos = world_pos + vec3<f32>(n.x, 0.0, n.z) * water_reflect.march.y;
        let lift = 2.0 * (params.x - mirror_pos.y);
        let lifted = position_world_to_ndc(mirror_pos + vec3<f32>(0.0, lift, 0.0));
        // Behind the eye (reverse-Z puts `w < 0` there, and the divide flips `z` negative with it)
        // or thrown well off the side, the reprojection means nothing; the uncorrected UV is the
        // image this drew before the correction existed and is the honest fallback.
        let usable = lifted.z > 0.0 && all(abs(lifted.xy) < vec2<f32>(1.5));
        let uv = select(frag_coord_to_uv(frag_coord), ndc_to_uv(lifted.xy), usable);
        let ruv = clamp(vec2<f32>(1.0 - uv.x, uv.y), vec2<f32>(0.002), vec2<f32>(0.998));
        var mirrored: vec4<f32>;
        if (second) {
            mirrored = sample_mirror2(ruv, water_reflect.flags.x);
        } else {
            mirrored = sample_mirror(ruv, water_reflect.flags.x);
        }
        return vec4<f32>(mirrored.rgb, saturate(min(params.y, 1.0) * mirrored.a * trust));
    }
}

#ifdef WATER_PREVIEW
// ---- the reflections processed by roughness (Phase 3 preview: REF-2, REF-4, REF-5, REF-6, REF-12) -
//
// A module for `liquid.wgsl`'s stylised region. It is pasted after `planar_read`, inside
// `#ifdef WATER_STYLISED` (which always carries `WATER_SCENE_DEPTH`), and reads that file's
// bindings and helpers: `water_reflect`, `view`, `reflection_tex`/`reflection_tex2`,
// `reflection_samp`, `probe_tex`/`probe_samp`, `floor_occ`/`scene_samp`, `sample_mirror`,
// `sample_mirror2`, `plane_trust`, `probe_dir_march`, `probe_dir_depth`, `probe_fog_shares`,
// `probe_refog`, `FLOOR_CONTACT_REACH`, `ProbeSlot`. Nothing in it takes a derivative or an
// implicit-LOD sample, so every function may be called under the per-fragment branches.
//
// What a rough surface reflects. A surface whose slopes spread α per axis (GGX, α² = 2σ²) sends
// its reflected ray √2·α wide; a reflection doubles a facet's tilt. A tilt in the plane of
// incidence turns the ray up or down by twice the tilt, a tilt across it turns the ray sideways by
// twice the tilt times μ = n·v, so at a grazing view the lobe is tall and narrow: the vertical
// streak. Only what the surface no longer resolves blurs: the roughness tensor less the calm
// surface's α0² (`surface_lost2`'s law per axis), so near water, where every octave resolves,
// reads the mirror exactly as before, and the ripple's slide of the lookup point stays the
// reflection's only motion there.
//
// The lobe is an angle at the water; the image it blurs is seen from elsewhere. Something `D`
// yards behind the reflecting point, seen by the eye `d_eye` yards in front of it, spreads over
// `D/(d_eye + D)` of the lobe on the screen: 0 at a contact, the whole lobe for the sky. That is
// what keeps a post, a hull and a bank attached to their reflections while the far shore softens.
// Seen from a probe's capture point `d_geom` yards from the same thing, the lobe spreads `D/d_geom`
// of its angle in the cube.
//
// Rough facets the eye sees lean toward it, so a rough surface sees higher sky: the lobe's centre
// climbs by about 0.7·α at grazing, scaled with the blur and taken through the same contact share,
// so a reflection stays on its waterline.
//
// The blur reads prefiltered images, never a wide kernel on a sharp one: the mirrors' chains
// (`reflection_chain`, five levels from half the capture's size, premultiplied by coverage) and
// the probe cube's cone chain (`probe_filter.wgsl`). A level is picked by the lobe's narrow side;
// a column of taps, evenly spaced up the screen's vertical through the lookup point and never
// further apart than twice the level's own spread, makes the tall side. Taps that close sum to a
// smooth profile, so nothing is drawn twice, and none is jittered, so a still camera gets a still
// image.

// The planar mirrors' roughness chains (`benilla_water`'s mirror chain dispatch in the water
// compute pass): each capture downsampled through five levels from half its size by a [1 3 3 1]
// tent, coverage-premultiplied as the capture is (it clears to transparent black), so level k
// holds the capture blurred to about `RR_CHAIN_SIGMA0`·2^k capture texels. Read through
// `reflection_samp` (linear, clamped, trilinear).
@group(#{MATERIAL_BIND_GROUP}) @binding(123) var reflection_chain: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(124) var reflection_chain2: texture_2d<f32>;
#ifdef WATER_MIRROR_DIST
// The mirrors' hit distances (REF-1): the mirrored eye's path length to what each capture texel
// shows, yards, 10000 for the dome, the nearest of each 2×2 capture texels at half the capture's
// size, each further level the nearest of four. Read by `textureLoad`.
@group(#{MATERIAL_BIND_GROUP}) @binding(125) var reflection_dist: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(126) var reflection_dist2: texture_2d<f32>;
#endif

/// How far a rough surface's reflection climbs toward higher sky, as a share of its unresolved α
/// along the view, at full lobe (`flags.w` = 1) and grazing.
const RR_LIFT: f32 = 0.7;

/// The extra spread, in capture texels, under which a mirror is read exactly as before
/// ([`sample_mirror`]'s tent).
const RR_CALM_TEXELS: f32 = 0.5;

/// [`sample_mirror`]'s own spread at radius 1, capture texels, one σ: four bilinear taps on the
/// diagonals at ±1 texel (σ² = 1 + 1/6).
const RR_TENT_SIGMA: f32 = 1.08;

/// The chain's level-0 spread in capture texels, one σ: the [1 3 3 1] tent (σ² = 0.75) read
/// bilinearly at twice the capture's texel (σ² = 4/6). Each level below doubles it, within a
/// tenth.
const RR_CHAIN_SIGMA0: f32 = 1.2;

/// The extra spread, capture texels, of a mirror read at no plane trust: a badly fitted surface
/// blurs as it fades into the tiers beneath.
const RR_TRUST_BLUR: f32 = 4.0;

/// The eye's height over the higher of the water and the mirror plane, yards, from which the
/// mirror fades in, and at which it is whole. The capture's oblique clip degenerates within a
/// quarter yard of its plane.
const RR_EYE_FADE: vec2<f32> = vec2<f32>(0.3, 0.9);

/// The angle the screen's axes are measured over at a lookup point, radians.
const RR_AXIS_ANGLE: f32 = 0.01;

/// The probe chain's spread per level, radians, one σ a side: level k convolves level k − 1 with a
/// cosine cone (`probe::probe_cone`: 3, 6, 12, 22 and 36°), whose taps spread cone/2 a side. Each
/// pair is log2 of the level's spread and one over log2 of the step to the next.
const RR_PROBE_SIGMA1: f32 = 0.026;
const RR_PROBE_L1: f32 = -5.2653;
const RR_PROBE_L2: f32 = -4.0933;
const RR_PROBE_L3: f32 = -3.0613;
const RR_PROBE_L4: f32 = -2.1436;
const RR_PROBE_K1: f32 = 0.8532;
const RR_PROBE_K2: f32 = 0.9690;
const RR_PROBE_K3: f32 = 1.0897;
const RR_PROBE_K4: f32 = 1.2916;

// The column's taps per tier: up to about 1.4:1, 2.4:1 and 3.6:1 of stretch before the level
// rises. Low's two mirror taps cost what `sample_mirror`'s four bilinear taps do, and its probe is
// one mip-biased read.
#ifdef WATER_TIER_HIGH
const RR_MIRROR_TAPS: i32 = 6;
const RR_PROBE_TAPS: i32 = 4;
#else
#ifdef WATER_TIER_LOW
const RR_MIRROR_TAPS: i32 = 2;
const RR_PROBE_TAPS: i32 = 1;
#else
const RR_MIRROR_TAPS: i32 = 4;
const RR_PROBE_TAPS: i32 = 3;
#endif
#endif

/// The reflected lobe at a fragment, radians: its spread up (in the plane of incidence) and
/// across, one σ, and how far its centre climbs toward higher sky.
struct RoughLobe {
    up: f32,
    side: f32,
    lift: f32,
};

/// The lobe of a surface with roughness tensor `a2` (world XZ: xx, zz, xz, `WaterSurface::a2`)
/// seen along `to_view` over its flat normal `flat_n`, at `gain` of the physical lobe
/// (`water_reflect.flags.w`; 0 is none). The tensor less α0² on each axis is what the surface no
/// longer resolves, read along the view's bearing and across it.
fn rough_lobe(a2: vec3<f32>, to_view: vec3<f32>, flat_n: vec3<f32>, gain: f32) -> RoughLobe {
    let a0 = water_reflect.look.surf.x;
    let lost = vec3<f32>(max(a2.x - a0 * a0, 0.0), max(a2.y - a0 * a0, 0.0), a2.z);
    // The view's bearing over the surface; looking straight down every bearing is the same.
    let bearing = to_view.xz;
    let bl = length(bearing);
    let t = select(vec2<f32>(1.0, 0.0), bearing / max(bl, 1e-6), bl > 1e-4);
    let along = lost.x * t.x * t.x + lost.y * t.y * t.y + 2.0 * lost.z * t.x * t.y;
    let across = lost.x * t.y * t.y + lost.y * t.x * t.x - 2.0 * lost.z * t.x * t.y;
    let a_up = sqrt(max(along, 0.0));
    let mu = saturate(dot(flat_n, to_view));
    let k = 1.41421356 * max(gain, 0.0);
    return RoughLobe(
        k * a_up,
        k * sqrt(max(across, 0.0)) * mu,
        RR_LIFT * max(gain, 0.0) * a_up * (1.0 - mu),
    );
}

/// The share of the lobe a reflected thing `d_hit` yards behind the water shows to an eye `d_eye`
/// yards in front of it: `D/(d_eye + D)`. A negative `d_hit` is not known, and takes the whole
/// lobe.
fn rough_contact(d_hit: f32, d_eye: f32) -> f32 {
    if (d_hit < 0.0) {
        return 1.0;
    }
    return d_hit / max(d_hit + d_eye, 1e-3);
}

/// How far behind mirror water the reflected thing lies, yards, at the fragment `frag_coord`: the
/// compute pass's plane march along the flat reflected ray (`floor_occ.a`, its `rho_c`, with
/// [`floor_rho_c`] undone: `D = R·√(ρ/(1 − ρ))`), the bank, post or deck it met, or how far it got
/// before it left the frame, which the thing lies at least as far as. A texel with no plane point
/// holds 0, a contact. Negative, not known, on a frame the pass did not march.
fn rough_mirror_hit(frag_coord: vec2<f32>) -> f32 {
    if (water_reflect.look.quality.y < 0.5) {
        return -1.0;
    }
    let uv = frag_coord_to_uv(frag_coord);
    let rho = clamp(textureSampleLevel(floor_occ, scene_samp, uv, 0.0).a, 0.0, 0.999);
    return FLOOR_CONTACT_REACH * sqrt(rho / (1.0 - rho));
}

/// `dir` turned toward the zenith by about `lift` radians near the horizon, less toward the top.
fn rough_lift_dir(dir: vec3<f32>, lift: f32) -> vec3<f32> {
    if (lift <= 0.0) {
        return dir;
    }
    return normalize(dir + vec3<f32>(0.0, lift, 0.0));
}

/// The α the sky floor is read with under the lobe, so the floor and the mirrors blur one sky
/// alike: the lobe's spread up as an α (σ = √2·α), never more than the floor's own `alpha`.
fn rough_floor_alpha(lobe: RoughLobe, alpha: f32) -> f32 {
    return min(lobe.up * 0.70710678, alpha);
}

/// How much a mirror shows at the eye's height `eye_y` over a surface at `surface_y` served by the
/// plane `plane_y`: none at or below either, whole from [`RR_EYE_FADE`]'s end.
fn rough_eye_fade(eye_y: f32, surface_y: f32, plane_y: f32) -> f32 {
    return smoothstep(RR_EYE_FADE.x, RR_EYE_FADE.y, eye_y - max(surface_y, plane_y));
}

/// The capture's UV per radian at the lookup point `p` (world: the lifted point the main camera
/// projects to `uv`), in the capture's `1 − u` frame: `xy` toward higher reflected elevation, down
/// the screen along the perspective's vertical through `p`, `zw` across it. The projection's
/// centre scale where `p` does not project (`usable` false, or an axis behind the eye).
fn rough_mirror_axes(p: vec3<f32>, uv: vec2<f32>, usable: bool) -> vec4<f32> {
    let centre = vec4<f32>(
        0.0,
        0.5 * view.clip_from_view[1][1],
        0.5 * view.clip_from_view[0][0],
        0.0,
    );
    if (!usable) {
        return centre;
    }
    let to_p = p - view.world_position.xyz;
    let dist = length(to_p);
    let v = to_p / max(dist, 1e-4);
    // Up across the ray in its vertical plane, and the horizontal across the ray; |v × up| is
    // |up − v·v.y| for a unit ray.
    let up_raw = vec3<f32>(0.0, 1.0, 0.0) - v * v.y;
    let ul = length(up_raw);
    if (ul < 1e-4) {
        return centre;
    }
    let up_dir = up_raw / ul;
    let side_dir = cross(v, vec3<f32>(0.0, 1.0, 0.0)) / ul;
    let reach = RR_AXIS_ANGLE * dist;
    // Lower in the mirrored world is higher in the reflected one.
    let a = position_world_to_ndc(p - up_dir * reach);
    let b = position_world_to_ndc(p + side_dir * reach);
    if (a.z <= 0.0 || b.z <= 0.0) {
        return centre;
    }
    let da = (ndc_to_uv(a.xy) - uv) / RR_AXIS_ANGLE;
    let db = (ndc_to_uv(b.xy) - uv) / RR_AXIS_ANGLE;
    return vec4<f32>(-da.x, da.y, -db.x, db.y);
}

/// Whether mirror `second`'s chain is this capture's: half its size, rounded down. A material
/// holding the placeholder, or a chain not yet rebuilt after a resize, is not.
fn rough_chain_ready(second: bool) -> bool {
    var cap: vec2<u32>;
    var chain: vec2<u32>;
    if (second) {
        cap = textureDimensions(reflection_tex2);
        chain = textureDimensions(reflection_chain2);
    } else {
        cap = textureDimensions(reflection_tex);
        chain = textureDimensions(reflection_chain);
    }
    return all(chain == max(cap / 2u, vec2<u32>(1u)));
}

/// The chain level for a lobe `sv` capture texels up and `ss` across, one σ, read by a column of
/// `taps`: the level whose spread is the lobe's across, or the least the column can stretch to its
/// up, at most `top`. `x` the level, `y` its spread in capture texels.
fn rough_chain_level(sv: f32, ss: f32, taps: i32, top: f32) -> vec2<f32> {
    let n = f32(taps);
    let reach = sqrt(1.0 + (n * n - 1.0) / 3.0);
    let want = max(max(ss, sv / reach), RR_CHAIN_SIGMA0);
    let lod = clamp(log2(want / RR_CHAIN_SIGMA0), 0.0, top);
    return vec2<f32>(lod, RR_CHAIN_SIGMA0 * exp2(lod));
}

/// Mirror `second`'s chain read over a lobe `sv_x` capture texels up and `ss_x` across (one σ,
/// beyond the tent's own) at `uv`, with `dv` and `ds` the UV of one capture texel up and across:
/// [`RR_MIRROR_TAPS`] taps evenly up the vertical at [`rough_chain_level`]'s level, their spread
/// with the level's making `sv`, alternate taps a quarter of the level's spread to either side so
/// a coarse level's bilinear facets do not line up down the column. The mean is premultiplied by
/// coverage, as [`sample_mirror`]'s is.
fn rough_mirror_column(
    second: bool,
    uv: vec2<f32>,
    dv: vec2<f32>,
    ds: vec2<f32>,
    sv_x: f32,
    ss_x: f32,
    texel: vec2<f32>,
) -> vec4<f32> {
    let n = f32(RR_MIRROR_TAPS);
    let sv = sqrt(sv_x * sv_x + RR_TENT_SIGMA * RR_TENT_SIGMA);
    let ss = sqrt(ss_x * ss_x + RR_TENT_SIGMA * RR_TENT_SIGMA);
    let levels = select(
        textureNumLevels(reflection_chain),
        textureNumLevels(reflection_chain2),
        second,
    );
    let lv = rough_chain_level(sv, ss, RR_MIRROR_TAPS, f32(levels - 1u));
    var gap = 0.0;
    var zig = 0.0;
    if (RR_MIRROR_TAPS > 1) {
        gap = min(sqrt(max(sv * sv - lv.y * lv.y, 0.0) * 12.0 / (n * n - 1.0)), 2.0 * lv.y);
        zig = 0.25 * lv.y;
    }
    // One capture texel in from each edge: half a chain texel, where the bilinear read stays
    // inside.
    let edge = 1.0 / texel;
    var acc = vec4<f32>(0.0);
    for (var i = 0; i < RR_MIRROR_TAPS; i = i + 1) {
        let along = (f32(i) - 0.5 * (n - 1.0)) * gap;
        let aside = select(-zig, zig, (i & 1) == 1);
        let q = clamp(uv + dv * along + ds * aside, edge, vec2<f32>(1.0) - edge);
        if (second) {
            acc = acc + textureSampleLevel(reflection_chain2, reflection_samp, q, lv.x);
        } else {
            acc = acc + textureSampleLevel(reflection_chain, reflection_samp, q, lv.x);
        }
    }
    acc = acc / n;
    return vec4<f32>(acc.rgb / max(acc.a, 1.0e-4), acc.a);
}

#ifdef WATER_MIRROR_DIST
/// How far behind the water the nearest thing mirror `second` shows over a lobe's reach lies,
/// yards: its distance chain at `level` (each texel the nearest of the capture texels under it)
/// at `ruv`, less the eye's `d_eye`, the mirrored eye's path to the plane along the same ray.
/// Negative for the dome.
fn rough_mirror_dist(second: bool, ruv: vec2<f32>, level: f32, d_eye: f32) -> f32 {
    var path = 0.0;
    if (second) {
        let l = i32(clamp(level, 0.0, f32(textureNumLevels(reflection_dist2) - 1u)));
        let size = vec2<f32>(textureDimensions(reflection_dist2, l));
        let px = vec2<i32>(clamp(ruv, vec2<f32>(0.0), vec2<f32>(0.9999)) * size);
        path = textureLoad(reflection_dist2, px, l).r;
    } else {
        let l = i32(clamp(level, 0.0, f32(textureNumLevels(reflection_dist) - 1u)));
        let size = vec2<f32>(textureDimensions(reflection_dist, l));
        let px = vec2<i32>(clamp(ruv, vec2<f32>(0.0), vec2<f32>(0.9999)) * size);
        path = textureLoad(reflection_dist, px, l).r;
    }
    if (path > 9000.0) {
        return -1.0;
    }
    return max(path - d_eye, 0.0);
}
#endif

/// One planar mirror's read for this fragment over the surface's lobe: [`planar_read`]'s gates,
/// plane-error reprojection and slide of the lookup point by the ripple, then the capture read
/// over the lobe its distance `d_hit` ([`rough_mirror_hit`], negative unknown) lets the screen
/// show, lifted toward higher sky; within [`RR_CALM_TEXELS`] of no blur, [`sample_mirror`]'s read
/// exactly. `a` is the weight, strength times coverage times [`plane_trust`] times the eye's
/// fade. With the lever off (`flags.w` 0) or no chain, [`planar_read`] itself.
fn planar_read_rough(
    params: vec4<f32>,
    second: bool,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    frag_coord: vec2<f32>,
    tilt: f32,
    lobe: RoughLobe,
    d_hit: f32,
) -> vec4<f32> {
    if (pv_rough_gain() <= 0.0 || !rough_chain_ready(second)) {
        return planar_read(params, second, world_pos, n, frag_coord, tilt);
    }
    let trust = plane_trust(abs(world_pos.y - params.x), params.w, tilt);
    let eye = view.world_position.xyz;
    let fade = rough_eye_fade(eye.y, world_pos.y, params.x);
    if (params.y <= 0.0 || trust <= 0.0 || fade <= 0.0) {
        return vec4<f32>(0.0);
    }
    // `planar_read`'s lookup: the rippled point lifted by twice its plane error, as the main
    // camera projects it, read at `1 − u`.
    let mirror_pos = world_pos + vec3<f32>(n.x, 0.0, n.z) * water_reflect.march.y;
    let p = mirror_pos + vec3<f32>(0.0, 2.0 * (params.x - mirror_pos.y), 0.0);
    let lifted = position_world_to_ndc(p);
    let usable = lifted.z > 0.0 && all(abs(lifted.xy) < vec2<f32>(1.5));
    let uv = select(frag_coord_to_uv(frag_coord), ndc_to_uv(lifted.xy), usable);
    let ruv = vec2<f32>(1.0 - uv.x, uv.y);
    let axes = rough_mirror_axes(p, uv, usable);
    var texel: vec2<f32>;
    if (second) {
        texel = vec2<f32>(textureDimensions(reflection_tex2));
    } else {
        texel = vec2<f32>(textureDimensions(reflection_tex));
    }
    // Capture texels per radian up and across.
    let jv = max(length(axes.xy * texel), 1e-3);
    let js = max(length(axes.zw * texel), 1e-3);
    let d_eye = distance(eye, world_pos);
#ifdef WATER_MIRROR_DIST
    // The nearest thing over the widest the lobe can reach, a level past the column's.
    let wide = rough_chain_level(lobe.up * jv, lobe.side * js, RR_MIRROR_TAPS, 4.0);
    let h = rough_contact(rough_mirror_dist(second, ruv, wide.x + 2.0, d_eye), d_eye);
#else
    let h = rough_contact(d_hit, d_eye);
#endif
    let trust_px = RR_TRUST_BLUR * (1.0 - trust);
    let sv = sqrt((lobe.up * h * jv) * (lobe.up * h * jv) + trust_px * trust_px);
    let ss = sqrt((lobe.side * h * js) * (lobe.side * h * js) + trust_px * trust_px);
    let centre = ruv + axes.xy * (lobe.lift * h);
    var mirrored: vec4<f32>;
    if (max(sv, ss) < RR_CALM_TEXELS) {
        let c = clamp(centre, vec2<f32>(0.002), vec2<f32>(0.998));
        if (second) {
            mirrored = sample_mirror2(c, water_reflect.flags.x);
        } else {
            mirrored = sample_mirror(c, water_reflect.flags.x);
        }
    } else {
        mirrored = rough_mirror_column(
            second,
            centre,
            axes.xy / jv,
            axes.zw / js,
            sv,
            ss,
            texel,
        );
    }
    return vec4<f32>(mirrored.rgb, saturate(min(params.y, 1.0) * mirrored.a * trust * fade));
}

/// The probe chain level whose spread is `sigma` radians: between the sharp capture and level 1
/// a straight blend, past it one level for each step of [`RR_PROBE_SIGMA1`]'s table, at most 5.
fn rough_probe_lod(sigma: f32) -> f32 {
    if (sigma <= RR_PROBE_SIGMA1) {
        return max(sigma, 0.0) / RR_PROBE_SIGMA1;
    }
    let l = log2(sigma);
    if (l <= RR_PROBE_L2) {
        return 1.0 + (l - RR_PROBE_L1) * RR_PROBE_K1;
    }
    if (l <= RR_PROBE_L3) {
        return 2.0 + (l - RR_PROBE_L2) * RR_PROBE_K2;
    }
    if (l <= RR_PROBE_L4) {
        return 3.0 + (l - RR_PROBE_L3) * RR_PROBE_K3;
    }
    return min(4.0 + (l - RR_PROBE_L4) * RR_PROBE_K4, 5.0);
}

/// [`rough_probe_lod`] undone: the spread of the probe chain at `lod`, radians.
fn rough_probe_sigma(lod: f32) -> f32 {
    if (lod <= 1.0) {
        return max(lod, 0.0) * RR_PROBE_SIGMA1;
    }
    if (lod <= 2.0) {
        return exp2(RR_PROBE_L1 + (lod - 1.0) / RR_PROBE_K1);
    }
    if (lod <= 3.0) {
        return exp2(RR_PROBE_L2 + (lod - 2.0) / RR_PROBE_K2);
    }
    if (lod <= 4.0) {
        return exp2(RR_PROBE_L3 + (lod - 3.0) / RR_PROBE_K3);
    }
    return exp2(RR_PROBE_L4 + (min(lod, 5.0) - 4.0) / RR_PROBE_K4);
}

/// The probe's calm read along `dir` in slot `idx`: the capture at level 0 and, at a softening
/// `spread` radians, four more reads around it, averaged ([`sample_probe`]'s).
fn rough_probe_soft(dir: vec3<f32>, idx: i32, spread: f32) -> vec3<f32> {
    let centre = textureSampleLevel(probe_tex, probe_samp, dir, idx, 0.0).rgb;
    if (spread <= 0.0) {
        return centre;
    }
    let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(dir.y) > 0.9);
    let t1 = normalize(cross(dir, helper)) * spread;
    let t2 = cross(dir, t1);
    var sum = centre;
    sum = sum + textureSampleLevel(probe_tex, probe_samp, dir + t1, idx, 0.0).rgb;
    sum = sum + textureSampleLevel(probe_tex, probe_samp, dir - t1, idx, 0.0).rgb;
    sum = sum + textureSampleLevel(probe_tex, probe_samp, dir + t2, idx, 0.0).rgb;
    sum = sum + textureSampleLevel(probe_tex, probe_samp, dir - t2, idx, 0.0).rgb;
    return sum * 0.2;
}

/// One probe's contribution over the lobe: [`sample_probe`]'s correction of the reflected
/// direction against the cube's stored distances and its fog re-base, the colour read on the
/// cube's chain. The lobe at the water `p` spreads `D/d_geom` of its angle seen from the capture
/// point, `D` the hit's distance from `p` and `d_geom` from the capture point (the whole lobe for
/// sky); within the calm softening's own spread, [`sample_probe`]'s read at level 0. With the lever
/// off or a cube without a chain (`probe_box_max.z` 0), [`sample_probe`] itself.
fn sample_probe_rough(
    sl: ProbeSlot,
    idx: i32,
    p: vec3<f32>,
    r: vec3<f32>,
    steps: i32,
    lobe: RoughLobe,
) -> vec3<f32> {
    let top = water_reflect.probe_box_max.z;
    if (pv_rough_gain() <= 0.0 || top < 1.0) {
        return sample_probe(sl, idx, p, r, steps);
    }
    var dir: vec3<f32>;
    if (water_reflect.probe_extra.z > 1.5) {
        dir = probe_dir_march(p, r, sl.at.xyz, idx);
    } else {
        dir = probe_dir_depth(p, r, sl.at.xyz, idx, steps);
    }
    let d_geom = textureSampleLevel(probe_tex, probe_samp, dir, idx, 0.0).a;
    let shares = probe_fog_shares(dir, d_geom, sl.at.xyz, p);
    let hit = sl.at.xyz + dir * d_geom;
    let scale = select(clamp(distance(p, hit) / max(d_geom, 1e-3), 0.0, 2.0), 1.0, d_geom > 9000.0);
    let up = lobe.up * scale;
    let side = lobe.side * scale;
    let d = rough_lift_dir(dir, lobe.lift * scale);
    let texel0 = water_reflect.probe_box_min.y;
    let spread = water_reflect.march.z * texel0;
    if (max(up, side) < max(spread, texel0)) {
        return probe_refog(rough_probe_soft(d, idx, spread), shares);
    }
    if (textureSampleLevel(probe_tex, probe_samp, d, idx, 1.0).a <= 0.0) {
        return probe_refog(rough_probe_soft(d, idx, spread), shares);
    }
    let n = f32(RR_PROBE_TAPS);
    let reach = sqrt(1.0 + (n * n - 1.0) / 3.0);
    let lod = min(rough_probe_lod(max(side, up / reach)), top);
    let s_lod = max(rough_probe_sigma(lod), texel0);
    var gap = 0.0;
    var zig = 0.0;
    if (RR_PROBE_TAPS > 1) {
        gap = min(sqrt(max(up * up - s_lod * s_lod, 0.0) * 12.0 / (n * n - 1.0)), 2.0 * s_lod);
        zig = 0.25 * s_lod;
    }
    // Up across `d` in its vertical plane, and the horizontal across it.
    let up_raw = vec3<f32>(0.0, 1.0, 0.0) - d * d.y;
    let ul = length(up_raw);
    let e_up = select(vec3<f32>(1.0, 0.0, 0.0), up_raw / max(ul, 1e-6), ul > 1e-4);
    let e_side = cross(d, e_up);
    var sum = vec3<f32>(0.0);
    for (var i = 0; i < RR_PROBE_TAPS; i = i + 1) {
        let along = (f32(i) - 0.5 * (n - 1.0)) * gap;
        let aside = select(-zig, zig, (i & 1) == 1);
        sum = sum
            + textureSampleLevel(probe_tex, probe_samp, d + e_up * along + e_side * aside, idx, lod).rgb;
    }
    return probe_refog(sum / n, shares);
}

#ifdef WATER_VIEW
/// The lobe as a debug view: r its spread up over 0.1 rad, g across over 0.02 rad, b the share of
/// it a reflected thing `d_hit` yards behind the water shows the eye `d_eye` in front. The id is
/// past the ids the other preview modules take; the integration assigns the final one.
const WV_REFL_ROUGH: u32 = 35u;

fn rough_lobe_view(lobe: RoughLobe, d_hit: f32, d_eye: f32) -> vec3<f32> {
    return vec3<f32>(
        saturate(lobe.up / 0.1),
        saturate(lobe.side / 0.02),
        rough_contact(d_hit, d_eye),
    );
}
#endif
#endif // WATER_PREVIEW


/// **The current**, Valve's two-phase flow map (Vlachos, "Water Flow in Portal 2", 2010): the ripple
/// is carried along the current for one period and restarted, twice, half a period apart, each
/// weighted by a triangle that is zero at its own restart, so neither reset is ever seen. The
/// current is the map's (`ATTRIBUTE_WOW_FLOW`); on still water both phases read the same point and
/// this is the unflowed layer exactly.
const FLOW_PERIOD_S: f32 = 1.6;

// ---- the surface's roughness (WAV-2) ------------------------------------------------------------
//
// Each ripple octave is read through its mip, never deleted, and the slope variance the mip and the
// feature fade remove becomes GGX roughness. What the five octaves lose together is a function of the
// pixel's footprint alone (`RippleStats`' per-level table at each octave's own LOD, w²-weighted), so
// `benilla_water`'s look rows tabulate it over the footprint (`look.ripple_loss`) and a pixel reads
// one row per axis.

/// Σ w² of the five octaves' weights (0.85, 1.0, 1.0, 0.70, 0.20): the fully unresolved surface's
/// slope variance in units of the map's.
const OCTAVE_W2_SUM: f32 = 3.2525;

/// The pixel's footprint on the water in world XZ: its major and minor extents in yards, log2 of
/// the major, the major axis (unit), and the map's feature wavelength in pixels at one tile a yard.
struct PixelFootprint {
    major: f32,
    minor: f32,
    log_maj: f32,
    e: vec2<f32>,
    feat_px1: f32,
};

/// The footprint from world XZ's screen derivatives, taken once for every octave in uniform control
/// flow: an octave's drift is uniform and its flow phases a smooth shift, so they add nothing, and
/// the footprint holds still while they move.
fn pixel_footprint(xz: vec2<f32>) -> PixelFootprint {
    let dx = dpdx(xz);
    let dy = dpdy(xz);
    let lx = length(dx);
    let ly = length(dy);
    var p: PixelFootprint;
    p.major = max(max(lx, ly), 1e-6);
    p.minor = max(min(lx, ly), 1e-6);
    p.log_maj = log2(p.major);
    p.e = select(dy, dx, lx >= ly) / p.major;
    p.feat_px1 = water_reflect.look.ripple_mean.w / (p.major * f32(textureDimensions(ripples).x));
    return p;
}

/// An octave's footprint: the map's feature wavelength in pixels along the major axis, and how much
/// of the octave the feature fade keeps.
struct OctaveFootprint {
    feat_px: f32,
    keep: f32,
};

/// The pixel's footprint on an octave `inv_wavelength` tiles a yard. Rotation keeps lengths, so the
/// lattice's turn does not enter.
fn octave_footprint(px: PixelFootprint, inv_wavelength: f32) -> OctaveFootprint {
    var f: OctaveFootprint;
    f.feat_px = px.feat_px1 / inv_wavelength;
    // Features under 2 px along the major axis are not resolved (Nyquist): the octave fades out
    // between 4 and 2 px and its variance joins the loss.
    f.keep = smoothstep(water_reflect.look.surf.z, water_reflect.look.surf.w, f.feat_px);
    return f;
}

/// The loss table (`look.ripple_loss`): its rows, and row 0's footprint LOD at one texel a yard
/// (log2 yards plus the view's mip bias), with as many rows to each doubling of the footprint.
const RIPPLE_LOSS_ROWS: u32 = 120u;
const RIPPLE_LOSS_FROM: f32 = -8.0;
const RIPPLE_LOSS_PER_LOD: f32 = 8.0;

/// The slope variance the five octaves lose per axis at footprint LOD `lod` (one texel a yard),
/// w²-weighted: x what their mips and their feature fade remove, y their mips' share alone.
fn ripple_loss_at(lod: f32) -> vec2<f32> {
    let t = clamp((lod - RIPPLE_LOSS_FROM) * RIPPLE_LOSS_PER_LOD, 0.0, f32(RIPPLE_LOSS_ROWS));
    let i = min(u32(t), RIPPLE_LOSS_ROWS - 1u);
    // A row holds its own point and the next one's, so one read interpolates.
    let row = water_reflect.look.ripple_loss[i];
    return mix(row.xy, row.zw, t - f32(i));
}

/// One octave's mip-filtered slope on its rotated lattice, back in world XZ (as Rᵀ g), less the
/// map's mean (the bias of its 8-bit truncation), × weight × the feature fade's keep.
fn octave_slope(uv: vec2<f32>, fp: OctaveFootprint, weight: f32, rot: vec2<f32>) -> vec2<f32> {
    let r = mat2x2<f32>(rot.x, -rot.y, rot.y, rot.x);
    let g = textureSampleBias(ripples, ripples_samp, uv, view.mip_bias).rg * 2.0 - 1.0
        - water_reflect.look.ripple_mean.xy;
    return transpose(r) * g * (weight * fp.keep);
}

/// One octave: its slope (world XZ, × weight × keep) and its feature wavelength in pixels.
struct Octave {
    slope: vec2<f32>,
    feat_px: f32,
};

/// One still octave on its own rotated lattice.
fn ripple_octave(uv: vec2<f32>, fp: OctaveFootprint, weight: f32, rot: vec2<f32>) -> Octave {
    var o: Octave;
    o.slope = octave_slope(uv, fp, weight, rot);
    o.feat_px = fp.feat_px;
    return o;
}

/// An octave's lattice coordinate: world XZ turned by `rot`, in tiles, plus its drift so far.
fn octave_uv(xz: vec2<f32>, rot: vec2<f32>, inv_wavelength: f32, drift: vec2<f32>) -> vec2<f32> {
    let r = mat2x2<f32>(rot.x, -rot.y, rot.y, rot.x);
    return (r * xz) * inv_wavelength + drift;
}

/// [`ripple_octave`] carried along `flow` (world XZ yards a second) by the two-phase flow map at
/// phase `ph0`, both phases on the unflowed footprint `fp`.
fn ripple_octave_flowing(
    xz: vec2<f32>,
    rot: vec2<f32>,
    inv_wavelength: f32,
    drift: vec2<f32>,
    weight: f32,
    flow: vec2<f32>,
    ph0: f32,
    fp: OctaveFootprint,
) -> Octave {
    let ph1 = fract(ph0 + 0.5);
    let w0 = 1.0 - abs(2.0 * ph0 - 1.0);
    // The second phase reads another stretch of the map, scaled by the current, so still water
    // keeps its one pattern.
    let apart = vec2<f32>(0.37, 0.61) * saturate(length(flow)) / inv_wavelength;
    let a = octave_slope(
        octave_uv(xz - flow * ph0 * FLOW_PERIOD_S, rot, inv_wavelength, drift),
        fp,
        weight,
        rot,
    );
    let b = octave_slope(
        octave_uv(xz - flow * ph1 * FLOW_PERIOD_S + apart, rot, inv_wavelength, drift),
        fp,
        weight,
        rot,
    );
    var o: Octave;
    o.slope = a * w0 + b * (1.0 - w0);
    o.feat_px = fp.feat_px;
    return o;
}

/// The surface at a fragment: the filtered full-band normal and the resolved low band's, each with
/// the wake; the isotropic GGX α = det(A)^¼ of the roughness tensor A (world XZ: xx, zz, xz); α's
/// place between the calm surface and the fully unresolved one; the slope scale.
struct WaterSurface {
    n: vec3<f32>,
    n_lo: vec3<f32>,
    alpha: f32,
    a2: vec3<f32>,
    rho_a: f32,
    k: f32,
};

/// An octave's slope if its features span at least ~8 px: the resolved low band.
fn octave_low(o: Octave) -> vec2<f32> {
    return o.slope * smoothstep(4.0, 8.0, o.feat_px);
}

/// The five ripple octaves, tiles of 110.4, 40.8, 12, 3.6 and 1.44 yd, each on its own rotated
/// lattice so their periods never line up, with the live wake's slope; called in uniform control
/// flow.
fn water_surface(xz: vec2<f32>, flow: vec2<f32>, wake_slope: vec2<f32>) -> WaterSurface {
    let px = pixel_footprint(xz);
    let ph0 = water_reflect.cycle.x;
    // Each octave's lattice: its turn, its tiles a yard (12 yd a tile at scale 1), its drift so
    // far in tiles.
    let rot0 = vec2<f32>(1.0, 0.0);
    let rot1 = vec2<f32>(0.8572, 0.5150);
    let rot2 = vec2<f32>(0.4540, 0.8910);
    let rot3 = vec2<f32>(-0.2924, 0.9563);
    let rot4 = vec2<f32>(-0.8572, 0.5150);
    let f0 = 1.0 / (12.0 * 9.2);
    let f1 = 1.0 / (12.0 * 3.4);
    let f2 = 1.0 / 12.0;
    let f3 = 1.0 / (12.0 * 0.30);
    let f4 = 1.0 / (12.0 * 0.12);
    let d0 = water_reflect.drift[0];
    let d1 = water_reflect.drift[1];
    let d2 = water_reflect.drift[2];
    let d3 = water_reflect.drift[3];
    let d4 = water_reflect.drift[4];
    let fp0 = octave_footprint(px, f0);
    let fp1 = octave_footprint(px, f1);
    let fp2 = octave_footprint(px, f2);
    let fp3 = octave_footprint(px, f3);
    let fp4 = octave_footprint(px, f4);
    // The two longest octaves are still; the three shorter ones ride the current.
    let o0 = ripple_octave(octave_uv(xz, rot0, f0, d0), fp0, 0.85, rot0);
    let o1 = ripple_octave(octave_uv(xz, rot1, f1, d1), fp1, 1.00, rot1);
    let o2 = ripple_octave_flowing(xz, rot2, f2, d2, 1.00, flow, ph0, fp2);
    let o3 = ripple_octave_flowing(xz, rot3, f3, d3, 0.70, flow, ph0, fp3);
    let o4 = ripple_octave_flowing(xz, rot4, f4, d4, 0.20, flow, ph0, fp4);
    let slope = o0.slope + o1.slope + o2.slope + o3.slope + o4.slope;
    let slope_lo = octave_low(o0) + octave_low(o1) + octave_low(o2) + octave_low(o3)
        + octave_low(o4);
    // `probe_box_min.w` scales the whole ripple: 1, 0 for a dead-flat surface.
    let scale = water_reflect.probe_box_min.w;
    var s: WaterSurface;
    s.k = NORMAL_STRENGTH * scale;
    s.n = normalize(vec3<f32>(
        slope.x * s.k + wake_slope.x * scale,
        1.0,
        slope.y * s.k + wake_slope.y * scale,
    ));
    // The wake is in the low band: its window sits at the camera, where its texels outsize the
    // pixels.
    s.n_lo = normalize(vec3<f32>(
        slope_lo.x * s.k + wake_slope.x * scale,
        1.0,
        slope_lo.y * s.k + wake_slope.y * scale,
    ));
    // GGX α² is twice the slope variance per axis (Beckmann): A = α0² I + 2k² cov, with cov the
    // loss at the footprint alone, so on still water α holds still while the phases move. Along
    // the footprint's major axis A takes the whole loss; across it the finer mips an anisotropic
    // sampler's probes read lose less, and the fade is the same.
    let a0 = water_reflect.look.surf.x;
    let k2 = 2.0 * s.k * s.k;
    let maj = ripple_loss_at(px.log_maj + view.mip_bias);
    var mips_min = maj.y;
#ifndef WATER_TIER_LOW
    // The sampler's anisotropy clamp N: each probe's extent is ρmax / min(ρmax / ρmin, N). Low
    // takes the loss as isotropic.
    let n = water_reflect.look.surf.y;
    if (n > 1.0) {
        mips_min = ripple_loss_at(log2(max(px.minor, px.major / n)) + view.mip_bias).y;
    }
#endif
    let c_maj = a0 * a0 + k2 * maj.x;
    let c_min = c_maj - k2 * (maj.y - mips_min);
    let d = c_maj - c_min;
    s.a2 = vec3<f32>(
        c_min + d * px.e.x * px.e.x,
        c_min + d * px.e.y * px.e.y,
        d * px.e.x * px.e.y,
    );
    // det(A) = c_maj · c_min.
    s.alpha = sqrt(sqrt(c_maj * c_min));
    let alpha_full = sqrt(a0 * a0 + k2 * OCTAVE_W2_SUM * water_reflect.look.ripple_mean.z);
    s.rho_a = smoothstep(a0, max(alpha_full, a0 + 1e-4), s.alpha);
    return s;
}

/// The slope variance the surface no longer resolves, in GGX α² ([`WaterSurface::alpha`] over the
/// calm surface's α0, which the glint's own exponent already stands for).
fn surface_lost2(surf: WaterSurface) -> f32 {
    let a0 = water_reflect.look.surf.x;
    return max(surf.alpha * surf.alpha - a0 * a0, 0.0);
}

/// The glitter lobe: Blinn at exponent `p` on the filtered normal, widened by `lost2`, the slope
/// variance the surface no longer resolves (Toksvig). Blinn p is Beckmann α² = 2/(p + 2), the
/// lost variance adds to it, and the weight follows the lobe's normalisation (p' + 2)/(p + 2), so
/// its integral, the glint's energy, stays.
fn glint_lobe_p1(
    n: vec3<f32>,
    to_light: vec3<f32>,
    to_view: vec3<f32>,
    p: f32,
    lost2: f32,
) -> f32 {
    let a2 = 2.0 / (p + 2.0) + lost2;
    let p_eff = max(2.0 / a2 - 2.0, 1.0);
    let h = normalize(to_light + to_view);
    return pow(max(dot(n, h), 0.0), p_eff) * (p_eff + 2.0) / (p + 2.0);
}

#ifdef WATER_PREVIEW
// ---- the living surface (WAV-1, WAV-3, WAV-4, WAV-5) -------------------------------------------
//
// The five ripple octaves stay as `water_surface` reads them, through their mips and the loss
// table, the three shorter ones on the flow map. Around them: each kind of water weighs them
// (`wg_wave_kind`), the wind and a world-space gust and slick field scale them (`wg_gains`), the
// wind shapes their slope so crests lie across it (`wg_wind_frame`), three analytic swell trains
// travel under them at their own phase speeds (`wg_swell`), a capillary octave rides on top
// (`wg_capillary`), and the wind's sub-pixel roughness joins the roughness tensor. Everything the
// pixel cannot resolve becomes GGX α², so the far field stays alive as roughness.

/// The prevailing wind's to-direction in world XZ, unit (`benilla_water::wind`'s
/// `WIND_PREVAILING`). The swell trains, the gust lattice and the octaves' drifts are built on it
/// and never turn, so nothing far from the origin swims when the wind veers; the live direction
/// (`waves.wind.xy`) shapes the slope and the glint.
const WG_WIND0: vec2<f32> = vec2<f32>(0.6, 0.8);
/// The prevailing wind's left normal, `(−WG_WIND0.y, WG_WIND0.x)`.
const WG_WIND0_PERP: vec2<f32> = vec2<f32>(-0.8, 0.6);
/// The wind speed at which the octaves keep their authored weights, yards a second (4 m/s).
const WG_U_REF: f32 = 4.4;
/// Yards a second per metre a second.
const WG_YD_PER_M: f32 = 1.0936;

/// The gains per kind of water on the long octaves (0, 1), the short octaves (2-4), the swell
/// trains and the capillary layer: the sea is the roughest water and carries swell, still fresh
/// water is fetch-limited with a little swell, flowing water carries none and keeps its detail on
/// the current, a WMO exterior canal and a WMO interior pool are calm.
const WG_KIND_OCEAN: vec4<f32> = vec4<f32>(1.35, 1.15, 1.0, 1.0);
const WG_KIND_LAKE: vec4<f32> = vec4<f32>(0.85, 1.05, 0.35, 1.0);
const WG_KIND_RIVER: vec4<f32> = vec4<f32>(0.45, 1.0, 0.0, 0.6);
const WG_KIND_CANAL: vec4<f32> = vec4<f32>(0.5, 0.8, 0.0, 0.7);
const WG_KIND_POOL: vec4<f32> = vec4<f32>(0.4, 0.6, 0.0, 0.4);

/// The gust and slick fields' tiles in yards, along and across the prevailing wind. The map's
/// height features span 0.23 of a tile, so cat's paws of about 19 × 14 yd and slick lanes of about
/// 59 × 20 yd.
const WG_GUST_TILE: vec2<f32> = vec2<f32>(84.0, 60.0);
const WG_SLICK_TILE: vec2<f32> = vec2<f32>(260.0, 90.0);
/// The slick field's own stretch of the map, so it never lines up with the gusts.
const WG_SLICK_SHIFT: vec2<f32> = vec2<f32>(0.37, 0.71);
/// Where each field is cut on the map's height (B, 0.5 ± 0.17), and the coverage each cut leaves
/// on average, which a pixel too coarse to resolve the field takes instead.
const WG_GUST_CUT: vec2<f32> = vec2<f32>(0.54, 0.66);
const WG_SLICK_CUT: vec2<f32> = vec2<f32>(0.66, 0.74);
const WG_GUST_MEAN: f32 = 0.33;
const WG_SLICK_MEAN: f32 = 0.12;
/// The map's height change per texel at its feature scale (0.17 · 2π / 58): the cut widens by
/// this per texel of footprint, so its edge never falls inside a pixel.
const WG_FIELD_SLOPE: f32 = 0.018;

/// The three swell trains: their to-directions (the prevailing wind turned by +0.20, −0.33 and
/// +0.55 rad), wavelengths in yards and slope amplitudes a·k at gain 1. Their periods,
/// 2π/√(g·k) with g = 10.73 yd/s², are the clock's (`benilla_water::clock`'s `SWELL_PERIOD_S`).
const WG_SWELL_DIR0: vec2<f32> = vec2<f32>(0.4291, 0.9033);
const WG_SWELL_DIR1: vec2<f32> = vec2<f32>(0.8269, 0.5624);
const WG_SWELL_DIR2: vec2<f32> = vec2<f32>(0.0934, 0.9956);
const WG_SWELL_LAMBDA: vec3<f32> = vec3<f32>(23.0, 13.7, 8.3);
const WG_SWELL_SLOPE: vec3<f32> = vec3<f32>(0.050, 0.042, 0.034);
/// The trains' second-order (Stokes) term: sharp crests, flat troughs.
const WG_SWELL_STOKES: f32 = 0.22;
/// The trains' whole slope variance at gain 1, as GGX α²: Σ (a·k)² · (1 + Stokes²).
const WG_SWELL_FULL2: f32 = 0.005682;

/// The capillary octave: the map tiled every 0.42 yd (features of about 3 cm) on its own lattice
/// turned −75°, its weight at gain 1, and how far octave 2's slope warps its lookup, in tiles per
/// unit of slope, so its tile never repeats in place.
const WG_CAP_TILE: f32 = 0.42;
const WG_CAP_ROT: vec2<f32> = vec2<f32>(0.2588, -0.9659);
const WG_CAP_WEIGHT: f32 = 0.16;
const WG_CAP_WARP: f32 = 10.0;

/// The share of Cox and Munk's (1954) wind-driven slope variance, 0.00512 per m/s on a clean
/// surface, that lies below the capillary octave and joins the roughness as GGX α².
const WG_CM: f32 = 0.35;

/// Σ w² of the long octaves (0.85, 1.0) and of the short ones (1.0, 0.7, 0.2).
const WG_W2_LONG: f32 = 1.7225;
const WG_W2_SHORT: f32 = 1.53;

/// What one kind of water does with the wind, before the gusts: the gains on the long octaves,
/// the short octaves, the swell and the capillary layer, how far the wind shapes the slope, and
/// whether the wind reaches the water at all (0 in a WMO room).
struct WaveKind {
    gain: vec4<f32>,
    aniso: f32,
    exposed: f32,
};

/// The surface's gains at a pixel: the kind's, under the wind and the gust field; `rough2` the
/// wind's sub-pixel roughness as GGX α², `aniso` the kind's wind shaping.
struct WaveGains {
    long_oct: f32,
    short_oct: f32,
    swell: f32,
    cap: f32,
    rough2: f32,
    aniso: f32,
};

/// The swell trains at a pixel: their slope as the octaves carry it (the normal's horizontal
/// part, −∇h), its resolved low band, and the slope variance the footprint no longer resolves as
/// GGX α² (xx, zz, xz).
struct Swell {
    slope: vec2<f32>,
    slope_lo: vec2<f32>,
    lost: vec3<f32>,
};

/// The capillary octave at a pixel: its slope (× weight × keep) and the slope variance its fade
/// drops, in the map's units.
struct Capillary {
    slope: vec2<f32>,
    lost: f32,
};

/// The wind lane: its direction (unit, the prevailing one while the lane is empty), its speed in
/// yards a second and the storm blend.
fn wg_wind() -> vec4<f32> {
    let lane = water_reflect.preview.wind;
    let len = length(lane.xy);
    let dir = select(WG_WIND0, lane.xy / max(len, 1e-4), len > 1e-4);
    return vec4<f32>(dir, max(lane.z, 0.0), saturate(lane.w));
}

/// The kind of water under the pixel and what it does with the wind: the sea, still fresh water
/// or flowing water (blended by the current's speed, yards a second), a WMO exterior canal or a
/// WMO interior pool; shallow water (the bed `sb` within a yard and a half), water near the shore
/// (`shore` yards from the waterline) and sloped water (`tilt`, [`surface_tilt`]) calm the long
/// octaves and the swell.
fn wg_wave_kind(flow: vec2<f32>, shore: f32, sb: SceneBehind, tilt: f32) -> WaveKind {
    let river = smoothstep(0.05, 0.4, length(flow));
    var base = mix(select(WG_KIND_LAKE, WG_KIND_OCEAN, w.kind.y > 0.5), WG_KIND_RIVER, river);
    base = select(base, WG_KIND_CANAL, w.path.x > 0.5);
    base = select(base, WG_KIND_POOL, w.path.x > 1.5);
    let indoor = w.path.x > 1.5;
    // The column under the pixel, deep where the bed is not known.
    let depth = select(8.0, max(sb.vertical, 0.0), sb.known);
    let deep_long = 0.35 + 0.65 * smoothstep(0.15, 1.5, depth);
    let deep_short = 0.6 + 0.4 * smoothstep(0.05, 0.6, depth);
    let fetch = mix(0.45, 1.0, smoothstep(1.0, 20.0, shore));
    let level = 1.0 - smoothstep(0.03, 0.12, tilt);
    var k: WaveKind;
    k.gain = base * vec4<f32>(
        deep_long * fetch * level,
        deep_short * mix(0.6, 1.0, level),
        deep_long * fetch * level,
        deep_short,
    );
    k.aniso = mix(1.0, 0.4, river) * select(1.0, 0.0, indoor);
    k.exposed = select(1.0, 0.0, indoor);
    return k;
}

/// The gust and slick fields at world `xz`: x how gusty (0 calm to 1 a cat's paw), y how slick
/// (glassy, where no gust is). Two taps of the ripple map's height in the prevailing wind's frame,
/// carried downwind by the clock (`waves.gust`), each cut at a threshold its own footprint
/// widens, and taken at its mean coverage where the pixel covers more than the field resolves.
/// Low takes the means.
fn wg_gust(xz: vec2<f32>, px: PixelFootprint) -> vec2<f32> {
#ifdef WATER_TIER_LOW
    return vec2<f32>(WG_GUST_MEAN, WG_SLICK_MEAN);
#else
    let q = vec2<f32>(dot(xz, WG_WIND0), dot(xz, WG_WIND0_PERP));
    let ofs = water_reflect.preview.gust;
    let texels = f32(textureDimensions(ripples).x);
    let lod_g = log2(max(px.major * texels / WG_GUST_TILE.y, 1e-6)) + view.mip_bias;
    let lod_s = log2(max(px.major * texels / WG_SLICK_TILE.y, 1e-6)) + view.mip_bias;
    let bg = textureSampleLevel(
        ripples,
        ripples_samp,
        q / WG_GUST_TILE + ofs.xy,
        max(lod_g, 0.0),
    ).b;
    let bs = textureSampleLevel(
        ripples,
        ripples_samp,
        q / WG_SLICK_TILE + ofs.zw + WG_SLICK_SHIFT,
        max(lod_s, 0.0),
    ).b;
    let edge_g = WG_FIELD_SLOPE * exp2(clamp(lod_g, 0.0, 6.0));
    let edge_s = WG_FIELD_SLOPE * exp2(clamp(lod_s, 0.0, 6.0));
    let gust = smoothstep(WG_GUST_CUT.x - edge_g, WG_GUST_CUT.y + edge_g, bg);
    let slick = smoothstep(WG_SLICK_CUT.x - edge_s, WG_SLICK_CUT.y + edge_s, bs);
    let g = mix(WG_GUST_MEAN, gust, 1.0 - smoothstep(3.0, 5.0, lod_g));
    let s = mix(WG_SLICK_MEAN, slick, 1.0 - smoothstep(3.0, 5.0, lod_s)) * (1.0 - g);
    return vec2<f32>(g, s);
#endif
}

/// The gains at a pixel of kind `k` under the gust field `gs` ([`wg_gust`]) and the wind `wind`
/// ([`wg_wind`]). The wind scales every slope by √(U / U_ref) (slope variance grows with the wind
/// speed); a gust lifts the short octaves up to ×1.9 and the capillaries ×2.4, a slick takes them
/// down to a fifth and a tenth; the swell is remote and ignores both. Cox and Munk's sub-pixel
/// share grows with the local wind.
fn wg_gains(k: WaveKind, gs: vec2<f32>, wind: vec4<f32>) -> WaveGains {
    let g = gs.x * k.exposed;
    let s = gs.y * k.exposed;
    let gw = mix(1.0, sqrt(clamp(wind.z / WG_U_REF, 0.25, 3.0)), k.exposed);
    var o: WaveGains;
    o.long_oct = k.gain.x * gw * mix(0.95, 1.15, g) * (1.0 - 0.25 * s);
    o.short_oct = k.gain.y * gw * mix(0.8, 1.9, g) * (1.0 - 0.8 * s);
    o.swell = k.gain.z * gw;
    o.cap = k.gain.w * gw * mix(0.6, 2.4, g) * (1.0 - 0.9 * s);
    o.rough2 = WG_CM * 0.00512 * (wind.z / WG_YD_PER_M) * (1.0 + 1.3 * g) * (1.0 - 0.85 * s)
        * k.gain.w * k.exposed;
    o.aniso = k.aniso;
    return o;
}

/// The wind's shaping of the octaves' slope, symmetric (xx, zz, xz): `1 + 0.25·a` along the
/// to-direction `dir`, `1 − 0.3·a` across it, so crests lie across the wind at about the same mean
/// variance.
fn wg_wind_frame(dir: vec2<f32>, a: f32) -> vec3<f32> {
    let along = 1.0 + 0.25 * a;
    let across = 1.0 - 0.3 * a;
    return vec3<f32>(
        along * dir.x * dir.x + across * dir.y * dir.y,
        along * dir.y * dir.y + across * dir.x * dir.x,
        (along - across) * dir.x * dir.y,
    );
}

/// The symmetric `m` (xx, zz, xz) times `v`.
fn wg_sym_mul(m: vec3<f32>, v: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(m.x * v.x + m.z * v.y, m.z * v.x + m.y * v.y);
}

/// `M·C·M` for the symmetric `m` and `c` (xx, zz, xz): the covariance `c` of a slope that `m`
/// shapes.
fn wg_sym_sandwich(m: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let p00 = m.x * c.x + m.z * c.z;
    let p01 = m.x * c.z + m.z * c.y;
    let p10 = m.z * c.x + m.y * c.z;
    let p11 = m.z * c.z + m.y * c.y;
    return vec3<f32>(p00 * m.x + p01 * m.z, p10 * m.z + p11 * m.y, p00 * m.z + p01 * m.y);
}

/// The share by which the gains scale what the octaves lose at this footprint: each octave's w²
/// by how much its feature fade has dropped (`1 − keep²`), long ones at the long gain², short
/// ones at the short gain²; the short gain² where nothing has faded, as the mips take the finest
/// octaves first. `k` the keeps of octaves 0-3, `k4` octave 4's.
fn wg_loss_gain(k: vec4<f32>, k4: f32, g: WaveGains) -> f32 {
    let lost = vec4<f32>(0.7225, 1.0, 1.0, 0.49) * (vec4<f32>(1.0) - k * k);
    let lost4 = 0.04 * (1.0 - k4 * k4);
    let den = lost.x + lost.y + lost.z + lost.w + lost4;
    let gl2 = g.long_oct * g.long_oct;
    let gs2 = g.short_oct * g.short_oct;
    let num = (lost.x + lost.y) * gl2 + (lost.z + lost.w + lost4) * gs2;
    return select(gs2, num / max(den, 1e-4), den > 1e-4);
}

/// One swell train along `d` (unit) of wavelength `lambda` yards, slope amplitude `amp`, at phase
/// `phase` (fractions of its period): its slope, as the octaves carry it, faded by its wavelength
/// in pixels along `d`, and what the fade drops as GGX α² along `d`.
fn wg_train(
    xz: vec2<f32>,
    px: PixelFootprint,
    d: vec2<f32>,
    lambda: f32,
    amp: f32,
    phase: f32,
) -> Swell {
    let c = dot(px.e, d);
    let fp = sqrt(c * c * px.major * px.major + (1.0 - c * c) * px.minor * px.minor);
    let feat = lambda / max(fp, 1e-6);
    let keep = smoothstep(water_reflect.look.surf.z, water_reflect.look.surf.w, feat);
    let th = TAU * fract(dot(xz, d) / lambda - phase);
    let s = amp * (sin(th) + WG_SWELL_STOKES * sin(2.0 * th));
    var o: Swell;
    o.slope = d * (s * keep);
    o.slope_lo = d * (s * keep * smoothstep(4.0, 8.0, feat));
    let lost = amp * amp * (1.0 + WG_SWELL_STOKES * WG_SWELL_STOKES) * (1.0 - keep * keep);
    o.lost = vec3<f32>(d.x * d.x, d.y * d.y, d.x * d.y) * lost;
    return o;
}

/// The three swell trains at `gain` ([`wg_train`]), at the clock's phases (`waves.swell`); none
/// where the gain is 0 (rivers, canals, pools).
fn wg_swell(xz: vec2<f32>, px: PixelFootprint, gain: f32) -> Swell {
    if (gain <= 0.0) {
        return Swell(vec2<f32>(0.0), vec2<f32>(0.0), vec3<f32>(0.0));
    }
    let p = water_reflect.preview.swell;
    let a = wg_train(xz, px, WG_SWELL_DIR0, WG_SWELL_LAMBDA.x, WG_SWELL_SLOPE.x * gain, p.x);
    let b = wg_train(xz, px, WG_SWELL_DIR1, WG_SWELL_LAMBDA.y, WG_SWELL_SLOPE.y * gain, p.y);
    let c = wg_train(xz, px, WG_SWELL_DIR2, WG_SWELL_LAMBDA.z, WG_SWELL_SLOPE.z * gain, p.z);
    return Swell(
        a.slope + b.slope + c.slope,
        a.slope_lo + b.slope_lo + c.slope_lo,
        a.lost + b.lost + c.lost,
    );
}

/// The capillary octave at `gain`: the map on its own lattice, drifting with the clock
/// (`waves.cap`), its lookup warped by octave 2's slope `warp`; through its mip and its feature
/// fade like any octave ([`octave_slope`]), so call it in uniform control flow. Low has none.
fn wg_capillary(xz: vec2<f32>, px: PixelFootprint, warp: vec2<f32>, gain: f32) -> Capillary {
#ifdef WATER_TIER_LOW
    return Capillary(vec2<f32>(0.0), 0.0);
#else
    let f = 1.0 / WG_CAP_TILE;
    let fp = octave_footprint(px, f);
    let uv = octave_uv(xz, WG_CAP_ROT, f, water_reflect.preview.cap.xy) + warp * WG_CAP_WARP;
    let wt = WG_CAP_WEIGHT * gain;
    var c: Capillary;
    c.slope = octave_slope(uv, fp, wt, WG_CAP_ROT);
    c.lost = wt * wt * water_reflect.look.ripple_mean.z * (1.0 - fp.keep * fp.keep);
    return c;
#endif
}

/// [`water_surface`] with the wind in it: the same five octaves on the same lattices, drifts,
/// flow phases and loss table, weighed by `wk` ([`wg_wave_kind`]) under the gusts and the wind,
/// shaped along the wind, with the capillary octave and the swell trains added and every faded
/// slope, the wind's sub-pixel share among them, in the roughness tensor. The same
/// `WaterSurface`, so every reader of it is unchanged; called in uniform control flow.
fn water_surface_wg(
    xz: vec2<f32>,
    flow: vec2<f32>,
    wake_slope: vec2<f32>,
    wk: WaveKind,
) -> WaterSurface {
    let px = pixel_footprint(xz);
    let ph0 = water_reflect.cycle.x;
    let rot0 = vec2<f32>(1.0, 0.0);
    let rot1 = vec2<f32>(0.8572, 0.5150);
    let rot2 = vec2<f32>(0.4540, 0.8910);
    let rot3 = vec2<f32>(-0.2924, 0.9563);
    let rot4 = vec2<f32>(-0.8572, 0.5150);
    let f0 = 1.0 / (12.0 * 9.2);
    let f1 = 1.0 / (12.0 * 3.4);
    let f2 = 1.0 / 12.0;
    let f3 = 1.0 / (12.0 * 0.30);
    let f4 = 1.0 / (12.0 * 0.12);
    let d0 = pv_drift(0);
    let d1 = pv_drift(1);
    let d2 = pv_drift(2);
    let d3 = pv_drift(3);
    let d4 = pv_drift(4);
    let fp0 = octave_footprint(px, f0);
    let fp1 = octave_footprint(px, f1);
    let fp2 = octave_footprint(px, f2);
    let fp3 = octave_footprint(px, f3);
    let fp4 = octave_footprint(px, f4);
    let wind = wg_wind();
    let g = wg_gains(wk, wg_gust(xz, px), wind);
    // The two longest octaves are still; the three shorter ones ride the current.
    let o0 = ripple_octave(octave_uv(xz, rot0, f0, d0), fp0, 0.85 * g.long_oct, rot0);
    let o1 = ripple_octave(octave_uv(xz, rot1, f1, d1), fp1, 1.00 * g.long_oct, rot1);
    let o2 = ripple_octave_flowing(xz, rot2, f2, d2, 1.00 * g.short_oct, flow, ph0, fp2);
    let o3 = ripple_octave_flowing(xz, rot3, f3, d3, 0.70 * g.short_oct, flow, ph0, fp3);
    let o4 = ripple_octave_flowing(xz, rot4, f4, d4, 0.20 * g.short_oct, flow, ph0, fp4);
    let cap = wg_capillary(xz, px, o2.slope, g.cap);
    let m = wg_wind_frame(wind.xy, g.aniso);
    let slope = wg_sym_mul(m, o0.slope + o1.slope + o2.slope + o3.slope + o4.slope + cap.slope);
    let slope_lo = wg_sym_mul(
        m,
        octave_low(o0) + octave_low(o1) + octave_low(o2) + octave_low(o3) + octave_low(o4),
    );
    let sw = wg_swell(xz, px, g.swell);
    // `probe_box_min.w` scales the whole surface: 1, 0 for a dead-flat one.
    let scale = water_reflect.probe_box_min.w;
    var s: WaterSurface;
    s.k = NORMAL_STRENGTH * scale;
    s.n = normalize(vec3<f32>(
        slope.x * s.k + (sw.slope.x + wake_slope.x) * scale,
        1.0,
        slope.y * s.k + (sw.slope.y + wake_slope.y) * scale,
    ));
    s.n_lo = normalize(vec3<f32>(
        slope_lo.x * s.k + (sw.slope_lo.x + wake_slope.x) * scale,
        1.0,
        slope_lo.y * s.k + (sw.slope_lo.y + wake_slope.y) * scale,
    ));
    // The roughness tensor: the loss table's view-aligned loss at the gains, the capillary
    // octave's faded share and the wind's sub-pixel share, all shaped along the wind; the swell's
    // faded share along each train; the calm surface's α0.
    let a0 = water_reflect.look.surf.x;
    let k2 = 2.0 * s.k * s.k;
    let maj = ripple_loss_at(px.log_maj + view.mip_bias);
    var mips_min = maj.y;
#ifndef WATER_TIER_LOW
    let n_af = water_reflect.look.surf.y;
    if (n_af > 1.0) {
        mips_min = ripple_loss_at(log2(max(px.minor, px.major / n_af)) + view.mip_bias).y;
    }
#endif
    let lg = wg_loss_gain(vec4<f32>(fp0.keep, fp1.keep, fp2.keep, fp3.keep), fp4.keep, g);
    let micro = k2 * cap.lost + g.rough2 * scale * scale;
    let l_maj = k2 * lg * maj.x + micro;
    let l_min = l_maj - k2 * lg * (maj.y - mips_min);
    let dd = l_maj - l_min;
    let lost_view = vec3<f32>(
        l_min + dd * px.e.x * px.e.x,
        l_min + dd * px.e.y * px.e.y,
        dd * px.e.x * px.e.y,
    );
    s.a2 = wg_sym_sandwich(m, lost_view) + sw.lost * (scale * scale)
        + vec3<f32>(a0 * a0, a0 * a0, 0.0);
    s.alpha = sqrt(sqrt(max(s.a2.x * s.a2.y - s.a2.z * s.a2.z, 1e-12)));
    let cw = WG_CAP_WEIGHT * g.cap;
    let full = a0 * a0
        + k2 * water_reflect.look.ripple_mean.z * (WG_W2_LONG * g.long_oct * g.long_oct
            + WG_W2_SHORT * g.short_oct * g.short_oct + cw * cw)
        + (g.rough2 + WG_SWELL_FULL2 * g.swell * g.swell) * scale * scale;
    s.rho_a = smoothstep(a0, max(sqrt(full), a0 + 1e-4), s.alpha);
    return s;
}

// ---- the sun and the moon on the water (SPEC-1, SPEC-2, SPEC-3, SPEC-4) ----------------------
//
// One normalised GGX lobe per body on the surface's roughness tensor, which already holds every
// slope the pixel does not resolve, so the path is a crisp glitter near the eye and widens into a
// sheen with distance, gusts and the wind. The bodies are the visible ones (`water_reflect.sun`,
// `.moon`); their light is the celestial tint (`wow_light.light_spec`, IntBand row 9, which tints
// both discs) times a radiance over the display's white and the disc's solid angle; each is
// reached as its lane's `w` says and seen through the pixel's open reflected sky.

/// The sun's and the moon's disc radius as the water reflects them, radians, overhead; toward the
/// horizon the sun's doubles and the moon's grows by half (`WowLighting::sun_disc_scale`,
/// `moon_disc_scale`). The disc widens the lobe by half its radius in slope.
const WG_SUN_RADIUS: f32 = 0.03;
const WG_MOON_RADIUS: f32 = 0.03;
/// Each body's radiance over the display's white, linear: a reflected disc and a calm path bloom,
/// a rough far path stays a sheen.
const WG_SUN_HDR: f32 = 24.0;
const WG_MOON_HDR: f32 = 10.0;
/// The moon's cast over the celestial tint's hue: night vision reads moonlight cooler.
const WG_MOON_COOL: vec3<f32> = vec3<f32>(0.80, 0.90, 1.0);
/// The highlight's knee, linear: the brightest channel approaches it and the hue holds.
const WG_SPEC_KNEE: f32 = 1.4;
/// The comic plateau on resolved glints: where the shaped peak crosses the band, a flat lift of
/// `WG_CRISP` on the share of the surface the pixel resolves, so near glints read as painted
/// shapes with an edge.
const WG_CRISP_BAND: vec2<f32> = vec2<f32>(0.55, 0.85);
const WG_CRISP: f32 = 0.35;
/// SPEC-4's art 'glint sun': the sine of the highest elevation the glint is drawn at, on the true
/// azimuth; 0 is off, the physical default. On, it is a documented deviation, and the reflected
/// disc and the glint's visibility move with it.
const WG_GLINT_ELEV_MAX: f32 = 0.0;
/// The default linear glint weight (`benilla_water::look`'s `GLINT_WEIGHT_LIN`). `look.composite.y`
/// over it scales both bodies' light, so `$WOW_WATER_GLINT` tunes the glint.
const WG_GLINT_WEIGHT_REF: f32 = 0.03575443;

/// `ω_xzᵀ·A·ω_xz` for the roughness tensor `a` (xx, zz, xz): `α_ω²·tan²θ·ω.y²` along `ω`.
fn wg_proj(a: vec3<f32>, o: vec3<f32>) -> f32 {
    return a.x * o.x * o.x + 2.0 * a.z * o.x * o.z + a.y * o.z * o.z;
}

/// The light from `l` the surface sends toward `v` per unit of the light's irradiance, both unit
/// and above the plane: GGX in slope space about world up, centred on the resolved normal `n` and
/// spread by the roughness tensor `a2` (GGX α², world XZ: xx, zz, xz) and the light's own disc
/// `disc2`; Smith's G on the tensor's α along each direction; Schlick's F on the half vector.
/// `D·G·F / (4·v.y)`.
fn wg_ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, a2: vec3<f32>, disc2: f32) -> f32 {
    if (l.y <= 0.0 || v.y <= 0.0) {
        return 0.0;
    }
    let h = normalize(v + l);
    if (h.y <= 1e-3) {
        return 0.0;
    }
    let a = a2 + vec3<f32>(disc2, disc2, 0.0);
    let det = max(a.x * a.y - a.z * a.z, 1e-12);
    // The half vector's slope against the resolved normal's.
    let d = n.xz / max(n.y, 1e-3) - h.xz / h.y;
    let q = (a.y * d.x * d.x - 2.0 * a.z * d.x * d.y + a.x * d.y * d.y) / det;
    let c2 = h.y * h.y;
    let dist = 1.0 / (3.14159265 * sqrt(det) * c2 * c2 * (1.0 + q) * (1.0 + q));
    let lam_v = 0.5 * (sqrt(1.0 + wg_proj(a, v) / (v.y * v.y)) - 1.0);
    let lam_l = 0.5 * (sqrt(1.0 + wg_proj(a, l) / (l.y * l.y)) - 1.0);
    let g = 1.0 / (1.0 + lam_v + lam_l);
    let f = F0_WATER + (1.0 - F0_WATER) * pow(1.0 - saturate(dot(v, h)), 5.0);
    return dist * g * f / (4.0 * v.y);
}

/// A body's disc radius `r` (overhead) toward elevation sine `y`, `grow` times it at the horizon.
fn wg_disc(r: f32, grow: f32, y: f32) -> f32 {
    return r * mix(grow, 1.0, smoothstep(0.0, 0.3, y));
}

/// The direction the glint is drawn toward for the body at `l`: `l` itself, or with the art
/// clamp on ([`WG_GLINT_ELEV_MAX`]) the same azimuth at that elevation when the body stands higher.
fn wg_glint_dir(l: vec3<f32>) -> vec3<f32> {
    if (WG_GLINT_ELEV_MAX <= 0.0 || l.y <= WG_GLINT_ELEV_MAX) {
        return l;
    }
    let flat_xz = normalize(l.xz + vec2<f32>(1e-6, 0.0));
    let c = sqrt(1.0 - WG_GLINT_ELEV_MAX * WG_GLINT_ELEV_MAX);
    return vec3<f32>(flat_xz.x * c, WG_GLINT_ELEV_MAX, flat_xz.y * c);
}

/// How much of a body the surface point sees along its reflected ray: the floor's open sky there
/// ([`floor_occlusion`]: the compute pass's answer on the mirror planes, the pixel's own march off
/// them), sharpened so a crown, a deck or a bank in the way takes the glint out; 1 where the floor
/// was not occluded. A glint lies where the reflected ray meets the body, so what stands on that
/// ray stands between the point and the body.
fn wg_glint_open(occ: FloorOcc) -> f32 {
    return smoothstep(0.08, 0.92, occ.v);
}

/// Whether the water is under the open sky for its glints: the eye above it, and not a WMO
/// interior pool or a room under its fog gate, whose sky is a ceiling. A function of its own
/// because `stylised_water` shadows `w` with its probe weights.
fn wg_lit_sky(eye_above: bool, room_fog: u32) -> bool {
    return eye_above && w.path.x < 1.5 && !(w.kind.z > 0.5 && room_fog != 0u);
}

/// The highlight `c` (linear) shaped for the gamma target: the brightest channel follows a soft
/// knee toward [`WG_SPEC_KNEE`] and the others keep their ratio to it, so the hue holds; on the
/// `resolved` share of the surface a flat lift where the peak crosses [`WG_CRISP_BAND`].
fn wg_spec_shape(c: vec3<f32>, resolved: f32) -> vec3<f32> {
    let m = max(c.r, max(c.g, c.b));
    if (m <= 1e-6) {
        return vec3<f32>(0.0);
    }
    let soft = WG_SPEC_KNEE * (1.0 - exp(-m / WG_SPEC_KNEE));
    let crisp = smoothstep(WG_CRISP_BAND.x, WG_CRISP_BAND.y, soft) * resolved * WG_CRISP;
    return c * ((soft + crisp) / m);
}

/// The glints in linear light, before the fog: the sun's and the moon's lobes ([`wg_ggx`]) on the
/// normal `n` and the surface's roughness tensor, toward the eye `to_view`. The sun's light is the
/// celestial tint, the moon's its hue at a fixed brightness cooled by [`WG_MOON_COOL`]; each times
/// its HDR radiance, its disc's solid angle and the glint weight's share, under its reach
/// (`sun.w`, `moon.w`) and the pixel's open reflected sky `open` ([`wg_glint_open`]). None where
/// `lit` is false ([`wg_lit_sky`]).
fn wg_glints(
    n: vec3<f32>,
    to_view: vec3<f32>,
    to_sun: vec3<f32>,
    to_moon: vec3<f32>,
    surf: WaterSurface,
    open: f32,
    lit: bool,
) -> vec3<f32> {
    if (!lit) {
        return vec3<f32>(0.0);
    }
    let weight = water_reflect.look.composite.y / WG_GLINT_WEIGHT_REF;
    let tint = water_dec(wow_light.light_spec.rgb);
    var c = vec3<f32>(0.0);
    let reach = pv_glint_reach();
    if (reach > 0.0) {
        let l = wg_glint_dir(to_sun);
        let r = wg_disc(WG_SUN_RADIUS, 2.0, l.y);
        let e = tint * (WG_SUN_HDR * 3.14159265 * r * r * weight * reach);
        c += e * wg_ggx(n, to_view, l, surf.a2, 0.25 * r * r);
    }
    if (water_reflect.moon.w > 0.0) {
        let r = wg_disc(WG_MOON_RADIUS, 1.5, to_moon.y);
        // The tint's hue at unit luminance, whatever the night row's brightness; white without one.
        let lum = luma709(tint);
        let hue = select(vec3<f32>(1.0), tint / max(lum, 1e-4), lum > 1e-4);
        let e = hue * WG_MOON_COOL
            * (WG_MOON_HDR * 3.14159265 * r * r * weight * water_reflect.moon.w);
        c += e * wg_ggx(n, to_view, to_moon, surf.a2, 0.25 * r * r);
    }
    return wg_spec_shape(c * open, 1.0 - surf.rho_a);
}
#endif // WATER_PREVIEW

// ---- the sky floor's occlusion (REF-7) ----------------------------------------------------------
//
// The sky the floor reflects is open only where nothing the frame shows stands above the water
// along the reflected ray: a canopy, a deck, a bank. Each pixel marches the depth pyramids for it
// ([`floor_march`]) and replaces the occluded share of the sky with the occluder as the frame
// shows it.

/// The sky floor cube's level-0 texel angle at a face centre (2/64 in tangent units), radians.
const SKY_LUT_TEXEL0: f32 = 0.03125;
/// Its last mip: 64² down to 2².
const SKY_LUT_TOP: f32 = 5.0;

/// The sky a surface reflects along `dir` with `alpha` of roughness it does not resolve: the dome
/// and its drawn clouds, unlit and in linear light, blurred by a Gaussian of the reflected lobe's
/// width. Slopes per axis spread as N(0, α²/2), and a reflection doubles the normal's deviation, so
/// σ ≈ √2·α. The lobe is held within half the ray's elevation: a wave sends what would leave below
/// the horizon back up off its neighbour, so the fog row under the horizon is never averaged into
/// the water and the dome's gradient over its lowest rings, which the mirrors show sharp, stays.
fn sky_floor(dir: vec3<f32>, alpha: f32) -> vec3<f32> {
    let sigma = max(min(1.41421356 * alpha, 0.5 * max(dir.y, 0.0)), SKY_LUT_TEXEL0);
    let lod = clamp(log2(sigma / SKY_LUT_TEXEL0), 0.0, SKY_LUT_TOP);
    return textureSampleLevel(sky_lut, scene_samp, dir, lod).rgb;
}

/// The sky the surface's underside reflects, in gamma: the dome's zenith and horizon stops
/// (`WowLighting::sky` rows 0 and 4) at the reflected ray's elevation, `sqrt` on the blend because
/// the dome's gradient is weighted toward its horizon stop. The sky floor is the look from above.
fn underside_sky(dir: vec3<f32>) -> vec3<f32> {
    let t = sqrt(saturate(dir.y));
    return mix(water_reflect.sky_horizon.rgb, water_reflect.sky_zenith.rgb, t);
}

/// Whether this fragment's floor is occluded: the eye above the surface (the underside is not
/// this look's), not a WMO interior pool (its floor is unlit and unoccluded), the occlusion lane
/// on, and the floor's share of the pixel worth a march, or the grading reading `rho_c`.
fn floor_occlusion_wanted(floor_share: f32, world_pos: vec3<f32>) -> bool {
    return view.world_position.y > world_pos.y
        && w.path.x < 1.5
        && water_reflect.look.quality.y > 0.5
        && (floor_share > 1.0 / 1024.0 || water_reflect.look.grade.x > 0.0);
}

/// The share of this fragment the compute pass's plane result describes: 1 within 0.75 yd of an
/// active mirror plane, 0 beyond a yard (the classification's rule: no plane spans more than a
/// yard), crossfaded between and scaled by the plane's own strength, so a plane's election and
/// drop fade rather than pop.
fn occl_plane_weight(y: f32) -> f32 {
    let a = (1.0 - smoothstep(0.75, 1.0, abs(y - water_reflect.params.x)))
        * saturate(water_reflect.params.y);
    let b = (1.0 - smoothstep(0.75, 1.0, abs(y - water_reflect.mirror2.x)))
        * saturate(water_reflect.mirror2.y);
    return max(a, b);
}

fn floor_occ_mix(a: FloorOcc, b: FloorOcc, t: f32) -> FloorOcc {
    return FloorOcc(
        mix(a.rgb, b.rgb, t),
        mix(a.v, b.v, t),
        mix(a.rho_c, b.rho_c, t),
        mix(a.known, b.known, t),
    );
}

/// The sky floor's occlusion at `world_pos`. On a mirror plane, the compute pass's quarter-
/// resolution answer at the rippled lookup point, read bilinearly, so it is smooth and never
/// binary. Off the planes the fragment marches itself, and the VERDICT comes from the unrippled
/// point along the flat reflection, so a still camera gets a still answer and no silhouette
/// flickers with the ripple; only the occluder's colour tap follows the rippled lookup point (`n`
/// slides it, as it slides the mirrors' and the probe's), shifted on screen by as much as the ripple
/// moves it. Between the two, both run and crossfade ([`occl_plane_weight`]). The plane's answer
/// is read after the march, so it is not carried through the march's loop.
fn floor_occlusion(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    flat_n: vec3<f32>,
    to_view: vec3<f32>,
) -> FloorOcc {
    let p_look = world_pos + vec3<f32>(n.x, 0.0, n.z) * water_reflect.march.y;
    let uv_look = ndc_to_uv(position_world_to_ndc(p_look).xy);
    let wp = occl_plane_weight(world_pos.y);
    if (wp >= 1.0) {
        return floor_occ_plane(uv_look);
    }
    let m = floor_march(
        world_pos + vec3<f32>(0.0, 0.05, 0.0),
        reflect(-to_view, flat_n),
        FLOOR_MARCH_LEVEL,
        FLOOR_MARCH_STEPS,
    );
    let shift = uv_look - ndc_to_uv(position_world_to_ndc(world_pos).xy);
    let marched = floor_from_march(m, FLOOR_MARCH_LEVEL, shift);
    var plane = FloorOcc(vec3<f32>(0.0), 1.0, 0.0, 0.0);
    if (wp > 0.0) {
        plane = floor_occ_plane(uv_look);
    }
    return floor_occ_mix(marched, plane, wp);
}

/// The compute pass's plane answer at `uv_look`, read bilinearly.
fn floor_occ_plane(uv_look: vec2<f32>) -> FloorOcc {
    let o = textureSampleLevel(water_occl, scene_samp, uv_look, 0.0);
    let c = textureSampleLevel(floor_occ, scene_samp, uv_look, 0.0);
    return FloorOcc(c.rgb, o.r, c.a, o.a);
}

// ---- the fog, once (REF-8) ----------------------------------------------------------------------
//
// Every term of the water is fogged once, by the world's own law at its own distance. The world the
// mirror reflects is fogged as its camera drew it, at the eye Z of the image beyond the surface;
// the probe and the floor's occluder were fogged at another distance (the face camera's, the
// frame's at the hit) and are re-based to the mirror's; the bed is the frame's, fogged at its own
// eye Z. The in-scatter, the glints and foam are fogged at the surface's, and so is the reflected
// sky: the dome is drawn unfogged, and its light crosses the fog from the surface to the eye.

/// The eye Z of the point `p`, yards: the planar depth 1.12's fog coordinate is.
fn water_eye_z(p: vec3<f32>) -> f32 {
    return -(view.view_from_world * vec4<f32>(p, 1.0)).z;
}

/// [`apply_fog`]'s GL_LINEAR law: the share of a colour that stays at eye Z `eye_z` under the
/// fog's start and end `fog_span`.
fn fog_law(eye_z: f32, fog_span: vec2<f32>) -> f32 {
    let denom = max(fog_span.y - fog_span.x, 0.001);
    return clamp((fog_span.y - eye_z) / denom, 0.0, 1.0);
}

/// The fog on a surface point: `g` its colour (gamma), `f` the share of the surface that stays.
struct WaterFog {
    g: vec3<f32>,
    f: f32,
};

/// The fog [`apply_fog`] lays on the surface point `world_pos`: the scene's, or a WMO room's haze
/// under the room gate; `f` 1 with the fog off.
fn water_fog(world_pos: vec3<f32>, room_fog: u32) -> WaterFog {
    var colour = wow_light.fog_color;
    var span = wow_light.fog_params.xy;
    if (w.kind.z > 0.5 && room_fog != 0u) {
        colour = wow_light.wmo_fog_color;
        span = wow_light.wmo_fog_params.xy;
    }
    return WaterFog(colour.rgb, select(1.0, fog_law(water_eye_z(world_pos), span), colour.w > 0.5));
}

/// A linear radiance `c` fogged as 1.12 fogs a surface: [`apply_fog`]'s mix in gamma, at the share
/// `f` that stays and the fog colour `g`; where the fog leaves all of it, `c` as it came.
fn fog_diffuse(c: vec3<f32>, f: f32, g: vec3<f32>) -> vec3<f32> {
    if (f >= 1.0) {
        return c;
    }
    return water_dec(mix(g, water_enc(c), f));
}

/// A colour `c_g` (gamma) fogged at the share `f_was` re-fogged at the share `f_now`, in the fog
/// colour `g`. Where little survived the first fog the un-fog would amplify what did (a blurred
/// near silhouette over a fogged background), so the re-base fades to the colour as it came between
/// the shares 0.35 and 0.2, and the un-fogged colour stays within 0.25 of the colour or the fog.
fn fog_rebase(c_g: vec3<f32>, f_was: f32, f_now: f32, g: vec3<f32>) -> vec3<f32> {
    let clear = clamp(
        (c_g - (1.0 - f_was) * g) / max(f_was, 0.2),
        vec3<f32>(0.0),
        max(c_g, g) + vec3<f32>(0.25),
    );
    return mix(c_g, mix(g, clear, f_now), smoothstep(0.2, 0.35, f_was));
}

/// The fog shares the probe's colour along `dir` is re-based between ([`probe_refog`]): `x` the
/// share the face camera left, at its own eye Z, the stored distance `d_geom` from the capture
/// point `capture` along the face's axis (`dir`'s largest component); `y` the share the mirror
/// would leave, at the eye Z of the hit's image in the water's plane through the surface point
/// `p`. The sky (no hit) the face drew unfogged, and its light takes the fog between the surface
/// and the eye: `x` 1, `y` the surface's. Both 1, which re-bases nothing, with the fog off, the eye
/// below the surface (its fog is the water's, not the faces') or the lane off (`refr.z`).
fn probe_fog_shares(dir: vec3<f32>, d_geom: f32, capture: vec3<f32>, p: vec3<f32>) -> vec2<f32> {
    let fog = wow_light.fog_color;
    if (fog.w <= 0.5 || water_reflect.look.body.refr.z < 0.5 || view.world_position.y <= p.y) {
        return vec2<f32>(1.0);
    }
    let span = wow_light.fog_params.xy;
    if (d_geom > 9000.0) {
        return vec2<f32>(1.0, fog_law(water_eye_z(p), span));
    }
    let hit = capture + dir * d_geom;
    let image = vec3<f32>(hit.x, 2.0 * p.y - hit.y, hit.z);
    return vec2<f32>(
        fog_law(d_geom * max(abs(dir.x), max(abs(dir.y), abs(dir.z))), span),
        fog_law(water_eye_z(image), span),
    );
}

/// The probe's colour `c_g` (gamma) fogged as the mirror fogs it: re-based from the face camera's
/// fog to the mirror's by their `shares` ([`probe_fog_shares`]), in the scene's fog, as the faces
/// draw it.
fn probe_refog(c_g: vec3<f32>, shares: vec2<f32>) -> vec3<f32> {
    return fog_rebase(c_g, shares.x, shares.y, wow_light.fog_color.rgb);
}

/// The mirror's colour `m_g` (gamma) in linear light, with the dome it drew fogged at the surface.
/// The mirror camera fogs the world at the eye Z of its image beyond the surface, which counts the
/// fog between the surface and the eye already, but draws the dome as it is, and the dome's light
/// crosses that fog too. The share of the colour that is the dome is read off its agreement with
/// the sky floor `sky` along the same flat ray (linear; the two show one sky): within a fifth all of
/// it, past two fifths none, so the world's reflection keeps the fog its image was drawn with.
fn mirror_fog(m_g: vec3<f32>, sky: vec3<f32>, fog: WaterFog) -> vec3<f32> {
    let m = water_dec(m_g);
    if (fog.f >= 1.0) {
        return m;
    }
    let off = length(m - sky) / max(length(sky), 1e-3);
    let dome = 1.0 - smoothstep(0.2, 0.4, off);
    return water_dec(mix(fog.g, m_g, mix(1.0, fog.f, dome)));
}

/// The floor's occluder fogged as the mirror fogs it. Its colour is the frame's at the hit
/// (`occ.rgb`, times the shut share `1 − v`), fogged at the hit's own eye Z; the mirror would fog
/// it at the eye Z of its image beyond the surface, along the view ray. The hit's distance from the
/// surface comes back out of `rho_c` ([`floor_rho_c`]: `D = R·√(ρ/(1 − ρ))`) along the flat
/// reflection. The scene's fog only; an open floor, a hit over 2000 yd off (ρ ≥ 0.999), a hit and
/// an image both nearer than the fog's start, and the lane off (`refr.w`) keep the colour.
fn floor_refog(
    occ: FloorOcc,
    world_pos: vec3<f32>,
    to_view: vec3<f32>,
    flat_n: vec3<f32>,
) -> FloorOcc {
    let fog = wow_light.fog_color;
    let shut = 1.0 - occ.v;
    if (fog.w <= 0.5
        || shut < 1.0 / 255.0
        || occ.rho_c >= 0.999
        || water_reflect.look.body.refr.w < 0.5) {
        return occ;
    }
    let d = FLOOR_CONTACT_REACH * sqrt(max(occ.rho_c, 0.0) / (1.0 - occ.rho_c));
    let span = wow_light.fog_params.xy;
    let f_hit = fog_law(water_eye_z(world_pos + reflect(-to_view, flat_n) * d), span);
    let f_image = fog_law(water_eye_z(world_pos - to_view * d), span);
    if (f_hit >= 1.0 && f_image >= 1.0) {
        return occ;
    }
    let c_g = fog_rebase(water_enc(occ.rgb / shut), f_hit, f_image, fog.rgb);
    return FloorOcc(water_dec(c_g) * shut, occ.v, occ.rho_c, occ.known);
}

// ---- the linear composite (FND-2) ---------------------------------------------------------------
//
// The framebuffer holds gamma values and light adds in linear, so the water is built as
// premultiplied linear layers and handed to the blender as a remainder against the snapshot: the
// pixel lands on `enc(l + t·dec(background))` wherever the framebuffer still holds the snapshot.
// `liquid_composite` mirrors every function here on the CPU and tests the algebra.

/// One premultiplied layer: `l` the linear radiance it adds, `t` how much of what is behind passes.
struct WaterLayer {
    l: vec3<f32>,
    t: vec3<f32>,
};

/// The surface: `f·r + s + (1 − f)·w_in·i` added, `(1 − f)·t_bed` passing; `w_in` weighs the
/// in-scatter and `t_bed` the bed (`w_in + t_bed ≤ 1`).
fn layer_surface(
    f: f32,
    r: vec3<f32>,
    s: vec3<f32>,
    i: vec3<f32>,
    w_in: vec3<f32>,
    t_bed: vec3<f32>,
) -> WaterLayer {
    return WaterLayer(f * r + s + (1.0 - f) * w_in * i, (1.0 - f) * t_bed);
}

/// Edge coverage `c`: the layer scaled toward nothing.
fn layer_cover(a: WaterLayer, c: f32) -> WaterLayer {
    return WaterLayer(c * a.l, vec3<f32>(1.0) - c * (vec3<f32>(1.0) - a.t));
}

/// An opaque `colour` over the layer with coverage `cover` (foam).
fn layer_over(colour: vec3<f32>, cover: f32, a: WaterLayer) -> WaterLayer {
    return WaterLayer(cover * colour + (1.0 - cover) * a.l, (1.0 - cover) * a.t);
}

/// The snapshot's texel under this fragment, the colour the blender's destination holds wherever
/// nothing drew between the snapshot and the water; 0 while the snapshot is not this view's size.
fn water_snapshot_px(frag_xy: vec2<f32>) -> vec3<f32> {
    let ok = all(vec2<i32>(textureDimensions(scene_tex)) == vec2<i32>(view.viewport.zw));
    return select(vec3<f32>(0.0), textureLoad(scene_tex, vec2<i32>(frag_xy), 0).rgb, ok);
}

#ifdef WATER_DUAL_SOURCE
var<private> water_src1: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
#endif

/// The bed the water's transmittance passes, linear: the snapshot under the pixel, and under
/// `WATER_REFRACT` the snapshot's texel at the refracted lookup `refr_uv` (a texel centre) by its
/// weight `accept`, which is 0 where the snapshot is not the viewport's size
/// ([`water_body_optics`]).
fn water_bed(snap_px: vec3<f32>, refr_uv: vec2<f32>, accept: f32) -> vec3<f32> {
    var bed = water_dec(snap_px);
#ifdef WATER_REFRACT
    // Read whatever the weight, so the read does not wait on the depth test that weighs it.
    let px = vec2<i32>(floor(refr_uv * view.viewport.zw + view.viewport.xy));
    let refr = water_dec(textureLoad(scene_tex, px, 0).rgb);
    bed = mix(bed, refr, accept);
#endif
    return bed;
}

/// The bed's radiance and transmittance and the in-scatter's weight, as the composite takes them.
struct BodyHeld {
    bed: vec3<f32>,
    t_bed: vec3<f32>,
    w_in: f32,
};

/// The refracted bed and the transmittance in half precision, four words: they cross the tiers,
/// where the shader's register peak lies at the edge of an occupancy step, in the room the
/// transmittance and the in-scatter's weight alone took.
fn body_hold(bed: vec3<f32>, t_bed: vec3<f32>, w_in: f32) -> vec4<u32> {
    return vec4<u32>(
        pack2x16float(bed.rg),
        pack2x16float(vec2<f32>(bed.b, t_bed.r)),
        pack2x16float(t_bed.gb),
        bitcast<u32>(w_in),
    );
}

/// [`body_hold`]'s words back as floats.
fn body_release(h: vec4<u32>) -> BodyHeld {
    let a = unpack2x16float(h.x);
    let b = unpack2x16float(h.y);
    let c = unpack2x16float(h.z);
    return BodyHeld(vec3<f32>(a, b.x), vec3<f32>(b.y, c), bitcast<f32>(h.w));
}

/// The water's output: `C = enc(l + t·bed)` less `β` times the snapshot, with `1 − β` in alpha, so
/// the premultiplied blend `src + β·dst` lands on `C` where `dst` is the snapshot, and a transparent
/// drawn before the water shows through attenuated by `β = enc(luma(t))`. `bed` is the bed the
/// layer passes, linear ([`water_bed`], refracted on Medium and High); the remainder always
/// subtracts the unrefracted texel `snap_px`, the one the blender's `dst` holds. `cov` is the share
/// of the pixel the surface covers.
fn water_output(
    a: WaterLayer,
    frag_xy: vec2<f32>,
    snap_px: vec3<f32>,
    bed: vec3<f32>,
    cov: f32,
) -> vec4<f32> {
    let c_lin = a.l + a.t * bed;
    let c = water_enc(c_lin);
    let beta = pow(max(dot(a.t, LUMA_709), 0.0), 1.0 / WATER_GAMMA);
#ifdef WATER_DUAL_SOURCE
    let beta3 = water_enc(a.t);
    water_src1 = vec4<f32>(beta3, beta);
    return vec4<f32>(c - beta3 * snap_px, 1.0 - beta);
#else
    return vec4<f32>(c - beta * snap_px, 1.0 - beta);
#endif
}

// ---- the reflectance (FND-2) --------------------------------------------------------------------
//
// Schlick on the filtered normal, lowered by the share of a rough lobe the split-sum env-BRDF says
// still reflects, and floored at what water reflects head-on. `liquid_composite` mirrors it.

/// Water's reflectance head-on: `((n − 1)/(n + 1))²` for n = 1.333 is 0.0204.
const F0_WATER: f32 = 0.02;

/// Karis' analytic fit of the split-sum environment BRDF ("Physically Based Shading on Mobile",
/// 2014) at reflectance `f0`, cosine `n_dot_v` and perceptual roughness `rough`.
fn env_brdf_approx(f0: f32, n_dot_v: f32, rough: f32) -> f32 {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

/// Where the fit's `min(r.x², exp2(−9.28μ))` changes arm for GGX `alpha`: `−log2((1 − √α)²)/9.28`
/// (0.0829 at the calm α0, 0.1208 at the fully unresolved surface's α).
fn fresnel_kink(alpha: f32) -> f32 {
    let rough = sqrt(clamp(alpha, 0.0, 1.0));
    return clamp(-log2(max((1.0 - rough) * (1.0 - rough), 1e-6)) / 9.28, 0.0, 1.0);
}

/// The reflectance of water with GGX roughness `alpha` at cosine `n_dot_v`: Schlick times the fit's
/// ratio of the rough lobe to the calm one, which is exact Schlick on a calm surface. Below the
/// kink the ratio falls toward grazing, a dim band under the horizon, so the cosine is held there:
/// the result never rises away from grazing, and peaks at 0.504 at α0, 0.365 at the unresolved α.
fn water_fresnel(n_dot_v: f32, alpha: f32) -> f32 {
    let rough = sqrt(clamp(alpha, 0.0, 1.0));
    let mu = clamp(max(n_dot_v, fresnel_kink(alpha)), 1e-4, 1.0);
    let schlick = F0_WATER + (1.0 - F0_WATER) * pow(1.0 - mu, 5.0);
    let k = env_brdf_approx(F0_WATER, mu, rough) / env_brdf_approx(F0_WATER, mu, 0.0);
    return max(saturate(schlick * k), F0_WATER);
}

// ---- the reflection's grading (REF-13) ----------------------------------------------------------
//
// A reflection loses a share of its contrast at every distance and more with the distance and the
// roughness that show it: the grading keeps `K = clamp(1 − s·(C_NEAR + C_FAR·ρ), 0, 1)` of it, in
// two stages, `look.grade` holding s, C_FAR and C_NEAR. `liquid_composite` mirrors it.

/// The grading's reach: the surface's unresolved roughness share `rho_a` and the reflected thing's
/// distance share `rho_c` as independent losses.
fn grade_rho(rho_a: f32, rho_c: f32) -> f32 {
    return 1.0 - (1.0 - rho_a) * (1.0 - rho_c);
}

/// The contrast the grading keeps at reach `rho`.
fn grade_keep(rho: f32) -> f32 {
    let g = water_reflect.look.grade;
    return clamp(1.0 - g.x * (g.z + g.y * rho), 0.0, 1.0);
}

/// The grading in two stages, as the affine map `r ↦ max(a·r + b, 0)` it is on the reflection `r`,
/// returned `(b, a)`. Both stages pivot on what the reflection is of: where the floor's march
/// reached a verdict, on the occluded `floor` (the occluder where it hit, the sky where it
/// escaped), keeping `K = k_n·k_f` of `r`, so water under a deck is pulled toward the deck's
/// underside and never toward the sky; where it did not, the near stage (`k_n = 1 − s·C_NEAR`)
/// has no pivot and the far stage keeps `k_f` of `r` about the open `sky`. An unknown share
/// carries no occluder and the whole sky, so `floor = known·P + (1 − known)·sky` with `P` the
/// floor of the share with a verdict: the far pivot is `floor` itself and the near pivot
/// `mix(r, P, known) = floor + (1 − known)·(r − sky)`, which counts a partial cover once. s = 0 is
/// the identity. Two terms instead of the sky and the verdict keep the tier chain light.
fn grade_affine(sky: vec3<f32>, floor: vec3<f32>, known: f32, rho: f32) -> vec4<f32> {
    let g = water_reflect.look.grade;
    let k = grade_keep(rho);
    let k_n = clamp(1.0 - g.x * g.z, 0.0, 1.0);
    let k_f = select(k / max(k_n, 1e-4), 0.0, k_n <= 1e-4);
    let a = mix(k_f, k, known);
    let b = (1.0 - k) * floor - (k_f - k) * (1.0 - known) * sky;
    return vec4<f32>(b, a);
}

// ---- the body (BODY-2): Beer-Lambert on the true path in the zone's own spectrum ---------------
//
// The bed passes `t_bed = exp(−σ_abs·L)·exp(−σ_turb·L)` per channel and the zone's colour replaces
// `w_in = 1 − exp(−σ_turb·L)` of it, along `L`, the refracted view ray's length from the surface
// down to the bed's depth plus the way down to the bed along the refracted lighting sun, the depth
// on ADT water at most the authored column's. The colour is the zone's swatch ramp (`look.body`'s
// rows from Light.dbc) read at the turbidity's optical depth, in three soft bands.
// `benilla_water::body` derives σ from the swatch on the CPU.

/// Fresh water's index of refraction and its inverse.
const WATER_IOR: f32 = 1.333;
const INV_IOR: f32 = 0.7501875;
/// The cosine of the critical angle (48.61°): a ray refracted into water lies within it.
const MU_D_MIN: f32 = 0.6612;
/// The refracted lookup's largest offset in pixels: the water's screen rectangle is grown by 32 px
/// (`world_view::WATER_RECT_MARGIN_PX`), so a lookup this far off a water pixel reads this frame's
/// linear depth, with a pixel's guard on each side.
const REFRACT_MAX_PX: f32 = 30.0;
/// Yards per unit of the authored depth coordinate, for a pixel whose bed is not known: the river
/// LUT's span (`0x810380`), and 255 over the ocean's 1.72 bytes per yard.
const REF_RIVER_V_YD: f32 = 4.6667;
const REF_OCEAN_V_YD: f32 = 148.26;

/// What `fragment` knows about the body under a water pixel.
struct BodyIn {
    // The albedo stops in gamma: swatch row 0, and the zone's line from it to row 63 at half the
    // deep ceiling and at the ceiling (`look.body.spare.y`).
    s: vec3<f32>,
    m: vec3<f32>,
    d: vec3<f32>,
    // The kind's absorption per yard per channel and its turbidity per yard.
    sig_abs: vec3<f32>,
    sig_turb: f32,
    // The lift: `look.body.look.w` × the frame sheet's mean alpha; 0 on unlit interiors.
    lift: f32,
    // The authored depth in yards: the column of a bed that is not known, and on ADT water the
    // most water a known bed's light crosses.
    fallback_z: f32,
    // A WMO interior pool: lit by 1, as the reference draws it.
    unlit: bool,
    // ADT water, which the terrain's shadow covers.
    adt: bool,
};

/// The water column under a pixel and what it does to light.
struct BodyOptics {
    // The bed was found behind the pixel, seen from above.
    known: bool,
    // The water's height above the bed, the view ray's path to it and the optical path, yards;
    // both paths cross at most the authored column on ADT water.
    z: f32,
    l_view: f32,
    l_opt: f32,
    // The bed's transmittance and the in-scatter's weight (`w_in + t_bed ≤ 1`).
    t_bed: vec3<f32>,
    w_in: f32,
    // The ramp coordinate after the bands, the albedo there in linear and gamma.
    x: f32,
    a_lin: vec3<f32>,
    a_g: vec3<f32>,
    // The bed's refracted lookup, its weight and its blur level (no refraction: the pixel's own).
    bed_uv_r: vec2<f32>,
    accept: f32,
    bed_lod: f32,
};

/// The body's light: its in-scatter radiance and the illuminant on the body and on foam.
struct BodyLight {
    in_scatter: vec3<f32>,
    irradiance: vec3<f32>,
    foam_irradiance: vec3<f32>,
};

/// Three soft bands over the ramp coordinate `x`, their edges widened by the pixel's own footprint
/// `fw` of `3x` so they never alias.
fn band_soft(x: f32, fw: f32) -> f32 {
    let y = x * 3.0;
    let e = water_reflect.look.body.look.z + min(fw, 0.5);
    return (floor(y) + smoothstep(0.5 - e, 0.5 + e, fract(y))) / 3.0;
}

/// The albedo in linear light at ramp coordinate `x` between the gamma stops `s`, `m` and `d`,
/// mixed in linear; only the two stops `x` lies between are decoded.
fn albedo_ramp(s: vec3<f32>, m: vec3<f32>, d: vec3<f32>, x: f32) -> vec3<f32> {
    let low = x < 0.5;
    let a = select(m, s, low);
    let b = select(d, m, low);
    return mix(water_dec(a), water_dec(b), select(2.0 * x - 1.0, 2.0 * x, low));
}

/// The reference's sheen lift on an albedo `a_g` (gamma): `lift` in the albedo's hue, its peak
/// channel at most `lift` (`benilla_water::body::body_lift`).
fn body_lift(a_g: vec3<f32>, lift: f32) -> vec3<f32> {
    return lift * a_g / max(max(a_g.r, max(a_g.g, a_g.b)), max(lift, 1e-4));
}

/// The map's north-west corner from its centre, 32 ADT tiles, and the shadow torus's span, 32 MCNK
/// chunks of 64 MCSH texels, in yards (`benilla_water::mcsh`'s `MAP_HALF_YD`, `MCSH_SPAN_YD`).
const MAP_HALF_YD: f32 = 17066.666;
const MCSH_SPAN_YD: f32 = 1066.6666;

/// Whether the lighting sun reaches the surface point `world_pos` over a water column `column`
/// yards deep, 1 lit and 0 shadowed: the terrain's MCSH at the ground point the sun's ray through
/// the surface meets, read bilinearly across the torus (the terrain clamps each chunk's block half
/// a texel in). Lit outside the shadow's window.
fn lighting_sun_mcsh(world_pos: vec3<f32>, column: f32) -> f32 {
    let win = water_reflect.look.body.shadow;
    let l = normalize(wow_light.light_sun.xyz);
    let g = world_pos.xz + l.xz * (min(max(column, 0.0), 16.0) / max(-l.y, 0.2));
    let inside = win.w > 0.5 && all(abs(g - win.xy) < vec2<f32>(win.z));
    let uv = fract((g + vec2<f32>(MAP_HALF_YD)) / MCSH_SPAN_YD);
    let shadowed = textureSampleLevel(water_shadow, ripples_samp, uv, 0.0).r;
    return select(1.0, 1.0 - shadowed, inside);
}

/// The light on the body and on foam, linear: the sky's irradiance seen through the open share of
/// the hemisphere and the lighting sun's under its shadow. One light, two chroma treatments.
struct Irr {
    body: vec3<f32>,
    foam: vec3<f32>,
};

/// The illuminant at the surface point `world_pos` over a column `column` yards deep, seen through
/// `vis` (`g` the sky's open share, `b` the sun's, [`body_visibility`]): the body takes the
/// chroma-bounded rows, foam the rows in the terrain light's colour, both at the terrain's
/// open-sky luminance (`benilla_water::body::body_light`), so open foam is 1.12's `FOAM_COLOR·lit`
/// and shadowed foam darkens without changing hue. On ADT water the sun is also under the terrain's
/// MCSH at the ground its ray meets; in shadow it keeps `e_sky.w` of itself, so shadowed water
/// darkens as the terrain's `terrainp_s.bls` law darkens the bank beside it (0.7 in gamma).
fn body_irradiance(world_pos: vec3<f32>, column: f32, vis: vec4<f32>, adt: bool) -> Irr {
    let b = water_reflect.look.body;
    let s = select(1.0, lighting_sun_mcsh(world_pos, column), adt) * vis.b;
    let s_eff = mix(b.e_sky.w, 1.0, s);
    return Irr(
        b.e_sky.rgb * vis.g + b.e_sun.rgb * s_eff,
        b.e_sky_foam.rgb * vis.g + b.e_sun_foam.rgb * s_eff,
    );
}

/// How much of the sky and of the lighting sun the frame shows open above the surface point
/// `world_pos` (`g` the sky, `b` the sun; `r` the reflected ray's open sky): on a mirror plane the
/// compute pass's world-anchored rays at the pixel, read bilinearly and weighed by the plane's
/// share; 1 off the planes, under the eye, and on a frame the rays were not traced (Low, or no
/// plane water on screen), where MCSH and its shadow floor alone shade the body. The specular ray's
/// verdict is never a diffuse visibility.
fn body_visibility(world_pos: vec3<f32>, frag_coord: vec2<f32>) -> vec4<f32> {
    let wp = occl_plane_weight(world_pos.y);
    if (water_reflect.look.quality.z < 0.5 || wp <= 0.0 || view.world_position.y <= world_pos.y) {
        return vec4<f32>(1.0);
    }
    let uv = (frag_coord - view.viewport.xy) / view.viewport.zw;
    let o = textureSampleLevel(water_occl, scene_samp, uv, 0.0);
    return mix(vec4<f32>(1.0), vec4<f32>(o.rgb, 1.0), wp);
}

/// The light the eye-below look lights the dome's stops with, gamma: the sky's and the sun's
/// irradiance open and unshadowed, in the terrain light's colour at its luminance, the terrain's
/// open-sky light (1.12's `lit` for an upward normal).
fn open_illuminant_g() -> vec3<f32> {
    let b = water_reflect.look.body;
    return water_enc(b.e_sky_foam.rgb + b.e_sun_foam.rgb);
}

/// The water column behind this pixel and its transmittance, the in-scatter's weight and the
/// albedo. Takes `fwidth`: call it in uniform control flow.
fn water_body_optics(
    world_pos: vec3<f32>,
    frag_coord: vec2<f32>,
    sb: SceneBehind,
    n_lo: vec3<f32>,
    flat_n: vec3<f32>,
    to_view: vec3<f32>,
    bi: BodyIn,
    wake_push: f32,
) -> BodyOptics {
    var o: BodyOptics;
    // From below the surface the column is not this pass's to draw: the deep body, opaque.
    let below = view.world_position.y < world_pos.y;
    o.known = sb.known && !below;
    // μ_d the lighting sun refracted into the water, μ_v the view ray refracted into it: the bed's
    // light reaches the eye along the refracted ray, which is never steeper than the critical
    // angle, so a grazing view crosses at most 1/0.66 yd of water per yard of depth. z is the depth
    // of the bed the bed's lookup reads: the straight ray's, and under `WATER_REFRACT` the
    // refracted lookup's by its weight. Where the bed drops away from the eye the path over-counts
    // the water, where it rises it under-counts.
    let mu_s = max(-normalize(wow_light.light_sun.xyz).y, 0.0);
    let mu_d = max(sqrt(1.0 - (1.0 - mu_s * mu_s) * INV_IOR * INV_IOR), MU_D_MIN);
    let ci = saturate(dot(flat_n, to_view));
    let mu_v = max(sqrt(1.0 - (1.0 - ci * ci) * INV_IOR * INV_IOR), MU_D_MIN);
    let uv = (frag_coord - view.viewport.xy) / view.viewport.zw;
    var z_bed = max(sb.vertical, 0.0);
    o.bed_uv_r = uv;
    o.accept = 0.0;
#ifdef WATER_REFRACT
    // The bed seen along the refracted ray. Only the ripple's tilt bends it, by (1 − 1/n) of the
    // path to the bed, at most the path over which a tenth of the bed's light survives; there is
    // no mean apparent-depth shift, so the waterline stays where it is. The tilt is the horizontal
    // part of the resolved low band's normal, the one the mirrors slide by: the octaves finer than
    // ~8 px of footprint would tear a submerged edge row to row. The normal is built about world
    // up ([`water_surface`]), so a sloping stream's grade, which `flat_n` carries, bends nothing.
    let vp = view.viewport.zw;
    let tau = vec3<f32>(n_lo.x, 0.0, n_lo.z);
    let sig_bar = dot(bi.sig_abs, LUMA_709) + bi.sig_turb;
    let reach = min(sb.path, 2.302585 / max(sig_bar, 1e-3));
    let refr = water_reflect.look.body.refr;
    let p_r = sb.bed - (1.0 - INV_IOR) * refr.x * reach * tau;
    let clip = position_world_to_clip(p_r);
    let uv_r0 = vec2<f32>(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
    // The offset in pixels, held inside the water's screen rectangle's margin, where the linear
    // depth is this frame's.
    let dpx = (uv_r0 - uv) * vp;
    let lim = min(refr.y * vp.y, REFRACT_MAX_PX);
    // The lookup is the centre of the texel whose depth is tested, so the bed read is exactly the
    // surface the rejection judged: a bilinear tap would blend in a neighbour (a pier leg's edge)
    // that no test saw.
    let at = clamp(
        floor(uv * vp + dpx * min(1.0, lim / max(length(dpx), 1e-4))),
        vec2<f32>(0.0),
        vp - 1.0,
    );
    let uv_r = (at + 0.5) / vp;
    let d_r = scene_view_z(vec2<i32>(at + view.viewport.xy));
    let p_rr = scene_world_pos(uv_r, max(d_r, 1e-3));
    // A surface in front of the water (a pier leg, a swimmer), above its plane (a bank, a rock
    // above the waterline) or standing above this pixel's own bed by more than a 45° slope over
    // the offset (a reed, a rock's submerged face) is not the bed the bent ray reaches: on screen
    // it lies on another surface point's ray, nearer the eye. The lookup is rejected, softly, so
    // silhouettes do not flicker. A snapshot not of the viewport's size (a frame of a resize)
    // holds no texel for the lookup, so neither the bed's colour nor its depth refracts there.
    let z_surf = -(view.view_from_world * vec4<f32>(world_pos, 1.0)).z;
    let a_front = smoothstep(-0.25, 0.0, d_r - z_surf);
    let a_below = 1.0 - smoothstep(0.0, 0.15, p_rr.y - world_pos.y);
    let rise = distance(p_r, sb.bed);
    let a_bed = 1.0 - smoothstep(rise, rise + 0.15, p_rr.y - sb.bed.y);
    let snap_fits = all(vec2<i32>(textureDimensions(scene_tex)) == vec2<i32>(vp));
    o.accept = select(0.0, a_front * a_below * a_bed, o.known && d_r > 0.0 && snap_fits);
    o.bed_uv_r = uv_r;
    z_bed = max(world_pos.y - mix(sb.bed.y, p_rr.y, o.accept), 0.0);
#endif
    o.z = select(bi.fallback_z, z_bed, o.known);
    // On ADT water the light crosses at most the authored column, the MCLQ depth byte in yards:
    // 1.12 draws a bed through as much water as the byte says, so a bed the artists left visible
    // under a deep lake stays visible. A WMO's byte is an opacity, not a depth.
    let z_path = select(o.z, min(o.z, bi.fallback_z), bi.adt);
    o.l_view = z_path / mu_v;
    o.l_opt = min(o.l_view + z_path / mu_d, 4096.0);
    let tau_t = bi.sig_turb * o.l_opt;
    let t_turb = exp(-tau_t);
    o.t_bed = select(vec3<f32>(0.0), exp(-bi.sig_abs * o.l_opt) * t_turb, o.known);
    o.w_in = select(1.0, 1.0 - t_turb, o.known);
    let look = water_reflect.look.body.look;
    let x0 = select(1.0 - exp(-look.x * tau_t), 1.0, below);
    let fw = fwidth(x0 * 3.0);
    o.x = saturate(mix(x0, band_soft(x0, fw), look.y) + wake_push);
    o.a_lin = albedo_ramp(bi.s, bi.m, bi.d, o.x);
    o.a_g = water_enc(o.a_lin);
    o.bed_lod = 0.0;
    return o;
}

/// The body's in-scatter: the albedo under the illuminant ([`body_irradiance`]; 1 on an unlit
/// interior, as the reference draws it), the lift on it in gamma, the ripple's shading; the
/// composite fogs it. `vis` is the open share of the sky and the sun ([`body_visibility`]),
/// `wave_tilt` the facet's tilt along the sun's bearing.
fn water_body_light(
    o: BodyOptics,
    world_pos: vec3<f32>,
    vis: vec4<f32>,
    bi: BodyIn,
    wave_tilt: f32,
) -> BodyLight {
    var l: BodyLight;
    var e = body_irradiance(world_pos, o.z, vis, bi.adt);
    if (bi.unlit) {
        e = Irr(vec3<f32>(1.0), vec3<f32>(1.0));
    }
    // `enc(a·E) = enc(a)·enc(E)` under the pure power law.
    let in_g = o.a_g * water_enc(e.body) + body_lift(o.a_g, bi.lift);
    l.in_scatter = water_dec(in_g) * (1.0 + BODY_RIPPLE_SHADE * wave_tilt);
    l.irradiance = e.body;
    l.foam_irradiance = e.foam;
    return l;
}

/// The in-scatter and foam's illuminant as the composite takes them.
struct LightHeld {
    in_scatter: vec3<f32>,
    foam: vec3<f32>,
};

/// The body's in-scatter and foam's illuminant in half precision, three words: like the bed
/// ([`body_hold`]) they cross the reflection tiers, where the shader's register peak lies at the
/// edge of an occupancy step.
fn light_hold(l: BodyLight) -> vec3<u32> {
    return vec3<u32>(
        pack2x16float(l.in_scatter.rg),
        pack2x16float(vec2<f32>(l.in_scatter.b, l.foam_irradiance.r)),
        pack2x16float(l.foam_irradiance.gb),
    );
}

/// [`light_hold`]'s words back as floats.
fn light_release(h: vec3<u32>) -> LightHeld {
    let a = unpack2x16float(h.x);
    let b = unpack2x16float(h.y);
    let c = unpack2x16float(h.z);
    return LightHeld(vec3<f32>(a, b.x), vec3<f32>(b.y, c));
}

#ifdef WATER_VIEW
// ---- the debug views (`WaterDebug::view`, `$WOW_WATER_VIEW`) -----------------------------------
//
// Compiled only into the `WATER_VIEW` variant, which the stylised pipelines take while a view is
// chosen. The id rides `flags.y`; `benilla_world::dev_state::WATER_VIEWS` names each one.
const WV_MIRROR1: u32 = 1u;
const WV_MIRROR2: u32 = 2u;
const WV_PROBE: u32 = 3u;
const WV_TIERS: u32 = 4u;
const WV_FLAT: u32 = 5u;
const WV_FLOW: u32 = 6u;
const WV_FOAM: u32 = 7u;
const WV_THICKNESS: u32 = 8u;
const WV_FRESNEL: u32 = 9u;
const WV_ROUGHNESS: u32 = 10u;
const WV_ROUGHNESS_ANISO: u32 = 11u;
const WV_SLOPE_FULL: u32 = 12u;
const WV_SLOPE_LO: u32 = 13u;
const WV_FLOOR_OCC: u32 = 14u;
const WV_SKY_FLOOR: u32 = 15u;
const WV_SIGMA_SHAPE: u32 = 16u;
// The composite's own views, which change the layer rather than paint over it: identity hands the
// blender a layer that passes everything, unrefracted (the frame must equal the snapshot), discard
// returns 0 (the frame is untouched). Then the body's: enc(t_bed), enc(in-scatter) and the albedo
// at the pixel's ramp coordinate.
const WV_COMPOSITE_IDENTITY: u32 = 17u;
const WV_DISCARD: u32 = 18u;
const WV_TRANSMITTANCE: u32 = 19u;
const WV_IN_SCATTER: u32 = 20u;
const WV_ALBEDO: u32 = 21u;
// The grading's: r ρ_α, g ρ_c, b the contrast kept K; the reflection before and after it, enc.
const WV_GRADE: u32 = 22u;
const WV_REFL_UNGRADED: u32 = 23u;
const WV_REFL_GRADED: u32 = 24u;
// The refraction's: the bed as the composite passes it, enc; the refracted lookup's weight in r,
// its offset in g as a share of `REFRACT_MAX_PX`, and in b the weight of a read that is not bed
// ([`view_refract_leak`]).
const WV_REFRACTED_BED: u32 = 25u;
const WV_REFRACT_REJECT: u32 = 26u;
// The lighting sun's terrain shadow at the surface over the pixel's column, white lit
// ([`lighting_sun_mcsh`]).
const WV_MCSH: u32 = 27u;
// The body's light: the sky's and the lighting sun's open share as the body reads them (1 off
// the mirror planes), and foam's illuminant left of the screen's centre against the body's right
// of it, enc.
const WV_SKY_VIS: u32 = 28u;
const WV_SUN_VIS: u32 = 29u;
const WV_FOAM_LIGHT: u32 = 30u;
// The surface's fog: the share that stays in r, the fog colour's green and blue in g and b.
const WV_FOG: u32 = 31u;

fn water_view_id() -> u32 {
    return u32(water_reflect.flags.y + 0.5);
}

// The absorption per yard of this material's kind (`WaterBody`'s ocean row on the ocean swatch).
fn view_sigma_abs() -> vec3<f32> {
    let body = water_reflect.look.body;
    return select(body.sig_river.rgb, body.sig_ocean.rgb, w.kind.y > 0.5);
}

// The share of the refracted bed's read that comes from a surface the rejection exists to keep
// out, by its weight: a texel whose opaque surface lies in front of this fragment (a pier leg, a
// swimmer), more than 0.15 yd above the water's plane (a bank, a rock above the waterline) or is
// sky. 0 where the read is bed.
fn view_refract_leak(world_pos: vec3<f32>, uv_r: vec2<f32>, accept: f32) -> f32 {
    let z_surf = -(view.view_from_world * vec4<f32>(world_pos, 1.0)).z;
    let d = scene_view_z(vec2<i32>(floor(uv_r * view.viewport.zw + view.viewport.xy)));
    let p = scene_world_pos(uv_r, max(d, 1e-3));
    let out = d <= 0.0 || d < z_surf - 0.25 || p.y > world_pos.y + 0.15;
    return select(0.0, accept, out);
}
#endif

#ifdef WATER_PREVIEW
// ── The shore and its contacts (preview: SHORE-1, SHORE-2, SHORE-4; BODY-7's wet faces) ────────
//
// Pure functions for `liquid.wgsl`'s stylised region, to sit just before `stylised_water`. They
// read only what that function already holds, plus two spare lanes of the look block (see
// `shore-contact.md`):
//
//   * **The waterline.** A thin, dim, broken wet line where the water touches the bank, coverage
//     prefiltered so that at distance it fades instead of becoming a pixel outline; the bed under
//     the last yard of water darkened as wet ground is. With each breath of the shore swell
//     (`cycle.y`) the water's edge draws back seaward and runs up again to the traced waterline: a
//     hand's breadth on still water, yards on a sea beach, where the drained ground keeps a glossy
//     film that dries, a foam front rides the edge, and bore foam trails each wave that breaks in
//     water shallower than its height allows.
//   * **Contacts.** Foam where a steep surface pierces the water (a post, a hull, a rock, a quay
//     wall, a dam's foot, a swimmer), found from the frame's linear depth, so any geometry on any
//     map: the pixel's own ray meeting a wall under the surface and, on High, four depth taps
//     finding a wall standing beside it. Wider on the face a current runs into and trailing
//     downstream; breathing with the swell at sea; thin and broken on still water. A beach is not
//     steep and gets none. The wall's face under the first yard of water darkens as wet stone.
//   * **Whitecaps** on the open sea, on the long octaves' crests where wave groups meet, by the
//     sea state.
//
// Foam is painted: one bubble pattern ([`foam_noise`], non-repeating) cut at the coverage, so a
// sheet opens into lace and then islands, with a crisp outer edge, a thinner interior, cream where
// it is thick and a cool lace where it is thin, lit by foam's illuminant (`light.foam`, the
// terrain light's colour under the body's shadow), so it is warm at dusk and blue under the moon.
//
// Uniformity: [`sc_geometry`] takes every derivative this needs and must be called in uniform
// control flow; the rest takes explicit gradients and `textureLoad`s and may run under any branch.

/// The wet line's width at the water's edge, yards: a hairline of foam where the water touches the
/// bank.
const SC_LINE_YD: f32 = 0.07;
/// How far past the swash's draw-back the waterline terms are evaluated, yards.
const SC_SHORE_REACH: f32 = 1.0;
/// The edge's fade over the water's column, yards: the whole water (reflection and glints included)
/// fades over the first `max(SC_EDGE_YD, 1.5·fwidth(column))` of depth, so the contact is
/// coverage-correct at any distance.
const SC_EDGE_YD: f32 = 0.06;
/// The largest column footprint the fade takes, yards: past it the neighbour is across a silhouette
/// in the depth, not down a slope.
const SC_EDGE_FW_MAX: f32 = 0.25;
/// Contact foam's reach from a steep surface at full energy, yards.
const SC_CONTACT_YD: f32 = 0.34;
/// The deepest point on a steep surface the pixel's own ray may meet and still be at a contact,
/// yards below the surface.
const SC_CONTACT_DEEP: f32 = 0.6;
/// Where a surface starts and finishes counting as a wall, in `1 − |n.y|`: about 40° to 57°.
const SC_STEEP: vec2<f32> = vec2<f32>(0.22, 0.45);
/// Contact energy on still water: thin, broken foam.
const SC_STILL_ENERGY: f32 = 0.3;
/// The meniscus band at a contact, yards, and the share of the reflection it adds.
const SC_MENISCUS_YD: f32 = 0.05;
const SC_MENISCUS_F: f32 = 0.3;
/// The wet bed's darkening where `look.shore.y` is not written.
const SC_WET_DEFAULT: f32 = 0.61;

/// The ocean's sea state, 0..1, where nothing writes `look.shore_kind.w`.
const SC_SEA_DEFAULT: f32 = 0.6;
/// How far the sea's edge draws back between two waves at sea state 1, yards.
const SC_SWASH_YD: f32 = 2.4;
/// The deepest column the backwash drains at sea state 1, yards, and on still water.
const SC_DRAIN_H: f32 = 0.14;
const SC_LAP_DRAIN_H: f32 = 0.05;
/// The uprush's share of one breath, and the backwash's ease (its edge is `x^p` of the way out).
const SC_UPRUSH: f32 = 0.32;
const SC_BACKWASH_POW: f32 = 1.35;
/// One breath, seconds: the clock's `SHORE_SWELL_S`, the period `cycle.y` counts.
const SC_SWELL_S: f32 = 5.5;
/// How long drained ground stays glossy, seconds (e-folding), and the film's share of the water's
/// reflection.
const SC_FILM_S: f32 = 2.8;
const SC_FILM_GLOSS: f32 = 0.55;
/// The along-shore spacing of the swash's tongues, yards per tile of the map.
const SC_CUSP_YD: f32 = 36.0;
/// The wave height at sea state 1, yards, and the depth-limited breaking index H/h.
const SC_WAVE_H: f32 = 0.5;
const SC_BREAK_GAMMA: f32 = 0.78;
/// The bore's trailing sheet lifetime, seconds (e-folding), and the surf band's outer limit, yards.
const SC_SHEET_S: f32 = 1.9;
const SC_SURF_YD: f32 = 14.0;
/// Gravity, yards a second squared (9.81 m/s²).
const SC_G_YD: f32 = 10.73;
/// Whitecaps start and are full this far from the shore, yards (the per-vertex distance, which
/// stops at 24).
const SC_CAP_SHORE: vec2<f32> = vec2<f32>(8.0, 20.0);

/// The bubble pattern's scale on the shore and at contacts, and on whitecaps (lookups a yard).
const SC_FOAM_SCALE: f32 = 0.6;
const SC_CAP_SCALE: f32 = 0.22;
/// Foam's albedo, gamma: cream where it is thick, a cool lace where it is thin.
const SC_FOAM_CREAM: vec3<f32> = vec3<f32>(0.96, 0.92, 0.83);
const SC_FOAM_LACE: vec3<f32> = vec3<f32>(0.78, 0.79, 0.90);

/// The High tier's search for a wall beside the pixel: the highest a point may stand above the
/// water and still be the wall's foot, yards, and the least rise per yard of distance that makes it
/// a wall (a beach climbs well under this).
const SC_PIERCE_RISE: f32 = 1.2;
const SC_PIERCE_SLOPE: f32 = 0.8;

/// The opaque surface behind a water pixel, as the shore's terms need it.
struct ScGeom {
    // Horizontal yards from the surface point to where its ray meets the opaque surface behind;
    // −1 where that surface is not known.
    h_ray: f32,
    // The water's column over that point, yards; the authored depth where it is not known.
    col: f32,
    // How steep that surface is, `1 − |n.y|`: 1 at a wall and across a silhouette in the depth.
    steep: f32,
    // The column's screen footprint (`fwidth`), yards, at most [`SC_EDGE_FW_MAX`].
    fw_col: f32,
    // The unit horizontal direction from the surface point to that point.
    to_obs: vec2<f32>,
};

/// What [`shore_contact`] reads of `stylised_water`.
struct ScIn {
    world_pos: vec3<f32>,
    frag_coord: vec2<f32>,
    // The waterline's distance, yards: `stylised_water`'s `to_shore` (exact near the line).
    to_shore: f32,
    // The interpolated per-vertex distance (`shore`), which stops at 24 yd: the sea's exposure.
    shore_far: f32,
    // The offset to the nearest waterline point, world XZ yards.
    shore_offset: vec2<f32>,
    // The current, world XZ yards a second.
    flow: vec2<f32>,
    // `fwidth(to_shore)` clamped (`stylised_water`'s `aa`), and raw.
    aa: f32,
    fw_shore: f32,
    // World XZ yards per pixel along screen x and y.
    d_xz_dx: vec2<f32>,
    d_xz_dy: vec2<f32>,
};

/// The shore's terms for the composite.
struct ShoreContact {
    // The water's coverage at its edge: the column's fade times the swash's draw-back.
    cover: f32,
    // Foam's coverage over the surface: the wet line, the swash front, the bore sheet, contact
    // foam and whitecaps.
    foam: f32,
    // Foam's albedo, linear; foam's illuminant multiplies it.
    foam_albedo: vec3<f32>,
    // The wet bed's weight: the bed passes `mix(1, look.shore.y, wet)` of itself.
    wet: f32,
    // Drained ground's gloss, 0..1.
    film: f32,
    // The meniscus' weight at the water's edge and at contacts.
    meniscus: f32,
    // Contact foam alone, for the foam view.
    contact: f32,
};

/// The contact geometry under a water pixel from what is behind it ([`scene_behind`]):
/// `fallback_col` is the authored depth the column takes where the bed is not known. Takes
/// `dpdxFine`, `dpdyFine` and `fwidth`: call it in uniform control flow.
fn sc_geometry(world_pos: vec3<f32>, sb: SceneBehind, fallback_col: f32) -> ScGeom {
    let bx = dpdxFine(sb.bed);
    let by = dpdyFine(sb.bed);
    let sx = dpdxFine(world_pos);
    let sy = dpdyFine(world_pos);
    let col = select(fallback_col, max(sb.vertical, 0.0), sb.known);
    let fw_col = fwidth(col);
    var g: ScGeom;
    g.col = col;
    g.fw_col = min(fw_col, SC_EDGE_FW_MAX);
    // The bed's facet from its neighbours. A neighbour's bed far from this one's, against the
    // surface's own step, is across a silhouette: the surface the ray met ends there, and met just
    // under the water that is a wall's edge at its waterline (deeper, a stone's on the bed).
    let nb = cross(bx, by);
    let ln = length(nb);
    let step_bed = max(length(bx), length(by));
    let step_surf = max(max(length(sx), length(sy)), 1e-4);
    let steep = select(0.0, 1.0 - abs(nb.y) / max(ln, 1e-12), ln > 1e-12);
    g.steep = select(steep, 1.0, step_bed > 4.0 * step_surf && col < 0.2);
    let d = sb.bed.xz - world_pos.xz;
    let h = length(d);
    g.h_ray = select(-1.0, h, sb.known);
    g.to_obs = select(vec2<f32>(0.0), d / max(h, 1e-5), h > 1e-5);
    return g;
}

/// [`ScGeom`] in half precision, three words: it crosses the reflection tiers to the composite,
/// where the shader's register peak lies.
fn sc_geometry_hold(g: ScGeom) -> vec3<u32> {
    return vec3<u32>(
        pack2x16float(vec2<f32>(g.h_ray, g.col)),
        pack2x16float(vec2<f32>(g.steep, g.fw_col)),
        pack2x16snorm(g.to_obs),
    );
}

/// [`sc_geometry_hold`]'s words back.
fn sc_geometry_release(h: vec3<u32>) -> ScGeom {
    let a = unpack2x16float(h.x);
    let b = unpack2x16float(h.y);
    return ScGeom(a.x, a.y, b.x, b.y, unpack2x16snorm(h.z));
}

/// The terms where the shore draws nothing: the edge's coverage `cover` and today's foam colour.
fn sc_neutral(cover: f32) -> ShoreContact {
    return ShoreContact(cover, 0.0, water_dec(FOAM_COLOR), 0.0, 0.0, 0.0, 0.0);
}

/// The edge's coverage from the water's column ([`SC_EDGE_YD`]); 1 where the bed is not known.
fn sc_edge_cover(g: ScGeom) -> f32 {
    if (g.h_ray < 0.0) {
        return 1.0;
    }
    return saturate(g.col / max(SC_EDGE_YD, 1.5 * g.fw_col));
}

/// How far a surface `1 − |n.y|` steep counts as a wall.
fn sc_steep_gate(steep: f32) -> f32 {
    return smoothstep(SC_STEEP.x, SC_STEEP.y, steep);
}

/// The share of the drained band the water covers at the breath's phase `a`: 0 drawn back to the
/// full lap, 1 up to the waterline. The uprush rises fast and eases out; the backwash drains
/// slowly and speeds up.
fn sc_swash_fill(a: f32) -> f32 {
    let up = saturate(a / SC_UPRUSH);
    let back = saturate((a - SC_UPRUSH) / (1.0 - SC_UPRUSH));
    return select(1.0 - pow(back, SC_BACKWASH_POW), 1.0 - (1.0 - up) * (1.0 - up), a < SC_UPRUSH);
}

/// Seconds since the backwash uncovered ground `d` yards from the waterline, at phase `a`, when
/// it draws back `lap` yards: in the backwash this breath's, in the uprush the last one's.
fn sc_film_age(d: f32, lap: f32, a: f32) -> f32 {
    let x = pow(saturate(d / max(lap, 1e-3)), 1.0 / SC_BACKWASH_POW);
    let a_d = SC_UPRUSH + (1.0 - SC_UPRUSH) * x;
    let da = select(a - a_d, a + 1.0 - a_d, a < a_d);
    return max(da, 0.0) * SC_SWELL_S;
}

/// The bubble pattern at `xz`, `scale` lookups a yard, spread over 0..1, with its screen
/// footprint: [`foam_noise`], carried along the current by the two-phase flow map where the water
/// moves, the two phases blended so the pattern's contrast holds.
fn sc_pattern(xz: vec2<f32>, flow: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, scale: f32) -> FoamNoise {
    var n: FoamNoise;
    if (dot(flow, flow) > 0.0225) {
        let ph0 = water_reflect.cycle.x;
        let ph1 = fract(ph0 + 0.5);
        let w0 = 1.0 - abs(2.0 * ph0 - 1.0);
        let a = foam_noise((xz - flow * ph0 * FLOW_PERIOD_S) * scale, dx * scale, dy * scale);
        let b = foam_noise(
            (xz - flow * ph1 * FLOW_PERIOD_S) * scale + vec2<f32>(0.37, 0.61),
            dx * scale,
            dy * scale,
        );
        let norm = inverseSqrt(w0 * w0 + (1.0 - w0) * (1.0 - w0));
        n.v = 0.5 + ((a.v - 0.5) * w0 + (b.v - 0.5) * (1.0 - w0)) * norm;
        n.fw = (a.fw * w0 + b.fw * (1.0 - w0)) * norm;
    } else {
        n = foam_noise(xz * scale, dx * scale, dy * scale);
    }
    // The map's height crowds about its middle; spread it over the range the cut travels.
    return FoamNoise(saturate((n.v - 0.5) * 3.5 + 0.5), n.fw * 3.5);
}

/// Foam of coverage `c` drawn through the bubble pattern `pat` of footprint `fw`: the pattern cut
/// at `1 − c`, so a sheet opens into lace and then islands as `c` falls, with a crisp outer edge and
/// a thinner interior. Where the footprint spans the pattern, the cut gives way to the mean.
fn sc_bubble_mask(c: f32, pat: f32, fw: f32) -> f32 {
    let th = 1.0 - c;
    let e = max(1.5 * fw, 0.04);
    let edge = smoothstep(th - e, th + e, pat);
    let inner = smoothstep(th, th + 0.35, pat);
    let far = smoothstep(0.25, 0.6, fw);
    return mix(edge * (0.62 + 0.38 * inner), c, far) * smoothstep(0.0, 0.04, c);
}

/// The long octaves' crests at `xz`, 0..1: the 40.8 yd octave's height where the 110.4 yd one's
/// is high, so caps gather on the groups' crests and come and go as the two drift past each other.
/// The lattices are `water_surface`'s `o0` and `o1`; two taps.
fn sc_crest(xz: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) -> f32 {
    let rot0 = vec2<f32>(1.0, 0.0);
    let rot1 = vec2<f32>(0.8572, 0.5150);
    let f0 = 1.0 / (12.0 * 9.2);
    let f1 = 1.0 / (12.0 * 3.4);
    let z = vec2<f32>(0.0);
    let b0 = textureSampleGrad(
        ripples,
        ripples_samp,
        octave_uv(xz, rot0, f0, pv_octave_drift(0)),
        octave_uv(dx, rot0, f0, z),
        octave_uv(dy, rot0, f0, z),
    ).b;
    let b1 = textureSampleGrad(
        ripples,
        ripples_samp,
        octave_uv(xz, rot1, f1, pv_octave_drift(1)),
        octave_uv(dx, rot1, f1, z),
        octave_uv(dy, rot1, f1, z),
    ).b;
    let c0 = saturate((b0 - 0.5) * 3.5 + 0.5);
    let c1 = saturate((b1 - 0.5) * 3.5 + 0.5);
    return c1 * smoothstep(0.3, 0.7, c0);
}

/// One tap of the wall search: the depth `o` pixels off, and whether the point it shows rises
/// steeply out of the water within `reach` yards of `world_pos`, as (horizontal distance, unit
/// direction), 1000 for none. A pier leg, a hull or a rock face rises faster than
/// [`SC_PIERCE_SLOPE`] a yard; a beach climbs far less, and a far bank or a body near the eye is not
/// near this point at all.
fn sc_pierce_tap(world_pos: vec3<f32>, frag_coord: vec2<f32>, o: vec2<f32>, reach: f32) -> vec3<f32> {
    let px = frag_coord + o;
    let z = scene_view_z(vec2<i32>(floor(px)));
    let uv = (px - view.viewport.xy) / view.viewport.zw;
    let s = scene_world_pos(uv, max(z, 1e-3));
    let rise = s.y - world_pos.y;
    let d = s.xz - world_pos.xz;
    let dh = length(d);
    let ok = z > 0.0 && rise > -0.05 && rise < SC_PIERCE_RISE && dh < reach
        && rise > SC_PIERCE_SLOPE * dh;
    return select(vec3<f32>(1000.0, 0.0, 0.0), vec3<f32>(dh, d / max(dh, 1e-5)), ok);
}

/// The nearest wall standing beside the pixel within `reach` yards, from four depth taps on the
/// screen axes `reach` away (`yd_px` yards a pixel, held to 1.5-16 px, inside the depth's margin
/// round the water's rectangle): (distance, unit direction), 1000 for none. Skipped where the depth
/// pyramid shows nothing nearer than the water in the pixel's neighbourhood, as over open water.
fn sc_pierce(world_pos: vec3<f32>, frag_coord: vec2<f32>, yd_px: f32, reach: f32) -> vec3<f32> {
    let r = clamp(reach / max(yd_px, 1e-4), 1.5, 16.0);
    let top = i32(textureNumLevels(hiz_tex)) - 1;
    let lvl = clamp(i32(ceil(log2(2.0 * r))) + 1, 0, top);
    let px = vec2<i32>(frag_coord - view.viewport.xy);
    let near = textureLoad(hiz_tex, px >> vec2<u32>(u32(lvl)), lvl).r;
    if (near <= law_world_to_screen(world_pos).z) {
        return vec3<f32>(1000.0, 0.0, 0.0);
    }
    var best = sc_pierce_tap(world_pos, frag_coord, vec2<f32>(r, 0.0), reach);
    let b = sc_pierce_tap(world_pos, frag_coord, vec2<f32>(-r, 0.0), reach);
    best = select(best, b, b.x < best.x);
    let c = sc_pierce_tap(world_pos, frag_coord, vec2<f32>(0.0, r), reach);
    best = select(best, c, c.x < best.x);
    let d = sc_pierce_tap(world_pos, frag_coord, vec2<f32>(0.0, -r), reach);
    best = select(best, d, d.x < best.x);
    return best;
}

/// The shore's terms at a water pixel seen from above: `i` what `stylised_water` holds, `g` the
/// contact geometry ([`sc_geometry_release`]). Gains from the look's shore rows: `look.shore` the
/// wet line, the wet bed's darkening, contact foam and the meniscus; `look.shore_kind` the wet
/// line on still water, rapids and the ocean, and in `w` the sea state.
fn shore_contact(i: ScIn, g: ScGeom) -> ShoreContact {
    let gains = water_reflect.look.shore;
    let kinds = water_reflect.look.shore_kind;
    let ocean = w.kind.y > 0.5;
    let speed = length(i.flow);
    let rapids = smoothstep(0.8, 2.2, speed);
    let sea = select(0.0, pv_sea(), ocean && pv_on(PV_SURF));
    let aa = i.aa;
    let xz = i.world_pos.xz;
    // World yards a pixel spans on the surface.
    let yd_px = max(sqrt(length(i.d_xz_dx) * length(i.d_xz_dy)), 1e-4);

    var o = sc_neutral(sc_edge_cover(g));
    o.foam_albedo = water_dec(SC_FOAM_CREAM);
    // The opaque foam's coverage before the bubble cut, and the wet line's.
    var foam_c = 0.0;
    var line = 0.0;
    // The breath's phase where the waterline sets it; out at sea, a slow plane wave.
    var a = fract(water_reflect.cycle.y + dot(xz, vec2<f32>(0.013, 0.017)));

    // ---- the waterline -----------------------------------------------------------------------
    let lap = select(SHORE_RUNUP_YARDS, max(SC_SWASH_YD * sea, SHORE_RUNUP_YARDS), ocean);
    var surf = ocean && sea > 0.0 && i.to_shore < SC_SURF_YD;
#ifdef WATER_TIER_LOW
    surf = false;
#endif
    if (i.to_shore < lap + SC_SHORE_REACH || surf) {
        // The nearest waterline point: a field read there varies along the shore and not across
        // it, so the whole band breathes as one and the swash's tongues run up the beach.
        let at = xz + i.shore_offset;
        let ph = textureSampleGrad(
            ripples,
            ripples_samp,
            at / SHORE_SWELL_SPREAD,
            i.d_xz_dx / SHORE_SWELL_SPREAD,
            i.d_xz_dy / SHORE_SWELL_SPREAD,
        ).b;
        a = fract(water_reflect.cycle.y + 2.0 * ph);
        let fill = sc_swash_fill(a);
        let cusp = textureSampleGrad(
            ripples,
            ripples_samp,
            at / SC_CUSP_YD + vec2<f32>(0.31, 0.77),
            i.d_xz_dx / SC_CUSP_YD,
            i.d_xz_dy / SC_CUSP_YD,
        ).b;
        let tongue = saturate((cusp - 0.5) * 3.5 + 0.5) - 0.5;
        // The water's edge this instant, yards seaward of the waterline.
        let edge = max(lap * (1.0 - fill) * (1.0 + 0.7 * tongue), 0.0);
        let d_edge = i.to_shore - edge;
        // A waterline the depth shows standing in deep water is a seam between two batches of the
        // same water, not a shore: the line, the front and the meniscus stay off it.
        let real = select(1.0, 1.0 - smoothstep(0.3, 0.7, g.col), g.h_ray >= 0.0);

        // Shoreward of the edge, thin water has drained: the ground shows, wet, and glossy for a
        // while after the water left it.
        let h_drain = select(SC_LAP_DRAIN_H, max(SC_DRAIN_H * sea, SC_LAP_DRAIN_H), ocean);
        let open = 1.0 - smoothstep(edge - aa, edge + aa, i.to_shore);
        let thin = 1.0 - smoothstep(0.6 * h_drain, h_drain, g.col);
        let drained = open * thin;
        o.cover = o.cover * (1.0 - drained);
        o.film = drained * exp(-sc_film_age(i.to_shore, lap, a) / SC_FILM_S);
        // The bed under the last of the water, and the drained ground, wet.
        o.wet = (1.0 - smoothstep(lap + 0.25, lap + 0.75, i.to_shore))
            * (1.0 - smoothstep(0.22, 0.5, g.col));
        // The wet line and the meniscus ride the edge: hairlines, and at distance only the share of
        // the pixel they cover, so neither becomes an outline.
        let fw = max(i.fw_shore, 1e-4);
        o.meniscus = (1.0 - smoothstep(0.0, SC_MENISCUS_YD + aa, d_edge)) * smoothstep(-aa, 0.0, d_edge)
            * saturate(SC_MENISCUS_YD / fw) * real;
        let hw = max(0.5 * SC_LINE_YD, aa);
        let shape = 1.0 - smoothstep(0.5 * hw, hw, abs(d_edge - hw));
        let kind_gain = select(mix(kinds.x, kinds.y, rapids), kinds.z, ocean);
        line = gains.x * kind_gain * shape * saturate(SC_LINE_YD / fw) * real;

        if (ocean && sea > 0.0) {
            // The swash front: dense on the uprush, thinning as the backwash drains.
            let up = a < SC_UPRUSH;
            let back_s = max(a - SC_UPRUSH, 0.0) * SC_SWELL_S;
            let strength = select(0.55 * exp(-back_s / 1.8) + 0.15, 0.9, up);
            let wf = (0.2 + 0.45 * sea) * select(0.6, 1.0, up);
            let front = (1.0 - smoothstep(0.35 * wf, wf, d_edge)) * smoothstep(-aa, aa, d_edge);
            foam_c = max(foam_c, front * strength * smoothstep(0.0, 0.3, sea) * real);
        }
        if (surf) {
            // A wave travels in at the shallow-water speed √(g·h) and breaks where the water is
            // shallower than its height allows (H/h > 0.78); the time to the shore over a sloping
            // bed is 2d/√(g·h), and the bore reaches the waterline as the uprush begins. Behind each
            // bore a foam sheet thins with its age.
            let hb = SC_WAVE_H * sea / SC_BREAK_GAMMA;
            let c = sqrt(SC_G_YD * max(g.col, 0.06));
            let tau = min(2.0 * i.to_shore / c, 3.0 * SC_SWELL_S);
            let fr = fract(a + tau / SC_SWELL_S);
            let fw_fr = clamp(2.0 * i.fw_shore / (c * SC_SWELL_S), 0.005, 0.25);
            let sheet = max(exp(-fr * SC_SWELL_S / SC_SHEET_S), smoothstep(1.0 - fw_fr, 1.0, fr));
            let breaking = (1.0 - smoothstep(0.75 * hb, hb, g.col))
                * smoothstep(edge, edge + 0.4, i.to_shore)
                * (1.0 - smoothstep(0.7 * SC_SURF_YD, SC_SURF_YD, i.to_shore));
            foam_c = max(foam_c, breaking * sheet * (0.35 + 0.5 * sea));
        }
    }

    // ---- contacts ----------------------------------------------------------------------------
    let wall = sc_steep_gate(g.steep) * select(0.0, 1.0, g.h_ray >= 0.0);
    // A wall's face under the first yard of water, seen through it, wet.
    o.wet = max(o.wet, wall * (1.0 - smoothstep(0.35, 0.9, g.col)));
    let energy = select(max(SC_STILL_ENERGY, saturate(speed * 0.5)), 0.55 + 0.45 * sea, ocean);
    let breath = select(1.0, 1.0 + 0.5 * sea * (0.5 - 0.5 * cos(TAU * a)), ocean);
    let reach = SC_CONTACT_YD * (0.5 + 0.5 * energy) * breath;
    var dc = select(1000.0, g.h_ray, wall > 0.0 && g.col < SC_CONTACT_DEEP);
    var gate = wall * (1.0 - smoothstep(0.6 * SC_CONTACT_DEEP, SC_CONTACT_DEEP, g.col));
    var dir = g.to_obs;
#ifdef WATER_TIER_HIGH
    let p = sc_pierce(i.world_pos, i.frag_coord, yd_px, 2.0 * reach);
    if (p.x < dc) {
        dc = p.x;
        dir = p.yz;
        gate = 1.0;
    }
#endif
    if (dc < 4.0 * reach) {
        // The current piles foam on the face it runs into and draws a tail off the far one.
        let fdir = select(vec2<f32>(0.0), i.flow / max(speed, 1e-4), speed > 1e-3);
        let along = dot(dir, fdir);
        let fl = saturate(speed / 1.5);
        let tail = fl * max(-along, 0.0);
        let r = reach * (1.0 + 0.8 * fl * max(along, 0.0)) * (1.0 + 2.2 * tail);
        let band = 1.0 - smoothstep(0.45 * r, r, dc);
        o.contact = gate * band * (1.0 - 0.45 * tail) * gains.z * (0.35 + 0.65 * energy);
        foam_c = max(foam_c, o.contact);
        o.meniscus = max(
            o.meniscus,
            gate * (1.0 - smoothstep(0.0, SC_MENISCUS_YD + yd_px, dc)) * saturate(SC_MENISCUS_YD / yd_px),
        );
    }

    // ---- the foam ------------------------------------------------------------------------------
    var thick = 0.0;
    if (foam_c > 0.0 || line > 0.0) {
        let pat = sc_pattern(xz, i.flow, i.d_xz_dx, i.d_xz_dy, SC_FOAM_SCALE);
        let m = sc_bubble_mask(foam_c, pat.v, pat.fw);
        // The line breaks into beads and dashes along the shore.
        let beads = line * (0.35 + 0.65 * smoothstep(0.3, 0.7, pat.v));
        o.foam = max(m, beads);
        thick = saturate((pat.v - (1.0 - foam_c)) / 0.35) * step(beads, m);
    }
#ifndef WATER_TIER_LOW
    if (ocean && sea > 0.05 && i.shore_far > SC_CAP_SHORE.x) {
        let crest = sc_crest(xz, i.d_xz_dx, i.d_xz_dy);
        let thr = mix(0.93, 0.72, sea);
        let cap = smoothstep(thr, thr + 0.1, crest)
            * smoothstep(SC_CAP_SHORE.x, SC_CAP_SHORE.y, i.shore_far)
            * (0.6 + 0.4 * sea);
        if (cap > 0.0) {
            let lace = sc_pattern(xz, vec2<f32>(0.0), i.d_xz_dx, i.d_xz_dy, SC_CAP_SCALE);
            let mc = sc_bubble_mask(cap, lace.v, lace.fw);
            thick = select(thick, saturate((lace.v - (1.0 - cap)) / 0.35), mc > o.foam);
            o.foam = max(o.foam, mc);
        }
    }
#endif
    // Cream where no shore foam lies, so river and wake foam are not lace.
    o.foam_albedo = water_dec(select(SC_FOAM_CREAM, mix(SC_FOAM_LACE, SC_FOAM_CREAM, thick), o.foam > 0.0));
    return o;
}

/// The shore's terms on the surface's layer, after its edge coverage: the bed under the wet band
/// passes `look.shore.y` of itself; drained ground keeps a film's share of the reflection `refl`
/// and the glints `spec`, at the surface's reflectance `f_surf`; the meniscus adds a share of the
/// reflection. Linear light; identity on [`sc_neutral`]'s terms.
fn sc_layer(a: WaterLayer, sc: ShoreContact, f_surf: f32, refl: vec3<f32>, spec: vec3<f32>) -> WaterLayer {
    let gains = water_reflect.look.shore;
    let f_wet = select(SC_WET_DEFAULT, gains.y, gains.y > 0.0);
    let t = a.t * mix(1.0, f_wet, sc.wet);
    let film = (1.0 - sc.cover) * sc.film * SC_FILM_GLOSS * (f_surf * refl + spec);
    let rim = sc.meniscus * gains.w * SC_MENISCUS_F * refl;
    return WaterLayer(a.l + film + rim, t);
}
#endif // WATER_PREVIEW

fn stylised_water(
    world_pos: vec3<f32>,
    frag_coord: vec2<f32>,
    // What is behind this fragment ([`scene_behind`]); where it is not known, every term below
    // falls back to the authored depth byte.
    sb: SceneBehind,
    shore: f32,
    shore_offset: vec2<f32>,
    // The heightfield's smooth normal (zero where the mesh has none).
    smooth_normal: vec3<f32>,
    /// The mirrors' weight, then the two probe spots and the second's share — see
    /// `ATTRIBUTE_WOW_PLANAR`.
    planar: vec4<f32>,
    /// The current, world XZ yards a second — see `ATTRIBUTE_WOW_FLOW`.
    flow: vec2<f32>,
    // The body's stops, optics rows and light ([`water_body_optics`]).
    bi: BodyIn,
    room_fog: u32,
) -> vec4<f32> {
    let xz = world_pos.xz;
    // Yards of water between this fragment and the opaque surface behind it along the view axis,
    // negative where that is not known.
    let thickness = select(-1.0, sb.gap_z, sb.known);
    // In uniform control flow, before anything branches — see [`surface_tilt`].
    let tilt = surface_tilt(world_pos);
    // Beside the tilt, and for the same reason: both read screen-space derivatives, which are only
    // defined before the per-fragment branches below. This is the direction the march is cast on.
    // The heightfield's smooth normal where the mesh carries one, so the reflection bends across a
    // slope instead of stepping at each triangle; the facet's own normal otherwise. Both read in
    // uniform control flow.
    let facet_n = surface_normal(world_pos);
    let flat_n = select(facet_n, normalize(smooth_normal), dot(smooth_normal, smooth_normal) > 0.25);
#ifndef WATER_REFRACT
    // The snapshot under this pixel, read once: the composite's bed and the remainder's subtrahend.
    let snap_px = water_snapshot_px(frag_coord);
#endif

    // ---- the live wave field ----------------------------------------------------------------
    //
    // The ambient ripple is a texture of waves. This is water that has actually been pushed on: a
    // height field integrated on the CPU under the 2-D wave equation, with every swimmer in range a
    // moving source in it (`benilla_world::liquid::ripple_sim`). What arrives here is its slope,
    // and it is simply added to the ambient ripple's — a wake is not a decal drawn over the water,
    // it is the water being a different shape.
    //
    // Sampled UNCONDITIONALLY, with the window handled afterwards: a `textureSample` inside an `if`
    // is non-uniform control flow, its implicit derivatives are undefined there, and WGSL rejects
    // it. The sampler clamps, so a lookup from outside the window returns its rim, which the CPU's
    // absorbing band has already brought to zero.
    let sim_uv = (xz - water_reflect.sim.xy) * water_reflect.sim.z;
    let sim_tex = textureSample(
        wake_tex,
        wake_samp,
        clamp(sim_uv, vec2<f32>(0.0), vec2<f32>(1.0)),
    );
    let inside = select(
        0.0,
        1.0,
        all(sim_uv > vec2<f32>(0.0)) && all(sim_uv < vec2<f32>(1.0)),
    );
    // Distance to the nearest edge of the window, in window fractions — the fade the far side of
    // [`WAKE_EDGE_FADE`] describes.
    let sim_edge = min(min(sim_uv.x, 1.0 - sim_uv.x), min(sim_uv.y, 1.0 - sim_uv.y));
    let sim_w = min(water_reflect.sim.w, 1.0)
        * inside
        * smoothstep(0.0, WAKE_EDGE_FADE, sim_edge);
    let wake_slope = (sim_tex.rg * 2.0 - 1.0) * sim_w;
    // The wave's own height, signed — how much water the disturbance has put under this pixel.
    let wake_height = (sim_tex.b * 2.0 - 1.0) * sim_w;
    let wake_foam = saturate((length(wake_slope) - WAKE_FOAM_FLOOR) * WAKE_FOAM_GAIN);

    // The surface: the ripple octaves through their mips, the wake, and the roughness the mips
    // removed ([`water_surface`]).
#ifdef WATER_PREVIEW
    // `waves`: the wind's field around the same octaves ([`water_surface_wg`]); a uniform branch,
    // so both arms' implicit-LOD taps stay legal.
    var surf: WaterSurface;
    if (pv_on(PV_WAVES)) {
        surf = water_surface_wg(xz, flow, wake_slope, wg_wave_kind(flow, shore, sb, tilt));
    } else {
        surf = water_surface(xz, flow, wake_slope);
    }
#else
    let surf = water_surface(xz, flow, wake_slope);
#endif
    // The reflections' lookup point slides by the resolved low band only: the octaves finer than
    // ~8 px of footprint would make a reflected trunk wander sideways row to row.
#ifdef WATER_VIEW
    // `flat` is a mode, not a paint: the whole look runs on the unrippled normal.
    let flat_view = water_view_id() == WV_FLAT;
    let n = select(surf.n, flat_n, flat_view);
    let n_slide = select(surf.n_lo, flat_n, flat_view);
#else
    let n = surf.n;
    let n_slide = surf.n_lo;
#endif
    let to_view = normalize(view.world_position.xyz - world_pos);
#ifdef WATER_PREVIEW
    // `rough`: the reflections' lobe, the surface's unresolved roughness; none from below.
    let rr_gain = select(0.0, pv_rough_gain(), view.world_position.y > world_pos.y);
    let lobe = rough_lobe(surf.a2, to_view, flat_n, rr_gain);
#endif

    // The wake rides the body's column with the world's own bathymetry — see [`WAKE_BODY_DEPTH`].
    let wake_push = wake_height * WAKE_BODY_DEPTH;
    // The sun's bearing, taken flat — see [`BODY_RIPPLE_SHADE`]. The epsilon is for the frame at
    // startup before the lanes are written, and for the instant the sun crosses the zenith.
    let sun_xz = normalize(water_reflect.sun.xz + vec2<f32>(1e-4, 1e-4));
    let wave_tilt = dot(n.xz, sun_xz);
    // The body as transport: the column behind the pixel, its transmittance and its albedo, in
    // uniform control flow for the bands' `fwidth`; then its light.
    let optics = water_body_optics(world_pos, frag_coord, sb, n_slide, flat_n, to_view, bi, wake_push);
#ifdef WATER_PREVIEW
    // `shore`: the contact geometry while derivatives are defined, held across the tiers.
    let sc_held = sc_geometry_hold(sc_geometry(world_pos, sb, bi.fallback_z));
#endif
#ifdef WATER_REFRACT
    // The refracted bed, read here where its lookup is made, held with the bed's transmittance
    // ([`body_hold`]) across the tiers to the composite.
    let held = body_hold(
        water_bed(water_snapshot_px(frag_coord), optics.bed_uv_r, optics.accept),
        optics.t_bed,
        optics.w_in,
    );
#endif
    // The light on the body: the sky and the lighting sun through what the frame shows standing
    // over the surface, and the sun under the terrain's shadow.
    let vis = body_visibility(world_pos, frag_coord);
    let body_light = water_body_light(optics, world_pos, vis, bi, wave_tilt);
    // Held to the composite in half precision ([`light_hold`]).
    let light_held = light_hold(body_light);

    // What the surface reflects where no tier has a better answer: the sky, the dome and its clouds
    // as drawn, unlit and linear ([`sky_floor`]). The sky is at infinity, where the mirrors' law
    // (the flat reflection of a point the ripple slides sideways) reads it along the flat reflected
    // ray; the floor reads it there too, blurred only by the slope variance the surface no longer
    // resolves, so a mirror and the floor show one sky and the ripple shows through the Fresnel
    // and the body as on mirror water.
    let r_rip = reflect(-to_view, n);
#ifdef WATER_PREVIEW
    // `rough_sky`: the floor on the mirrors' lobe, lifted and blurred alike, where their chain runs.
    var sky_dir = reflect(-to_view, flat_n);
    var sky_alpha = sqrt(surface_lost2(surf));
    if (rr_gain > 0.0 && pv_on(PV_ROUGH_SKY) && rough_chain_ready(false)) {
        sky_dir = rough_lift_dir(sky_dir, lobe.lift);
        sky_alpha = rough_floor_alpha(lobe, sky_alpha);
    }
    let sky = sky_floor(sky_dir, sky_alpha);
#else
    let sky = sky_floor(reflect(-to_view, flat_n), sqrt(surface_lost2(surf)));
#endif
    // Seen from below, the surface reflects the dome's two stops along the rippled ray under the
    // open illuminant ([`underside_sky`], [`open_illuminant_g`]), in gamma; a WMO interior pool's
    // under 1, as it is drawn unlit.
    let eye_above = view.world_position.y > world_pos.y;
    let under_e = select(open_illuminant_g(), vec3<f32>(1.0), bi.unlit);
    let sky_g = select(underside_sky(r_rip) * under_e, water_enc(sky), eye_above);
    // **A surface has one reflection, and Fresnel is how big its share of the fragment is**: the
    // tiers below are answers to *what* is reflected, and they share it. The share is
    // [`water_fresnel`] on the filtered normal at the surface's roughness, under the grazing limit
    // `look.composite.x` (1 for none).
    let f_surf = min(water_fresnel(dot(n, to_view), surf.alpha), water_reflect.look.composite.x);

    // ---- the cubemap probe, tier four ---------------------------------------------------------
    //
    // **Directly over the sky and under everything else**, which is its confidence order: it knows
    // the neighbourhood but not this fragment, so anything that knows this fragment outranks it. It
    // answers the one thing none of the other three can — off-screen content at the RIGHT
    // ORIENTATION. The march sees only what is on the frame; the capture sees off the frame but at
    // one plane's orientation; the sky knows nothing. A cube anchored to the water is a picture of
    // what is actually around it, read in the fragment's own reflected direction.
    //
    // The strength lane is zero until every face has been captured, so the frames while the six
    // cameras take their turns show the sky mix rather than a half-built cube.
    //
    // **Considering a probe is free; SAMPLING one is not.** The loop below is weight arithmetic
    // over every slot and costs no memory traffic, while each tap is a colour fetch plus its depth
    // march — five cube reads at the default march depth. That is why the answer to "how many
    // probes blend" is "look at all of them, read the best few": `PROBE_TAPS + 1` are tracked, the
    // extra one never sampled, existing only to supply the weight every tap is measured above.
    // `probe_box_min.x` pins every fragment to one slot, so two renders can be differenced to
    // measure what the blend between those probes actually has to hide. See `probe::probe_force`.
    let forced = i32(water_reflect.probe_box_min.x);
    let taps = select(
        min(i32(water_reflect.probe_box_max.y), PROBE_TAPS_MAX),
        1,
        forced >= 0,
    );
    // **Ranked by weight.** A probe's weight is its availability times the share this fragment's
    // cell gives its spot, and the taps are the heaviest few; the next one is the cutoff each tap's
    // weight is measured above. A probe fading in starts at weight zero, so it cannot push a
    // contributing probe out of the taps, and it takes a tap only at the weight of the probe it
    // replaces, which is where both sit at zero above the cutoff.
    var idx = array<i32, 4>(-1, -1, -1, -1);
    var wt = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var i = 0; i < PROBE_SLOT_MAX; i = i + 1) {
        let sl = water_reflect.probes[i];
        if (sl.at.w <= 0.0 || (forced >= 0 && i != forced)) {
            continue;
        }
        // The probes stand on the map's fixed spots: a fragment reads only its own region's probes,
        // by the share its cell gives each — never the nearest probe of some other water.
        // Availability times share (`box_min.w`, 1 except in the live probe's crossfade).
        let id = sl.bind.x;
        let own = select(0.0, 1.0 - planar.w, abs(id - planar.y) < 0.5)
            + select(0.0, planar.w, abs(id - planar.z) < 0.5);
        let u = sl.at.w * sl.box_min.w * own;
        // Insertion into the short list, heaviest first, shifting the tail down.
        for (var k = 0; k <= taps; k = k + 1) {
            if (u > wt[k]) {
                for (var m = taps; m > k; m = m - 1) {
                    wt[m] = wt[m - 1];
                    idx[m] = idx[m - 1];
                }
                wt[k] = u;
                idx[k] = i;
                break;
            }
        }
    }

    var probe_w = 0.0;
    var probe_rgb = vec3<f32>(0.0);
    let cut = wt[taps];
    var wsum = 0.0;
    var w = array<f32, 3>(0.0, 0.0, 0.0);
    // The taps' cover, summed here so the ranked weights are not held across the cube reads.
    var cover = 0.0;
    for (var k = 0; k < taps; k = k + 1) {
        w[k] = max(wt[k] - cut, 0.0);
        wsum = wsum + w[k];
        cover = cover + wt[k];
    }
    if (wsum > 1e-8) {
        // The cube is read through a FLATTER normal than the water draws with — see
        // `liquid::probe::probe_normal` for the measurement and for why a prefiltered cube is the
        // real answer.
        // **Flat direction, rippled lookup point** — the same correction the march and the mirror
        // got. Perturbing the DIRECTION swings the sample across the cube, and at a grazing angle a
        // slight tilt swings it far: measured at the Stranglethorn camera this tier scored 3.69 on
        // the repeat signature against a waves-off 1.26, the worst of the three. The cube is
        // parallax-corrected against its own capture point, so moving the point the correction is
        // solved for is a real change of viewpoint on a real place, and cannot repeat.
        // About the facet's geometric normal, as the march does: on a sloping stream straight up
        // is off by the slope, and the reflected ray twice that.
        let r = reflect(-to_view, flat_n);
        let probe_pos = world_pos + vec3<f32>(n_slide.x, 0.0, n_slide.z) * water_reflect.march.y;
        // **The tier's strength is how much of this cell's spot shares a cube covers**: the sum of
        // the taps' weights (availability times share), so it does not depend on which tap ranks
        // first. A cell between two spots whose second has no cube (only the spots nearest the
        // player are captured) gives the floor that spot's share, which the classification ramps
        // across the cells between them, so the probe hands its water to the floor over a stretch
        // instead of a step at the cell where the held spot's share ends. A probe serves its whole
        // region, so it does not fade with distance from it. The taps are blended by weight and
        // nothing cleverer: the probe is a soft term under the screen-space march, not an attempt
        // to place reflections exactly.
        for (var k = 0; k < taps; k = k + 1) {
            if (w[k] <= 0.0) {
                continue;
            }
            let sl = water_reflect.probes[idx[k]];
#ifdef WATER_PREVIEW
            probe_rgb = probe_rgb
                + sample_probe_rough(sl, idx[k], probe_pos, r, i32(water_reflect.probe_extra.w), lobe)
                    * (w[k] / wsum);
#else
            probe_rgb = probe_rgb
                + sample_probe(sl, idx[k], probe_pos, r, i32(water_reflect.probe_extra.w))
                    * (w[k] / wsum);
#endif
        }
        probe_w = water_reflect.probe_cfg.x * min(cover, 1.0);
    }

    // `$WOW_PROBE_SHOW` — the cube as the water reads it, black where the tier is not contributing.
    // It exists because a cubemap of the WRONG PLACE is still a plausible reflection: a face
    // captured at the wrong orientation, or a cube that never received its copy, yields water that
    // looks like water and reflects somewhere else. The final image cannot show that; this can.
    if (water_reflect.probe_cfg.w > 0.5 && water_reflect.probe_cfg.w < 1.5) {
        return vec4<f32>(probe_rgb, 1.0);
    }
    // `$WOW_PROBE_CUBE` — the cube unwrapped across the water as a lat-long map: screen X is the
    // bearing, screen Y the elevation, horizon across the middle. No reflection, no ripple, no
    // Fresnel; just what the six cameras captured. See `liquid::probe`'s `probe_cube_show`.
    if (water_reflect.probe_cfg.w > 2.5) {
        // **The latitude sweep is packed into the bottom half of the screen, deliberately.** This
        // debug can only paint WATER fragments, and water occupies roughly the lower half of a
        // frame at any normal camera — so mapping screen Y straight to latitude showed only the
        // cube's lower hemisphere, horizon down to nadir, while the upper half of the image was
        // the real world above the waterline. That is an easy thing to misread as "the cube has
        // trees in it" when the trees being looked at are the actual trees. `fract(y * 2)` sweeps
        // zenith to nadir across whichever half the water is in, so the whole sphere is visible.
        let uv = frag_coord_to_uv(frag_coord);
        // Longitude across the screen, latitude down the LOWER HALF, linearly and with no wrap.
        //
        // The first version used `fract(uv.y * 2)` to pack a full sphere into the half of the frame
        // water occupies, and `fract` wraps: the bottom row returned to the zenith instead of
        // reaching the nadir, so the map carried a seam and every latitude read off it was suspect.
        // Two wrong readings of this image came from that. `uv.y = 0.5` is straight up, `uv.y = 1`
        // is straight down, and the arithmetic between them is a straight line — which is what
        // makes the horizon land exactly halfway down the band and a mis-oriented face obvious.
        let lon = (uv.x * 2.0 - 1.0) * 3.14159265;
        let lat = (1.5 - 2.0 * uv.y) * 3.14159265;
        let dir = vec3<f32>(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
        // One slot's cube — the debug unwraps ONE probe, because a lat-long of "whichever probe is
        // nearest this fragment" would be a mosaic of several and unreadable as a check on any of
        // them. `probe_extra.y` picks which: the captured slot nearest the eye
        // (`benilla_water`'s `probe_debug_slot`).
        let texel = textureSampleLevel(
            probe_tex,
            probe_samp,
            dir,
            i32(water_reflect.probe_extra.y),
            0.0,
        );
        return vec4<f32>(texel.rgb, 1.0);
    }

    // ---- the planar reflection ----------------------------------------------------------------
    //
    // Replaces the sky mix above wherever the mirrored camera actually drew something. Three gates,
    // and each is a case where the image is not this surface's reflection: strength 0 (the pass did
    // not run), a surface further from the mirror plane than the tolerance (one capture is right
    // for one plane, and a world of terraced pools has many), and an eye below the surface (the
    // reflection is on the other side of it).
    //
    // `1 - u` is the mirrored camera's negated right vector undone — see `reflect`'s module doc; the
    // normal's XZ displaces the sample, which is what makes it read as a reflection in water rather
    // than in glass. Alpha is coverage, so where the mirrored view saw nothing the sky mix stands.
    // **Every water body in the frame reflects, not just the one the capture was built for.**
    // `params.w` is no longer a yes/no threshold on the plane error — it is the distance over which
    // the correction below is trusted, and past it the surface fades back to its sky mix instead of
    // switching off. See [`plane_trust`] for why a hard edge was the wrong shape: a pool and the
    // ocean are routinely in one shot, they are ten yards apart, and whichever one lost the coin
    // toss had no reflection at all.
    // ---- screen-space reflection, over the planar capture, over the sky ------------------------
    //
    // The layering is the whole design, and each tier covers the one below's blind spot. SSR is
    // right for any surface orientation because it traces the fragment's real normal, and blind to
    // everything off the screen. The planar capture sees off-screen and behind the camera, and is
    // right for one plane at one orientation. The sky is always available and always plausible.
    //
    // They fail in complementary places, which is why this is a hybrid rather than a compromise:
    // open water is flat (where the plane is exact) and grazing (where the march runs off the
    // frame), while a stream is sloped (where the plane is wrong by twice its tilt) and enclosed by
    // banks a few yards away (where the march has plenty to hit).
    var ssr: SsrHit;
    ssr.rgb = vec3<f32>(0.0);
    ssr.conf = 0.0;
#ifdef WATER_SSR
    if (water_reflect.flags.z > 0.0 && view.world_position.y > world_pos.y) {
        ssr = ssr_trace(
            world_pos + vec3<f32>(n_slide.x, 0.0, n_slide.z) * water_reflect.march.y,
            reflect(-to_view, flat_n),
        );
    }
#endif

    // ---- the tiers' weights ----------------------------------------------------------------------
    //
    // **One reflection, four answers to what is in it**, in confidence order: the sky floor, then
    // the probe, the mirrors and the march, each covering its weight `w` of what the ones beneath
    // it give. The shares are explicit products:
    // `refl_l = ((floor·(1 − w_p) + probe·w_p)·(1 − w_m) + mirror·w_m)·(1 − w_s) + ssr·w_s`, and the
    // floor's share is `(1 − w_p)(1 − w_m)(1 − w_s)`. Where the march found the answer itself,
    // nothing beneath it averages in: two reflections of one surface blended together are a double
    // image, not a better one.
    let w_p = saturate(probe_w);
    // Each mirror serves the water near its own plane: its colour and its weight here — strength
    // times coverage times trust. Where both reach a fragment they blend by weight.
    // Off probe water the mirrors stand down, so the probe beneath them shows through.
    let serve = vec4<f32>(1.0, 1.0, 1.0, smoothstep(0.0, 1.0, planar.x));
#ifdef WATER_PREVIEW
    // `rough`: each mirror read on its chain over the lobe its hit distance lets the screen show.
    let rr_hit = rough_mirror_hit(frag_coord);
    let m1 = planar_read_rough(water_reflect.params, false, world_pos, n_slide, frag_coord, tilt, lobe, rr_hit)
        * serve;
    let m2 = planar_read_rough(water_reflect.mirror2, true, world_pos, n_slide, frag_coord, tilt, lobe, rr_hit)
        * serve;
#else
    let m1 = planar_read(water_reflect.params, false, world_pos, n_slide, frag_coord, tilt) * serve;
    let m2 = planar_read(water_reflect.mirror2, true, world_pos, n_slide, frag_coord, tilt) * serve;
#endif
    // The colour favours the mirror whose plane is nearer this water — both are trusted across
    // their whole tolerance, so without this a body between the two planes took an even mix of a
    // right image and a wrong one. The overall weight below is untouched.
    let k1 = m1.a / (1.0 + abs(world_pos.y - water_reflect.params.x));
    let k2 = m2.a / (1.0 + abs(world_pos.y - water_reflect.mirror2.x));
    let msum = k1 + k2;
    var w_m = 0.0;
    var mirror_g = vec3<f32>(0.0);
    if (msum > 0.0) {
        mirror_g = (m1.rgb * k1 + m2.rgb * k2) / msum;
        // `$WOW_MIRROR_SHOW` — the capture as the water sees it.
        if (water_reflect.params.y > 1.5) {
            return vec4<f32>(mirror_g, 1.0);
        }
        // Either mirror's weight counts; together they are an "either", not a sum past one.
        w_m = saturate(m1.a + m2.a - m1.a * m2.a);
    }

    // …and the march's own weight.
    let w_s = saturate(water_reflect.flags.z * ssr.conf);

    // The sky is the FLOOR, not a layer: the answer for a direction no tier has a better one for,
    // less what the frame shows standing above the water along the reflected ray. Its share of the
    // pixel decides whether that is worth a march; the occluder replaces the shut share of the sky
    // in linear light.
    let w_floor = (1.0 - w_p) * (1.0 - w_m) * (1.0 - w_s);
    var occ = FloorOcc(vec3<f32>(0.0), 1.0, 0.0, 0.0);
    var occ_marched = 0.0;
    var floor_share = w_floor * f_surf;
#ifdef WATER_VIEW
    // The occlusion view shows what the floor gets wherever a march may run, the tiers over it or not.
    if (water_view_id() == WV_FLOOR_OCC) {
        floor_share = 1.0;
    }
#endif
    if (floor_occlusion_wanted(floor_share, world_pos)) {
        occ = floor_occlusion(world_pos, n_slide, flat_n, to_view);
        occ = floor_refog(occ, world_pos, to_view, flat_n);
        occ_marched = 1.0 - occl_plane_weight(world_pos.y);
    }
#ifdef WATER_PREVIEW
    // `glint`: the open reflected sky the glints are seen through.
    let wg_open = wg_glint_open(occ);
#endif
    // The reflected sky's light crosses the fog between the surface and the eye, which the dome as
    // drawn never took: the open sky the floor reflects and the dome the mirror drew are fogged at
    // the surface ([`water_fog`]), as the in-scatter is. What the tiers drew of the world they
    // fogged at its image's distance already ([`mirror_fog`], [`probe_refog`], [`floor_refog`]).
    let fog = water_fog(world_pos, room_fog);
    let sky_seen = fog_diffuse(sky, fog.f, fog.g);
    // The chain in linear light: each tier decoded where it has a share, the floor as the
    // occlusion built it.
    var floor_l = occ.rgb + occ.v * sky_seen;
    if (!eye_above) {
        floor_l = water_dec(sky_g);
    }
    // The grading's map, built while the floor's verdict and the open sky are at hand
    // ([`grade_affine`]): the pivot is the occluded floor where the march reached a verdict, the
    // open sky where it did not. From below the surface, or at strength 0, nothing is graded.
    let graded = eye_above && water_reflect.look.grade.x > 0.0;
    var grade_ab = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (graded) {
        grade_ab = grade_affine(sky_seen, floor_l, occ.known, grade_rho(surf.rho_a, occ.rho_c));
    }
    var refl_l = floor_l;
    if (w_p > 0.0) {
        refl_l = mix(refl_l, water_dec(probe_rgb), w_p);
    }
    if (w_m > 0.0) {
        refl_l = mix(refl_l, mirror_fog(mirror_g, sky, fog), w_m);
    }
    if (w_s > 0.0) {
        refl_l = mix(refl_l, water_dec(ssr.rgb), w_s);
    }
    var refl_g = refl_l;
    if (graded) {
        refl_g = max(grade_ab.w * refl_l + grade_ab.xyz, vec3<f32>(0.0));
    }
    // **The probe lane draws its reflection and nothing else** (`probe_at.w`), ungraded. Not a
    // debug switch: it is what that lane is while the tier is being made correct, because a
    // reflection arriving at a few per cent over a lit green lake hides every error it contains.
    // See `liquid::probe`.
    if (water_reflect.probe_at.w > 0.5) {
        return vec4<f32>(water_enc(refl_l), 1.0);
    }

    // The glitter path's lights: the same Blinn highlight the reference computes, evaluated against
    // the rippled normal per fragment in the composite below. The fine ripple layer above breaks it
    // into pieces; the exponent and the weight decide how hard and how bright those pieces are —
    // which is to say, how ROUGH the water is. See [`SPEC_POWER`].
    //
    // **Against the VISIBLE sun, not the lighting one.** The engine carries two (see
    // `lighting::daynight`): `light_sun` is pinned at compass 225 deg and never sets — it exists so
    // MCSH shadows can be baked once — while the disc in the sky rides `celestial_sun_direction`
    // at compass 45 deg and genuinely rises and sets. They measure 25-32 deg apart, so a highlight
    // aimed by `light_sun` lands in a different part of the water than the sun it is supposed to be
    // a reflection of, and it lands there at midnight too. `water_reflect.sun` is the one you can
    // point at.
    // Guarded: the lanes are zero for one frame at startup, before the sun pass has run, and
    // `normalize` of a zero vector is a NaN that would survive the multiply by a zero reach.
    let sun_len = length(water_reflect.sun.xyz);
    let to_light = select(vec3<f32>(0.0, 1.0, 0.0), water_reflect.sun.xyz / sun_len, sun_len > 1e-4);
    // ...and only as far as the sun actually reaches: below the horizon, behind a cloud, or behind
    // the ridge you are standing under, there is no glitter to have. The scalar is slewed on the
    // CPU, so this fades over about a second rather than switching.
    let sun_reach = water_reflect.sun.w;
    // Widened, at the same energy, by the roughness the surface cannot resolve ([`glint_lobe_p1`]).
    let lost2 = surface_lost2(surf);

    // ...and the moon's, which is the same computation against the other body in the sky. Water
    // does not care which one is up; it was only ever the sun here because the sun was the only
    // direction this shader had been handed.
    let moon_len = length(water_reflect.moon.xyz);
    let to_moon = select(
        vec3<f32>(0.0, 1.0, 0.0),
        water_reflect.moon.xyz / moon_len,
        moon_len > 1e-4,
    );

    // The white line where the water meets the land, and the bubbles behind it.
    //
    // `to_shore` is the distance to the waterline in yards, traced on the CPU from where the water
    // surface crosses the ground and carried per vertex (`benilla_world::liquid::surface`). The
    // cells it runs through are subdivided so that distance is described finely enough to draw a
    // band under a yard wide; at the bare 4.17 yd lattice the band appeared only where a corner
    // happened to fall near the water's edge, which is what made it look like teeth.
    // **The distance, measured here rather than interpolated.** `shore` is the same quantity carried
    // per vertex, and it is what the band used to be drawn from — but distance to a curve creases
    // along the curve, and a linear interpolation across that crease cuts its corner: the
    // reconstructed band lands off by up to half a vertex spacing, so at half a yard between
    // vertices and a fifth of a yard of band it visibly jogged, thinned and doubled as the camera
    // moved. The offset has no crease, so it interpolates honestly and the length is exact.
    //
    // The interpolated scalar still wins beyond the near field, for two reasons: past a couple of
    // yards nothing is drawn either way, and an offset is only meaningful where a nearest point
    // exists — on the ridge halfway between two facing banks the nearest point flips sides, and
    // interpolating across that flip would collapse the offset to nothing and draw a foam line down
    // the middle of a channel. Inside the near field the two agree to within the mesh's own error;
    // where they do not, the scalar is the conservative answer and is taken.
    let exact = length(shore_offset);
    var to_shore = shore;
    if (shore < SHORE_EXACT_REACH && abs(exact - shore) < SHORE_EXACT_TRUST) {
        to_shore = exact;
    }

    // The derivatives the band needs, taken HERE, where the flow is still uniform. Everything below
    // sits behind [`FOAM_REACH`], and past that gate neither `fwidth` nor an implicit-LOD
    // `textureSample` is defined — the same rule the wake field's unconditional tap is written
    // around, answered the other way: hoist the derivative rather than the sample.
    //
    // Anti-aliasing against how much distance one pixel covers, bounded at both ends: below so a
    // near-view edge stays an edge, above because past a fraction of the band's own width this
    // stops being anti-aliasing and becomes a smear.
    let fw_shore = fwidth(to_shore);
    let aa = clamp(fw_shore, 0.015, 0.20);
    // World yards per pixel, for the taps inside the gate.
    let d_xz_dx = dpdx(xz);
    let d_xz_dy = dpdy(xz);

    // The edge's coverage over the bed (see below, where the composite takes it).
    var soft = 1.0;
    if (thickness >= 0.0) {
        soft = saturate(thickness / SOFT_EDGE_YARDS);
    }
#ifdef WATER_PREVIEW
    // `shore`: the wet line, the swash, contacts and whitecaps ([`shore_contact`]), seen from above.
    let sc_on = pv_on(PV_SHORE) && eye_above;
    var sc = sc_neutral(soft);
    if (sc_on) {
        sc = shore_contact(
            ScIn(world_pos, frag_coord, to_shore, shore, shore_offset, flow, aa, fw_shore,
                d_xz_dx, d_xz_dy),
            sc_geometry_release(sc_held),
        );
    }
#else
    let sc_on = false;
#endif
    var line = 0.0;
    var bubbles = 0.0;
    if (to_shore < FOAM_REACH && !sc_on) {
        // 1. The line. Its outer edge wavers only slightly, so it reads as an even edge rather than
        //    as something ragged; the raggedness belongs to the bubbles. No time in this term —
        //    foam gathers where the shore's shape makes it gather, and an edge that crawls reads as
        //    a bug.
        let warp = textureSampleGrad(
            ripples,
            ripples_samp,
            xz * 0.05,
            d_xz_dx * 0.05,
            d_xz_dy * 0.05,
        ).b * 2.0 - 1.0;
        // The swell: the band's outer edge advances up the shore and slides back down, out of step
        // from bay to bay (see [`SHORE_RUNUP_YARDS`]). The phase comes from a very coarse read of
        // the same map, so it drifts along a coastline instead of switching at some boundary.
        let swell_phase = textureSampleGrad(
            ripples,
            ripples_samp,
            xz / SHORE_SWELL_SPREAD,
            d_xz_dx / SHORE_SWELL_SPREAD,
            d_xz_dy / SHORE_SWELL_SPREAD,
        ).b * TAU;
        let swell = sin(TAU * water_reflect.cycle.y + swell_phase);
        let edge = (FOAM_LINE_YARDS + SHORE_RUNUP_YARDS * swell) * (1.0 + FOAM_EDGE_WARP * warp);
        line = 1.0 - smoothstep(edge - aa, edge + aa, to_shore);

        // 2. Behind it, bubbles: the same white, but broken into flecks that thin out with distance
        //    until there is only water. `out` runs 0 at the line to 1 where they stop, and it is
        //    the THRESHOLD the noise is cut at — so the coverage falls away on its own, and there
        //    is no second hard edge anywhere out in the water to alias against.
        let out = saturate((to_shore - edge) / FOAM_BUBBLE_YARDS);
        // Fine on purpose: the band is barely a yard, so the flecks have to be a good deal smaller
        // than that or only one spans it and the break-up never reads as bubbles at all — and
        // non-repeating, which at that size the map is not on its own (see [`foam_noise`]).
        let raw = foam_noise(xz, d_xz_dx, d_xz_dy);
        // Spread before cutting. The map's height channel is a sum of Perlin octaves normalised on
        // its single largest texel, so its values crowd hard around 0.5 and only the extremes ever
        // approach 0 or 1 (`benilla_world::liquid::ripple`). Cut at a threshold sweeping the whole
        // 0..1 it behaves as a step — everything passes, then nothing — which collapsed the bubbles
        // into a narrow ring and made the line's outer edge read as hard. Widened about its middle,
        // the field spans the range the threshold actually travels.
        let bub = saturate((raw.v - 0.5) * 3.5 + 0.5);
        // The cut is softened by the noise's own pixel footprint, so at distance the flecks resolve
        // to their average instead of sparkling. The footprint is carried out of `foam_noise` on
        // the same fetches that gave the value — `fwidth` is not available on this side of the gate
        // — and the `* 3.5` is the spread above, which multiplies the derivative with the value.
        let bw = max(0.07, raw.fw * 3.5 * 1.5);
        // The taper must reach exactly zero where the band ends, not merely dim. `out` saturates at
        // 1 out in open water, and a threshold sitting at 1 still admits the top of the spread
        // field — which put flecks across the whole surface the first time this was tried.
        bubbles = smoothstep(out - bw, out + bw, bub) * (1.0 - out);
    }

    // ...and the wake's own aeration alongside them. `max`, not a sum: these are three ways for the
    // same surface to be white, and adding them blows out where a swimmer crosses a waterline —
    // which is precisely where a player spends their time in the water.
#ifdef WATER_PREVIEW
    let foam = max(
        max(max(line * FOAM_LINE_LEVEL, bubbles * FOAM_BUBBLE_LEVEL), sc.foam),
        wake_foam * WAKE_FOAM_LEVEL,
    );
#else
    let foam = max(
        max(line * FOAM_LINE_LEVEL, bubbles * FOAM_BUBBLE_LEVEL),
        wake_foam * WAKE_FOAM_LEVEL,
    );
#endif

    // The BODY thins to nothing where something passes through the surface: a sheet of water
    // meeting a rock along a hard line is the giveaway that it is a sheet, while real water fades
    // over the last hand's breadth because by then there is barely any water left to be opaque
    // with. Needs to know what is behind the surface: [`scene_behind`]'s column.
    //
    // **The foam is taken after the fade, not before it.** Foam sits ON the surface — it is spray,
    // not water column — so softening it against the very intersection it gathers at erases the
    // shoreline exactly where the line belongs, which is what the first attempt did.
#ifdef WATER_VIEW
    // The debug views paint one quantity over the water, opaque and unfogged; the composite's own
    // views change its layer below instead.
    let view_id = water_view_id();
    if (view_id != 0u && view_id != WV_FLAT && view_id < WV_COMPOSITE_IDENTITY) {
        var v = vec3<f32>(0.0);
        if (view_id == WV_MIRROR1) {
            v = m1.rgb;
        } else if (view_id == WV_MIRROR2) {
            v = m2.rgb;
        } else if (view_id == WV_PROBE) {
            v = probe_rgb;
        } else if (view_id == WV_TIERS) {
            // Each tier's share of `refl` as the chain above mixed them: the probe over the sky,
            // the mirrors over both, the march over all. Sky grey, probe green, the first mirror
            // blue, the second cyan, the march red.
            let s_ssr = saturate(water_reflect.flags.z * ssr.conf);
            let s_mirrors = saturate(m1.a + m2.a - m1.a * m2.a) * (1.0 - s_ssr);
            let s_probe = saturate(probe_w) * (1.0 - s_mirrors - s_ssr);
            let s_sky = 1.0 - s_probe - s_mirrors - s_ssr;
            let s_m1 = select(0.0, k1 / msum, msum > 0.0) * s_mirrors;
            v = vec3<f32>(0.5) * s_sky
                + vec3<f32>(0.0, 1.0, 0.0) * s_probe
                + vec3<f32>(0.0, 0.0, 1.0) * s_m1
                + vec3<f32>(0.0, 1.0, 1.0) * (s_mirrors - s_m1)
                + vec3<f32>(1.0, 0.0, 0.0) * s_ssr;
        } else if (view_id == WV_FLOW) {
            // The current at 2 yd/s full scale: 0.5 + flow/4 in rg, the speed in b.
            v = vec3<f32>(vec2<f32>(0.5) + 0.25 * flow, saturate(length(flow) * 0.5));
        } else if (view_id == WV_FOAM) {
            // Each foam's coverage as mixed above: the shore's line and bubbles in r, the wake's
            // in b; g is the contact foam's, which this look does not draw.
            v = vec3<f32>(
                max(line * FOAM_LINE_LEVEL, bubbles * FOAM_BUBBLE_LEVEL),
                0.0,
                wake_foam * WAKE_FOAM_LEVEL,
            );
        } else if (view_id == WV_THICKNESS) {
            // What is behind the pixel, yards / 8: r the view-Z gap, g the vertical column, b the
            // path through the water; 0 where it is not known.
            let column = max(vec3<f32>(sb.gap_z, sb.vertical, sb.path), vec3<f32>(0.0)) / 8.0;
            v = select(vec3<f32>(0.0), column, sb.known);
        } else if (view_id == WV_FRESNEL) {
            v = vec3<f32>(f_surf);
        } else if (view_id == WV_ROUGHNESS || view_id == WV_ROUGHNESS_ANISO) {
            // α between the calm surface (0) and the fully unresolved one (1).
            let a0 = water_reflect.look.surf.x;
            let k2 = 2.0 * surf.k * surf.k;
            let a_full = sqrt(a0 * a0 + k2 * OCTAVE_W2_SUM * water_reflect.look.ripple_mean.z);
            let g = saturate((surf.alpha - a0) / max(a_full - a0, 1e-6));
            if (view_id == WV_ROUGHNESS) {
                // Grey, red where α has reached the unresolved surface's.
                v = select(vec3<f32>(g), vec3<f32>(1.0, 0.0, 0.0), surf.rho_a >= 0.999);
            } else {
                // r (α_major / α_minor − 1)·4 from the tensor's eigenvalues, g the normalised α.
                let mid = 0.5 * (surf.a2.x + surf.a2.y);
                let half_gap = length(vec2<f32>(0.5 * (surf.a2.x - surf.a2.y), surf.a2.z));
                let ratio = sqrt((mid + half_gap) / max(mid - half_gap, 1e-12));
                v = vec3<f32>(saturate((ratio - 1.0) * 4.0), g, 0.0);
            }
        } else if (view_id == WV_SLOPE_FULL) {
            v = vec3<f32>(vec2<f32>(0.5) + 4.0 * n.xz, 0.0);
        } else if (view_id == WV_SLOPE_LO) {
            v = vec3<f32>(vec2<f32>(0.5) + 4.0 * surf.n_lo.xz, 0.0);
        } else if (view_id == WV_SKY_FLOOR) {
            v = water_enc(sky);
        } else if (view_id == WV_SIGMA_SHAPE) {
            // The fragment's kind's absorption per channel over its largest channel.
            let sigma = view_sigma_abs();
            v = sigma / max(max(sigma.r, max(sigma.g, sigma.b)), 1e-6);
        } else if (view_id == WV_FLOOR_OCC) {
            // r the open sky v, g rho_c, b the per-pixel march's share (1 off the planes, 0 on
            // them, between in the crossfade band); half bright where the verdict is unknown.
            v = vec3<f32>(occ.v, occ.rho_c, occ_marched) * (0.5 + 0.5 * occ.known);
        }
        return vec4<f32>(v, 1.0);
    }
#endif
    // ---- the composite, in linear light ---------------------------------------------------------
    //
    // The water as one premultiplied layer: one reflectance for every tier, the
    // tiers' linear chain graded above, the glints in linear, the body's in-scatter and the bed's
    // transmittance, the edge's coverage, the foam over it, each term fogged once: the reflection
    // as its tier fogged it, the glints and foam at the surface's eye Z, the bed as the frame did.
    // The glints in linear light at the `composite.y` weight, the gamma glint weight 0.22 raised to
    // the gamma, so the peak over black is the gamma weight's; the moon's weight is raised likewise.
    // A highlight adds light, so the fog takes its share away rather than mixing toward a colour.
    let gw = water_reflect.look.composite.y;
#ifdef WATER_PREVIEW
    // `glint`: the sun's and the moon's GGX lobes on the surface's roughness ([`wg_glints`]).
    var spec_glint: vec3<f32>;
    if (pv_on(PV_GLINT)) {
        spec_glint = wg_glints(n, to_view, to_light, to_moon, surf, wg_open, wg_lit_sky(eye_above, room_fog));
    } else {
        spec_glint = water_dec(wow_light.light_spec.rgb)
            * glint_lobe_p1(n, to_light, to_view, SPEC_POWER, lost2)
            * (gw * sun_reach)
            + water_dec(MOON_SPEC_COLOR)
                * glint_lobe_p1(n, to_moon, to_view, MOON_SPEC_POWER, lost2)
                * (gw * pow(MOON_SPEC_WEIGHT, WATER_GAMMA) * water_reflect.moon.w);
    }
    let spec_l = spec_glint * fog.f;
#else
    let spec_l = (water_dec(wow_light.light_spec.rgb)
        * glint_lobe_p1(n, to_light, to_view, SPEC_POWER, lost2)
        * (gw * sun_reach)
        + water_dec(MOON_SPEC_COLOR)
            * glint_lobe_p1(n, to_moon, to_view, MOON_SPEC_POWER, lost2)
            * (gw * pow(MOON_SPEC_WEIGHT, WATER_GAMMA) * water_reflect.moon.w))
        * fog.f;
#endif
#ifdef WATER_VIEW
    // The body's own views, opaque and unfogged: what passes from the bed, what the body scatters,
    // and the albedo the ramp reads at this pixel; then the surface's fog.
    if (view_id == WV_TRANSMITTANCE) {
        return vec4<f32>(water_enc(optics.t_bed), 1.0);
    } else if (view_id == WV_IN_SCATTER) {
        return vec4<f32>(water_enc(body_light.in_scatter), 1.0);
    } else if (view_id == WV_ALBEDO) {
        return vec4<f32>(optics.a_g, 1.0);
    } else if (view_id == WV_GRADE) {
        let keep = select(1.0, grade_keep(grade_rho(surf.rho_a, occ.rho_c)), eye_above);
        return vec4<f32>(surf.rho_a, occ.rho_c, keep, 1.0);
    } else if (view_id == WV_REFL_UNGRADED) {
        return vec4<f32>(water_enc(refl_l), 1.0);
    } else if (view_id == WV_REFL_GRADED) {
        return vec4<f32>(water_enc(refl_g), 1.0);
    } else if (view_id == WV_REFRACTED_BED) {
        let bed_r = water_bed(water_snapshot_px(frag_coord), optics.bed_uv_r, optics.accept);
        return vec4<f32>(water_enc(bed_r), 1.0);
    } else if (view_id == WV_REFRACT_REJECT) {
        let off = length((optics.bed_uv_r - frag_coord_to_uv(frag_coord)) * view.viewport.zw);
        let leak = view_refract_leak(world_pos, optics.bed_uv_r, optics.accept);
        return vec4<f32>(optics.accept, off / REFRACT_MAX_PX, leak, 1.0);
    } else if (view_id == WV_MCSH) {
        return vec4<f32>(vec3<f32>(lighting_sun_mcsh(world_pos, optics.z)), 1.0);
    } else if (view_id == WV_SKY_VIS) {
        return vec4<f32>(vec3<f32>(vis.g), 1.0);
    } else if (view_id == WV_SUN_VIS) {
        return vec4<f32>(vec3<f32>(vis.b), 1.0);
    } else if (view_id == WV_FOAM_LIGHT) {
        let left = frag_coord.x < view.viewport.x + 0.5 * view.viewport.z;
        let e = select(body_light.irradiance, body_light.foam_irradiance, left);
        return vec4<f32>(water_enc(e), 1.0);
    } else if (view_id == WV_FOG) {
        return vec4<f32>(fog.f, fog.g.g, fog.g.b, 1.0);
    }
#endif
#ifdef WATER_REFRACT
    let body = body_release(held);
#else
    let body = BodyHeld(vec3<f32>(0.0), optics.t_bed, optics.w_in);
#endif
    // The body scatters `w_in` of its light back; `t_bed` of the bed passes.
    let light = light_release(light_held);
    // From below the surface the floor is the dome's stops ([`underside_sky`]), which no camera drew
    // and so none fogged: the fog between the surface and the eye takes the reflection here.
    var refl_out = refl_g;
    if (!eye_above) {
        refl_out = fog_diffuse(refl_g, fog.f, fog.g);
    }
    var layer = layer_surface(
        f_surf,
        refl_out,
        spec_l,
        fog_diffuse(light.in_scatter, fog.f, fog.g),
        vec3<f32>(body.w_in),
        body.t_bed,
    );
#ifdef WATER_PREVIEW
    // `shore`: the edge's coverage with the swash's draw-back, then the wet bed, film and rim.
    layer = layer_cover(layer, sc.cover);
    layer = sc_layer(layer, sc, f_surf, refl_out, spec_l);
#else
    layer = layer_cover(layer, soft);
#endif
    // Foam is coverage: spray over the surface, hiding what is under it.
    // Lit by the body's light in the terrain light's colour (warm at dusk, blue under the moon),
    // and fogged as a surface is.
    if (foam > 0.0) {
#ifdef WATER_PREVIEW
        let foam_l = fog_diffuse(sc.foam_albedo * light.foam, fog.f, fog.g);
#else
        let foam_l = fog_diffuse(water_dec(FOAM_COLOR) * light.foam, fog.f, fog.g);
#endif
        layer = layer_over(foam_l, foam, layer);
    }
#ifdef WATER_REFRACT
    let snap_px = water_snapshot_px(frag_coord);
    var bed_out = body.bed;
#else
    var bed_out = water_dec(snap_px);
#endif
#ifdef WATER_VIEW
    if (view_id == WV_COMPOSITE_IDENTITY) {
        layer = WaterLayer(vec3<f32>(0.0), vec3<f32>(1.0));
        bed_out = water_dec(snap_px);
    } else if (view_id == WV_DISCARD) {
        return vec4<f32>(0.0);
    }
#endif
    return water_output(layer, frag_coord, snap_px, bed_out, 1.0);
}
#endif // WATER_STYLISED

// The vertex input — bevy 0.18's `forward_io::Vertex` fields at bevy's own shader locations, plus
// the shore offset at 10 under `LIQUID_SHORE_OFFSET` (`LiquidExt::specialize` sets the def and
// appends the attribute to the buffer layout when the mesh carries it). Declared here rather than
// imported for the one reason the model shader declares its own: a struct you did not write cannot
// gain a field.
struct LiquidVertex {
    @builtin(instance_index) instance_index: u32,
#ifdef VERTEX_POSITIONS
    @location(0) position: vec3<f32>,
#endif
#ifdef VERTEX_NORMALS
    @location(1) normal: vec3<f32>,
#endif
#ifdef VERTEX_UVS_A
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_UVS_B
    @location(3) uv_b: vec2<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
#endif
#ifdef LIQUID_SHORE_OFFSET
    // Vertex → nearest point on the waterline, in mesh-local XZ yards.
    @location(10) shore_offset: vec2<f32>,
#endif
#ifdef LIQUID_SURFACE_NORMAL
    @location(11) surface_normal: vec3<f32>,
#endif
#ifdef LIQUID_PLANAR
    @location(12) planar: vec4<f32>,
#endif
#ifdef LIQUID_FLOW
    // The current, mesh-local XZ yards a second.
    @location(13) flow: vec2<f32>,
#endif
}

@vertex
fn vertex(in: LiquidVertex) -> LiquidVsOut {
    var out: LiquidVsOut;
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    out.world_position =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.position, 1.0));
    out.clip_position = position_world_to_clip(out.world_position.xyz);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index);
#ifdef LIQUID_SURFACE_NORMAL
    out.surface_normal =
        mesh_functions::mesh_normal_local_to_world(in.surface_normal, in.instance_index);
#else
    out.surface_normal = vec3<f32>(0.0);
#endif
#ifdef LIQUID_PLANAR
    out.planar = in.planar;
#else
    out.planar = vec4<f32>(1.0, 65535.0, 65535.0, 0.0);
#endif
    out.uv = in.uv;
#ifdef VERTEX_COLORS
    out.vcolor = in.color;
#else
    out.vcolor = vec4<f32>(1.0);
#endif
    // Depth coordinate V (0..1) in UV1.x: the swatch row on ADT water, the alpha ramp on WMO.
    out.depth = in.uv_b.x;
    // UV1.y is the distance to the waterline in yards, for the far field and for the shore foam on
    // a mesh without the offset attribute.
    out.shore = in.uv_b.y;
#ifdef LIQUID_SHORE_OFFSET
    // A direction: the placement's rotation and scale, not its translation.
    let off_local = vec3<f32>(in.shore_offset.x, 0.0, in.shore_offset.y);
    let off_world = mat3x3<f32>(
        world_from_local[0].xyz,
        world_from_local[1].xyz,
        world_from_local[2].xyz,
    ) * off_local;
    out.shore_offset = off_world.xz;
#else
    out.shore_offset = vec2<f32>(in.uv_b.y, 0.0);
#endif
#ifdef LIQUID_FLOW
    // The current, a direction like the offset: the placement's rotation and scale only.
    out.flow = (mat3x3<f32>(
        world_from_local[0].xyz,
        world_from_local[1].xyz,
        world_from_local[2].xyz,
    ) * vec3<f32>(in.flow.x, 0.0, in.flow.y)).xz;
#else
    out.flow = vec2<f32>(0.0);
#endif
    out.secondary_vtx = sun_sheen(out.world_normal, out.world_position.xyz);
    // ADT surfaces carry no `MeshTag`, so they take the scene fog.
    out.room_fog = mesh_functions::get_tag(in.instance_index) & 0x40000000u;
    return out;
}

// ── The ADT depth swatch ─────────────────────────────────────────────────────────────────────
//
// `0x68a830` fills an 8×64 texture, each row the same across its 8 columns, with an exact
// byte-space integer accumulator, `row(i) = c0 + floor(i * (c1 - c0) / 64)` for i = 0..63, so
// row 63 stops short of the deep endpoint. On the ocean (selector 0) the last row's HSV value is
// scaled by 0.9 (`0x68aa13`, `[0x8102ec]`), which `floor(0.9 * byte)` reproduces within 1/255,
// and its alpha is forced to 255 (`0x7bbec0`/`0x7bbec8`). Sampling is LINEAR, no mip, clamped
// (flags `0x201`), so V maps to texel `V*64 - 0.5` and the ocean darkening ramps over the last
// 1/64 of V. The WMO arms use the 256-entry ramp `0xca7f10` instead, which a plain lerp
// reproduces.
fn swatch_row(shallow: vec4<f32>, deep: vec4<f32>, i: f32, ocean: bool) -> vec4<f32> {
    // RGB endpoints are bytes already (`0x68a8fb`/`0x68a902`), so they round back exactly; alpha
    // endpoints are `LightParams` floats the reference quantizes with `floor(v*255)`.
    let c0 = vec4<f32>(round(shallow.rgb * 255.0), floor(shallow.w * 255.0));
    let c1 = vec4<f32>(round(deep.rgb * 255.0), floor(deep.w * 255.0));
    let row = c0 + floor(i * (c1 - c0) / 64.0);
    if ocean && i >= 63.0 {
        return vec4<f32>(floor(row.rgb * 0.9), 255.0) / 255.0;
    }
    return row / 255.0;
}

/// The swatch sampled at depth coord `v`, LINEAR across the two rows it falls between.
fn swatch_at(shallow: vec4<f32>, deep: vec4<f32>, v: f32, ocean: bool) -> vec4<f32> {
    let t = clamp(v * 64.0 - 0.5, 0.0, 63.0);
    let i0 = floor(t);
    return mix(
        swatch_row(shallow, deep, i0, ocean),
        swatch_row(shallow, deep, min(i0 + 1.0, 63.0), ocean),
        t - i0,
    );
}

@fragment
fn fragment(in: LiquidVsOut) -> @location(0) vec4<f32> {
    // The far-clip wall, as terrain and models: discard beyond `fog_params.w` (0 disables it).
    if (wow_light.fog_params.w > 0.0) {
        let clip_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        if (clip_z > wow_light.fog_params.w) {
            discard;
        }
    }

    // The animated frame; `view.mip_bias` is the render-scale LOD compensation, 0 at native.
    let detail = textureSampleBias(
        frames,
        frames_samp,
        apply_scroll(in.uv),
        frame_layer(),
        view.mip_bias,
    );

    // Magma/slime: the sheet is the opaque body, unmodulated (the ADT vertex has no colour, the WMO
    // one is `0xffffffff`) and unlit (lighting off on both paths), but fogged.
    if (w.kind.x > 0.5) {
        // The stylised lane grades magma into crust and seam ([`stylised_lava`]); slime keeps the
        // sheet, having no heat to render, and the faithful lane keeps it for both. Fog applies to
        // all four cases — see the module note on why skipping it was wrong.
        //
        // **`select`, not an `if`, and that is a correctness fix rather than a style choice.**
        // `detail` above is a `textureSampleBias`, which needs implicit derivatives, and wrapping
        // the code after it in a branch was enough to cost it a mip level: the FAITHFUL lane's lava
        // came back duller and greyer (−76 red, +9 green, +13 blue over the magma silhouette — the
        // signature of a blurrier mip, not of a grade), on a path where none of this code runs.
        // Caught by capturing `WOW_WATER_STYLE=0` before and after and diffing; it would have been
        // invisible in any stylised shot. A `select` has no control flow for the sample to be
        // reachable-from differently, so the derivatives stay put.
        //
        // The cost of evaluating the grade on surfaces that discard it is two `textureSampleLevel`
        // taps on magma and slime fragments only, which is a rounding error next to being wrong.
        let graded = stylised_lava(detail.rgb, in.world_position.xz, anim_time());
        let full = select(detail.rgb, graded, w.path.y > 0.5 && w.path.z > 0.5);
        return vec4<f32>(apply_fog(full, in.world_position.xyz, in.room_fog), 1.0);
    }

    // V, from the authored depth byte CPU-side: clamp(byte/42) on river/lake (LUT `0xc81768`,
    // `0x68d790`, saturating near 5 yd), clamp(byte/255) on ocean (LUT `0xc7fcd8`,
    // `0x68d690`), both built in `0x68c4c0`. One V indexes colour and alpha alike.
    let depth = clamp(in.depth, 0.0, 1.0);
    var shallow = wow_light.water_river[0];
    var deep = wow_light.water_river[1];
    if (w.kind.y > 0.5) {
        shallow = wow_light.water_ocean[0];
        deep = wow_light.water_ocean[1];
    }

    // ---- The stylised look ------------------------------------------------------------------
    //
    // It replaces all three water arms at once, and it runs HERE — after the world's own swatch and
    // depth are resolved, before any of the reference's three combines — because those two are the
    // only world inputs it takes.
    //
    // Each arm keeps its own colour source, so what changes is the treatment and never the
    // palette: the ADT ramp lerps the zone swatch by depth; a WMO exterior canal takes the deep
    // river band flat, having no bathymetry to lerp over; and a WMO interior pool keeps its
    // authored `MOMT.diffColor` body, unlit, because that arm has no normal to light with and its
    // rooms are not lit by the scene's sun. The other two are lit by the sky and the lighting sun
    // under the terrain's shadow ([`body_irradiance`]), at the terrain's open-sky luminance.
#ifdef WATER_STYLISED
    if (w.path.y > 0.5) {
        var body_shallow = shallow;
        var body_deep = deep;
        if (w.path.x > 1.5) {
            body_shallow = vec4<f32>(in.vcolor.rgb, shallow.w);
            body_deep = vec4<f32>(in.vcolor.rgb, deep.w);
        } else if (w.path.x > 0.5) {
            body_shallow = deep;
        }
        // What stands behind this water pixel, from the complete linear depth (`liquid::hiz`): the
        // static world, entity-path models and units included.
        let sb = scene_behind(in.world_position.xyz, in.clip_position.xy);
        // The body's stops lie on the zone's own line from swatch row 0 toward row 63 (without the
        // ocean's ×0.9 tail): the deep stop at the deep ceiling, the mid stop at half of it, so the
        // three bands keep equal spans. A WMO exterior's pair is its deep river row twice, an
        // interior's its pool colour twice.
        let ocean = w.kind.y > 0.5;
        let r0 = swatch_row(body_shallow, body_deep, 0.0, ocean).rgb;
        let r63 = swatch_row(body_shallow, body_deep, 63.0, false).rgb;
        let dc = water_reflect.look.body.spare.y;
        let sig = select(water_reflect.look.body.sig_river, water_reflect.look.body.sig_ocean, ocean);
        let unlit = w.path.x > 1.5;
        let bi = BodyIn(
            r0,
            mix(r0, r63, 0.5 * dc),
            mix(r0, r63, dc),
            sig.xyz,
            sig.w,
            select(water_reflect.look.body.look.w * w.anim.x, 0.0, unlit),
            depth * select(REF_RIVER_V_YD, REF_OCEAN_V_YD, ocean),
            unlit,
            w.path.x < 0.5,
        );
        return stylised_water(
            in.world_position.xyz,
            in.clip_position.xy,
            sb,
            in.shore,
            in.shore_offset,
            in.surface_normal,
            in.planar,
            in.flow,
            bi,
            in.room_fog,
        );
    }
#endif

    // ---- The WMO water arms: opacity is the per-vertex authored byte through the zone's linear
    // alpha ramp, carried in `in.depth`.
    let vtx_alpha = mix(shallow.w, deep.w, depth);
    if (w.path.x > 1.5) {
        // ---- WMO interior (`0x6b6420`): fixed-function whatever the CVars (`[0xc9607c]` unread),
        // lighting off (`0x0e = 0`), fog on (`0x0f = 1`), combine preset `(0x1f, 3)`:
        //     rgb = clamp(Cf + Ct)      alpha = clamp(Af + At)
        // Cf is the pool's raw `MOMT.diffColor`; the vertex has no normal, so no sun and no sheen.
        // Preset 3 is `GL_ADD` through `GL_COMBINE` on both channels (`COMBINE_ALPHA` at
        // `0x85c2fc`), so the ripple adds to the pool's opacity rather than multiplying it.
        let body = clamp(in.vcolor.rgb + detail.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(
            apply_fog(body, in.world_position.xyz, in.room_fog),
            clamp(vtx_alpha + detail.a, 0.0, 1.0),
        );
    }
    if (w.path.x > 0.5) {
        // ---- WMO exterior (`0x6b6630`): `Shaders\Pixel\MapObjExtWater0.bls` (bound `0x6b6654`),
        // the shader leg of the `[0xc9607c]` gate:
        //     rgb = primary.rgb + detail.rgb + secondary·detail.a      alpha = primary.a
        // No `+0.25`, which is the ADT program's own. Lighting is on (`0x0e` never set), so
        // primary = band · clamp(ambient + diffuse·max(N·L, 0)), the band one flat colour for
        // every nibble: the deep river row, `LightIntBand` sub-17 (`water_river[1]`), an immediate
        // at `0x6b66be`. A DayNight slot is not a sub: `0x6d64d0` moves sub-8 out to `+0x4c`, so
        // the kernel's slot 16 is sub-17.
        let n_ext = normalize(in.world_normal);
        let to_light_ext = -normalize(wow_light.light_sun.xyz);
        let primary_ext = clamp(
            wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb
                * max(dot(n_ext, to_light_ext), 0.0),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        ) * deep.rgb;
        let rgb_ext = primary_ext + detail.rgb + in.secondary_vtx * detail.a;
        // Alpha is `fragment.color.primary` alone: the bound program bypasses the texture
        // environment, so the interior arm's `+ At` does not apply.
        return vec4<f32>(apply_fog(rgb_ext, in.world_position.xyz, in.room_fog), vtx_alpha);
    }

    // The ADT arm. `primary`: the lit white vertex.
    let n = normalize(in.world_normal);
    let to_light = -normalize(wow_light.light_sun.xyz);
    let ndotl = max(dot(n, to_light), 0.0);
    let primary = clamp(
        wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb * ndotl,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

    let secondary = in.secondary_vtx;

    // colorTex: the stage-0 depth swatch.
    let swatch = swatch_at(shallow, deep, depth, w.kind.y > 0.5);

    // The `ocean0_s.bls` combine.
    var rgb = primary * swatch.rgb + detail.rgb + (secondary + vec3<f32>(0.25)) * detail.a;

    // `colorTex.a`, over the same V as the colour: deeper water is more opaque.
    let alpha = swatch.w;

    rgb = apply_fog(rgb, in.world_position.xyz, in.room_fog);

    // Raw gamma out; alpha blends in gamma space like the reference's bytes.
    return vec4<f32>(rgb, alpha);
}
