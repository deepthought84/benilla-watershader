// FFXGlow, the reference's full-screen glow (`FFXGlow.bls`, `FFXGauss4.bls`):
//   blur = Gauss4(Gauss4(Box4(scene → ¼), horizontal), vertical)
//   out  = lerp(screen, blur, z) + w · blur²
// `w` is the zone's LightParams glow weight, `z` the drunk or underwater haze mix. Tap offsets
// count source texels from texel-centre UVs, with no half-texel shift. The math runs on gamma
// bytes, and every scene read clamps to 1.0 because the float target does not saturate.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var in_tex: texture_2d<f32>;
@group(0) @binding(1) var in_samp: sampler;
// Combine-pass bindings, unbound in the downsample and blur passes.
@group(0) @binding(2) var blur_tex: texture_2d<f32>;
struct FfxCombine {
    // x = zone glow weight `w` (both combines), y = FFXDeath gate (1 while a ghost, no ramp),
    // z = haze mix, w = deband dither on.
    lane: vec4<f32>,
    // xy = the GlowWave phases `(t mod 3174)/3174` and `(t mod 2805)/2805`, t in ms; zw unused.
    wave: vec4<f32>,
}
@group(0) @binding(3) var<uniform> ffx: FfxCombine;
// The 128×128 wave LUT and its repeat sampler: bound on every combine, sampled only underwater.
@group(0) @binding(4) var wave_tex: texture_2d<f32>;
@group(0) @binding(5) var wave_samp: sampler;

#ifdef WATER_GLOW
// The Improved Water preview's glow (FND-3): the water's highlight excess (`liquid.wgsl`'s
// `water_glow_acc`), 16.16 fixed point, one texel of the quarter target per entry, summed over its
// 4×4 source pixels. Bound on the water downsample alone.
@group(0) @binding(8) var<storage, read> water_glow_acc: array<u32>;
const WATER_GLOW_FIXED: f32 = 65536.0;

// FFXGlow's Box4 with the water's excess in alpha: rgb exactly `fs_downsample`'s, alpha the mean
// excess over the texel's 4×4 block, which the two Gauss passes blur with the colour.
@fragment
fn fs_downsample_water(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(in_tex));
    let one = vec4<f32>(1.0);
    let rgb = (min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(-1.5, -1.5) * texel), one)
        + min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(0.5, -1.5) * texel), one)
        + min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(0.5, 0.5) * texel), one)
        + min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(-1.5, 0.5) * texel), one))
        * 0.25;
    let q = max(vec2<u32>(textureDimensions(in_tex)) / 4u, vec2<u32>(8u));
    let p = min(vec2<u32>(in.position.xy), q - 1u);
    let i = p.y * q.x + p.x;
    var e = 0.0;
    if (i < arrayLength(&water_glow_acc)) {
        e = f32(water_glow_acc[i]) / (16.0 * WATER_GLOW_FIXED);
    }
    return vec4<f32>(rgb.rgb, e);
}

// Round 2: the excess spread wider than FFXGlow's own kernel, so it reads as a halo round the
// glints rather than a fill between them: the blurred quarter texel at `uv` and a ring of eight at
// three quarter texels, its alpha a 30/70 blend of the two; the colour stays the centre's.
fn water_glow_wide(uv: vec2<f32>) -> vec4<f32> {
    let c = textureSample(blur_tex, in_samp, uv);
    let t = 3.0 / vec2<f32>(textureDimensions(blur_tex));
    var ring = 0.0;
    for (var k = 0; k < 8; k = k + 1) {
        let a = f32(k) * 0.7853982 + 0.3927;
        ring = ring + textureSample(blur_tex, in_samp, uv + vec2<f32>(cos(a), sin(a)) * t).a;
    }
    return vec4<f32>(c.rgb, 0.3 * c.a + 0.7 * ring / 8.0);
}

// The weight's trim (round 2): weight 6 fused a sunset's glitter into one white column; this
// takes the default to an effective 2.
const WATER_GLOW_TRIM: f32 = 0.34;

// The blurred excess `b.a` over the combine's gamma output `outg`, as light: weight `wave.z`, capped
// at `wave.w`, in the hue of the blurred neighbourhood, added under a square-law decode so a dark
// night bay blooms more than a bright noon one.
fn water_glow_over(outg: vec3<f32>, b: vec4<f32>) -> vec3<f32> {
    let e = min(max(b.a, 0.0), ffx.wave.w) * ffx.wave.z * WATER_GLOW_TRIM;
    let m = max(b.r, max(b.g, b.b));
    let hue = select(vec3<f32>(1.0), b.rgb / m, m > 1e-3);
    return min(sqrt(outg * outg + e * hue), vec3<f32>(1.0));
}
#endif



// Box4, full → ¼: four bilinear taps over a 4×4 footprint (`0xce89cc`), each clamped to 1.0 like
// the reference's byte target; an unclamped super-white would square into a hard glow disc.
@fragment
fn fs_downsample(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(in_tex));
    let one = vec4<f32>(1.0);
    return (min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(-1.5, -1.5) * texel), one)
        + min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(0.5, -1.5) * texel), one)
        + min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(0.5, 0.5) * texel), one)
        + min(textureSample(in_tex, in_samp, in.uv + vec2<f32>(-1.5, 0.5) * texel), one))
        * 0.25;
}

// Gauss4: taps at ±0.5 and ±2.5 source texels (0x6cad10), weights from FFXGauss4.bls.
fn gauss4(uv: vec2<f32>, axis: vec2<f32>) -> vec4<f32> {
    let texel = axis / vec2<f32>(textureDimensions(in_tex));
    return textureSample(in_tex, in_samp, uv - 2.5 * texel) * 0.125
        + textureSample(in_tex, in_samp, uv - 0.5 * texel) * 0.375
        + textureSample(in_tex, in_samp, uv + 0.5 * texel) * 0.375
        + textureSample(in_tex, in_samp, uv + 2.5 * texel) * 0.125;
}

@fragment
fn fs_gauss_h(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return gauss4(in.uv, vec2<f32>(1.0, 0.0));
}

@fragment
fn fs_gauss_v(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return gauss4(in.uv, vec2<f32>(0.0, 1.0));
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

// Deviation, opt-in (`WOW_DITHER=1`): a dither against banding, which the reference's 8-bit
// framebuffer lacks. The hash is Bevy's `screen_space_dither`, copied here because Bevy applies
// it only in the tonemapping pass that `Tonemapping::None` skips.
fn screen_space_dither(frag_coord: vec2<f32>) -> vec3<f32> {
    var dither = vec3<f32>(dot(vec2<f32>(171.0, 231.0), frag_coord)).xxx;
    dither = fract(dither.rgb / vec3<f32>(103.0, 71.0, 97.0));
    return (dither - vec3<f32>(0.5)) / 255.0;
}

// The combine's exit: dither in gamma space, where the present-encode rounds, then the frame's
// one decode, which the sRGB present-encode reverses byte-exactly; none under `GAMMA_OUT`.
fn combine_out(outg: vec3<f32>, alpha: f32, frag_coord: vec2<f32>) -> vec4<f32> {
    let dithered = clamp(
        outg + screen_space_dither(frag_coord) * step(0.5, ffx.lane.w),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
#ifdef GAMMA_OUT
    // The UI camera's ground pass stores premultiplied bytes like `ui_quad.wgsl`; the lane decodes.
    return vec4<f32>(dithered * alpha, alpha);
#else
    return vec4<f32>(srgb_to_linear(dithered), alpha);
#endif
}

// Both combines, at the plain or the warped UV; `frag_coord` stays unwarped for the dither.
fn combine_body(uv: vec2<f32>, frag_coord: vec2<f32>) -> vec4<f32> {
    let scene = textureSample(in_tex, in_samp, uv);
    // Clamped like the downsample taps: the screen term must saturate before the glow add.
    let sg = clamp(scene.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
#ifdef WATER_GLOW
    let blur4 = water_glow_wide(uv);
    let bg = max(blur4.rgb, vec3<f32>(0.0));
#else
    let bg = max(textureSample(blur_tex, in_samp, uv).rgb, vec3<f32>(0.0));
#endif
    // Scene alpha passes through; only the transparent-clear create booth reads it.
    if ffx.lane.y > 0.0 {
        // FFXDeath replaces the glow while a ghost, unhazed (one pass slot, `[0xce8bb4]`). The
        // 0.144 blue weight is as shipped; the tint is packed at `0x6cb930`.
        let glowed = sg + ffx.lane.x * bg * bg;
        let luma = clamp(dot(glowed, vec3<f32>(0.299, 0.587, 0.144)), 0.0, 1.0);
        let p = clamp(4.0 * luma * (1.0 - luma), 0.0, 1.0);
        let ghost_tint = vec3<f32>(83.0, 147.0, 168.0) / 255.0; // 0x5393A8
        let outg = min(vec3<f32>(luma) + ghost_tint * p, vec3<f32>(1.0));
        return combine_out(outg, scene.a, frag_coord);
    }
    // The haze fades toward the blur before the glow add; z is the larger of the drunk fraction
    // and 84/255 while submerged.
    let hazed = mix(sg, min(bg, vec3<f32>(1.0)), ffx.lane.z);
#ifdef WATER_GLOW
    let outg = water_glow_over(min(hazed + ffx.lane.x * bg * bg, vec3<f32>(1.0)), blur4);
#else
    let outg = min(hazed + ffx.lane.x * bg * bg, vec3<f32>(1.0));
#endif
    return combine_out(outg, scene.a, frag_coord);
}

@fragment
fn fs_combine(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return combine_body(in.uv, in.position.xy);
}

// FFXGlowWave, the warp the reference runs instead of FFXGlow while the eye is in any liquid
// (`0x6cc630`, `0x6cb310`): the same combine, both texcoords displaced by a 128×128 sine LUT. The
// reference's per-vertex wave texcoord (`0x7bca80`) is affine in the screen UV, so per fragment
// is exact: uv' = uv · R(10°) · S(W/128, 0.88·H/128) · T(p1, p2), fixed to screen pixels.
const WAVE_ROT_RAD: f32 = 0.174532925; // 10°, the reference's rotation row
const WAVE_LUT_EDGE: f32 = 128.0; // the LUT is 128×128 texels ([0xce89a0])
const WAVE_V_SCALE: f32 = 0.88; // [0xce89c0]: the v axis only; u's scale is 1.0
const WAVE_AMPLITUDE_PX: f32 = 3.0; // full-res pixels, both samples

// The warp's screen-UV offset in UV units; `screen` is the full-res scene size in pixels.
fn wave_offset(uv: vec2<f32>, screen: vec2<f32>) -> vec2<f32> {
    let c = cos(WAVE_ROT_RAD);
    let s = sin(WAVE_ROT_RAD);
    let scale = vec2<f32>(screen.x, WAVE_V_SCALE * screen.y) / WAVE_LUT_EDGE;
    let tex = vec2<f32>(c * uv.x - s * uv.y, s * uv.x + c * uv.y) * scale + ffx.wave.xy;
    // Sampled, not evaluated: the reference's sine is a filtered LUT. The sampler must repeat,
    // as the texcoord spans about 10 cycles; the unbias matches the shipped ps_2_0 shader.
    let d = (textureSample(wave_tex, wave_samp, tex).rg - vec2<f32>(0.5)) * 2.0;
    return d * WAVE_AMPLITUDE_PX / screen;
}

@fragment
fn fs_combine_wave(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let screen = vec2<f32>(textureDimensions(in_tex));
    return combine_body(in.uv + wave_offset(in.uv, screen), in.position.xy);
}

#ifdef UNDERWATER_IMPROVED
// ---- the Improved Water preview's underwater combine (UW-2/3/4) ---------------------------------
//
// `liquid.wgsl`'s eye-below rows and the wake-underwater module's law, pasted between the same
// markers: pure functions, no bindings.
/// The eye-below rows, written by `benilla_water`'s `under` lanes every frame: the member `under`
/// in the preview block, and the improved underwater combine's uniform (`ffx_glow.wgsl`).
struct UnderLanes {
    // xyz the eye, w the near plane's distance.
    cam_pos: vec4<f32>,
    // xyz the view's forward axis, w the angle one output pixel subtends (radians).
    cam_fwd: vec4<f32>,
    // xyz the right axis times tan(half the horizontal field); then the up axis times tan(half the
    // vertical field); w spare on both.
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    // x the water surface's world Y over the eye, y the eye's depth under it, z 1 while the eye is
    // in a water kind (0 dry, magma or slime), w the feature bits.
    surface: vec4<f32>,
    // rgb the absorption per yard, w the in-scatter's extinction per yard.
    sigma: vec4<f32>,
    // rgb the swapped (underwater) fog colour, then the dry fog colour, gamma; w spare.
    fog_wet: vec4<f32>,
    fog_dry: vec4<f32>,
    // x/y the swapped fog's start and end, z/w the dry fog's, yards.
    fog_span: vec4<f32>,
    // rgb the dry dome's zenith and horizon stops, gamma; w spare.
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    // rgb the dry sun's colour (Light.dbc row 9), gamma; w spare.
    sun_dry: vec4<f32>,
    // xyz the lighting sun refracted into the water, unit, pointing up toward it; w spare.
    sun_refr: vec4<f32>,
    // rgb the lighting sun's irradiance, linear; w its share of the light under the surface (0 at
    // night or under cloud).
    sun_e: vec4<f32>,
    // x the underwater clock in seconds reduced by `UW_CLOCK_PERIOD`, y the quality tier; zw spare.
    clock: vec4<f32>,
    spare: vec4<f32>,
};

// ---- wu law: begin ----------------------------------------------------------------------------
// ---- the switches -------------------------------------------------------------------------------

/// One bit per preview feature, `WOW_PREVIEW_WAKE_UNDERWATER`'s names in order.
const WU_KELVIN: u32 = 1u;
const WU_BOW: u32 = 2u;
const WU_COLLAR: u32 = 4u;
const WU_TRAIL: u32 = 8u;
const WU_RINGS: u32 = 16u;
const WU_WINDOW: u32 = 32u;
const WU_TINT: u32 = 64u;
const WU_CAUSTICS: u32 = 128u;
const WU_SHAFTS: u32 = 256u;
const WU_WATERLINE: u32 = 512u;
const WU_CAUSTICS_ABOVE: u32 = 1024u;
const WU_BODY_CUT: u32 = 2048u;

/// Whether the bits lane `bits` (an integer held in a float) has feature `b`.
fn wu_has(bits: f32, b: u32) -> bool {
    return (u32(max(bits, 0.0) + 0.5) & b) != 0u;
}

// ---- constants ----------------------------------------------------------------------------------

const WU_TAU: f32 = 6.2831853;
/// Gravity in yards per second squared (9.81 m/s²).
const WU_G: f32 = 10.7283;
/// Fresh water's index, and per channel at about 650, 550 and 450 nm (blue a little high, so the
/// window's rim shows its fringe).
const WU_ETA: f32 = 1.333;
const WU_ETA_RGB: vec3<f32> = vec3<f32>(1.3315, 1.333, 1.3355);
/// Water's reflectance head-on.
const WU_F0: f32 = 0.02;
const WU_LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);
/// Distance standing in for "nothing along the ray", yards.
const WU_FAR: f32 = 100000.0;
/// The wake's and the underwater clocks' periods, seconds; every time term turns a whole number
/// of times in its period, so neither clock's wrap shows.
const WU_CLOCK_PERIOD: f32 = 64.0;
const UW_CLOCK_PERIOD: f32 = 256.0;

/// The slots the lanes hold.
const WU_SWIMMERS: u32 = 4u;
const WU_RINGS_MAX: u32 = 8u;

/// A swimmer makes waves from this speed and its wake is whole from the second, yd/s.
const WU_SPEED_MIN: f32 = 0.6;
const WU_SPEED_FULL: f32 = 3.0;
/// The body's pressure patch: its σ over the half width at the waterline, about a third of the
/// body's length, the scale that narrows the wake at swim speed (to about 11° at 4.72 yd/s).
const WU_PATCH_SIGMA: f32 = 2.0;
/// The Kelvin field's steepest crest at `WU_KELVIN_REF_YD` (above the ambient ripple's tilt there,
/// or the arms vanish into it), its decay length behind the bow and its reach (inside the wave
/// field's 24-yard half window, faded over its last 30%), yards.
const WU_KELVIN_SLOPE: f32 = 0.45;
const WU_KELVIN_REF_YD: f32 = 3.0;
const WU_KELVIN_DECAY: f32 = 14.0;
const WU_KELVIN_REACH: f32 = 22.0;
/// The transverse system's gain over the divergent one's (1 physical): at swim speed the patch
/// spectrum leaves them faint, and a little more lets the wedge's inside read.
const WU_KELVIN_TRANSVERSE: f32 = 1.6;
/// tan and cos of the Kelvin cusp's polar angle, 19.47° (tan = 1/(2√2)).
const WU_CUSP_TAN: f32 = 0.35355339;
const WU_CUSP_COS: f32 = 0.94280904;
/// The crest profile e^{q(cos φ − 1)}: its sharpness, its mean (e^{−q}·I0(q)), its steepest
/// slope.
const WU_CREST_Q: f32 = 1.1;
const WU_CREST_MEAN: f32 = 0.4414;
const WU_CREST_SLOPE: f32 = 0.57;
/// How far the crests carry past the cusp line, in units of the Airy width (r/K²)^(1/3).
const WU_FEATHER: f32 = 1.16;
/// The bow wave's gain, and its crest's apex ahead of the bow point in head units.
const WU_BOW_GAIN: f32 = 0.55;
const WU_BOW_APEX: f32 = 0.015;
/// The steepest normal tilt the summed wake may give.
const WU_SLOPE_MAX: f32 = 0.85;
/// A fragment takes a swimmer's wake only within this height of the swimmer's water, yards.
const WU_SURFACE_MATCH: f32 = 1.0;
/// The collar's strength on a swimmer at rest, against one under way.
const WU_COLLAR_REST: f32 = 0.35;
/// A ring's slope at its packet, per unit of strength (about 0.14 half a yard out from a hand at
/// strength 0.8, against an ambient ripple tilt of 0.3 to 0.7 near the eye).
const WU_RING_SLOPE: f32 = 1.0;

/// Snell's window: its gain over the n² radiance the air's light gains entering the water, the
/// land haze's share at its rim, the sun's halo and disc, and the transmitted glint's weight.
const WU_WINDOW_GAIN: f32 = 1.25;
const WU_RIM_LAND: f32 = 0.3;
const WU_SUN_HALO: f32 = 0.6;
const WU_SUN_DISC: f32 = 6.0;
const WU_WINDOW_GLINT: f32 = 2.0;
const WU_GLINT_ALPHA_MIN: f32 = 0.06;
/// The internal mirror: the mirrored bed's share over the in-scatter, and its blur in pixels.
const WU_TIR_BED: f32 = 0.38;
const WU_TIR_BLUR_PX: f32 = 3.0;
/// Foam seen from below: the light it passes, against its lit top.
const WU_FOAM_UNDER: f32 = 0.45;
/// The aerated water under a trail: how far it pales toward foam.
const WU_AER_MILK: f32 = 0.3;

/// The in-scatter's palette from the swapped fog colour: its luminance ceiling (linear), the floor
/// its weakest channel keeps of its strongest, the deep and the shallow ends against the mid.
const WU_INSCATTER_LUMA_MAX: f32 = 0.30;
const WU_INSCATTER_SAT_FLOOR: f32 = 0.12;
const WU_DEEP_SHARE: f32 = 0.30;
const WU_SHALLOW_GAIN: f32 = 1.9;
const WU_SHALLOW_LIFT: f32 = 0.01;
/// The in-scatter's darkening with depth under the surface, per yard.
const WU_DEPTH_DIM: f32 = 0.02;
/// The shafts' and the caustics' strength, the caustics' fade depth (yards) and their cell rates
/// (cells a yard: 56 and 80 to the 64-yard tile).
const WU_SHAFT_GAIN: f32 = 2.2;
const WU_CAUSTIC_GAIN: f32 = 1.5;
const WU_CAUSTIC_DEPTH: f32 = 14.0;
const WU_CAUSTIC_F1: f32 = 0.875;
const WU_CAUSTIC_F2: f32 = 1.25;
const WU_CAUSTIC_MEAN: f32 = 0.22;
/// The light curtains: three plane waves on the 64-yard lattice (2π/64 × (11, 4), (−6, 10),
/// (3, −12), about 5.4 yd apart), their phase turns per clock period, and the fundamentals' and
/// harmonics' amplitudes.
const WU_SHAFT_K0: vec2<f32> = vec2<f32>(1.0799225, 0.3926991);
const WU_SHAFT_K1: vec2<f32> = vec2<f32>(-0.5890486, 0.9817477);
const WU_SHAFT_K2: vec2<f32> = vec2<f32>(0.2945243, -1.1780972);
const WU_SHAFT_M0: f32 = 23.0;
const WU_SHAFT_M1: f32 = -17.0;
const WU_SHAFT_M2: f32 = 29.0;
const WU_SHAFT_A: f32 = 0.24;
const WU_SHAFT_H: f32 = 0.12;
/// The waterline's band at the near plane, pixels.
const WU_MENISCUS_PX: f32 = 9.0;
/// The combine's output range under water, gamma: no channel reaches 0 or 1.
const WU_OUT_FLOOR: f32 = 0.016;
const WU_OUT_TOP: f32 = 0.965;

// ---- helpers ------------------------------------------------------------------------------------

/// Gamma to linear light and back, the water's pure 2.2 power law (`liquid.wgsl`'s `WATER_GAMMA`).
fn wu_dec(g: vec3<f32>) -> vec3<f32> {
    return pow(max(g, vec3<f32>(0.0)), vec3<f32>(2.2));
}

fn wu_enc(l: vec3<f32>) -> vec3<f32> {
    return pow(max(l, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

/// A lattice point hashed to [0, 1) (Hoskins' hash12).
fn wu_hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + vec3<f32>(33.33));
    return fract((p3.x + p3.y) * p3.z);
}

/// A lattice point hashed to two values in [0, 1) (Hoskins' hash22).
fn wu_hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + dot(p3, p3.yzx + vec3<f32>(33.33));
    return fract((p3.xx + p3.yz) * p3.zy);
}

/// Value noise in [0, 1], smooth across cells.
fn wu_vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (vec2<f32>(3.0) - 2.0 * f);
    let a = wu_hash21(i);
    let b = wu_hash21(i + vec2<f32>(1.0, 0.0));
    let c = wu_hash21(i + vec2<f32>(0.0, 1.0));
    let d = wu_hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

/// Worley's nearest and second-nearest feature distances on a jittered cell lattice; each feature
/// point circles its cell's centre, `phase` radians along.
fn wu_worley(p: vec2<f32>, phase: f32) -> vec2<f32> {
    let i = floor(p);
    let f = p - i;
    var d1 = 8.0;
    var d2 = 8.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let c = vec2<f32>(f32(x), f32(y));
            let h = wu_hash22(i + c);
            let o = vec2<f32>(0.5) + 0.38 * sin(vec2<f32>(phase) + WU_TAU * h);
            let d = length(c + o - f);
            if (d < d1) {
                d2 = d1;
                d1 = d;
            } else if (d < d2) {
                d2 = d;
            }
        }
    }
    return vec2<f32>(d1, d2);
}

/// GGX's distribution at `nh` for roughness `alpha`, over its peak: 1 on the mirror direction.
fn wu_ggx_peak1(nh: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 * a2 / max(d * d, 1e-8);
}

/// The exposure ceiling, linear: the peak channel rolls from `knee` toward `top` on a soft
/// shoulder with the hue kept, and no channel falls under `floor_share` of the luminance.
fn wu_ceiling(c: vec3<f32>, knee: f32, top: f32, floor_share: f32) -> vec3<f32> {
    let m = max(max(c.r, c.g), max(c.b, 1e-6));
    let span = max(top - knee, 1e-4);
    let m2 = select(m, knee + span * (1.0 - exp(-(m - knee) / span)), m > knee);
    let o = c * (m2 / m);
    return max(o, vec3<f32>(dot(o, WU_LUMA) * floor_share));
}

// ---- the wake (INT-1/2/4/5) ---------------------------------------------------------------------

/// What the wake adds at a fragment.
struct WuWake {
    // The height's gradient in world XZ; the normal's horizontal part is its negative.
    grad: vec2<f32>,
    // The height, yards.
    h: f32,
    // Foam coverage, 0-1: collar, bow crest, kicked trail, splash rings.
    foam: f32,
    // The aerated water's share under the foam, 0-1.
    aer: f32,
    // The share of the live wave field kept here.
    keep: f32,
};

fn wu_wake_none() -> WuWake {
    return WuWake(vec2<f32>(0.0), 0.0, 0.0, 0.0, 1.0);
}

/// Two wakes on one fragment: the waves add, the foams and aerations take the larger, the wave
/// field keeps the smaller share.
fn wu_wake_add(a: WuWake, b: WuWake) -> WuWake {
    return WuWake(
        a.grad + b.grad,
        a.h + b.h,
        max(a.foam, b.foam),
        max(a.aer, b.aer),
        min(a.keep, b.keep),
    );
}

/// The summed wake's normal tilt −∇h, held under `WU_SLOPE_MAX` so no crest tips past it.
fn wu_wake_tilt(w: WuWake) -> vec2<f32> {
    let m = length(w.grad);
    return -w.grad * min(1.0, WU_SLOPE_MAX / max(m, 1e-6));
}

/// The far field of a pressure patch moving at `u` yd/s over deep water, by stationary phase
/// (Kelvin's ship wake): both wave systems that meet a point inside the 19.47° wedge, the crests
/// fixed to the swimmer, on a peaked profile so they read as thin bright lines. `lat` is the offset
/// across the track and `back` the distance behind the source, yards; `sig2` the patch's (σ/2)²,
/// `fp` the pixel's footprint. Returns the height and its derivatives across and behind, unscaled
/// ([`wu_kelvin_norm`]). Waves shorter than about three pixels fade out; past the cusp the crests
/// carry on as feathers over an Airy width.
fn wu_kelvin(lat: f32, back: f32, u: f32, sig2: f32, fp: f32) -> vec3<f32> {
    if (back < 0.05 || u < WU_SPEED_MIN) {
        return vec3<f32>(0.0);
    }
    // K = g/U², the transverse wavenumber: 2πU²/g is 13 yd at swim speed.
    let kw = WU_G / (u * u);
    let y = abs(lat);
    let sy = select(1.0, -1.0, lat < 0.0);
    let r = max(sqrt(lat * lat + back * back), 0.5);
    var tau = y / back;
    var out_d = 0.0;
    if (tau > WU_CUSP_TAN) {
        out_d = (y - WU_CUSP_TAN * back) * WU_CUSP_COS;
        tau = WU_CUSP_TAN;
    }
    // tan φ = T/(1 + 2T²) has two roots T = tan θ: the transverse (θ < 35.26°) and the divergent.
    let disc = sqrt(max(1.0 - 8.0 * tau * tau, 1e-12));
    let ca = inverseSqrt(1.0 + tau * tau);
    let sa = tau * ca;
    // The cusp's caustic softened over its Airy scale, so the arms are crests and not beads.
    let eps = 2.4 * pow(kw * r, -0.33333333);
    let near = inverseSqrt(max(r, 1.0) / WU_KELVIN_REF_YD);
    var acc = vec3<f32>(0.0);
    for (var j = 0; j < 2; j = j + 1) {
        let transverse = j == 0;
        let t = select(
            min((1.0 + disc) / (4.0 * tau + 1e-4), 40.0),
            2.0 * tau / (1.0 + disc),
            transverse,
        );
        let sec2 = 1.0 + t * t;
        let c = inverseSqrt(sec2);
        let s = t * c;
        let k = kw * sec2;
        // The phase's second derivative over θ, which sets the stationary-phase amplitude.
        let f2 = sec2 * (c * ca + s * sa) * (1.0 - 2.0 * t * t);
        var amp = sec2 * sec2 * exp(-k * k * sig2) * inverseSqrt(sqrt(f2 * f2 + eps * eps));
        amp = amp * (1.0 - smoothstep(0.22, 0.45, fp * k / WU_TAU));
        amp = amp * select(1.0, WU_KELVIN_TRANSVERSE, transverse);
        let ph = k * (back * c + y * s) + select(-0.78539816, 0.78539816, transverse);
        let ex = exp(WU_CREST_Q * (cos(ph) - 1.0));
        let g = -WU_CREST_Q * sin(ph) * ex;
        acc = acc + amp * vec3<f32>(ex - WU_CREST_MEAN, g * k * s * sy, g * k * c);
    }
    let band = WU_FEATHER * pow(r / (kw * kw), 0.33333333);
    let fade = exp(-pow(max(out_d / band, 1e-6), 1.3)) * smoothstep(0.1, 1.1, back);
    return acc * (fade * near);
}

/// The scale that brings [`wu_kelvin`]'s steepest crest near a slope of 1 at `WU_KELVIN_REF_YD`:
/// over its divergent waves' spectral peak (sec²θ·e^{−k²(σ/2)²} at sec⁴θ = 1/(2K²(σ/2)²), the
/// transverse wave's at a slow source), their wavenumber there and the crest profile's slope.
fn wu_kelvin_norm(u: f32, sig2: f32) -> f32 {
    let kw = WU_G / max(u * u, 1e-3);
    let sec2 = max(inverseSqrt(2.0 * kw * kw * sig2), 1.0);
    let k = kw * sec2;
    let peak = sec2 * exp(-k * k * sig2);
    return 1.0 / max(peak * k * WU_CREST_SLOPE, 1e-4);
}

/// The bow wave in head units (half a shoulder span is 0.285), `q` = (across, ahead) of the bow
/// point: a crest wrapped round the head as a hyperbola opening at about 32° with a trough inside
/// it, both gone within a body length. Returns the height (head units) and the crest's share.
fn wu_bow(q: vec2<f32>) -> vec2<f32> {
    let xb = WU_BOW_APEX - q.y;
    let y = abs(q.x);
    let yy = y / 0.19;
    let sq = sqrt(1.0 + yy * yy);
    let xc = 0.30 * (sq - 1.0) - 0.05;
    let sl = 0.30 * yy / (0.19 * sq);
    // Signed distance to the crest line, positive ahead of and outside it.
    let d = (xc - xb) * inverseSqrt(1.0 + sl * sl);
    let behind = max(xb, 0.0);
    let along = sqrt(y * y + behind * behind);
    let wd = 0.05 + 0.06 * along;
    let cr = exp(-d * d / (wd * wd * select(1.0, 0.45, d > 0.0)));
    let dt = d + 1.9 * wd;
    let tr = exp(-dt * dt / (2.0 * wd * wd));
    let a = 0.055 * exp(-along / 0.42);
    return vec2<f32>(a * (cr - 0.8 * tr), cr * exp(-along / 0.45));
}

/// Distance from `q` (across, ahead; yards from the bow point) to the swimmer's footprint at the
/// waterline: a capsule from the head back to the hips, `hw` its half width at the head, tapering.
fn wu_body_dist(q: vec2<f32>, hw: f32) -> f32 {
    let head = -0.9 * hw;
    let hips = -3.2 * hw;
    let t = clamp((q.y - head) / (hips - head), 0.0, 1.0);
    let p = vec2<f32>(q.x, q.y - mix(head, hips, t));
    return length(p) - hw * mix(1.0, 0.7, t);
}

/// The foam collar at the body's waterline: a thin broken cream band just outside the footprint,
/// wider at the head where the bow piles water up. `dc` the footprint distance and `aa` the
/// antialias width (yards), `front` 1 at the head and 0 at the hips, `brk` a cell-border break-up
/// in [0, 1], `wob` the band's wobble in [−0.5, 0.5].
fn wu_collar(dc: f32, hw: f32, front: f32, brk: f32, wob: f32, aa: f32) -> f32 {
    // The sketch's widths are metres on a 0.356 m shoulder; here they scale with the body.
    let bs = hw / 0.356;
    let cw = (0.030 + 0.045 * front + 0.03 * wob) * bs;
    let band = 1.0 - smoothstep(cw - aa - 0.01, cw, dc);
    return band * mix(0.45, 1.0, brk) * smoothstep(-0.02, 0.0, dc);
}

/// The kicked trail's coverage behind a swimmer, the sketch's law in its metres: `lat` across and
/// `b` behind the kick's origin, `age` seconds since the water was kicked, `n1` and `n2` two value
/// noises in world space. The churn just behind the feet, then a trail that widens with √age and
/// thins over 0.8 s and 3 s.
fn wu_trail_cover(lat: f32, b: f32, age: f32, n1: f32, n2: f32) -> f32 {
    let churn = smoothstep(0.8, 1.4, b) * (1.0 - smoothstep(1.9, 3.0, b))
        * (1.0 - smoothstep(0.18, 0.45, lat));
    let latn = lat / (0.20 + 0.15 * sqrt(age));
    var cov = (0.30 * exp(-age / 0.8) + 0.15 * exp(-age / 3.0))
        * (1.0 - smoothstep(0.35, 1.1, latn + 0.45 * (n1 - 0.5)));
    cov = max(cov, churn * 0.42) * smoothstep(0.05, 0.45, b) * (0.55 + 0.9 * n2);
    return cov;
}

/// Foam as a web: the borders of warped bubble cells at two sizes. Coverage `c` widens the threads
/// from lace to a holed sheet; as it falls they thin, break into islands and go. `aa` the pixel's
/// footprint in cell units. A crisp outer edge, the thread's centre denser than its rim, and
/// threads too thin to resolve fade instead of aliasing.
fn wu_foam_lace(p: vec2<f32>, c: f32, aa: f32) -> f32 {
    let q = p + 1.1 * vec2<f32>(wu_vnoise(p * 0.5), wu_vnoise(p * 0.5 + vec2<f32>(5.2)));
    let w1 = wu_worley(q * 0.85, 0.0);
    let w2 = wu_worley(q * 2.2 + vec2<f32>(7.0), 0.0);
    let e = min(w1.y - w1.x, (w2.y - w2.x) * 0.8 + 0.10 * (1.0 - c));
    let wid = mix(0.02, 0.36, c * c) * (0.78 + 0.44 * wu_vnoise(q * 1.3 + vec2<f32>(3.0)));
    var a = 1.0 - smoothstep(wid - aa * 0.5, wid + aa, e);
    a = a * mix(0.72, 1.0, smoothstep(0.0, wid * 0.8 + 1e-3, wid - e));
    // Older foam: whole patches of the web gone.
    a = a * smoothstep(0.30, 0.52, wu_vnoise(q * 0.55 + vec2<f32>(17.0)) * 0.8 + 0.1 + 0.45 * c);
    return a * mix(0.70, 0.97, c) * smoothstep(0.2, 1.2, wid / max(aa, 1e-3))
        * smoothstep(0.04, 0.22, c);
}

/// One swimmer's wake at `xz`: the Kelvin field, the bow wave, the collar and the kicked trail,
/// from its two lane rows `a` and `b` ([`WakeLanes::swimmer`]). `surf_y` is the fragment's surface
/// height, `fp` its footprint and `aa` its antialias width in yards, `cfg` the wake's switch row.
/// The trail and the collar's break-up read world space, so foam stays where it was made while the
/// waves ride with the swimmer.
fn wu_swimmer(
    xz: vec2<f32>,
    surf_y: f32,
    fp: f32,
    aa: f32,
    a: vec4<f32>,
    b: vec4<f32>,
    cfg: vec4<f32>,
) -> WuWake {
    var o = wu_wake_none();
    let bits = cfg.x;
    let fwd = b.xy;
    let side = vec2<f32>(fwd.y, -fwd.x);
    let d = xz - a.xz;
    // (across, ahead) of the bow point.
    let q = vec2<f32>(dot(d, side), dot(d, fwd));
    let back = -q.y;
    let u = b.z;
    let hw = max(b.w, 0.1);
    if (abs(surf_y - a.y) > WU_SURFACE_MATCH
        || q.y > 4.0 * hw + 1.0
        || back > WU_KELVIN_REACH
        || abs(q.x) > 1.5 + 0.5 * max(back, 0.0)) {
        return o;
    }
    let st = a.w;
    let ramp = smoothstep(WU_SPEED_MIN, WU_SPEED_FULL, u);
    let ph = WU_TAU * cfg.y / WU_CLOCK_PERIOD;
    // Gradient in (across, ahead).
    var grad = vec2<f32>(0.0);
    var h = 0.0;

    if (wu_has(bits, WU_KELVIN) && back > 0.05) {
        let sig = WU_PATCH_SIGMA * hw;
        let sig2 = 0.25 * sig * sig;
        let gain = st * ramp * WU_KELVIN_SLOPE * wu_kelvin_norm(u, sig2)
            * exp(-back / WU_KELVIN_DECAY)
            * (1.0 - smoothstep(0.7 * WU_KELVIN_REACH, WU_KELVIN_REACH, back));
        let kv = wu_kelvin(q.x, back, u, sig2, fp) * gain;
        h = h + kv.x;
        grad = grad + vec2<f32>(kv.y, -kv.z);
    }

    var crest = 0.0;
    if (wu_has(bits, WU_BOW)) {
        let sl = 0.285 / hw;
        let ql = q * sl;
        if (ql.y > -3.5 && ql.y < 1.5 && abs(ql.x) < 2.4) {
            let e = 0.015;
            let b0 = wu_bow(ql);
            let bx = wu_bow(ql + vec2<f32>(e, 0.0));
            let bz = wu_bow(ql + vec2<f32>(0.0, e));
            // The crest is 0.05 head units wide; past a pixel it becomes roughness, not a line.
            let keep = (1.0 - smoothstep(0.5, 1.5, fp * sl / 0.05)) * st * ramp * WU_BOW_GAIN;
            h = h + b0.x / sl * keep;
            // Slopes are the same in head units and yards.
            grad = grad + vec2<f32>(bx.x - b0.x, bz.x - b0.x) / e * keep;
            crest = b0.y * st * ramp;
        }
    }

    var foam = 0.0;
    var aer = 0.0;
    if (wu_has(bits, WU_COLLAR)) {
        let dc = wu_body_dist(q, hw);
        if (dc < 0.6 * hw + 0.1) {
            let front = smoothstep(-2.5 * hw, 0.0, q.y);
            let ang = atan2(q.x, q.y + 0.9 * hw);
            // Whole turns of the angle, so the wobble meets itself behind the head.
            let wob = 0.35 * sin(ang * 5.0 + ph * 19.0) + 0.15 * sin(ang * 11.0 - ph * 31.0);
            let wc = wu_worley(xz * 12.8, ph * 7.0);
            let brk = smoothstep(0.08, 0.22, wc.y - wc.x);
            let rest = mix(WU_COLLAR_REST, 1.0, ramp);
            foam = max(foam, wu_collar(dc, hw, front, brk, wob, aa) * st * rest);
        }
    }
    if (crest > 0.01) {
        // Two still noises crossfaded by the clock: the crest's foam breathes and never pops.
        let n = mix(
            wu_vnoise(xz * 8.2),
            wu_vnoise(xz * 8.2 + vec2<f32>(7.3)),
            0.5 + 0.5 * sin(ph * 23.0),
        );
        foam = max(foam, smoothstep(0.45, 0.75, crest * (0.7 + 0.6 * n)));
    }
    if (wu_has(bits, WU_TRAIL) && u > WU_SPEED_MIN) {
        // Sketch metres for this body, measured from the kick's origin 0.73 m behind the head.
        let bs = hw / 0.356;
        let bk = back / bs - 0.73;
        let age = max(back - 0.73 * bs, 0.0) / u;
        if (bk > 0.05 && age < 4.5) {
            let n1 = wu_vnoise(xz * 1.92 + vec2<f32>(3.0));
            let n2 = wu_vnoise(xz * 1.28 + vec2<f32>(9.0));
            let cov = wu_trail_cover(abs(q.x) / bs, bk, age, n1, n2) * st * ramp;
            aer = max(aer, saturate(cov * 1.9) * 0.62);
            if (cov > 0.02) {
                // 3.2 cells a metre.
                let s = 2.93;
                let lace = wu_foam_lace(xz * s, saturate(cov / 0.58), fp * s * 1.5 + 0.015);
                foam = max(foam, lace);
            }
        }
    }

    o.grad = grad.x * side + grad.y * fwd;
    o.h = h;
    o.foam = foam;
    o.aer = aer;
    // The wave field stands back inside the wedge, where this draws the swimmer's own waves.
    let wedge = (1.0 - smoothstep(0.0, 1.5, abs(q.x) - 0.25 * max(back, 0.0) - 1.0))
        * (1.0 - smoothstep(0.6 * WU_KELVIN_REACH, WU_KELVIN_REACH, back));
    o.keep = mix(1.0, cfg.z, wedge * st * ramp);
    return o;
}

/// The waves from one impulse at the surface (Cauchy–Poisson): at range `r` and age τ the local
/// wavenumber is k = g·τ²/(4r²), so crests are born at the centre and run outward at twice the
/// packet's speed. `a` the source's radius sets the packet, `decay` its life, `fp` the footprint.
/// Returns the height in units of `a` and its gradient (world XZ, `dq` from the centre).
fn wu_ring(dq: vec2<f32>, age: f32, a: f32, decay: f32, fp: f32) -> vec3<f32> {
    let r = length(dq) + 1e-3;
    let tau = age + 0.12;
    let k = WU_G * tau * tau / (4.0 * r * r);
    let ka = k * a;
    var env = ka * ka * exp(-ka * ka) * exp(-age / decay) / (1.0 + r / (6.0 * a));
    env = env * (1.0 - smoothstep(0.22, 0.45, fp * k / WU_TAU));
    let ph = k * r;
    return env * vec3<f32>(cos(ph), sin(ph) * k * dq / r);
}

/// The splash an impulse leaves: a broken foam ring spreading from the source and a short-lived
/// core, gone within a second. `r` the range and `a` the source's radius (yards), `brk` the ring's
/// angular break-up and `core_n` the core's, in [0, 1].
fn wu_ring_foam(r: f32, age: f32, a: f32, brk: f32, core_n: f32) -> f32 {
    // The sketch's hand is 0.07 yd; larger sources spread wider foam.
    let s = sqrt(max(a, 0.01) / 0.07);
    let rad = (0.07 + 0.30 * sqrt(age)) * s;
    let life = 1.0 - smoothstep(0.2, 0.9, age);
    let ring = (1.0 - smoothstep(0.010 * s, 0.035 * s, abs(r - rad))) * life
        * smoothstep(0.4, 0.65, brk);
    let core = (1.0 - smoothstep(rad * 0.2, rad * 0.8, r)) * (1.0 - smoothstep(0.05, 0.35, age))
        * smoothstep(0.3, 0.5, core_n);
    return max(ring, core * 0.9);
}

/// One ring event's waves and splash at `xz`, from its two lane rows `ra` and `rb`
/// ([`WakeLanes::ring`]); nothing past the packet's reach.
fn wu_ring_event(xz: vec2<f32>, fp: f32, ra: vec4<f32>, rb: vec4<f32>) -> WuWake {
    var o = wu_wake_none();
    let dq = xz - ra.xy;
    let r = length(dq);
    let age = ra.z;
    let a = max(rb.x, 0.02);
    // The packet sits where k·a ≈ 1: r = τ·√(g·a/4).
    let reach = 0.5 + 2.5 * (age + 0.12) * sqrt(WU_G * a * 0.25);
    if (r > reach) {
        return o;
    }
    let rv = wu_ring(dq, age, a, max(rb.y, 0.1), fp);
    let g = ra.w * WU_RING_SLOPE * a;
    o.h = rv.x * g;
    o.grad = rv.yz * g;
    if (rb.z > 0.0 && age < 0.9) {
        // The break-up reads the ring's direction, not its angle, so it has no seam behind.
        let dirn = dq / max(r, 1e-4);
        let seed = fract(ra.xy * 0.37);
        let brk = wu_vnoise(dirn * 2.6 + vec2<f32>(age * 2.0, 0.0) + seed * 17.0);
        let core_n = wu_vnoise(dq * (25.0 / sqrt(a / 0.07)) + seed * 29.0);
        o.foam = wu_ring_foam(r, age, a, brk, core_n) * rb.z;
    }
    return o;
}

// ---- the underside, seen from below (UW-1) ------------------------------------------------------

/// The view ray leaving the water upward.
struct WuSnell {
    // Its direction in the air above, at green's index.
    a: vec3<f32>,
    // The transmittance per channel: Fresnel's on the air side's cosine, 0 past the critical angle.
    t: vec3<f32>,
};

/// Snell's window: the view ray `rd` (eye to surface, upward) refracted out through the surface of
/// normal `n` (upward), per channel for the rim's fringe. `soft` widens the critical angle's edge
/// by the roughness the surface does not resolve, so the rim breaks up instead of aliasing.
fn wu_snell(rd: vec3<f32>, n: vec3<f32>, soft: f32) -> WuSnell {
    let ci = clamp(dot(rd, n), 0.0, 1.0);
    let s2 = WU_ETA_RGB * WU_ETA_RGB * (1.0 - ci * ci);
    let k = vec3<f32>(1.0) - s2;
    // Never the square root of an exact 0, which this backend can turn into a NaN.
    let ct = sqrt(max(k, vec3<f32>(1e-12)));
    let f = vec3<f32>(WU_F0) + (1.0 - WU_F0) * pow(vec3<f32>(1.0) - ct, vec3<f32>(5.0));
    let t = (vec3<f32>(1.0) - f) * smoothstep(vec3<f32>(-soft), vec3<f32>(soft), k);
    // refract(rd, −n, η) written out: η·rd + (√k − η·cos i)·n.
    let a = normalize(WU_ETA * rd + (sqrt(max(k.g, 1e-4)) - WU_ETA * ci) * n);
    return WuSnell(a, t);
}

/// The dry sky seen along the air direction `a`, linear: the dome's two stops by elevation (the
/// underside's `sqrt` blend), and toward the horizon a band of land haze in the dry fog's colour,
/// which the window compresses into its rim. Gamma stops in.
fn wu_window_sky(
    a: vec3<f32>,
    zenith_g: vec3<f32>,
    horizon_g: vec3<f32>,
    land_g: vec3<f32>,
) -> vec3<f32> {
    let e = max(a.y, 1e-6);
    let sky = mix(horizon_g, zenith_g, sqrt(e));
    let rim = 1.0 - smoothstep(0.03, 0.16, e);
    let land = mix(horizon_g, land_g, 0.5);
    return wu_dec(mix(sky, land, rim * WU_RIM_LAND));
}

/// The sun broken through the ripples to the eye below: GGX on the refraction half-vector
/// −(L + η·V), over its peak.
fn wu_window_glint(n: vec3<f32>, to_view: vec3<f32>, to_sun: vec3<f32>, alpha: f32) -> f32 {
    var h = -(to_sun + WU_ETA * to_view);
    h = select(h, -h, h.y < 0.0);
    h = normalize(h + vec3<f32>(0.0, 1e-6, 0.0));
    return wu_ggx_peak1(max(dot(n, h), 0.0), alpha);
}

/// The point the internal mirror reads: the view ray `rd` reflected off the flat underside, from
/// the lookup point slid by the resolved ripple (`n_lo`'s tilt times `slide` yards, the mirrors'
/// rule), carried down to a bed `bed_depth` yards under the surface.
fn wu_tir_point(
    world_pos: vec3<f32>,
    rd: vec3<f32>,
    n_lo: vec3<f32>,
    slide: f32,
    bed_depth: f32,
) -> vec3<f32> {
    let r = vec3<f32>(rd.x, -abs(rd.y), rd.z);
    let p = world_pos + vec3<f32>(n_lo.x, 0.0, n_lo.z) * slide;
    return p + r * (bed_depth / max(-r.y, 0.08));
}

/// The underside's radiance, linear and opaque.
struct WuUnderside {
    l: vec3<f32>,
    // The window's share at this fragment (green's transmittance).
    win: f32,
};

/// The surface seen from below: inside Snell's window the dry sky and the sun through the ripples,
/// past the critical angle the internal mirror (`tir_base` the in-scatter along the reflected ray,
/// `tir_bed` the mirrored bed by its weight `tir_bed_w`, both linear), under the exposure ceiling.
/// `n` the rippled normal, `to_view` toward the eye, `alpha` the unresolved roughness; the stops
/// and the sun colour in gamma, `to_sun` the visible sun and `sun_reach` how much of it gets
/// through.
fn wu_underside(
    n: vec3<f32>,
    to_view: vec3<f32>,
    alpha: f32,
    zenith_g: vec3<f32>,
    horizon_g: vec3<f32>,
    land_g: vec3<f32>,
    sun_g: vec3<f32>,
    to_sun: vec3<f32>,
    sun_reach: f32,
    tir_base: vec3<f32>,
    tir_bed: vec3<f32>,
    tir_bed_w: f32,
) -> WuUnderside {
    let rd = -to_view;
    // Round 2: the window reads the ripple's tilt held to 0.12, so steep facets far outside the
    // critical cone do not throw detached blobs of sky, and a softer edge than a hard cut.
    let tilt = vec2<f32>(n.x, n.z) / max(n.y, 0.2);
    let tl = length(tilt);
    let t_k = select(1.0, 0.07 / tl, tl > 0.07);
    let n_w = normalize(vec3<f32>(tilt.x * t_k, 1.0, tilt.y * t_k));
    let sn = wu_snell(rd, n_w, 0.08 + 2.0 * alpha * alpha);
    let sun_l = wu_dec(sun_g) * sun_reach;
    var win = wu_window_sky(sn.a, zenith_g, horizon_g, land_g);
    let sd = max(dot(sn.a, to_sun), 0.0);
    let disc = WU_SUN_DISC * smoothstep(0.9990, 0.9996, sd);
    win = win + sun_l * (WU_SUN_HALO * pow(sd, 24.0) + disc);
    // Radiance gains n² entering the denser medium.
    win = win * (WU_ETA * WU_ETA * WU_WINDOW_GAIN);
    let glint = wu_window_glint(n, to_view, to_sun, max(alpha, WU_GLINT_ALPHA_MIN))
        * WU_WINDOW_GLINT;
    let tir = mix(tir_base, tir_bed, tir_bed_w);
    var l = win * sn.t + tir * (vec3<f32>(1.0) - sn.t) + sun_l * (glint * sn.t.g);
    l = wu_ceiling(l, 0.80, 1.0, 0.04);
    return WuUnderside(l, sn.t.g);
}

// ---- under water: absorption, in-scatter, shafts, caustics, waterline (UW-2/3) ------------------

/// The view ray through screen `uv` (0..1, y down) of the lanes' camera.
fn wu_post_ray(uv: vec2<f32>, u: UnderLanes) -> vec3<f32> {
    let s = uv * 2.0 - vec2<f32>(1.0);
    return normalize(u.cam_fwd.xyz + s.x * u.cam_right.xyz - s.y * u.cam_up.xyz);
}

/// The GL linear fog's surviving share at eye Z `ez` between `start` and `end` (1.12's law).
fn wu_fog_share(ez: f32, start: f32, end: f32) -> f32 {
    return clamp((end - ez) / max(end - start, 0.001), 0.0, 1.0);
}

/// A colour `c_g` the world fogged at share `f` toward `fog_g` (gamma), the fog taken back off.
/// Where little of it survived the un-fog would amplify noise, so it fades to the colour as it came
/// between the shares 0.35 and 0.2, and stays within 0.25 of the colour or the fog.
fn wu_unfog(c_g: vec3<f32>, f: f32, fog_g: vec3<f32>) -> vec3<f32> {
    let clear = clamp(
        (c_g - (1.0 - f) * fog_g) / max(f, 0.2),
        vec3<f32>(0.0),
        max(c_g, fog_g) + vec3<f32>(0.25),
    );
    return mix(c_g, clear, smoothstep(0.2, 0.35, f));
}

/// The in-scatter palette's mid, linear: the swapped fog colour (gamma) under a luminance ceiling,
/// its weakest channel held at a share of its strongest, so the zone's hue stays and no channel
/// clips or dies.
fn wu_palette(fog_g: vec3<f32>) -> vec3<f32> {
    let l = wu_dec(fog_g);
    let y = dot(l, WU_LUMA);
    let c = l * min(1.0, WU_INSCATTER_LUMA_MAX / max(y, 1e-4));
    let m = max(c.r, max(c.g, c.b));
    return max(c, vec3<f32>(m * WU_INSCATTER_SAT_FLOOR));
}

/// The light the water scatters toward the eye along `rd`, linear: the palette's deep end looking
/// down, its mid level (the authored fog colour), its shallow end looking up, a forward glow toward
/// the refracted sun `sun_up` of irradiance `sun_l`, darkening `depth` yards under the surface.
fn wu_inscatter(
    rd: vec3<f32>,
    mid: vec3<f32>,
    sun_up: vec3<f32>,
    sun_l: vec3<f32>,
    depth: f32,
) -> vec3<f32> {
    let x = clamp(0.5 + 0.6 * rd.y, 0.0, 1.0);
    let deep = mid * WU_DEEP_SHARE;
    let shallow = mid * WU_SHALLOW_GAIN + vec3<f32>(WU_SHALLOW_LIFT);
    var c = select(mix(mid, shallow, 2.0 * x - 1.0), mix(deep, mid, 2.0 * x), x < 0.5);
    let mu = max(dot(rd, sun_up), 0.0);
    c = c + shallow * (0.10 * pow(mu, 6.0)) + sun_l * (0.012 * pow(mu, 40.0));
    return c * exp(-WU_DEPTH_DIM * max(depth, 0.0));
}

/// The light-curtain field at the surface point `q`: three plane waves on the 64-yard lattice and
/// their first harmonics, in about [−1, 1]; `phase` is the underwater clock in radians of its
/// period.
fn wu_shaft_field(q: vec2<f32>, phase: f32) -> f32 {
    let a0 = dot(WU_SHAFT_K0, q) + WU_SHAFT_M0 * phase;
    let a1 = dot(WU_SHAFT_K1, q) + WU_SHAFT_M1 * phase;
    let a2 = dot(WU_SHAFT_K2, q) + WU_SHAFT_M2 * phase;
    return WU_SHAFT_A * (cos(a0) + cos(a1) + cos(a2))
        + WU_SHAFT_H * (cos(2.0 * a0) + cos(2.0 * a1) + cos(2.0 * a2));
}

/// ∫ e^{−bt}·cos(α + ωt) dt over [t0, t1], in closed form; `e0`, `e1` are e^{−b·t0} and e^{−b·t1}.
fn wu_sheet(alpha: f32, omega: f32, b: f32, t0: f32, t1: f32, e0: f32, e1: f32) -> f32 {
    let nx = e1 * cos(omega * t1) - e0 * cos(omega * t0);
    let ny = e1 * sin(omega * t1) - e0 * sin(omega * t0);
    // (nx + i·ny) / (−b + iω)
    let den = max(b * b + omega * omega, 1e-8);
    let qx = (ny * omega - nx * b) / den;
    let qy = (-ny * b - nx * omega) / den;
    return qx * cos(alpha) - qy * sin(alpha);
}

/// The shafts along a view segment [`t0`, `t1`] of the ray from `eye` along `rd`, as a modulation
/// of the in-scatter: the curtain field is constant along the refracted sun `s` (up), so each point
/// takes the field where its light entered, and the integral over the segment, weighted by the
/// in-scatter's extinction `beta`, is closed-form per wave. Over the segment's mean, so 0 is no
/// shaft; strongest looking along the sun, where the field holds still along the ray.
fn wu_shafts(
    eye: vec3<f32>,
    rd: vec3<f32>,
    t0: f32,
    t1: f32,
    beta: f32,
    surf_y: f32,
    s: vec3<f32>,
    phase: f32,
) -> f32 {
    if (t1 <= t0) {
        return 0.0;
    }
    let sy = max(s.y, 0.2);
    // Where the light reaching eye + rd·t entered: q0 + t·qd.
    let q0 = eye.xz + s.xz * ((surf_y - eye.y) / sy);
    let qd = rd.xz - s.xz * (rd.y / sy);
    let b = max(beta, 1e-3);
    let e0 = exp(-b * t0);
    let e1 = exp(-b * t1);
    let mean = (e0 - e1) / b;
    // Each wave's phase where the segment's light entered, and its rate along the ray.
    let a0 = dot(WU_SHAFT_K0, q0) + WU_SHAFT_M0 * phase;
    let a1 = dot(WU_SHAFT_K1, q0) + WU_SHAFT_M1 * phase;
    let a2 = dot(WU_SHAFT_K2, q0) + WU_SHAFT_M2 * phase;
    let w0 = dot(WU_SHAFT_K0, qd);
    let w1 = dot(WU_SHAFT_K1, qd);
    let w2 = dot(WU_SHAFT_K2, qd);
    var acc = WU_SHAFT_A * wu_sheet(a0, w0, b, t0, t1, e0, e1);
    acc = acc + WU_SHAFT_A * wu_sheet(a1, w1, b, t0, t1, e0, e1);
    acc = acc + WU_SHAFT_A * wu_sheet(a2, w2, b, t0, t1, e0, e1);
    acc = acc + WU_SHAFT_H * wu_sheet(2.0 * a0, 2.0 * w0, b, t0, t1, e0, e1);
    acc = acc + WU_SHAFT_H * wu_sheet(2.0 * a1, 2.0 * w1, b, t0, t1, e0, e1);
    acc = acc + WU_SHAFT_H * wu_sheet(2.0 * a2, 2.0 * w2, b, t0, t1, e0, e1);
    return acc / max(mean, 1e-5);
}

/// One light column of the cell `cell` in sun space (round 2, the underwater sketch's): its
/// centre swaying in the cell, its radius, and its strength (0 for a cell without one, flickering
/// otherwise), from the clock `t` in seconds.
fn wu_col(cell: vec2<f32>, t: f32, seed: f32, prob: f32, r0: f32, r1: f32) -> vec4<f32> {
    let h = wu_hash22(cell + vec2<f32>(seed));
    let c = cell + vec2<f32>(0.5) + 0.36 * sin(t * vec2<f32>(0.31, 0.27) * (vec2<f32>(0.6) + h) + WU_TAU * h);
    let w = wu_hash21(cell + vec2<f32>(seed + 13.7));
    let st = step(1.0 - prob, w) * (0.6 + 0.4 * sin(t * (1.3 + w) + w * 20.0));
    return vec4<f32>(c, r0 + r1 * w, st);
}

fn wu_erf(x: f32) -> f32 {
    return tanh(clamp(1.2025 * x, -6.0, 6.0));
}

/// The light columns' in-scatter along the first `l` yards of the ray from `eye` along `rd`
/// (round 2): sparse thin columns parallel to the refracted sun `s` (up), each a point of sun
/// space (where its light entered the surface at `surf_y`, times `sc` cells a yard) with a
/// Gaussian cross-section; the walk takes the ray's projection one cell at a time, gives each
/// column to one step, and integrates its Gaussian along the ray in closed form, so nothing
/// averages out and nothing is marched. Weighted by the extinction `beta` and the depth.
fn wu_shaft_columns(
    eye: vec3<f32>,
    rd: vec3<f32>,
    l: f32,
    t: f32,
    surf_y: f32,
    s: vec3<f32>,
    beta: f32,
) -> f32 {
    let sc = 0.75;
    let sy = max(s.y, 0.2);
    let q0 = (eye.xz + s.xz * ((surf_y - eye.y) / sy)) * sc;
    let qd = (rd.xz - s.xz * (rd.y / sy)) * sc;
    let qn = max(length(qd), 1e-4);
    let dn = qd / qn;
    let pn = vec2<f32>(-dn.y, dn.x);
    let ls = l * qn;
    let kk = min(ceil(ls), 10.0);
    var acc = 0.0;
    for (var k = 0; k < 10; k = k + 1) {
        let fk = f32(k);
        if (fk >= kk) {
            break;
        }
        let base = floor(q0 + dn * (fk + 0.5));
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let col = wu_col(base + vec2<f32>(f32(x), f32(y)), t, 0.0, 0.5, 0.05, 0.08);
                if (col.w <= 0.0) {
                    continue;
                }
                let al = dot(col.xy - q0, dn);
                if (clamp(floor(al), 0.0, kk - 1.0) != fk) {
                    continue;
                }
                let r = col.z;
                let pe = dot(col.xy - q0, pn);
                if (abs(pe) > 3.5 * r) {
                    continue;
                }
                let uc = clamp(al / qn, 0.0, l);
                let ii = exp(-pe * pe / (r * r)) * (r / qn) * 0.8862
                    * (wu_erf((ls - al) / r) - wu_erf(-al / r));
                let depth = max(surf_y - (eye.y + rd.y * uc), 0.0);
                // Past the walk's ten cells the columns fade out instead of stopping on a line.
                let tail = select(1.0, 1.0 - smoothstep(6.0, 10.0, al), ls > 10.0);
                let w = exp(-0.09 * depth / sy - max(beta, 0.02) * uc - 0.12 * depth)
                    * smoothstep(0.8, 3.5, uc) * (1.0 - smoothstep(12.0, 24.0, uc)) * tail;
                acc = acc + col.w * ii * w;
            }
        }
    }
    return acc;
}

/// The caustic web on a surface under `depth` yards of water whose sunlight entered at `q` (world
/// XZ): two layers of cell borders whose points circle at whole turns of the clock, sharpest in the
/// first yards, blue focusing a little deeper; over its mean, so 1 is plain light. Faded to 1 where
/// the footprint `fp` (yards) cannot hold a cell.
fn wu_caustic(q: vec2<f32>, depth: f32, fp: f32, phase: f32) -> vec3<f32> {
    let sharp = mix(7.0, 3.0, smoothstep(0.3, 3.0, depth));
    let rq = vec2<f32>(0.4536 * q.x - 0.8912 * q.y, 0.8912 * q.x + 0.4536 * q.y);
    let w1 = wu_worley(q * WU_CAUSTIC_F1, phase * 48.0);
    let w2 = wu_worley(rq * WU_CAUSTIC_F2 + vec2<f32>(3.1), -phase * 37.0);
    let sh = vec3<f32>(0.96, 1.0, 1.05) * sharp;
    let c1 = exp(-(w1.y - w1.x) * sh);
    let c2 = exp(-(w2.y - w2.x) * sh);
    let c = (c1 * c2 * 2.2 + (c1 + c2) * 0.28) / WU_CAUSTIC_MEAN;
    let keep = 1.0 - smoothstep(0.12, 0.35, fp * WU_CAUSTIC_F2);
    return mix(vec3<f32>(1.0), vec3<f32>(0.4) + 0.6 * c, keep);
}

/// The water's height over the plane where the near plane meets it, a few centimetres of slosh.
fn wu_waterline(p: vec2<f32>, phase: f32) -> f32 {
    return 0.035 * sin(p.x * 2.1 + phase * 61.0) + 0.025 * sin(p.y * 3.3 - phase * 87.0)
        + 0.015 * sin(dot(p, vec2<f32>(1.7, -2.3)) + phase * 143.0);
}

/// The dry sky along `rd` for the waterline's dry half, gamma (round 2): the frame holds only the
/// swapped fog there. The dry fog's colour at the horizon rising to the dome's zenith stop, the
/// dry sun's glow round the sun (unrefracted from the lanes), and the zone glow's square-law lift
/// the dry frame's sky gets, so the sky does not drop to a flat grey-blue between two frames.
fn wu_dry_sky(rd: vec3<f32>, u: UnderLanes) -> vec3<f32> {
    let sxz = u.sun_refr.xz * WU_ETA;
    let sun = normalize(vec3<f32>(sxz.x, sqrt(max(1.0 - dot(sxz, sxz), 1e-6)), sxz.y));
    let up = max(rd.y, 0.0);
    var c = mix(u.fog_dry.rgb, u.sky_zenith.rgb, smoothstep(0.03, 0.75, sqrt(up)));
    c = mix(c, u.sky_horizon.rgb, 0.25 * (1.0 - smoothstep(0.0, 0.3, up)));
    let mu = max(dot(rd, sun), 0.0);
    let day = smoothstep(0.0, 0.1, u.sun_e.w);
    c = c + u.sun_dry.rgb * ((0.5 * pow(mu, 32.0) + 0.22 * pow(mu, 4.0)) * day);
    return min(c + 0.6 * c * c, vec3<f32>(1.0));
}

/// The frame under water, per pixel, gamma in and out: `sg` the scene as drawn (the world fogged by
/// the swapped underwater fog), `rd` the pixel's ray, `view_z` the opaque world's view Z behind it
/// (0 where nothing was drawn), `n_recv` the receiving surface's normal from the depth's
/// derivatives. Below the waterline the swapped fog comes off and the water's own transport goes
/// on: per-channel absorption, in-scatter by direction and depth with the shafts in it, caustics on
/// what lies under the plane, under the exposure ceiling; above it, the dry fog and sky come back;
/// across it, the meniscus. The lanes' `surface.z` 0 returns the scene unchanged.
fn wu_post(
    sg: vec3<f32>,
    rd: vec3<f32>,
    view_z: f32,
    n_recv: vec3<f32>,
    u: UnderLanes,
) -> vec3<f32> {
    let bits = u.surface.w;
    if (u.surface.z < 0.5 || !wu_has(bits, WU_TINT)) {
        return sg;
    }
    let eye = u.cam_pos.xyz;
    let h = u.surface.x;
    let cosf = max(dot(rd, u.cam_fwd.xyz), 1e-3);
    let phase = WU_TAU * u.clock.x / UW_CLOCK_PERIOD;
    let nr = select(-n_recv, n_recv, dot(n_recv, rd) < 0.0);
    // The opaque world behind the pixel, and where the ray crosses the plane ahead of the eye.
    let t_d = select(WU_FAR, view_z / cosf, view_z > 0.0);
    let t_raw = (h - eye.y) / select(rd.y, 1e-6, abs(rd.y) < 1e-6);
    let t_c = select(WU_FAR, t_raw, t_raw > 0.0);
    // The near plane's point on this ray, against the sloshing waterline.
    let t_near = u.cam_pos.w / cosf;
    let p0 = eye + rd * t_near;
    let line = h + wu_waterline(p0.xz, phase);
    let wl = wu_has(bits, WU_WATERLINE);
    let starts_dry = wl && p0.y > line;
    // What the pixel shows is the nearer of the two; the world fogged it at its eye Z.
    let t_vis = min(t_d, t_c);
    let ez = min(t_vis, WU_FAR) * cosf;
    let f_wet = wu_fog_share(ez, u.fog_span.x, u.fog_span.y);
    var c_g = sg;
    if (starts_dry) {
        if (t_vis >= WU_FAR && rd.y > 0.0) {
            c_g = wu_dry_sky(rd, u);
        } else {
            // Round 2: where the swapped fog took an object's colour the un-fog cannot bring it
            // back, so it keeps its shading in the dry fog's hue instead of staying cyan.
            let clear = wu_unfog(sg, f_wet, u.fog_wet.rgb);
            let lost = 1.0 - smoothstep(0.2, 0.6, f_wet);
            let y = dot(clear, WU_LUMA);
            let hue = u.fog_dry.rgb / max(dot(u.fog_dry.rgb, WU_LUMA), 1e-3);
            let est = mix(clear, min(y * hue, vec3<f32>(1.0)), lost);
            c_g = mix(u.fog_dry.rgb, est, wu_fog_share(ez, u.fog_span.z, u.fog_span.w));
        }
    } else {
        // The water segment: from the eye to what the pixel shows, or to the surface's underside on
        // a ray leaving upward; for an eye just above the plane, from where a downward ray enters
        // it, and none on a ray going up.
        let above = eye.y > h;
        let down = rd.y < 0.0;
        let t0 = select(0.0, select(0.0, t_c, down), above);
        let t1 = select(min(t_d, t_c), select(0.0, t_d, down), above);
        let open_water = t1 >= 0.5 * WU_FAR;
        let wet_l = select(max(t1 - t0, 0.0), WU_FAR, open_water);
        var lin = vec3<f32>(0.0);
        if (!open_water) {
            lin = wu_dec(wu_unfog(sg, f_wet, u.fog_wet.rgb));
            // The segment ends on the opaque world: caustics where it lies under the plane.
            if (wu_has(bits, WU_CAUSTICS) && t_d < 0.5 * WU_FAR && t_d <= t1 + 1e-3) {
                let p = eye + rd * t_d;
                let depth = h - p.y;
                if (depth > 0.05) {
                    let s = u.sun_refr.xyz;
                    let q = p.xz + s.xz * (depth / max(s.y, 0.2));
                    let cau = wu_caustic(q, depth, t_d * u.cam_fwd.w, phase)
                        * (1.0 + 0.35 * wu_shaft_field(q, phase));
                    let facing = smoothstep(0.05, 0.5, dot(nr, s));
                    let k = WU_CAUSTIC_GAIN * u.sun_e.w * facing * smoothstep(0.1, 0.6, depth)
                        * exp(-depth / WU_CAUSTIC_DEPTH);
                    lin = lin * (vec3<f32>(1.0) + k * (cau - vec3<f32>(1.0)));
                }
            }
        }
        let tr = exp(-(u.sigma.rgb + vec3<f32>(u.sigma.w)) * min(wet_l, 4096.0));
        let w_in = 1.0 - exp(-u.sigma.w * min(wet_l, 4096.0));
        let t_mid = 0.5 * (t0 + min(t1, 60.0));
        let depth_mid = h - (eye.y + rd.y * t_mid);
        let mid = wu_palette(u.fog_wet.rgb);
        let ins = wu_inscatter(rd, mid, u.sun_refr.xyz, u.sun_e.rgb, depth_mid);
        lin = lin * tr + ins * w_in;
        if (wu_has(bits, WU_SHAFTS) && u.sun_e.w > 0.01) {
            // Round 2: the light columns, added as the sun's light they scatter, strongest
            // looking up the sun.
            let l_seg = min(max(t1 - t0, 0.0), 24.0);
            let acc = wu_shaft_columns(eye + rd * t0, rd, l_seg, u.clock.x, h, u.sun_refr.xyz, u.sigma.w);
            let ph = 0.3 + 0.7 * pow(saturate(dot(rd, u.sun_refr.xyz) * 0.5 + 0.5), 3.0);
            let tint = mix(vec3<f32>(0.6, 0.95, 1.0), vec3<f32>(1.0), 0.4);
            lin = lin + u.sun_e.rgb * tint
                * ((1.0 - exp(-1.3 * acc)) * ph * WU_SHAFT_GAIN * smoothstep(0.02, 0.2, u.sun_e.w));
        }
        c_g = wu_enc(wu_ceiling(lin, 0.55, 0.85, 0.04));
    }
    if (wl) {
        // The meniscus where the near plane cuts the water: a dark lens of water, a bright lip.
        let w = max(t_near * u.cam_fwd.w * WU_MENISCUS_PX, 1e-4);
        let off = p0.y - line;
        let m = 1.0 - smoothstep(0.0, w, abs(off));
        let lip = 1.0 - smoothstep(0.0, 0.35 * w, abs(off - 0.6 * w));
        c_g = mix(c_g, c_g * 0.55 + wu_enc(wu_palette(u.fog_wet.rgb)) * 0.2, m * 0.8)
            + vec3<f32>(lip * 0.22);
    }
    return mix(vec3<f32>(WU_OUT_FLOOR), vec3<f32>(WU_OUT_TOP), saturate(c_g));
}
// ---- wu law: end ------------------------------------------------------------------------------

@group(0) @binding(6) var<uniform> uw: UnderLanes;
// The opaque world's linear depth (view Z, yards; 0 where nothing was drawn), the water's own.
@group(0) @binding(7) var uw_depth: texture_2d<f32>;

/// The stock warp's reach kept, and the haze's: the water's own absorption replaces most of the
/// haze's blur.
const UW_WARP_KEEP: f32 = 0.7;
const UW_HAZE_KEEP: f32 = 0.35;

/// The improved underwater combine: the warp, the water's own transport on the warped scene
/// ([`wu_post`]), then the haze at `UW_HAZE_KEEP` of its mix and the zone glow.
@fragment
fn fs_combine_wave_improved(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let screen = vec2<f32>(textureDimensions(in_tex));
    let uv = in.uv + wave_offset(in.uv, screen) * UW_WARP_KEEP;
    let dd = vec2<f32>(textureDimensions(uw_depth));
    let z = textureLoad(uw_depth, vec2<i32>(clamp(uv * dd, vec2<f32>(0.0), dd - vec2<f32>(1.0))), 0).r;
    let rd = wu_post_ray(uv, uw);
    let t = select(1000.0, z / max(dot(rd, uw.cam_fwd.xyz), 1e-3), z > 0.0);
    let p = uw.cam_pos.xyz + rd * t;
    // The receiving surface's facing from the depth's own derivatives, in uniform control flow.
    let n_recv = normalize(cross(dpdx(p), dpdy(p)) + vec3<f32>(0.0, 1e-7, 0.0));
    let scene = textureSample(in_tex, in_samp, uv);
    let sg = wu_post(clamp(scene.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), rd, z, n_recv, uw);
    let bg = max(textureSample(blur_tex, in_samp, uv).rgb, vec3<f32>(0.0));
    let hazed = mix(sg, min(bg, vec3<f32>(1.0)), ffx.lane.z * UW_HAZE_KEEP);
    let outg = min(hazed + ffx.lane.x * bg * bg, vec3<f32>(1.0));
    return combine_out(outg, scene.a, in.position.xy);
}
#endif
