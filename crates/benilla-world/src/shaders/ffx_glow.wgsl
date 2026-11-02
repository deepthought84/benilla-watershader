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
// Round 3: tighter and dimmer (the round 2 halo filled the gaps of a broken glitter path into one
// bar): the ring at one and a half texels, 60/40 with the centre, at 0.75 of its strength.
fn water_glow_wide(uv: vec2<f32>) -> vec4<f32> {
    let c = textureSample(blur_tex, in_samp, uv);
    let t = 1.5 / vec2<f32>(textureDimensions(blur_tex));
    var ring = 0.0;
    for (var k = 0; k < 8; k = k + 1) {
        let a = f32(k) * 0.7853982 + 0.3927;
        ring = ring + textureSample(blur_tex, in_samp, uv + vec2<f32>(cos(a), sin(a)) * t).a;
    }
    return vec4<f32>(c.rgb, (0.6 * c.a + 0.4 * ring / 8.0) * 0.75);
}

// The blurred excess `b.a` over the combine's gamma output `outg`, as light: weight `wave.z`, capped
// at `wave.w`, in the hue of the blurred neighbourhood, added under a square-law decode so a dark
// night bay blooms more than a bright noon one.
fn water_glow_over(outg: vec3<f32>, b: vec4<f32>) -> vec3<f32> {
    let e = min(max(b.a, 0.0), ffx.wave.w) * ffx.wave.z;
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
    return combine_body_on(uv, frag_coord, clamp(scene.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), scene.a);
}

// [`combine_body`] on the scene colour `sg` (gamma, clamped) and alpha `scene_a`.
fn combine_body_on(uv: vec2<f32>, frag_coord: vec2<f32>, sg: vec3<f32>, scene_a: f32) -> vec4<f32> {
    let scene = vec4<f32>(sg, scene_a);
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
    // vertical field); w the caustic net's octave drift (tiles), x then y.
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    // x the water surface's world Y over the eye, y the eye's depth under it, z 1 while the eye is
    // in a water kind (0 dry, magma or slime), w the feature bits.
    surface: vec4<f32>,
    // rgb the absorption per yard, w the in-scatter's extinction per yard.
    sigma: vec4<f32>,
    // rgb the swapped (underwater) fog colour, then the dry fog colour, gamma; w the kind's
    // turbidity per yard, then the caustic net's gain (0 with `caustics` off).
    fog_wet: vec4<f32>,
    fog_dry: vec4<f32>,
    // x/y the swapped fog's start and end, z/w the dry fog's, yards.
    fog_span: vec4<f32>,
    // rgb the dry dome's zenith and horizon stops, gamma; w the sun's share kept in shadow
    // (`look.body.e_sky.w`), then the open sky's luminance on the body.
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    // rgb the dry sun's colour (Light.dbc row 9), gamma; w the lighting sun's luminance on the body.
    sun_dry: vec4<f32>,
    // xyz the lighting sun refracted into the water, unit, pointing up toward it; w the caustic
    // net's clock phase (`preview.clocks.y`).
    sun_refr: vec4<f32>,
    // rgb the lighting sun's irradiance, linear; w its share of the light under the surface (0 at
    // night or under cloud).
    sun_e: vec4<f32>,
    // x the underwater clock in seconds reduced by `UW_CLOCK_PERIOD`, y the quality tier, z the
    // veil's mist and w its falls on (above the water).
    clock: vec4<f32>,
    // The terrain shadow's window (`look.body.shadow`): centre XZ, half extent, valid.
    shadow: vec4<f32>,
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
/// Fresh water's index, and per channel at about 650, 550 and 450 nm (blue a little high, so the
/// window's rim shows its fringe).
const WU_ETA: f32 = 1.333;
const WU_ETA_RGB: vec3<f32> = vec3<f32>(1.3315, 1.333, 1.3355);
/// Water's reflectance head-on.
const WU_F0: f32 = 0.02;
const WU_LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);
/// Distance standing in for "nothing along the ray", yards.
const WU_FAR: f32 = 100000.0;
/// The underwater clock's period, seconds; every time term turns a whole number of times in it,
/// so its wrap never shows.
const UW_CLOCK_PERIOD: f32 = 256.0;

/// Snell's window: its gain over the n² radiance the air's light gains entering the water, the
/// land haze's share at its rim, the sun's halo and disc, and the transmitted glint's weight.
const WU_WINDOW_GAIN: f32 = 1.25;
const WU_RIM_LAND: f32 = 0.12;
const WU_SUN_HALO: f32 = 0.6;
const WU_SUN_DISC: f32 = 6.0;
const WU_WINDOW_GLINT: f32 = 6.0;
const WU_GLINT_ALPHA_MIN: f32 = 0.025;
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
/// The shafts' strength.
const WU_SHAFT_GAIN: f32 = 1.6;
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
const WU_MENISCUS_PX: f32 = 14.0;
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

/// The distance outside the nearest round hole of a jittered lattice, in cell units: positive in the
/// foam, negative in a hole. Each hole is `rad` times its own size in [1 − spread, 1 + spread], its
/// centre jittered `jit` about its cell's. The union of discs, so a hole's rim is an arc and the foam
/// between holes thins to a neck and thickens into a Plateau border where three meet; no `sin` and
/// no sort, so it costs less than [`wu_worley`].
fn wu_holes(p: vec2<f32>, rad: f32, jit: f32, spread: f32) -> f32 {
    let i = floor(p);
    let f = p - i;
    var s = 8.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let c = vec2<f32>(f32(x), f32(y));
            let h = wu_hash22(i + c);
            let o = vec2<f32>(0.5) + jit * (2.0 * h - vec2<f32>(1.0));
            let size = 1.0 + spread * (2.0 * fract(h.x * 7.13 + h.y * 3.71) - 1.0);
            s = min(s, length(c + o - f) - rad * size);
        }
    }
    return s;
}

/// Bubbles at two sizes (big cells 0.85 and small 2.2 a unit of `p`, warped): the distance to the
/// nearest hole's rim in units of `p`, the holes `ra` and `rb` in their cells' units.
fn wu_bubbles(p: vec2<f32>, ra: f32, rb: f32) -> f32 {
    let sa = wu_holes(p * 0.85, ra, 0.36, 0.2) / 0.85;
    let sb = wu_holes(p * 2.2 + vec2<f32>(7.0), rb, 0.42, 0.4) / 2.2;
    return min(sa, sb);
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
    // Round 3: two-yard cells (sparser than round 2's 1.3 yd), so the walk's cells span the whole
    // 24 yd segment and no column stops on the walk's end; thinner columns.
    let sc = 0.6;
    let sy = max(s.y, 0.2);
    let q0 = (eye.xz + s.xz * ((surf_y - eye.y) / sy)) * sc;
    let qd = (rd.xz - s.xz * (rd.y / sy)) * sc;
    let qn = max(length(qd), 1e-4);
    let dn = qd / qn;
    let pn = vec2<f32>(-dn.y, dn.x);
    let ls = l * qn;
    let kk = min(ceil(ls), 14.0);
    var acc = 0.0;
    for (var k = 0; k < 14; k = k + 1) {
        let fk = f32(k);
        if (fk >= kk) {
            break;
        }
        let base = floor(q0 + dn * (fk + 0.5));
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let col = wu_col(base + vec2<f32>(f32(x), f32(y)), t, 0.0, 0.5, 0.03, 0.05);
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
                // Past the walk's cells the columns fade out instead of stopping on a line; each
                // fades with depth under the surface, so it hangs from the light and not the bed.
                let tail = select(1.0, 1.0 - smoothstep(8.0, 14.0, al), ls > 14.0);
                let w = exp(-0.09 * depth / sy - max(beta, 0.02) * uc - 0.15 * depth)
                    * smoothstep(0.8, 3.5, uc) * (1.0 - smoothstep(10.0, 24.0, uc)) * tail;
                acc = acc + col.w * ii * w;
            }
        }
    }
    return acc;
}

// ---- one caustic net, seen from above and from below (BODY-5, UW-4) ------------------------------
//
// The bed is lit through the surface's lenses: a ray of the lighting sun entering at `x` lands at
// `x + D·κ·∇h(x)` on a bed `D` deep along the refracted ray, κ = 1 − 1/n, so the bed's irradiance is
// the sun's over |det(I + D·κ·H)|, H the surface's Hessian. The lenses are two sets of six ripples,
// 1.0-0.3 yd and 3.6-1.6 yd, laid on the visible surface's 3.6-yd ripple octave (`water_surface`'s
// fourth, its lattice turn and tile), each a whole number of cycles a tile on spread headings, so
// the net rides that octave's drift and the current with it, and the octave's wrap moves it by
// whole periods: the net moves with the water the eye sees. Each ripple also runs at a whole number
// of cycles in the preview's weather clock. The fine set's curvatures are scaled so its net is
// sharpest at `WU_CAU_FOCUS_YD` (a soft mottle in the first hand's breadth, bright lines at a
// yard), the coarse set's `WU_CAU_COARSE` times deeper, and the fine net hands over to the coarse
// one as the bed falls away; neither is drawn past its sharpest, where the net would break into
// speckle. The intensity is regularised (`WU_CAU_EPS`) and divided by its mean at that depth
// (`wu_caustic_mean`), so the bed keeps its energy and only the sun's direct share is
// redistributed. The water above (`nwg_caustic_gain`) and the improved underwater combine
// (`wu_post`) both light the bed with it; only what the eye looks through differs.

/// The octave the net rides: its lattice turn and its tile, yards (`water_surface`'s `rot3`, `f3`).
const WU_CAU_ROT: vec2<f32> = vec2<f32>(-0.2924, 0.9563);
const WU_CAU_TILE_YD: f32 = 3.6;
/// The path along the refracted sun, yards, at which a curvature of one focuses: the net is
/// sharpest at about 1.25 of it.
const WU_CAU_FOCUS_YD: f32 = 0.72;
/// The intensity's regularisation, 1/sqrt(det² + ε²): the brightest line is 1/ε over the mean.
const WU_CAU_EPS: f32 = 0.15;
/// The path over which the net's contrast fades out, yards.
const WU_CAU_FADE_YD: vec2<f32> = vec2<f32>(6.0, 20.0);
/// The deepest focus ratio either set's net is drawn at: past it the net would break into speckle.
const WU_CAU_RHO_MAX: f32 = 1.6;
/// Each ripple's curvature per unit focus.
const WU_CAU_CURV: f32 = 0.57735;
/// How much deeper the coarse set focuses than the fine one (its ripples' length over theirs).
const WU_CAU_COARSE: f32 = 3.3;
/// A ripple's wavelength in output pixels between which it fades out of the net: the net's lines
/// are a tenth of the ripple wide, so a ripple under 8 px would alias to speckle, and one at 16 px
/// draws whole. Each ripple fades by its own wavelength, the fine set's first, and the net's
/// focus follows what is left ([`wu_cwave`], [`wu_caustic_lens`]).
const WU_CAU_RIPPLE_PX: vec2<f32> = vec2<f32>(8.0, 16.0);
/// How fast turbidity scatters the net away, per unit of the kind's turbidity per yard.
const WU_CAU_TURB: f32 = 0.6;
/// The eye's adaptation under water: the bed lies behind yards of in-scatter that flatten it, and
/// the eye adapted to the dim body reads its light's contrast this much stronger. Only the view
/// from below takes it.
const WU_CAU_UNDER_GAIN: f32 = 2.5;
/// The flow map's restart period, seconds (`FLOW_PERIOD_S`).
const WU_CAU_FLOW_PERIOD_S: f32 = 1.6;

/// World XZ `x` on the net's octave lattice at its drift `drift` (tiles).
fn wu_caustic_uv(x: vec2<f32>, drift: vec2<f32>) -> vec2<f32> {
    let r = WU_CAU_ROT;
    return vec2<f32>(r.x * x.x + r.y * x.y, -r.y * x.x + r.x * x.y) / WU_CAU_TILE_YD + drift;
}

/// One ripple's share of the Hessian, `(h_xx, h_zz, h_xz)` per unit focus, and in `w` the square
/// of what the footprint `fp` (yards an output pixel on the bed) keeps of it: `mn` its cycles a
/// tile on the lattice, `m` its cycles in the clock's period, `p0` its phase; its curvature
/// `WU_CAU_CURV` along its world heading, faded as its wavelength nears the pixel
/// ([`WU_CAU_RIPPLE_PX`]).
fn wu_cwave(uv: vec2<f32>, phi: f32, mn: vec2<f32>, m: f32, p0: f32, fp: f32) -> vec4<f32> {
    let r = WU_CAU_ROT;
    let d = normalize(vec2<f32>(r.x * mn.x - r.y * mn.y, r.y * mn.x + r.x * mn.y));
    let lambda = WU_CAU_TILE_YD / length(mn);
    let k = smoothstep(WU_CAU_RIPPLE_PX.x, WU_CAU_RIPPLE_PX.y, lambda / max(fp, 1e-5));
    let w = WU_CAU_CURV * k * vec3<f32>(d.x * d.x, d.y * d.y, d.x * d.y);
    return vec4<f32>(-sin(WU_TAU * fract(dot(uv, mn) - m * phi + p0)) * w, k * k);
}

/// The fine set's Hessian at lattice point `uv`, per unit focus (ripples of 1.0-0.3 yd), and in
/// `w` the mean square of what the footprint `fp` keeps of its ripples; `phi` the weather clock's
/// phase.
fn wu_caustic_hessian(uv: vec2<f32>, phi: f32, fp: f32) -> vec4<f32> {
    var h = wu_cwave(uv, phi, vec2<f32>(2.0, -3.0), 40.0, 0.0000, fp);
    h += wu_cwave(uv, phi, vec2<f32>(1.0, 5.0), 45.0, 0.6180, fp);
    h += wu_cwave(uv, phi, vec2<f32>(-4.0, -4.0), 50.0, 0.2361, fp);
    h += wu_cwave(uv, phi, vec2<f32>(7.0, 0.0), 56.0, 0.8541, fp);
    h += wu_cwave(uv, phi, vec2<f32>(-7.0, 6.0), 63.0, 0.4721, fp);
    h += wu_cwave(uv, phi, vec2<f32>(1.0, -12.0), 71.0, 0.0902, fp);
    return vec4<f32>(h.xyz, h.w / 6.0);
}

/// The coarse set's (ripples of 3.6-1.6 yd, slower, as longer waves run).
fn wu_caustic_hessian_coarse(uv: vec2<f32>, phi: f32, fp: f32) -> vec4<f32> {
    var h = wu_cwave(uv, phi, vec2<f32>(1.0, 0.0), 21.0, 0.3820, fp);
    h += wu_cwave(uv, phi, vec2<f32>(2.0, 1.0), 30.0, 0.7639, fp);
    h += wu_cwave(uv, phi, vec2<f32>(1.0, 2.0), 32.0, 0.1459, fp);
    h += wu_cwave(uv, phi, vec2<f32>(0.0, 1.0), 22.0, 0.5279, fp);
    h += wu_cwave(uv, phi, vec2<f32>(-1.0, 1.0), 25.0, 0.9098, fp);
    h += wu_cwave(uv, phi, vec2<f32>(-2.0, 1.0), 33.0, 0.2918, fp);
    return vec4<f32>(h.xyz, h.w / 6.0);
}

/// The regularised intensity's mean over the pattern at focus ratio `rho` (path over
/// `WU_CAU_FOCUS_YD`): a rational fit, within 11% from 0 to 6.
fn wu_caustic_mean(rho: f32) -> f32 {
    let r = min(rho, 6.0);
    let r2 = r * r;
    let r3 = r2 * r;
    return (0.989 + 2.1614 * r2 - 0.1102 * r3) / (1.0 - 0.1993 * r2 + 0.6322 * r3);
}

/// The caustic of the Hessian `h` (per unit focus, `w` the mean square its ripples keep) at focus
/// ratio `rho`, mean 1: a set whose ripples the footprint has faded is the whole set at a shallower
/// focus, `rho` times the rms it keeps, so its mean is read there; a set faded out is 1 everywhere.
fn wu_caustic_lens(h: vec4<f32>, rho: f32) -> f32 {
    let jxx = 1.0 + rho * h.x;
    let jzz = 1.0 + rho * h.y;
    let jxz = rho * h.z;
    let det = jxx * jzz - jxz * jxz;
    return inverseSqrt(det * det + WU_CAU_EPS * WU_CAU_EPS) / wu_caustic_mean(rho * sqrt(h.w));
}

/// Both sets' net at lattice point `uv` down a path of `path` yards, seen through a footprint of
/// `fp` yards an output pixel on the bed, mean 1: the fine net sharp in the first yard or two,
/// handing over to the coarse net, which focuses `WU_CAU_COARSE` times deeper, as the bed falls
/// away; each ripple fades as the footprint nears it, the fine set's first.
fn wu_caustic_at(uv: vec2<f32>, path: f32, phi: f32, fp: f32) -> f32 {
    let wf = 1.0 - smoothstep(1.0, 3.5, path);
    let wc = smoothstep(0.4, 2.0, path);
    var c = 1.0;
    if (wf > 0.0) {
        let rho = min(path / WU_CAU_FOCUS_YD, WU_CAU_RHO_MAX);
        c += wf * (wu_caustic_lens(wu_caustic_hessian(uv, phi, fp), rho) - 1.0);
    }
    if (wc > 0.0) {
        let rho = min(path / (WU_CAU_FOCUS_YD * WU_CAU_COARSE), WU_CAU_RHO_MAX);
        c += wc * (wu_caustic_lens(wu_caustic_hessian_coarse(uv, phi, fp), rho) - 1.0);
    }
    return c;
}

/// The net, mean 1, on the bed point whose refracted sun ray entered the surface at `entry` (world
/// XZ) and ran `path` yards down to it, seen through `fp` yards an output pixel on the bed: on the
/// octave's lattice at its `drift`, carried along `flow` (world XZ yards a second) by the
/// two-phase flow map at phase `ph0` as the octave is.
fn wu_caustic_net(
    entry: vec2<f32>,
    path: f32,
    phi: f32,
    drift: vec2<f32>,
    flow: vec2<f32>,
    ph0: f32,
    fp: f32,
) -> f32 {
    if (dot(flow, flow) < 1e-4) {
        return wu_caustic_at(wu_caustic_uv(entry, drift), path, phi, fp);
    }
    // Two phases half a period apart, blended so the net's variance holds through the restart.
    let ph1 = fract(ph0 + 0.5);
    let w0 = 1.0 - abs(2.0 * ph0 - 1.0);
    let apart = vec2<f32>(0.37, 0.61) * saturate(length(flow)) * WU_CAU_TILE_YD;
    let a = wu_caustic_at(wu_caustic_uv(entry - flow * (ph0 * WU_CAU_FLOW_PERIOD_S), drift), path, phi, fp)
        - 1.0;
    let b = wu_caustic_at(
        wu_caustic_uv(entry - flow * (ph1 * WU_CAU_FLOW_PERIOD_S) + apart, drift),
        path,
        phi,
        fp,
    ) - 1.0;
    return 1.0 + (a * w0 + b * (1.0 - w0)) * inverseSqrt(w0 * w0 + (1.0 - w0) * (1.0 - w0));
}

/// The net's contrast kept over a path of `path` yards through water of turbidity `sig_turb` per
/// yard; the footprint fades the net's ripples one by one ([`wu_cwave`]).
fn wu_caustic_keep(path: f32, sig_turb: f32) -> f32 {
    return (1.0 - smoothstep(WU_CAU_FADE_YD.x, WU_CAU_FADE_YD.y, path))
        * exp(-sig_turb * path * WU_CAU_TURB);
}

/// The lighting sun's direct share of the bed's light under its shadow `lit` (1 lit): the sun's
/// luminance `sun` over it and the open sky's `sky`, the sun keeping `keep` of itself in shadow.
fn wu_sun_share(lit: f32, sun: f32, sky: f32, keep: f32) -> f32 {
    return saturate(sun * lit / max(sky + sun * mix(keep, 1.0, lit), 1e-4));
}

/// The bed's gain under the net, 1 where there is none: the net at `amp` of its contrast.
fn wu_caustic_gain(c: f32, amp: f32) -> f32 {
    return max(1.0 + amp * (c - 1.0), 0.0);
}

/// The water's height over the plane where the near plane meets it, a few centimetres of slosh.
fn wu_waterline(p: vec2<f32>, phase: f32) -> f32 {
    // Round 3: twice the slosh, so the line is not ruler-straight at 1:1.
    return 0.07 * sin(p.x * 2.1 + phase * 61.0) + 0.05 * sin(p.y * 3.3 - phase * 87.0)
        + 0.03 * sin(dot(p, vec2<f32>(1.7, -2.3)) + phase * 143.0);
}

/// The dry sky along `rd` for the waterline's dry half, gamma (round 2): the frame holds only the
/// swapped fog there. The dry fog's colour at the horizon rising to the dome's zenith stop, the
/// dry sun's glow round the sun (unrefracted from the lanes), and the zone glow's square-law lift
/// the dry frame's sky gets, so the sky does not drop to a flat grey-blue between two frames.
fn wu_dry_sky(rd: vec3<f32>, u: UnderLanes) -> vec3<f32> {
    let sxz = u.sun_refr.xz * WU_ETA;
    let sun = normalize(vec3<f32>(sxz.x, sqrt(max(1.0 - dot(sxz, sxz), 1e-6)), sxz.y));
    let up = max(rd.y, 0.0);
    // Round 3: the horizon stop carries the low sky (a sunset's cream and pink) and the dry fog only
    // hazes the last few degrees; round 2 rose from the fog and read as a blue gradient.
    var c = mix(u.sky_horizon.rgb, u.sky_zenith.rgb, smoothstep(0.3, 1.0, sqrt(up)));
    c = mix(c, u.fog_dry.rgb, 0.35 * (1.0 - smoothstep(0.0, 0.12, up)));
    let mu = max(dot(rd, sun), 0.0);
    let day = smoothstep(0.0, 0.1, u.sun_e.w);
    c = c + u.sun_dry.rgb * ((0.5 * pow(mu, 32.0) + 0.22 * pow(mu, 4.0)) * day);
    return min(c + 0.6 * c * c, vec3<f32>(1.0));
}

/// The frame under water, per pixel, gamma in and out: `sg` the scene as drawn (the world fogged by
/// the swapped underwater fog), `rd` the pixel's ray, `view_z` the opaque world's view Z behind it
/// (0 where nothing was drawn), `n_recv` the receiving surface's normal from the depth's
/// derivatives, `lit` the lighting sun's terrain shadow there (1 lit). Below the waterline the
/// swapped fog comes off and the water's own transport goes on: per-channel absorption, in-scatter
/// by direction and depth with the shafts in it, the caustic net ([`wu_caustic_net`]) on what lies
/// under the plane, under the exposure ceiling; above it, the dry fog and sky come back; across it,
/// the meniscus. The lanes' `surface.z` 0 returns the scene unchanged.
fn wu_post(
    sg: vec3<f32>,
    rd: vec3<f32>,
    view_z: f32,
    n_recv: vec3<f32>,
    lit: f32,
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
            // Round 3: more of it, in a hue half way to grey, so the wreck is not teal.
            let clear = wu_unfog(sg, f_wet, u.fog_wet.rgb);
            let lost = 1.0 - smoothstep(0.45, 0.9, f_wet);
            let y = dot(clear, WU_LUMA);
            let hue = mix(u.fog_dry.rgb / max(dot(u.fog_dry.rgb, WU_LUMA), 1e-3), vec3<f32>(1.0), 0.5);
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
            // The segment ends on the opaque world: the one caustic net where it lies under the
            // plane, its contrast the sun's direct share under the shadow `lit` and its facing.
            if (wu_has(bits, WU_CAUSTICS) && t_d < 0.5 * WU_FAR && t_d <= t1 + 1e-3) {
                let p = eye + rd * t_d;
                let depth = h - p.y;
                let s = u.sun_refr.xyz;
                if (depth > 0.0 && s.y > 0.02) {
                    let path = depth / max(s.y, 0.2);
                    let facing = smoothstep(0.05, 0.5, dot(nr, s));
                    let amp = wu_caustic_keep(path, u.fog_wet.w)
                        * wu_sun_share(lit * facing, u.sun_dry.w, u.sky_horizon.w, u.sky_zenith.w)
                        * u.fog_dry.w * WU_CAU_UNDER_GAIN;
                    if (amp >= 1.0 / 256.0) {
                        let entry = p.xz + s.xz * path;
                        // The footprint on the bed: an output pixel's angle over the distance,
                        // stretched by the ray's grazing on the bed.
                        let fp = t_d * u.cam_fwd.w / max(abs(dot(rd, nr)), 0.1);
                        // No current from below: the eye's body has none on the CPU.
                        let c = wu_caustic_net(entry, path, u.sun_refr.w,
                            vec2<f32>(u.cam_right.w, u.cam_up.w), vec2<f32>(0.0), 0.0, fp);
                        lin = lin * wu_caustic_gain(c, amp);
                    }
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
            let tint = mix(vec3<f32>(0.85, 0.97, 1.0), vec3<f32>(1.0), 0.5);
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
            + vec3<f32>(lip * 0.35);
    }
    return mix(vec3<f32>(WU_OUT_FLOOR), vec3<f32>(WU_OUT_TOP), saturate(c_g));
}
// ---- wu law: end ------------------------------------------------------------------------------

/// Round 3: the veil's rows (`benilla_water::preview_rows`'s `v`), after the under rows.
struct VeilLanes {
    // x armed, y the mirror plane's world Y, z the rain, w the storm.
    a: vec4<f32>,
    // x the dry fog's end (yards), y the mist on, z the falls on, w spare.
    b: vec4<f32>,
    // xyz the visible sun's direction, w how much of it gets through.
    sun: vec4<f32>,
    // rgb the sky's and the sun's light on foam, linear.
    e_sky: vec4<f32>,
    e_sun: vec4<f32>,
    spare0: vec4<f32>,
    spare1: vec4<f32>,
    spare2: vec4<f32>,
    // The four nearest falls: the foot line's ends (XZ), then the pool's surface, the drop, the
    // pool's spread (not the spray's: a curtain on the bank throws as much) and the strength
    // (`preview.falls`).
    falls: array<vec4<f32>, 8>,
};

struct FfxUwBlock {
    u: UnderLanes,
    veil: VeilLanes,
};

@group(0) @binding(6) var<uniform> uwb: FfxUwBlock;
// The opaque world's linear depth (view Z, yards; 0 where nothing was drawn), the water's own.
@group(0) @binding(7) var uw_depth: texture_2d<f32>;
// The lighting sun's terrain shadow torus (`benilla_water::mcsh`, 1 shadowed), inside the window
// `uwb.u.shadow`; the depth stands in, unread, while the window is not valid.
@group(0) @binding(9) var uw_shadow: texture_2d<f32>;

/// The map's north-west corner from its centre and the shadow torus's span, yards
/// (`liquid.wgsl`'s `MAP_HALF_YD`, `MCSH_SPAN_YD`).
const UW_MAP_HALF_YD: f32 = 17066.666;
const UW_MCSH_SPAN_YD: f32 = 1066.6666;

/// Whether the lighting sun reaches the ground at world point `p`, 1 lit and 0 shadowed: the
/// terrain's MCSH there, read bilinearly across the torus, as `liquid.wgsl`'s `lighting_sun_mcsh`
/// reads it for the bed seen from above. Lit outside the window.
fn uw_sun_lit(p: vec3<f32>, win: vec4<f32>) -> f32 {
    if (win.w < 0.5 || any(abs(p.xz - win.xy) >= vec2<f32>(win.z))) {
        return 1.0;
    }
    let dims = vec2<i32>(textureDimensions(uw_shadow));
    let t = fract((p.xz + vec2<f32>(UW_MAP_HALF_YD)) / UW_MCSH_SPAN_YD) * vec2<f32>(dims) - 0.5;
    let i = vec2<i32>(floor(t));
    let f = t - floor(t);
    let s00 = uw_shadow_texel(i, dims);
    let s10 = uw_shadow_texel(i + vec2<i32>(1, 0), dims);
    let s01 = uw_shadow_texel(i + vec2<i32>(0, 1), dims);
    let s11 = uw_shadow_texel(i + vec2<i32>(1, 1), dims);
    return 1.0 - mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

/// One texel of the shadow torus, its index wrapped.
fn uw_shadow_texel(i: vec2<i32>, dims: vec2<i32>) -> f32 {
    return textureLoad(uw_shadow, ((i % dims) + dims) % dims, 0).r;
}

/// The stock warp's reach kept, and the haze's: the water's own absorption replaces most of the
/// haze's blur.
const UW_WARP_KEEP: f32 = 0.7;
const UW_HAZE_KEEP: f32 = 0.35;

/// The improved underwater combine: the warp, the water's own transport on the warped scene
/// ([`wu_post`]), then the haze at `UW_HAZE_KEEP` of its mix and the zone glow.
@fragment
fn fs_combine_wave_improved(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uw = uwb.u;
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
    let lit = uw_sun_lit(p, uw.shadow);
    let sg = wu_post(clamp(scene.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), rd, z, n_recv, lit, uw);
    let bg = max(textureSample(blur_tex, in_samp, uv).rgb, vec3<f32>(0.0));
    let hazed = mix(sg, min(bg, vec3<f32>(1.0)), ffx.lane.z * UW_HAZE_KEEP);
    let outg = min(hazed + ffx.lane.x * bg * bg, vec3<f32>(1.0));
    return combine_out(outg, scene.a, in.position.xy);
}

// ---- the veil (round 3, WX-2 and RIV-4) -------------------------------------------------------
//
// Above the water, over everything the frame shows: the mist layer lying on the mirror plane
// (exponential in height over `VEIL_LAYER_YD`, its density by the storm, the rain, a low sun and a
// near fog end as the water's own `nwg_mist` sets it, closed form along the ray to the depth), and
// each of the four nearest falls' spray, a plume rising from its foot line, billowing upward on the
// clock, marched through the line's own box. The water drew the mist on its own pixels only, so
// posts, hulls and the far shore stood crisp in milky water; the falls' spray could not rise above
// the waterline at all.

const VEIL_LAYER_YD: f32 = 2.5;
const VEIL_SIGMA: f32 = 0.011;
const VEIL_MAX: f32 = 0.75;
/// The spray: its extinction per yard of plume, and how much of the frame it hides at most, in its
/// dense base and over the curtain above it (the curtain's streaks stay crisp through it).
const VEIL_SPRAY_K: f32 = 0.9;
const VEIL_SPRAY_MAX: f32 = 0.66;
const VEIL_SPRAY_TOP: f32 = 0.24;
const VEIL_SPRAY_STEPS: i32 = 10;
/// A plume's spread off its foot line and its rise, yards: by the drop, capped in yards, so a
/// cascade's spray hugs its foot instead of fogging the whole fall.
const VEIL_SPRAY_REACH: vec2<f32> = vec2<f32>(0.8, 3.2);
const VEIL_SPRAY_RISE: vec2<f32> = vec2<f32>(1.0, 5.0);

/// The veil over the scene colour `sg` (gamma) along the ray `rd` from the eye to what the frame
/// shows `t_d` yards out, gamma.
fn veil_apply(sg: vec3<f32>, rd: vec3<f32>, t_d: f32) -> vec3<f32> {
    let u = uwb.u;
    let v = uwb.veil;
    let eye = u.cam_pos.xyz;
    var lin = wu_dec(sg);
    let sun_l = normalize(v.sun.xyz + vec3<f32>(0.0, 1e-5, 0.0));
    let sun_c = wu_dec(u.sun_dry.rgb);
    let forward = pow(saturate(dot(rd, sun_l)), 6.0);
    if (v.b.y > 0.5) {
        let low = smoothstep(-0.12, 0.0, sun_l.y) * (1.0 - smoothstep(0.08, 0.3, sun_l.y));
        let thick = 1.0 - smoothstep(80.0, 250.0, v.b.x);
        let m = max(max(0.6 * v.a.w, 0.6 * v.a.z), max(0.45 * low, 0.7 * thick));
        if (m > 0.0) {
            let e0 = max(eye.y - v.a.y, 0.0);
            let dy = rd.y;
            // To what the pixel shows, or to the plane where the ray meets the water.
            var tt = t_d;
            if (dy < -1e-4) {
                tt = min(tt, e0 / -dy);
            }
            let h = VEIL_LAYER_YD;
            var od = 0.0;
            if (abs(dy) < 1e-4) {
                od = exp(-e0 / h) * tt;
            } else {
                od = h / dy * (exp(-e0 / h) - exp(-(e0 + dy * tt) / h));
            }
            let cover = (1.0 - exp(-VEIL_SIGMA * m * max(od, 0.0))) * VEIL_MAX;
            let colour = mix(wu_dec(u.fog_dry.rgb), wu_dec(u.sky_horizon.rgb), 0.6) * 0.95
                + sun_c * (0.6 * forward * low * saturate(v.sun.w));
            lin = mix(lin, colour, cover);
        }
    }
    if (v.b.z > 0.5) {
        // The spray's light: the sky's and the sun's on foam (in the hue of the light on the
        // terrain), the sun's scattered forward (Henyey-Greenstein, g 0.5, 1 on average), so the
        // mist takes the scene's warm or cool light and glows backlit; held to grey it read as smoke.
        let cosv = dot(rd, sun_l);
        let hg = 0.75 / pow(1.25 - cosv, 1.5);
        let mist_c = (v.e_sky.rgb * 0.75
            + v.e_sun.rgb * (0.5 + 0.35 * min(hg, 4.0) * saturate(v.sun.w))) * 0.9;
        for (var i = 0u; i < 4u; i = i + 1u) {
            let foot = v.falls[2u * i];
            let at = v.falls[2u * i + 1u];
            if (at.w <= 0.0) {
                continue;
            }
            // The line's frame: along it from `a`, across it, up from the pool.
            let ab = foot.zw - foot.xy;
            let len = length(ab);
            let ua = select(vec2<f32>(1.0, 0.0), ab / max(len, 1e-4), len > 1e-3);
            let un = vec2<f32>(-ua.y, ua.x);
            // A short foot line (a trickle) throws a narrow, faint plume.
            let wide = smoothstep(0.5, 8.0, len);
            let drop = min(at.y, 100.0);
            let reach = clamp(0.8 + 0.03 * drop, VEIL_SPRAY_REACH.x, VEIL_SPRAY_REACH.y)
                * mix(0.55, 1.0, wide);
            let rise = clamp(1.0 + 0.05 * drop, VEIL_SPRAY_RISE.x, VEIL_SPRAY_RISE.y)
                * mix(0.7, 1.0, wide);
            let strength = at.w * mix(0.35, 1.0, wide);
            // The ray in that frame, and its span through the plume's box: across to where the
            // widened plume is windowed out, past the ends a short taper, up to the window's top.
            let rel = eye.xz - foot.xy;
            let o = vec3<f32>(dot(rel, ua), eye.y - at.x, dot(rel, un));
            let dl = vec3<f32>(dot(rd.xz, ua), rd.y, dot(rd.xz, un));
            let hw = 3.0 * 1.7 * reach;
            let lo = vec3<f32>(-1.6 * reach, -0.3, -hw);
            let hi = vec3<f32>(len + 1.6 * reach, 3.4 * rise, hw);
            let inv = vec3<f32>(1.0) / select(dl, vec3<f32>(1e-6), abs(dl) < vec3<f32>(1e-6));
            let t0 = (lo - o) * inv;
            let t1 = (hi - o) * inv;
            let tn = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), 0.0));
            let tf = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), min(max(t0.z, t1.z), t_d));
            if (tf <= tn) {
                continue;
            }
            let dt = (tf - tn) / f32(VEIL_SPRAY_STEPS);
            var od = 0.0;
            var oy = 0.0;
            for (var k = 0; k < VEIL_SPRAY_STEPS; k = k + 1) {
                let ts = tn + (f32(k) + 0.5) * dt;
                let q = o + dl * ts;
                let y = q.y;
                if (y < -0.1) {
                    continue;
                }
                let yr = y / rise;
                // Widening as it rises; past the line's ends it stops within a short taper.
                let side = abs(q.z) / (reach * (1.0 + 0.35 * min(yr, 2.0)));
                let past = max(max(-q.x, q.x - len), 0.0) / (0.6 * reach);
                let along = clamp(q.x, 0.0, len);
                // Billows: lumps rolling upward on the clock, chunky low down and thinning higher.
                let n = 0.65 * wu_vnoise(vec2<f32>(along * 0.5 + q.z * 0.25, (y - u.clock.x * 1.1) * 0.55))
                    + 0.35 * wu_vnoise(vec2<f32>(along * 1.3 + 7.0 + q.z * 0.5, (y - u.clock.x * 1.9) * 1.3));
                let lump = smoothstep(0.2, 0.7, n + 0.3 - 0.3 * min(yr / 2.5, 1.0));
                let win = (1.0 - smoothstep(2.2, 3.0, side)) * (1.0 - smoothstep(2.6, 3.4, yr))
                    * (1.0 - smoothstep(1.5, 2.6, past));
                // Soft against what the frame shows inside the box, so no edge cuts round it.
                let soft = smoothstep(0.0, 1.5, t_d - ts);
                let d = exp(-0.8 * side * side - yr - past * past) * lump * win * soft * dt;
                od = od + d;
                oy = oy + d * yr;
            }
            if (od <= 0.0) {
                continue;
            }
            // Dense and bright at the base; over the curtain it hides little.
            let high = smoothstep(0.8, 2.2, oy / od);
            let cap = mix(VEIL_SPRAY_MAX, VEIL_SPRAY_TOP, high);
            let cover = (1.0 - exp(-VEIL_SPRAY_K * strength * od)) * cap;
            lin = mix(lin, mist_c * mix(1.0, 0.85, high), cover);
        }
    }
    return wu_enc(lin);
}

/// The dry combine with the veil (round 3): the veil over the scene, then the glow as ever.
@fragment
fn fs_combine_veil(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let u = uwb.u;
    let uv = in.uv;
    let dd = vec2<f32>(textureDimensions(uw_depth));
    let z = textureLoad(uw_depth, vec2<i32>(clamp(uv * dd, vec2<f32>(0.0), dd - vec2<f32>(1.0))), 0).r;
    let rd = wu_post_ray(uv, u);
    let t_d = select(4000.0, z / max(dot(rd, u.cam_fwd.xyz), 1e-3), z > 0.0);
    let scene = textureSample(in_tex, in_samp, uv);
    let sg = veil_apply(clamp(scene.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), rd, t_d);
    return combine_body_on(uv, in.position.xy, sg, scene.a);
}
#endif
