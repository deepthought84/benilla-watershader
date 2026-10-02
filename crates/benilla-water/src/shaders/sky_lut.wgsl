// The sky floor's cube (`benilla_water::sky_lut`): the sky the water reflects where no tier has a
// better answer, as a 64² cube with five prefiltered mips, rebuilt every frame the stylised water is
// on screen. Mip 0 is the dome's own gradient (the dome law, the sky pass's text) under the drawn
// cloud layer (the cloud law, the cloud dome's mesh mapping inverted), unlit and in linear light;
// each mip below it widens the one above by a Gaussian, so the water reads the sky blurred by its
// own roughness (`liquid.wgsl`'s `sky_floor`).

// The reflection block (`liquid.wgsl`'s `WaterReflect`) as rows; the dome's seven from DOME_ROW.
@group(0) @binding(0) var<storage, read> lanes: array<vec4<f32>>;
// The cloud dome's colored tile, raw gamma, alpha the coverage (`clouds::SkyCloudLayer`).
@group(0) @binding(1) var cloud_tex: texture_2d<f32>;
@group(0) @binding(2) var cloud_samp: sampler;
// The mip this dispatch writes, its six faces as layers.
@group(0) @binding(3) var out_mip: texture_storage_2d_array<rgba16float, write>;
// The mip above it, as a cube (`lut_prefilter`).
@group(0) @binding(4) var in_mip: texture_cube<f32>;
@group(0) @binding(5) var lut_samp: sampler;
// x 1 while the cloud dome is drawn, 0 while the sky hides it (a WMO skybox, under water).
@group(0) @binding(6) var<uniform> params: vec4<f32>;

// The dome's first row in the block (`hooks::DOME_LANES`).
const DOME_ROW: u32 = #{DOME_ROW};

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

// ---- cloud law: begin (the cloud dome's mesh, `clouds::layer`'s `cloud_dome_mesh`, inverted) ----
//
// The dome is a 12-ring cap of the unit sphere recentred by `−cos(π/4)` so its rim sits at eye
// level, every vertex pushed out along its own direction; ring `i` sits at co-latitude
// `CLOUD_RING_COLAT[i]·π` with the polar UV radius `i/24` and the ring's alpha. `clouds::layer`'s
// tests pin these arrays to the mesh's.
const CLOUD_SHIFT: f32 = 0.70710678;
const CLOUD_RING_COLAT = array<f32, 12>(0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.205, 0.23, 0.245, 0.25);
const CLOUD_RING_ALPHA = array<f32, 12>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.50196078, 0.0, 0.0);

/// The cloud tile UV the drawn dome shows along `dir` (xy), and the dome's ring alpha there (z, 0
/// below the rim).
fn cloud_dome_uv(dir: vec3<f32>) -> vec3<f32> {
    // The far intersection of the ray from the eye with the recentred sphere: the point on the
    // unit sphere the mesh vertex along `dir` was pushed out from.
    let b = CLOUD_SHIFT * dir.y;
    let t = -b + sqrt(max(b * b - (CLOUD_SHIFT * CLOUD_SHIFT - 1.0), 0.0));
    let q = t * dir + vec3<f32>(0.0, CLOUD_SHIFT, 0.0);
    let colat = acos(clamp(q.y, -1.0, 1.0)) / 3.14159265;
    // Indexed at run time, so read through locals.
    var colats = CLOUD_RING_COLAT;
    var alphas = CLOUD_RING_ALPHA;
    var ring = 11.0;
    var alpha = 0.0;
    for (var i = 0; i < 11; i = i + 1) {
        if (colat < colats[i + 1]) {
            let f = (colat - colats[i]) / (colats[i + 1] - colats[i]);
            ring = f32(i) + f;
            alpha = mix(alphas[i], alphas[i + 1], f);
            break;
        }
    }
    let h = length(q.xz);
    let sc = select(vec2<f32>(0.0), q.xz / h, h > 1e-6);
    return vec3<f32>(sc * (ring / 24.0) + 0.5, alpha);
}
// ---- cloud law: end ----

/// The water's display gamma, `liquid.wgsl`'s (a test compares them).
const WATER_GAMMA: f32 = 2.2;

fn water_dec(g: vec3<f32>) -> vec3<f32> {
    return pow(max(g, vec3<f32>(0.0)), vec3<f32>(WATER_GAMMA));
}

/// The level-0 texel's angle at a face centre (2/64 in tangent units), radians.
const SKY_LUT_TEXEL0: f32 = 0.03125;

/// The direction through face `face` at `uv` (each in −1..1, u right, v down), in the wgpu cube
/// convention: +X −X +Y −Y +Z −Z.
fn cube_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let u = uv.x;
    let v = uv.y;
    switch face {
        case 0u: { return vec3<f32>(1.0, -v, -u); }
        case 1u: { return vec3<f32>(-1.0, -v, u); }
        case 2u: { return vec3<f32>(u, 1.0, v); }
        case 3u: { return vec3<f32>(u, -1.0, -v); }
        case 4u: { return vec3<f32>(u, -v, 1.0); }
        default: { return vec3<f32>(-u, -v, -1.0); }
    }
}

/// The direction through the centre of texel `xy` of a `size`² face.
fn texel_dir(face: u32, xy: vec2<u32>, size: u32) -> vec3<f32> {
    return normalize(cube_dir(face, (vec2<f32>(xy) + 0.5) / f32(size) * 2.0 - 1.0));
}

/// Mip 0: the dome along each texel's direction, the cloud layer over it as the sky pass blends
/// it (premultiplied, in gamma), decoded to linear light.
@compute @workgroup_size(8, 8, 1)
fn lut_base(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_mip).x;
    if (gid.x >= size || gid.y >= size) {
        return;
    }
    let dir = texel_dir(gid.z, gid.xy, size);
    let dome = DomeColors(
        lanes[DOME_ROW + 0u],
        lanes[DOME_ROW + 1u],
        lanes[DOME_ROW + 2u],
        lanes[DOME_ROW + 3u],
        lanes[DOME_ROW + 4u],
        lanes[DOME_ROW + 5u],
        lanes[DOME_ROW + 6u],
    );
    let sky_g = dome_color(dome, dir);
    let c = cloud_dome_uv(dir);
    let texel = textureSampleLevel(cloud_tex, cloud_samp, c.xy, 0.0);
    let a = texel.a * c.z * params.x;
    let out_g = texel.rgb * a + sky_g * (1.0 - a);
    textureStore(out_mip, vec2<i32>(gid.xy), i32(gid.z), vec4<f32>(water_dec(out_g), 1.0));
}

/// Mip k from mip k − 1: a Gaussian of the width that takes mip k − 1's blur to mip k's,
/// `σ_k = SKY_LUT_TEXEL0·2^k`, so `σ_inc = √(σ_k² − σ_(k−1)²) = SKY_LUT_TEXEL0·2^(k−1)·√3`,
/// importance-sampled on 16 golden-angle taps in the tangent plane.
@compute @workgroup_size(8, 8, 1)
fn lut_prefilter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_mip).x;
    if (gid.x >= size || gid.y >= size) {
        return;
    }
    let k = f32(firstTrailingBit(64u / size));
    let c = texel_dir(gid.z, gid.xy, size);
    // Frisvad's orthonormal basis about c.
    var t: vec3<f32>;
    var b: vec3<f32>;
    if (c.z < -0.9999999) {
        t = vec3<f32>(0.0, -1.0, 0.0);
        b = vec3<f32>(-1.0, 0.0, 0.0);
    } else {
        let a = 1.0 / (1.0 + c.z);
        let d = -c.x * c.y * a;
        t = vec3<f32>(1.0 - c.x * c.x * a, d, -c.x);
        b = vec3<f32>(d, 1.0 - c.y * c.y * a, -c.y);
    }
    let sigma = SKY_LUT_TEXEL0 * exp2(k - 1.0) * 1.7320508;
    var acc = vec3<f32>(0.0);
    for (var j = 0; j < 16; j = j + 1) {
        let r = sigma * sqrt(-2.0 * log(1.0 - (f32(j) + 0.5) / 16.0));
        let th = f32(j) * 2.39996323;
        let d = normalize(c + r * (cos(th) * t + sin(th) * b));
        acc = acc + textureSampleLevel(in_mip, lut_samp, d, 0.0).rgb;
    }
    textureStore(out_mip, vec2<i32>(gid.xy), i32(gid.z), vec4<f32>(acc / 16.0, 1.0));
}
