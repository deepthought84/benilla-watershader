// The depth pyramid's seed, and the water's linear depth, from one read of the view's depth.
//
// One invocation per pyramid texel of mip 0. It loads the view-depth texels its footprint covers
// and writes the NEAREST and the FARTHEST of them into mip 0 of the two pyramids (`liquid::hiz`
// has why both, and why the nearest is a `max`: Bevy is reverse-Z). The same loads also give the
// full-resolution linear depth the water reads behind each of its pixels: each view pixel belongs
// to exactly one pyramid texel, and that texel's invocation writes it.
//
// **All integer.** The pyramid is the previous power of two of the view on each axis, so a texel's
// footprint is a ratio below 2 and spans at most three view texels per axis. The footprint and the
// ownership are exact integer expressions, mirrored and tested on the CPU in `liquid::hiz`.
//
// **Linear depth.** Bevy's perspective is `perspective_infinite_reverse_rh`: `d = near / z_view`,
// so `z_view = near / d`, in yards along the view axis. `d == 0` is the cleared far plane (sky, or
// nothing drawn) and stays 0, which every reader takes as "unknown".

struct SeedParams {
    src_size: vec2<u32>,    // view depth size, physical px
    dst_size: vec2<u32>,    // pyramid mip-0 size (prev_pot per axis, >= 8)
    lin_min: vec2<u32>,     // linear-depth write rect, inclusive (WaterScreenRect)
    lin_max: vec2<u32>,     // exclusive
    tile_origin: vec2<u32>, // first pyramid texel of this dispatch
    write_hiz: u32,         // bit 0: write mip 0 of both pyramids; bit 1: orthographic, linear depth 0
    near: f32,              // clip_from_view.w_axis.z: Bevy's infinite reverse-Z near plane, yd
};

#ifdef SEED_MS
@group(0) @binding(1) var src_ms: texture_depth_multisampled_2d;
const SEED_SAMPLES: i32 = #{SEED_SAMPLES};
#else
@group(0) @binding(0) var src: texture_depth_2d;
#endif
@group(0) @binding(2) var hiz_near: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var hiz_far: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var lin_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var<uniform> p: SeedParams;

const WRITE_HIZ: u32 = 1u;
const ORTHOGRAPHIC: u32 = 2u;

// The first view texel pyramid texel `d` covers, per axis.
fn footprint_lo(d: u32, src: u32, dst: u32) -> u32 {
    return (d * src) / dst;
}

// The last view texel pyramid texel `d` covers, per axis (inclusive).
fn footprint_hi(d: u32, src: u32, dst: u32) -> u32 {
    return min(((d + 1u) * src + dst - 1u) / dst, src) - 1u;
}

// The pyramid texel view texel `x`'s centre falls in, per axis: its one owner.
fn owner(x: u32, src: u32, dst: u32) -> u32 {
    return ((2u * x + 1u) * dst) / (2u * src);
}

// One view pixel's depth as (nearest, farthest): its value, or the max and min over its samples.
fn load_depth(px: vec2<u32>) -> vec2<f32> {
#ifdef SEED_MS
    var near = 0.0;
    var far = 1.0;
    for (var i = 0; i < SEED_SAMPLES; i = i + 1) {
        let v = textureLoad(src_ms, vec2<i32>(px), i);
        near = max(near, v);
        far = min(far, v);
    }
    return vec2<f32>(near, far);
#else
    let v = textureLoad(src, vec2<i32>(px), 0);
    return vec2<f32>(v, v);
#endif
}

// Write the linear depth of `px` if pyramid texel `t` owns it and it lies inside the water's rect.
// Under MSAA `d` is the pixel's nearest sample, so a sub-pixel edge reads as the thing in front.
fn put_linear(px: vec2<u32>, t: vec2<u32>, d: f32) {
    if (owner(px.x, p.src_size.x, p.dst_size.x) != t.x || owner(px.y, p.src_size.y, p.dst_size.y) != t.y) {
        return;
    }
    if (any(px < p.lin_min) || any(px >= p.lin_max)) {
        return;
    }
    let z = select(0.0, p.near / d, d > 0.0 && (p.write_hiz & ORTHOGRAPHIC) == 0u);
    textureStore(lin_depth, vec2<i32>(px), vec4<f32>(z, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_seed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = p.tile_origin + gid.xy;
    if (any(t >= p.dst_size)) {
        return;
    }
    let lo = vec2<u32>(
        footprint_lo(t.x, p.src_size.x, p.dst_size.x),
        footprint_lo(t.y, p.src_size.y, p.dst_size.y),
    );
    let hi = vec2<u32>(
        footprint_hi(t.x, p.src_size.x, p.dst_size.x),
        footprint_hi(t.y, p.src_size.y, p.dst_size.y),
    );
    var near = 0.0;
    var far = 1.0;
    if (hi.x - lo.x <= 2u && hi.y - lo.y <= 2u) {
        // **Nine independent loads, not a loop**: a loop with data-dependent bounds serialises its
        // loads. A clamped 3x3 covers the footprint; a texel read twice is harmless to a max, a
        // min, or a store of the same value.
        let x1 = min(lo.x + 1u, hi.x);
        let x2 = min(lo.x + 2u, hi.x);
        let y1 = min(lo.y + 1u, hi.y);
        let y2 = min(lo.y + 2u, hi.y);
        let a = load_depth(vec2<u32>(lo.x, lo.y));
        let b = load_depth(vec2<u32>(x1, lo.y));
        let c = load_depth(vec2<u32>(x2, lo.y));
        let e = load_depth(vec2<u32>(lo.x, y1));
        let f = load_depth(vec2<u32>(x1, y1));
        let g = load_depth(vec2<u32>(x2, y1));
        let h = load_depth(vec2<u32>(lo.x, y2));
        let k = load_depth(vec2<u32>(x1, y2));
        let m = load_depth(vec2<u32>(x2, y2));
        near = max(max(max(a.x, b.x), max(c.x, e.x)), max(max(f.x, g.x), max(max(h.x, k.x), m.x)));
        far = min(min(min(a.y, b.y), min(c.y, e.y)), min(min(f.y, g.y), min(min(h.y, k.y), m.y)));
        put_linear(vec2<u32>(lo.x, lo.y), t, a.x);
        put_linear(vec2<u32>(x1, lo.y), t, b.x);
        put_linear(vec2<u32>(x2, lo.y), t, c.x);
        put_linear(vec2<u32>(lo.x, y1), t, e.x);
        put_linear(vec2<u32>(x1, y1), t, f.x);
        put_linear(vec2<u32>(x2, y1), t, g.x);
        put_linear(vec2<u32>(lo.x, y2), t, h.x);
        put_linear(vec2<u32>(x1, y2), t, k.x);
        put_linear(vec2<u32>(x2, y2), t, m.x);
    } else {
        // A view and pyramid whose sizes disagree by more than the power-of-two rule allows (the
        // node's size guard keeps this unreachable): still every covered texel, just slower.
        for (var y = lo.y; y <= hi.y; y = y + 1u) {
            for (var x = lo.x; x <= hi.x; x = x + 1u) {
                let v = load_depth(vec2<u32>(x, y));
                near = max(near, v.x);
                far = min(far, v.y);
                put_linear(vec2<u32>(x, y), t, v.x);
            }
        }
    }
    if ((p.write_hiz & WRITE_HIZ) != 0u) {
        textureStore(hiz_near, vec2<i32>(t), vec4<f32>(near, 0.0, 0.0, 0.0));
        textureStore(hiz_far, vec2<i32>(t), vec4<f32>(far, 0.0, 0.0, 0.0));
    }
}
