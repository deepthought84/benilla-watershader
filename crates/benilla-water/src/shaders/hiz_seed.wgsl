// The depth pyramid's seed, and the water's linear depth, from one read of the view's depth.
//
// One invocation per pyramid texel of mip 0. It loads the view-depth texels its footprint covers
// and writes the NEAREST and the FARTHEST of them into mip 0 of the two pyramids (`liquid::hiz`
// has why both, and why the nearest is a `max`: Bevy is reverse-Z). The same dispatch also writes
// the full-resolution linear depth the water reads behind each of its pixels, inside the water's
// rectangle: each workgroup takes 8 × 8 blocks of the rectangle, one pixel an invocation, the
// dispatch's workgroup count of blocks apart.
//
// **Perf: in 8 × 8 blocks, not by owner.** Written by each pyramid texel's invocation for the view
// pixels it owns, a wave's stores scattered over a 13 × 14 patch in nine masked stores; the
// stores were most of the seed's 1.07 ms at 3200 × 1800 (STV, an RX 6500 XT). A block a workgroup
// lands on the image's tiling: 0.71 ms (a row-order walk, 1.02).
//
// **All integer.** The pyramid is the previous power of two of the view on each axis, so a texel's
// footprint is a ratio below 2 and spans at most three view texels per axis. The footprint is an
// exact integer expression, mirrored and tested on the CPU in `liquid::hiz`.
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

// Write the linear depth of `px`. Under MSAA `d` is the pixel's nearest sample, so a sub-pixel edge
// reads as the thing in front.
fn put_linear(px: vec2<u32>, d: f32) {
    let z = select(0.0, p.near / d, d > 0.0 && (p.write_hiz & ORTHOGRAPHIC) == 0u);
    textureStore(lin_depth, vec2<i32>(px), vec4<f32>(z, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_seed(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    // The linear depth: the rectangle's 8 × 8 blocks, this workgroup's every `stride`-th.
    let span = p.lin_max - p.lin_min;
    let blocks = (span + vec2<u32>(7u)) / 8u;
    let count = blocks.x * blocks.y;
    let stride = nwg.x * nwg.y;
    let inner = vec2<u32>(li % 8u, li / 8u);
    for (var j = wg.y * nwg.x + wg.x; j < count; j = j + stride) {
        let px = p.lin_min + vec2<u32>(j % blocks.x, j / blocks.x) * 8u + inner;
        if (all(px < p.lin_max)) {
            put_linear(px, load_depth(px).x);
        }
    }
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
    } else {
        // A view and pyramid whose sizes disagree by more than the power-of-two rule allows (the
        // node's size guard keeps this unreachable): still every covered texel, just slower.
        for (var y = lo.y; y <= hi.y; y = y + 1u) {
            for (var x = lo.x; x <= hi.x; x = x + 1u) {
                let v = load_depth(vec2<u32>(x, y));
                near = max(near, v.x);
                far = min(far, v.y);
            }
        }
    }
    if ((p.write_hiz & WRITE_HIZ) != 0u) {
        textureStore(hiz_near, vec2<i32>(t), vec4<f32>(near, 0.0, 0.0, 0.0));
        textureStore(hiz_far, vec2<i32>(t), vec4<f32>(far, 0.0, 0.0, 0.0));
    }
}
