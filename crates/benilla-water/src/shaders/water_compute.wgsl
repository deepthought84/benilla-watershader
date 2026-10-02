// The water's compute pass (`benilla_water::water_compute`), one dispatch per entry on the world
// view, after the depth pyramids and the scene snapshot and before the water draws.
//
// `occl_plane`: the sky floor's occlusion for the water on an active mirror plane, at a quarter of
// the view's resolution (an eighth on Low). Each texel finds the plane point under its pixel and
// marches the reflected ray through the pyramids by the floor march law, the same text the water's
// fragment runs per pixel off the planes; the fragment reads the two images bilinearly at its
// rippled lookup point.

#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var hiz_tex: texture_2d<f32>;
@group(0) @binding(2) var hiz_far_tex: texture_2d<f32>;
// The opaque world's view Z behind each pixel, yards, 0 where nothing was drawn.
@group(0) @binding(3) var scene_depth_tex: texture_2d<f32>;
@group(0) @binding(4) var scene_tex: texture_2d<f32>;
@group(0) @binding(5) var scene_samp: sampler;
// The reflection block (`liquid.wgsl`'s `WaterReflect`) as rows.
@group(0) @binding(6) var<storage, read> lanes: array<vec4<f32>>;
// r the open sky v, g and b the body's sky and sun visibility (v and 1 until they are traced),
// a 1 where the march reached a verdict.
@group(0) @binding(7) var water_occl: texture_storage_2d<rgba8unorm, write>;
// rgb the occluder's radiance times 1 − v, linear; a rho_c.
@group(0) @binding(8) var floor_occ: texture_storage_2d<rgba16float, write>;

struct OcclParams {
    // The texels of the dispatch: the quarter-resolution tiles covering the water's rectangle.
    tile_origin: vec2<u32>,
    tile_end: vec2<u32>,
    // The lighting sun's travel direction; written 0 and read by no entry point.
    sun_travel: vec4<f32>,
};
@group(0) @binding(9) var<uniform> params: OcclParams;

// The mirror planes' rows in the block: `params` and `mirror2` (`hooks::MIRROR2_LANES`).
const PARAMS_ROW: u32 = #{PARAMS_ROW};
const MIRROR2_ROW: u32 = #{MIRROR2_ROW};

// The pass's tier: its march's lowest level and step budget.
#ifdef WATER_TIER_HIGH
const OCC_LEVEL: i32 = 1;
const OCC_STEPS: i32 = 16;
#else
#ifdef WATER_TIER_LOW
const OCC_LEVEL: i32 = 3;
const OCC_STEPS: i32 = 8;
#else
const OCC_LEVEL: i32 = 2;
const OCC_STEPS: i32 = 12;
#endif
#endif

/// The water's display gamma, `liquid.wgsl`'s (a test compares them).
const WATER_GAMMA: f32 = 2.2;

fn water_dec(g: vec3<f32>) -> vec3<f32> {
    return pow(max(g, vec3<f32>(0.0)), vec3<f32>(WATER_GAMMA));
}

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
/// The march's hard step bound: the largest tier's budget.
const LAW_MAX_STEPS: i32 = 16;

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

/// Where a floor march ended: its verdict, the screen UV and device depth it reached, and how far
/// along the ray that point is from the origin, yards.
struct FloorMarch {
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
/// - a hit is validated against the slab of the `min_level` texel it stopped in.
///
/// `travelled` is the distance from the origin to the point the verdict was reached at: the hit,
/// where the ray left the frame, or where the budget ran out. The reflected thing is at least that
/// far behind the water.
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
    // Past the far plane the depth cannot place the point: as far as it reaches.
    let at = law_ndc_to_world(vec3<f32>(law_uv_to_ndc(p.xy), max(p.z, 1e-6)));
    return FloorMarch(state, p.xy, p.z, distance(origin, at));
}

/// The sky floor's occlusion at one point: `rgb` the occluder's radiance times `1 − v` (linear),
/// `v` how much of the sky is open, `rho_c` how far behind the water the reflected thing lies as a
/// share of the eye's distance (0 at a contact, toward 1 far off), `known` 1 where the march
/// reached a verdict.
struct FloorOcc {
    rgb: vec3<f32>,
    v: f32,
    rho_c: f32,
    known: f32,
};

/// The march's verdict as the floor takes it. `rho_c = D/(d_eye + D)` at the distance `D` the
/// verdict was reached: a hit at `D`, an escape or an unknown past it, all say the reflected thing
/// is at least `D` behind the water, so it is continuous from a contact to the far field. The
/// occluder's colour is read at the hit shifted by `tap_shift`, the ripple's slide of the lookup
/// point on screen.
fn floor_from_march(m: FloorMarch, d_eye: f32, min_level: i32, tap_shift: vec2<f32>) -> FloorOcc {
    let rho = m.travelled / max(d_eye + m.travelled, 1e-4);
    if (m.state == FLOOR_HIT) {
        return FloorOcc(water_dec(snapshot_box(m.uv + tap_shift, min_level)), 0.0, rho, 1.0);
    }
    if (m.state == FLOOR_ESCAPED) {
        return FloorOcc(vec3<f32>(0.0), 1.0, rho, 1.0);
    }
    return FloorOcc(vec3<f32>(0.0), 1.0, rho, 0.0);
}
// ---- floor march law: end ----

/// The candidate point on a mirror plane at height `h` under the view ray `dir` from the eye, and
/// whether it qualifies: an active plane below the eye, in front of the opaque surface the pixel
/// shows (`z_s`, view Z yards, 0 where nothing was drawn).
fn plane_point(h: f32, strength: f32, dir: vec3<f32>, z_s: f32) -> vec4<f32> {
    let eye = view.world_position.xyz;
    if (strength <= 0.0 || dir.y >= 0.0 || eye.y <= h) {
        return vec4<f32>(0.0);
    }
    let p = eye + dir * ((h - eye.y) / dir.y);
    let z = -(view.view_from_world * vec4<f32>(p, 1.0)).z;
    // A water surface lies just above its own bed: a plane point behind the opaque surface is
    // another body's plane seen through this one's bed.
    let ok = z_s <= 0.0 || z < z_s;
    return vec4<f32>(p, select(0.0, z, ok));
}

@compute @workgroup_size(8, 8, 1)
fn occl_plane(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = params.tile_origin + gid.xy;
    if (any(q >= params.tile_end)) {
        return;
    }
    let dims = textureDimensions(water_occl);
    if (any(q >= dims)) {
        return;
    }
    // The texel's centre as the fragment's bilinear read sees it, and the view pixel under it.
    let uv = (vec2<f32>(q) + 0.5) / vec2<f32>(dims);
    let px = vec2<i32>(uv * view.viewport.zw);
    let z_s = textureLoad(scene_depth_tex, px, 0).r;
    let eye = view.world_position.xyz;
    let dir = normalize(law_ndc_to_world(vec3<f32>(law_uv_to_ndc(uv), 1e-3)) - eye);
    // The farthest qualifying plane point: over the sea the pool plane's point floats in front of
    // the sea bed and the sea plane's is the water's, while over the pool the sea plane's lies
    // behind the pool bed and is rejected.
    let a = plane_point(lanes[PARAMS_ROW].x, lanes[PARAMS_ROW].y, dir, z_s);
    let b = plane_point(lanes[MIRROR2_ROW].x, lanes[MIRROR2_ROW].y, dir, z_s);
    let pick = select(a, b, b.w > a.w);
    if (pick.w <= 0.0) {
        textureStore(water_occl, vec2<i32>(q), vec4<f32>(1.0, 1.0, 1.0, 0.0));
        textureStore(floor_occ, vec2<i32>(q), vec4<f32>(0.0));
        return;
    }
    let p = pick.xyz;
    let m = floor_march(p + vec3<f32>(0.0, 0.05, 0.0), vec3<f32>(dir.x, -dir.y, dir.z), OCC_LEVEL, OCC_STEPS);
    let o = floor_from_march(m, distance(p, eye), OCC_LEVEL, vec2<f32>(0.0));
    textureStore(water_occl, vec2<i32>(q), vec4<f32>(o.v, o.v, 1.0, o.known));
    textureStore(floor_occ, vec2<i32>(q), vec4<f32>(o.rgb, o.rho_c));
}
