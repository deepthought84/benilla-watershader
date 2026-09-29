// The reflection march's hierarchical depth — a pyramid of NEAREST depths, one mip per halving.
//
// **Why a pyramid at all.** A linear march samples the depth buffer at a fixed cadence and is
// therefore blind between its samples: geometry thinner than a step is jumped over, and at a
// grazing angle the steps grow long enough that this happens constantly. The march in
// `liquid.wgsl` gave up on 13.7% of a grazing water surface at the Stranglethorn camera, and a
// fragment that gives up falls back to the sky — which at sunset is far brighter than the bank it
// should have reflected, so the failures are painted as bright bands. That is the artefact.
//
// A hierarchy removes the blindness rather than reducing it. Each texel of mip `n` holds the
// nearest surface anywhere in its 2^n by 2^n tile, so a ray that passes in front of that value has
// provably missed everything in the tile and can skip the whole tile in one step. A ray that does
// not can descend and ask a finer question. Nothing is jumped over, and the cost is logarithmic in
// the distance travelled instead of linear.
//
// **NEAREST, which is `max` here, and this is the one thing easy to get backwards.** Bevy is
// reverse-Z: the far plane is 0 and nearer surfaces have larger depth. Bevy ships its own depth
// pyramid in `bevy_core_pipeline::mip_generation`, and it reduces with `min` — the FURTHEST surface
// in each tile — because it is built for occlusion culling, where the safe error is to draw
// something that turns out to be hidden. A reflection march needs the opposite safety: it must
// never skip a tile that contains something, so it wants the nearest. That is why this exists
// rather than binding Bevy's.
//
// Sky is depth 0, the far plane, so it loses every `max` against real geometry and a tile holding
// both reports the geometry. That is the conservative answer and it needs no special case.
//
// **And the FARTHEST, into a second pyramid** (`min` here — the other way round from the first).
// The nearest proves a tile empty in FRONT of a ray; the farthest is what proves a ray has passed
// BEHIND everything in a tile, once each surface is given a thickness — see `liquid::hiz`'s image
// doc and `ssr_trace`. Sky, at 0, wins every `min`, so a tile holding any sky can never be passed
// behind: the ray has to descend and find it.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

// Every level below the base, from the level above; the base, mip 0, is `hiz_seed.wgsl`'s
// compute pass from the view's depth. Deliberately no sampler: every read here is a `textureLoad`
// of an exact texel, because a filtered value between two depths is a depth where nothing stands.
@group(0) @binding(1) var lvl: texture_2d<f32>;
// The farthest-depth pyramid's level above, beside `lvl` (the nearest's) — see `liquid::hiz`.
@group(0) @binding(3) var lvl_far: texture_2d<f32>;

// Every pass writes both pyramids: location 0 the nearest, location 1 the farthest.
struct HizOut {
    @location(0) near: f32,
    @location(1) far: f32,
};

/// Every level below: the nearest of the four texels above, and the farthest.
@fragment
fn fs_reduce(in: FullscreenVertexOutput) -> HizOut {
    // `lvl` is a view of the level ABOVE, so its dimensions are the source's and the destination
    // is half of them. Taking the source's as the destination's — and then doubling on top — reads
    // off the end of the texture, which clamps to the edge and leaves every level below the base
    // holding one corner of the level above. That is what an empty pyramid looked like.
    let src_dims = vec2<i32>(textureDimensions(lvl));
    let dst_dims = max(src_dims / 2, vec2<i32>(1));
    let p = vec2<i32>(in.uv * vec2<f32>(dst_dims)) * 2;
    let hi = src_dims - vec2<i32>(1);
    let pa = clamp(p, vec2<i32>(0), hi);
    let pb = clamp(p + vec2<i32>(1, 0), vec2<i32>(0), hi);
    let pc = clamp(p + vec2<i32>(0, 1), vec2<i32>(0), hi);
    let pd = clamp(p + vec2<i32>(1, 1), vec2<i32>(0), hi);
    let near = max(
        max(textureLoad(lvl, pa, 0).r, textureLoad(lvl, pb, 0).r),
        max(textureLoad(lvl, pc, 0).r, textureLoad(lvl, pd, 0).r),
    );
    let far = min(
        min(textureLoad(lvl_far, pa, 0).r, textureLoad(lvl_far, pb, 0).r),
        min(textureLoad(lvl_far, pc, 0).r, textureLoad(lvl_far, pd, 0).r),
    );
    return HizOut(near, far);
}
