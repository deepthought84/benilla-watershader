// The planar mirrors' roughness chains (`mirror_chain.rs`), for the preview's rough reflections:
// each level is the level above (the capture for level 0) at half its size, through a [1 3 3 1]
// tent: four bilinear taps on the diagonals at ±0.75 source texel. The capture clears to
// transparent black, so its colour is premultiplied by coverage and the mean stays so.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn mirror_chain(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(dst);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let o = 0.75 / vec2<f32>(textureDimensions(src));
    var acc = textureSampleLevel(src, src_samp, uv + vec2<f32>(-o.x, -o.y), 0.0);
    acc += textureSampleLevel(src, src_samp, uv + vec2<f32>(o.x, -o.y), 0.0);
    acc += textureSampleLevel(src, src_samp, uv + vec2<f32>(-o.x, o.y), 0.0);
    acc += textureSampleLevel(src, src_samp, uv + vec2<f32>(o.x, o.y), 0.0);
    textureStore(dst, vec2<i32>(id.xy), acc * 0.25);
}
