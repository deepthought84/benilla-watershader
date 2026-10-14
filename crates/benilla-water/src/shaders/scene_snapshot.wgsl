// The water's scene snapshot: the main texture after the opaque world, copied texel for texel into
// the 10-bit `WaterSceneColor` (`liquid::scene_color`). The main texture is the resolved,
// single-sampled one, so one `textureLoad` per pixel is the whole copy; the format conversion
// clamps to [0, 1], which only fullbright magma and slime exceed.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var main_tex: texture_2d<f32>;

@fragment
fn fs_snapshot(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return textureLoad(main_tex, vec2<i32>(in.position.xy), 0);
}
