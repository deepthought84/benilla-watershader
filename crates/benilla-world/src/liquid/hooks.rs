//! The seam the Improved Water plugs into: what the reference water and the rest of the engine
//! must name whether or not that plugin is built. The render layers, the camera markers the static
//! pass and the cull filter on, the probe overlay the minimap reads, the layout of the reflection
//! block `liquid.wgsl` binds, and the pass images every liquid material holds.
//!
//! Without the plugin the pass images are one-texel placeholders and the reflection block is
//! zeros, which the shader reads as "no tier contributing"; the plugin publishes its own targets
//! into [`WaterPassImages`] in [`WaterPassSet`], before the materials are built.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{
    BufferDescriptor, BufferUsages, Extent3d, TextureDimension, TextureFormat,
    TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::renderer::RenderDevice;

/// The render layer liquid surfaces ride, and the one the reflection camera does not render.
///
/// Public because the app's booth-layer ladder (`portrait`) must stay clear of it, and asserts that
/// it does — a silent layer collision there is exactly the class of bug that ladder was created to
/// stop, and this is the engine's one claim on a layer index outside it.
pub const WATER_RENDER_LAYER: usize = 31;

/// The render layer for **world geometry that must never appear in the mirror** — the overhead unit
/// names, the raid target marks, the questgiver `!` and `?` markers, and the sun's and moon's
/// lens-flare glares.
///
/// They are not UI. The 1.12 client draws overhead names inside the world pass, depth-tested, as
/// real camera-facing billboard meshes (`benilla_app::nameplates`), which is why walls occlude
/// them — and it is also why the mirrored camera drew them, floating under the surface of every
/// lake, back to front. A name is a label attached to the viewer's own eye; it has no reflection,
/// any more than a cursor does. A questgiver marker is the same kind of label, and a glare is the
/// eye's own: 1.12 draws it on its lens-flare path (`0x483740 → 0x6d48c0 → 0x7e57e0`) at
/// `cam + 12·dir`, sized by where the camera looks, so a mirror or probe that drew the quad would
/// show the viewer's lens, not the sky. Layers do not propagate: every such mesh carries this one
/// itself.
///
/// Distinct from [`WATER_RENDER_LAYER`] even though both mean "the world camera draws it, the
/// mirror does not", because the two say different things and only one of them is also a statement
/// about the water: liquid is excluded so it cannot wash the image teal from in front of the
/// mirrored lens, and merging them would make that reason cover labels too.
pub const UNMIRRORED_RENDER_LAYER: usize = 30;

/// The world camera's layers: the world, the labels over it, and the water. A mesh on none of them
/// belongs to a booth's own camera and scene.
pub(crate) fn world_camera_layers() -> bevy::camera::visibility::RenderLayers {
    bevy::camera::visibility::RenderLayers::from_layers(&[
        0,
        UNMIRRORED_RENDER_LAYER,
        WATER_RENDER_LAYER,
    ])
}

/// The world camera renders the world, the water, and the labels over it; the reflection camera
/// renders only the world. One system rather than a component at the spawn, because there are three
/// spawn sites for the same camera (the client's, its no-data fallback, and the world viewer's) and
/// none of them should have to know that water is layered.
pub(super) fn stamp_world_camera_layers(
    mut commands: Commands,
    cameras: Query<
        Entity,
        (
            With<crate::view::WorldCamera>,
            Added<crate::view::WorldCamera>,
        ),
    >,
) {
    for entity in &cameras {
        commands.entity(entity).insert(world_camera_layers());
    }
}

/// The mirror's **view-key shape** — the components that decide which pipelines its draws need,
/// kept in one place because a second camera has to reproduce them exactly.
///
/// A pipeline is specialized against the VIEW as well as the material, so this bundle decides
/// whether the mirror shares the world camera's pipelines: no multisampling and no glow pass (this
/// image is sampled through a rippling normal at half resolution, where neither is recoverable).
/// Neither camera carries a depth prepass, so with MSAA off the mirror's key is the world's and its
/// draws reuse the world camera's pipelines.
///
/// Extracted so `pipe_warm`'s warm mirror can be built from the SAME bundle. Warming a view key by
/// hand-copying four components is how a warm pass silently stops covering the thing it was written
/// for: the copy keeps compiling, the keys quietly diverge, and the only symptom is the tripwire
/// firing months later with no obvious connection to whatever changed here.
pub fn mirror_view_shape() -> impl Bundle {
    (
        Camera3d::default(),
        bevy::render::view::Msaa::Off,
        bevy::render::view::Hdr,
        bevy::core_pipeline::tonemapping::Tonemapping::None,
    )
}

/// Marks a mirrored camera; the retained static pass draws into exactly the views it is told to,
/// and the water's mirrors are among them (see `static_gx::render`'s marker).
#[derive(Component)]
pub struct ReflectionCamera(pub usize);

/// Which cube face a capture camera owns, in wgpu's layer order: +X, -X, +Y, -Y, +Z, -Z.
///
/// Extracted to the render world so the probe's node can pair each face's colour target with that
/// view's `ViewDepthTexture` — the depth is what becomes the cube's alpha.
#[derive(Component, Clone, Copy, bevy::render::extract_component::ExtractComponent)]
pub struct ProbeFace(pub usize);

/// What the probe needs the static cull to admit while it is capturing.
///
/// **A cubemap needs every direction and the cull list only has one.** `static_gx` builds a single
/// visible set from the WORLD camera's frustum and every view draws it, so a probe face aimed
/// anywhere the player is not looking receives terrain and sky and none of the trees, doodads or
/// buildings that make a reflection worth having. The mirror accepts that knowingly — it is aimed
/// roughly where the camera is — but for a cube it is not an approximation, it is the whole failure.
///
/// So during a capture the cull additionally admits everything within [`radius`](Self::radius) of
/// the probe, frustum and farclip ignored. It is a sphere test per cell, it runs on six frames out
/// of every three hundred and sixty, and the extra cells it admits are real world geometry that the
/// main view simply clips — so the only cost is a little draw time on those frames. A second full
/// cull walk was rejected for the mirror on cost grounds and would be the wrong answer here too.
#[derive(Resource, Clone, Copy, Default)]
pub struct ProbeCull {
    /// Set for the whole capture and a few frames either side of it.
    ///
    /// **Held rather than pulsed, because system order is not guaranteed.** The cull and the
    /// capture driver both run in `Update` with no ordering between them, so a flag raised as a
    /// face is activated can easily be read by a cull that already ran — and that face captures a
    /// world with no trees in it while its neighbours capture one with. It showed as two of the six
    /// faces coming back as flat sky panels in the unwrapped cube while the other four held the
    /// forest. A countdown that spans the whole six-frame capture cannot be raced.
    pub active: bool,
    /// The capture point, in Bevy world yards.
    pub at: Vec3,
    /// How far around it to admit, in yards: the scenery the water's reflected rays meet
    /// (`benilla_water`'s `PROBE_CULL_YD`).
    pub radius: f32,
    /// Frames remaining on the hold. Counted down by the capture driver, read as `active` by the
    /// cull.
    pub hold: u32,
}

/// One probe as the debug overlays see it: where it stands, and whether the water can actually read
/// it this frame.
///
/// A flat published view rather than access to the slots themselves, because the overlays want
/// exactly two facts and the slot carries eight — including the resident/target split, which is an
/// invariant of the capture scheduler and nobody else's business.
#[derive(Clone, Copy, Debug)]
pub struct ProbeMark {
    /// The capture point, in **Bevy** world yards.
    pub at: Vec3,
    /// Whether this probe holds a cube the water is reading. False while a slot is waiting for its
    /// first capture.
    pub live: bool,
    /// **This probe's share of the blend at the player's position**, 0 to 1 — normalised across the
    /// taps, so the lit markers are the probes the water is actually reading and the rest are dark.
    pub fade: f32,
}

/// The live probes, republished each frame for the minimap and world overlays; empty without the
/// Improved Water plugin.
#[derive(Resource, Default)]
pub struct ProbeMarks(pub Vec<ProbeMark>);

/// The probe slot ceiling the shader's array is built for.
pub const PROBE_SLOT_MAX: usize = 16;
/// Where the probe slots sit in the reflection block: four `vec4`s per slot.
pub const PROBE_LANES: std::ops::Range<usize> = 48..48 + PROBE_SLOT_MAX * 16;
/// Where the dome's seven rows sit in the reflection block — `sky0..sky4`, `fog`, `warp`, as
/// [`crate::sky::dome_uniforms`] hands them to the dome itself. The shader's `WaterReflect::dome`.
pub const DOME_LANES: std::ops::Range<usize> = PROBE_LANES.end..PROBE_LANES.end + 28;
/// The march's own switches — the shader's `WaterReflect::march`: `x` rays may pass behind tiles,
/// `y` the ripple's origin shift, `z` the probe's softening, `w` probes bound to fixed spots.
pub const MARCH_LANES: std::ops::Range<usize> = DOME_LANES.end..DOME_LANES.end + 4;
/// The second mirror's row: plane, strength, distortion, tolerance — the same four as the first
/// mirror's `params`, zero while it is off.
pub const MIRROR2_LANES: std::ops::Range<usize> = MARCH_LANES.end..MARCH_LANES.end + 4;
/// The stylised water's clock, every phase already reduced by its own period: eight map offsets
/// (the five ripple octaves, then the shore bubbles' three reads) and a row of cycles (flow
/// restart, shore swell) — the shader's `WaterReflect::drift` and `cycle`.
pub const CLOCK_LANES: std::ops::Range<usize> = MIRROR2_LANES.end..MIRROR2_LANES.end + 20;
/// The stylised look's rows, the shader's `WaterReflect::look` (a `WaterLook`): 139 `vec4`s, one
/// writer per row (`benilla_water`'s `look`, and the render world for [`LOOK_WATER_RECT`]).
pub const LOOK_LANES: std::ops::Range<usize> =
    CLOCK_LANES.end..CLOCK_LANES.end + 76 + 4 * RIPPLE_LOSS_ROWS;
/// `quality`: the tier (a copy), the sky floor's occlusion.
pub const LOOK_QUALITY: usize = LOOK_LANES.start;
/// `composite`: the grazing ceiling, the linear glint weight, a forced glow excess.
pub const LOOK_COMPOSITE: usize = LOOK_LANES.start + 4;
/// `water_rect`: the water's screen rectangle, written in the render world.
pub const LOOK_WATER_RECT: usize = LOOK_LANES.start + 8;
/// `surf`: the calm roughness, the ripple's anisotropy clamp, its feature fade.
pub const LOOK_SURF: usize = LOOK_LANES.start + 12;
/// `grade`: the reflection grading's strength and shares.
pub const LOOK_GRADE: usize = LOOK_LANES.start + 16;
/// `detail`: the painted detail's contrast.
pub const LOOK_DETAIL: usize = LOOK_LANES.start + 20;
/// `ripple_loss`: the slope the ripple octaves lose over the pixel's footprint, one row per eighth
/// of a LOD ([`RIPPLE_LOSS_ROWS`]).
pub const LOOK_RIPPLE_LOSS: usize = LOOK_LANES.start + 24;
/// The rows of `ripple_loss` (`liquid.wgsl`'s `RIPPLE_LOSS_ROWS`).
pub const RIPPLE_LOSS_ROWS: usize = 120;
/// `ripple_mean`: the ripple's mean slope, variance and feature wavelength.
pub const LOOK_RIPPLE_MEAN: usize = LOOK_RIPPLE_LOSS + 4 * RIPPLE_LOSS_ROWS;
/// `body`: the ten rows of a `WaterBody`.
pub const LOOK_BODY: usize = LOOK_RIPPLE_MEAN + 4;
/// `shore`: the wet line, wet darkening, contact foam and meniscus gains.
pub const LOOK_SHORE: usize = LOOK_BODY + 40;
/// `shore_kind`: the wet line per kind of water, and the shore lap's phase.
pub const LOOK_SHORE_KIND: usize = LOOK_SHORE + 4;
/// The Improved Water preview's rows, the shader's `WaterReflect::preview` (a `WaterPreview`):
/// [`PREVIEW_ROWS`] `vec4`s, zero where the preview or a feature of it is off.
pub const PREVIEW_LANES: std::ops::Range<usize> = LOOK_LANES.end..LOOK_LANES.end + 4 * PREVIEW_ROWS;
/// The preview block's rows.
pub const PREVIEW_ROWS: usize = 51 + 3 * S2_TRAINS + 3 * S2_BANDS;
/// `mask`: x the feature bits (`benilla_assets::preview`), y the rough reflection's cone share,
/// z 1 − the rivers' slow tier's gain (`WOW_PREVIEW_RIVER_SLOW`; zero runs it), w 1 while the
/// dispersive surface draws (`WOW_PREVIEW_SURFACE2`).
pub const PV_MASK: usize = PREVIEW_LANES.start;
/// `gains`: x caustics, y lights, z sparkles, w river foam.
pub const PV_GAINS: usize = PV_MASK + 4;
/// `clocks`: phases in [0, 1): x river, y weather, z wake; w spare.
pub const PV_CLOCKS: usize = PV_MASK + 8;
/// `wind`: xy the to-direction (unit), z the speed in yd/s with the storm in it, w the storm.
pub const PV_WIND: usize = PV_MASK + 12;
/// `weather`: x rain, y snow, z the sea state 0..1, w the river mist's gain.
pub const PV_WEATHER: usize = PV_MASK + 16;
/// `swell`: xyz the three swell trains' phases, w the glint's own sun reach (0: `sun.w`).
pub const PV_SWELL: usize = PV_MASK + 20;
/// `gust`: xy the gust field's offset, zw the slick field's, in their tiles.
pub const PV_GUST: usize = PV_MASK + 24;
/// `cap`: xy the capillary octave's drift in its tiles, z the band the dispersive surface thins
/// from at Medium (1-based; 0 the shader's default), w spare.
pub const PV_CAP: usize = PV_MASK + 28;
/// `drift`: the five ripple octaves' downwind drifts under `waves`, (d0, d1), (d2, d3), (d4, –).
pub const PV_DRIFT: usize = PV_MASK + 32;
/// `falls`: eight fall lines, two rows each.
pub const PV_FALLS: usize = PV_MASK + 44;
/// `under`: the view from below the surface.
pub const PV_UNDER: usize = PV_MASK + 108;
/// `looks`: the owner's look switches, each lane 1 where its switch is off (so an unwritten lane
/// is on): x the soft shoreline, y the lively reflection, z the surface height, w spare.
pub const PV_LOOKS: usize = PV_MASK + 172;
/// `spare`: room for shader-only iteration.
pub const PV_SPARE: usize = PV_MASK + 176;
/// `trains`: the dispersive surface's wave trains (`benilla_water`'s `surface2`), three rows each:
/// (k·d / 2π in cycles a yard, the phase in cycles, the wind sea's slope amplitude), (the swell's
/// slope amplitude, Q·λ/2π of the orbital displacement, the Stokes share, –), (the direction, 1/λ²,
/// 1 + St²).
pub const PV_TRAINS: usize = PV_MASK + 204;
/// `bands`: the trains' bands, longest first, three rows each: the band's slope variance sums at
/// unit gains (wind sea², 2·wind sea·swell, swell², the kind gain's share), (its longest wavelength
/// squared, its first train, its trains, –), and its direction tensor (xx, zz, xz, –) of trace 1.
pub const PV_BANDS: usize = PV_TRAINS + 4 * 3 * S2_TRAINS;
/// The dispersive surface's trains and bands (`liquid.wgsl`'s `S2_TRAINS`, `S2_BANDS`).
pub const S2_TRAINS: usize = 40;
pub const S2_BANDS: usize = 9;
/// The reflection block's length in floats — `liquid.wgsl`'s `WaterReflect`.
pub const REFLECT_FLOATS: usize = PREVIEW_LANES.end;

/// The pass targets and the reflection block every liquid material binds. The reference water
/// never samples them; the Improved Water plugin publishes its own in [`WaterPassSet`], and
/// re-points the materials itself when it rebuilds one.
#[derive(Resource, Clone)]
pub struct WaterPassImages {
    /// The first planar mirror's capture.
    pub reflection: Handle<Image>,
    /// The second planar mirror's capture.
    pub reflection2: Handle<Image>,
    /// The two mirrors' roughness chains (the preview's rough reflections); the placeholder is one
    /// texel, which the shader reads as no chain.
    pub reflection_chain: Handle<Image>,
    pub reflection_chain2: Handle<Image>,
    /// The scene colour snapshot the march reads.
    pub scene_color: Handle<Image>,
    /// The probe cube array.
    pub probe: Handle<Image>,
    /// The nearest-depth pyramid.
    pub hiz: Handle<Image>,
    /// The farthest-depth pyramid.
    pub hiz_far: Handle<Image>,
    /// The opaque world's linear depth behind each pixel: view Z in yards, `R32Float` at the view's
    /// size, 0 where nothing was drawn; what the stylised lane measures its water column against.
    /// Written inside the water's screen rectangle alone; a texel outside it holds what an earlier
    /// frame wrote there, or 0 where none has since the image was made (at boot or a resize). The
    /// placeholder is one texel, which the shader reads as unknown.
    pub scene_depth: Handle<Image>,
    /// The sky floor's occlusion over the mirror planes and the occluder's colour, a quarter of the
    /// view's resolution. The placeholders read as the sky open, the verdict unknown and no
    /// occluder.
    pub water_occl: Handle<Image>,
    pub floor_occ: Handle<Image>,
    /// The sky floor's cube; the placeholder is black.
    pub sky_lut: Handle<Image>,
    /// The lighting sun's terrain shadow (the ADT chunks' MCSH) around the camera, `R8Unorm`, 1
    /// shadowed; the placeholder is one lit texel.
    pub shadow: Handle<Image>,
    /// The wave simulation's field.
    pub wake: Handle<Image>,
    /// The reflection block, [`REFLECT_FLOATS`] floats, written once a frame in the render world.
    pub reflect_buf: bevy::render::render_resource::Buffer,
    /// The preview glow's accumulator; the placeholder is 16 bytes, which the shader's length test
    /// writes nothing into.
    pub glow_acc: bevy::render::render_resource::Buffer,
}

/// The variants a water plugin asks the stylised liquid pipelines to compile; all off without one.
/// Every stylised material carries [`Self::bits`] in `path.w`, which the material's
/// [`LiquidKey`](benilla_assets::materials::LiquidKey) reads back, so a variant is a key and never a
/// branch in the shipped shader.
#[derive(Resource, Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct WaterPipelineCaps {
    /// The debug views (`WATER_VIEW`), while `WaterDebug::view` names one.
    pub view: bool,
    /// The quality tier ([`super::WaterQuality::tier`]), 0 Low, 1 Medium, 2 High; a larger value
    /// is High.
    pub tier: u8,
    /// The highlight excess stored for the glow.
    pub glow: bool,
    /// Per-channel transmittance through dual-source blending.
    pub dual_source: bool,
    /// The authored sheet as painted detail.
    pub painted: bool,
}

impl WaterPipelineCaps {
    /// The `path.w` integer: the key's cap bits shifted down by its `CAPS_SHIFT`, built from the
    /// key's own consts so the two cannot drift.
    pub fn bits(self) -> u32 {
        use benilla_assets::materials::LiquidKey as K;
        let mut k: u16 = u16::from(self.tier.min(2)) << K::TIER_SHIFT;
        for (on, bit) in [
            (self.view, K::VIEW),
            (self.glow, K::GLOW),
            (self.dual_source, K::DUAL_SOURCE),
            (self.painted, K::PAINTED),
        ] {
            if on {
                k |= bit;
            }
        }
        u32::from(k >> K::CAPS_SHIFT)
    }
}

/// The ripple map's anisotropic filtering clamp, 1 for none. A water plugin sets it from the
/// quality tier; the liquid plugin puts it on the map's sampler, and the plugin's look rows hand it
/// to the shader, whose roughness then counts the probes the hardware takes.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug)]
pub struct RippleAniso(pub u16);

impl Default for RippleAniso {
    fn default() -> Self {
        Self(1)
    }
}

/// Where a water plugin publishes [`WaterPassImages`]; the liquid materials are built after it.
#[derive(bevy::ecs::schedule::SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WaterPassSet;

/// A one-texel image of `format`, `layers` deep, viewed as `view`, zero.
fn placeholder(format: TextureFormat, layers: u32, view: Option<TextureViewDimension>) -> Image {
    let texel = vec![0u8; format.block_copy_size(None).unwrap_or(4) as usize];
    placeholder_filled(format, &texel, layers, view)
}

/// A one-texel image of `format` holding `texel`, `layers` deep, viewed as `view`.
fn placeholder_filled(
    format: TextureFormat,
    texel: &[u8],
    layers: u32,
    view: Option<TextureViewDimension>,
) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: layers,
        },
        TextureDimension::D2,
        texel,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_view_descriptor = view.map(|dimension| TextureViewDescriptor {
        dimension: Some(dimension),
        ..default()
    });
    image
}

/// Publish placeholder pass images when no water plugin has: one texel each, typed as the bindings
/// declare them, and a zeroed reflection block the shader reads as every tier off.
pub(super) fn placeholder_pass_images(
    mut commands: Commands,
    device: Option<Res<RenderDevice>>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(device) = device else {
        return;
    };
    let flat = images.add(placeholder(TextureFormat::Rgba16Float, 1, None));
    let depth = images.add(placeholder(TextureFormat::R32Float, 1, None));
    let cube = images.add(placeholder(
        TextureFormat::Rgba16Float,
        6,
        Some(TextureViewDimension::CubeArray),
    ));
    let sky_cube = images.add(placeholder(
        TextureFormat::Rgba16Float,
        6,
        Some(TextureViewDimension::Cube),
    ));
    // The sky open (r, g, b) and the verdict unknown (a).
    let occl = images.add(placeholder_filled(
        TextureFormat::Rgba8Unorm,
        &[255, 255, 255, 0],
        1,
        None,
    ));
    let shadow = images.add(placeholder(TextureFormat::R8Unorm, 1, None));
    let reflect_buf = device.create_buffer(&BufferDescriptor {
        label: Some("water_reflect_params"),
        size: (REFLECT_FLOATS * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let glow_acc = device.create_buffer(&BufferDescriptor {
        label: Some("water_glow_acc_placeholder"),
        size: 16,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    commands.insert_resource(WaterPassImages {
        reflection: flat.clone(),
        reflection2: flat.clone(),
        reflection_chain: flat.clone(),
        reflection_chain2: flat.clone(),
        scene_color: flat.clone(),
        probe: cube,
        hiz: depth.clone(),
        hiz_far: depth.clone(),
        scene_depth: depth,
        water_occl: occl,
        floor_occ: flat.clone(),
        sky_lut: sky_cube,
        shadow,
        wake: flat,
        reflect_buf,
        glow_acc,
    });
}

#[cfg(test)]
mod tests {
    use super::WaterPipelineCaps;
    use benilla_assets::materials::LiquidKey;
    use bevy::math::Vec4;

    /// Every caps combination round-trips through `path.w` into the key's matching bits and tier,
    /// on the stylised lane only; magma and slime stay fullbright whatever `path.w` holds.
    #[test]
    fn the_caps_round_trip_through_path_w() {
        for bools in 0u8..16 {
            for tier in 0..=2 {
                let on = |i: u8| bools & (1 << i) != 0;
                let caps = WaterPipelineCaps {
                    view: on(0),
                    tier,
                    glow: on(1),
                    dual_source: on(2),
                    painted: on(3),
                };
                let lane = caps.bits() as f32;
                let water = LiquidKey::from_lanes(Vec4::ZERO, Vec4::new(0.0, 1.0, 0.0, lane));
                assert!(water.composite());
                assert_eq!(water.tier(), tier, "{caps:?}");
                for (want, bit) in [
                    (caps.view, LiquidKey::VIEW),
                    (caps.glow, LiquidKey::GLOW),
                    (caps.dual_source, LiquidKey::DUAL_SOURCE),
                    (caps.painted, LiquidKey::PAINTED),
                ] {
                    assert_eq!(water.has(bit), want, "{caps:?} bit {bit:#x}");
                }
                let reference = LiquidKey::from_lanes(Vec4::ZERO, Vec4::new(0.0, 0.0, 0.0, lane));
                assert_eq!(
                    reference,
                    LiquidKey::default(),
                    "{caps:?} on the reference lane"
                );
                let magma = LiquidKey::from_lanes(Vec4::X, Vec4::new(0.0, 1.0, 1.0, lane));
                assert!(magma.has(LiquidKey::FULLBRIGHT) && !magma.composite());
                assert_eq!(magma, LiquidKey::from_lanes(Vec4::X, Vec4::ZERO));
            }
        }
        // A tier past High clamps rather than spilling into the next bit.
        let past = WaterPipelineCaps {
            tier: 3,
            ..Default::default()
        };
        let key = LiquidKey::from_lanes(Vec4::ZERO, Vec4::new(0.0, 1.0, 0.0, past.bits() as f32));
        assert_eq!(key.tier(), 2);
        assert!(!key.has(LiquidKey::GLOW));
    }
}
