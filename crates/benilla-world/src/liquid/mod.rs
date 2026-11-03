//! Liquid: the animated surfaces (`surface`), where the liquid is and whether a subject or the
//! camera eye is in it (`query`, indexed by `spatial`), and the underwater drift cloud (`drift`).
//! The MCLQ parse and flat mesh are `benilla_formats::liquid`; the shading is `liquid.wgsl`.
//!
//! The reference has three liquid renderers, each with its own arm in the shader: ADT MCLQ, WMO
//! MLIQ water (`0x6b62e0` category 0, exterior `0x6b6630` or interior `0x6b6420` by the group's
//! `MOGP.flags & 0x48`: one texture stage, no depth ramp, alpha from a per-vertex authored byte)
//! and magma/slime.
//!
//! The ADT combiner is the asset `Shaders\Pixel\ocean0_s.bls` in `patch.MPQ`:
//! `rgb = primary·colorTex.rgb + detailTex.rgb + (secondary + 0.25)·detailTex.a`,
//! `alpha = colorTex.a`. `primary` is the lit vertex colour `clamp(ambient + N·L·sun)`;
//! `detailTex` is the animated `lake_a`/`ocean_h` frame, near-black RGB and a ripple alpha.
//! `colorTex` is a 64-row swatch, RGB between the zone's raw `Light.dbc` water rows (IntBand 16/17
//! river/lake, 14/15 ocean) and alpha between its `LightParams` shallow and deep alphas, filled by
//! `0x68a830` as `row(i) = c0 + floor(i·(c1−c0)/64)`, so it never reaches the deep endpoint. The
//! ocean's last row alone is darkened by an HSV `V ×= 0.9` and forced opaque; about 80% of ocean
//! vertices sample it. The dirty flags `[0xc8117c]`/`[0xc81b70]` clear every world frame
//! (`0x680b90`/`0x680b97`, refill `0x58acd0`), so colour and opacity track the zone and the clock.
//!
//! One per-vertex depth `V` indexes colour and alpha alike: `min(byte/42, 1)` on river/lake
//! (`0xc81768`, read at `0x68d818` by `0x68d790`), `min(byte/255, 1)` on ocean (`0xc7fcd8`, read
//! at `0x68d718` by `0x68d690`), both tables built by `0x68c4c0`. `0xc7fbc0`'s `1.6·(i/63)^8` is no
//! water alpha: it fills the sky glare texture `[0xc7fbb8]` (`0x68c250` via `0x68c4a0`), whose one
//! liquid binding, `0x685257`, sits in a branch that never runs (`[0xc800ec]` is only ever 0).

use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;

use benilla_assets::materials::LiquidMaterial;
use benilla_assets::AssetSet;

mod drift;
mod hooks;
mod query;
#[cfg(test)]
mod real_data;
mod ripple;
mod spatial;
mod surface;
pub(crate) use hooks::world_camera_layers;
pub use hooks::{
    mirror_view_shape, ProbeCull, ProbeFace, ProbeMark, ProbeMarks, ReflectionCamera, RippleAniso,
    WaterPassImages, WaterPassSet, WaterPipelineCaps, CLOCK_LANES, DOME_LANES, LOOK_BODY,
    LOOK_COMPOSITE, LOOK_DETAIL, LOOK_GRADE, LOOK_LANES, LOOK_QUALITY, LOOK_RIPPLE_LOSS,
    LOOK_RIPPLE_MEAN, LOOK_SHORE, LOOK_SHORE_KIND, LOOK_SURF, LOOK_WATER_RECT, MARCH_LANES,
    MIRROR2_LANES, PREVIEW_LANES, PREVIEW_ROWS, PROBE_LANES, PROBE_SLOT_MAX, PV_BANDS, PV_CAP,
    PV_CLOCKS, PV_DRIFT, PV_FALLS, PV_GAINS, PV_GUST, PV_LOOKS, PV_MASK, PV_SPARE, PV_SWELL,
    PV_TRAINS, PV_UNDER, PV_WEATHER, PV_WIND, REFLECT_FLOATS, RIPPLE_LOSS_ROWS, S2_BANDS,
    S2_TRAINS, UNMIRRORED_RENDER_LAYER, WATER_RENDER_LAYER,
};

// The submodules are private: this list is everything the rest of the client may name.
/// **Which water look the client draws** — the `waterStyle` CVar's knob, and the only setting in
/// this subsystem that is a matter of taste rather than of fidelity.
///
/// [`WaterStyle::Reference`] is the default and is what every other line under `liquid/` is about:
/// the 1.12 client's own combine, its swatch, its sheet, its opacity. [`WaterStyle::Stylised`] is
/// benilla's own — an animated surface normal, a specular glitter path, a Fresnel sky mix and shore
/// foam, ported from the `water-test` sandbox and described in `liquid.wgsl`'s `stylised_water`. It
/// keeps the zone's colours, depth and lighting and changes only what is done with them, so it
/// follows the world's day/night and its zones exactly as the faithful path does.
///
/// The switch rewrites lanes and reloads nothing: [`surface::apply_water_style`] rewrites `path.y`
/// (and `path.w`, [`WaterPipelineCaps::bits`]) on the handful of shared liquid materials when this
/// resource changes. The lanes are the materials' pipeline key (`LiquidKey`), so a flip moves every
/// liquid onto the other lane's pipelines; the world camera's view key stays the same on both
/// lanes (`benilla_water::depth`).
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug)]
pub enum WaterStyle {
    /// The 1.12 client's water, which is what the rest of this subsystem implements.
    Reference,
    /// benilla's Improved Water: the stylised surface, planar mirrors over flat water and fixed
    /// cube probes over falls and small pools (the `benilla-water` plugin), with the
    /// screen-space march a debug option.
    Stylised,
}

/// `$WOW_WATER_STYLE=1` boots into Improved Water — the lever for the world viewer, which has no
/// CVar host, and for captures. Session-only: `cvars` marks the row env-overridden so a run cannot
/// pin itself into `config.toml`.
impl Default for WaterStyle {
    fn default() -> Self {
        match std::env::var("WOW_WATER_STYLE").as_deref() {
            Ok("0") | Err(_) => Self::Reference,
            Ok(_) => Self::Stylised,
        }
    }
}

impl WaterStyle {
    /// The CVar's parse: 0 is the reference water, anything else Improved Water.
    pub fn from_cvar(v: f32) -> Self {
        if v != 0.0 {
            Self::Stylised
        } else {
            Self::Reference
        }
    }

    /// Is this benilla's own look, as opposed to the 1.12 client's?
    pub fn is_stylised(self) -> bool {
        self == Self::Stylised
    }

    /// Does this look drive the planar reflection cameras?
    pub fn wants_mirror(self) -> bool {
        self.is_stylised()
    }

    /// Does this look drive the cube probes?
    pub fn wants_probe(self) -> bool {
        self.is_stylised()
    }

    /// What `GetCVar("waterStyle")` answers for this state.
    pub fn cvar(self) -> &'static str {
        match self {
            Self::Reference => "0",
            Self::Stylised => "1",
        }
    }

    /// The shader lane (`LiquidParams.path.y`).
    pub fn shader_flag(self) -> f32 {
        f32::from(self.is_stylised())
    }
}

/// **How much Improved Water spends** — the `waterQuality` CVar's knob, benilla's own like
/// [`WaterStyle`]: the reference has one water and no quality dial for it. It reaches the stylised
/// pipelines as their key's tier (`LiquidKey::tier`, one `WATER_TIER_*` def each), so a switch
/// moves the materials onto that tier's pipeline, one of the keys the pipeline warm pass compiles
/// (`LiquidKey::shipped`); the reference lane, magma and slime never read it.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug)]
pub enum WaterQuality {
    Low,
    /// The default.
    Medium,
    High,
}

/// `$WOW_WATER_QUALITY=<0|1|2>` (or `low`, `medium`, `high`) boots at that tier — the lever the
/// golden water scenes pin Medium with; anything else is Medium. Session-only, like
/// `$WOW_WATER_STYLE`.
impl Default for WaterQuality {
    fn default() -> Self {
        std::env::var("WOW_WATER_QUALITY")
            .ok()
            .and_then(|v| Self::parse(&v))
            .unwrap_or(Self::Medium)
    }
}

impl WaterQuality {
    /// The CVar's parse: the integer part, below 1 Low and from 2 High.
    pub fn from_cvar(v: f32) -> Self {
        if v < 1.0 {
            Self::Low
        } else if v < 2.0 {
            Self::Medium
        } else {
            Self::High
        }
    }

    /// A tier by its number or its name, any case; `None` for anything else.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        match s.to_ascii_lowercase().as_str() {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => s.parse::<f32>().ok().map(Self::from_cvar),
        }
    }

    /// What `GetCVar("waterQuality")` answers for this state.
    pub fn cvar(self) -> &'static str {
        match self {
            Self::Low => "0",
            Self::Medium => "1",
            Self::High => "2",
        }
    }

    /// The pipeline key's tier ([`WaterPipelineCaps::tier`]): 0 Low, 1 Medium, 2 High.
    pub fn tier(self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Medium => 1,
            Self::High => 2,
        }
    }
}

pub use query::{
    camera_claim, describe_at, liquid_at, player_claim, surfaces_at, unit_claim, water_surface_at,
    EyeLiquid, FoamPatch, LiquidClaim, LiquidHit, LiquidSource, RoomPlacements, SubmergedEye,
    Underwater, WaterChunkInfo, WmoPool,
};
pub use ripple::{RippleStats, SIZE as RIPPLE_SIZE};
pub(crate) use spatial::maintain_water_index;
pub use spatial::WaterIndex;
pub(crate) use surface::{
    spawn_liquids, spawn_wmo_liquids, LiquidAssets, LiquidSoundSource, WetLattice,
};
pub use surface::{warm_liquid_mesh, LiquidSurface, RippleMap, WaterMapRef};

/// Present when a plugin draws Improved Water (`benilla-water`); without it that option draws the
/// reference, since the stylised lane alone has no reflection tiers to read.
#[derive(Resource)]
pub struct ImprovedWaterPresent;

/// Hold [`WaterStyle::Reference`] while no plugin draws Improved Water.
fn hold_reference_without_plugin(mut style: ResMut<WaterStyle>) {
    style.set_if_neq(WaterStyle::Reference);
}

/// `WOW_FORCE_SUB=<frames>`: hold the camera-eye verdict submerged for the first `<frames>` frames,
/// then release it, so the wet-to-dry crossing lands on a known frame with no water or server.
fn forced_submersion_frames() -> u32 {
    std::env::var("WOW_FORCE_SUB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// The [`forced_submersion_frames`] override, run after the real probe so it overwrites the
/// verdict rather than racing it; logs the release frame.
fn force_submersion(mut underwater: ResMut<Underwater>, mut frame: Local<u32>) {
    let hold = forced_submersion_frames();
    *frame += 1;
    if *frame <= hold {
        underwater.0 = benilla_formats::Submersion::Water;
    } else if *frame == hold + 1 {
        info!(
            "WOW_FORCE_SUB: frame {} — eye verdict released to Dry",
            *frame
        );
    }
}

/// The set that writes [`Underwater`]; every consumer of the verdict runs after it, since a
/// frame-old verdict shows one mixed frame (murk with the sky still up) on every surface crossing.
#[derive(bevy::ecs::schedule::SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SubmersionVerdict;

/// The liquid subsystem: the shared materials at startup, the submersion verdict and the drift
/// cloud each frame. The terrain streamer spawns the surfaces with their tile ([`spawn_liquids`]).
pub(crate) struct LiquidPlugin;

impl Plugin for LiquidPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<LiquidMaterial>::default())
            .init_resource::<Underwater>()
            .init_resource::<SubmergedEye>()
            .init_resource::<WaterIndex>()
            .init_resource::<WaterStyle>()
            .init_resource::<WaterQuality>()
            .init_resource::<WaterPipelineCaps>()
            .init_resource::<RippleAniso>()
            .init_resource::<ProbeMarks>()
            // PreUpdate: a surface streamed by last frame's commands is indexed before anyone asks.
            .add_systems(PreUpdate, maintain_water_index)
            .add_systems(
                PreUpdate,
                hold_reference_without_plugin.run_if(
                    not(resource_exists::<ImprovedWaterPresent>)
                        .and(resource_changed::<WaterStyle>),
                ),
            )
            // The pass images: a water plugin's, published in `WaterPassSet`, or placeholders.
            .add_systems(
                Startup,
                (
                    hooks::placeholder_pass_images.run_if(not(resource_exists::<WaterPassImages>)),
                    surface::setup_liquid,
                )
                    .chain()
                    .after(AssetSet::Open)
                    .after(WaterPassSet),
            )
            .add_systems(Update, hooks::stamp_world_camera_layers)
            // The Improved Water's fine ring, for its surface height.
            .add_systems(Update, surface::swap_fine_liquids)
            .add_systems(
                Update,
                (
                    // After the portal pass publishes this frame's `CameraInteriorClaim`: a
                    // frame-old room flashes the wrong atmosphere on every doorway crossing.
                    query::detect_submersion
                        .after(crate::wmo_portal::WmoPvsSet)
                        .in_set(SubmersionVerdict),
                    // The look toggle and the pipeline variants: change-gated, so the steady state
                    // schedules nothing and a flip touches the shared materials once. It runs on
                    // the frame the CVar lands and on the first frame after startup, which is what
                    // carries a persisted `waterStyle` from `config.toml` onto materials built
                    // before it was read.
                    // After the frame's script calls, whose `waterStyle` callback writes it.
                    surface::apply_water_style
                        .after(crate::schedule::WorldStage::Present)
                        .run_if(
                            resource_changed::<WaterStyle>
                                .or(resource_changed::<WaterPipelineCaps>),
                        ),
                    // The ripple map's anisotropy: a change re-uploads the map and rebuilds every
                    // liquid bind group, so it runs on a tier change and once the map exists.
                    surface::apply_ripple_aniso
                        .run_if(resource_changed::<RippleAniso>.or(resource_added::<RippleMap>)),
                ),
            )
            // The `WOW_NO_LIQUID` override: after both per-frame `Visibility` owners of a surface
            // (the exterior cull, the model-visibility authority) and before Bevy reads it.
            .add_systems(
                PostUpdate,
                surface::hide_liquid_surfaces
                    .after(crate::exterior_cull::ExteriorCullSet)
                    .before(bevy::camera::visibility::VisibilitySystems::VisibilityPropagate)
                    .run_if(|| std::env::var_os("WOW_NO_LIQUID").is_some()),
            );
        // Added only when `WOW_FORCE_SUB` names a hold: an ordinary run carries no system for it.
        if forced_submersion_frames() > 0 {
            app.add_systems(
                Update,
                force_submersion
                    .after(query::detect_submersion)
                    .in_set(SubmersionVerdict),
            );
        }
        drift::register(app);
    }

    /// Check the liquid layout against this device's per-stage limits, once the mesh pipeline
    /// exists (`MeshRenderPlugin::finish`).
    fn finish(&self, app: &mut App) {
        check_liquid_limits(app);
    }
}

/// Name, at startup, every per-stage binding limit the liquid pipelines exceed on this device.
/// There is one liquid layout for both looks, so over a limit every liquid pipeline fails, the
/// reference water's with the rest, and its draws drop with only a pipeline error in the log.
///
/// A liquid pipeline binds the two view groups, the mesh group (`model_only`: no skin, morph or
/// lightmap) and the material group, which `StandardMaterial` and `LiquidExt` share because
/// `LiquidExt` is not bindless. Each view key the world camera takes is counted: MSAA on or off
/// (it carries no depth prepass). Headless, it does nothing.
fn check_liquid_limits(app: &App) {
    use benilla_assets::materials::StageCount;
    use bevy::pbr::{MeshPipeline, MeshPipelineViewLayoutKey as ViewKey};
    use bevy::render::render_resource::{AsBindGroup, ShaderStages};
    use bevy::render::renderer::{RenderAdapter, RenderDevice};

    let Some(render_app) = app.get_sub_app(bevy::render::RenderApp) else {
        return;
    };
    let world = render_app.world();
    let (Some(device), Some(adapter), Some(mesh)) = (
        world.get_resource::<RenderDevice>(),
        world.get_resource::<RenderAdapter>(),
        world.get_resource::<MeshPipeline>(),
    ) else {
        warn!("liquid: no mesh pipeline at startup, so the binding-limit check did not run");
        return;
    };
    let material = LiquidMaterial::bind_group_layout_entries(device, false);
    let limits = device.limits();
    let downlevel = adapter.get_downlevel_capabilities().flags;
    let count = |stage, key: ViewKey| {
        let view = &mesh.view_layouts.0[key.bits() as usize];
        StageCount::of(
            stage,
            view.main_layout
                .entries
                .iter()
                .chain(&view.binding_array_layout.entries)
                .chain(&mesh.mesh_layouts.model_only.entries)
                .chain(&material),
        )
    };
    for stage in [ShaderStages::VERTEX, ShaderStages::FRAGMENT] {
        for key in [ViewKey::empty(), ViewKey::MULTISAMPLED] {
            let over = count(stage, key).overruns(stage, &limits, downlevel);
            if !over.is_empty() {
                error!(
                    "liquid: the {stage:?} stage of the liquid layout (view key {key:?}) does not \
                     fit this device: {}. No liquid will draw.",
                    over.join("; ")
                );
            }
        }
    }
    let frag = count(ShaderStages::FRAGMENT, ViewKey::empty());
    info!(
        "liquid: the fragment stage binds {} sampled textures of {}, {} samplers of {}, {} \
         storage buffers of {}",
        frag.sampled_textures,
        limits.max_sampled_textures_per_shader_stage,
        frag.samplers,
        limits.max_samplers_per_shader_stage,
        frag.storage_buffers,
        limits.max_storage_buffers_per_shader_stage,
    );
}

#[cfg(test)]
mod tests {
    use super::WaterQuality::{self, High, Low, Medium};

    /// The CVar's value, its name and its env spelling reach the same tier, and `cvar` reads back
    /// what `from_cvar` took; out-of-range values clamp.
    #[test]
    fn the_quality_parses_as_its_cvar_and_its_lever() {
        for (q, tier) in [(Low, 0), (Medium, 1), (High, 2)] {
            assert_eq!(q.tier(), tier);
            assert_eq!(WaterQuality::from_cvar(q.cvar().parse().unwrap()), q);
            assert_eq!(WaterQuality::parse(q.cvar()), Some(q));
        }
        for (v, q) in [(-1.0, Low), (0.9, Low), (1.5, Medium), (7.0, High)] {
            assert_eq!(WaterQuality::from_cvar(v), q, "{v}");
        }
        for (s, q) in [
            ("low", Low),
            (" MEDIUM ", Medium),
            ("High", High),
            ("2.0", High),
        ] {
            assert_eq!(WaterQuality::parse(s), Some(q), "{s:?}");
        }
        assert_eq!(WaterQuality::parse("ultra"), None);
    }
}
