//! **Improved Water**, benilla's own water look, as a plugin on the engine's liquid seam
//! (`benilla_world::liquid::hooks`): the whole-map classification of which water the planar
//! mirrors serve and where the fixed cube probes stand (`liquid_planar`, cached by `water_map`),
//! the two planar mirrors (`reflect`), the probes (`probe`), the depth pyramid and scene colour
//! the screen-space march reads (`depth`, `hiz`, `scene_color`), the sky floor's occlusion over the
//! mirror planes and the sky floor's cube (`water_compute`, `sky_lut`), whether the water on screen
//! can show the sky floor at all (`floor_share`), the wave simulation (`ripple_sim`), the surface's
//! clock (`clock`), the body's optics and light (`body`), the terrain shadow's window (`mcsh`)
//! and the look's knobs (`look`). The surface shading
//! is `liquid.wgsl`'s stylised lane, in the engine.
//!
//! Every pass is inert on the reference lane (`WaterStyle::Reference`); the plugin only makes
//! the Improved Water option draw what it names.

use std::path::PathBuf;

use bevy::prelude::*;

pub mod body;
mod clock;
mod depth;
mod floor_share;
mod flow;
mod hiz;
mod liquid_planar;
mod look;
mod mcsh;
mod probe;
mod reflect;
mod ripple_sim;
mod scene_color;
mod sky_lut;
mod water_compute;
mod water_map;
mod world_view;

pub use liquid_planar::PlanarMap;

// The engine items the moved passes name by their old paths.
use benilla_world::liquid::{
    RippleAniso, WaterChunkInfo, WaterIndex, WaterMapRef, WaterPassImages, WaterPassSet,
    WaterPipelineCaps, WaterQuality, WaterStyle, UNMIRRORED_RENDER_LAYER,
};
use benilla_world::{dev_state, lighting, sky, sun, view, water_fx, world_unit};

/// The Improved Water plugin. Add it beside the engine's `WorldPlugins`; `cache_dir` is where each
/// map's classification is cached (the state folder), `None` to keep nothing.
pub struct ImprovedWaterPlugin {
    /// The classification cache's folder.
    pub cache_dir: Option<PathBuf>,
}

impl Plugin for ImprovedWaterPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(benilla_world::liquid::ImprovedWaterPresent);
        water_map::set_state_dir(self.cache_dir.clone());
        benilla_assets::set_water_classifier(benilla_assets::WaterClassifier {
            map: water_map::water_map,
            batch: liquid_planar::classify_batch,
        });
        bevy::asset::embedded_asset!(app, "shaders/probe_face.wgsl");
        bevy::asset::embedded_asset!(app, "shaders/probe_filter.wgsl");
        bevy::asset::embedded_asset!(app, "shaders/hiz_reduce.wgsl");
        bevy::asset::embedded_asset!(app, "shaders/hiz_seed.wgsl");
        bevy::asset::embedded_asset!(app, "shaders/water_compute.wgsl");
        bevy::asset::embedded_asset!(app, "shaders/sky_lut.wgsl");
        bevy::asset::embedded_asset!(app, "shaders/scene_snapshot.wgsl");
        // The surface's clock, the mirrors' block and targets, then the wave simulation, the depth
        // passes and the probes, each pass inert unless the Improved Water look is selected.
        clock::register(app);
        app.init_resource::<look::WaterLook>()
            .init_resource::<mcsh::WaterShadowWindow>();
        reflect::register(app);
        ripple_sim::register(app);
        world_view::register(app);
        hiz::register(app);
        scene_color::register(app);
        sky_lut::register(app);
        water_compute::register(app);
        floor_share::register(app);
        depth::register(app);
        probe::register(app);
        app.add_systems(
            Startup,
            publish_pass_images
                .in_set(WaterPassSet)
                .after(benilla_assets::AssetSet::Open),
        );
        register_caps(app);
    }
}

/// [`drive_pipeline_caps`] in `PreUpdate`, so the materials take a changed variant in the same
/// frame's `Update`, on the frames a switch it reads changed.
fn register_caps(app: &mut App) {
    app.add_systems(
        PreUpdate,
        drive_pipeline_caps
            .run_if(resource_changed::<dev_state::DebugState>.or(resource_changed::<WaterQuality>)),
    );
}

/// Ask the stylised liquid pipelines for the variants this plugin's switches need and the ripple
/// map for the tier's anisotropy: the quality tier, the debug views while one is chosen, the
/// screen-space march while its switch is on. A no-op unless the answer changes, so the materials
/// are touched only then. The depth pyramid's readers are asked each frame
/// (`floor_share::drive_floor_readers`).
fn drive_pipeline_caps(
    dev: Res<dev_state::DebugState>,
    quality: Res<WaterQuality>,
    mut caps: ResMut<WaterPipelineCaps>,
    mut aniso: ResMut<RippleAniso>,
) {
    let next = WaterPipelineCaps {
        view: dev.water.view != 0,
        tier: quality.tier(),
        ssr: dev.water.ssr,
        ..*caps
    };
    caps.set_if_neq(next);
    aniso.set_if_neq(RippleAniso(look::ripple_aniso(quality.tier())));
}

/// Hand the engine's liquid materials this plugin's targets; each pass re-points them itself when
/// it rebuilds one.
fn publish_pass_images(
    mut commands: Commands,
    reflect: Option<Res<reflect::WaterReflect>>,
    reflect_buf: Option<Res<reflect::WaterReflectBuffer>>,
    scene_color: Option<Res<scene_color::WaterSceneColor>>,
    probe: Option<Res<probe::WaterProbe>>,
    hiz: Option<Res<hiz::WaterHiz>>,
    occ: Option<Res<water_compute::FloorOcc>>,
    lut: Option<Res<sky_lut::SkyLut>>,
    sim: Option<Res<ripple_sim::RippleSim>>,
) {
    let (
        Some(reflect),
        Some(reflect_buf),
        Some(scene_color),
        Some(probe),
        Some(hiz),
        Some(occ),
        Some(lut),
        Some(sim),
    ) = (reflect, reflect_buf, scene_color, probe, hiz, occ, lut, sim)
    else {
        return;
    };
    commands.insert_resource(WaterPassImages {
        reflection: reflect.image.clone(),
        reflection2: reflect.image2.clone(),
        scene_color: scene_color.image.clone(),
        probe: probe.cube.clone(),
        hiz: hiz.image.clone(),
        hiz_far: hiz.far.clone(),
        scene_depth: hiz.lin.clone(),
        water_occl: occ.occl.clone(),
        floor_occ: occ.col.clone(),
        sky_lut: lut.image.clone(),
        wake: sim.image.clone(),
        reflect_buf: reflect_buf.0.clone(),
    });
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;
    use bevy::prelude::*;

    use benilla_assets::materials::LiquidKey;

    /// Each quality tier reaches the stylised pipeline key through the caps' `path.w`, and a switch
    /// leaves the other variants where they were.
    #[test]
    fn the_quality_is_the_stylised_keys_tier() {
        use super::WaterQuality::{High, Low, Medium};
        let mut world = World::new();
        world.init_resource::<super::dev_state::DebugState>();
        world.init_resource::<super::WaterPipelineCaps>();
        world.init_resource::<super::RippleAniso>();
        world
            .resource_mut::<super::dev_state::DebugState>()
            .water
            .ssr = true;
        for quality in [High, Low, Medium] {
            world.insert_resource(quality);
            world.run_system_once(super::drive_pipeline_caps).unwrap();
            let caps = *world.resource::<super::WaterPipelineCaps>();
            let key =
                LiquidKey::from_lanes(Vec4::ZERO, Vec4::new(0.0, 1.0, 0.0, caps.bits() as f32));
            assert_eq!(key.tier(), quality.tier(), "{quality:?}");
            assert!(
                key.has(LiquidKey::SSR),
                "{quality:?}: the march's variant stays"
            );
            assert_eq!(
                world.resource::<super::RippleAniso>().0,
                super::look::ripple_aniso(quality.tier()),
                "{quality:?}: the ripple map's anisotropy follows the tier"
            );
        }
    }

    /// **A `waterQuality` change reaches the key on its frame**, through the plugin's own schedule
    /// and run condition, which `run_system_once` skips; a frame on which no switch changed leaves
    /// the caps alone.
    #[test]
    fn a_quality_change_reaches_the_caps_through_the_schedule() {
        use super::WaterQuality::{High, Low, Medium};
        let mut app = App::new();
        app.init_resource::<super::dev_state::DebugState>()
            .init_resource::<super::WaterPipelineCaps>()
            .init_resource::<super::RippleAniso>()
            .insert_resource(Low);
        super::register_caps(&mut app);
        let tier = |app: &App| app.world().resource::<super::WaterPipelineCaps>().tier;
        // Each condition counts changes since it was last evaluated, and `or` evaluates one only
        // when those before it are false: two frames let each of the two see the state once.
        for _ in 0..2 {
            app.update();
        }
        assert_eq!(tier(&app), Low.tier(), "the boot tier");
        let set = |app: &mut App, t: u8| {
            app.world_mut()
                .resource_mut::<super::WaterPipelineCaps>()
                .tier = t;
        };
        set(&mut app, High.tier());
        app.update();
        assert_eq!(
            tier(&app),
            High.tier(),
            "no switch changed, so the caps stand"
        );
        app.insert_resource(Medium);
        app.update();
        assert_eq!(tier(&app), Medium.tier(), "a new tier reaches the caps");
    }
}
