//! The water passes' WGSL tripwire: each of this crate's shaders composed with naga_oil and
//! validated with naga, as Bevy's shader cache composes it (`benilla-assets`'
//! `tests/support/wgsl.rs`), under the defs of every pipeline the crate queues. A shader that fails
//! here fails its pipeline, and the pass that waits for it never runs: the sky floor's cube stays
//! black, the occlusion stays at its placeholder, the pyramid is never built.
//!
//! And the reflection block's layout as naga lays it out: the rows the CPU writes at
//! `benilla_world::liquid`'s lanes are the members the shader reads there.

#[path = "../../benilla-assets/tests/support/wgsl.rs"]
mod wgsl;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use benilla_assets::materials::LiquidKey;
use benilla_world::liquid::{
    CLOCK_LANES, DOME_LANES, LOOK_BODY, LOOK_COMPOSITE, LOOK_DETAIL, LOOK_GRADE, LOOK_LANES,
    LOOK_QUALITY, LOOK_RIPPLE_LOSS, LOOK_RIPPLE_MEAN, LOOK_SHORE, LOOK_SHORE_KIND, LOOK_SURF,
    LOOK_WATER_RECT, MARCH_LANES, MIRROR2_LANES, PROBE_LANES, REFLECT_FLOATS,
};
use naga_oil::compose::ShaderDefValue;

/// A flag def.
fn flag(name: &str) -> (String, ShaderDefValue) {
    (name.into(), ShaderDefValue::Bool(true))
}

/// A value def.
fn value(name: &str, v: u32) -> (String, ShaderDefValue) {
    (name.into(), ShaderDefValue::UInt(v))
}

/// Every pipeline this crate queues, as (shader, defs): `water_compute`'s floor occlusion at each
/// tier (`init_compute_pipelines`), the sky cube's two entry points (`init_lut_pipelines`), the
/// pyramid's seed single- and multisampled (`HizPipelines::queue_seed`, at 4 samples) and its
/// reduce, the scene snapshot's blit, and the probe's face blit and filter, which push no defs.
fn pipelines() -> Vec<(&'static str, Vec<(String, ShaderDefValue)>)> {
    let mut out = Vec::new();
    for tier in LiquidKey::TIER_DEFS {
        out.push((
            "water_compute.wgsl",
            vec![
                flag(tier),
                value("PARAMS_ROW", 0),
                value("MIRROR2_ROW", (MIRROR2_LANES.start / 4) as u32),
            ],
        ));
    }
    out.push((
        "sky_lut.wgsl",
        vec![value("DOME_ROW", (DOME_LANES.start / 4) as u32)],
    ));
    out.push(("hiz_seed.wgsl", vec![]));
    out.push((
        "hiz_seed.wgsl",
        vec![flag("SEED_MS"), value("SEED_SAMPLES", 4)],
    ));
    for shader in [
        "hiz_reduce.wgsl",
        "scene_snapshot.wgsl",
        "probe_face.wgsl",
        "probe_filter.wgsl",
    ] {
        out.push((shader, vec![]));
    }
    out
}

#[test]
fn the_water_passes_compile_under_every_def_set() {
    let Some(modules) = wgsl::bevy_modules() else {
        return;
    };
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
    let mut failures = Vec::new();
    let mut shaders: Vec<&str> = Vec::new();
    for (shader, defs) in pipelines() {
        let path: PathBuf = dir.join(shader);
        let source = std::fs::read_to_string(&path).expect("the shader");
        let defs: HashMap<String, ShaderDefValue> = defs
            .into_iter()
            .chain([(
                wgsl::STORAGE_BINDINGS.0.to_owned(),
                wgsl::STORAGE_BINDINGS.1,
            )])
            .collect();
        let named: Vec<String> = defs.keys().cloned().collect();
        let verdict = wgsl::composer_for(&source, &modules)
            .and_then(|mut c| wgsl::compose(&mut c, &source, &path, defs))
            .and_then(|m| wgsl::validate(&m));
        if let Err(e) = verdict {
            failures.push(format!("{shader} under {named:?}: {e}"));
        }
        if !shaders.contains(&shader) {
            shaders.push(shader);
        }
    }
    let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".wgsl"))
        .collect();
    on_disk.sort();
    let mut covered: Vec<String> = shaders.iter().map(|s| (*s).to_owned()).collect();
    covered.sort();
    assert_eq!(
        covered, on_disk,
        "a shader of this crate is not composed here"
    );
    assert!(
        failures.is_empty(),
        "{} water pipelines fail; their passes never run:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The members of struct `name` in `module`, as (member, byte offset), and its span in bytes.
fn layout(module: &naga::Module, name: &str) -> (Vec<(String, u32)>, u32) {
    let (_, ty) = module
        .types
        .iter()
        .find(|(_, t)| t.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("liquid.wgsl has no struct `{name}`"));
    let naga::TypeInner::Struct { members, span } = &ty.inner else {
        panic!("`{name}` is not a struct");
    };
    let members = members
        .iter()
        .map(|m| (m.name.clone().unwrap_or_default(), m.offset))
        .collect();
    (members, *span)
}

/// **The reflection block is laid out where the CPU writes it.** naga's own layout of
/// `WaterReflect` and `WaterLook`, on a pipeline that compiles them: each row the CPU writes at a
/// `benilla_world::liquid` lane is the member at that lane's byte offset, and the block is
/// [`REFLECT_FLOATS`] floats long.
#[test]
fn the_reflection_block_is_laid_out_where_the_cpu_writes_it() {
    let Some(modules) = wgsl::bevy_modules() else {
        return;
    };
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../benilla-assets/src/shaders/liquid.wgsl");
    let liquid = std::fs::read_to_string(&path).expect("liquid.wgsl");
    let mut composer = wgsl::composer_for(&liquid, &modules).expect("liquid.wgsl's imports");
    let flags = wgsl::LIQUID_LAYOUTS[0].iter().copied();
    let module = wgsl::compose(&mut composer, &liquid, &path, wgsl::liquid_defs(flags))
        .expect("liquid.wgsl composes");
    let at = |lane: usize| (lane * 4) as u32;

    let (reflect, span) = layout(&module, "WaterReflect");
    assert_eq!(span, at(REFLECT_FLOATS), "WaterReflect's size");
    let offset = |m: &str| {
        reflect
            .iter()
            .find(|(n, _)| n == m)
            .unwrap_or_else(|| panic!("WaterReflect has no `{m}`"))
            .1
    };
    for (member, lane) in [
        ("probes", PROBE_LANES.start),
        ("dome", DOME_LANES.start),
        ("march", MARCH_LANES.start),
        ("mirror2", MIRROR2_LANES.start),
        ("drift", CLOCK_LANES.start),
        ("cycle", CLOCK_LANES.start + 16),
        ("look", LOOK_LANES.start),
    ] {
        assert_eq!(offset(member), at(lane), "WaterReflect::{member}");
    }

    let (look, span) = layout(&module, "WaterLook");
    assert_eq!(span, at(LOOK_LANES.len()), "WaterLook's size");
    let rows = [
        ("quality", LOOK_QUALITY),
        ("composite", LOOK_COMPOSITE),
        ("water_rect", LOOK_WATER_RECT),
        ("surf", LOOK_SURF),
        ("grade", LOOK_GRADE),
        ("detail", LOOK_DETAIL),
        ("ripple_loss", LOOK_RIPPLE_LOSS),
        ("ripple_mean", LOOK_RIPPLE_MEAN),
        ("body", LOOK_BODY),
        ("shore", LOOK_SHORE),
        ("shore_kind", LOOK_SHORE_KIND),
    ];
    assert_eq!(
        look,
        rows.map(|(m, lane)| (m.to_owned(), at(lane - LOOK_LANES.start)))
            .to_vec(),
        "WaterLook's rows and their offsets in the look block"
    );
}
