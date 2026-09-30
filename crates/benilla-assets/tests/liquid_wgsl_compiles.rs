//! The WGSL tripwire: `liquid.wgsl` composed with naga_oil and validated with naga, as Bevy's
//! shader cache composes it (`bevy_shader-0.18.1` `ShaderCache::get`), under the defs of every
//! liquid pipeline a player reaches ([`LiquidKey::shipped`]) and the debug variants, on each view
//! key the world camera takes and each liquid vertex layout.
//!
//! One liquid layout and one shader file serve every pipeline, and naga validates the whole
//! composed module, called or not: a helper outside the `WATER_STYLISED` region that names a
//! symbol inside it fails the reference and magma pipelines, whose draws then drop with nothing
//! but a pipeline error in the log. This finds that without a GPU.
//!
//! The Bevy modules come from the registry sources of the resolved `bevy_*` packages (`cargo
//! metadata`); where they are absent (a vendored or offline tree) the test says so and passes.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use benilla_assets::materials::LiquidKey;
use naga_oil::compose::{
    get_preprocessor_data, ComposableModuleDescriptor, Composer, NagaModuleDescriptor,
    ShaderDefValue, ShaderLanguage, ShaderType,
};

/// The defs Bevy's mesh pipeline gives a liquid draw on the world camera, apart from the view key
/// and the vertex layout: the `fs_defs` of `$WOW_PIPE_TRACE` (`pipe_warm`) over the
/// `water-loch-modan` capture on 2026-09-30, both looks.
const BASE: &[(&str, ShaderDefValue)] = &[
    ("IRRADIANCE_VOLUMES_ARE_USABLE", ShaderDefValue::Bool(true)),
    ("MATERIAL_BIND_GROUP", ShaderDefValue::UInt(3)),
    ("MESH_PIPELINE", ShaderDefValue::Bool(true)),
    ("MULTIPLE_LIGHTMAPS_IN_ARRAY", ShaderDefValue::Bool(true)),
    ("MULTIPLE_LIGHT_PROBES_IN_ARRAY", ShaderDefValue::Bool(true)),
    (
        "SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS",
        ShaderDefValue::Int(8),
    ),
    ("SHADOW_FILTER_METHOD_GAUSSIAN", ShaderDefValue::Bool(true)),
    ("VERTEX_OUTPUT_INSTANCE_INDEX", ShaderDefValue::Bool(true)),
    ("VIEW_PROJECTION_PERSPECTIVE", ShaderDefValue::Bool(true)),
];

/// The def the pipeline cache adds to every shader (`bevy_render-0.18.1`
/// `pipeline_cache.rs:287`): the device's `max_storage_buffers_per_shader_stage`, here wgpu's
/// default limit.
const STORAGE_BINDINGS: (&str, ShaderDefValue) =
    ("AVAILABLE_STORAGE_BUFFER_BINDINGS", ShaderDefValue::UInt(8));

/// The defs a module carries of its own, as `load_shader_library!` settings give them: the light
/// array sizes of `mesh_view_types` (`bevy_pbr-0.18.1` `render/mesh.rs:132`, desktop values).
/// `mesh_bindings`' batch size (`:296`) is set only where storage buffers are missing.
fn module_defs(module: &str) -> Vec<(&'static str, ShaderDefValue)> {
    match module {
        "bevy_pbr::mesh_view_types" => vec![
            ("MAX_DIRECTIONAL_LIGHTS", ShaderDefValue::UInt(10)),
            ("MAX_CASCADES_PER_LIGHT", ShaderDefValue::UInt(4)),
        ],
        _ => Vec::new(),
    }
}

/// The view-key defs of the world camera: MSAA on or off. It carries no depth prepass
/// (`benilla_water::depth`).
const VIEW_KEYS: [&[&str]; 2] = [&[], &["MULTISAMPLED"]];

/// The vertex layouts' defs: every liquid surface's (`benilla_world::liquid::warm_liquid_mesh`),
/// the same with an interior pool's vertex colour, and the bare grid whose `#else` arms stand in
/// for a missing attribute.
const LAYOUTS: [&[&str]; 3] = [
    &[
        "VERTEX_POSITIONS",
        "VERTEX_NORMALS",
        "VERTEX_UVS",
        "VERTEX_UVS_A",
        "VERTEX_UVS_B",
        "LIQUID_SHORE_OFFSET",
        "LIQUID_SURFACE_NORMAL",
        "LIQUID_PLANAR",
        "LIQUID_FLOW",
    ],
    &[
        "VERTEX_POSITIONS",
        "VERTEX_NORMALS",
        "VERTEX_UVS",
        "VERTEX_UVS_A",
        "VERTEX_UVS_B",
        "VERTEX_COLORS",
        "LIQUID_SHORE_OFFSET",
        "LIQUID_SURFACE_NORMAL",
        "LIQUID_PLANAR",
        "LIQUID_FLOW",
    ],
    &[
        "VERTEX_POSITIONS",
        "VERTEX_NORMALS",
        "VERTEX_UVS",
        "VERTEX_UVS_A",
        "VERTEX_UVS_B",
    ],
];

/// The `src` folders of the resolved `bevy_*` packages, from `cargo metadata` offline and for the
/// host alone (another platform's packages need not be downloaded); `None` when cargo cannot say
/// or a package's sources are not on disk.
fn bevy_sources() -> Option<Vec<PathBuf>> {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = std::process::Command::new(rustc).arg("-vV").output().ok()?;
    let host = String::from_utf8(version.stdout)
        .ok()?
        .lines()
        .find_map(|l| l.strip_prefix("host: ").map(str::to_owned))?;
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let out = std::process::Command::new(cargo)
        .args(["metadata", "--format-version", "1", "--offline"])
        .args(["--filter-platform", &host])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let json = String::from_utf8(out.stdout).ok()?;
    let key = "\"manifest_path\":\"";
    let mut roots: Vec<PathBuf> = json
        .match_indices(key)
        .filter_map(|(at, _)| {
            let rest = &json[at + key.len()..];
            let manifest = Path::new(&rest[..rest.find('"')?]);
            let dir = manifest.parent()?;
            let name = dir.file_name()?.to_str()?;
            name.starts_with("bevy_").then(|| dir.join("src"))
        })
        .collect();
    roots.sort();
    roots.dedup();
    let wanted = ["bevy_pbr-", "bevy_render-"];
    let found = wanted.iter().all(|w| {
        roots.iter().any(|r| {
            r.is_dir()
                && r.parent()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(w))
        })
    });
    found.then_some(roots)
}

/// Every `.wgsl` file under `root` with a `#define_import_path`, by module name.
fn collect_modules(root: &Path, into: &mut HashMap<String, (PathBuf, String)>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_modules(&path, into);
        } else if path.extension().is_some_and(|e| e == "wgsl") {
            let Ok(source) = std::fs::read_to_string(&path) else {
                continue;
            };
            if let (Some(name), _, _) = get_preprocessor_data(&source) {
                into.entry(name).or_insert((path, source));
            }
        }
    }
}

/// Add `module` to the composer after everything it imports, as `add_import_to_composer` does.
fn add_module(
    composer: &mut Composer,
    modules: &HashMap<String, (PathBuf, String)>,
    module: &str,
) -> Result<(), String> {
    if composer.contains_module(module) {
        return Ok(());
    }
    let (path, source) = modules
        .get(module)
        .ok_or_else(|| format!("no source defines `{module}`"))?;
    for import in get_preprocessor_data(source).1 {
        add_module(composer, modules, &import.import)?;
    }
    composer
        .add_composable_module(ComposableModuleDescriptor {
            source,
            file_path: &path.to_string_lossy(),
            language: ShaderLanguage::Wgsl,
            as_name: None,
            additional_imports: &[],
            shader_defs: module_defs(module)
                .into_iter()
                .chain([STORAGE_BINDINGS])
                .map(|(k, v)| (k.into(), v))
                .collect(),
        })
        .map(|_| ())
        .map_err(|e| format!("{module}: {e:?}"))
}

/// Every key the tripwire covers: the union of [`LiquidKey::shipped`] over its arguments, and each
/// stylised one with a debug variant's bit added.
fn covered_keys() -> Vec<LiquidKey> {
    let mut keys = Vec::new();
    for triple in 0..8 {
        for key in LiquidKey::shipped(triple & 1 != 0, triple & 2 != 0, triple & 4 != 0) {
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
    }
    let debug: Vec<LiquidKey> = keys
        .iter()
        .filter(|k| k.composite())
        .flat_map(|k| {
            [LiquidKey::VIEW, LiquidKey::SSR].map(|bit| {
                let (y, w) = k.path_lanes();
                let caps = (bit >> LiquidKey::CAPS_SHIFT) as f32;
                LiquidKey::from_lanes(
                    bevy::math::Vec4::ZERO,
                    bevy::math::Vec4::new(0.0, y, 0.0, w + caps),
                )
            })
        })
        .collect();
    for key in debug {
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    keys
}

#[test]
fn liquid_wgsl_compiles_for_every_liquid_pipeline() {
    let Some(roots) = bevy_sources() else {
        println!("skipped: bevy sources not found");
        return;
    };
    let mut modules = HashMap::new();
    for root in &roots {
        collect_modules(root, &mut modules);
    }
    let liquid_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/liquid.wgsl");
    let liquid = std::fs::read_to_string(&liquid_path).expect("liquid.wgsl");
    let mut composer =
        Composer::non_validating().with_capabilities(naga::valid::Capabilities::all());
    for import in get_preprocessor_data(&liquid).1 {
        add_module(&mut composer, &modules, &import.import).unwrap_or_else(|e| {
            panic!(
                "liquid.wgsl's import {} does not compose: {e}",
                import.import
            )
        });
    }

    // Keys that push the same defs compile the same text: compose each def set once.
    let keys = covered_keys();
    assert_eq!(
        keys.len(),
        22 + 40,
        "22 shipped keys and 40 debug variants of the stylised 20"
    );
    let mut by_defs: HashMap<Vec<&str>, Vec<LiquidKey>> = HashMap::new();
    for key in keys {
        by_defs.entry(key.frag_defs()).or_default().push(key);
    }
    let mut failures = Vec::new();
    let mut composed = 0;
    for (key_defs, keys) in &by_defs {
        for view in VIEW_KEYS {
            for layout in LAYOUTS {
                let named: BTreeSet<&str> =
                    key_defs.iter().chain(view).chain(layout).copied().collect();
                let mut defs: HashMap<String, ShaderDefValue> =
                    BASE.iter().map(|(k, v)| ((*k).into(), *v)).collect();
                defs.insert(STORAGE_BINDINGS.0.into(), STORAGE_BINDINGS.1);
                defs.extend(
                    named
                        .iter()
                        .map(|d| ((*d).into(), ShaderDefValue::Bool(true))),
                );
                let module = composer.make_naga_module(NagaModuleDescriptor {
                    source: &liquid,
                    file_path: &liquid_path.to_string_lossy(),
                    shader_type: ShaderType::Wgsl,
                    shader_defs: defs,
                    additional_imports: &[],
                });
                composed += 1;
                let verdict = module.map_err(|e| format!("compose: {e:?}")).and_then(|m| {
                    naga::valid::Validator::new(
                        naga::valid::ValidationFlags::all(),
                        naga::valid::Capabilities::all(),
                    )
                    .validate(&m)
                    .map(|_| ())
                    .map_err(|e| format!("validate: {e:?}"))
                });
                if let Err(e) = verdict {
                    failures.push(format!("keys {keys:?}, defs {named:?}: {e}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "liquid.wgsl fails {} of {composed} def sets; the pipelines of these keys drop every \
         liquid draw they make:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert_eq!(composed, by_defs.len() * VIEW_KEYS.len() * LAYOUTS.len());
    println!(
        "liquid.wgsl validated under {composed} def sets ({} key def sets)",
        by_defs.len()
    );
}
