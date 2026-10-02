//! Composing a WGSL file the way Bevy's shader cache does (`bevy_shader-0.18.1`
//! `ShaderCache::get`): naga_oil, with every module it imports added from the registry sources of
//! the resolved `bevy_*` packages (`cargo metadata`), then validated with naga. Shared by the
//! shader tripwires of `benilla-assets` and `benilla-water`, which include this file by path.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use naga_oil::compose::{
    get_preprocessor_data, ComposableModuleDescriptor, Composer, NagaModuleDescriptor,
    ShaderDefValue, ShaderLanguage, ShaderType,
};

/// The defs Bevy's mesh pipeline gives a liquid draw on the world camera, apart from the view key
/// and the vertex layout: the `fs_defs` of `$WOW_PIPE_TRACE` (`pipe_warm`) over the
/// `water-loch-modan` capture on 2026-09-30, both looks.
pub const LIQUID_BASE: &[(&str, ShaderDefValue)] = &[
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
pub const STORAGE_BINDINGS: (&str, ShaderDefValue) =
    ("AVAILABLE_STORAGE_BUFFER_BINDINGS", ShaderDefValue::UInt(8));

/// The view-key defs of the world camera: MSAA on or off. It carries no depth prepass
/// (`benilla_water::depth`).
pub const LIQUID_VIEW_KEYS: [&[&str]; 2] = [&[], &["MULTISAMPLED"]];

/// The liquid vertex layouts' defs: every liquid surface's
/// (`benilla_world::liquid::warm_liquid_mesh`), the same with an interior pool's vertex colour,
/// and the bare grid whose `#else` arms stand in for a missing attribute.
pub const LIQUID_LAYOUTS: [&[&str]; 3] = [
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

/// `liquid.wgsl`'s defs on one liquid pipeline: [`LIQUID_BASE`], [`STORAGE_BINDINGS`] and the
/// key's, the view key's and the layout's flags.
pub fn liquid_defs<'a>(
    flags: impl IntoIterator<Item = &'a str>,
) -> HashMap<String, ShaderDefValue> {
    let mut defs: HashMap<String, ShaderDefValue> =
        LIQUID_BASE.iter().map(|(k, v)| ((*k).into(), *v)).collect();
    defs.insert(STORAGE_BINDINGS.0.into(), STORAGE_BINDINGS.1);
    defs.extend(
        flags
            .into_iter()
            .map(|d| (d.into(), ShaderDefValue::Bool(true))),
    );
    defs
}

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
    let wanted = ["bevy_core_pipeline-", "bevy_pbr-", "bevy_render-"];
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

/// The importable Bevy modules, by name. Where the sources are not on disk this is the project's
/// data-gated skip ([`benilla_formats::skipped`]): printed, counted under `$BENILLA_SKIP_LOG`, and
/// a failure under `BENILLA_REQUIRE_DATA=1`.
pub fn bevy_modules() -> Option<HashMap<String, (PathBuf, String)>> {
    let Some(roots) = bevy_sources() else {
        benilla_formats::skipped(
            "bevy sources not found (`cargo metadata --offline`)",
            &[PathBuf::from(env!("CARGO_MANIFEST_DIR"))],
        );
        return None;
    };
    let mut modules = HashMap::new();
    for root in &roots {
        collect_modules(root, &mut modules);
    }
    Some(modules)
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

/// A composer holding every module `source` imports, or which import fails to compose.
pub fn composer_for(
    source: &str,
    modules: &HashMap<String, (PathBuf, String)>,
) -> Result<Composer, String> {
    let mut composer =
        Composer::non_validating().with_capabilities(naga::valid::Capabilities::all());
    for import in get_preprocessor_data(source).1 {
        add_module(&mut composer, modules, &import.import)?;
    }
    Ok(composer)
}

/// `source` composed under `defs`, as the pipeline cache composes it.
pub fn compose(
    composer: &mut Composer,
    source: &str,
    path: &Path,
    defs: HashMap<String, ShaderDefValue>,
) -> Result<naga::Module, String> {
    composer
        .make_naga_module(NagaModuleDescriptor {
            source,
            file_path: &path.to_string_lossy(),
            shader_type: ShaderType::Wgsl,
            shader_defs: defs,
            additional_imports: &[],
        })
        .map_err(|e| format!("compose: {e:?}"))
}

/// Validate a composed module with every check and capability.
pub fn validate(module: &naga::Module) -> Result<(), String> {
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map(|_| ())
    .map_err(|e| format!("validate: {e:?}"))
}
