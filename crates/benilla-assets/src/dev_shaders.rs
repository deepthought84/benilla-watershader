//! `$WOW_DEV_SHADERS=<repo root>`: the water's shaders read from the working tree at startup
//! instead of the copies compiled in, so a shader-only change needs a client restart and no build.
//! Dev builds only; without the variable, or in the player build, every path is the embedded one.
//!
//! One read-only asset source per shader folder (never the repo root, whose `target/` is huge),
//! and no writer. A dev iteration may change code only, never bindings, struct sizes or entry
//! points: the CPU side of those is compiled in.

use bevy::asset::AssetPath;
#[cfg(feature = "dev")]
use bevy::prelude::*;

/// The embedded crate, its folder under `crates/` and the asset source the folder is read through.
#[cfg(feature = "dev")]
const SOURCES: [(&str, &str, &str); 3] = [
    ("benilla_assets", "benilla-assets", "devsh-assets"),
    ("benilla_world", "benilla-world", "devsh-world"),
    ("benilla_water", "benilla-water", "devsh-water"),
];

/// The repo root `$WOW_DEV_SHADERS` names, if it holds `crates/`; read once.
#[cfg(feature = "dev")]
fn root() -> Option<&'static std::path::Path> {
    static ROOT: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    ROOT.get_or_init(|| {
        let raw = std::env::var_os("WOW_DEV_SHADERS")?;
        let p = std::fs::canonicalize(raw).ok()?;
        p.join("crates").is_dir().then_some(p)
    })
    .as_deref()
}

/// Register the working tree's shader folders as asset sources; call before `AssetPlugin` builds.
/// A no-op without `$WOW_DEV_SHADERS`.
#[cfg(feature = "dev")]
pub fn register_dev_shader_sources(app: &mut App) {
    use bevy::asset::io::{AssetSource, AssetSourceBuilder};
    let Some(root) = root() else {
        return;
    };
    for (_, dir, id) in SOURCES {
        let p = root.join("crates").join(dir).join("src/shaders");
        let p = p.to_string_lossy().into_owned();
        app.register_asset_source(
            id,
            AssetSourceBuilder::new(AssetSource::get_default_reader(p)),
        );
    }
    info!(
        "dev shaders: {} (the water's shaders read from the working tree)",
        root.display()
    );
}

/// No-op in the player build.
#[cfg(not(feature = "dev"))]
pub fn register_dev_shader_sources(_app: &mut bevy::prelude::App) {}

/// `embedded://<crate>/shaders/<file>` as the working tree's copy (`devsh-…://<file>`) when
/// `$WOW_DEV_SHADERS` is set and the file exists there; the embedded path otherwise.
pub fn shader_path(embedded: &'static str) -> AssetPath<'static> {
    #[cfg(feature = "dev")]
    if let Some(root) = root() {
        for (krate, dir, id) in SOURCES {
            let prefix = format!("embedded://{krate}/shaders/");
            if let Some(file) = embedded.strip_prefix(prefix.as_str()) {
                if root
                    .join("crates")
                    .join(dir)
                    .join("src/shaders")
                    .join(file)
                    .is_file()
                {
                    return AssetPath::parse(&format!("{id}://{file}")).into_owned();
                }
            }
        }
    }
    AssetPath::parse(embedded)
}
