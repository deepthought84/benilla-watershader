//! The WGSL tripwire: `liquid.wgsl` composed with naga_oil and validated with naga, as Bevy's
//! shader cache composes it ([`wgsl`]), under the defs of every liquid pipeline a player reaches
//! ([`LiquidKey::shipped`]) and the debug variants, on each view key the world camera takes and
//! each liquid vertex layout.
//!
//! One liquid layout and one shader file serve every pipeline, and naga validates the whole
//! composed module, called or not: a helper outside the `WATER_STYLISED` region that names a
//! symbol inside it fails the reference and magma pipelines, whose draws then drop with nothing
//! but a pipeline error in the log. This finds that without a GPU.
//!
//! The Bevy modules come from the registry sources of the resolved `bevy_*` packages; where they
//! are absent the test is skipped as a data-gated test is, which fails under
//! `BENILLA_REQUIRE_DATA=1`.

#[path = "support/wgsl.rs"]
mod wgsl;

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use benilla_assets::materials::LiquidKey;

/// Every key the tripwire covers: the union of [`LiquidKey::shipped`] over its arguments, and each
/// stylised one with each combination of the debug variants' bits and the linear composite's
/// (`$WOW_WATER_COMPOSITE`), which the debug panel and the environment allow at once.
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
            [
                LiquidKey::VIEW,
                LiquidKey::SSR,
                LiquidKey::VIEW | LiquidKey::SSR,
                LiquidKey::COMPOSITE,
                LiquidKey::COMPOSITE | LiquidKey::VIEW,
                LiquidKey::COMPOSITE | LiquidKey::SSR,
                LiquidKey::COMPOSITE | LiquidKey::VIEW | LiquidKey::SSR,
            ]
            .map(|bits| {
                let (y, w) = k.path_lanes();
                let caps = (bits >> LiquidKey::CAPS_SHIFT) as f32;
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
    let Some(modules) = wgsl::bevy_modules() else {
        return;
    };
    let liquid_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders/liquid.wgsl");
    let liquid = std::fs::read_to_string(&liquid_path).expect("liquid.wgsl");
    let mut composer = wgsl::composer_for(&liquid, &modules)
        .unwrap_or_else(|e| panic!("liquid.wgsl's imports do not compose: {e}"));

    // Keys that push the same defs compile the same text: compose each def set once.
    let keys = covered_keys();
    assert_eq!(
        keys.len(),
        22 + 140,
        "22 shipped keys and 140 debug and composite variants of the stylised 20"
    );
    let mut by_defs: HashMap<Vec<&str>, Vec<LiquidKey>> = HashMap::new();
    for key in keys {
        by_defs.entry(key.frag_defs()).or_default().push(key);
    }
    let mut failures = Vec::new();
    let mut composed = 0;
    for (key_defs, keys) in &by_defs {
        for view in wgsl::LIQUID_VIEW_KEYS {
            for layout in wgsl::LIQUID_LAYOUTS {
                let named: BTreeSet<&str> =
                    key_defs.iter().chain(view).chain(layout).copied().collect();
                composed += 1;
                let verdict = wgsl::compose(
                    &mut composer,
                    &liquid,
                    &liquid_path,
                    wgsl::liquid_defs(named.iter().copied()),
                )
                .and_then(|m| wgsl::validate(&m));
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
    assert_eq!(
        composed,
        by_defs.len() * wgsl::LIQUID_VIEW_KEYS.len() * wgsl::LIQUID_LAYOUTS.len()
    );
    println!(
        "liquid.wgsl validated under {composed} def sets ({} key def sets)",
        by_defs.len()
    );
}
