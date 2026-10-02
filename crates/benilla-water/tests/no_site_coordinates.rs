//! Improved Water runs on whatever map is loaded, so its runtime code names no place: every
//! position it uses is derived from the loaded map. This fails on a site's coordinates written as
//! a literal position into the crate's `src/` or into `liquid.wgsl`, the water's shader in
//! `benilla-assets`: three numbers, or a horizontal pair. Test modules may name places and are not
//! scanned. A lone number, such as a site's height, cannot be told from a constant and passes.
//!
//! The rule reads shape, not meaning. Any pair of place-sized floats in a vector or an array reads
//! as a position, a size or a fade range packed as `vec2(100.0, 400.0)` or `[512.0, 512.0]` too:
//! pass such a range as a function's two arguments. A pair returned as a tuple or built with
//! `array<f32, 2>(…)`, a pair with an expression in it, and a triple with an expression in it and
//! fewer than two place-sized literals are not read as positions. A triple of literals is one when
//! a literal is place-sized and two are not 0: `vec3(0.0, 10000.0, 0.0)`, on an axis, passes.

use std::path::{Path, PathBuf};

/// The least magnitude, in yards, that makes a literal a coordinate in the world rather than a
/// colour, a direction or an offset: a map spans thousands of yards around its origin.
const PLACE_YD: f64 = 100.0;

/// The vector constructors a position is written with, in Rust and WGSL.
const VECTORS: &[&str] = &[
    "vec2",
    "vec3",
    "vec2f",
    "vec3f",
    "Vec2::new",
    "Vec3::new",
    "Vec3A::new",
];

/// A numeric literal's value: an optional sign, digits, and a Rust or WGSL float suffix.
fn number(s: &str) -> Option<f64> {
    let s = s.trim();
    let digits = s.strip_prefix('-').unwrap_or(s).trim_start();
    if !digits.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    let digits = digits.replace('_', "");
    let digits = ["f32", "f64", "f"]
        .iter()
        .find_map(|suffix| digits.strip_suffix(suffix))
        .unwrap_or(&digits);
    let v: f64 = digits.parse().ok()?;
    Some(if s.starts_with('-') { -v } else { v })
}

/// Whether the `(` or `[` at `at` in `code` opens a list of coordinates: an array, a tuple or a
/// vector's constructor, not another function's arguments, which a range such as
/// `smoothstep(100.0, 400.0, d)` fills.
fn opens_coordinates(code: &str, at: usize) -> bool {
    if code[at..].starts_with('[') {
        return true;
    }
    let before = code[..at].trim_end();
    let name_len: usize = before
        .chars()
        .rev()
        .take_while(|c| c.is_alphanumeric() || "_:<>!".contains(*c))
        .map(char::len_utf8)
        .sum();
    let name = &before[before.len() - name_len..];
    let name = ["<f32>", "<f64>"]
        .iter()
        .find_map(|t| name.strip_suffix(t))
        .unwrap_or(name);
    name.is_empty() || VECTORS.iter().any(|v| name.ends_with(v))
}

/// Whether the two or three comma-separated `parts` of a list are a position: a float among them,
/// not a 0-255 colour, and either three numbers with a place-sized one off an axis, or, in a list
/// of coordinates, a horizontal pair or a triple with one computed part, each with two
/// place-sized numbers.
fn is_position(parts: &[&str], coordinates: bool) -> bool {
    if !parts.iter().any(|p| p.contains('.')) {
        return false;
    }
    let values: Vec<f64> = parts.iter().filter_map(|p| number(p)).collect();
    let far = values.iter().filter(|v| v.abs() >= PLACE_YD).count();
    if values.len() < parts.len() {
        // `Vec3::new(-420.0, 48.64 + LIFT, 9385.0)`.
        return coordinates && parts.len() == 3 && far >= 2;
    }
    if values
        .iter()
        .all(|v| v.fract() == 0.0 && (0.0..=255.0).contains(v))
    {
        return false;
    }
    match parts.len() {
        // Off an axis: `vec3(0.0, 10000.0, 0.0)` is a direction's sentinel.
        3 => far >= 1 && values.iter().filter(|v| **v != 0.0).count() >= 2,
        2 => coordinates && far == 2,
        _ => false,
    }
}

/// Every innermost `(…)` or `[…]` in `code` that [`is_position`]: `Vec3::new(-420.0, 48.64,
/// 9385.0)`, `[-9385.0, 420.0, 48.64]`, `vec2<f32>(-420.0, 9385.0)`. Returned as `(line, text)`.
fn literal_positions(code: &str) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    for (open, close) in [('(', ')'), ('[', ']')] {
        for (at, _) in code.match_indices(open) {
            let inner = &code[at + 1..];
            let Some(end) = inner.find(close) else {
                continue;
            };
            let inner = &inner[..end];
            if inner.contains(['(', ')', '[', ']']) {
                continue;
            }
            let parts: Vec<&str> = inner
                .split(',')
                .map(str::trim)
                .filter(|p| !p.is_empty())
                .collect();
            if (2..=3).contains(&parts.len()) && is_position(&parts, opens_coordinates(code, at)) {
                let line = code[..at].matches('\n').count() + 1;
                found.push((line, format!("{open}{inner}{close}")));
            }
        }
    }
    found
}

/// A source file with its `#[cfg(test)] mod tests` removed and every `//` comment blanked,
/// keeping the line count so a finding names its real line.
fn runtime_code(text: &str) -> String {
    let text = text
        .match_indices("#[cfg(test)]")
        .find(|(at, m)| text[at + m.len()..].trim_start().starts_with("mod tests"))
        .map_or(text, |(at, _)| &text[..at]);
    text.lines()
        .map(|l| l.find("//").map_or(l, |c| &l[..c]))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every `.rs` and `.wgsl` file under `dir`.
fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("the crate's src is readable") {
        let path = entry.expect("a readable entry").path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs" || e == "wgsl") {
            out.push(path);
        }
    }
}

/// The detector finds the literal the probe's old fixed capture point was written as, in each
/// spelling a position takes here, and passes the other kinds of two- and three-number literal
/// the water's code holds.
#[test]
fn a_literal_world_position_is_found() {
    for site in [
        "const PROBE_AT: Vec3 = Vec3::new(-420.0, 48.64, 9385.0);",
        "let at = [-9385.0, 420.0, 48.64];",
        "let at = vec3<f32>(-9385.0, 420.0, 48.64);",
        "let at = Vec3::new(\n    -420.0,\n    48.64,\n    9_385.0,\n);",
        "let at = vec2<f32>(-420.0, 9385.0);",
        "let at = (-9385.0, 420.0);",
        "let at = Vec3::new(-420.0, 48.64 + LIFT, 9385.0);",
    ] {
        assert_eq!(literal_positions(site).len(), 1, "not found: {site}");
    }
    for not_a_site in [
        "Vec3::new(0.0, 1.0, 0.0)",
        "Color::srgb(0.25, 0.95, 0.35)",
        "vec3<f32>(255.0, 128.0, 0.0)",
        "vec3<f32>(0.0, 10000.0, 0.0)",
        "vec2<f32>(0.0, 10000.0)",
        "@workgroup_size(256, 1, 1)",
        "UVec2::new(640, 360)",
        "smoothstep(100.0, 400.0, d)",
        "clamp(d, 100.0, 400.0)",
        "Vec3::new(0.0, 4000.0 * k, 0.0)",
    ] {
        assert!(
            literal_positions(not_a_site).is_empty(),
            "flagged: {not_a_site}"
        );
    }
    assert!(literal_positions(&runtime_code("// Mirror Lake (-9385.0, 420.0, 48.64)")).is_empty());
}

/// **No place in the world is written into the runtime code.** Positions come from the loaded
/// map (its classification's spots, its liquid, its terrain); a site's coordinates in `src/` or
/// in the shader make the look depend on one place.
#[test]
fn the_runtime_code_names_no_place() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate sits in the workspace's crates/");
    let mut files = Vec::new();
    sources(&crates.join("benilla-water/src"), &mut files);
    assert!(!files.is_empty(), "no sources under benilla-water/src");
    files.push(crates.join("benilla-assets/src/shaders/liquid.wgsl"));
    let mut found = Vec::new();
    for path in files {
        let text = std::fs::read_to_string(&path).expect("a readable source");
        for (line, literal) in literal_positions(&runtime_code(&text)) {
            let path = path.strip_prefix(crates).unwrap_or(&path).display();
            found.push(format!("crates/{path}:{line}: {literal}"));
        }
    }
    assert!(
        found.is_empty(),
        "a literal world position in runtime code; derive it from the loaded map:\n{}",
        found.join("\n")
    );
}
