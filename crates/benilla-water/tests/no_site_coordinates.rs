//! Improved Water runs on whatever map is loaded, so its runtime code names no place: every
//! position it uses is derived from the loaded map. This fails on a site's coordinates written
//! into `src/` as a literal position. Test modules may name places and are not scanned.

use std::path::{Path, PathBuf};

/// The least magnitude, in yards, that makes a three-number literal a place in the world rather
/// than a colour, a direction or an offset: a map spans thousands of yards around its origin.
const PLACE_YD: f64 = 100.0;

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

/// Every innermost `(…)` or `[…]` in `code` holding exactly three numeric literals, at least one of
/// them a float, with a component of at least [`PLACE_YD`]: `Vec3::new(-420.0, 48.64, 9385.0)`,
/// `[-9385.0, 420.0, 48.64]`, `vec3<f32>(…)`. Returned as `(line, text)`.
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
            let values: Vec<f64> = parts.iter().filter_map(|p| number(p)).collect();
            if parts.len() == 3
                && values.len() == 3
                && parts.iter().any(|p| p.contains('.'))
                && values.iter().any(|v| v.abs() >= PLACE_YD)
            {
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
/// spelling a position takes here, and passes the crate's other kinds of three-number literal.
#[test]
fn a_literal_world_position_is_found() {
    for site in [
        "const PROBE_AT: Vec3 = Vec3::new(-420.0, 48.64, 9385.0);",
        "let at = [-9385.0, 420.0, 48.64];",
        "let at = vec3<f32>(-9385.0, 420.0, 48.64);",
        "let at = Vec3::new(\n    -420.0,\n    48.64,\n    9_385.0,\n);",
    ] {
        assert_eq!(literal_positions(site).len(), 1, "not found: {site}");
    }
    for not_a_site in [
        "Vec3::new(0.0, 1.0, 0.0)",
        "Color::srgb(0.25, 0.95, 0.35)",
        "@workgroup_size(256, 1, 1)",
        "UVec2::new(640, 360)",
        "smoothstep(100.0, 400.0, d)",
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
/// map (its classification's spots, its liquid, its terrain); a site's coordinates in `src/`
/// make the look depend on one place.
#[test]
fn the_runtime_code_names_no_place() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    sources(&src, &mut files);
    assert!(!files.is_empty(), "no sources under {}", src.display());
    let mut found = Vec::new();
    for path in files {
        let text = std::fs::read_to_string(&path).expect("a readable source");
        for (line, literal) in literal_positions(&runtime_code(&text)) {
            let path = path.strip_prefix(&src).unwrap_or(&path).display();
            found.push(format!("src/{path}:{line}: {literal}"));
        }
    }
    assert!(
        found.is_empty(),
        "a literal world position in runtime code; derive it from the loaded map:\n{}",
        found.join("\n")
    );
}
