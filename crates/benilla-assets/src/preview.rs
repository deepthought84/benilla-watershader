//! The Improved Water **preview** levers: which of the preview's features run, read once from the
//! environment and shared by every crate that draws or feeds them (the liquid pipeline's
//! `WATER_PREVIEW` def, the reflection block's mask row, the CPU inputs).
//!
//! - `$WOW_PREVIEW` sets the start: unset or `all` every feature, `0`/`off`/`none` none, a comma
//!   list of names exactly those, a list of `-name` entries every feature but those.
//! - `$WOW_PREVIEW_<NAME>=0` then turns one feature off (`=1` on), e.g. `WOW_PREVIEW_WAVES=0`.
//!
//! With no feature on, no stylised pipeline compiles the preview and the frame is today's.

/// The features, by bit: each name is its lever's (`WOW_PREVIEW_<NAME>`, upper case).
pub const NAMES: [&str; 23] = [
    "waves",
    "glint",
    "rough",
    "rough_sky",
    "shore",
    "surf",
    "rivers",
    "falls",
    "wake",
    "wake_foam",
    "rings",
    "window",
    "tint",
    "shafts",
    "uw_caustics",
    "waterline",
    "caustics",
    "rain",
    "lights",
    "sparkles",
    "crest",
    "mist",
    "glow",
];

/// WAV-1/3/4/5: the wind's wave field, swell, gusts and slicks, capillaries.
pub const WAVES: u32 = 1 << 0;
/// SPEC-1..4: the sun's and the moon's GGX glint on the surface's roughness.
pub const GLINT: u32 = 1 << 1;
/// REF-2/4/5/6/12: the reflections blurred by the surface's unresolved roughness.
pub const ROUGH: u32 = 1 << 2;
/// REF-4's sky floor on the same lobe as the mirrors.
pub const ROUGH_SKY: u32 = 1 << 3;
/// SHORE-1/2, BODY-7: the wet line, coverage-correct edge, contact foam, wet faces.
pub const SHORE: u32 = 1 << 4;
/// SHORE-4: the sea's swash, bores and whitecaps.
pub const SURF: u32 = 1 << 5;

/// Every feature bit.
pub const ALL: u32 = (1 << NAMES.len()) - 1;

/// The bit of the feature called `name` (any case), if there is one.
pub fn bit(name: &str) -> Option<u32> {
    NAMES
        .iter()
        .position(|n| n.eq_ignore_ascii_case(name.trim()))
        .map(|i| 1 << i)
}

/// The mask `$WOW_PREVIEW` = `master` and the per-feature levers `get(NAME)` give.
fn parse(master: Option<&str>, get: impl Fn(&str) -> Option<String>) -> u32 {
    let mut mask = match master.map(str::trim) {
        None | Some("") | Some("all") | Some("1") | Some("on") => ALL,
        Some("0") | Some("off") | Some("none") => 0,
        Some(list) => {
            let items: Vec<&str> = list
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect();
            if items.iter().all(|s| s.starts_with('-')) {
                items
                    .iter()
                    .filter_map(|s| bit(&s[1..]))
                    .fold(ALL, |m, b| m & !b)
            } else {
                items.iter().filter_map(|s| bit(s)).fold(0, |m, b| m | b)
            }
        }
    };
    for (i, name) in NAMES.iter().enumerate() {
        match get(&name.to_ascii_uppercase()).as_deref().map(str::trim) {
            Some("0") | Some("off") => mask &= !(1 << i),
            Some("1") | Some("on") => mask |= 1 << i,
            _ => {}
        }
    }
    mask
}

/// The preview's feature bits this run, read once.
pub fn mask() -> u32 {
    static MASK: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MASK.get_or_init(|| {
        let master = std::env::var("WOW_PREVIEW").ok();
        let m = parse(master.as_deref(), |n| {
            std::env::var(format!("WOW_PREVIEW_{n}")).ok()
        });
        bevy::log::info!("preview: {}", describe(m));
        m
    })
}

/// Whether any of `bits` is on.
pub fn on(bits: u32) -> bool {
    mask() & bits != 0
}

/// Whether the preview runs at all: any feature on.
pub fn any() -> bool {
    mask() != 0
}

/// The names of the features on in `mask`, for the log.
pub fn describe(mask: u32) -> String {
    if mask == 0 {
        return "off (WOW_PREVIEW=0)".into();
    }
    let on: Vec<&str> = NAMES
        .iter()
        .enumerate()
        .filter(|(i, _)| mask & (1 << i) != 0)
        .map(|(_, n)| *n)
        .collect();
    format!("on: {}", on.join(" "))
}

/// A sub-lever `$WOW_PREVIEW_<NAME>` read as a finite number within `lo..=hi`, `default` otherwise.
pub fn number(name: &str, default: f32, lo: f32, hi: f32) -> f32 {
    std::env::var(format!("WOW_PREVIEW_{name}"))
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && (lo..=hi).contains(v))
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_levers_parse() {
        let none = |_: &str| None;
        assert_eq!(parse(None, none), ALL);
        assert_eq!(parse(Some("0"), none), 0);
        assert_eq!(parse(Some("waves,shore"), none), WAVES | SHORE);
        assert_eq!(parse(Some("-glint"), none), ALL & !GLINT);
        let one_off = |n: &str| (n == "ROUGH").then(|| "0".to_string());
        assert_eq!(parse(None, one_off), ALL & !ROUGH);
        let one_on = |n: &str| (n == "SURF").then(|| "1".to_string());
        assert_eq!(parse(Some("off"), one_on), SURF);
        assert!(ALL < 1 << 24, "the mask stays exact in an f32");
    }
}
