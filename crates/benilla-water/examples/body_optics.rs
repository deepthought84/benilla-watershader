//! The body's optics and light at the golden water scenes, from the install's `Light.dbc`: for
//! each kind the clear-noon swatch, the absorption and turbidity `body::kind_optics` derives from
//! it, and the depth at which a bed seen straight down fades to a tenth; at the scene's own minute
//! the swatch, the capped deep stop (`BodyLook::deep_ceiling` along the zone's line, the colour
//! `deep_hue` rows judge) with its hue, and the body's illuminant. Prints Blizzard-derived numbers:
//! never commit the output.
//! `cargo run --release -p benilla-water --example body_optics [-- <map> <x> <y> <z> <minute>]`

use benilla_formats::{LightCatalog, LiquidKind, Submersion};
use benilla_water::body::{body_light, deep_stop, kind_optics, luma, BodyLook};
use benilla_world::lighting::{celestial_sun_direction, sun_direction, WowLighting};
use benilla_world::sky::dome_uniforms;

/// The golden water scenes' eyes and minutes (`benilla-app`'s `capture/scenarios.rs`), one per
/// distinct spot: name, map, WoW x, y, z, minute.
const SCENES: &[(&str, u32, [f32; 3], u32)] = &[
    ("water-noon", 0, [-9527.0, -310.6, 70.8], 720),
    ("water-night", 0, [-9527.0, -310.6, 70.8], 0),
    ("water-loch-modan", 0, [-5240.3, -3119.5, 300.8], 720),
    ("water-elwynn-falls", 0, [-8322.0, -340.0, 147.2], 720),
    ("water-falfarren-river", 1, [2666.2, -2545.9, 175.2], 720),
    ("water-savage-coast", 0, [-12619.3, -41.0, 29.9], 720),
    ("water-savage-sunset", 0, [-12619.3, -41.0, 29.9], 1180),
    ("water-savage-night", 0, [-12727.1, -2.1, 2.4], 1364),
    ("water-menethil-sunset", 0, [-3977.1, -899.0, 10.6], 1180),
    (
        "water-menethil-wreck-noon",
        0,
        [-3977.14, -898.95, 10.84],
        720,
    ),
    ("water-stormwind-canal", 0, [-8748.7, 526.2, 98.4], 720),
    (
        "water-alameth-bank-noon",
        1,
        [9356.42, 1191.51, 1254.81],
        720,
    ),
    ("water-darkshore-pier-noon", 1, [6450.0, 526.0, 10.48], 720),
    ("water-felwood-jademir", 1, [-3057.92, 1837.42, 31.45], 720),
    ("water-swamp-sorrows", 0, [-10323.0, -4045.0, 21.6], 720),
    ("water-darrowmere", 0, [1293.0, -2428.0, 57.2], 720),
    ("water-forgotten-coast", 1, [-4347.46, 2415.11, 10.01], 720),
    ("water-lordamere-rock", 0, [982.0, 537.0, 36.5], 720),
    ("water-low-moon", 0, [-5444.4, -3504.4, 301.29], 150),
    ("water-crystal-3p-noon", 0, [-9535.03, -357.67, 66.57], 720),
    ("water-baradin-noon", 0, [-2537.88, -1692.24, 6.3], 720),
    (
        "water-baradin-sunset-45",
        0,
        [-2537.88, -1692.24, 6.3],
        1180,
    ),
    ("water-westfall-noon", 0, [-10589.6, 2100.2, 3.8], 720),
    ("water-dam-foot", 0, [-4550.9, -3318.7, 182.0], 720),
    ("water-booty-underdeck", 0, [-14380.0, 409.0, 3.86], 720),
    ("water-booty-harbour", 0, [-14297.2, 530.99, 10.78], 720),
    ("water-stv-river", 0, [-13647.2, 482.3, 26.92], 720),
    ("water-stv-mouth", 0, [-12757.0, 269.4, 11.1], 720),
];

/// HSV hue in degrees, and the chroma.
fn hue(c: [f32; 3]) -> (f32, f32) {
    let (max, min) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
    let d = max - min;
    if d <= 0.0 {
        return (0.0, 0.0);
    }
    let h = if max == c[0] {
        ((c[1] - c[2]) / d).rem_euclid(6.0)
    } else if max == c[1] {
        (c[2] - c[0]) / d + 2.0
    } else {
        (c[0] - c[1]) / d + 4.0
    };
    (60.0 * h, d)
}

fn bytes(c: [f32; 3]) -> [i32; 3] {
    c.map(|v| (v * 255.0).round() as i32)
}

fn main() -> anyhow::Result<()> {
    let data = benilla_formats::wow_data_or_skip!(Ok(()));
    let mut chain = benilla_formats::open_chain(&data)?;
    let catalog = LightCatalog::load(&mut chain)?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let one;
    let scenes: &[(&str, u32, [f32; 3], u32)] = if args.len() == 5 {
        let n = |i: usize| args[i].parse::<f32>();
        one = [(
            "spot",
            args[0].parse()?,
            [n(1)?, n(2)?, n(3)?],
            args[4].parse()?,
        )];
        &one
    } else {
        SCENES
    };
    let look = BodyLook::default();
    // The refracted lighting sun's cosine at 30° elevation, for the nadir fade depth.
    let mu_d = (1.0 - (60f32.to_radians().sin() / 1.333).powi(2)).sqrt();
    for &(name, map, pos, minute) in scenes {
        let noon = catalog.sample_blended(map, pos, 1440, false, Submersion::Dry, false);
        let now = catalog.sample_blended(map, pos, minute * 2, false, Submersion::Dry, false);
        println!("{name} (map {map}, {pos:?}, minute {minute})");
        for (kind, label) in [(LiquidKind::Still, "river"), (LiquidKind::Ocean, "ocean")] {
            let ocean = kind == LiquidKind::Ocean;
            let (rows_noon, alpha_noon, rows_now) = if ocean {
                (noon.water_ocean, noon.water_ocean_alpha, now.water_ocean)
            } else {
                (noon.water_river, noon.water_river_alpha, now.water_river)
            };
            let o = kind_optics(rows_noon[0], rows_noon[1], alpha_noon[0], ocean, &look);
            let s = o.sigma_abs;
            // Where Y(t_bed) falls to 0.1 straight down: the path is z·(1 + 1/μ_d).
            let fade =
                |z: f32| luma(s.map(|a| (-(a + o.sigma_turb) * z * (1.0 + 1.0 / mu_d)).exp()));
            let mut z = 0.0;
            while fade(z) > 0.1 && z < 60.0 {
                z += 0.05;
            }
            let stop = deep_stop(rows_now[0], rows_now[1], ocean, look.deep_ceiling);
            let (h, c) = hue(stop);
            println!(
                "  {label:5} noon S {:?} D {:?} a {:.2}  sigma_abs ({:.3}, {:.3}, {:.3}) /yd  \
                 sigma_turb {:.3} /yd  T=0.1 at {z:.1} yd  | now S {:?} D {:?}  deep stop {:?} \
                 hue {h:.0} chroma {:.0}",
                bytes(rows_noon[0]),
                bytes(rows_noon[1]),
                alpha_noon[0],
                s[0],
                s[1],
                s[2],
                o.sigma_turb,
                bytes(rows_now[0]),
                bytes(rows_now[1]),
                bytes(stop),
                c * 255.0,
            );
        }
        let minute_f = minute as f32;
        let mut light = WowLighting::default();
        light.sky = now.sky;
        light.fog_color = now.fog_color;
        light.celestial_dir = celestial_sun_direction(minute_f);
        let e = body_light(
            now.ambient,
            now.sun_diffuse,
            sun_direction(minute_f),
            light.celestial_dir,
            &dome_uniforms(&light),
            &look,
        );
        let enc = |c: [f32; 3]| bytes(c.map(|v| v.max(0.0).powf(1.0 / 2.2)));
        let open: [f32; 3] = std::array::from_fn(|i| e.e_sky[i] + e.e_sun[i]);
        let raw: [f32; 3] = std::array::from_fn(|i| e.e_sky_raw[i] + e.e_sun_raw[i]);
        println!(
            "  light  k_E {:.2}  sun share {:.2}  s0 {:.2}  night {:.2}  open E {:?}  raw E {:?}",
            e.k_e,
            e.sun_share,
            e.s0,
            e.night,
            enc(open),
            enc(raw),
        );
    }
    Ok(())
}
