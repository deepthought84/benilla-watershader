//! The Improved Water composite's blend algebra, mirrored on the CPU and tested before the shader
//! uses it: premultiplied linear layers over the snapshot, and the remainder the blender needs so
//! the gamma framebuffer lands on `enc(L + t·dec(background))`.
//!
//! The framebuffer holds gamma values and light adds in linear, so the water cannot hand the
//! blender its linear colour: `src + dst·β` sums in gamma. The water knows the snapshot `s`, which
//! is `dst` wherever nothing drew between the snapshot and the water, and outputs
//! `src0 = C − β·s`; the blender's `src0 + β·dst` then lands on `C` exactly for any `β`, and a
//! transparent drawn before the water shows through attenuated by `β`.

/// An RGB triple, linear or gamma as the name says.
type V3 = [f32; 3];

/// The display decode: a pure power law, so `enc(a·b) = enc(a)·enc(b)` holds for the residual.
const WATER_GAMMA: f32 = 2.2;
/// Rec. 709 luminance, which weighs the transmittance into the scalar blend factor.
const LUMA_709: V3 = [0.2126, 0.7152, 0.0722];
/// The layer weight's floor where fog divides a layer by its own coverage.
const WATER_LAYER_EPS: f32 = 1e-4;
/// Water's reflectance head-on: `((n − 1)/(n + 1))²` for n = 1.333 is 0.0204.
const F0_WATER: f32 = 0.02;

fn zip(a: V3, b: V3, f: impl Fn(f32, f32) -> f32) -> V3 {
    [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2])]
}

fn scale(a: V3, k: f32) -> V3 {
    a.map(|x| x * k)
}

fn add(a: V3, b: V3) -> V3 {
    zip(a, b, |x, y| x + y)
}

fn mul(a: V3, b: V3) -> V3 {
    zip(a, b, |x, y| x * y)
}

fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Gamma to linear, negatives clamped (no NaN from `powf`).
fn dec(g: V3) -> V3 {
    g.map(|x| x.max(0.0).powf(WATER_GAMMA))
}

/// Linear to gamma, negatives clamped.
fn enc(l: V3) -> V3 {
    l.map(|x| x.max(0.0).powf(1.0 / WATER_GAMMA))
}

/// One premultiplied layer: `l` the linear radiance it adds, `t` the fraction of what lies behind
/// that passes.
#[derive(Clone, Copy, Debug)]
struct Layer {
    l: V3,
    t: V3,
}

/// The surface: `F·R + S + (1 − F)·w_in⊙I` added, `(1 − F)·t_bed` passing. `w_in` weighs the
/// in-scatter and `t_bed` the bed, `w_in + t_bed ≤ 1`; a plain body transmittance `T` is the case
/// `w_in = 1 − T`, `t_bed = T`.
fn surface(f: f32, r: V3, s: V3, i: V3, w_in: V3, t_bed: V3) -> Layer {
    Layer {
        l: add(add(scale(r, f), s), scale(mul(w_in, i), 1.0 - f)),
        t: scale(t_bed, 1.0 - f),
    }
}

/// Edge coverage `c`: the layer scaled toward nothing.
fn cover(a: Layer, c: f32) -> Layer {
    Layer {
        l: scale(a.l, c),
        t: a.t.map(|t| 1.0 - c * (1.0 - t)),
    }
}

/// An opaque `colour` over the layer with coverage `cover` (foam).
fn over(colour: V3, cover: f32, a: Layer) -> Layer {
    Layer {
        l: add(scale(colour, cover), scale(a.l, 1.0 - cover)),
        t: scale(a.t, 1.0 - cover),
    }
}

/// The shader's gamma fog law (`apply_fog`) on the layer's own colour, `mix(fog_c, x, k)` with `k`
/// the share that stays, weighted by the layer's opacity so an empty layer adds nothing.
fn fog(a: Layer, k: f32, fog_c: V3) -> Layer {
    let o = a.t.map(|t| 1.0 - t);
    let own = enc(zip(a.l, o, |l, o| l / o.max(WATER_LAYER_EPS)));
    let fogged = zip(fog_c, own, |f, x| f + (x - f) * k);
    Layer {
        l: mul(o, dec(fogged)),
        t: a.t,
    }
}

/// The pixel the composite is after: `enc(L + t⊙dec(snap))`.
fn target(a: Layer, snap: V3) -> V3 {
    enc(add(a.l, mul(a.t, dec(snap))))
}

/// The scalar remainder: `β = enc(luma(t))`, `src0 = C − β·s`, alpha `1 − β`.
fn src_scalar(a: Layer, snap: V3) -> [f32; 4] {
    let c = target(a, snap);
    let beta = dot(a.t, LUMA_709).max(0.0).powf(1.0 / WATER_GAMMA);
    let rgb = zip(c, snap, |c, s| c - beta * s);
    [rgb[0], rgb[1], rgb[2], 1.0 - beta]
}

/// The dual-source remainder: `β = enc(t)` per channel; `src1 = (β, β_s)`.
fn src_dual(a: Layer, snap: V3) -> ([f32; 4], [f32; 4]) {
    let c = target(a, snap);
    let beta = enc(a.t);
    let beta_s = dot(a.t, LUMA_709).max(0.0).powf(1.0 / WATER_GAMMA);
    let rgb = zip(c, zip(beta, snap, |b, s| b * s), |c, bs| c - bs);
    (
        [rgb[0], rgb[1], rgb[2], 1.0 - beta_s],
        [beta[0], beta[1], beta[2], beta_s],
    )
}

/// Karis' analytic fit of the split-sum environment BRDF (2014), as the shader has it.
fn env_brdf_approx(f0: f32, n_dot_v: f32, rough: f32) -> f32 {
    let c0 = [-1.0, -0.0275, -0.572, 0.022];
    let c1 = [1.0, 0.0425, 1.04, -0.04];
    let r: [f32; 4] = std::array::from_fn(|k| rough * c0[k] + c1[k]);
    let a004 = (r[0] * r[0]).min((-9.28 * n_dot_v).exp2()) * r[0] + r[1];
    let ab = [-1.04 * a004 + r[2], 1.04 * a004 + r[3]];
    f0 * ab[0] + ab[1]
}

/// The fit's switch `μ_p`, where `min(r.x², exp2(−9.28μ))` changes arm: `−log2((1 − √α)²)/9.28`.
fn fresnel_kink(alpha: f32) -> f32 {
    let rough = alpha.clamp(0.0, 1.0).sqrt();
    (-((1.0 - rough) * (1.0 - rough)).max(1e-6).log2() / 9.28).clamp(0.0, 1.0)
}

/// Schlick scaled by the env-BRDF's roughness ratio, held at the kink below it so it never falls
/// toward grazing, and floored at `F0_WATER`: calm water reflects 2% head-on.
fn water_fresnel(n_dot_v: f32, alpha: f32) -> f32 {
    let rough = alpha.clamp(0.0, 1.0).sqrt();
    let mu = n_dot_v.max(fresnel_kink(alpha)).clamp(1e-4, 1.0);
    let schlick = F0_WATER + (1.0 - F0_WATER) * (1.0 - mu).powi(5);
    let k = env_brdf_approx(F0_WATER, mu, rough) / env_brdf_approx(F0_WATER, mu, 0.0);
    (schlick * k).clamp(0.0, 1.0).max(F0_WATER)
}

/// The grading's reach: roughness and distance as independent losses.
fn grade_rho(rho_a: f32, rho_c: f32) -> f32 {
    1.0 - (1.0 - rho_a) * (1.0 - rho_c)
}

/// The contrast kept at reach `rho` under strength `s` and shares `c_near`, `c_far`.
fn grade_keep(s: f32, c_near: f32, c_far: f32, rho: f32) -> f32 {
    (1.0 - s * (c_near + c_far * rho)).clamp(0.0, 1.0)
}

/// The two-stage grading: `k_n` about `p_near`, then the rest of `K` about `sky`.
fn grade_reflection(r: V3, sky: V3, p_near: V3, rho: f32, s: f32, c_near: f32, c_far: f32) -> V3 {
    let k = grade_keep(s, c_near, c_far, rho);
    let k_n = (1.0 - s * c_near).clamp(0.0, 1.0);
    let k_f = if k_n <= 1e-4 { 0.0 } else { k / k_n.max(1e-4) };
    let r1 = add(p_near, scale(zip(r, p_near, |a, b| a - b), k_n));
    zip(
        add(sky, scale(zip(r1, sky, |a, b| a - b), k_f)),
        [0.0; 3],
        f32::max,
    )
}

/// How the blender's arithmetic is modelled: plain f32, or a half-float target whose sources and
/// result round through `f16` (the framebuffer's own values are f16 already).
#[derive(Clone, Copy, Debug)]
enum Blender {
    F32,
    F16,
}

impl Blender {
    fn round(self, x: f32) -> f32 {
        match self {
            Self::F32 => x,
            Self::F16 => half::f16::from_f32(x).to_f32(),
        }
    }

    /// `(One, OneMinusSrcAlpha)`: `src.rgb + (1 − src.a)·dst`.
    fn premul(self, src: [f32; 4], dst: V3) -> V3 {
        let a = 1.0 - self.round(src[3]);
        std::array::from_fn(|k| self.round(self.round(src[k]) + a * dst[k]))
    }

    /// Colour `(One, Src1)`: `src0.rgb + src1.rgb⊙dst`.
    fn dual(self, src0: [f32; 4], src1: [f32; 4], dst: V3) -> V3 {
        std::array::from_fn(|k| self.round(self.round(src0[k]) + self.round(src1[k]) * dst[k]))
    }
}

/// A seeded LCG (Knuth's MMIX constants) over `[0, 1)`: fixed cases without a `rand` dependency.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    fn v3(&mut self, lo: f32, hi: f32) -> V3 {
        [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
    }

    /// A framebuffer colour: a half float, as the target stores it.
    fn texel(&mut self, hi: f32) -> V3 {
        self.v3(0.0, hi).map(|x| half::f16::from_f32(x).to_f32())
    }
}

/// One random case: a surface, covered, foamed and fogged, and the snapshot behind it. The surface
/// weighs its in-scatter by a scalar under `1 − max(t_bed)`, or, with `plain_t`, is a plain body
/// transmittance (`w_in = 1 − t_bed`).
struct Case {
    layer: Layer,
    snap: V3,
}

fn random_case(rng: &mut Lcg, plain_t: bool) -> Case {
    let f = rng.range(0.02, 1.0);
    let t_bed = rng.v3(0.0, 1.0);
    let w_in = if plain_t {
        t_bed.map(|t| 1.0 - t)
    } else {
        let max = t_bed.iter().copied().fold(0.0, f32::max);
        [rng.range(0.0, 1.0 - max); 3]
    };
    let (r, i, s) = (rng.v3(0.0, 1.2), rng.v3(0.0, 1.2), rng.v3(0.0, 1.2));
    let mut layer = surface(f, r, s, i, w_in, t_bed);
    layer = cover(layer, rng.unit());
    layer = over(rng.v3(0.0, 1.2), rng.unit(), layer);
    layer = fog(layer, rng.unit(), rng.v3(0.0, 1.0));
    Case {
        layer,
        snap: rng.texel(1.2),
    }
}

fn max_diff(a: V3, b: V3) -> f32 {
    (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f32::max)
}

#[test]
fn the_remainder_lands_on_the_linear_composite_where_dst_is_the_snapshot() {
    let mut rng = Lcg(0x5EED_0001);
    let bound = 2f32.powi(-10);
    for n in 0..20_000 {
        let Case { layer, snap } = random_case(&mut rng, n % 2 == 1);
        let c = target(layer, snap);
        let scalar = src_scalar(layer, snap);
        let (src0, src1) = src_dual(layer, snap);
        for blender in [Blender::F32, Blender::F16] {
            let got = blender.premul(scalar, snap);
            assert!(
                max_diff(got, c) <= bound,
                "case {n}, {blender:?} scalar: {got:?} against {c:?}"
            );
            let got = blender.dual(src0, src1, snap);
            assert!(
                max_diff(got, c) <= bound,
                "case {n}, {blender:?} dual: {got:?} against {c:?}"
            );
        }
    }
}

#[test]
fn the_composite_is_the_linear_sum_encoded_once() {
    let mut rng = Lcg(0x5EED_0002);
    for n in 0..20_000 {
        // Gamma inputs, as the shader holds them before decoding.
        let (f, cov, foam, k) = (rng.range(0.02, 1.0), rng.unit(), rng.unit(), rng.unit());
        let t_bed = rng.v3(0.0, 1.0);
        let w_in = t_bed.map(|t| 1.0 - t);
        let (r_g, i_g, foam_g, fog_g) = (
            rng.v3(0.0, 1.0),
            rng.v3(0.0, 1.0),
            rng.v3(0.0, 1.0),
            rng.v3(0.0, 1.0),
        );
        let s = rng.v3(0.0, 0.5);
        let snap = rng.texel(1.0);
        let layer = fog(
            over(
                dec(foam_g),
                foam,
                cover(surface(f, dec(r_g), s, dec(i_g), w_in, t_bed), cov),
            ),
            k,
            fog_g,
        );
        let got = target(layer, snap);
        // The same pixel written out term by term, per channel.
        let want: V3 = std::array::from_fn(|ch| {
            let dec1 = |g: f32| g.powf(2.2);
            let surface_l = f * dec1(r_g[ch]) + s[ch] + (1.0 - f) * w_in[ch] * dec1(i_g[ch]);
            let surface_t = (1.0 - f) * t_bed[ch];
            let covered_l = cov * surface_l;
            let covered_t = 1.0 - cov * (1.0 - surface_t);
            let foamed_l = foam * dec1(foam_g[ch]) + (1.0 - foam) * covered_l;
            let foamed_t = (1.0 - foam) * covered_t;
            let o = 1.0 - foamed_t;
            let own = (foamed_l / o.max(1e-4)).powf(1.0 / 2.2);
            let fogged_l = o * (fog_g[ch] + (own - fog_g[ch]) * k).max(0.0).powf(2.2);
            (fogged_l + foamed_t * dec1(snap[ch])).powf(1.0 / 2.2)
        });
        assert!(
            max_diff(got, want) <= 1e-5,
            "case {n}: {got:?} against {want:?}"
        );
    }
}

#[test]
fn the_gamma_sum_is_forty_three_percent_too_bright() {
    // A reflection of 0.2 over a bed of 0.1, both linear, nothing absorbed.
    let layer = Layer {
        l: [0.2; 3],
        t: [1.0; 3],
    };
    let snap = enc([0.1; 3]);
    let ours = target(layer, snap)[0];
    assert!((ours - 0.579).abs() <= 1e-3, "{ours}");
    let through_blender = Blender::F32.premul(src_scalar(layer, snap), snap)[0];
    assert!((through_blender - ours).abs() <= 1e-6);
    // Summed on the gamma target: enc(0.2) + enc(0.1).
    let naive = enc([0.2; 3])[0] + snap[0];
    assert!((naive - 0.832).abs() <= 1e-3, "{naive}");
    assert!(naive / ours > 1.43, "{naive} against {ours}");
}

#[test]
fn a_transparent_drawn_before_the_water_is_attenuated_by_beta() {
    let mut rng = Lcg(0x5EED_0004);
    for n in 0..5_000 {
        let Case { layer, snap } = random_case(&mut rng, n % 2 == 1);
        let delta = rng.v3(0.0, 0.3);
        let dst = add(snap, delta);
        let c = target(layer, snap);
        // The blend factors from their definitions: the encoded transmittance, per channel and
        // as its Rec. 709 luminance.
        let beta_s = dot(layer.t, LUMA_709).powf(1.0 / 2.2);
        let beta = enc(layer.t);
        let got = Blender::F32.premul(src_scalar(layer, snap), dst);
        assert!(
            max_diff(got, add(c, scale(delta, beta_s))) <= 1e-3,
            "case {n} scalar"
        );
        let (src0, src1) = src_dual(layer, snap);
        let got = Blender::F32.dual(src0, src1, dst);
        assert!(
            max_diff(got, add(c, mul(beta, delta))) <= 1e-3,
            "case {n} dual"
        );
        // A layer that adds no light: the dual residual is the power law's own product.
        let clear = Layer {
            l: [0.0; 3],
            t: layer.t,
        };
        let (src0, src1) = src_dual(clear, snap);
        let got = Blender::F32.dual(src0, src1, dst);
        let want = enc(mul(clear.t, dec(dst)));
        assert!(
            max_diff(got, want) <= 1e-4,
            "case {n} clear: {got:?} {want:?}"
        );
    }
}

#[test]
fn opaque_water_ignores_the_framebuffer_and_empty_water_leaves_it() {
    let mut rng = Lcg(0x5EED_0005);
    let bound = 2f32.powi(-11);
    for n in 0..5_000 {
        let (r, i, s) = (rng.v3(0.0, 1.0), rng.v3(0.0, 1.0), rng.v3(0.0, 0.5));
        let snap = rng.texel(1.0);
        let other = rng.texel(1.0);
        let opaque = cover(
            surface(rng.range(0.02, 1.0), r, s, i, [1.0; 3], [0.0; 3]),
            1.0,
        );
        for blender in [Blender::F32, Blender::F16] {
            let src = src_scalar(opaque, snap);
            assert_eq!(1.0 - src[3], 0.0, "case {n}: opaque water passes nothing");
            assert_eq!(
                blender.premul(src, snap),
                blender.premul(src, other),
                "case {n}"
            );
            let (src0, src1) = src_dual(opaque, snap);
            assert_eq!(
                blender.dual(src0, src1, snap),
                blender.dual(src0, src1, other)
            );
        }
        let empty = cover(
            surface(rng.range(0.02, 1.0), r, s, i, [0.5; 3], [0.5; 3]),
            0.0,
        );
        for blender in [Blender::F32, Blender::F16] {
            let got = blender.premul(src_scalar(empty, snap), snap);
            assert!(
                max_diff(got, snap) <= bound,
                "case {n} {blender:?}: {got:?} {snap:?}"
            );
            let (src0, src1) = src_dual(empty, snap);
            let got = blender.dual(src0, src1, snap);
            assert!(max_diff(got, snap) <= bound, "case {n} {blender:?} dual");
        }
    }
}

#[test]
fn fog_is_todays_law_on_opaque_water_and_absent_on_clear() {
    let mut rng = Lcg(0x5EED_0006);
    for n in 0..5_000 {
        let (l, fog_c, k, snap) = (
            rng.v3(0.0, 1.2),
            rng.v3(0.0, 1.0),
            rng.unit(),
            rng.texel(1.2),
        );
        let opaque = fog(Layer { l, t: [0.0; 3] }, k, fog_c);
        let todays = zip(fog_c, enc(l), |f, x| f + (x - f) * k);
        assert!(
            max_diff(target(opaque, snap), todays) <= 1e-5,
            "case {n} opaque"
        );
        let clear = fog(
            Layer {
                l: [0.0; 3],
                t: [1.0; 3],
            },
            k,
            fog_c,
        );
        assert!(
            max_diff(target(clear, snap), snap) <= 1e-5,
            "case {n} clear"
        );
    }
}

#[test]
fn the_output_stays_finite() {
    let mut rng = Lcg(0x5EED_0007);
    for n in 0..5_000 {
        let snap = rng.texel(8.0);
        let layer = Layer {
            l: rng.v3(0.0, 8.0),
            t: rng.v3(0.0, 1.0),
        };
        let scalar = src_scalar(layer, snap);
        let (src0, src1) = src_dual(layer, snap);
        // The remainder can reach −8 where a bright snapshot sits behind clear water.
        let low = [-8.0, -8.0, -8.0, scalar[3]];
        for blender in [Blender::F32, Blender::F16] {
            for got in [
                blender.premul(scalar, snap),
                blender.premul(low, snap),
                blender.dual(src0, src1, snap),
            ] {
                assert!(got.iter().all(|x| x.is_finite()), "case {n}: {got:?}");
            }
        }
    }
}

#[test]
fn calm_water_reflects_two_percent_head_on_and_fresnel_is_monotone() {
    assert!((water_fresnel(1.0, 0.0) - 0.02).abs() <= 1e-6);
    // The fit reads 0.0165 head-on at α0; the floor holds it at F0.
    assert_eq!(water_fresnel(1.0, 0.0548), F0_WATER);
    for alpha in [0.0, 0.0548, 0.08, 0.1037] {
        let mut prev = f32::INFINITY;
        for step in 1..=10_000 {
            let mu = step as f32 * 1e-4;
            let f = water_fresnel(mu, alpha);
            assert!(
                f <= prev + 1e-7,
                "α {alpha}: F rises at μ {mu}: {prev} → {f}"
            );
            assert!((F0_WATER..=1.0).contains(&f), "α {alpha}, μ {mu}: {f}");
            prev = f;
        }
    }
    // The plateau below the kink, and the kink itself.
    for (alpha, mu_p, plateau) in [
        (0.0548, 0.0829, 0.5038),
        (0.08, 0.1034, 0.4237),
        (0.1037, 0.1208, 0.3648),
    ] {
        assert!((fresnel_kink(alpha) - mu_p).abs() <= 5e-4, "α {alpha}");
        for mu in [0.0, 1e-3, mu_p * 0.5, mu_p - 1e-3] {
            let f = water_fresnel(mu, alpha);
            assert!((f - plateau).abs() <= 5e-4, "α {alpha}, μ {mu}: {f}");
        }
    }
    // Calm water's roughness ratio is 1: plain Schlick.
    for step in 0..=100 {
        let mu = (step as f32 / 100.0).max(1e-4);
        let schlick = F0_WATER + (1.0 - F0_WATER) * (1.0 - mu).powi(5);
        assert!((water_fresnel(mu, 0.0) - schlick).abs() <= 1e-6, "μ {mu}");
    }
}

/// The shader's colour transfer is this mirror's: the same gamma and the same luminance weights.
#[test]
fn the_shaders_transfer_is_the_mirrors() {
    let wgsl = include_str!("shaders/liquid.wgsl");
    let gamma = format!("const WATER_GAMMA: f32 = {WATER_GAMMA:?};");
    assert!(wgsl.contains(&gamma), "liquid.wgsl lost `{gamma}`");
    let [r, g, b] = LUMA_709;
    let luma = format!("const LUMA_709: vec3<f32> = vec3<f32>({r:?}, {g:?}, {b:?});");
    assert!(wgsl.contains(&luma), "liquid.wgsl lost `{luma}`");
}

/// The shader's composite is this mirror's algebra: the layers, the fog guard, the scalar β and
/// the two remainders, line for line.
#[test]
fn the_shaders_composite_is_the_mirrors() {
    let wgsl = include_str!("shaders/liquid.wgsl");
    assert_eq!(WATER_LAYER_EPS, 1e-4);
    for line in [
        "const WATER_LAYER_EPS: f32 = 1e-4;",
        "return WaterLayer(f * r + s + (1.0 - f) * w_in * i, (1.0 - f) * t_bed);",
        "return WaterLayer(c * a.l, vec3<f32>(1.0) - c * (vec3<f32>(1.0) - a.t));",
        "return WaterLayer(cover * colour + (1.0 - cover) * a.l, (1.0 - cover) * a.t);",
        "let own = water_enc(a.l / max(o, vec3<f32>(WATER_LAYER_EPS)));",
        "return WaterLayer(o * water_dec(apply_fog(own, world_pos, room_fog)), a.t);",
        "let c_lin = a.l + a.t * bed;",
        "let beta = pow(max(dot(a.t, LUMA_709), 0.0), 1.0 / WATER_GAMMA);",
        "return vec4<f32>(c - beta * snap_px, 1.0 - beta);",
        "let beta3 = water_enc(a.t);",
        "return vec4<f32>(c - beta3 * snap_px, 1.0 - beta);",
    ] {
        assert!(wgsl.contains(line), "liquid.wgsl lost `{line}`");
    }
}

/// The shader's reflectance is this mirror's: the fit's coefficients, its kink, the held cosine and
/// the floor, line for line.
#[test]
fn the_shaders_fresnel_is_the_mirrors() {
    let wgsl = include_str!("shaders/liquid.wgsl");
    assert_eq!(F0_WATER, 0.02);
    for line in [
        "const F0_WATER: f32 = 0.02;",
        "let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);",
        "let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);",
        "let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;",
        "let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;",
        "return f0 * ab.x + ab.y;",
        "return clamp(-log2(max((1.0 - rough) * (1.0 - rough), 1e-6)) / 9.28, 0.0, 1.0);",
        "let mu = clamp(max(n_dot_v, fresnel_kink(alpha)), 1e-4, 1.0);",
        "let schlick = F0_WATER + (1.0 - F0_WATER) * pow(1.0 - mu, 5.0);",
        "let k = env_brdf_approx(F0_WATER, mu, rough) / env_brdf_approx(F0_WATER, mu, 0.0);",
        "return max(saturate(schlick * k), F0_WATER);",
    ] {
        assert!(wgsl.contains(line), "liquid.wgsl lost `{line}`");
    }
}

/// At s = 0 the grading is the identity. Where the floor's ray escaped (the near pivot is the open
/// sky) it keeps `K` of the reflection's distance from the sky; where the march has no verdict
/// (the pivot is the reflection) only the far stage acts, `K / k_n`; at reach 0 only the near stage
/// acts: the reflection is pulled toward the occluder by `1 − k_n`, never toward the sky.
#[test]
fn the_grading_keeps_k_and_pulls_shaded_water_toward_its_occluder() {
    let (c_near, c_far) = (0.2, 0.3);
    let mut rng = Lcg(0x5EED_0009);
    for _ in 0..20_000 {
        let (r, sky, occ) = (rng.v3(0.0, 1.2), rng.v3(0.0, 1.2), rng.v3(0.0, 1.2));
        let rho = grade_rho(rng.unit(), rng.unit());
        assert!((0.0..=1.0).contains(&rho));
        let same = grade_reflection(r, sky, occ, rho, 0.0, c_near, c_far);
        assert!(max_diff(same, r) <= 1e-6, "s 0 moved {r:?} to {same:?}");
        let s = rng.range(0.0, 2.4);
        let k = grade_keep(s, c_near, c_far, rho);
        let k_n = (1.0 - s * c_near).clamp(0.0, 1.0);
        let about_sky = |kept: f32| {
            zip(
                add(sky, scale(zip(r, sky, |a, b| a - b), kept)),
                [0.0; 3],
                f32::max,
            )
        };
        let escaped = grade_reflection(r, sky, sky, rho, s, c_near, c_far);
        assert!(max_diff(escaped, about_sky(k)) <= 1e-5, "s {s}, rho {rho}");
        if k_n > 1e-4 {
            let unknown = grade_reflection(r, sky, r, rho, s, c_near, c_far);
            assert!(
                max_diff(unknown, about_sky(k / k_n)) <= 1e-5,
                "s {s}, rho {rho}: the far stage alone"
            );
        }
        let near = grade_reflection(r, sky, occ, 0.0, s, c_near, c_far);
        let want = add(occ, scale(zip(r, occ, |a, b| a - b), k_n));
        assert!(max_diff(near, want) <= 1e-5, "s {s}: the near stage alone");
    }
    // Under a deck: a dark reflection of the underside, a bright sky. The near pivot keeps it dark;
    // a single pivot on the sky would lift it toward the sky.
    let (under, sky) = ([0.03, 0.03, 0.025], [0.6, 0.7, 0.9]);
    let graded = grade_reflection(under, sky, under, 0.0, 1.0, c_near, c_far);
    assert!(max_diff(graded, under) <= 1e-6);
    let single = zip(
        add(
            sky,
            scale(
                zip(under, sky, |a, b| a - b),
                grade_keep(1.0, c_near, c_far, 0.0),
            ),
        ),
        [0.0; 3],
        f32::max,
    );
    assert!(dot(single, LUMA_709) > 2.0 * dot(graded, LUMA_709));
}

/// The shader's grading is this mirror's, line for line.
#[test]
fn the_shaders_grading_is_the_mirrors() {
    let wgsl = include_str!("shaders/liquid.wgsl");
    for line in [
        "return 1.0 - (1.0 - rho_a) * (1.0 - rho_c);",
        "return clamp(1.0 - g.x * (g.z + g.y * rho), 0.0, 1.0);",
        "let k_n = clamp(1.0 - g.x * g.z, 0.0, 1.0);",
        "let k_f = select(k / max(k_n, 1e-4), 0.0, k_n <= 1e-4);",
        "let r1 = p_near + k_n * (r - p_near);",
        "return max(sky + k_f * (r1 - sky), vec3<f32>(0.0));",
        "let p_near = mix(refl_l, occ.rgb + occ.v * sky, occ.known);",
    ] {
        assert!(wgsl.contains(line), "liquid.wgsl lost `{line}`");
    }
}
