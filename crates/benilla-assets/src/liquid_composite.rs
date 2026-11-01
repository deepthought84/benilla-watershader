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

/// The shader's gamma fog law (`apply_fog`) on one diffuse term's linear radiance `c`:
/// `mix(fog_c, enc(c), k)` decoded, with `k` the share that stays (`fog_diffuse`), and `c` as it
/// came where the fog leaves all of it. Each term is fogged once on its own; the layer is never
/// fogged as a whole, so the bed it passes keeps the fog the frame drew it with.
fn fog_diffuse(c: V3, k: f32, fog_c: V3) -> V3 {
    if k >= 1.0 {
        return c;
    }
    dec(zip(fog_c, enc(c), |f, x| f + (x - f) * k))
}

/// GLSL's `smoothstep`.
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A gamma colour `c` fogged at the share `was` re-fogged at the share `now` in the fog colour `g`
/// (`fog_rebase`, the probe's re-fog): un-fogged, its un-fog held within
/// 0.25 of the colour or the fog, and faded to the colour as it came where under 0.35 of it
/// survived the first fog, down to none at 0.2.
fn fog_rebase(c: V3, was: f32, now: f32, g: V3) -> V3 {
    let w = smoothstep(0.2, 0.35, was);
    std::array::from_fn(|ch| {
        let clear =
            ((c[ch] - (1.0 - was) * g[ch]) / was.max(0.2)).clamp(0.0, c[ch].max(g[ch]) + 0.25);
        let refogged = g[ch] + (clear - g[ch]) * now;
        c[ch] + (refogged - c[ch]) * w
    })
}

/// The reach at which a reflected thing counts as half far, yards (`FLOOR_CONTACT_REACH`).
const FLOOR_CONTACT_REACH: f32 = 64.0;

/// How far behind the water a reflected thing lies at distance `d` (`floor_rho_c`).
fn floor_rho_c(d: f32) -> f32 {
    d * d / (d * d + FLOOR_CONTACT_REACH * FLOOR_CONTACT_REACH)
}

/// The distance [`floor_rho_c`] was taken at, from its value (`floor_tone`).
fn floor_reach(rho: f32) -> f32 {
    FLOOR_CONTACT_REACH * (rho.max(0.0) / (1.0 - rho)).sqrt()
}

/// The mirror's gamma colour `m_g` in linear light with the share of it that is the dome fogged at
/// the surface's share `k` (`mirror_fog`): the dome share is the colour's agreement with the sky
/// floor `sky` (linear), all of it within a fifth, none past two fifths.
fn mirror_fog(m_g: V3, sky: V3, k: f32, fog_c: V3) -> V3 {
    let m = dec(m_g);
    if k >= 1.0 {
        return m;
    }
    let len = |v: V3| dot(v, v).sqrt();
    let off = len(zip(m, sky, |a, b| a - b)) / len(sky).max(1e-3);
    let dome = 1.0 - smoothstep(0.2, 0.4, off);
    let share = 1.0 + (k - 1.0) * dome;
    dec(zip(fog_c, m_g, |f, x| f + (x - f) * share))
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

/// The two-stage grading: `k_n` about `p_near`, then the rest of `K` about `p_far`.
fn grade_reflection(r: V3, p_far: V3, p_near: V3, rho: f32, s: f32, c_near: f32, c_far: f32) -> V3 {
    let k = grade_keep(s, c_near, c_far, rho);
    let k_n = (1.0 - s * c_near).clamp(0.0, 1.0);
    let k_f = if k_n <= 1e-4 { 0.0 } else { k / k_n.max(1e-4) };
    let r1 = add(p_near, scale(zip(r, p_near, |a, b| a - b), k_n));
    zip(
        add(p_far, scale(zip(r1, p_far, |a, b| a - b), k_f)),
        [0.0; 3],
        f32::max,
    )
}

/// The two-stage grading as the shader runs it: affine in the reflection `r`, `max(a·r + b, 0)`;
/// `(a, b)`. The occluded `floor` is `known·P + (1 − known)·sky`, `P` the floor of the share with a
/// verdict (an unknown share carries no occluder and the whole sky), so the far pivot is `floor`
/// itself and the near pivot `mix(r, P, known)` is `floor + (1 − known)·(r − sky)`, expanded.
fn grade_affine(
    sky: V3,
    floor: V3,
    known: f32,
    rho: f32,
    s: f32,
    c_near: f32,
    c_far: f32,
) -> (f32, V3) {
    let k = grade_keep(s, c_near, c_far, rho);
    let k_n = (1.0 - s * c_near).clamp(0.0, 1.0);
    let k_f = if k_n <= 1e-4 { 0.0 } else { k / k_n.max(1e-4) };
    let a = k_f + (k - k_f) * known;
    let b = add(
        scale(floor, 1.0 - k),
        scale(sky, -(k_f - k) * (1.0 - known)),
    );
    (a, b)
}

/// The affine map applied: `max(a·r + b, 0)`.
fn grade_apply((a, b): (f32, V3), r: V3) -> V3 {
    zip(add(scale(r, a), b), [0.0; 3], f32::max)
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

/// One random case: a surface with its in-scatter, glints and foam fogged, covered and foamed, and
/// the snapshot behind it. The surface weighs its in-scatter by a scalar under `1 − max(t_bed)`, or,
/// with `plain_t`, is a plain body transmittance (`w_in = 1 − t_bed`).
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
    let (k, fog_c) = (rng.unit(), rng.v3(0.0, 1.0));
    let mut layer = surface(f, r, scale(s, k), fog_diffuse(i, k, fog_c), w_in, t_bed);
    layer = cover(layer, rng.unit());
    layer = over(fog_diffuse(rng.v3(0.0, 1.2), k, fog_c), rng.unit(), layer);
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
        let layer = over(
            fog_diffuse(dec(foam_g), k, fog_g),
            foam,
            cover(
                surface(
                    f,
                    dec(r_g),
                    scale(s, k),
                    fog_diffuse(dec(i_g), k, fog_g),
                    w_in,
                    t_bed,
                ),
                cov,
            ),
        );
        let got = target(layer, snap);
        // The same pixel written out term by term, per channel: the in-scatter and foam fogged in
        // gamma, the glints scaled by the share that stays, the reflection and the bed as they came.
        let want: V3 = std::array::from_fn(|ch| {
            let dec1 = |g: f32| g.powf(2.2);
            let fogged = |g: f32| (fog_g[ch] + (g - fog_g[ch]) * k).powf(2.2);
            let surface_l = f * dec1(r_g[ch]) + k * s[ch] + (1.0 - f) * w_in[ch] * fogged(i_g[ch]);
            let surface_t = (1.0 - f) * t_bed[ch];
            let covered_l = cov * surface_l;
            let covered_t = 1.0 - cov * (1.0 - surface_t);
            let foamed_l = foam * fogged(foam_g[ch]) + (1.0 - foam) * covered_l;
            let foamed_t = (1.0 - foam) * covered_t;
            (foamed_l + foamed_t * dec1(snap[ch])).powf(1.0 / 2.2)
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

/// Each term is fogged once. Opaque water that only scatters fogs as 1.12 fogs a surface, the
/// bed a clear layer passes keeps the frame's own fog, a share of 1 changes nothing, and in a full
/// fog the in-scatter and foam become the fog's colour, the glints go and the reflection stays: the
/// far water converges on `F·R + (1 − F)·fog`, the mirror's fogged horizon over the fog itself.
#[test]
fn each_term_is_fogged_once() {
    let mut rng = Lcg(0x5EED_0006);
    for n in 0..5_000 {
        let (i, fog_c, k, snap) = (
            rng.v3(0.0, 1.2),
            rng.v3(0.0, 1.0),
            rng.unit(),
            rng.texel(1.2),
        );
        let matte = surface(
            0.0,
            [0.0; 3],
            [0.0; 3],
            fog_diffuse(i, k, fog_c),
            [1.0; 3],
            [0.0; 3],
        );
        let todays = zip(fog_c, enc(i), |f, x| f + (x - f) * k);
        assert!(
            max_diff(target(matte, snap), todays) <= 1e-5,
            "case {n} matte"
        );
        let clear = surface(0.0, [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3], [1.0; 3]);
        assert!(
            max_diff(target(clear, snap), snap) <= 1e-5,
            "case {n} clear"
        );
        assert_eq!(fog_diffuse(i, 1.0, fog_c), i, "case {n} unfogged");
        let (f, r, s) = (rng.range(0.02, 1.0), rng.v3(0.0, 1.2), rng.v3(0.0, 1.2));
        let far = surface(
            f,
            r,
            scale(s, 0.0),
            fog_diffuse(i, 0.0, fog_c),
            [1.0; 3],
            [0.0; 3],
        );
        let want = enc(add(scale(r, f), scale(dec(fog_c), 1.0 - f)));
        assert!(max_diff(target(far, snap), want) <= 1e-5, "case {n} far");
    }
}

/// The re-base is exact where the first fog left enough to un-fog: a clear colour fogged at one
/// share and re-based to another is the clear colour fogged at the second, so a reflected hill the
/// probe's face saw nearer than the mirror would comes out fogged as the mirror's. It never
/// amplifies past 0.25 over the colour or the fog, and under a share of 0.2 it keeps the colour as
/// it came. The same share in and out is the identity.
#[test]
fn the_fog_rebase_is_exact_where_it_can_be_and_bounded_where_it_cannot() {
    let mut rng = Lcg(0x5EED_000A);
    let mut exact = 0;
    for n in 0..20_000 {
        let (x, g) = (rng.v3(0.0, 1.0), rng.v3(0.0, 1.0));
        let (was, now) = (rng.range(0.35, 1.0), rng.unit());
        let fog_at = |c: V3, k: f32| zip(g, c, |g, c| g + (c - g) * k);
        let fogged = fog_at(x, was);
        let got = fog_rebase(fogged, was, now, g);
        if (0..3).all(|ch| x[ch] <= fogged[ch].max(g[ch]) + 0.25) {
            exact += 1;
            assert!(
                max_diff(got, fog_at(x, now)) <= 1e-5,
                "case {n}: {x:?} at {was} to {now}"
            );
            let same = fog_rebase(fogged, was, was, g);
            assert!(
                max_diff(same, fogged) <= 1e-5,
                "case {n}: the same share moved it"
            );
        }
        let c = rng.v3(0.0, 1.0);
        let low = rng.range(0.0, 0.35);
        let got = fog_rebase(c, low, rng.unit(), g);
        for ch in 0..3 {
            assert!(
                got[ch] >= 0.0 && got[ch] <= c[ch].max(g[ch]) + 0.25 + 1e-6,
                "case {n}: {got:?} from {c:?} at {low}"
            );
        }
        let lost = rng.range(0.0, 0.2);
        assert_eq!(fog_rebase(c, lost, rng.unit(), g), c, "case {n} at {lost}");
    }
    assert!(
        exact > 15_000,
        "only {exact} of 20000 cases un-fog unbounded"
    );
}

/// The floor's re-fog finds the reflected thing's distance from `rho_c` again, to a hundredth of a
/// yard per hundred over the 2000 yd it re-fogs (ρ < 0.999), so the image beyond the surface is
/// where the march put it.
#[test]
fn the_floors_distance_comes_back_out_of_rho_c() {
    for step in 0..=2000 {
        let d = step as f32;
        let rho = floor_rho_c(d);
        if rho >= 0.999 {
            break;
        }
        assert!(
            (floor_reach(rho) - d).abs() <= 0.01 * (1.0 + d / 100.0),
            "{d} yd: {rho}"
        );
    }
    assert!(floor_rho_c(1999.0) < 0.999 && floor_rho_c(2030.0) >= 0.999);
}

/// The mirror's reflection of the dome is fogged at the surface and its reflection of the world is
/// not: a texel showing the sky floor's own colour takes the surface's fog as the in-scatter does, a
/// texel a far cry from the sky (a hill, a mast) keeps the fog its image was drawn with, and with
/// the fog leaving everything the colour is the plain decode.
#[test]
fn the_mirror_fogs_the_dome_it_drew_and_not_the_world() {
    let mut rng = Lcg(0x5EED_000B);
    for n in 0..5_000 {
        let (sky_g, fog_c, k) = (rng.v3(0.2, 1.0), rng.v3(0.0, 1.0), rng.unit() * 0.999);
        let sky = dec(sky_g);
        let dome = mirror_fog(sky_g, sky, k, fog_c);
        assert!(
            max_diff(dome, fog_diffuse(sky, k, fog_c)) <= 1e-5,
            "case {n}: the dome {sky_g:?} at {k}"
        );
        let world_g = scale(sky_g, 0.3);
        assert!(
            max_diff(mirror_fog(world_g, sky, k, fog_c), dec(world_g)) <= 1e-6,
            "case {n} the world"
        );
        assert_eq!(mirror_fog(sky_g, sky, 1.0, fog_c), sky, "case {n} unfogged");
    }
}

/// The shader's re-fog is this mirror's: the bounded un-fog, its fade, and the floor's distance
/// from `rho_c` against the law that made it, line for line.
#[test]
fn the_shaders_refog_is_the_mirrors() {
    let wgsl = include_str!("shaders/liquid.wgsl");
    for line in [
        "(c_g - (1.0 - f_was) * g) / max(f_was, 0.2),",
        "max(c_g, g) + vec3<f32>(0.25),",
        "return mix(c_g, mix(g, clear, f_now), smoothstep(0.2, 0.35, f_was));",
        "const FLOOR_CONTACT_REACH: f32 = 64.0;",
        "return d2 / (d2 + FLOOR_CONTACT_REACH * FLOOR_CONTACT_REACH);",
        "let d = FLOOR_CONTACT_REACH * sqrt(max(occ.rho_c, 0.0) / (1.0 - occ.rho_c));",
        "let off = length(m - sky) / max(length(sky), 1e-3);",
        "let dome = 1.0 - smoothstep(0.2, 0.4, off);",
        "return water_dec(mix(fog.g, m_g, mix(1.0, fog.f, dome)));",
    ] {
        assert!(wgsl.contains(line), "liquid.wgsl lost `{line}`");
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

/// A 10-bit unorm texel, as the snapshot (`Rgb10a2Unorm`) stores a framebuffer colour.
fn ten_bit(c: V3) -> V3 {
    c.map(|x| (x.clamp(0.0, 1.0) * 1023.0).round() / 1023.0)
}

#[test]
fn a_ten_bit_snapshot_is_within_half_a_step() {
    let mut rng = Lcg(0x5EED_0009);
    let step = 0.5 / 1023.0;
    let slack = 2f32.powi(-10);
    for n in 0..20_000 {
        let Case { layer, .. } = random_case(&mut rng, n % 2 == 1);
        // The framebuffer holds the half-float frame; the water reads its 10-bit snapshot.
        let dst = rng.texel(1.0);
        let snap = ten_bit(dst);
        let c = target(layer, snap);
        let beta_s = dot(layer.t, LUMA_709).max(0.0).powf(1.0 / WATER_GAMMA);
        let beta = enc(layer.t).iter().copied().fold(0.0, f32::max);
        let scalar = src_scalar(layer, snap);
        let (src0, src1) = src_dual(layer, snap);
        for blender in [Blender::F32, Blender::F16] {
            let got = blender.premul(scalar, dst);
            assert!(
                max_diff(got, c) <= beta_s * step + slack,
                "case {n}, {blender:?} scalar: {got:?} against {c:?}"
            );
            let got = blender.dual(src0, src1, dst);
            assert!(
                max_diff(got, c) <= beta * step + slack,
                "case {n}, {blender:?} dual: {got:?} against {c:?}"
            );
        }
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

/// The shader's composite is this mirror's algebra: the layers, the fog on a diffuse term, the
/// scalar β and the two remainders, line for line.
#[test]
fn the_shaders_composite_is_the_mirrors() {
    let wgsl = include_str!("shaders/liquid.wgsl");
    for line in [
        "return WaterLayer(f * r + s + (1.0 - f) * w_in * i, (1.0 - f) * t_bed);",
        "return WaterLayer(c * a.l, vec3<f32>(1.0) - c * (vec3<f32>(1.0) - a.t));",
        "return WaterLayer(cover * colour + (1.0 - cover) * a.l, (1.0 - cover) * a.t);",
        "return water_dec(mix(g, water_enc(c), f));",
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

/// The grading's numbers at the owner's band midpoints (C_NEAR 0.2, C_FAR 0.3): the reach of a
/// rough share 0.3 and a distance share 0.4 is 0.58, and at s = 1 the contrast kept is 0.8 at
/// reach 0 and 0.5 at reach 1.
#[test]
fn the_grading_keeps_the_band_midpoints() {
    let (c_near, c_far) = (0.2, 0.3);
    assert!((grade_rho(0.3, 0.4) - 0.58).abs() <= 1e-6);
    assert!((grade_keep(1.0, c_near, c_far, 0.0) - 0.8).abs() <= 1e-6);
    assert!((grade_keep(1.0, c_near, c_far, 1.0) - 0.5).abs() <= 1e-6);
    assert_eq!(grade_keep(0.0, c_near, c_far, 1.0), 1.0);
    assert_eq!(grade_keep(4.0, c_near, c_far, 1.0), 0.0);
}

/// At s = 0 the grading is the identity. Where the floor's march has a verdict both stages pivot
/// on the occluded floor and keep `K` of the reflection's distance from it (an escaped ray's floor
/// is the open sky); where it has none the near stage has no pivot and only the far stage acts,
/// `K / k_n` about the sky; at reach 0 with a verdict only the near stage acts. At a partial
/// verdict the floor is `known·P + (1 − known)·sky`, and the shader's affine form is the two stages
/// with the far pivot that floor and the near pivot `mix(r, P, known)`.
#[test]
fn the_grading_keeps_k_about_what_the_reflection_is_of() {
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
        let about = |p: V3, kept: f32| {
            zip(
                add(p, scale(zip(r, p, |a, b| a - b), kept)),
                [0.0; 3],
                f32::max,
            )
        };
        let hit = grade_apply(grade_affine(sky, occ, 1.0, rho, s, c_near, c_far), r);
        assert!(
            max_diff(hit, about(occ, k)) <= 1e-5,
            "s {s}, rho {rho}: about the occluder"
        );
        let escaped = grade_apply(grade_affine(sky, sky, 1.0, rho, s, c_near, c_far), r);
        assert!(
            max_diff(escaped, about(sky, k)) <= 1e-5,
            "s {s}, rho {rho}: about the sky"
        );
        if k_n > 1e-4 {
            let unknown = grade_apply(grade_affine(sky, sky, 0.0, rho, s, c_near, c_far), r);
            assert!(
                max_diff(unknown, about(sky, k / k_n)) <= 1e-5,
                "s {s}, rho {rho}: the far stage alone"
            );
        }
        let near = grade_reflection(r, occ, occ, 0.0, s, c_near, c_far);
        assert!(
            max_diff(near, about(occ, k_n)) <= 1e-5,
            "s {s}: the near stage alone"
        );
        let known = rng.unit();
        let floor = add(scale(occ, known), scale(sky, 1.0 - known));
        let p_near = add(scale(r, 1.0 - known), scale(occ, known));
        let staged = grade_reflection(r, floor, p_near, rho, s, c_near, c_far);
        let affine = grade_apply(grade_affine(sky, floor, known, rho, s, c_near, c_far), r);
        assert!(
            max_diff(staged, affine) <= 1e-5,
            "s {s}, rho {rho}, known {known}"
        );
        let identity = grade_apply(grade_affine(sky, floor, known, rho, 0.0, c_near, c_far), r);
        assert!(max_diff(identity, r) <= 1e-6);
    }
}

/// A partly covered floor, as an unknown verdict the frame's end covers by `c` returns it
/// (`rgb = c·frame`, `v = 1 − c`, `known = c`, the shape the quarter-resolution plane's bilinear
/// read gives every silhouette too): the floor is `c·frame + (1 − c)·sky`, and with the near stage
/// off the grading keeps `K` of the reflection about exactly that. With both stages on it is the
/// staged law with the far pivot that floor and the near pivot `mix(r, frame, c)`. A far pivot
/// `mix(sky, floor, known)` would count the cover twice and let `c(1 − c)(1 − K)` of the sky's
/// lead over the occluder in: under a canopy at half cover that is a dark reflection lifted by
/// a tenth of the sky.
#[test]
fn the_grading_counts_partial_cover_once() {
    let (c_near, c_far) = (0.2, 0.3);
    let mut rng = Lcg(0x5EED_0011);
    for _ in 0..20_000 {
        let (r, sky, frame) = (rng.v3(0.0, 1.2), rng.v3(0.0, 1.2), rng.v3(0.0, 1.2));
        let (c, rho, s) = (rng.unit(), rng.unit(), rng.range(0.0, 2.4));
        let (rgb, v, known) = (scale(frame, c), 1.0 - c, c);
        let floor_l = add(rgb, scale(sky, v));
        let pivot = add(scale(frame, c), scale(sky, 1.0 - c));
        assert!(max_diff(floor_l, pivot) <= 1e-6);
        let k = grade_keep(s, 0.0, c_far, rho);
        let one_stage = grade_apply(grade_affine(sky, floor_l, known, rho, s, 0.0, c_far), r);
        let want = zip(
            add(pivot, scale(zip(r, pivot, |a, b| a - b), k)),
            [0.0; 3],
            f32::max,
        );
        assert!(
            max_diff(one_stage, want) <= 1e-5,
            "c {c}, s {s}: about c·frame + (1 − c)·sky"
        );
        let p_near = add(scale(r, 1.0 - c), scale(frame, c));
        let staged = grade_reflection(r, pivot, p_near, rho, s, c_near, c_far);
        let affine = grade_apply(grade_affine(sky, floor_l, known, rho, s, c_near, c_far), r);
        assert!(max_diff(staged, affine) <= 1e-5, "c {c}, s {s}, rho {rho}");
    }
    let (canopy, sky, c, rho, s) = ([0.05; 3], [0.8; 3], 0.5, 0.25, 1.65);
    let floor_l = add(scale(canopy, c), scale(sky, 1.0 - c));
    let k = grade_keep(s, c_near, c_far, rho);
    let graded = grade_apply(grade_affine(sky, floor_l, c, rho, s, c_near, c_far), canopy);
    let twice = grade_reflection(
        canopy,
        add(scale(sky, 1.0 - c), scale(floor_l, c)),
        add(scale(canopy, 1.0 - c), scale(floor_l, c)),
        rho,
        s,
        c_near,
        c_far,
    );
    let lift = dot(twice, LUMA_709) - dot(graded, LUMA_709);
    let want = c * (1.0 - c) * (1.0 - k) * (0.8 - 0.05);
    assert!((lift - want).abs() <= 1e-5, "lift {lift}, want {want}");
    assert!(lift > 0.06, "lift {lift}");
}

/// Under a deck, at the reach the Booty's under-deck water measured (rho 0.57) and a hit verdict:
/// the reflection is the dark underside and the floor the coarser read of it, and the grading
/// pulls the one toward the other, never toward the sky, at any strength up to 2.4: with the
/// floor a hair off the underside it reads no brighter than 1.05x itself, and with the floor
/// clearly brighter or darker it lies between the two, channel by channel. The mutation it
/// guards against, a far stage about the open sky, reads more than 4x.
#[test]
fn the_grading_keeps_under_deck_water_dark() {
    let (c_near, c_far, rho) = (0.2, 0.3, 0.57);
    let (under, sky) = ([0.03, 0.03, 0.025], [0.6, 0.7, 0.9]);
    let close = [0.031, 0.03, 0.026];
    for floor in [close, [0.08, 0.075, 0.07], [0.01, 0.012, 0.008]] {
        for i in 0..=24 {
            let s = i as f32 * 0.1;
            let graded = grade_apply(grade_affine(sky, floor, 1.0, rho, s, c_near, c_far), under);
            for ch in 0..3 {
                let (lo, hi) = (under[ch].min(floor[ch]), under[ch].max(floor[ch]));
                assert!(
                    graded[ch] >= lo - 1e-6 && graded[ch] <= hi + 1e-6,
                    "s {s}, floor {floor:?}: {graded:?}"
                );
            }
            if floor == close {
                assert!(
                    dot(graded, LUMA_709) <= 1.05 * dot(under, LUMA_709),
                    "s {s}: {graded:?}"
                );
            }
        }
    }
    let sky_pivot = grade_reflection(under, sky, close, rho, 1.0, c_near, c_far);
    assert!(dot(sky_pivot, LUMA_709) > 4.0 * dot(under, LUMA_709));
}

/// The shader's grading is this mirror's, line for line, and wired as the composite takes it: the
/// map built from the floor's verdict, applied to the tier chain only with the eye above and s
/// above 0, and the graded chain is what the surface layer reflects at the one Fresnel `f_surf`
/// (fogged there only from below the surface, where no camera fogged it).
#[test]
fn the_shaders_grading_is_the_mirrors() {
    let wgsl = include_str!("shaders/liquid.wgsl");
    for line in [
        "return 1.0 - (1.0 - rho_a) * (1.0 - rho_c);",
        "return clamp(1.0 - g.x * (g.z + g.y * rho), 0.0, 1.0);",
        "let k_n = clamp(1.0 - g.x * g.z, 0.0, 1.0);",
        "let k_f = select(k / max(k_n, 1e-4), 0.0, k_n <= 1e-4);",
        "let a = mix(k_f, k, known);",
        "let b = (1.0 - k) * floor - (k_f - k) * (1.0 - known) * sky;",
        "return vec4<f32>(b, a);",
        "let graded = eye_above && water_reflect.look.grade.x > 0.0;",
        "grade_ab = grade_affine(sky_seen, floor_l, occ.known, grade_rho(surf.rho_a, occ.rho_c));",
        "refl_g = max(grade_ab.w * refl_l + grade_ab.xyz, vec3<f32>(0.0));",
        "var refl_out = refl_g;",
        "refl_out = fog_diffuse(refl_g, fog.f, fog.g);",
        "var layer = layer_surface(\n        f_surf,\n        refl_out,\n",
    ] {
        assert!(wgsl.contains(line), "liquid.wgsl lost `{line}`");
    }
}
