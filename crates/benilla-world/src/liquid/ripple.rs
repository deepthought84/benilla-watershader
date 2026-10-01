//! The stylised water's ripple map — a generated, seamlessly tiling slope texture.
//!
//! **This is not the reference's water and does not pretend to be.** Everything else under
//! `liquid/` reproduces what the 1.12 client does, byte by byte where the bytes are known; this
//! file exists to feed the *alternative* look the `waterStyle` CVar selects (see
//! [`super::WaterStyle`]), and its numbers answer to the eye rather than to `WoW.exe`. The
//! reference's own animated sheet (`lake_a`/`ocean_h`) is untouched and still drives the faithful
//! path — the two share a material and a draw, and nothing here can reach the reference look.
//!
//! The map holds the *slope* of a smooth noise height field in R/G and the height itself in B:
//!
//! - **R/G — slope.** The shader rebuilds a world-space normal as `vec3(r, 1, g)` and needs no
//!   tangent frame to do it, because a liquid surface is a flat, axis-aligned, never-rotating
//!   plane. Signed values live around 0.5 in an unorm texture, which is also what makes the mip
//!   chain correct: the average of encoded slopes is the encoding of the average slope, i.e.
//!   distant water flattens, which is exactly what it should do.
//! - **B — the height.** A second, uncorrelated scalar field the shore foam masks itself with,
//!   free: it is the field the slopes were differentiated from.
//!
//! Generated rather than shipped for the same reason the rest of benilla generates nothing: there
//! are no bundled art assets in this project, and a stylised look that needed one would have to
//! ship one. Perlin over a *wrapped lattice* is what makes it tile — a ripple map with a seam
//! draws a grid across the whole ocean, once every twelve yards.
//!
//! The mip chain is not optional. Without one the GPU point-samples a 256² map through pixels
//! that, out toward the horizon, each cover many tiles of it, and the beat between the two grids
//! is a moiré that crawls over distant water. (The reference's own sheet ships nine authored mips
//! whose alpha deliberately dies with distance, for the same reason — see `liquid.wgsl`.)
//!
//! What a mip averages away is not lost to the look: [`RippleStats`] measures, from the uploaded
//! bytes, how much slope variance each level has lost against level 0, so the shader can turn the
//! slope it can no longer resolve into roughness.

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::math::Vec2;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

/// Edge length of the generated map. Small on purpose: the shader samples it at three different
/// world scales, so its own texel grid never becomes the visible detail.
const SIZE: usize = 256;

/// The chain's levels, [`SIZE`] down to 1: the look block's nine `ripple_stats` rows.
const LEVELS: usize = SIZE.trailing_zeros() as usize + 1;

/// The octave ladder: `(lattice period, amplitude)`. Every period divides [`SIZE`], so each layer
/// tiles on its own and the sum tiles with them.
const OCTAVES: &[(i32, f32)] = &[(4, 1.0), (8, 0.5), (16, 0.28), (32, 0.14)];

/// Perlin noise that tiles seamlessly with the given lattice period.
///
/// Ordinary fbm does not tile. Wrapping the *lattice coordinates* — rather than cross-fading the
/// edges, which softens the whole texture — makes the repeat exact: the gradient is looked up at
/// the wrapped lattice point while the distance vector uses the unwrapped one, and mixing those
/// two up is what tears the noise at the seam.
fn tileable_perlin(p: Vec2, period: i32, seed: u32) -> f32 {
    const R: f32 = std::f32::consts::FRAC_1_SQRT_2;
    const DIRS: [(f32, f32); 8] = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (R, R),
        (-R, R),
        (R, -R),
        (-R, -R),
    ];

    let xi = p.x.floor();
    let zi = p.y.floor();
    let xf = p.x - xi;
    let zf = p.y - zi;
    let wrap = |v: i32| v.rem_euclid(period);
    let (x0, z0) = (wrap(xi as i32), wrap(zi as i32));
    let (x1, z1) = (wrap(x0 + 1), wrap(z0 + 1));

    let g = |gx: i32, gz: i32, dx: f32, dz: f32| {
        let mut h = (gx as u32)
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add((gz as u32).wrapping_mul(0x85EB_CA6B))
            .wrapping_add(seed.wrapping_mul(0xC2B2_AE35));
        h ^= h >> 15;
        h = h.wrapping_mul(0x2545_F491);
        h ^= h >> 13;
        let (ux, uz) = DIRS[(h & 7) as usize];
        ux * dx + uz * dz
    };

    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (u, v) = (fade(xf), fade(zf));

    let n00 = g(x0, z0, xf, zf);
    let n10 = g(x1, z0, xf - 1.0, zf);
    let n01 = g(x0, z1, xf, zf - 1.0);
    let n11 = g(x1, z1, xf - 1.0, zf - 1.0);
    let a = n00 + u * (n10 - n00);
    let b = n01 + u * (n11 - n01);
    a + v * (b - a)
}

/// The summed octaves, normalised to ±0.5.
fn height_field(seed: u32) -> Vec<f32> {
    let mut height = vec![0.0f32; SIZE * SIZE];
    let mut peak = 0.0f32;
    for z in 0..SIZE {
        for x in 0..SIZE {
            let uv = Vec2::new(x as f32 / SIZE as f32, z as f32 / SIZE as f32);
            let mut h = 0.0;
            for &(period, amp) in OCTAVES {
                h += tileable_perlin(uv * period as f32, period, seed + period as u32) * amp;
            }
            height[z * SIZE + x] = h;
            peak = peak.max(h.abs());
        }
    }
    let scale = if peak > 0.0 { 0.5 / peak } else { 1.0 };
    for h in &mut height {
        *h *= scale;
    }
    height
}

/// Box-filter one level down to half its edge. A 2×2 box never crosses the edge, so the wrap the
/// generator went to such lengths for survives the chain untouched.
fn halve(level: &[u8], n: usize) -> Vec<u8> {
    let half = n / 2;
    let mut next = vec![0u8; half * half * 4];
    for z in 0..half {
        for x in 0..half {
            for c in 0..4 {
                let at =
                    |dx: usize, dz: usize| level[((z * 2 + dz) * n + x * 2 + dx) * 4 + c] as u32;
                next[(z * half + x) * 4 + c] =
                    ((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1) + 2) / 4) as u8;
            }
        }
    }
    next
}

/// What each level of the ripple chain has lost against level 0, in the shader's decoded slope
/// (`tex·2 − 1`): the table the stylised lane turns into roughness. A box mip is a block mean (to
/// the nearest code), so by the law of total variance the variance a level-`L` sample has lost is
/// `cov₀ − cov_L`.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct RippleStats {
    /// Per level 0..=8: `[lost_xx, lost_zz, lost_xz, resolved]`, the covariance the level has lost
    /// and the per-axis variance `(cov_xx + cov_zz) / 2` it keeps.
    pub levels: [[f32; 4]; LEVELS],
    /// `[mean_x, mean_z, var0, wavelength]`: level 0's mean slope (the `as u8` truncation's
    /// bias), its per-axis variance, and its features' wavelength in texels.
    pub mean: [f32; 4],
}

/// The slope statistics of `chain`, the bytes [`ripple_map`] uploads: per level, the covariance
/// of the decoded R/G slope about the level's own mean, and what the level has lost against level
/// 0; then level 0's mean, variance and Rice wavelength ([`rice_wavelength`]), averaged over the
/// two slope axes.
fn ripple_stats(chain: &[u8]) -> RippleStats {
    let decoded: [f64; 256] = std::array::from_fn(|b| b as f64 / 255.0 * 2.0 - 1.0);
    let decode = |b: u8| decoded[usize::from(b)];
    let mut cov = [[0.0f64; 3]; LEVELS];
    let mut mean = [0.0f64; 2];
    let mut wavelength = 0.0;
    let mut at = 0;
    for (level, c) in cov.iter_mut().enumerate() {
        let n = SIZE >> level;
        let texels = chain[at..at + n * n * 4].as_chunks::<4>().0;
        at += n * n * 4;
        let count = (n * n) as f64;
        let (mut sx, mut sz) = (0.0, 0.0);
        for t in texels {
            sx += decode(t[0]);
            sz += decode(t[1]);
        }
        let (mx, mz) = (sx / count, sz / count);
        let (mut cxx, mut czz, mut cxz) = (0.0, 0.0, 0.0);
        for t in texels {
            let (x, z) = (decode(t[0]) - mx, decode(t[1]) - mz);
            cxx += x * x;
            czz += z * z;
            cxz += x * z;
        }
        *c = [cxx / count, czz / count, cxz / count];
        if level == 0 {
            mean = [mx, mz];
            wavelength = (0..2)
                .map(|axis| {
                    let field: Vec<f64> = texels.iter().map(|t| decode(t[axis])).collect();
                    rice_wavelength(&field)
                })
                .sum::<f64>()
                / 2.0;
        }
    }
    let c0 = cov[0];
    RippleStats {
        levels: cov.map(|c| {
            [
                (c0[0] - c[0]).max(0.0) as f32,
                (c0[1] - c[1]).max(0.0) as f32,
                (c0[2] - c[2]) as f32,
                (0.5 * (c[0] + c[1])) as f32,
            ]
        }),
        mean: [
            mean[0] as f32,
            mean[1] as f32,
            (0.5 * (c0[0] + c0[1])) as f32,
            wavelength as f32,
        ],
    }
}

/// Rice's mean wavelength of a wrapped [`SIZE`]² field, `2π·sqrt(var f / (var ∂x f + var ∂z f))`,
/// each derivative the wrapped central difference per texel: the map's feature size, without an
/// FFT.
fn rice_wavelength(f: &[f64]) -> f64 {
    let at = |x: usize, z: usize| f[(z % SIZE) * SIZE + x % SIZE];
    // The sum and the sum of squares of the field, ∂x and ∂z.
    let mut sums = [[0.0f64; 2]; 3];
    for z in 0..SIZE {
        for x in 0..SIZE {
            let values = [
                at(x, z),
                (at(x + 1, z) - at(x + SIZE - 1, z)) / 2.0,
                (at(x, z + 1) - at(x, z + SIZE - 1)) / 2.0,
            ];
            for (sum, v) in sums.iter_mut().zip(values) {
                sum[0] += v;
                sum[1] += v * v;
            }
        }
    }
    let n = (SIZE * SIZE) as f64;
    let [field, dx, dz] = sums.map(|[s, s2]| s2 / n - (s / n) * (s / n));
    std::f64::consts::TAU * (field / (dx + dz)).sqrt()
}

/// Build the ripple map: R/G = wrapped central-difference slope, B = the height it came from,
/// with a complete box-filtered mip chain and a repeating trilinear sampler, and the chain's
/// [`RippleStats`].
///
/// `Rgba8Unorm`, never `Srgb`: this holds geometry, not colour, and gamma-decoding a slope bends
/// it.
pub(super) fn ripple_map(seed: u32) -> (Image, RippleStats) {
    let height = height_field(seed);
    let at = |x: usize, z: usize| height[(z % SIZE) * SIZE + (x % SIZE)];

    let mut mip0 = vec![0u8; SIZE * SIZE * 4];
    for z in 0..SIZE {
        for x in 0..SIZE {
            // Central differences, wrapping — the slope has to tile as well as the height does.
            let dx = at(x + 1, z) - at((x + SIZE - 1) % SIZE, z);
            let dz = at(x, z + 1) - at(x, (z + SIZE - 1) % SIZE);
            let i = (z * SIZE + x) * 4;
            mip0[i] = ((dx * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            mip0[i + 1] = ((dz * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            mip0[i + 2] = ((at(x, z) + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            mip0[i + 3] = 255;
        }
    }

    let mut chain = mip0.clone();
    let mut level = mip0;
    let mut n = SIZE;
    let mut levels = 1;
    while n > 1 {
        level = halve(&level, n);
        n /= 2;
        chain.extend_from_slice(&level);
        levels += 1;
    }

    let started = std::time::Instant::now();
    let stats = ripple_stats(&chain);
    debug!(
        "liquid: the ripple chain's slope statistics in {:.3} ms",
        started.elapsed().as_secs_f64() * 1e3
    );

    let mut image = Image::new_uninit(
        Extent3d {
            width: SIZE as u32,
            height: SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
        // Kept in the main world too (349 KB): a sampler change after upload re-extracts the
        // image from this copy, where a render-world-only image fails to re-extract and the GPU
        // keeps the old sampler.
        RenderAssetUsages::default(),
    );
    image.data = Some(chain);
    image.texture_descriptor.mip_level_count = levels;
    image.sampler = ImageSampler::Descriptor(ripple_sampler(1));
    (image, stats)
}

/// The ripple map's sampler: repeating, trilinear, and `anisotropy` probes at most along a
/// footprint's major axis (1 for none; the quality tier's `RippleAniso`, wgpu takes 1-16 and
/// anything above 1 only with every filter linear). The reference's own filter policy is a process
/// global this map, benilla's own, does not read.
pub(super) fn ripple_sampler(anisotropy: u16) -> ImageSamplerDescriptor {
    ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: anisotropy.clamp(1, 16),
        ..default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The clamp reaches the descriptor within wgpu's 1-16, and every filter stays linear, which
    /// wgpu requires of a clamp above 1.
    #[test]
    fn the_sampler_takes_the_anisotropy_clamp() {
        for (asked, got) in [(0, 1), (1, 1), (4, 4), (8, 8), (40, 16)] {
            let d = ripple_sampler(asked);
            assert_eq!(d.anisotropy_clamp, got, "{asked}");
            assert!([d.mag_filter, d.min_filter, d.mipmap_filter]
                .iter()
                .all(|f| *f == ImageFilterMode::Linear),);
            assert_eq!(d.address_mode_u, ImageAddressMode::Repeat);
            assert_eq!(d.address_mode_v, ImageAddressMode::Repeat);
        }
    }

    /// The wrapped lattice is the whole tiling mechanism: one period out has to land on the same
    /// value, or the repeat draws a seam across the water every twelve yards.
    #[test]
    fn perlin_tiles() {
        for i in 0..16 {
            let p = Vec2::new(i as f32 * 0.37, i as f32 * 0.11);
            let here = tileable_perlin(p, 8, 1234);
            for step in [Vec2::new(8.0, 0.0), Vec2::new(0.0, 8.0), Vec2::splat(8.0)] {
                let there = tileable_perlin(p + step, 8, 1234);
                assert!(
                    (here - there).abs() < 1e-5,
                    "seam at {p:?} + {step:?}: {here} vs {there}"
                );
            }
        }
    }

    /// A complete chain down to 1x1, and exactly the bytes the descriptor promises — a short
    /// chain is a validation error at bind time, and a long one is silently sampled garbage.
    #[test]
    fn mip_chain_is_complete() {
        let (image, _) = ripple_map(7);
        let levels = image.texture_descriptor.mip_level_count;
        assert_eq!(levels, 9, "256 down to 1 is nine levels");
        assert_eq!(levels as usize, LEVELS);
        let expected: usize = (0..levels)
            .map(|l| {
                let n = SIZE >> l;
                n * n * 4
            })
            .sum();
        assert_eq!(image.data.as_ref().map(Vec::len), Some(expected));
    }

    /// The statistics of the map the client draws.
    fn shipped() -> RippleStats {
        ripple_map(super::super::surface::RIPPLE_SEED).1
    }

    /// Level 0 loses nothing, each coarser level loses at least what the finer one did, and the
    /// one-texel level has lost the whole of level 0's variance. The middle one holds for block
    /// means; the chain's rounding to a code could break it where a level keeps almost nothing,
    /// and on the shipped map does not.
    #[test]
    fn stats_are_the_law_of_total_variance() {
        let stats = shipped();
        assert_eq!(stats.levels[0][..3], [0.0; 3]);
        for pair in stats.levels.windows(2) {
            for axis in 0..2 {
                assert!(pair[1][axis] >= pair[0][axis], "{pair:?}");
            }
        }
        let last = stats.levels[LEVELS - 1];
        assert!(
            (0.5 * (last[0] + last[1]) - stats.mean[2]).abs() < 1e-6,
            "{last:?}"
        );
        assert!(last[3].abs() < 1e-9, "a one-texel level resolves nothing");
    }

    /// The `as u8` truncation leaves every decoded slope half a code low: a mean of −0.5/255·2.
    #[test]
    fn stats_measure_the_truncation_bias() {
        let [mx, mz, ..] = shipped().mean;
        for m in [mx, mz] {
            assert!((m - (-0.5 / 255.0 * 2.0)).abs() < 2e-4, "{m}");
        }
    }

    /// The map is isotropic: the cross term is under a tenth of what each level loses, and the
    /// two axes' variances agree within a tenth.
    #[test]
    fn stats_are_isotropic() {
        let stats = shipped();
        for (level, [xx, _, xz, _]) in stats.levels.iter().enumerate().skip(1) {
            assert!(
                xz.abs() <= 0.1 * xx,
                "level {level}: xz {xz} against xx {xx}"
            );
        }
        // The one-texel level has lost level 0's whole covariance, per axis.
        let [xx, zz, ..] = stats.levels[LEVELS - 1];
        assert!((xx - zz).abs() <= 0.1 * xx, "level 0: xx {xx}, zz {zz}");
    }

    /// The numbers the shader's roughness rests on, for the seed the client ships: a change of
    /// seed, octaves or quantisation fails here first.
    #[test]
    fn the_map_the_shader_sees_is_pinned() {
        let stats = shipped();
        let [.., var0, wavelength] = stats.mean;
        assert!((var0 / 6.536e-4 - 1.0).abs() < 0.01, "var0 {var0}");
        assert!((wavelength - 17.9).abs() < 0.5, "wavelength {wavelength}");
        let [xx, zz, ..] = stats.levels[3];
        let lost = (xx + zz) / (2.0 * var0);
        assert!((lost - 0.367).abs() < 0.01, "level 3 lost {lost}");
    }
}
