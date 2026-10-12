"""Fixtures for scripts/water_metrics.py: synthetic frames on which each row's answer is known.

    python3 -m unittest discover -s scripts/tests
"""

import contextlib
import io
import json
import math
import pathlib
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import water_metrics as wm  # noqa: E402

W, H = 320, 240


def flat(rgb, w=W, h=H):
    return np.broadcast_to(np.asarray(rgb, dtype=np.float32), (h, w, 3)).copy()


def noisy(rgb, sd=4.0, seed=1, w=W, h=H):
    rng = np.random.default_rng(seed)
    return np.clip(flat(rgb, w, h) + rng.normal(0, sd, (h, w, 1)), 0, 255).astype(np.float32)


def banded_column(push, edge, h=900, span=180.0):
    """Y along a column of the albedo view: a linear ramp through the body's three bands,
    xb = mix(x, band_soft(x), push), with
    band_soft = (floor(3x) + smoothstep(.5 − e, .5 + e, fract(3x))) / 3."""
    x = np.linspace(0.0, 1.0, h, endpoint=False)
    y = 3.0 * x
    f = y - np.floor(y)
    if edge > 0:
        t = np.clip((f - (0.5 - edge)) / (2 * edge), 0, 1)
        s = t * t * (3 - 2 * t)
    else:
        s = (f >= 0.5).astype(np.float64)
    xb = (1 - push) * x + push * (np.floor(y) + s) / 3.0
    return 40.0 + span * xb


def view_of(column, w=8):
    return np.repeat(np.repeat(column[:, None, None], w, axis=1), 3, axis=2).astype(np.float32)


def png(a):
    """An image through an 8-bit PNG and back, as a capture reaches the metrics."""
    buf = io.BytesIO()
    Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8)).save(buf, "PNG")
    buf.seek(0)
    return wm.load_rgb(buf)


def save(a, path):
    Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8)).save(path)


def hsv_patch(h_deg, s, v):
    import colorsys

    return [c * 255 for c in colorsys.hsv_to_rgb(h_deg / 360.0, s, v)]


class BandStep(unittest.TestCase):
    """Each case as float codes and through an 8-bit PNG, the way a capture arrives."""

    box = [0, 0, 8, 900]

    def test_a_hard_band_scores_infinity(self):
        v = view_of(banded_column(0.5, 0.0))
        for img in (v, png(v)):
            r = wm.band_step(img, self.box)
            self.assertTrue(math.isinf(r["value"]))
            self.assertEqual(r["row"], 150)  # fract(3x) crosses 0.5 at x = 1/6

    def test_a_soft_band_scores_its_analytic_slope(self):
        # Push 0.5, edge 0.22: max slope 0.5 + 0.5 · 1.5 / 0.44 = 2.2045 times the ramp's; the
        # 40-row window rounds the smoothstep's peak off by about 1.5%.
        v = view_of(banded_column(0.5, 0.22))
        for img in (v, png(v)):
            r = wm.band_step(img, self.box)["value"]
            self.assertAlmostEqual(r, 2.2045, delta=0.05)
            self.assertTrue(wm.judge(r, {"max": 3.0}))

    def test_an_unbanded_ramp_scores_one(self):
        for col in (banded_column(0.0, 0.22), banded_column(0.0, 0.22, h=400, span=60.0)):
            v = view_of(col)
            for img in (v, png(v)):
                r = wm.band_step(img, [0, 0, 8, len(col)])["value"]
                self.assertAlmostEqual(r, 1.0, delta=0.02)

    def test_a_silhouette_in_the_box_is_not_water(self):
        # A rock over one band's span of water, in the view and the still alike: only the still's
        # mask keeps its edge from reading as a hard band.
        view = view_of(banded_column(0.5, 0.22), w=40)
        still = view * 0.6  # the lit water: ≥ 16 codes off the view everywhere
        view[300:600, 10:30] = still[300:600, 10:30] = [30, 40, 20]
        view, still = png(view), png(still)
        box = [0, 0, 40, 900]
        self.assertTrue(math.isinf(wm.band_step(view, box)["value"]))
        r = wm.band_step(view, box, still=still)["value"]
        self.assertAlmostEqual(r, 2.2045, delta=0.05)

    def test_a_flat_view_has_no_ramp_to_band(self):
        self.assertIsNone(wm.band_step(flat([90, 120, 130]), [0, 0, W, H])["value"])

    def test_against_the_unbanded_view_soft_bands_on_a_bare_ramp_score_their_slope(self):
        banded, plain = view_of(banded_column(0.5, 0.22)), view_of(banded_column(0.0, 0.22))
        r = wm.band_step(png(banded), self.box, unbanded=png(plain))
        self.assertAlmostEqual(r["value"], 2.2045, delta=0.06)
        self.assertEqual(r["basis"], "banded/unbanded")
        self.assertFalse(wm.judge(r["value"], wm.ROWS["band_step"].gate))
        same = wm.band_step(png(plain), self.box, unbanded=png(plain))["value"]
        self.assertAlmostEqual(same, 1.0, places=6)
        self.assertTrue(wm.judge(same, wm.ROWS["band_step"].gate))

    def test_the_scenes_own_edges_are_not_bands(self):
        # A seabed silhouette (a hard 30-code rise) and a steep shore ramp in both captures: alone
        # each view scores ∞; the ratio is 1, and a hard band added on top still reads ∞.
        def scene(push, hard_edge):
            v = view_of(banded_column(push, 0.0 if hard_edge else 0.22), w=40) * 0.3
            v[450:, :] += 30.0
            v[600:700, 20:] += np.linspace(0, 60, 100)[:, None, None]
            return png(v)

        plain, soft, hard = scene(0.0, False), scene(0.5, False), scene(0.5, True)
        box = [0, 0, 40, 900]
        self.assertTrue(math.isinf(wm.band_step(plain, box)["value"]))
        self.assertAlmostEqual(wm.band_step(plain, box, unbanded=plain)["value"], 1.0, places=6)
        r = wm.band_step(soft, box, unbanded=plain)
        self.assertGreater(r["scene_steps"], 0)
        self.assertLess(r["value"], 1.1)
        r = wm.band_step(hard, box, unbanded=plain)
        self.assertTrue(math.isinf(r["value"]))
        self.assertEqual(r["row"], 150)


class DeepWater(unittest.TestCase):
    box = [40, 40, 280, 200]

    def test_a_desaturated_in_scatter_fails_the_no_neon_chroma_ratio(self):
        # Water where the two views differ; the bank (the frame's border) the same in both.
        albedo = flat([60, 60, 60])
        albedo[20:220, 20:300] = hsv_patch(160, 0.6, 0.4)
        same = albedo.copy()
        same[20:220, 20:300] = hsv_patch(160, 0.6, 0.7)  # brighter, the albedo's chroma
        r = wm.chroma_ratio(png(same), png(albedo), [0, 0, W, H])
        self.assertAlmostEqual(r["value"], 1.0, delta=0.02)
        self.assertEqual(r["pixels"], 196 * 276)  # the water eroded by 2 px
        self.assertTrue(wm.judge(r["value"], wm.ROWS["chroma_ratio"].gate))
        warm = albedo.copy()
        warm[20:220, 20:300] = hsv_patch(160, 0.45, 0.7)
        r = wm.chroma_ratio(png(warm), png(albedo), [0, 0, W, H])
        self.assertAlmostEqual(r["value"], 0.75, delta=0.02)
        self.assertEqual(r["inside"], 0.0)
        self.assertFalse(wm.judge(r["value"], wm.ROWS["chroma_ratio"].gate))

    def test_a_hue_shifted_patch_fails_deep_hue(self):
        albedo = flat(hsv_patch(200, 0.6, 0.5))
        same = wm.deep_hue(albedo, self.box, albedo=albedo)
        self.assertEqual(same["value"], 0.0)
        self.assertTrue(wm.judge(same["value"], wm.ROWS["deep_hue"].gate))
        shifted = flat(hsv_patch(230, 0.6, 0.5))
        r = wm.deep_hue(shifted, self.box, albedo=albedo)
        self.assertAlmostEqual(r["value"], 30.0, delta=0.5)
        self.assertFalse(wm.judge(r["value"], wm.ROWS["deep_hue"].gate))

    def test_the_transmittance_view_masks_the_shallows(self):
        albedo = flat(hsv_patch(200, 0.6, 0.5))
        final = albedo.copy()
        final[:, :160] = hsv_patch(120, 0.6, 0.5)  # a hue-shifted shallow half …
        t = flat([200, 200, 200])
        t[:, 160:] = 10  # … that the transmittance view says is not deep
        self.assertEqual(wm.deep_hue(final, self.box, albedo=albedo, transmittance=t)["value"], 0.0)

    def test_the_box_hue_without_views_is_informational(self):
        r = wm.deep_hue(flat(hsv_patch(108, 0.4, 0.5)), self.box)
        self.assertAlmostEqual(r["value"], 108.0, delta=0.5)
        self.assertFalse(r["gated"])
        r = wm.deep_hue(flat(hsv_patch(108, 0.4, 0.5)), self.box, ref_hue=191.0)
        self.assertAlmostEqual(r["value"], 83.0, delta=0.5)

    def test_a_darkened_deep_patch_fails_deep_luma(self):
        base = wm.deep_luma(flat([30, 90, 110]), self.box)["value"]
        row = wm.ROWS["deep_luma"]
        self.assertTrue(wm.against(wm.deep_luma(flat([29, 88, 108]), self.box)["value"], base, row))
        darker = wm.deep_luma(flat([20, 70, 90]), self.box)["value"]
        self.assertFalse(wm.against(darker, base, row))

    def test_a_desaturated_deep_patch_fails_deep_chroma(self):
        base = wm.deep_chroma(flat([30, 90, 110]), self.box)["value"]
        row = wm.ROWS["deep_chroma"]
        close = wm.deep_chroma(flat([32, 90, 108]), self.box)["value"]
        self.assertTrue(wm.against(close, base, row))
        grey = wm.deep_chroma(flat([60, 90, 100]), self.box)["value"]
        self.assertFalse(wm.against(grey, base, row))


def enc(lin):
    """Linear 0-1 to gamma codes, the water's enc()."""
    return 255.0 * np.asarray(lin, dtype=np.float64) ** (1 / 2.2)


class LinearLight(unittest.TestCase):
    """Laws and views in linear light: codes decode through the water's 2.2 gamma."""

    shade, lit = [100, 0, 200, 50], [0, 0, 100, 50]
    bank_shade, bank_lit = [100, 100, 200, 150], [0, 100, 100, 150]

    def frame(self, water_lit, water_shade):
        img = flat([90, 90, 90])
        bank = np.array([150.0, 120.0, 80.0])
        img[0:50, 0:100] = water_lit
        img[0:50, 100:200] = water_shade
        img[100:150, 0:100] = bank
        img[100:150, 100:200] = np.rint(bank * 0.7)  # the terrain's shadow: 0.7 in gamma
        return png(img)

    def ratio(self, img, in_scatter=None):
        return wm.shade_ratio(img, self.shade, self.lit, self.bank_shade, self.bank_lit, in_scatter)

    def test_shade_under_the_terrain_law_scores_0_456(self):
        water = np.array([60.0, 110.0, 130.0])
        r = self.ratio(self.frame(water, np.rint(water * 0.7)))
        self.assertAlmostEqual(r["value"], 0.456, delta=0.01)
        self.assertAlmostEqual(r["bank"], 0.456, delta=0.01)
        self.assertTrue(wm.judge(r["value"], wm.ROWS["shade_ratio"].gate))

    def test_lit_water_beside_a_shadowed_bank_fails_and_so_does_a_gamma_0_456(self):
        water = np.array([60.0, 110.0, 130.0])
        lit = self.ratio(self.frame(water, water))["value"]
        self.assertAlmostEqual(lit, 1.0, places=5)
        self.assertFalse(wm.judge(lit, wm.ROWS["shade_ratio"].gate))
        dark = self.ratio(self.frame(water, np.rint(water * 0.456)))["value"]
        self.assertAlmostEqual(dark, 0.456**2.2, delta=0.01)
        self.assertFalse(wm.judge(dark, wm.ROWS["shade_ratio"].gate))

    def test_the_in_scatter_view_decodes(self):
        view = flat([0, 0, 0])
        view[0:50, 0:100] = enc(0.30)
        view[0:50, 100:200] = enc(0.30 * 0.456)
        water = np.array([60.0, 110.0, 130.0])
        r = self.ratio(self.frame(water, water), png(view))
        self.assertAlmostEqual(r["value"], 0.456, delta=0.01)

    def test_the_deep_mask_decodes_the_transmittance_view(self):
        # t_bed 0.05 is code 65 (deep, Y < 0.1); 0.15 is code 108 (not deep).
        box = [0, 0, W, H]
        final = flat([20, 60, 80])
        final[:, 160:] = [60, 120, 140]
        t = flat(enc([0.05, 0.05, 0.05]))
        t[:, 160:] = enc([0.15, 0.15, 0.15])
        r = wm.deep_luma(final, box, transmittance=png(t))
        self.assertAlmostEqual(r["value"], float(wm.luma(np.array([20.0, 60.0, 80.0]))), places=3)

    def test_the_deep_mask_reads_fog_f_from_red(self):
        # fog_f 0.9 under a dark fog colour is deep; fog_f 0.59 under a bright one is not.
        box = [0, 0, W, H]
        final = flat([20, 60, 80])
        final[:, 160:] = [60, 120, 140]
        fog = flat([230, 60, 90])
        fog[:, 160:] = [150, 250, 250]
        r = wm.deep_luma(final, box, fog=png(fog))
        self.assertAlmostEqual(r["value"], float(wm.luma(np.array([20.0, 60.0, 80.0]))), places=3)


class Structure(unittest.TestCase):
    def test_a_straight_bright_line_trips_straight_line(self):
        img = noisy([60, 90, 100], sd=4.0, w=640, h=400)
        box = [0, 0, 640, 400]
        quiet = wm.straight_line(img, box)["value"]
        self.assertTrue(wm.judge(quiet, wm.ROWS["straight_line"].gate), quiet)
        for x in range(40, 600):  # a 560 px line, gently sloped
            y = 120 + (x - 40) // 7
            img[y, x] = [160, 170, 175]
        r = wm.straight_line(img, box)
        self.assertGreaterEqual(r["value"], 400)
        self.assertFalse(wm.judge(r["value"], wm.ROWS["straight_line"].gate))

    def test_zero_band_finds_the_flat_rows(self):
        img = noisy([90, 130, 150])
        img[100:140] = [111, 190, 190]
        r = wm.zero_band(img, [0, 60, W, 180])
        self.assertEqual(r["rows"], [100, 140])
        self.assertEqual(r["value"], 40)
        self.assertEqual(r["mad"], 0.0)
        self.assertEqual(wm.zero_band(noisy([90, 130, 150]), [0, 60, W, 180])["value"], 0)

    def test_refract_leak_counts_the_occluder_pulled_into_the_strip(self):
        frame = flat([200, 40, 40], w=100, h=100)  # a red occluder on the left …
        frame[:, 50:] = [40, 80, 160]  # … over blue water on the right
        mask = np.zeros((100, 100, 3), dtype=np.float32)
        mask[:, 50:] = 255
        bed = flat([40, 80, 160], w=100, h=100)
        refracted = bed.copy()
        refracted[:, 50:54] = [200, 40, 40]  # four columns of the strip read the occluder
        r = wm.refract_leak(refracted, bed, frame, mask, [0, 0, 100, 100])
        self.assertEqual(r["strip"], 800)
        self.assertAlmostEqual(r["value"], 0.5)
        self.assertEqual(wm.refract_leak(bed, bed, frame, mask, [0, 0, 100, 100])["value"], 0.0)

    def test_autocorr_peak_finds_a_period_along_the_bank(self):
        img = noisy([60, 90, 100], sd=1.0, w=400, h=40)
        img[20, ::25] = 200  # a bright dash every 25 px
        r = wm.autocorr_peak(img, [0, 10, 400, 30], lag_min=2, lag_max=100)
        self.assertEqual(r["lag"], 25)
        self.assertGreater(r["value"], 0.5)
        quiet = wm.autocorr_peak(noisy([60, 90, 100], sd=6.0, w=400, h=40), [0, 10, 400, 30])
        self.assertLess(quiet["value"], 0.3)

    def test_halo_count_counts_isolated_outliers(self):
        img = flat([80, 80, 80])
        img[50, 50] = [200, 200, 200]
        img[60, 90] = [0, 0, 0]
        self.assertEqual(wm.halo_count(img, [0, 0, W, H])["value"], 2)

    def test_stair_steps_count_on_a_staircase_bank_and_not_on_a_smooth_one(self):
        def bank(edge_of_x):
            img = flat([40, 60, 30])
            for x in range(W):
                img[edge_of_x(x) :, x] = [50, 110, 130]
            return img

        box = [0, 20, W, 220]
        self.assertEqual(wm.stair_step(bank(lambda x: 60 + (x // 20) * 8), box)["value"], 15)
        self.assertEqual(wm.stair_step(bank(lambda x: 60 + int(x * 0.4)), box)["value"], 0)


class Ratios(unittest.TestCase):
    def test_clip_fraction_counts_pixels_at_the_ceiling_and_black_ones(self):
        img = flat([100, 100, 100])
        img[0:10, 0:10] = [255, 120, 80]  # one channel clipped
        img[0:10, 10:20] = 255  # white
        img[0:10, 20:30] = 0  # black, a NaN's look
        r = wm.clip_fraction(img, [0, 0, 40, 20])
        self.assertAlmostEqual(r["value"], 200 / 800)
        self.assertAlmostEqual(r["white"], 100 / 800)
        self.assertAlmostEqual(r["black"], 100 / 800)
        frames = [flat([100, 100, 100]), img, flat([255, 255, 255])]
        b = wm.burst_clip_fraction(frames)
        self.assertEqual((b["value"], b["at"]), (1.0, 2))
        self.assertAlmostEqual(b["black"], 100 / (W * H))

    def test_refl_contrast_is_the_reflected_over_the_direct_michelson(self):
        img = flat([100, 100, 100])
        img[0:10, 0:10] = 63  # direct crown
        img[0:10, 20:30] = 200  # direct sky
        img[50:60, 0:10] = 110  # reflected crown
        img[50:60, 20:30] = 150  # reflected sky
        boxes = [0, 0, 10, 10], [20, 0, 30, 10], [0, 50, 10, 60], [20, 50, 30, 60]
        r = wm.refl_contrast(img, *boxes)
        md, mr = (200 - 63) / 263, 40 / 260
        self.assertAlmostEqual(r["direct"], md, places=5)
        self.assertAlmostEqual(r["value"], mr / md, places=5)
        # An inverted reflection: the reflected crown lighter than the reflected sky.
        img[50:60, 0:10], img[50:60, 20:30] = 150, 110
        r = wm.refl_contrast(img, *boxes)
        self.assertAlmostEqual(r["reflected"], -mr, places=5)
        self.assertAlmostEqual(r["value"], -mr / md, places=5)
        self.assertIs(wm.against(r["value"], mr / md, wm.ROWS["refl_contrast"], within=[0.6, 1.0]),
                      False)
        # The same with a light object: the sign follows the direct pair's, not the order of the
        # boxes.
        img[0:10, 0:10], img[0:10, 20:30] = 200, 63
        self.assertAlmostEqual(wm.refl_contrast(img, *boxes)["value"], mr / md, places=5)
        img[50:60, 0:10], img[50:60, 20:30] = 110, 150
        self.assertAlmostEqual(wm.refl_contrast(img, *boxes)["value"], -mr / md, places=5)

    def test_a_seam_step_stays_inside_the_image(self):
        img = flat([100, 100, 100])
        img[120:] = 110
        r = wm.seam_step(img, [0, 0, W, H], 120)
        self.assertAlmostEqual(r["value"], 10.0, places=4)
        for at in (1, 238):  # 3 lines either side would wrap or run off the frame
            with self.assertRaises(ValueError):
                wm.seam_step(img, [0, 0, W, H], at)
        with self.assertRaises(ValueError):
            wm.ring_seam(img, [0, 0, W, H], 1, axis="x")

    def test_horizon_ratio_and_shore_band(self):
        img = flat([100, 100, 100])
        img[:50] = 200
        r = wm.horizon_ratio(img, [0, 60, W, 80], [0, 20, W, 40])
        self.assertAlmostEqual(r["value"], 0.5, places=5)
        img[150:152, 10:20] = 150
        img[200:, :] = 80
        r = wm.shore_band_ratio(img, [0, 140, W, 160], [0, 210, W, 230])
        self.assertAlmostEqual(r["value"], 150 / 80, delta=0.3)

    def test_the_glint_peak_sits_on_the_bright_column(self):
        img = flat([60, 70, 80])
        xs = np.arange(W)
        prof = 100 * np.exp(-0.5 * ((xs - 200) / 20.0) ** 2)
        img[100:160] += prof[None, :, None].astype(np.float32)
        r = wm.glint_peak(img, [0, 80, W, 180], mirror=(200, 110))
        self.assertLessEqual(abs(r["x"] - 200), 1)
        w = wm.glint_width(img, [0, 80, W, 180])["value"]
        self.assertGreater(w, 12)
        self.assertLess(w, 22)


class Bursts(unittest.TestCase):
    def test_a_static_burst_scores_view_std_zero(self):
        f = noisy([70, 100, 120])
        frames = [f.copy() for _ in range(8)]
        self.assertEqual(wm.frame_std(frames)["value"], 0.0)
        self.assertEqual(wm.view_std(frames, "foam", channel=1)["value"], 0.0)
        moving = [noisy([70, 100, 120], seed=i) for i in range(8)]
        self.assertGreater(wm.frame_std(moving)["value"], 1.0)
        self.assertGreater(wm.view_std(moving, "foam", channel=1)["value"], 1.0)

    def test_view_std_names_its_view_and_frame_std_the_shipped_frame(self):
        frames = [noisy([70, 100, 120], seed=i) for i in range(4)]
        self.assertEqual(wm.frame_std(frames)["basis"], "frame luma")
        self.assertEqual(wm.view_std(frames, "floor_occ", channel=1)["basis"], "view floor_occ g")
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "wg-std.toml").write_text(
                '[[row]]\nmetric = "view_std"\nbox = [0, 0, 40, 40]\n'
                '[[row]]\nmetric = "view_std"\nlabel = "occ"\nbox = [0, 0, 40, 40]\n'
                'view = "floor_occ"\nchannel = 1\n'
            )
            for i, f in enumerate(frames):
                save(f, d / f"wg-std-{i:03d}.png")
                save(f, d / f"wg-std@floor_occ-{i:03d}.png")
            res = wm.measure(d, d, ["wg-std"])["wg-std"]
            self.assertIsNone(res["view_std"]["value"])  # no view named: not the shipped frame
            self.assertIn("names its `view`", res["view_std"]["error"])
            self.assertGreater(res["occ"]["value"], 1.0)
            self.assertEqual(res["occ"]["basis"], "view floor_occ g")

    def test_far_band_std_is_normalised_by_the_fog(self):
        L = [np.full((10, 10), v, dtype=np.float32) for v in (10.0, 12.0)]
        self.assertAlmostEqual(wm.far_band_std(L)["value"], 1.0)
        self.assertAlmostEqual(wm.far_band_std(L, fog=0.75)["value"], 4.0)

    def test_pulse_finds_the_flow_period(self):
        rng = np.random.default_rng(3)
        base = rng.normal(0, 10, (60, 80)).astype(np.float32)
        t = np.arange(120) / 30.0
        frames = [100 + base * (1.0 + 0.4 * math.cos(2 * math.pi * 1.25 * ti)) for ti in t]
        self.assertAlmostEqual(wm.pulse(frames, t)["value"], 1.25, delta=0.05)

    def test_pop_names_the_jump(self):
        L = [np.full((4, 4), 100.0 + 0.5 * i) for i in range(6)]
        L[4] = L[4] + 20
        r = wm.pop(L)
        self.assertEqual(r["at"], 3)
        self.assertGreater(r["ratio"], 10)

    def test_a_world_anchored_box_holds_still_through_a_pan(self):
        anchor = (100.0, 0.0, 0.0)
        poses, frames = [], []
        for i in range(6):
            eye = (0.0, 0.0, 2.0)
            look = (100.0, 10.0 - 4.0 * i, 0.0)
            poses.append((eye, look))
            img = flat([50, 80, 100], w=320, h=180)
            c = wm.project(anchor, eye, look, (320, 180))
            x, y = int(round(c[0])), int(round(c[1]))
            img[y - 12 : y + 13, x - 12 : x + 13] = 200
            frames.append(img)
        r = wm.pan_luma_std(frames, poses, anchor, half=(6, 6))
        self.assertEqual(r["shots"], 6)
        self.assertEqual(r["value"], 0.0)

    def test_the_camera_centre_projects_to_the_image_centre(self):
        x, y = wm.project((10.0, 0.0, 2.0), (0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (1600, 900))
        self.assertAlmostEqual(x, 800.0)
        self.assertAlmostEqual(y, 450.0)
        # West (+y) is on the left of a camera facing north (+x); up is up.
        x, y = wm.project((10.0, 1.0, 3.0), (0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (1600, 900))
        self.assertLess(x, 800.0)
        self.assertLess(y, 450.0)


class Baselines(unittest.TestCase):
    """A row against a baseline run: only numbers that measure the same thing are compared."""

    box = "box = [40, 40, 280, 200]\n"

    def metrics(self, d, baseline=False):
        """main() on a directory holding its own region files; the first run writes base.json."""
        argv = [str(d), "--regions", str(d), "--baseline" if baseline else "--json"]
        argv.append(str(d / "base.json"))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = wm.main(argv)
        return rc, out.getvalue()

    def test_a_bare_deep_hue_has_no_baseline_verdict(self):
        # A hue angle has no better direction: from 190°, 200° is not worse and 100° not better.
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "wg-deep.toml").write_text('[[row]]\nmetric = "deep_hue"\n' + self.box)
            save(flat(hsv_patch(190, 0.6, 0.5)), d / "wg-deep.png")
            self.assertEqual(self.metrics(d)[0], 0)
            for hue in (200, 100, 20):
                save(flat(hsv_patch(hue, 0.6, 0.5)), d / "wg-deep.png")
                rc, out = self.metrics(d, baseline=True)
                self.assertEqual(rc, 0, out)
                self.assertRegex(out, r"base 19\d\.?\d* -  basis hue; mask none")

    def test_a_deep_row_on_another_mask_is_not_comparable(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "wg-deep.toml").write_text('[[row]]\nmetric = "deep_luma"\n' + self.box)
            final = flat([20, 60, 80])
            final[:, 160:] = [60, 120, 140]
            save(final, d / "wg-deep.png")
            self.assertEqual(self.metrics(d)[0], 0)  # the baseline: no views, the whole box
            t = flat(enc([0.05, 0.05, 0.05]))
            t[:, 160:] = enc([0.15, 0.15, 0.15])
            save(t, d / "wg-deep@transmittance.png")
            rc, out = self.metrics(d, baseline=True)
            self.assertEqual(rc, 1, out)
            self.assertIn("n/c (basis mask none -> mask transmittance)", out)
            # The region row switches the view off: the same basis compares again.
            (d / "wg-deep.toml").write_text(
                '[[row]]\nmetric = "deep_luma"\n' + self.box + "transmittance = false\n"
            )
            rc, out = self.metrics(d, baseline=True)
            self.assertEqual(rc, 0, out)
            self.assertIn(" ok", out)


    def test_an_informational_row_has_no_gate_and_no_baseline_verdict(self):
        # A fixed box over a sweeping camera: the region row says gate = false.
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "wg-pan.toml").write_text(
                '[[row]]\nmetric = "shimmer"\n' + self.box + "gate = false\n"
                '[[row]]\nmetric = "pop"\n' + self.box
            )
            for i in range(4):
                save(noisy([90, 110, 120], seed=i), d / f"wg-pan-{i:03d}.png")
            self.assertEqual(self.metrics(d)[0], 0)
            base = json.loads((d / "base.json").read_text())["wg-pan"]
            self.assertIs(base["shimmer"]["gated"], False)
            self.assertNotIn("gated", base["pop"])
            for i in range(4):  # louder: worse on both rows
                save(noisy([90, 110, 120], sd=12.0, seed=i), d / f"wg-pan-{i:03d}.png")
            rc, out = self.metrics(d, baseline=True)
            self.assertEqual(rc, 1, out)
            self.assertRegex(out, r"shimmer +\S+ +info .* -  ")
            self.assertRegex(out, r"pop +\S+ +- .* WORSE")


class TierConsistency(unittest.TestCase):
    """The sky floor against the mirror where both see open sky (R-C3): the floor-served still
    against the shipped one, on mirror-served water whose floor verdict is open."""

    def scene(self):
        still = flat([90, 130, 170])
        tiers = flat([128, 128, 128])  # the probe or the floor: not a mirror's
        tiers[:, :200] = [0, 0, 255]  # mirror 1
        tiers[100:140, :200] = [0, 255, 255]  # mirror 2
        occ = flat([255, 0, 0])  # open sky, known
        occ[:, 100:] = [128, 0, 0]  # open sky, unknown
        occ[200:, :] = [0, 0, 0]  # occluded
        return still, tiers, occ

    def test_one_sky_scores_zero_and_a_brighter_floor_its_excess(self):
        still, tiers, occ = self.scene()
        box = [0, 0, W, H]
        r = wm.tier_consistency(still, still.copy(), tiers, occ, box)
        self.assertEqual((r["value"], r["worst_dhue"]), (0.0, 0.0))
        self.assertEqual(r["n"], 200 * 200)  # the mirrors' open water, known or not
        floor = still.copy()
        floor[:, :200] *= 1.2
        r = wm.tier_consistency(still, floor, tiers, occ, box)
        self.assertAlmostEqual(r["value"], 0.2, places=5)
        self.assertFalse(wm.judge(r["value"], wm.ROWS["tier_consistency"].gate, r))
        # Brighter only where the mirror does not serve, or where the floor is occluded: the same.
        floor = still.copy()
        floor[:, 200:] *= 1.5
        floor[200:, :] *= 1.5
        self.assertEqual(wm.tier_consistency(still, floor, tiers, occ, box)["value"], 0.0)

    def test_one_band_off_fails_where_the_whole_mask_passes(self):
        still, tiers, occ = self.scene()
        floor = still.copy()
        floor[32:48, :200] *= 1.3  # one 16-row band, a tenth of the mask
        r = wm.tier_consistency(still, floor, tiers, occ, [0, 0, W, H])
        self.assertLess(abs(r["ratio"] - 1.0), 0.1)
        self.assertAlmostEqual(r["value"], 0.3, places=4)
        self.assertEqual(r["bands"], 13)  # rows 0-199 in 16-row bands, 200 px wide

    def test_a_hue_shift_fails_on_worst_dhue(self):
        still, tiers, occ = self.scene()
        still[:] = hsv_patch(200, 0.5, 0.6)
        floor = flat(hsv_patch(208, 0.5, 0.6))
        r = wm.tier_consistency(still, floor, tiers, occ, [0, 0, W, H])
        self.assertLess(r["value"], 0.1)
        self.assertAlmostEqual(r["worst_dhue"], 8.0, delta=0.5)
        self.assertFalse(wm.judge(r["value"], wm.ROWS["tier_consistency"].gate, r))

    def test_the_row_reads_its_variants_and_named_boxes_from_files(self):
        still, tiers, occ = self.scene()
        floor = still.copy()
        floor[:, :50] *= 1.05
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "wg-t.toml").write_text(
                "[box]\na = [0, 0, 50, 100]\nb = [0, 100, 300, 240]\n"
                '[[row]]\nmetric = "tier_consistency"\nbox = ["a", "b"]\n'
            )
            save(still, d / "wg-t.png")
            res = wm.measure(d, d, ["wg-t"])["wg-t"]["tier_consistency"]
            self.assertIn("wg-t~floor.png", res["missing"])
            self.assertIsNone(res["pass"])
            save(floor, d / "wg-t~floor.png")
            save(tiers, d / "wg-t@tiers.png")
            save(occ, d / "wg-t~floor@floor_occ.png")
            res = wm.measure(d, d, ["wg-t"])["wg-t"]["tier_consistency"]
            self.assertEqual(res["n"], 50 * 100 + 200 * 100)
            self.assertTrue(res["pass"])
            self.assertAlmostEqual(res["value"], 0.05, delta=0.01)


class TierSeam(unittest.TestCase):
    """The automatic tier seam: the boundaries traced from the TIERS view, the step measured on the
    still, the world's own edges (a capture with no liquid) left out."""

    def scene(self):
        # Water below row 60; the floor (grey) above row 120, the probe (green) from it, the first
        # mirror right of column 260. The world with no liquid is a flat bed.
        world = flat([60, 70, 50])
        world[:60] = [150, 160, 170]
        still = noisy([100, 100, 100], sd=3.0)
        still[:60] = world[:60]
        tiers = flat([128, 128, 128])
        tiers[120:] = [0, 255, 0]
        tiers[60:, 260:] = [0, 0, 255]
        tiers[:60] = world[:60]  # the view repaints water only
        return still, tiers, world

    def test_the_classes_read_mixes_and_leave_the_dry_world_out(self):
        still, tiers, world = self.scene()
        tiers[200:, :20] = [38, 217, 38]  # 0.3 floor + 0.7 probe
        tiers[200:, 20:40] = [90, 128, 166]  # 0.7 floor + 0.3 mirror 1: still the floor
        cls = wm.tier_classes(tiers, still, world)
        self.assertTrue((cls[:60] == -1).all())  # dry: the still is the world there
        self.assertTrue((cls[60:120, :260] == 0).all())
        self.assertTrue((cls[120:200, :260] == 1).all())
        self.assertTrue((cls[200:, :20] == 1).all())
        self.assertTrue((cls[200:, 20:40] == 0).all())
        self.assertTrue((cls[60:, 260:] == 2).all())
        # A colour no mix of the tiers makes (a falls drawn over the water) is not read.
        tiers[150:160, 100:110] = [200, 170, 120]
        tiers[160:170, 100:110] = [131, 92, 32]  # a sandy bank: red, green and blue mixed
        tiers[170:180, 100:110] = [178, 0, 77]  # 0.7 march + 0.3 mirror 1
        cls = wm.tier_classes(tiers, still, world)
        self.assertTrue((cls[150:170, 100:110] == -1).all())
        self.assertTrue((cls[170:180, 100:110] == 4).all())

    def test_a_step_across_the_boundary_is_the_seam_and_a_ramp_is_none(self):
        still, tiers, world = self.scene()
        r = wm.tier_seam(still, tiers, world)
        self.assertLess(r["value"], 1.0)  # the noise alone, averaged along the boundary
        still[120:] += 10.0  # the probe's water 10 codes brighter than the floor's
        r = wm.tier_seam(still, tiers, world, detail=True)
        self.assertAlmostEqual(r["worst"], 10.0, delta=1.5)
        self.assertEqual(r["worst_pair"], "floor/probe")
        fp = [s for s in r["all"] if s["pair"] == "floor/probe"]
        self.assertEqual(sum(s["n"] for s in fp), 260)  # one element per column
        self.assertTrue(all(s["n"] <= 32 for s in fp))  # cut every 32 px
        # The value weighs every kept stretch by its width, the floor/mirror and probe/mirror
        # ones (no step) too.
        self.assertLess(r["value"], r["worst"])
        self.assertAlmostEqual(
            r["value"] * r["px"],
            sum(s["n"] * abs(s["step"]) for s in r["all"] if s["kept"]),
            places=3,
        )
        # The same 10 codes spread over 40 rows is no seam: 3 px either side see 0.75 of it.
        ramp = still.copy()
        ramp[120:] -= 10.0
        ramp[100:140] += (np.arange(40) / 40.0 * 10.0)[:, None, None]
        ramp[140:] += 10.0
        self.assertLess(wm.tier_seam(ramp, tiers, world)["value"], 1.5)

    def test_the_worlds_own_edge_is_left_out(self):
        still, tiers, world = self.scene()
        # The bed itself steps 20 codes at the tier boundary, and the still shows it, dimmed.
        world[120:, :260] += 20.0
        still[120:, :260] += 12.0
        r = wm.tier_seam(still, tiers, world, detail=True)
        fp = [s for s in r["all"] if s["pair"] == "floor/probe"]
        self.assertTrue(fp and not any(s["kept"] for s in fp))
        self.assertGreaterEqual(r["excluded"], 260)
        self.assertLess(r["value"], 1.0)
        # A step well beyond what the world's edge explains is still a seam.
        still[120:, :260] += 20.0
        r = wm.tier_seam(still, tiers, world)
        self.assertAlmostEqual(r["worst"], 32.0, delta=1.5)

    def test_a_frame_without_a_boundary_scores_zero_and_the_row_reads_its_files(self):
        still, tiers, world = self.scene()
        one = tiers.copy()
        one[60:] = [0, 0, 255]
        self.assertEqual(wm.tier_seam(still, one, world)["value"], 0.0)
        still[120:, :260] += 10.0
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "wg-s.toml").write_text('[[row]]\nmetric = "tier_seam"\n')
            save(still, d / "wg-s.png")
            save(tiers, d / "wg-s@tiers.png")
            res = wm.measure(d, d, ["wg-s"])["wg-s"]["tier_seam"]
            self.assertIn("wg-s~noliquid.png", res["missing"])
            save(world, d / "wg-s~noliquid.png")
            res = wm.measure(d, d, ["wg-s"])["wg-s"]["tier_seam"]
            self.assertAlmostEqual(res["worst"], 10.0, delta=1.5)


class Profiles(unittest.TestCase):
    def test_roughness_rising_toward_the_far_rows_is_monotone(self):
        view = np.zeros((100, 4, 3), dtype=np.float32)
        view[..., 0] = np.linspace(250, 10, 100)[:, None]  # far (top) rough, near (bottom) calm
        r = wm.roughness_profile(view, [0, 0, 4, 100], rising=True)
        self.assertEqual((r["monotone"], r["back"]), (1.0, 0.0))
        self.assertTrue(wm.judge(r["value"], {"max": 10.8, "monotone": {"min": 1.0}}, r))
        view[40:45, :, 0] -= 6  # a dip: a step back toward the camera
        r = wm.roughness_profile(view, [0, 0, 4, 100], rising=True)
        self.assertLess(r["monotone"], 1.0)
        self.assertAlmostEqual(r["back"], 6.0 - 240 / 99, places=4)
        self.assertTrue(wm.judge(r["value"], {"max": 10.8}, r))  # the step gate alone passes it
        self.assertFalse(wm.judge(r["value"], {"max": 10.8, "monotone": {"min": 1.0}}, r))
        # A profile falling toward the far rows is monotone its own way, not the required one.
        view[..., 0] = np.linspace(10, 250, 100)[:, None]
        self.assertEqual(wm.roughness_profile(view, [0, 0, 4, 100])["monotone"], 1.0)
        self.assertEqual(wm.roughness_profile(view, [0, 0, 4, 100], rising=True)["monotone"], 0.0)

    def test_a_required_number_gates_the_row_from_its_region_file(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "wg-r.toml").write_text(
                '[[row]]\nmetric = "roughness_profile"\nbox = [0, 0, 4, 100]\nmax = 10.8\n'
                "rising = true\nrequire = { monotone = { min = 1.0 } }\n"
                '[[row]]\nmetric = "roughness_profile"\nlabel = "free"\nbox = [0, 0, 4, 100]\n'
                "max = 10.8\n"
            )
            view = np.zeros((100, 4, 3), dtype=np.float32)
            view[..., 0] = np.linspace(250, 10, 100)[:, None]
            view[40:45, :, 0] -= 6
            save(view, d / "wg-r@roughness.png")
            save(view, d / "wg-r.png")
            res = wm.measure(d, d, ["wg-r"])["wg-r"]
            gate = res["roughness_profile"]["gate"]
            self.assertEqual(gate, {"max": 10.8, "monotone": {"min": 1.0}})
            self.assertFalse(res["roughness_profile"]["pass"])
            self.assertTrue(res["free"]["pass"])
            self.assertEqual(wm.gate_text(gate), "max 10.8 monotone min 1")


class Scenes(unittest.TestCase):
    """The file naming and the region file, end to end through main()."""

    def test_a_scene_directory_is_measured_from_its_region_file(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            regions = d / "regions"
            regions.mkdir()
            (regions / "wg-test.toml").write_text(
                """scene = "wg-test"
[box]
water = [0, 120, 320, 240]
sky = [0, 0, 320, 100]
[[row]]
metric = "horizon_ratio"
water = "water"
sky = "sky"
[[row]]
metric = "frame_std"
box = "water"
[[row]]
metric = "lum_vs_ref"
box = "water"
[[row]]
metric = "band_step"
box = "water"
[[row]]
metric = "luma_sd"
label = "sky_sd"
box = "sky"
max = 1.0
"""
            )
            img = flat([100, 100, 100])
            img[:110] = 200
            Image.fromarray(img.astype(np.uint8)).save(d / "wg-test.png")
            Image.fromarray((img * 0.5).astype(np.uint8)).save(d / "wg-test-ref.png")
            log = []
            for i in range(3):
                p = d / f"wg-test-{i:03d}.png"
                Image.fromarray(img.astype(np.uint8)).save(p)
                log.append(f"capture: burst shot {i} at {2.5 + i / 30:.6f} s, writing {p}")
            (d / "wg-test.log").write_text("\n".join(log))
            out = d / "out.json"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = wm.main([str(d), "--regions", str(regions), "--json", str(out)])
            self.assertEqual(rc, 0)
            res = json.loads(out.read_text())["wg-test"]
            self.assertAlmostEqual(res["horizon_ratio"]["value"], 0.5)
            self.assertEqual(res["frame_std"]["value"], 0.0)
            self.assertAlmostEqual(res["lum_vs_ref"]["value"], 2.0, delta=0.02)
            self.assertIsNone(res["band_step"]["value"])
            self.assertIn("wg-test@albedo.png", res["band_step"]["missing"])
            self.assertEqual(res["sky_sd"]["pass"], True)
            times = wm.Scene(d, "wg-test").burst_times()
            for got, want in zip(times, (2.5, 2.5 + 1 / 30, 2.5 + 2 / 30), strict=True):
                self.assertAlmostEqual(got, want, places=5)
            # Against itself as the baseline nothing is worse.
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                rc = wm.main([str(d), "--regions", str(regions), "--baseline", str(out)])
            self.assertEqual(rc, 0)
            self.assertNotIn("WORSE", printed.getvalue())

    def test_band_step_reads_the_albedo_capture_over_the_stills_water(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "wg-band.toml").write_text(
                """[[row]]
metric = "band_step"
box = [0, 0, 40, 900]
unbanded = false
max = 3
[[row]]
metric = "band_step"
label = "unmasked"
box = [0, 0, 40, 900]
still = false
unbanded = false
max = 3
[[row]]
metric = "band_step"
label = "ratio"
box = [0, 0, 40, 900]
"""
            )
            view = view_of(banded_column(0.5, 0.22), w=40)
            still = view * 0.6
            view[300:600, 10:30] = still[300:600, 10:30] = [30, 40, 20]
            Image.fromarray(np.rint(view).astype(np.uint8)).save(d / "wg-band@albedo.png")
            Image.fromarray(np.rint(still).astype(np.uint8)).save(d / "wg-band.png")
            res = wm.measure(d, d, ["wg-band"])["wg-band"]
            self.assertAlmostEqual(res["band_step"]["value"], 2.2045, delta=0.05)
            self.assertTrue(res["band_step"]["pass"])
            self.assertTrue(math.isinf(res["unmasked"]["value"]))
            self.assertFalse(res["unmasked"]["pass"])
            self.assertIn("wg-band~unbanded@albedo.png", res["ratio"]["missing"])
            plain = view_of(banded_column(0.0, 0.22), w=40)
            plain[300:600, 10:30] = [30, 40, 20]
            Image.fromarray(np.rint(plain).astype(np.uint8)).save(d / "wg-band~unbanded@albedo.png")
            res = wm.measure(d, d, ["wg-band"])["wg-band"]
            self.assertAlmostEqual(res["ratio"]["value"], 2.2045, delta=0.06)
            self.assertEqual(res["ratio"]["gate"], {"max": 1.1})
            self.assertFalse(res["ratio"]["pass"])

    def test_a_row_measured_on_a_view_and_a_bare_hue_judged_by_its_own_gate(self):
        # The still's water is warm (a reflection); the in-scatter view holds the body's hue.
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "wg-hue.toml").write_text(
                """[box]
water = [0, 0, 100, 100]
[[row]]
metric = "deep_hue"
label = "hue"
box = "water"
[[row]]
metric = "deep_hue"
label = "body_hue"
box = "water"
on = "in_scatter"
albedo = false
min = 120
"""
            )
            save(flat(hsv_patch(60, 0.5, 0.5), w=100, h=100), d / "wg-hue.png")
            res = wm.measure(d, d, ["wg-hue"])["wg-hue"]
            self.assertIn("wg-hue@in_scatter.png", res["body_hue"]["missing"])
            save(flat(hsv_patch(130, 0.5, 0.5), w=100, h=100), d / "wg-hue@in_scatter.png")
            out = d / "out.json"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                wm.main([str(d), "wg-hue", "--regions", str(d), "--json", str(out)])
            res = json.loads(out.read_text())["wg-hue"]
            self.assertAlmostEqual(res["hue"]["value"], 60.0, delta=1.0)
            self.assertEqual((res["hue"]["gated"], res["hue"]["pass"]), (False, None))
            self.assertAlmostEqual(res["body_hue"]["value"], 130.0, delta=1.0)
            self.assertEqual(res["body_hue"]["basis"], "on in_scatter; hue; mask none")
            self.assertEqual(res["body_hue"]["gate"], {"min": 120})
            self.assertTrue(res["body_hue"]["pass"])
            self.assertIn("PASS min 120", printed.getvalue())
            # A bare hue has no better direction: against any baseline it gets no verdict.
            base = json.loads(out.read_text())
            base["wg-hue"]["body_hue"]["value"] = 170.0
            self.assertEqual(wm.versus(res["body_hue"], base["wg-hue"]["body_hue"]), (None, "-"))
            save(flat(hsv_patch(100, 0.5, 0.5), w=100, h=100), d / "wg-hue@in_scatter.png")
            res = wm.measure(d, d, ["wg-hue"])["wg-hue"]
            self.assertFalse(res["body_hue"]["pass"])

    def test_within_holds_a_row_between_two_shares_of_its_baseline(self):
        row = wm.ROWS["refl_contrast"]
        self.assertTrue(wm.against(0.15, 0.2391, row))  # lower is better: any softening passes
        for value, inside in ((0.1434, False), (0.1435, True), (0.2391, True), (0.2392, False)):
            self.assertEqual(wm.against(value, 0.2391, row, within=[0.6, 1.0]), inside, value)
        self.assertTrue(wm.against(-1.5, -2.0, row, within=[0.6, 1.0]))
        self.assertTrue(wm.against(1.0, 1.0, wm.ROWS["glint_peak"], within=[0.9, 1.1]))
        # A row's numpy number gets a plain verdict, which the exit status counts.
        self.assertIs(wm.against(np.float32(0.5), 0.2391, row), False)
        self.assertIs(wm.against(np.float32(0.5), 0.2391, row, within=[0.6, 1.0]), False)
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "wg-within.toml").write_text(
                """[box]
object = [0, 0, 10, 10]
sky = [10, 0, 20, 10]
object_refl = [0, 20, 10, 30]
sky_refl = [10, 20, 20, 30]
[[row]]
metric = "refl_contrast"
object = "object"
sky = "sky"
object_refl = "object_refl"
sky_refl = "sky_refl"
within = [0.6, 1.0]
"""
            )
            img = flat([100, 100, 100], w=20, h=30)
            img[0:10, 0:10], img[0:10, 10:20] = 50, 150  # direct contrast 0.5
            img[20:30, 0:10], img[20:30, 10:20] = 80, 120  # reflected 0.2: ratio 0.4
            save(img, d / "wg-within.png")
            out = d / "out.json"
            with contextlib.redirect_stdout(io.StringIO()):
                wm.main([str(d), "--regions", str(d), "--json", str(out)])
            base = json.loads(out.read_text())
            self.assertEqual(base["wg-within"]["refl_contrast"]["within"], [0.6, 1.0])
            for refl, rc, tag in (((60, 140), 1, "0.6-1x OUT"), ((85, 115), 0, "0.6-1x ok"),
                                  ((90, 110), 1, "0.6-1x OUT"),
                                  ((120, 80), 1, "0.6-1x OUT")):  # 2, 0.75, 0.5, -1 x baseline
                img[20:30, 0:10], img[20:30, 10:20] = refl
                save(img, d / "wg-within.png")
                with contextlib.redirect_stdout(io.StringIO()) as printed:
                    got = wm.main([str(d), "--regions", str(d), "--baseline", str(out)])
                self.assertEqual(got, rc, refl)
                self.assertIn(tag, printed.getvalue())

    def test_a_deferred_row_reports_its_miss_without_failing(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "wg-defer.toml").write_text(
                """[box]
water = [0, 0, 100, 100]
[[row]]
metric = "luma_sd"
label = "kept"
box = "water"
min = 5
defer = "P1-35"
[[row]]
metric = "luma_sd"
box = "water"
"""
            )
            save(flat([90, 110, 120], w=100, h=100), d / "wg-defer.png")
            out = d / "out.json"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                rc = wm.main([str(d), "--regions", str(d), "--json", str(out)])
            self.assertEqual(rc, 0)
            self.assertIn("DEFER", printed.getvalue())
            res = json.loads(out.read_text())["wg-defer"]
            self.assertEqual((res["kept"]["pass"], res["kept"]["defer"]), (False, "P1-35"))
            # Worse than its baseline: deferred too; the undeferred row's miss counts.
            base = json.loads(out.read_text())
            base["wg-defer"]["kept"]["value"] = base["wg-defer"]["luma_sd"]["value"] = 3.0
            (d / "base.json").write_text(json.dumps(base))
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                rc = wm.main([str(d), "--regions", str(d), "--baseline", str(d / "base.json")])
            self.assertEqual(rc, 1)
            self.assertIn("DEFER WORSE", printed.getvalue())
            self.assertEqual(printed.getvalue().count("WORSE"), 2)

    def test_contact_rows_read_the_foam_views_contact_coverage(self):
        # The contact view's r is the distance to the contact (1 at it): full everywhere, while
        # the foam view's g, c_contact, stands in 30 of 100 columns.
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "wg-quay.toml").write_text(
                """[box]
quay = [0, 0, 100, 40]
open = [0, 60, 100, 100]
[[row]]
metric = "contact_cov"
box = "quay"
[[row]]
metric = "foam_fp"
box = "open"
"""
            )
            save(flat([90, 110, 120], w=100, h=100), d / "wg-quay.png")
            contact = flat([255, 0, 0], w=100, h=100)
            save(contact, d / "wg-quay@contact.png")
            foam = flat([0, 0, 0], w=100, h=100)
            foam[30:34, 0:30, 1] = 200  # a collar along 30 columns
            foam[80, 5, 1] = 255  # one stray pixel of contact foam on open water
            foam[60:100, :, 0] = 255  # line foam, another channel
            save(foam, d / "wg-quay@foam.png")
            res = wm.measure(d, d, ["wg-quay"])["wg-quay"]
            self.assertAlmostEqual(res["contact_cov"]["value"], 30.0)
            self.assertFalse(res["contact_cov"]["pass"])
            self.assertAlmostEqual(res["foam_fp"]["value"], 100 / 4000)
            self.assertTrue(res["foam_fp"]["pass"])

    def test_contact_rows_count_water_only_against_a_second_view(self):
        # Two views of one scene differ on water only; the bank's green above the waterline (row
        # 20) is scene colour, not contact foam, and only counts without the second view.
        view = flat([40, 180, 40], w=100, h=40)  # a green bank
        view[20:, :] = 0  # water: no foam
        view[20:22, 0:25] = [0, 200, 0]  # a collar along 25 columns
        fres = view.copy()
        fres[20:, :] = 90  # the Fresnel view's grey on the same water
        box = [0, 0, 100, 40]
        self.assertAlmostEqual(wm.contact_cov(view, box)["value"], 100.0)
        r = wm.contact_cov(view, box, mask=fres)
        self.assertAlmostEqual(r["value"], 25.0)
        self.assertEqual((r["columns"], r["basis"]), (100, "water"))
        f = wm.foam_fp(view, box, mask=fres)
        self.assertAlmostEqual(f["value"], 100 * 50 / 2000)
        self.assertAlmostEqual(wm.foam_fp(view, box)["value"], 100 * (2000 + 50) / 4000)

    def test_every_row_has_a_definition_and_a_direction_or_none(self):
        for name, row in wm.ROWS.items():
            self.assertTrue(row.fn.__doc__, name)
            self.assertIn(row.better, ("lower", "higher", None), name)

    def test_the_region_files_name_their_scene_and_boxes_inside_the_frame(self):
        files = sorted(wm.REGIONS.glob("*.toml"))
        self.assertTrue(files, "no region file")
        for path in files:
            reg = wm.load_regions(path)
            self.assertEqual(reg.get("scene"), path.stem, path.name)
            boxes = dict(reg["box"])
            for name, b in boxes.items():
                x0, y0, x1, y1 = b
                self.assertTrue(0 <= x0 < x1 <= 1600 and 0 <= y0 < y1 <= 900, f"{path.name} {name}")
            for spec in reg["row"]:
                for k in spec.keys() & wm.BOX_PARAMS:
                    v = spec[k]
                    self.assertTrue(isinstance(v, list) or v in boxes, f"{path.name} {k} = {v!r}")


if __name__ == "__main__":
    unittest.main()
