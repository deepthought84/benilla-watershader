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
            self.assertTrue(wm.judge(r, wm.ROWS["band_step"].gate))

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


class DeepWater(unittest.TestCase):
    box = [40, 40, 280, 200]

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

    def test_a_seam_step_stays_inside_the_image(self):
        img = flat([100, 100, 100])
        img[120:] = 110
        r = wm.tier_seam(img, [0, 0, W, H], 120)
        self.assertAlmostEqual(r["value"], 10.0, places=4)
        for at in (1, 238):  # 3 lines either side would wrap or run off the frame
            with self.assertRaises(ValueError):
                wm.tier_seam(img, [0, 0, W, H], at)
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
        self.assertEqual(wm.view_std(frames)["value"], 0.0)
        self.assertEqual(wm.view_std(frames, channel=1)["value"], 0.0)
        moving = [noisy([70, 100, 120], seed=i) for i in range(8)]
        self.assertGreater(wm.view_std(moving)["value"], 1.0)

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
metric = "view_std"
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
            self.assertEqual(res["view_std"]["value"], 0.0)
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
[[row]]
metric = "band_step"
label = "unmasked"
box = [0, 0, 40, 900]
still = false
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

    def test_every_row_has_a_definition_and_a_direction_or_none(self):
        for name, row in wm.ROWS.items():
            self.assertTrue(row.fn.__doc__, name)
            self.assertIn(row.better, ("lower", "higher", None), name)


if __name__ == "__main__":
    unittest.main()
