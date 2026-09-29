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


def banded_column(push, edge, h=900):
    """Y along a column of the albedo view: a linear ramp through ART-2's three bands,
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
    return 40.0 + 180.0 * xb


def view_of(column, w=8):
    return np.repeat(np.repeat(column[:, None, None], w, axis=1), 3, axis=2).astype(np.float32)


def hsv_patch(h_deg, s, v):
    import colorsys

    return [c * 255 for c in colorsys.hsv_to_rgb(h_deg / 360.0, s, v)]


class BandStep(unittest.TestCase):
    def test_a_hard_band_scores_infinity(self):
        v = view_of(banded_column(0.5, 0.0))
        self.assertTrue(math.isinf(wm.band_step(v, [0, 0, 8, 900])["value"]))

    def test_a_soft_band_scores_its_analytic_slope(self):
        # Push 0.5, edge 0.22: max slope 0.5 + 0.5 · 1.5 / 0.44 = 2.2045 times the ramp's.
        v = view_of(banded_column(0.5, 0.22))
        r = wm.band_step(v, [0, 0, 8, 900])["value"]
        self.assertAlmostEqual(r, 2.2045, delta=0.02)
        self.assertTrue(wm.judge(r, wm.ROWS["band_step"].gate))

    def test_an_unbanded_ramp_scores_one(self):
        v = view_of(banded_column(0.0, 0.22))
        self.assertAlmostEqual(wm.band_step(v, [0, 0, 8, 900])["value"], 1.0, delta=0.01)


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

    def test_every_row_has_a_definition_and_a_direction_or_none(self):
        for name, row in wm.ROWS.items():
            self.assertTrue(row.fn.__doc__, name)
            self.assertIn(row.better, ("lower", "higher", None), name)


if __name__ == "__main__":
    unittest.main()
