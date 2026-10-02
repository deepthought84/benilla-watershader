"""Fixtures for scripts/water_baseline.py: a synthetic capture run whose table is known.

    python3 -m unittest discover -s scripts/tests
"""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import water_baseline as wb  # noqa: E402

REGION = """scene = "wg-t"
[box]
water = [0, 20, 40, 40]
[[row]]
metric = "luma_sd"
box = "water"
[[row]]
metric = "lum_vs_ref"
box = "water"
[[row]]
metric = "shimmer"
box = "water"
[[row]]
metric = "pop"
box = "water"
gate = false
"""


def save(rgb, path):
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def frame(v, w=40, h=40):
    img = np.full((h, w, 3), v, dtype=np.uint8)
    img[30:, :20] = v // 2  # structure in the water box: luma_sd > 0
    return img


def run(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rc = fn(*args)
    return rc, out.getvalue()


class Plan(unittest.TestCase):
    def test_all_runs_the_stills_views_bursts_and_repeats_in_order(self):
        p = wb.plan("all")
        self.assertEqual([(s, label) for s, label, _, _ in p], [
            ("stills", "stills"), ("stills", "view-fresnel"), ("stills", "view-foam"),
            ("stills", "view-tiers"), ("bursts", "bursts"), ("repeat", "stills"),
            ("repeat-bursts", "bursts"),
        ])
        views = {e["WATER_VIEW"]: e["WATER_SCENES"].split() for _, _, e, _ in p
                 if "WATER_VIEW" in e}
        self.assertEqual(views["fresnel"], wb.VIEW_SCENES)
        self.assertEqual(views["foam"], wb.VIEW_SCENES)
        self.assertEqual(views["tiers"], wb.VIEWS["tiers"])
        bursts = [e["WATER_SCENES"].split() + a for s, _, e, a in p if s == "bursts"]
        self.assertEqual(bursts, [wb.STATIC + wb.PANS + ["60", "2"]])
        self.assertEqual(wb.plan("stills"), [("stills", "stills", {}, [])])

    def test_capture_runs_each_sweep_through_water_shots_with_its_environment(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            stub = d / "shots.sh"
            calls = d / "calls.txt"
            line = '"$1|${2:-}|${3:-}|${WATER_SCENES:-}|${WATER_VIEW:-}"'
            stub.write_text(f"#!/bin/sh\necho {line} >> {calls}\n")
            stub.chmod(0o755)
            saved = wb.MIN_MEM_MB, wb.MIN_DISK_GB
            wb.MIN_MEM_MB = wb.MIN_DISK_GB = 0
            try:
                rc, _ = run(wb.capture, d / "run", "bursts", stub)
            finally:
                wb.MIN_MEM_MB, wb.MIN_DISK_GB = saved
            self.assertEqual(rc, 0)
            out, n, every, scenes, view = calls.read_text().strip().split("|")
            self.assertEqual((out, n, every, view), (str(d / "run" / "bursts"), "60", "2", ""))
            self.assertEqual(scenes.split(), wb.STATIC + wb.PANS)
            self.assertTrue((d / "run" / "bursts" / "vram.tsv").exists())

    def test_a_sweep_does_not_start_without_room(self):
        with tempfile.TemporaryDirectory() as d:
            saved = wb.MIN_MEM_MB
            wb.MIN_MEM_MB = 10**9
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    rc, _ = run(wb.capture, pathlib.Path(d), "stills", pathlib.Path("/bin/false"))
            finally:
                wb.MIN_MEM_MB = saved
            self.assertEqual(rc, 3)


class Vram(unittest.TestCase):
    def test_each_amdgpu_client_counts_once(self):
        a = "pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t7\ndrm-total-vram:\t1000 KiB\n" \
            "drm-total-gtt:\t20 KiB\n"
        other = "drm-driver:\ti915\ndrm-client-id:\t9\ndrm-total-vram:\t5 KiB\n"
        b = a.replace("\t7", "\t8").replace("1000", "24")
        self.assertEqual(wb.VramWatch.fdinfo_totals([a, a, b, other, "pos:\t0\n"]), (1024, 40))


class Table(unittest.TestCase):
    def test_a_run_becomes_both_lanes_the_burst_rows_and_ins4(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            regions, stills, bursts = d / "regions", d / "run" / "stills", d / "run" / "bursts"
            for p in (regions, stills, bursts):
                p.mkdir(parents=True)
            (regions / "wg-t.toml").write_text(REGION)
            save(frame(200), stills / "wg-t.png")
            save(frame(100), stills / "wg-t-ref.png")
            for i in range(3):
                save(frame(200 + i), bursts / f"wg-t-{i:03d}.png")
            journal = ["t,mean_ms,cpu_ms,main_ms,gpu_ms,gpu_water"]
            journal += [f"{i},10,20,5,{i},{i / 10}" for i in range(12)]
            (stills / "wg-t.fps.csv").write_text("\n".join(journal) + "\n")
            (stills / "wg-t.log").write_text(
                "\x1b[32m INFO\x1b[0m capture: image settled after 1571 frames, aging 1000\n"
                "GPU_PASSES frames=60 water_hiz_seed=9.0\n"
                "GPU_PASSES frames=60 water_hiz_seed=1.0 water_snapshot=0.4\n"
            )
            (stills / "vram.tsv").write_text(f"{stills / 'wg-t.png'}\t2048\t1024\n")
            rc, _ = run(wb.table, d / "run", d / "out", regions)
            self.assertEqual(rc, 0)
            base = json.loads((d / "out" / "baseline.json").read_text())
            self.assertEqual(sorted(base), ["wg-t", "wg-t-ref"])
            self.assertAlmostEqual(base["wg-t"]["lum_vs_ref"]["value"], 2.0, places=2)
            self.assertEqual(base["wg-t"]["shimmer"]["source"], "bursts")
            self.assertGreater(base["wg-t"]["luma_sd"]["value"], 0)
            self.assertNotIn("source", base["wg-t"]["luma_sd"])
            perf = json.loads((d / "out" / "ins4.json").read_text())["stills/wg-t"]
            self.assertEqual(perf["rows"], 8)
            self.assertEqual(perf["gpu_ms"], 7.5)  # the median of the last 8 rows, 4-11
            self.assertEqual(perf["water_hiz_seed"], 1.0)  # the last GPU_PASSES line
            self.assertEqual(perf["water_hiz_reduce"], 0.0)
            self.assertEqual((perf["settle"], perf["age"], perf["flat"]), (1571, 1000, False))
            self.assertEqual((perf["vram_mib"], perf["gtt_mib"]), (2.0, 1.0))
            md = (d / "out" / "table.md").read_text()
            self.assertIn("| shimmer | shimmer |", md)
            self.assertIn("| lum_vs_ref | lum_vs_ref | 2 | min 1 pass |", md)
            self.assertRegex(md, r"\| pop \| pop \| \S+ \|  info \|")
            self.assertIn("(burst) |", md)
            self.assertIn("| stills/wg-t | 7.5 |", md)


class Checks(unittest.TestCase):
    def test_selfcheck_names_the_pixels_that_differ(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = pathlib.Path(d) / "a", pathlib.Path(d) / "b"
            a.mkdir()
            b.mkdir()
            for p in (a, b):
                save(frame(90), p / "same.png")
            save(frame(90), a / "moved.png")
            img = frame(90)
            img[5:7, 10:13] += 6
            save(img, b / "moved.png")
            rc, out = run(wb.selfcheck, a, b)
            self.assertEqual(rc, 1)
            self.assertIn("moved.png: 6 px differ, max 6 codes", out)
            self.assertIn("bbox x10-12 y5-6, > 4 codes 6 px", out)
            self.assertIn("identical 1, differing 1, unmatched 0", out)
            (b / "moved.png").unlink()  # a shot only one run took is counted, not failed
            rc, out = run(wb.selfcheck, a, b)
            self.assertEqual(rc, 0)
            self.assertIn("identical 1, differing 0, unmatched 1", out)

    def test_diff_lists_moved_relabelled_and_informational_rows(self):
        old = {"s": {"a": {"metric": "m", "value": 1.0}, "b": {"metric": "view_std", "value": 2},
                     "c": {"metric": "pop", "value": 3}, "d": {"metric": "m", "value": 4}}}
        new = {"s": {"a": {"metric": "m", "value": 1.5}, "b2": {"metric": "frame_std", "value": 2},
                     "c": {"metric": "pop", "value": 3, "gated": False},
                     "d": {"metric": "m", "value": 4}}}
        self.assertEqual(wb.changes(old, new), [
            ("s", "a", "value 1 -> 1.5"),
            ("s", "b", "removed (view_std 2)"),
            ("s", "c", "informational"),
            ("s", "b2", "added (frame_std 2)"),
        ])


if __name__ == "__main__":
    unittest.main()
