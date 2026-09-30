"""Fixtures for scripts/water_perf.py: synthetic perf-shot logs whose summary is known.

    python3 -m unittest discover -s scripts/tests
"""

import contextlib
import io
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import water_perf as wp  # noqa: E402

HEADER = "t,mean_ms,main_ms,gpu_ms\n"


def gpu_passes(seed, fixed=0.078, prepass=0.3):
    return (f"GPU_PASSES frames=120 main_opaque_pass_3d={seed * 2:.3f} water_hiz_seed={seed:.3f} "
            f"early prepass={prepass:.3f} water_snapshot=0.400 {wp.FIXED_PASS}={fixed:.3f}")


def perf_log(seeds, frozen=(9.0, 9.0), fixed=0.078, probe=True):
    """A perf shot's log: GPU_PASSES under the frozen clock, the probe's mark, the probe seconds
    (the first SETTLE of them still mixed), and the FPS_PROBE line, with the ANSI colours a
    capture log carries."""
    lines = [gpu_passes(s, fixed) for s in frozen]
    if probe:
        lines.append("\x1b[2m2026-09-30T10:00:00Z\x1b[0m \x1b[32m INFO\x1b[0m probe: scene aged "
                     "1000 frames; vsync off, warming 60 frames")
    lines += [gpu_passes(s, fixed) for s in seeds]
    if probe:
        lines.append("FPS_PROBE scenario=water-test frames=1200 mean_ms=6.50 p50_ms=6.40 "
                     "p95_ms=7.10 fps=153.8 px=1600x900 world_px=3200x1800 cpu_ms=12.00 "
                     "cpu_pct=185 present=AutoNoVsync gpu_n=1100 gpu_mean=5.90 gpu_p50=5.80")
    return "\n".join(lines) + "\n"


class ParseLog(unittest.TestCase):
    def test_the_probe_seconds_are_the_ones_after_the_mark_less_the_settle(self):
        # Two frozen seconds at 9.0, two mixed probe seconds at 5.0, then 1.0, 1.2, 1.1.
        r = wp.parse_log(perf_log([5.0, 5.0, 1.0, 1.2, 1.1]))
        self.assertTrue(r["probe"])
        self.assertEqual(r["seconds"], 3)
        self.assertAlmostEqual(r["passes"]["water_hiz_seed"], 1.1)
        self.assertAlmostEqual(r["passes"]["main_opaque_pass_3d"], 2.2)
        # A pass name with a space keeps its whole name.
        self.assertAlmostEqual(r["passes"]["early prepass"], 0.3)
        self.assertAlmostEqual(r["gpu_water"], 1.1 + 0.4)
        self.assertAlmostEqual(r["gpu_passes"], 2.2 + 1.1 + 0.3 + 0.4 + 0.078)
        self.assertEqual(r["mean_ms"], 6.5)
        self.assertEqual(r["gpu_mean"], 5.9)
        self.assertEqual(r["present"], "AutoNoVsync")

    def test_a_probe_held_at_60_hz_with_the_gpu_idle_is_paced(self):
        r = wp.parse_log(perf_log([1, 1, 1]))
        self.assertFalse(wp.paced(r))
        held = perf_log([1, 1, 1]).replace("mean_ms=6.50", "mean_ms=16.67")
        self.assertTrue(wp.paced(wp.parse_log(held)))
        # A frame whose GPU work fills 1/60 s is its cost, not a pace.
        full = held.replace("gpu_mean=5.90", "gpu_mean=16.20")
        self.assertFalse(wp.paced(wp.parse_log(full)))

    def test_a_log_without_the_probe_has_no_seconds(self):
        r = wp.parse_log(perf_log([1.0, 1.0, 1.0], probe=False))
        self.assertFalse(r["probe"])
        self.assertNotIn("passes", r)


class Measure(unittest.TestCase):
    def write(self, d, stem, log, journal=None):
        (d / f"{stem}.perf.log").write_text(log)
        if journal is not None:
            (d / f"{stem}.perf.fps.csv").write_text(journal)

    def test_a_shot_whose_fixed_pass_runs_slow_is_flagged_as_clocked_down(self):
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            rows = "".join(f"{i},6.5,{m},6.0\n" for i, m in enumerate([9.0, 9.0, 4.0, 5.0, 6.0]))
            self.write(d, "water-a", perf_log([5, 5, 1, 1, 1]), "# adapter\n" + HEADER + rows)
            self.write(d, "water-a-ref", perf_log([5, 5, 1, 1, 1]))
            self.write(d, "water-b", perf_log([5, 5, 1, 1, 1], fixed=0.156))
            r = wp.measure(d, [])
            self.assertEqual(sorted(r), ["water-a", "water-a-ref", "water-b"])
            # main_ms: the journal's last `seconds` rows, the probe's own.
            self.assertAlmostEqual(r["water-a"]["main_ms"], 5.0)
            self.assertTrue(r["water-a"]["clock_ok"])
            self.assertFalse(r["water-b"]["clock_ok"])
            self.assertAlmostEqual(r["water-b"]["fixed_ratio"], 2.0)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(wp.main([str(d)]), 1)
            self.assertIn("clock?", out.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(wp.main([str(d), "water-a", "water-a-ref"]), 0)


if __name__ == "__main__":
    unittest.main()
