"""Fixtures for scripts/water-shots.sh: where it writes and what reaches the client, run in a
throwaway repo with a stub `cargo` on PATH and a stub client that lists one scene, `water-a`, echoes
the capture knobs it receives into its log and writes the files a capture would.

    python3 -m unittest discover -s scripts/tests
"""

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "water-shots.sh"

CLIENT = """#!/bin/bash
if [ "${WOW_CAPTURE:-}" = list ]; then echo water-a; exit 0; fi
[ -z "${WOW_CAPTURE_STABLE+x}" ] || echo "WOW_CAPTURE_STABLE is set: $WOW_CAPTURE_STABLE"
[ -z "${WOW_CAPTURE_AGE+x}" ] || echo "WOW_CAPTURE_AGE is set: $WOW_CAPTURE_AGE"
echo "WOW_CAPTURE_PAN=${WOW_CAPTURE_PAN:-}"
echo "WOW_MSAA=${WOW_MSAA:-}"
if [ -n "${STUB_ABORT:-}" ]; then echo "wgpu error: Validation Error"; exit 134; fi
if [ "${WOW_CAPTURE_BURST:-0}" -gt 0 ]; then
  for i in $(seq 0 $((WOW_CAPTURE_BURST - 1))); do
    : > "${WOW_CAPTURE_OUT%.png}-$(printf %03d "$i").png"
  done
else
  : > "$WOW_CAPTURE_OUT"
fi
"""


@unittest.skipUnless(shutil.which("bash"), "water-shots.sh is a bash script")
class Shots(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = pathlib.Path(tmp.name)
        self.repo = self.tmp / "repo"
        self.caller = self.tmp / "caller"
        for d in ("scripts", "target/release", "docs"):
            (self.repo / d).mkdir(parents=True)
        (self.tmp / "bin").mkdir()
        self.caller.mkdir()
        shutil.copy(SCRIPT, self.repo / "scripts" / "water-shots.sh")
        (self.tmp / "bin" / "cargo").write_text("#!/bin/sh\nexit 0\n")
        (self.repo / "target" / "release" / "benilla").write_text(CLIENT)
        for p in ("scripts/water-shots.sh", "target/release/benilla"):
            (self.repo / p).chmod(0o755)
        (self.tmp / "bin" / "cargo").chmod(0o755)

    def shots(self, cwd, *args, **env):
        """water-shots.sh run from `cwd`, with only the knobs given set."""
        base = {k: v for k, v in os.environ.items()
                if not k.startswith(("WATER_", "WOW_CAPTURE"))}
        base["PATH"] = f"{self.tmp / 'bin'}:{base['PATH']}"
        return subprocess.run([str(self.repo / "scripts" / "water-shots.sh"), *map(str, args)],
                              cwd=cwd, env=base | env, capture_output=True, text=True)

    def test_the_default_directory_is_target_visual_water(self):
        r = self.shots(self.caller)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.repo / "target/visual/water/water-a.png").exists())
        self.assertTrue((self.repo / "target/visual/water/water-a-ref.png").exists())

    def test_a_relative_directory_resolves_against_the_caller(self):
        r = self.shots(self.caller, "rel")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.caller / "rel" / "water-a.png").exists())

    def test_a_fresh_directory_in_the_repo_is_refused_and_removed(self):
        r = self.shots(self.caller, self.repo / "foo")
        self.assertEqual(r.returncode, 2)
        self.assertFalse((self.repo / "foo").exists())
        self.assertIn("is in the repo outside target/", r.stderr)

    def test_an_existing_directory_in_the_repo_is_refused_and_kept(self):
        r = self.shots(self.caller, self.repo / "docs")
        self.assertEqual(r.returncode, 2)
        self.assertTrue((self.repo / "docs").is_dir())
        self.assertEqual(list((self.repo / "docs").iterdir()), [])

    def test_a_relative_directory_from_the_repo_root_is_refused(self):
        r = self.shots(self.repo, "docs/sub")
        self.assertEqual(r.returncode, 2)
        self.assertFalse((self.repo / "docs" / "sub").exists())

    def test_every_level_a_refusal_made_is_removed_and_no_other(self):
        r = self.shots(self.caller, self.repo / "shots" / "deep" / "er")
        self.assertEqual(r.returncode, 2)
        self.assertFalse((self.repo / "shots").exists())
        (self.repo / "keep").mkdir()
        r = self.shots(self.caller, self.repo / "keep" / "new")
        self.assertEqual(r.returncode, 2)
        self.assertTrue((self.repo / "keep").is_dir())
        self.assertFalse((self.repo / "keep" / "new").exists())

    def test_a_directory_under_target_is_accepted(self):
        r = self.shots(self.caller, self.repo / "target" / "shots")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.repo / "target" / "shots" / "water-a.png").exists())

    def test_a_burst_writes_and_names_its_shots(self):
        r = self.shots(self.caller, self.tmp / "burst", 3)
        self.assertEqual(r.returncode, 0, r.stderr)
        for i in range(3):
            self.assertTrue((self.tmp / "burst" / f"water-a-{i:03d}.png").exists())
        self.assertFalse((self.tmp / "burst" / "water-a-ref-000.png").exists())
        self.assertIn("water-a-000.png … water-a-002.png", r.stdout)

    def test_the_settle_and_age_stay_the_scenes_own_and_other_knobs_pass(self):
        r = self.shots(self.caller, self.tmp / "env", WOW_CAPTURE_STABLE="30",
                       WOW_CAPTURE_AGE="5", WOW_CAPTURE_PAN="20,10")
        self.assertEqual(r.returncode, 0, r.stderr)
        log = (self.tmp / "env" / "water-a.log").read_text()
        self.assertNotIn("WOW_CAPTURE_STABLE", log)
        self.assertNotIn("WOW_CAPTURE_AGE", log)
        self.assertIn("WOW_CAPTURE_PAN=20,10", log)

    def test_msaa_is_off_whatever_the_caller_exported(self):
        r = self.shots(self.caller, self.tmp / "msaa", WOW_MSAA="4")
        self.assertEqual(r.returncode, 0, r.stderr)
        for stem in ("water-a", "water-a-ref"):
            log = (self.tmp / "msaa" / f"{stem}.log").read_text()
            self.assertIn("WOW_MSAA=off\n", log)

    def test_water_msaa_multisamples_both_lanes(self):
        r = self.shots(self.caller, self.tmp / "msaa", WATER_MSAA="4", WOW_MSAA="off")
        self.assertEqual(r.returncode, 0, r.stderr)
        for stem in ("water-a", "water-a-ref"):
            log = (self.tmp / "msaa" / f"{stem}.log").read_text()
            self.assertIn("WOW_MSAA=4\n", log)

    def test_water_msaa_refuses_the_default_directory(self):
        r = self.shots(self.caller, WATER_MSAA="4")
        self.assertEqual(r.returncode, 2)
        self.assertIn("give it a <dir> of its own", r.stderr)
        self.assertFalse((self.repo / "target" / "visual").exists())

    def test_every_log_names_the_msaa_level(self):
        for level, sub in (("4", "m4"), (None, "off")):
            env = {"WATER_MSAA": level} if level else {}
            r = self.shots(self.caller, self.tmp / sub, **env)
            self.assertEqual(r.returncode, 0, r.stderr)
            for stem in ("water-a", "water-a-ref"):
                log = (self.tmp / sub / f"{stem}.log").read_text()
                self.assertTrue(log.startswith(f"water-shots: msaa {level or 'off'}\n"), log)
                self.assertIn(f"WOW_MSAA={level or 'off'}\n", log)
            self.assertIn(f"(msaa {level or 'off'})", r.stdout)

    def test_an_msaa_level_wgpu_does_not_express_is_refused(self):
        r = self.shots(self.caller, self.tmp / "msaa", WATER_MSAA="3")
        self.assertEqual(r.returncode, 2)
        self.assertIn("not 3", r.stderr)
        self.assertFalse((self.tmp / "msaa").exists())

    def test_a_client_that_aborts_fails_the_run(self):
        r = self.shots(self.caller, self.tmp / "abort", WATER_MSAA="4", STUB_ABORT="1")
        self.assertEqual(r.returncode, 1)
        self.assertIn("capture water-a FAILED", r.stderr)
        self.assertIn("Validation Error", (self.tmp / "abort" / "water-a.log").read_text())
        self.assertFalse((self.tmp / "abort" / "water-a-ref.log").exists())

    def test_a_perf_run_takes_no_burst(self):
        r = self.shots(self.caller, self.tmp / "perf", 3, WATER_PERF="1200")
        self.assertEqual(r.returncode, 2)
        self.assertIn("not a burst or a view", r.stderr)


if __name__ == "__main__":
    unittest.main()
