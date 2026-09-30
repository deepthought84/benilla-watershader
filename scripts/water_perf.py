#!/usr/bin/env python3
"""Water perf (INS-4): the cost of each golden water scene, read from a `water-shots.sh` perf run.

    WATER_PERF=1200 scripts/water-shots.sh <dir>     # <scene>.perf.log/.fps.csv, both lanes
    scripts/water_perf.py <dir> [<stem> ...]         # one line per shot
    scripts/water_perf.py <dir> --json out.json      # also write the numbers

A perf shot builds and ages its scene like a still, then runs its frames with vsync off and the
loop continuous (the capture's `WOW_FPS_PROBE`), so the GPU stays busy and its clock stays up;
under a 60 Hz pace a light frame lets the clock idle down and every pass reads slower. Per shot,
from <stem>.perf.log:
- the FPS_PROBE line: wall ms per frame, the whole-frame GPU meter (`WOW_GPU_MS`, gpu_mean), CPU
  ms per frame over every thread;
- each probe second's GPU_PASSES line (ms per frame per pass), after the log's "probe: scene aged"
  line and without its first SETTLE seconds, which hold frozen-clock frames and the warm-up; the
  median of each pass over the rest;
- main_ms, the main thread's CPU per frame, the median over the journal's rows of the same seconds
  (<stem>.perf.fps.csv, one row per GPU_PASSES line).

FIXED_PASS does the same work on every scene and lane, so its time moves only with the GPU's
clock (`fixed_ratio`, against the run's median). A shot where it reads more than CLOCK_SLACK times
the median ran at a lower clock, as a light frame whose CPU is the limit can, and its passes read
slower by up to that factor ("clock?"). A probe whose wall frame sits at 1/60 s while the GPU idles
part of it was paced, not uncapped ("paced"), and its clock idled too.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics
import sys

# The UI's gamma decode: one full-window pass, the same work in every shot.
FIXED_PASS = "ui_gamma_decode"
CLOCK_SLACK = 1.25
# A wall frame this close to 1/60 s with the GPU busy under PACED_GPU of it is a 60 Hz pace.
PACE_MS = 1000.0 / 60.0
PACED_GPU = 0.9
# Probe seconds dropped after the probe starts: the first flush still holds frozen-clock frames,
# the next can hold the rest of the 60-frame warm-up.
SETTLE = 2
PROBE_MARK = "probe: scene aged"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# `name=value` pairs; a pass name can hold a space ("early prepass").
PAIR = re.compile(r"(\S(?:[^=]*\S)?)=(\S+)")
# Columns of the summary, from the shot's record.
SHOWN = ["frames", "mean_ms", "gpu_mean", "gpu_passes", "gpu_water", "water_hiz_seed",
         "water_hiz_reduce", "water_snapshot", "main_transparent_pass_3d", "main_opaque_pass_3d",
         "cpu_ms", "main_ms", FIXED_PASS]


def pairs(text: str) -> dict[str, str]:
    return {k: v for k, v in PAIR.findall(text)}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def parse_log(text: str) -> dict:
    """The probe's numbers from one perf shot's log: the FPS_PROBE fields and the per-pass
    medians over the probe seconds kept (`seconds`)."""
    lines = [ANSI.sub("", l) for l in text.splitlines()]
    start = next((i for i, l in enumerate(lines) if PROBE_MARK in l), None)
    rec: dict = {"probe": start is not None}
    probe = next((l for l in lines if l.startswith("FPS_PROBE ")), None)
    if probe:
        rec.update({k: num(v) for k, v in pairs(probe[len("FPS_PROBE "):]).items()})
    if start is None:
        return rec
    seconds = []
    for l in lines[start + 1:]:
        if l.startswith("GPU_PASSES "):
            seconds.append({k: float(v) for k, v in pairs(l[len("GPU_PASSES "):]).items()})
    kept = seconds[SETTLE:]
    rec["seconds"] = len(kept)
    passes: dict[str, float] = {}
    for name in sorted({k for s in kept for k in s} - {"frames"}):
        passes[name] = statistics.median(s.get(name, 0.0) for s in kept)
    rec["passes"] = passes
    rec["gpu_passes"] = sum(passes.values())
    rec["gpu_water"] = sum(v for k, v in passes.items() if k.startswith("water_"))
    return rec


def journal_main_ms(text: str, seconds: int) -> float | None:
    """The median main-thread ms over the journal's last `seconds` rows that carry GPU cells."""
    rows = [l for l in text.splitlines() if l and not l.startswith("#")]
    if len(rows) < 2 or seconds <= 0:
        return None
    body = [r for r in csv.DictReader(rows) if r.get("gpu_ms")]
    vals = [float(r["main_ms"]) for r in body[-seconds:] if r.get("main_ms")]
    return statistics.median(vals) if vals else None


def measure(directory: pathlib.Path, stems: list[str]) -> dict:
    """{stem: record} for every <stem>.perf.log in the directory, or the stems given."""
    logs = sorted(directory.glob("*.perf.log"))
    found = {p.name[: -len(".perf.log")]: p for p in logs}
    out: dict = {}
    for stem in stems or sorted(found):
        if stem not in found:
            print(f"{stem}: no {stem}.perf.log in {directory}", file=sys.stderr)
            continue
        rec = parse_log(found[stem].read_text(errors="replace"))
        journal = directory / f"{stem}.perf.fps.csv"
        if journal.exists():
            rec["main_ms"] = journal_main_ms(journal.read_text(), rec.get("seconds", 0))
        out[stem] = rec
    fixed = [r["passes"][FIXED_PASS] for r in out.values() if FIXED_PASS in r.get("passes", {})]
    ref = statistics.median(fixed) if fixed else None
    for r in out.values():
        f = r.get("passes", {}).get(FIXED_PASS)
        r["fixed_ratio"] = f / ref if f is not None and ref else None
        r["clock_ok"] = None if r["fixed_ratio"] is None else r["fixed_ratio"] <= CLOCK_SLACK
    return out


def paced(r: dict) -> bool:
    """The probe ran at the 60 Hz pace instead of its frame cost."""
    wall, gpu = r.get("mean_ms"), r.get("gpu_mean")
    if not isinstance(wall, float):
        return False
    return abs(wall - PACE_MS) < 0.35 and (not isinstance(gpu, float) or gpu < PACED_GPU * wall)


def value(r: dict, key: str):
    return r.get(key, r.get("passes", {}).get(key))


def fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def report(results: dict) -> int:
    """One line per shot; the count of shots that are not a clean saturated probe."""
    bad = 0
    heads = ["frames", "wall", "gpu", "passes", "water", "seed", "reduce", "snap", "transp",
             "opaque", "cpu", "main", "fixed"]
    print(f"{'shot':34s} " + " ".join(f"{h:>7s}" for h in heads))
    for stem, r in results.items():
        why = []
        if not r.get("probe") or "mean_ms" not in r:
            why.append("no probe")
        elif r.get("seconds", 0) < 3:
            why.append(f"{r.get('seconds', 0)} probe seconds")
        if r.get("present") not in (None, "AutoNoVsync", "Immediate", "Mailbox"):
            why.append(f"present {r.get('present')}")
        if paced(r):
            why.append("paced")
        if r.get("clock_ok") is False:
            why.append(f"clock? ({FIXED_PASS} {r['fixed_ratio']:.2f}x the run's)")
        bad += bool(why)
        cells = " ".join(f"{fmt(value(r, k)):>7s}" for k in SHOWN)
        print(f"{stem:34s} {cells}  {' '.join(why)}")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", type=pathlib.Path, help="directory of a WATER_PERF run")
    ap.add_argument("stems", nargs="*", help="shots, e.g. water-loch-modan water-loch-modan-ref")
    ap.add_argument("--json", type=pathlib.Path, help="write the records as JSON")
    a = ap.parse_args(argv)
    results = measure(a.dir, a.stems)
    bad = report(results)
    if a.json:
        a.json.write_text(json.dumps(results, indent=1))
    return 1 if bad or not results else 0


if __name__ == "__main__":
    sys.exit(main())
