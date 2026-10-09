#!/usr/bin/env python3
"""The water baseline: the golden sweeps it is captured from, its table, and the checks on it.

    scripts/water_baseline.py capture <dir> [stills|views|bursts|repeat|all]
    scripts/water_baseline.py table <dir> <out>          # <out>/baseline.json, ins4.json, table.md
    scripts/water_baseline.py selfcheck <dir a> <dir b>  # two runs of one build, PNG by PNG
    scripts/water_baseline.py diff <old.json> <new.json> # the rows that differ between two tables

`capture` runs `scripts/water-shots.sh` once per sweep, one client at a time, with the environment
water-shots.sh needs (a display, $WOW_DATA) taken from the caller:
- stills: every golden water scene, Improved and its Reference twin, into <dir>/stills;
- views: the WATER_VIEW captures the rows read (VIEWS), into <dir>/stills beside the stills;
- bursts: the static and pan bursts (STATIC, PANS), BURST shots EVERY steps apart, <dir>/bursts;
- repeat: the stills again into <dir>/repeat and two bursts into <dir>/repeat-bursts, the
  selfcheck's second run.
Before every sweep it requires MIN_MEM_MB of MemAvailable and MIN_DISK_GB free on <dir>'s file
system, and beside it samples each capture client's peak VRAM and GTT (amdgpu fdinfo) into
<sweep dir>/vram.tsv.

`table` measures every region file's rows (scripts/water_metrics.py): the Improved lane on
<dir>/stills, the Reference twins on <dir>/stills, the burst rows on <dir>/bursts, a row missing its
capture in stills taking the bursts' value. INS-4 per shot: the median of each journal column over
the last JOURNAL_ROWS aged one-second rows of <shot>.fps.csv, and the last GPU_PASSES line of its
log.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import threading

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import water_metrics as wm  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHOTS = ROOT / "scripts" / "water-shots.sh"

# The static bursts the temporal rows read, and the pans (a scene's own sweep, `scene_pan`).
STATIC = ["water-falfarren-river", "water-elwynn-stream", "water-crystal-3p-noon",
          "water-baradin-noon"]
PANS = ["water-pan-burst", "water-elwynn-stream-pan", "water-alameth-bank-pan",
        "water-falfarren-pan", "water-elwynn-falls-pan"]
# 60 shots 2 fixed steps apart: 2 s at 1/30 s.
BURST, EVERY = 60, 2
# The views the contact, deep and seam rows read, and the scenes each is captured at.
VIEW_SCENES = [
    "water-loch-modan", "water-menethil-wreck-noon", "water-booty-underdeck",
    "water-booty-harbour", "water-stv-river", "water-darkshore-pier-noon", "water-elwynn-falls",
    "water-stormwind-canal", "water-crystal-3p-noon", "water-baradin-noon", "water-savage-coast",
    "water-westfall-noon", "water-stv-mouth", "water-alameth-bank-noon", "water-felwood-jademir",
    "water-swamp-sorrows", "water-darrowmere", "water-forgotten-coast",
]
VIEWS = {
    "fresnel": VIEW_SCENES,
    "foam": VIEW_SCENES,
    "tiers": ["water-dam-foot", "water-elwynn-falls", "water-loch-modan", "water-baradin-noon"],
}
REPEAT_BURSTS = ["water-baradin-noon", "water-pan-burst"]

MIN_MEM_MB = 4000
MIN_DISK_GB = 10

# The fps journal's columns a shot's INS-4 record keeps, and the GPU_PASSES passes.
COLS = ["mean_ms", "cpu_ms", "main_ms", "gpu_ms", "gpu_opaque", "gpu_static", "gpu_transp",
        "gpu_glow", "gpu_post", "gpu_ui", "gpu_other", "gpu_water"]
PASSES = ["water_hiz_seed", "water_hiz_reduce", "water_snapshot", "main_transparent_pass_3d"]
# The last journal rows taken, all inside the 1000-step age.
JOURNAL_ROWS = 8
ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ---- capture ------------------------------------------------------------------------------------


def plan(what: str) -> list[tuple[str, str, dict, list[str]]]:
    """The sweeps `capture <what>` runs, in order: (sub-directory, label, environment,
    water-shots.sh's arguments after the directory)."""
    sweeps = []
    if what in ("stills", "all"):
        sweeps.append(("stills", "stills", {}, []))
    if what in ("views", "all"):
        for view, scenes in VIEWS.items():
            env = {"WATER_SCENES": " ".join(scenes), "WATER_VIEW": view}
            sweeps.append(("stills", f"view-{view}", env, []))
    if what in ("bursts", "all"):
        env = {"WATER_SCENES": " ".join(STATIC + PANS)}
        sweeps.append(("bursts", "bursts", env, [str(BURST), str(EVERY)]))
    if what in ("repeat", "all"):
        sweeps.append(("repeat", "stills", {}, []))
        env = {"WATER_SCENES": " ".join(REPEAT_BURSTS)}
        sweeps.append(("repeat-bursts", "bursts", env, [str(BURST), str(EVERY)]))
    return sweeps


def mem_available_mb() -> int:
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def room(directory: pathlib.Path) -> tuple[int, int]:
    """(MemAvailable MB, free GB on the file system holding `directory`)."""
    d = directory
    while not d.exists():
        d = d.parent
    return mem_available_mb(), shutil.disk_usage(d).free // 2**30


class VramWatch:
    """Each capture client's peak VRAM and GTT in KiB while it runs, keyed by its $WOW_CAPTURE_OUT:
    every `period` s, the sum over the client's amdgpu DRM clients (fdinfo, one per drm-client-id)
    of drm-total-vram and drm-total-gtt."""

    def __init__(self, period: float = 0.5):
        self.period = period
        self.peak: dict[str, list[int]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def fdinfo_totals(texts) -> tuple[int, int]:
        """(VRAM, GTT) KiB summed over the amdgpu clients the fdinfo texts name, each once."""
        vram, gtt = {}, {}
        for text in texts:
            fields = {}
            for line in text.splitlines():
                k, _, v = line.partition(":")
                if v.strip():
                    fields[k.strip()] = v.split()[0]
            if fields.get("drm-driver") != "amdgpu" or "drm-client-id" not in fields:
                continue
            cid = fields["drm-client-id"]
            if "drm-total-vram" in fields:
                vram[cid] = int(fields["drm-total-vram"])
            if "drm-total-gtt" in fields:
                gtt[cid] = int(fields["drm-total-gtt"])
        return sum(vram.values()), sum(gtt.values())

    def sample(self) -> None:
        for proc in pathlib.Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                if (proc / "comm").read_text().strip() != "benilla":
                    continue
                env = (proc / "environ").read_bytes().split(b"\0")
                out = next((e[len(b"WOW_CAPTURE_OUT="):].decode() for e in env
                            if e.startswith(b"WOW_CAPTURE_OUT=")), None)
                if not out:
                    continue
                texts = []
                for fd in (proc / "fdinfo").iterdir():
                    try:
                        texts.append(fd.read_text())
                    except OSError:
                        pass
            except OSError:
                continue
            v, g = self.fdinfo_totals(texts)
            peak = self.peak.setdefault(out, [0, 0])
            peak[0], peak[1] = max(peak[0], v), max(peak[1], g)

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            self.sample()

    def __enter__(self) -> VramWatch:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def append(self, tsv: pathlib.Path) -> None:
        """One line per capture: its output path, peak VRAM KiB, peak GTT KiB."""
        with tsv.open("a") as f:
            for out, (v, g) in self.peak.items():
                f.write(f"{out}\t{v}\t{g}\n")


def capture(directory: pathlib.Path, what: str, shots: pathlib.Path = SHOTS) -> int:
    """Run the sweeps of `plan(what)` into `directory`; stops at the first one that fails."""
    for sub, label, env, args in plan(what):
        mem, disk = room(directory)
        print(f"{label}: MemAvailable {mem} MB, {disk} GB free")
        if mem < MIN_MEM_MB or disk < MIN_DISK_GB:
            print(f"{label}: needs {MIN_MEM_MB} MB and {MIN_DISK_GB} GB; not started",
                  file=sys.stderr)
            return 3
        out = directory / sub
        out.mkdir(parents=True, exist_ok=True)
        log = out / f"sweep-{label}.log"
        with VramWatch() as watch, log.open("w") as f:
            rc = subprocess.run([str(shots), str(out), *args], env={**os.environ, **env},
                                stdout=f, stderr=subprocess.STDOUT).returncode
        watch.append(out / "vram.tsv")
        print(f"{label}: exit {rc}")
        if rc != 0:
            print("\n".join(log.read_text(errors="replace").splitlines()[-3:]), file=sys.stderr)
            return 1
    return 0


# ---- table --------------------------------------------------------------------------------------


def rows(directory: pathlib.Path, regions: pathlib.Path) -> dict:
    """{scene lane: {label: row}}: the Improved stills, burst rows where a still has none, and the
    Reference twins (`<scene>-ref`)."""

    def measure(d, lane=""):
        return wm.measure(d, regions, [], lane) if d.exists() else {}

    imp = measure(directory / "stills")
    ref = measure(directory / "stills", "-ref")
    bur = measure(directory / "bursts")
    merged = {}
    for scene in sorted(set(imp) | set(bur)):
        r = dict(imp.get(scene, {}))
        for label, b in bur.get(scene, {}).items():
            if label not in r or r[label].get("value") is None:
                r[label] = dict(b, source="bursts")
        merged[scene] = r
    merged.update(ref)
    return merged


def ins4(directory: pathlib.Path) -> dict:
    """{<sweep>/<shot>: record} from each shot's journal, log and its own sweep's vram.tsv (a
    scene shot in two sweeps has a peak in each)."""
    vram = {}
    for sub in ["stills", "bursts"]:
        t = directory / sub / "vram.tsv"
        if t.exists():
            for line in t.read_text().splitlines():
                out, v, g = line.split("\t")
                vram[f"{sub}/{pathlib.Path(out).stem}"] = (int(v) / 1024, int(g) / 1024)
    shots = {}
    for sub in ["stills", "bursts"]:
        for journal in sorted((directory / sub).glob("*.fps.csv")):
            stem = journal.name[: -len(".fps.csv")]
            lines = [l for l in journal.read_text().splitlines() if l and not l.startswith("#")]
            head = lines[0].split(",")
            recs = [dict(zip(head, l.split(","))) for l in lines[1:]]
            recs = [r for r in recs if r.get("gpu_ms")][-JOURNAL_ROWS:]
            rec = {"source": sub, "rows": len(recs)}
            for c in COLS:
                vals = [float(r[c]) for r in recs if r.get(c) not in (None, "")]
                rec[c] = statistics.median(vals) if vals else None
            log = directory / sub / f"{stem}.log"
            if log.exists():
                txt = ANSI.sub("", log.read_text(errors="replace"))
                gp = [l for l in txt.splitlines() if l.startswith("GPU_PASSES")]
                if gp:
                    kv = dict(tok.split("=", 1) for tok in gp[-1].split()[1:] if "=" in tok)
                    for p in PASSES:
                        rec[p] = float(kv[p]) if p in kv else 0.0
                rec["live_compiles"] = txt.count("pipeline compiled LIVE")
                rec["shader_errors"] = len(
                    re.findall(r"failed to process shader|Validation Error|panicked", txt)
                )
                m = re.search(r"image settled after (\d+) frames, aging (\d+)", txt)
                rec["settle"], rec["age"] = (int(m[1]), int(m[2])) if m else (None, None)
                rec["flat"] = "settled image is FLAT" in txt
            if f"{sub}/{stem}" in vram:
                rec["vram_mib"], rec["gtt_mib"] = vram[f"{sub}/{stem}"]
            shots[f"{sub}/{stem}"] = rec
    return shots


def fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v))


def markdown(directory: pathlib.Path, merged: dict, perf: dict) -> str:
    md = ["# Phase 1 baseline", "", f"Captures: `{directory}`", ""]
    for scene in sorted(merged):
        md += [f"## {scene}", "", "| row | metric | value | gate | detail |",
               "|---|---|---|---|---|"]
        for label, r in merged[scene].items():
            extra = ", ".join(
                f"{k} {fmt(v) if not isinstance(v, (list, dict)) else v}"
                for k, v in r.items()
                if k not in wm.SHOWN | {"source"} and v is not None
            )
            why = r.get("error") or ("needs " + ", ".join(r["missing"]) if r.get("missing") else "")
            gate = wm.gate_text(r.get("gate") or {})
            verdict = {True: "pass", False: "FAIL", None: "-"}[r.get("pass")]
            if r.get("gated") is False and not r.get("gate"):
                verdict = "info"
            if r.get("defer") is not None and r.get("pass") is False:
                verdict = "DEFER"
            burst = " (burst)" if r.get("source") == "bursts" else ""
            md.append(f"| {label} | {r.get('metric')} | {fmt(r.get('value'))} | {gate} {verdict} | "
                      f"{extra}{why}{burst} |")
        md.append("")
    md += [
        "## INS-4 (RS 2, 1600x900; ms per frame, median of the last 8 aged seconds)",
        "",
        "| shot | gpu_ms | gpu_transp | gpu_water | seed | reduce | snapshot | cpu_ms | main_ms "
        "| VRAM MiB | live compiles |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, r in sorted(perf.items()):
        cells = [r.get(c) for c in ("gpu_ms", "gpu_transp", "gpu_water", "water_hiz_seed",
                                    "water_hiz_reduce", "water_snapshot", "cpu_ms", "main_ms",
                                    "vram_mib")]
        md.append(f"| {k} | " + " | ".join(map(fmt, cells)) + f" | {r.get('live_compiles')} |")
    return "\n".join(md) + "\n"


def table(directory: pathlib.Path, out: pathlib.Path, regions: pathlib.Path = wm.REGIONS) -> int:
    out.mkdir(parents=True, exist_ok=True)
    merged = rows(directory, regions)
    plain = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731 (numpy scalars)
    (out / "baseline.json").write_text(json.dumps(merged, indent=1, default=plain))
    perf = ins4(directory)
    (out / "ins4.json").write_text(json.dumps(perf, indent=1))
    (out / "table.md").write_text(markdown(directory, merged, perf))
    print(f"{len(merged)} scene lanes, {len(perf)} shots -> {out}")
    return 0


# ---- checks -------------------------------------------------------------------------------------


def selfcheck(a: pathlib.Path, b: pathlib.Path) -> int:
    """Every PNG in `a` against its namesake in `b`: byte-identical, or the pixels that differ
    (count, max code, MAE, bounding box, how many by more than 4 codes), the flake boxes' source.
    Exit 1 when any differs; a PNG with no namesake is counted, not failed."""
    same = diff = missing = 0
    for pa in sorted(a.glob("*.png")):
        pb = b / pa.name
        if not pb.exists():
            missing += 1
            continue
        if pa.read_bytes() == pb.read_bytes():
            same += 1
            continue
        x = np.asarray(Image.open(pa).convert("RGB"), dtype=np.int16)
        y = np.asarray(Image.open(pb).convert("RGB"), dtype=np.int16)
        d = np.abs(x - y).max(axis=-1)
        if not d.any():
            same += 1
            continue
        diff += 1
        ys, xs = np.nonzero(d)
        mae = np.abs(x - y).mean()
        print(f"{pa.name}: {len(ys)} px differ, max {d.max()} codes, MAE {mae:.4f}, "
              f"bbox x{xs.min()}-{xs.max()} y{ys.min()}-{ys.max()}, "
              f"> 4 codes {int((d > 4).sum())} px")
    print(f"identical {same}, differing {diff}, unmatched {missing}")
    return 1 if diff else 0


def changes(old: dict, new: dict) -> list[tuple[str, str, str]]:
    """(scene lane, label, what changed) for every row whose value, metric, basis, gate,
    informational standing or deferral differs between two tables, or that only one of them
    holds."""
    out = []
    for scene in sorted(set(old) | set(new)):
        o, n = old.get(scene, {}), new.get(scene, {})
        for label in list(o) + [k for k in n if k not in o]:
            a, b = o.get(label), n.get(label)
            if b is None:
                out.append((scene, label, f"removed ({a.get('metric')} {fmt(a.get('value'))})"))
                continue
            if a is None:
                out.append((scene, label, f"added ({b.get('metric')} {fmt(b.get('value'))})"))
                continue
            what = []
            if a.get("metric") != b.get("metric"):
                what.append(f"metric {a.get('metric')} -> {b.get('metric')}")
            if a.get("value") != b.get("value"):
                what.append(f"value {fmt(a.get('value'))} -> {fmt(b.get('value'))}")
            if a.get("basis") != b.get("basis"):
                what.append(f"basis {a.get('basis')} -> {b.get('basis')}")
            if (a.get("gate") or {}) != (b.get("gate") or {}):
                old_gate = wm.gate_text(a.get("gate") or {}) or "none"
                what.append(f"gate {old_gate} -> {wm.gate_text(b.get('gate') or {}) or 'none'}")
            if (a.get("gated") is False) != (b.get("gated") is False):
                what.append("informational" if b.get("gated") is False else "gated")
            if a.get("defer") != b.get("defer"):
                what.append(f"deferred ({b['defer']})" if b.get("defer") else "no longer deferred")
            if what:
                out.append((scene, label, ", ".join(what)))
    return out


def diff(old: pathlib.Path, new: pathlib.Path) -> int:
    found = changes(json.loads(old.read_text()), json.loads(new.read_text()))
    for scene, label, what in found:
        print(f"{scene} {label}: {what}")
    print(f"{len(found)} rows differ")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="run the baseline's sweeps")
    c.add_argument("dir", type=pathlib.Path)
    c.add_argument("what", nargs="?", default="all",
                   choices=["stills", "views", "bursts", "repeat", "all"])
    t = sub.add_parser("table", help="measure a capture run into a baseline table")
    t.add_argument("dir", type=pathlib.Path)
    t.add_argument("out", type=pathlib.Path)
    t.add_argument("--regions", type=pathlib.Path, default=wm.REGIONS)
    s = sub.add_parser("selfcheck", help="compare two runs' PNGs")
    s.add_argument("a", type=pathlib.Path)
    s.add_argument("b", type=pathlib.Path)
    d = sub.add_parser("diff", help="the rows that differ between two baseline.json files")
    d.add_argument("old", type=pathlib.Path)
    d.add_argument("new", type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.cmd == "capture":
        return capture(a.dir.resolve(), a.what)
    if a.cmd == "table":
        return table(a.dir, a.out, a.regions)
    if a.cmd == "selfcheck":
        return selfcheck(a.a, a.b)
    return diff(a.old, a.new)


if __name__ == "__main__":
    sys.exit(main())
