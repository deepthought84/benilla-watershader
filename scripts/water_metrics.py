#!/usr/bin/env python3
"""Water image metrics: the numbers the water gates are judged on, measured on captures.

Every row reads the displayed frame (gamma-encoded 8-bit, as the player sees it) inside pixel boxes
that a per-scene region file names. Luma is Rec. 709 over the gamma codes, 0-255. Rows held to a
linear law, and views that hold enc(x), decode the codes first, (code/255)^2.2 (the water's display
gamma), and take the Rec. 709 luminance of that. Rates are taken only on bursts
(`scripts/water-shots.sh <dir> <n>`), whose shots stand on a held clock; a single still carries no
time.

    scripts/water_metrics.py <dir> [<scene> ...]           # every scene with a region file
    scripts/water_metrics.py <dir> --json out.json         # also write the numbers
    scripts/water_metrics.py <dir> --baseline base.json    # gate each row against a baseline run
    scripts/water_metrics.py <dir> --lane ref              # the Reference twins, <scene>-ref
    scripts/water_metrics.py --rows                        # the row catalogue
    python3 -m unittest discover -s scripts/tests          # the rows' synthetic fixtures

Files a scene reads from <dir> (only the ones its rows need):
    <scene>.png                 the Improved still (else the first burst shot)
    <scene>-000.png, -001 ...   a burst; <scene>.log gives each shot's clock ("burst shot N at T s")
    <scene>-ref.png             the Reference twin (and <scene>-ref-000.png ... its burst)
    <scene>@<view>.png          a WATER_VIEW debug capture (and <scene>@<view>-000.png ...)
    <scene>~<variant>.png       an A/B variant, e.g. ~noreflect for mirrors off
    <scene>.poses.csv           a moving burst's camera per shot: shot,ex,ey,ez,lx,ly,lz (WoW yd)

Region file, scripts/water_regions/<scene>.toml: pixel boxes only, never install data.

    scene = "wg-loch-fp-noon"
    [box]                        # [x0, y0, x1, y1], half-open, capture pixels (1600x900)
    crown = [540, 290, 600, 340]
    [[row]]
    metric = "refl_contrast"     # a row of the catalogue
    label = "crown"              # optional, tells two rows of one metric apart
    object = "crown"             # a string names a [box]; a list is an inline box
    ...                          # the row's other parameters; max/min/near override its gate

A row passes against an absolute gate (the catalogue's, or the region row's `max`, `min`, or
`near = [target, tolerance]`) and, with --baseline, against the same row of the baseline run in the
row's better direction (captures are hermetic, so no tolerance unless the row names one: `slack`
in value units, `factor` as a fraction of the baseline).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import sys
import tomllib

import numpy as np
from PIL import Image

REGIONS = pathlib.Path(__file__).resolve().parent / "water_regions"
LUMA_709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
# The world camera's vertical field of view (`view::CAM_FOVY`, π/4).
CAM_FOVY_DEG = 45.0
# The water's display gamma: its enc/dec are a pure 2.2 power law.
WATER_GAMMA = 2.2
# terrainp_s.bls darkens terrain in MCSH shadow by 0.7 in gamma: 0.7^2.2 = 0.456 linear.
SHADE_LAW = 0.456


# ---- images -------------------------------------------------------------------------------------


def load_rgb(path: pathlib.Path | str) -> np.ndarray:
    """An 8-bit capture as float32 RGB codes, HxWx3."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def luma(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 luma of gamma codes: the displayed value, 0-255."""
    return rgb[..., :3] @ LUMA_709


def dec(rgb: np.ndarray) -> np.ndarray:
    """Gamma codes, 0-255, to linear 0-1 through the water's display gamma."""
    return (np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 255.0) / 255.0) ** WATER_GAMMA


def lin_y(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance of decoded codes, linear 0-1."""
    return dec(rgb[..., :3]) @ LUMA_709.astype(np.float64)


def crop(a: np.ndarray, b) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in b)
    h, w = a.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"box {list(b)} is outside the {w}x{h} image")
    return a[y0:y1, x0:x1]


def box_mean(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)² window, edges clamped."""
    if r <= 0:
        return a.astype(np.float64)
    p = np.pad(a.astype(np.float64), r, mode="edge")
    c = np.cumsum(np.cumsum(p, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def median3(a: np.ndarray) -> np.ndarray:
    """3x3 median, edges clamped."""
    p = np.pad(a, 1, mode="edge")
    h, w = a.shape
    stack = [p[dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3)]
    return np.median(np.stack(stack), axis=0)


def smooth1(v: np.ndarray, k: int) -> np.ndarray:
    """Moving mean of odd width k along a 1-D profile, edges clamped."""
    if k <= 1:
        return v.astype(np.float64)
    r = k // 2
    p = np.pad(v.astype(np.float64), r, mode="edge")
    c = np.concatenate([[0.0], np.cumsum(p)])
    return (c[k:] - c[:-k]) / k


def hue_deg(rgb: np.ndarray) -> np.ndarray:
    """HSV hue in degrees, 0-360 (0 where the colour is grey)."""
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = np.where(mx - mn > 0, mx - mn, 1.0)
    h = np.where(mx == r, (g - b) / d % 6.0,
                 np.where(mx == g, (b - r) / d + 2.0, (r - g) / d + 4.0))
    return np.where(mx - mn > 0, h * 60.0, 0.0)


def hue_dist(a, b):
    """Absolute circular distance between hues, degrees, 0-180."""
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) % 360.0
    return np.minimum(d, 360.0 - d)


def lab(rgb: np.ndarray) -> np.ndarray:
    """CIE L*a*b* (D65) of sRGB codes, for ΔE76."""
    c = rgb.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], -1)


def michelson(a: float, b: float) -> float:
    return abs(a - b) / (a + b) if a + b > 0 else 0.0


def chroma(rgb) -> float:
    """(max − min) / max of one colour."""
    c = np.asarray(rgb, dtype=np.float64)
    return float((c.max() - c.min()) / c.max()) if c.max() > 0 else 0.0


# ---- the rows: pure functions over arrays -------------------------------------------------------
# Each returns a dict with "value" and its supporting numbers. `L` arguments are luma arrays.


def refl_contrast(img, object, sky, object_refl, sky_refl):
    """Michelson contrast of the reflected object against the reflected sky, over the same of the
    direct object against the direct sky (box means of luma). The phase gate: never above
    baseline."""
    L = luma(img)
    md = michelson(crop(L, object).mean(), crop(L, sky).mean())
    mr = michelson(crop(L, object_refl).mean(), crop(L, sky_refl).mean())
    return {"value": mr / md if md > 0 else math.inf, "direct": md, "reflected": mr}


def ridge_contrast(img, object, sky, object_refl, sky_refl):
    """refl_contrast's ratio on a far ridge: the reflected ridge keeps at least a share of the
    direct ridge's contrast against its sky."""
    return refl_contrast(img, object, sky, object_refl, sky_refl)


def horizon_ratio(img, water, sky):
    """Mean luma of the far water just under the horizon over the sky just above it."""
    L = luma(img)
    w, s = crop(L, water).mean(), crop(L, sky).mean()
    return {"value": w / s if s > 0 else math.inf, "water": w, "sky": s}


def zero_band(img, box, sd_max=0.5, step_max=2.0, min_rows=4):
    """The far-clip wall: the longest run of rows in the box (last water to the sky) that do not
    vary across the row (each channel's median absolute deviation from the row's median ≤ sd_max
    codes, so a mast or a rock in a few columns does not break it) and hold their colour from row
    to row (each channel's row median within step_max codes; the sky's gradient above moves).
    0 rows = no band; `mad` is the band's mean spread across its rows, codes."""
    C = crop(img, box)
    med = np.median(C, axis=1)
    mad = np.median(np.abs(C - med[:, None, :]), axis=1).max(axis=1)
    flat = mad <= sd_max
    steady = np.append(False, np.abs(np.diff(med, axis=0)).max(axis=1) <= step_max)
    best_len, best_at, run = 0, 0, 0
    for i in range(len(flat)):
        run = (run + 1 if steady[i] and run else 1) if flat[i] else 0
        if run > best_len:
            best_len, best_at = run, i + 1 - run
    if best_len < min_rows:
        return {"value": 0, "rows": None, "mad": None}
    y0 = int(box[1]) + best_at
    spread = float(mad[best_at : best_at + best_len].mean())
    return {"value": best_len, "rows": [y0, y0 + best_len], "mad": spread}


def far_band_std(L_frames, fog=0.0):
    """Per-pixel temporal std of luma over a burst, box mean, normalised by the fog's share
    (1 − fog factor): far water is bounded by it whatever the waves do."""
    sd = float(np.stack(L_frames).std(axis=0).mean())
    return {"value": sd / max(1.0 - fog, 1e-6), "raw": sd, "fog": fog}


def shimmer(L_frames):
    """Frame-to-frame high-frequency energy: rms of the change of each frame's 3x3 high-pass
    between consecutive shots, mean over the burst."""
    hp = [L - box_mean(L, 1) for L in L_frames]
    e = [float(np.sqrt(((b - a) ** 2).mean())) for a, b in zip(hp, hp[1:])]
    return {"value": float(np.mean(e)) if e else 0.0, "max": float(np.max(e)) if e else 0.0}


def pop(L_frames):
    """The largest frame-to-frame jump: box-mean |ΔL| per consecutive pair, the max, with its
    ratio to the burst's median change."""
    d = [float(np.abs(b - a).mean()) for a, b in zip(L_frames, L_frames[1:])]
    if not d:
        return {"value": 0.0, "median": 0.0, "at": None}
    med = float(np.median(d))
    i = int(np.argmax(d))
    return {"value": d[i], "median": med, "ratio": d[i] / med if med > 0 else math.inf, "at": i}


def lomb_scargle(t, x, freqs):
    """Normalised Lomb-Scargle power of x(t) at freqs (Hz)."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    var = x.var()
    out = np.zeros(len(freqs))
    if var <= 0:
        return out
    for k, f in enumerate(freqs):
        w = 2 * np.pi * f
        tau = np.arctan2(np.sin(2 * w * t).sum(), np.cos(2 * w * t).sum()) / (2 * w)
        c, s = np.cos(w * (t - tau)), np.sin(w * (t - tau))
        out[k] = ((x @ c) ** 2 / (c @ c) + (x @ s) ** 2 / (s @ s)) / (2 * var)
    return out


def pulse(L_frames, times, f_lo=0.3, f_hi=4.0):
    """The flow's breathing: per-shot detail energy (rms of the 9x9 high-pass) as a time series on
    the shots' clock, its Lomb-Scargle peak in Hz (1.25 Hz = the two-phase 1.6 s flow)."""
    e = [float(np.sqrt(((L - box_mean(L, 4)) ** 2).mean())) for L in L_frames]
    fr = np.linspace(f_lo, f_hi, 371)
    pw = lomb_scargle(times, e, fr)
    k = int(np.argmax(pw))
    mod = (max(e) - min(e)) / np.mean(e) if np.mean(e) else 0.0
    return {"value": float(fr[k]), "power": float(pw[k]), "modulation": mod}


def repeat_signature(img, box):
    """Vertical over horizontal high-frequency energy on reflected columns: the mean squared row
    step over the mean squared column step of the 9x9 high-pass of luma. A reflection that repeats
    a feature up a column scores high; flat water's own floor is the reference."""
    L = crop(luma(img), box)
    h = L - box_mean(L, 4)
    ey = float((np.diff(h, axis=0) ** 2).mean())
    ex = float((np.diff(h, axis=1) ** 2).mean())
    return {"value": ey / ex if ex > 0 else math.inf, "ey": ey, "ex": ex}


def trunk_wander(img, box, dark=True, smooth=5):
    """How far a reflected trunk wanders sideways: per row, the column of the darkest (or
    brightest) point of the 5-px smoothed row; the std of that path about its straight-line fit,
    and its largest row-to-row jump, in px."""
    L = crop(luma(img), box)
    xs = []
    for row in L:
        s = smooth1(row, smooth)
        xs.append(int(np.argmin(s) if dark else np.argmax(s)))
    xs = np.array(xs, dtype=np.float64)
    y = np.arange(len(xs), dtype=np.float64)
    fit = np.polyval(np.polyfit(y, xs, 1), y) if len(xs) > 1 else xs
    jump = float(np.abs(np.diff(xs)).max()) if len(xs) > 1 else 0.0
    return {"value": float((xs - fit).std()), "max_jump": jump}


def lum_vs_ref(img, ref, box):
    """Mean water luma on the Improved lane over its Reference twin at the same box."""
    a, b = crop(luma(img), box).mean(), crop(luma(ref), box).mean()
    return {"value": a / b if b > 0 else math.inf, "improved": a, "reference": b}


def luma_ratio(img, num, den):
    """Mean luma of one box over another (water over bank, under-deck water over the deck)."""
    L = luma(img)
    a, b = crop(L, num).mean(), crop(L, den).mean()
    return {"value": a / b if b > 0 else math.inf, "num": a, "den": b}


def water_bank_ratio(img, water, bank):
    """Mean water luma over the adjacent bank's."""
    return luma_ratio(img, water, bank)


def underdeck_ratio(img, water, deck):
    """Mean luma of the water under a deck over the deck's underside."""
    return luma_ratio(img, water, deck)


def seam_step(img, box, at, rows=3, axis="y"):
    """|mean luma of the `rows` lines before `at` − the `rows` lines from it| inside the box's
    span, across a horizontal seam (axis y, `at` a row) or a vertical one (axis x, a column)."""
    L = luma(img)
    x0, y0, x1, y1 = (int(v) for v in box)
    if axis == "y":
        a = L[at - rows : at, x0:x1].mean()
        b = L[at : at + rows, x0:x1].mean()
    else:
        a = L[y0:y1, at - rows : at].mean()
        b = L[y0:y1, at : at + rows].mean()
    return {"value": float(abs(a - b)), "before": float(a), "after": float(b)}


def tier_seam(img, box, at, rows=3, axis="y"):
    """The luma step across a reflection-tier ownership boundary (see seam_step)."""
    return seam_step(img, box, at, rows, axis)


def ring_seam(img, box, at, rows=3, axis="y"):
    """The luma step across the far ring's 0.9-far-clip depression row (see seam_step)."""
    return seam_step(img, box, at, rows, axis)


def glint(img, box, band=20, smooth=31, mirror=None):
    """The sun's (or moon's) glint on the water: per `band` rows, the row-mean luma profile
    smoothed over `smooth` px; the brightest band's peak (x, y, luma, rgb) and its horizontal σ,
    the second moment of the excess over the profile's median where it exceeds 10% of the
    peak's."""
    L, rgb = luma(img), img
    x0, y0, x1, y1 = (int(v) for v in box)
    best = None
    for yb in range(y0, y1, band):
        prof = smooth1(L[yb : min(yb + band, y1), x0:x1].mean(axis=0), smooth)
        k = int(np.argmax(prof))
        if best is None or prof[k] > best[2]:
            best = (k, yb, float(prof[k]), prof)
    k, yb, peak, prof = best
    ex = np.clip(prof - np.median(prof), 0, None)
    lo = k
    while lo > 0 and ex[lo - 1] > 0.1 * ex[k]:
        lo -= 1
    hi = k
    while hi < len(ex) - 1 and ex[hi + 1] > 0.1 * ex[k]:
        hi += 1
    w = ex[lo : hi + 1]
    xs = np.arange(lo, hi + 1)
    mu = (w * xs).sum() / w.sum() if w.sum() > 0 else k
    sigma = float(np.sqrt((w * (xs - mu) ** 2).sum() / w.sum())) if w.sum() > 0 else 0.0
    cx = x0 + k
    patch = rgb[yb : min(yb + band, y1), max(cx - 5, 0) : cx + 5].reshape(-1, 3).mean(0)
    rgb_peak = [round(float(v)) for v in patch]
    out = {"x": cx, "y": [yb, min(yb + band, y1)], "peak": peak, "rgb": rgb_peak, "sigma": sigma}
    if mirror is not None:
        out["offset"] = float(math.hypot(cx - mirror[0], (yb + band / 2) - mirror[1]))
    return out


def glint_peak(img, box, band=20, smooth=31, mirror=None):
    """The glint's peak luma (glint(): position, colour and offset from the predicted mirror
    point beside it)."""
    g = glint(img, box, band, smooth, mirror)
    return {"value": g["peak"], **{k: v for k, v in g.items() if k != "peak"}}


def glint_width(img, box, band=20, smooth=31, mirror=None):
    """The glint path's horizontal σ in px (glint())."""
    g = glint(img, box, band, smooth, mirror)
    return {"value": g["sigma"], "x": g["x"], "y": g["y"]}


def deep_mask(box, shape, transmittance=None, fresnel=None, fog=None):
    """The deep-water mask over a box, from the WATER_VIEW captures given (full frames, cropped
    here); a view not given masks nothing. The transmittance view holds enc(t_bed): deep where the
    decoded Y(t_bed) < 0.1. The Fresnel view holds F: F < 0.2 (its r). The fog view holds fog_f in
    r and the fog colour in g and b: fog_f > 0.8."""
    m = np.ones(shape, dtype=bool)
    if transmittance is not None:
        m &= lin_y(crop(transmittance, box)) < 0.1
    if fresnel is not None:
        m &= crop(fresnel, box)[..., 0] / 255.0 < 0.2
    if fog is not None:
        m &= crop(fog, box)[..., 0] / 255.0 > 0.8
    return m


def deep_hue(img, box, albedo=None, transmittance=None, fresnel=None, fog=None, ref_hue=None):
    """Hue of deep water. With the albedo view: the median of |hue(final) − hue(albedo)| over the
    deep mask, in degrees (pass ≤ 10). Without it: the hue of the box's mean colour, and its
    distance to `ref_hue` (the Light.dbc deep row's) when given."""
    fin = crop(img, box)
    m = deep_mask(box, fin.shape[:2], transmittance, fresnel, fog)
    if not m.any():
        return {"value": None, "pixels": 0}
    if albedo is not None:
        d = hue_dist(hue_deg(fin[m]), hue_deg(crop(albedo, box)[m]))
        return {"value": float(np.median(d)), "pixels": int(m.sum())}
    h = float(hue_deg(fin[m].mean(axis=0)))
    if ref_hue is None:
        return {"value": h, "gated": False}
    return {"value": float(hue_dist(h, ref_hue)), "hue": h, "ref_hue": ref_hue}


def deep_luma(img, box, transmittance=None, fog=None):
    """Mean luma over the deep mask without its F cut (the veil is what the player sees)."""
    fin = crop(img, box)
    m = deep_mask(box, fin.shape[:2], transmittance, None, fog)
    return {"value": float(luma(fin[m]).mean()) if m.any() else None}


def deep_chroma(img, box, transmittance=None, fog=None):
    """(max − min)/max of the deep mask's mean colour, without the F cut."""
    fin = crop(img, box)
    m = deep_mask(box, fin.shape[:2], transmittance, None, fog)
    if not m.any():
        return {"value": None}
    c = fin[m].mean(axis=0)
    return {"value": chroma(c), "rgb": [round(float(v), 1) for v in c]}


def water_mask(view, still, thr=3.0, erode=2):
    """Water pixels of a full frame: where a WATER_VIEW capture differs from the final still by
    more than `thr` codes in some channel (a view repaints water only), eroded by `erode` px so
    the pixels a silhouette's edge mixes with water drop out."""
    m = np.abs(view[..., :3] - still[..., :3]).max(axis=-1) > thr
    return box_mean(m, erode) > 1.0 - 1e-9 if erode > 0 else m


def ls_slopes(y: np.ndarray, w: int) -> np.ndarray:
    """Least-squares slope of y over every run of w consecutive samples."""
    k = np.arange(w) - (w - 1) / 2.0
    return np.convolve(y, (k / (k @ k))[::-1], mode="valid")


def band_step(view, box, still=None, window=None, hard=4.0):
    """Soft bands, on the albedo view, over water: the steepest least-squares slope of luma down
    a column over `window` rows, over the box's mean |∂Y/∂y| on water (its unbanded ramp). The
    window defaults to the rows that mean slope takes to climb 8 codes (4 to 128), so 8-bit steps
    average out; a column's run of water rows shorter than 2 windows + 3 rows is skipped. Water is
    where the view differs from `still` (the final frame; water_mask), else the whole box.
    Analytic 2.2 for 3 bands at push 0.5, edge 0.22, 1 unbanded. A hard band is ∞: a rise within
    2 rows of ≥ `hard` codes and over 4 times what the window above or below climbs in 2 rows,
    plus a code per end for rounding. Pass ≤ 3."""
    Y = crop(luma(view), box).astype(np.float64)
    m = crop(water_mask(view, still), box) if still is not None else np.ones(Y.shape, dtype=bool)
    both = m[1:] & m[:-1]
    d = np.abs(np.diff(Y, axis=0))
    mean = float(d[both].mean()) if both.any() else 0.0
    if mean <= 0:
        return {"value": None, "window": None, "runs": 0}
    w = int(window) if window else int(np.clip(math.ceil(8.0 / mean), 4, 128))
    worst, runs = 0.0, 0
    for c in range(Y.shape[1]):
        edges = np.flatnonzero(np.diff(np.concatenate([[0], m[:, c].astype(np.int8), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            y = Y[a:b, c]
            n = len(y)
            if n < 2 * w + 3:
                continue
            i = np.arange(w, n - w - 2)
            rise = np.abs(y[i + 2] - y[i])
            around = np.maximum(np.abs(y[i] - y[i - w]), np.abs(y[i + 2 + w] - y[i + 2])) / w
            jump = (rise >= hard) & (rise > 2.0 + 4.0 * 2.0 * around)
            if jump.any():
                k = int(i[np.argmax(jump)])
                at = int(box[1] + a + k + 1 + np.argmax(np.abs(np.diff(y[k : k + 3]))))
                return {"value": math.inf, "column": int(box[0]) + c, "row": at, "window": w}
            worst = max(worst, float(np.abs(ls_slopes(y, w)).max()))
            runs += 1
    return {"value": worst / mean if runs else None, "window": w, "runs": runs}


def shade_ratio(img, water_shade, water_lit, bank_shade, bank_lit, in_scatter=None):
    """Shadowed over lit water in linear luminance (box means; the in-scatter view, which holds
    enc(in_scatter), where given, else the final frame), beside the same ratio on the bank in the
    final frame. The terrain's shadow law is 0.456 linear: pass |water ratio − 0.456| ≤ 0.1."""
    src = in_scatter if in_scatter is not None else img
    w = lin_y(crop(src, water_shade)).mean() / max(lin_y(crop(src, water_lit)).mean(), 1e-9)
    b = lin_y(crop(img, bank_shade)).mean() / max(lin_y(crop(img, bank_lit)).mean(), 1e-9)
    return {"value": float(w), "bank": float(b)}


def refract_leak(refracted, bed, frame, mask, box, band=8, de=5.0):
    """Refraction pulling an occluder into the water: in the `band`-px strip of water along every
    non-water silhouette, the share of pixels whose refracted-bed view matches the nearest
    occluder's colour (ΔE76 < 5) while the unrefracted bed view does not. `mask` is the water
    view (non-zero on water). Pass ≤ 0.1%."""
    R, B, F = (crop(a, box) for a in (refracted, bed, frame))
    wm = crop(mask, box).max(axis=-1) > 0 if mask.ndim == 3 else crop(mask, box) > 0
    occ = F.copy()
    known = ~wm
    strip = np.zeros_like(wm)
    # Grow the occluders into the water one pixel a pass, carrying the nearest occluder's colour.
    for _ in range(band):
        grown = known.copy()
        for dst, src in (
            ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
            ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
            ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
            ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ):
            take = np.zeros_like(wm)
            take[dst] = known[src] & ~grown[dst]
            moved = np.zeros_like(occ)
            moved[dst] = occ[src]
            occ[take] = moved[take]
            grown |= take
        strip |= grown & wm
        known = grown
    if not strip.any():
        return {"value": 0.0, "strip": 0}
    lo = lab(occ[strip])
    near_r = np.linalg.norm(lab(R[strip]) - lo, axis=-1) < de
    near_b = np.linalg.norm(lab(B[strip]) - lo, axis=-1) < de
    leak = near_r & ~near_b
    return {"value": float(leak.mean()), "strip": int(strip.sum())}


def roughness_profile(view, box, channel=0):
    """A WATER_VIEW channel's row-mean profile from near (bottom) to far (top): its largest
    adjacent step in codes, and the share of steps that follow the profile's overall direction."""
    c = crop(view, box)[..., channel].astype(np.float64)
    p = c.mean(axis=1)[::-1]
    d = np.diff(p)
    sign = np.sign(p[-1] - p[0]) or 1.0
    moving = np.abs(d) > 0.5
    mono = float((np.sign(d[moving]) == sign).mean()) if moving.any() else 1.0
    return {"value": float(np.abs(d).max()) if len(d) else 0.0, "monotone": mono}


def floor_ncc(img, other, box, r=8):
    """Normalised cross-correlation of the water's high-passed luma (minus a 17x17 mean) in two
    captures of one scene, e.g. floor-served (mirrors off) against mirror-served."""
    a = crop(luma(img), box)
    b = crop(luma(other), box)
    ha = a - box_mean(a, r)
    hb = b - box_mean(b, r)
    ha -= ha.mean()
    hb -= hb.mean()
    den = math.sqrt(float((ha**2).sum() * (hb**2).sum()))
    return {"value": float((ha * hb).sum() / den) if den > 0 else 0.0}


def stair_step(img, box, rise=3, tread=4):
    """Stair-steps along a bank: per column, the row of the strongest vertical luma edge in the box
    (the waterline, rows smoothed over 3); a step is a riser, consecutive columns moving the edge
    by more than 1 px, rising ≥ `rise` px in all after a tread of ≥ `tread` columns within 1 px.
    The count over the box."""
    L = crop(luma(img), box).astype(np.float64)
    Ls = (L[:-2] + L[1:-1] + L[2:]) / 3.0
    edge = np.abs(np.diff(Ls, axis=0)).argmax(axis=0).astype(np.int64)
    steps, flat, climb = 0, 0, 0
    for v in np.append(np.diff(edge), 0):
        if abs(v) <= 1:
            if abs(climb) >= rise and flat >= tread:
                steps += 1
            if climb:
                flat = 0
            climb = 0
            flat += 1
        else:
            climb += v
    return {"value": steps}


def shore_band_ratio(img, band, bank):
    """The shore band's brightest point (3x3 mean of luma) over the bank's mean luma."""
    L = luma(img)
    peak = float(box_mean(crop(L, band), 1).max())
    b = float(crop(L, bank).mean())
    return {"value": peak / b if b > 0 else math.inf, "peak": peak, "bank": b}


def band_width(img, box, min_contrast=8.0):
    """How wide the shore band shows: per column, the run of rows around the brightest one that
    stays above half its height over the column's median; the median over the columns where the
    band stands ≥ min_contrast codes out, in px."""
    L = crop(luma(img), box)
    widths = []
    for col in L.T:
        base, k = float(np.median(col)), int(np.argmax(col))
        h = float(col[k]) - base
        if h < min_contrast:
            continue
        half = base + 0.5 * h
        lo = k
        while lo > 0 and col[lo - 1] >= half:
            lo -= 1
        hi = k
        while hi < len(col) - 1 and col[hi + 1] >= half:
            hi += 1
        widths.append(hi - lo + 1)
    return {"value": float(np.median(widths)) if widths else 0.0, "columns": len(widths)}


def autocorr_peak(img, box, lag_min=2, lag_max=200):
    """Repetition along a bank: the column profile of the box's per-column max luma, less its 31-px
    moving mean; the highest normalised autocorrelation at lags [lag_min, lag_max] px past the
    central lobe."""
    s = crop(luma(img), box).max(axis=0).astype(np.float64)
    s = s - smooth1(s, 31)
    s -= s.mean()
    v = float((s**2).sum())
    if v <= 0:
        return {"value": 0.0, "lag": None}
    n = len(s)
    r = np.array([float((s[: n - k] * s[k:]).sum()) / v for k in range(min(lag_max, n - 1) + 1)])
    zero = int(np.argmax(r <= 0)) if (r <= 0).any() else len(r)
    lo = max(lag_min, zero)
    if lo >= len(r):
        return {"value": 0.0, "lag": None}
    k = lo + int(np.argmax(r[lo:]))
    return {"value": float(r[k]), "lag": k}


def straight_line(img, box, thr=12.0, dark=False):
    """A straight line in the water: pixels standing more than `thr` codes above (or below) their
    7x7 mean vote in a Hough transform (1°, 2 px bins); the strongest line's votes, i.e. its length
    in px."""
    L = crop(luma(img), box).astype(np.float64)
    ridge = (L - box_mean(L, 3)) if not dark else (box_mean(L, 3) - L)
    ys, xs = np.nonzero(ridge > thr)
    if len(xs) == 0:
        return {"value": 0, "theta": None, "rho": None}
    th = np.deg2rad(np.arange(180))
    diag = int(math.ceil(math.hypot(*L.shape))) // 2 + 1
    rho = np.outer(xs, np.cos(th)) + np.outer(ys, np.sin(th))
    rho = np.floor(rho / 2.0).astype(np.int64) + diag
    acc = np.zeros((180, 2 * diag + 1), dtype=np.int64)
    np.add.at(acc, (np.broadcast_to(np.arange(180), rho.shape), rho), 1)
    t, rr = np.unravel_index(int(acc.argmax()), acc.shape)
    return {"value": int(acc[t, rr]), "theta": int(t), "rho": int(2 * (rr - diag))}


def contact_cov(view, box, channel=0, thr=0.2):
    """Share of the box's columns whose contact-foam coverage (a WATER_VIEW channel, 0-1) reaches
    `thr` somewhere, in %."""
    c = crop(view, box)[..., channel] / 255.0
    return {"value": float((c.max(axis=0) >= thr).mean() * 100.0)}


def foam_fp(view, box, channel=0, thr=0.2):
    """Foam where there is no contact: share of the open-water box's pixels whose coverage reaches
    `thr`, in %."""
    c = crop(view, box)[..., channel] / 255.0
    return {"value": float((c >= thr).mean() * 100.0)}


def horizon_step(img, box, water_y, sky_y):
    """|luma of the last water row − luma of the first sky row| over the box's columns."""
    L = luma(img)
    x0, x1 = int(box[0]), int(box[2])
    a, b = float(L[water_y, x0:x1].mean()), float(L[sky_y, x0:x1].mean())
    return {"value": abs(a - b), "water": a, "sky": b}


def luma_sd(img, box):
    """Standard deviation of luma in the box (in-water structure), codes."""
    return {"value": float(crop(luma(img), box).std())}


def luma_p5(img, box):
    """5th percentile of luma in the box (how dark the darkest water gets), codes."""
    return {"value": float(np.percentile(crop(luma(img), box), 5))}


def view_std(frames, channel=None):
    """Per-pixel temporal std over a static burst, in 8-bit codes, box mean (and p99): of a
    WATER_VIEW channel, or of luma when `channel` is None. A still scene scores 0."""
    a = np.stack([luma(f) if channel is None else f[..., channel] for f in frames])
    a = a.astype(np.float64)
    sd = a.std(axis=0)
    return {"value": float(sd.mean()), "p99": float(np.percentile(sd, 99))}


def project(point, eye, look, size, fovy_deg=CAM_FOVY_DEG):
    """Pixel of a WoW-coords point (x north, y west, z up) seen by a camera at `eye` looking at
    `look`, vertical fov `fovy_deg`, image `size` (w, h). None behind the eye."""
    p, e = np.asarray(point, dtype=np.float64), np.asarray(eye, dtype=np.float64)
    f = np.asarray(look, dtype=np.float64) - e
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    v = p - e
    z = v @ f
    if z <= 1e-6:
        return None
    t = math.tan(math.radians(fovy_deg) / 2)
    w, h = size
    nx = (v @ r) / (z * t * (w / h))
    ny = (v @ u) / (z * t)
    return ((nx + 1) / 2 * w, (1 - ny) / 2 * h)


def pan_luma_std(frames, poses, anchor, half=(8, 8), fovy_deg=CAM_FOVY_DEG):
    """A world-anchored box's luma through a moving burst: the box (`half` px each way) around
    `anchor` (WoW yd) reprojected per shot from its camera pose; the std of its mean luma over the
    shots where it lies in frame."""
    means = []
    for f, (eye, look) in zip(frames, poses):
        h, w = f.shape[:2]
        c = project(anchor, eye, look, (w, h), fovy_deg)
        if c is None:
            continue
        x, y = int(round(c[0])), int(round(c[1]))
        b = [x - half[0], y - half[1], x + half[0] + 1, y + half[1] + 1]
        if b[0] < 0 or b[1] < 0 or b[2] > w or b[3] > h:
            continue
        means.append(float(crop(luma(f), b).mean()))
    return {"value": float(np.std(means)) if means else None, "shots": len(means)}


def halo_count(img, box, thr=12.0):
    """Water pixels differing from their 3x3 median luma by more than `thr` codes."""
    L = crop(luma(img), box)
    return {"value": int((np.abs(L - median3(L)) > thr).sum())}


# ---- the catalogue ------------------------------------------------------------------------------
# better: which way a row improves ("lower", "higher", None = informational), used against a
# baseline; slack/factor its default tolerance there. gate: the absolute pass the specs give,
# applied unless a region row overrides it.
#
# needs: what each argument is read from, as (kind, argument, default name, required):
#   ("still", arg[, required])     the lane's still; optional (False) is skipped when absent or
#                                  when the region row sets `arg = false`
#   ("ref", "ref")                 the Reference twin
#   ("burst", mode)                the lane's burst, or a view's (row parameter `view`), cropped
#                                  to `box` as luma ("luma") or RGB ("rgb"), or whole ("full")
#   ("times",) / ("poses",)        the burst's clock (the log, else `every` steps of 1/60 s) / poses
#   ("view", arg, view, required)  a WATER_VIEW capture; a row parameter `arg` names another view
#   ("variant", arg, variant)      an A/B capture <scene>~<variant>.png


class Row:
    def __init__(self, fn, needs, better, gate=None, slack=0.0, factor=None):
        self.fn, self.needs, self.better, self.gate = fn, needs, better, gate
        self.slack, self.factor = slack, factor


STILL = [("still", "img")]
DEEP_VIEWS = [("view", "transmittance", "transmittance", False), ("view", "fog", "fog", False)]
ROWS: dict[str, Row] = {
    "refl_contrast": Row(refl_contrast, STILL, "lower"),
    "ridge_contrast": Row(ridge_contrast, STILL, "higher", {"min": 0.4}),
    "horizon_ratio": Row(horizon_ratio, STILL, "higher"),
    "zero_band": Row(zero_band, STILL, "lower"),
    "far_band_std": Row(far_band_std, [("burst", "luma")], "higher"),
    "shimmer": Row(shimmer, [("burst", "luma")], "lower"),
    "pulse": Row(pulse, [("burst", "luma"), ("times",)], None),
    "repeat_signature": Row(repeat_signature, STILL, "lower"),
    "trunk_wander": Row(trunk_wander, STILL, "lower"),
    "lum_vs_ref": Row(lum_vs_ref, STILL + [("ref", "ref")], "higher", {"min": 1.0}),
    "water_bank_ratio": Row(water_bank_ratio, STILL, "lower"),
    "underdeck_ratio": Row(underdeck_ratio, STILL, "lower"),
    "tier_seam": Row(tier_seam, STILL, "lower"),
    "pop": Row(pop, [("burst", "luma")], "lower"),
    "glint_peak": Row(glint_peak, STILL, None),
    "glint_width": Row(glint_width, STILL, None),
    "deep_hue": Row(
        deep_hue,
        STILL
        + [("view", "albedo", "albedo", False), ("view", "fresnel", "fresnel", False)]
        + DEEP_VIEWS,
        "lower",
        {"max": 10.0},
    ),
    "deep_luma": Row(deep_luma, STILL + DEEP_VIEWS, "higher", slack=3.0),
    "deep_chroma": Row(deep_chroma, STILL + DEEP_VIEWS, "higher", factor=0.8),
    "band_step": Row(
        band_step,
        [("view", "view", "albedo", True), ("still", "still", False)],
        "lower",
        {"max": 3.0},
    ),
    "shade_ratio": Row(
        shade_ratio,
        STILL + [("view", "in_scatter", "in_scatter", False)],
        None,
        {"near": [SHADE_LAW, 0.1]},
    ),
    "refract_leak": Row(
        refract_leak,
        [
            ("view", "refracted", "refracted", True),
            ("view", "bed", "bed", True),
            ("still", "frame"),
            ("view", "mask", "water", True),
        ],
        "lower",
        {"max": 0.001},
    ),
    "roughness_profile": Row(roughness_profile, [("view", "view", "roughness", True)], "lower"),
    "floor_ncc": Row(
        floor_ncc, STILL + [("variant", "other", "noreflect")], "higher", {"min": 0.7}
    ),
    "stair_step": Row(stair_step, STILL, "lower"),
    "shore_band_ratio": Row(shore_band_ratio, STILL, "lower", {"max": 1.1}),
    "band_width": Row(band_width, STILL, "lower", {"max": 4.0}),
    "autocorr_peak": Row(autocorr_peak, STILL, "lower", {"max": 0.2}),
    "straight_line": Row(straight_line, STILL, "lower", {"max": 150}),
    "contact_cov": Row(contact_cov, [("view", "view", "contact", True)], "higher", {"min": 60.0}),
    "foam_fp": Row(foam_fp, [("view", "view", "contact", True)], "lower", {"max": 0.5}),
    "ring_seam": Row(ring_seam, STILL, "lower", {"max": 2.0}),
    "horizon_step": Row(horizon_step, STILL, "lower", {"max": 3.0}),
    "luma_sd": Row(luma_sd, STILL, "higher"),
    "luma_p5": Row(luma_p5, STILL, None),
    "view_std": Row(view_std, [("burst", "rgb")], "lower"),
    "pan_luma_std": Row(pan_luma_std, [("burst", "full"), ("poses",)], "lower"),
    "halo_count": Row(halo_count, STILL, "lower"),
}


def judge(value, gate: dict | None) -> bool | None:
    """An absolute gate: {"max": v}, {"min": v} or {"near": [target, tol]}; None without one."""
    if not gate or value is None:
        return None
    if "max" in gate and not value <= gate["max"]:
        return False
    if "min" in gate and not value >= gate["min"]:
        return False
    if "near" in gate and not abs(value - gate["near"][0]) <= gate["near"][1]:
        return False
    return True


def against(value, base, row: Row, slack=None, factor=None) -> bool | None:
    """The row against its baseline value in its better direction: no tolerance unless `slack`
    (absolute) or `factor` (share of the baseline) says otherwise."""
    if value is None or base is None or row.better is None:
        return None
    slack = row.slack if slack is None else slack
    factor = row.factor if factor is None else factor
    if row.better == "lower":
        return value <= (base / factor if factor else base + slack)
    return value >= (base * factor if factor else base - slack)


# ---- scenes: files, region files, running the rows ----------------------------------------------

TIME_RE = re.compile(r"(?:burst shot \d+|shot 0) at ([0-9.]+) s, writing (\S+\.png)")


class Scene:
    """One scene's captures in a directory, found by the naming the module doc lists."""

    def __init__(self, directory: pathlib.Path, name: str):
        self.dir, self.name = directory, name
        self._cache: dict[str, np.ndarray | None] = {}

    def path(self, suffix: str) -> pathlib.Path:
        return self.dir / f"{self.name}{suffix}"

    def image(self, suffix: str = "") -> np.ndarray | None:
        """<scene><suffix>.png, else the first shot of its burst, else None."""
        if suffix not in self._cache:
            p = self.path(f"{suffix}.png")
            shots = [p] if p.exists() else self.burst_paths(suffix)
            self._cache[suffix] = load_rgb(shots[0]) if shots else None
        return self._cache[suffix]

    def burst_paths(self, suffix: str = "") -> list[pathlib.Path]:
        stem = f"{self.name}{suffix}"
        shot = re.compile(re.escape(stem) + r"-\d{3}\.png")
        return sorted(p for p in self.dir.glob("*.png") if shot.fullmatch(p.name))

    def burst_times(self, suffix: str = "") -> list[float] | None:
        """Each burst shot's clock from the capture log; None without a log that names them all."""
        log = self.path(f"{suffix}.log")
        if not log.exists():
            return None
        by_name = {}
        for line in log.read_text(errors="replace").splitlines():
            m = TIME_RE.search(line)
            if m:
                by_name[pathlib.Path(m.group(2)).name] = float(m.group(1))
        t = [by_name.get(p.name) for p in self.burst_paths(suffix)]
        return None if not t or any(v is None for v in t) else t

    def poses(self, suffix: str = "") -> list | None:
        """<scene>.poses.csv: shot,ex,ey,ez,lx,ly,lz, the camera eye and look-at per shot."""
        p = self.path(f"{suffix}.poses.csv")
        if not p.exists():
            return None
        rows = sorted(csv.DictReader(p.open()), key=lambda r: int(r["shot"]))
        return [
            ([float(r[k]) for k in ("ex", "ey", "ez")], [float(r[k]) for k in ("lx", "ly", "lz")])
            for r in rows
        ]


# Row parameters that name a box; a string is looked up in the region file's [box] table.
BOX_PARAMS = {"box", "object", "sky", "object_refl", "sky_refl", "water", "bank", "deck", "band",
              "water_shade", "water_lit", "bank_shade", "bank_lit", "num", "den"}
# Region-row keys that are not the function's parameters.
META = {"metric", "label", "max", "min", "near", "slack", "factor", "lane", "gate", "every"}


def run_row(scene: Scene, spec: dict, boxes: dict, lane: str) -> dict:
    """Measure one region-file row on a scene; `lane` is "" (Improved) or "-ref" (its row may
    say `lane = "ref"` to read the twin on an Improved run)."""
    name = spec["metric"]
    row = ROWS[name]
    base = "-ref" if spec.get("lane") == "ref" else lane
    params = {k: v for k, v in spec.items() if k not in META}
    for k in params.keys() & BOX_PARAMS:
        if isinstance(params[k], str):
            if params[k] not in boxes:
                raise KeyError(f"no box named {params[k]!r}")
            params[k] = boxes[params[k]]
    box = params.get("box")
    args: dict = {}
    missing: list[str] = []
    for need in row.needs:
        kind = need[0]
        if kind == "still":
            arg, required = need[1], need[2] if len(need) > 2 else True
            if not required and params.pop(arg, None) is False:
                continue
            img = scene.image(base)
            if img is not None:
                args[arg] = img
            elif required:
                missing.append(f"{scene.name}{base}.png")
        elif kind == "ref":
            args["ref"] = scene.image("-ref")
            if args["ref"] is None:
                missing.append(f"{scene.name}-ref.png")
        elif kind == "burst":
            view = params.pop("view", None)
            sfx = f"{base}@{view}" if view else base
            shots = scene.burst_paths(sfx)
            if len(shots) < 2:
                missing.append(f"{scene.name}{sfx}-NNN.png (a burst)")
                continue
            if need[1] == "full":
                params.pop("box", None)
                args["frames"] = [load_rgb(p) for p in shots]
            else:
                params.pop("box", None)
                frames = [crop(load_rgb(p), box) for p in shots]
                if need[1] == "luma":
                    args["L_frames"] = [luma(f) for f in frames]
                else:
                    args["frames"] = frames
        elif kind == "times":
            t = scene.burst_times(base)
            if t is None:
                every = float(spec.get("every", 2))
                t = [i * every / 60.0 for i in range(len(scene.burst_paths(base)))]
            args["times"] = t
        elif kind == "poses":
            args["poses"] = scene.poses(base)
            if args["poses"] is None:
                missing.append(f"{scene.name}{base}.poses.csv")
        elif kind == "view":
            _, arg, default, required = need
            view = params.pop(arg, default)
            if view is False:
                continue
            img = scene.image(f"{base}@{view}")
            if img is not None:
                args[arg] = img
            elif required:
                missing.append(f"{scene.name}{base}@{view}.png")
        elif kind == "variant":
            _, arg, default = need
            var = params.pop(arg, default)
            args[arg] = scene.image(f"{base}~{var}")
            if args[arg] is None:
                missing.append(f"{scene.name}{base}~{var}.png")
    if missing:
        return {"value": None, "missing": missing}
    return row.fn(**args, **params)


def load_regions(path: pathlib.Path) -> dict:
    with path.open("rb") as f:
        r = tomllib.load(f)
    r.setdefault("box", {})
    r.setdefault("row", [])
    for spec in r["row"]:
        if spec.get("metric") not in ROWS:
            raise ValueError(f"{path}: unknown metric {spec.get('metric')!r}")
    return r


def measure(directory: pathlib.Path, regions_dir: pathlib.Path, scenes: list[str],
            lane: str = "") -> dict:
    """{scene: {row label: result}} for the scenes given, or every region file with captures."""
    names = scenes or sorted(p.stem for p in regions_dir.glob("*.toml"))
    results: dict = {}
    for name in names:
        rp = regions_dir / f"{name}.toml"
        if not rp.exists():
            print(f"{name}: no region file {rp}", file=sys.stderr)
            continue
        reg = load_regions(rp)
        scene = Scene(directory, reg.get("scene", name))
        if scene.image(lane) is None:
            if scenes:
                print(f"{name}: no capture {scene.name}{lane}.png in {directory}", file=sys.stderr)
            continue
        rows: dict = {}
        for spec in reg["row"]:
            label = spec.get("label", spec["metric"])
            n = 2
            while label in rows:
                label = f"{spec.get('label', spec['metric'])}#{n}"
                n += 1
            try:
                res = run_row(scene, spec, reg["box"], lane)
            except (KeyError, ValueError, TypeError) as e:
                res = {"value": None, "error": str(e)}
            gate = {k: spec[k] for k in ("max", "min", "near") if k in spec}
            if spec.get("gate") is False or res.get("gated") is False:
                gate = {}
            elif not gate:
                gate = ROWS[spec["metric"]].gate or {}
            res.update(metric=spec["metric"], gate=gate)
            res.update(slack=spec.get("slack"), factor=spec.get("factor"))
            res["pass"] = judge(res.get("value"), gate)
            rows[label] = res
        results[f"{scene.name}{lane}"] = rows
    return results


def fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (float, np.floating)):
        return "inf" if math.isinf(v) else f"{float(v):.4g}"
    return str(v)


SHOWN = {"value", "metric", "gate", "pass", "slack", "factor", "gated", "missing", "error"}


def report(results: dict, baseline: dict | None) -> int:
    """Print one line per row; the count of failed gates and rows worse than the baseline."""
    failures = 0
    for scene, rows in results.items():
        print(f"== {scene}")
        for label, r in rows.items():
            extra = ", ".join(
                f"{k} {fmt(v)}" for k, v in r.items() if k not in SHOWN and v is not None
            )
            verdict = {True: "PASS", False: "FAIL", None: "-"}[r["pass"]]
            gate = " ".join(f"{k} {fmt(v)}" for k, v in r["gate"].items())
            base = ""
            if baseline is not None:
                b = baseline.get(scene, {}).get(label, {}).get("value")
                ok = against(r.get("value"), b, ROWS[r["metric"]], r.get("slack"), r.get("factor"))
                base = f"  base {fmt(b)} {({True: 'ok', False: 'WORSE', None: '-'})[ok]}"
                failures += ok is False
            failures += r["pass"] is False
            why = r.get("error") or (f"needs {', '.join(r['missing'])}" if r.get("missing") else "")
            value = fmt(r.get("value"))
            print(f"  {label:22s} {value:>10s}  {verdict:4s} {gate:14s}{base}  {extra}{why}")
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", nargs="?", type=pathlib.Path, help="directory of captures")
    ap.add_argument("scenes", nargs="*",
                    help="scene names (default: every region file with captures)")
    ap.add_argument("--regions", type=pathlib.Path, default=REGIONS, help="region-file directory")
    ap.add_argument("--lane", choices=["improved", "ref"], default="improved")
    ap.add_argument("--json", type=pathlib.Path, help="write the results as JSON")
    ap.add_argument("--baseline", type=pathlib.Path,
                    help="a --json file of an earlier run to gate against")
    ap.add_argument("--rows", action="store_true", help="print the row catalogue and exit")
    a = ap.parse_args(argv)
    if a.rows:
        for name, row in ROWS.items():
            doc = " ".join((row.fn.__doc__ or "").split())
            gate = f"; gate {row.gate}" if row.gate else ""
            print(f"{name} (better {row.better}{gate})\n    {doc}")
        return 0
    if a.dir is None:
        ap.error("a capture directory is required")
    results = measure(a.dir, a.regions, a.scenes, "-ref" if a.lane == "ref" else "")
    base = json.loads(a.baseline.read_text()) if a.baseline else None
    failures = report(results, base)
    if a.json:
        plain = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731 (numpy scalars)
        a.json.write_text(json.dumps(results, indent=1, default=plain))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
