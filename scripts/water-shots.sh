#!/usr/bin/env bash
# Golden water scenes: capture the `water-*` scenarios (`capture/scenarios.rs`), one client process
# per shot, like visual.sh. Opens a real window. Captures hold Blizzard-derived imagery, so they go
# to a directory outside the repo or under target/; the script refuses any other place in the repo.
#
#   scripts/water-shots.sh [<dir>]                # <dir>/<scene>.png, Improved Water, and its
#                                                 # Reference twin <dir>/<scene>-ref.png;
#                                                 # <dir> target/visual/water
#   scripts/water-shots.sh <dir> <n> [<every>]    # bursts, Improved only: <dir>/<scene>-000.png …
#                                                 # n shots, every <every> fixed 1/60 s steps
#                                                 # (default 2), and <scene>.poses.csv
#
# $WATER_SCENES="<scene> …" captures those scenes instead of every one. $WATER_VIEW=<view> captures
# that Improved Water debug view instead (`dev_state::WATER_VIEWS`), <scene>@<view>.png, Improved
# only and with the FFX glow off, so a pixel holds what the view wrote. $WOW_CAPTURE_PAN, and any
# other capture knob in the environment, reaches every client.
#
# The render state is pinned: 1600x900, render scale 2, far clip 777. Each scene carries its own
# settle (300 still frames) and age (1000 steps) (`capture`'s WATER_STABLE_FRAMES,
# WATER_AGE_FRAMES). Every shot also writes the perf journal of its run, <shot>.fps.csv (the GPU
# columns per pass group, `gpu_water` the water's own passes), and its log carries the per-pass
# GPU_PASSES lines.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd -P)"

dir="${1:-$root/target/visual/water}"
burst="${2:-0}"
every="${3:-2}"
# Resolved against the caller's directory, before the script moves to the repo root.
fresh=0
[ -d "$dir" ] || fresh=1
mkdir -p "$dir"
dir="$(cd "$dir" && pwd -P)"
case "$dir/" in
  "$root/target/"*) ;;
  "$root/"*)
    [ "$fresh" = 0 ] || rmdir "$dir"
    echo "water-shots.sh: $dir is in the repo outside target/; captures stay out of git" >&2
    exit 2 ;;
esac
cd "$root"

cargo build -q --release -p benilla
bin=./target/release/benilla

# The binary owns the scene list.
all=$(WOW_CAPTURE=list WOW_CAPTURE_LIST=water "$bin")
[ -n "$all" ] || { echo "the binary lists no water scenes" >&2; exit 1; }
scenes="${WATER_SCENES:-$all}"
for s in $scenes; do
  grep -qx -- "$s" <<< "$all" || { echo "$s is not a golden water scene" >&2; exit 1; }
done

# shot <scene> <stem> <style> [ENV=val …]: one client; the journal starts fresh, since it appends
# across runs.
shot() {
  rm -f "$dir/$2.fps.csv"
  env "${@:4}" WOW_CAPTURE="$1" WOW_CAPTURE_OUT="$dir/$2.png" \
    WOW_CAPTURE_BURST="$burst" WOW_CAPTURE_BURST_EVERY="$every" \
    WOW_WATER_STYLE="$3" WOW_RENDER_SCALE=2 WOW_FARCLIP=777 WOW_WIN=1600x900 \
    WOW_FPS_JOURNAL="$dir/$2.fps.csv" WOW_GPU_PASSES=1 \
    "$bin" > "$dir/$2.log" 2>&1 || { echo "capture $2 FAILED, log: $dir/$2.log" >&2; exit 1; }
}

for s in $scenes; do
  if [ -n "${WATER_VIEW:-}" ]; then
    stem="$s@$WATER_VIEW"
    [ "$burst" -gt 0 ] && echo "capture $s view $WATER_VIEW -> $dir/$stem-000.png … ($burst)" \
      || echo "capture $s view $WATER_VIEW -> $dir/$stem.png"
    shot "$s" "$stem" 1 WOW_WATER_VIEW="$WATER_VIEW" WOW_NO_FFX=1
  elif [ "$burst" -gt 0 ]; then
    echo "capture $s -> $dir/$s-000.png … $s-$(printf %03d $((burst - 1))).png (every $every)"
    shot "$s" "$s" 1
  else
    echo "capture $s -> $dir/$s.png, $dir/$s-ref.png"
    shot "$s" "$s" 1
    shot "$s" "$s-ref" 0
  fi
done
echo "water scenes written to $dir"
