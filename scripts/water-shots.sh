#!/usr/bin/env bash
# Golden water scenes: capture the `water-*` scenarios (`capture/scenarios.rs`) with Improved Water
# on, one client process per scene, like visual.sh. Opens a real window. Captures hold
# Blizzard-derived imagery, so they go to a directory outside the repo or under target/; the
# script refuses any other place in the repo.
#
#   scripts/water-shots.sh [<dir>]                # <dir>/<scene>.png, <dir> target/visual/water
#   scripts/water-shots.sh <dir> <n> [<every>]    # bursts: <dir>/<scene>-000.png … n shots, every
#                                                 # <every> fixed 1/60 s steps (default 2)
#
# The render state is pinned: 1600x900, render scale 2, far clip 777. Each scene carries its own
# settle (300 still frames, `capture`'s WATER_STABLE_FRAMES): at these spots the default 30 pass
# before the world streams in, and the shot is the one-colour clear frame.
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
scenes=$(WOW_CAPTURE=list WOW_CAPTURE_LIST=water "$bin")
[ -n "$scenes" ] || { echo "the binary lists no water scenes" >&2; exit 1; }

for s in $scenes; do
  if [ "$burst" -gt 0 ]; then
    echo "capture $s -> $dir/$s-000.png … $s-$(printf %03d $((burst - 1))).png (every $every)"
  else
    echo "capture $s -> $dir/$s.png"
  fi
  WOW_CAPTURE="$s" WOW_CAPTURE_OUT="$dir/$s.png" \
    WOW_CAPTURE_BURST="$burst" WOW_CAPTURE_BURST_EVERY="$every" \
    WOW_WATER_STYLE=1 WOW_RENDER_SCALE=2 WOW_FARCLIP=777 WOW_WIN=1600x900 \
    "$bin" > "$dir/$s.log" 2>&1 || { echo "capture $s FAILED, log: $dir/$s.log" >&2; exit 1; }
done
echo "water scenes written to $dir"
