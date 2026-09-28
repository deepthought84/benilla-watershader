#!/usr/bin/env bash
# Golden water scenes: capture the `water-*` scenarios (`capture/scenarios.rs`) with Improved Water
# on, one client process per scene, like visual.sh. Opens a real window. Captures hold
# Blizzard-derived imagery, so they go to a directory outside the repo or under target/.
#
#   scripts/water-shots.sh <dir>                  # <dir>/<scene>.png
#   scripts/water-shots.sh <dir> <n> [<every>]    # bursts: <dir>/<scene>-000.png … n shots, every
#                                                 # <every> fixed 1/60 s steps (default 2)
#
# The render state is pinned: 1600x900, render scale 2, far clip 777. The shutter waits for 300
# still frames: at these spots the default 30 pass before the world streams in, and the shot is
# the one-colour clear frame.
set -euo pipefail
cd "$(dirname "$0")/.."

dir="${1:?usage: water-shots.sh <dir> [<burst n> [<every>]]}"
burst="${2:-0}"
every="${3:-2}"
mkdir -p "$dir"

cargo build -q --release -p benilla
bin=./target/release/benilla

# The binary owns the scene list.
scenes=$(WOW_CAPTURE=list WOW_CAPTURE_LIST=water "$bin")
[ -n "$scenes" ] || { echo "the binary lists no water scenes" >&2; exit 1; }

for s in $scenes; do
  echo "capture $s -> $dir/$s.png (burst $burst, every $every)"
  WOW_CAPTURE="$s" WOW_CAPTURE_OUT="$dir/$s.png" \
    WOW_CAPTURE_BURST="$burst" WOW_CAPTURE_BURST_EVERY="$every" WOW_CAPTURE_STABLE=300 \
    WOW_WATER_STYLE=1 WOW_RENDER_SCALE=2 WOW_FARCLIP=777 WOW_WIN=1600x900 \
    "$bin" > "$dir/$s.log" 2>&1 || { echo "capture $s FAILED, log: $dir/$s.log" >&2; exit 1; }
done
echo "water scenes written to $dir"
