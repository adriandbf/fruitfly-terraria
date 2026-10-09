#!/usr/bin/env bash
# Stage M1 batch: calibration baselines + the fly's first fights, unattended.
# Run from the repo root on the Mac with Terraria in the arena world:
#   bash training/moonlord/run_batch.sh
# Leave the Terraria window in front for the whole run unless you are
# testing --keep-focus (the last block). Ctrl+C stops after the current fight.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
H="training/moonlord/harness.py"
N=${N:-5}
START=$(date +%Y%m%d_%H%M%S)

echo "== 1. idle baseline"
$PY $H --policy idle --episodes "$N" --seed 100 --whip-strategy balanced
echo "== 2. random keys baseline"
$PY $H --policy random --episodes "$N" --seed 100 --whip-strategy balanced
echo "== 3. scripted dodger, whip targets the nearest part (old)"
$PY $H --policy scripted --episodes "$N" --seed 100 --whip-strategy nearest
echo "== 4. scripted dodger, balanced whip targeting (hands low, head, hands, core)"
$PY $H --policy scripted --episodes "$N" --seed 100 --whip-strategy balanced
echo "== 5. fly, brain climb/dive + direction commitment"
$PY $H --policy fly --episodes "$N" --seed 100 --fly-vertical brain --fly-commit 10 --whip-strategy balanced
echo "== 6. fly, hand-rule climb/dive + direction commitment"
$PY $H --policy fly --episodes "$N" --seed 100 --fly-vertical rule --fly-commit 10 --whip-strategy balanced
echo "== report"
$PY training/moonlord/report.py --since "$START"
