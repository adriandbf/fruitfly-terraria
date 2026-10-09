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

echo "== 2. idle baseline (no movement)"
$PY $H --policy idle --episodes "$N" --seed 100
echo "== 3. random keys baseline"
$PY $H --policy random --episodes "$N" --seed 100
echo "== 4. scripted dodger (the ceiling)"
$PY $H --policy scripted --episodes "$N" --seed 100
echo "== 5. the fly (attention + flee-by-flying rule)"
$PY $H --policy fly --episodes "$N" --seed 100
echo "== 6. the fly without the flee-by-flying rule (ablation)"
$PY $H --policy fly --episodes "$N" --seed 100 --no-fly-flee
echo "== report"
$PY training/moonlord/report.py --since "$START"
