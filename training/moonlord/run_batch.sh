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

echo "== 2. idle baseline (no movement)"
$PY $H --policy idle --episodes "$N" --seed 100
echo "== 3. random keys baseline"
$PY $H --policy random --episodes "$N" --seed 100
echo "== 4. scripted dodger (the ceiling)"
$PY $H --policy scripted --episodes "$N" --seed 100
echo "== 5. the fly (stopgap projectile sense: projectiles shown to LC4 as enemies)"
$PY $H --policy fly --episodes "$N" --seed 100
echo "== 6. focus test: random keys with --keep-focus; click another app during it and see if the game keeps running"
$PY $H --policy random --episodes 1 --keep-focus --seed 999
echo "== report"
$PY training/moonlord/report.py
