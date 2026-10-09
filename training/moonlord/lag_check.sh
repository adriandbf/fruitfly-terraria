#!/usr/bin/env bash
# Is the fight slowdown caused by the whip? 2 idle fights with the whip assist,
# 2 without, then the game speed of each (1.00x = full speed).
# Live: idle fights ran at 1.0x before the whip worked, ~0.35x after.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
H="training/moonlord/harness.py"
$PY $H --policy idle --episodes 2 --seed 300
$PY $H --policy idle --episodes 2 --seed 300 --no-whip-assist
$PY - <<'PYEOF'
import json
rows = [json.loads(l) for l in open("logs/moonlord/results.jsonl")][-4:]
for r in rows:
    print(r["run"], "whip" if r["ledger"].get("assist") else "no whip", "game speed", r.get("game_speed"), "survived", r["seconds"], "s")
PYEOF
