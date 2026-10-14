"""Offline checks for the training tools (cmaes, the fake fight's domain
randomization, the fly's senses). No game and no connectome needed. Run:
python training/moonlord/test_training.py"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT / "brain"):
    sys.path.insert(0, str(p))
import fly_params  # noqa: E402
import harness  # noqa: E402
from cmaes import CMAES  # noqa: E402
from fake_bridge import randomized  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


# 1. The arena box the senses use comes from loadout.json.
lo = json.loads((HERE / "loadout.json").read_text())
box = fly_params.arena_px(lo)
check("arena box from the loadout", box["half_width_px"] == 16.0 * lo["arena"]["half_width_tiles"] and box["height_px"] > 0,
      str(box))

# 2. built-in CMA-ES finds a known optimum.
es = CMAES([0.9] * 5, 0.3, popsize=10, seed=1)
for _ in range(60):
    xs = es.ask()
    es.tell([float(((x - 0.3) ** 2).sum()) for x in xs])
check("CMA-ES converges on a quadratic", np.allclose(es.mean, 0.3, atol=0.02), str(np.round(es.mean, 3)))

# 3. Domain randomization: same seed -> same boss; Master fights jitter around Master values.
a, b = randomized({}, 7, 0.15), randomized({}, 7, 0.15)
check("randomized is seeded", a == b and a != randomized({}, 8, 0.15))
check("randomized is a no-op at r=0", randomized({"x": 1}, 7, 0.0) == {"x": 1})
m = randomized({"difficulty": "master"}, 7, 0.15)
check("Master fights keep difficulty=master", m["difficulty"] == "master" and m.keys() >= a.keys())

# 4. config_from_loadout = the fight harness.py runs.
c = harness.config_from_loadout(lo, "fight", "balanced")
check("config_from_loadout carries whip, heal and buffs",
      c["whip_strategy"] == "balanced" and c["whip_assist"] and c["heal_at_hp"] == lo["heal_at_hp"]
      and c["buffs"] == lo["buffs"] and c["summon_slots"] == lo["summon_slots"])

# 5. The fly's senses (scene3): time to contact, direction, beam.
import scene3  # noqa: E402
base = {"in_world": True, "player": {"cx": 0.0, "cy": 0.0, "x": -10.0, "y": -21.0, "vx": 0.0, "vy": 0.0},
        "npcs": [], "episode": {}}
fast_far = {**base, "projectiles": [{"type": 462, "cx": 400.0, "cy": 0.0, "vx": -16.0, "vy": 0.0}]}
slow_near = {**base, "projectiles": [{"type": 454, "cx": 150.0, "cy": 0.0, "vx": -2.0, "vy": 0.0}]}
rf, rs = scene3.scene_rates_v3(fast_far), scene3.scene_rates_v3(slow_near)
check("v3: a fast bolt 400 px out looms more than a slow sphere 150 px out (time to contact)",
      rf[("LC11", 1)] > rs[("LC11", 1)] > 0, f"{rf[('LC11', 1)]:.0f} vs {rs[('LC11', 1)]:.0f} Hz")
below = scene3.scene_rates_v3({**base, "projectiles": [{"type": 462, "cx": -50.0, "cy": 300.0, "vx": 0.0, "vy": -12.0}]})
check("v3: a projectile rising from below-left drives LC12 left, not LC11",
      below[("LC12", -1)] > 0 and below[("LC11", -1)] == 0 and below[("LC12", 1)] == 0)
beam = scene3.scene_rates_v3({**base, "projectiles": [{"type": 455, "cx": -600.0, "cy": -400.0, "vx": 0.83, "vy": 0.56}]})
check("v3: a deathray passing near drives LC22", max(beam[("LC22", -1)], beam[("LC22", 1)]) > 0)
# beam from (0,-900) pointing 0.4 rad to the right of straight down at the fly (0,0): ~360 px off the
# line (old band silent) but within 0.6 rad -> LC22 on the beam's side (right)
far_beam = scene3.scene_rates_v3({**base, "projectiles": [{"type": 455, "cx": 0.0, "cy": -900.0,
                                                            "vx": math.sin(0.4), "vy": math.cos(0.4)}]})
check("v3.1: a sweeping deathray 0.4 rad away already drives LC22 on its side",
      far_beam[("LC22", 1)] > 0 and far_beam[("LC22", -1)] == 0, f"{far_beam[('LC22', 1)]:.0f} Hz")

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
