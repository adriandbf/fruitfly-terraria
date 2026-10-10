"""Offline checks for the training tools (fly_params, cmaes, the response
table brain, train_fly_cma, imitate). No game and no connectome needed: the
brain is a small synthetic response table. Run:
python training/moonlord/test_training.py"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT / "brain", ROOT / "interface"):
    sys.path.insert(0, str(p))
import fly_params  # noqa: E402
import harness  # noqa: E402
import imitate  # noqa: E402
import neurons  # noqa: E402
import response_table as rt  # noqa: E402
from cmaes import CMAES  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


# 1. fly_params: defaults survive the unit round trip; bounds and ordering hold.
p = fly_params.unit_to_params(fly_params.default_unit())
check("defaults round-trip through unit space",
      abs(p["encoder"]["max_rate_hz"] - 150) < 1e-3 and p["decoder"]["flee_on_hz"] == 30.0
      and p["policy"]["threat_horizon_ticks"] == 90 and abs(p["policy"]["above_px"] - 60) < 1e-3, str(p["policy"]))
q = fly_params.unit_to_params([2.0] * len(fly_params.PARAM_SPEC))
check("unit values are clipped to the bounds", q["encoder"]["max_rate_hz"] == 300.0)
r = fly_params.unit_to_params([0.5] * len(fly_params.PARAM_SPEC))
check("flee_off_hz never exceeds flee_on_hz", r["decoder"]["flee_off_hz"] <= r["decoder"]["flee_on_hz"])
check("integer knobs are ints", isinstance(r["policy"]["commit_ticks"], int))
kw = fly_params.policy_kwargs(r)
check("policy_kwargs feeds FlyPolicy", set(kw) >= {"enc_params", "dec_params", "vertical", "above_px", "commit_ticks"})

# 2. built-in CMA-ES finds a known optimum.
es = CMAES([0.9] * 5, 0.3, popsize=10, seed=1)
for _ in range(60):
    xs = es.ask()
    es.tell([float(((x - 0.3) ** 2).sum()) for x in xs])
check("CMA-ES converges on a quadratic", np.allclose(es.mean, 0.3, atol=0.02), str(np.round(es.mean, 3)))

# 3. TableBrain: exact at grid points, linear between them, refuses untabled inputs.
tmp = Path(tempfile.mkdtemp())
levels = [0.0, 100.0, 300.0]
dn = list(neurons.ALL_DESCENDING)
L = len(levels)
table = np.zeros((1, L, L, L, L, len(dn)), dtype=np.float32)
gf = dn.index(neurons.DNP01_JUMP[0])
for i in range(L):
    table[0, i, :, :, :, gf] = levels[i] / 2  # GF = half the LC4-left rate
meta = {"engine": "synthetic", "null": "none", "null_seed": 0, "levels": levels, "k": 1, "warm_ms": 0,
        "channels": rt.CHANNELS, "dn_ids": [int(x) for x in dn]}
np.savez_compressed(tmp / "t.npz", table=table, meta=json.dumps(meta))
tb = rt.TableBrain(tmp / "t.npz")
ids = list(neurons.LC4_LOOMING_LEFT)
check("table exact at a grid point", tb.rate_for(tb.stimulate(ids, [100.0] * len(ids)), dn[gf]) == 50.0)
check("table interpolates between points", abs(tb.rate_for(tb.stimulate(ids, [200.0] * len(ids)), dn[gf]) - 100.0) < 1e-6)
try:
    tb.stimulate(list(neurons.SUGAR_WATER_GRN_LEFT), [50.0] * len(neurons.SUGAR_WATER_GRN_LEFT))
    check("table refuses the approach channel", False)
except ValueError:
    check("table refuses the approach channel", True)

# 4. config_from_loadout = the fight harness.py runs.
lo = json.loads((HERE / "loadout.json").read_text())
c = harness.config_from_loadout(lo, "fight", "balanced")
check("config_from_loadout carries whip, heal and buffs",
      c["whip_strategy"] == "balanced" and c["whip_assist"] and c["heal_at_hp"] == lo["heal_at_hp"]
      and c["buffs"] == lo["buffs"] and c["summon_slots"] == lo["summon_slots"])

# 5. train_fly_cma end to end on the synthetic table (1 generation).
import train_fly_cma  # noqa: E402
train_fly_cma.PARAMS_DIR = tmp
train_fly_cma.LOG_DIR = tmp
out = train_fly_cma.main(["--engine", "table", "--table", str(tmp / "t.npz"), "--workers", "1",
                          "--generations", "1", "--popsize", "4", "--train-seeds", "1", "--heldout-seeds", "1",
                          "--max-seconds", "20", "--name", "t"])
saved = json.loads(Path(out).read_text())
check("train_fly_cma writes a loadable params file",
      set(saved) >= {"encoder", "decoder", "policy", "meta"} and "heldout" in saved["meta"])
pol_kw = {**saved["policy"]}
import policies  # noqa: E402
fp = policies.FlyPolicy(brain=tb, readout_fn=tb.readout, **fly_params.policy_kwargs(saved))
check("FlyPolicy takes the tuned above_px", fp.ABOVE_PX == saved["policy"]["above_px"])

# 6. imitation scoring.
check("teacher labels: up counts as climb, down as dive",
      imitate.teacher_label({"left": True, "up": True}) == (-1, 1)
      and imitate.teacher_label({"right": True, "down": True}) == (1, -1)
      and imitate.teacher_label({}) == (0, 0))
check("balanced accuracy ignores class imbalance",
      abs(imitate.balanced_accuracy([1] * 10, [1] * 9 + [0]) - 0.5) < 1e-9)

# 7. Scene fly: senses -> (stub) brain -> learned readout, keys credited "brain".
import scene_fly  # noqa: E402


class _StubBrain:
    def __init__(self):
        self.last = None

    def stimulate(self, ids, rates, duration_ms=100):
        self.last = dict(zip(ids, rates))
        return self.last

    def rate_for(self, rates, fid):
        return 0.0

    def reset_state(self, seed=None):
        pass


from fake_bridge import FakeMoonLordBridge  # noqa: E402
b = FakeMoonLordBridge({"seed": 3})
b.request({"cmd": "spawn_npc", "type": 398})
st = b.request({"cmd": "get_state"})
sb = _StubBrain()
n_dn = len(scene_fly.SCENE["readout_dns"])
ro = {"wh": [0.0] * n_dn + [1.0], "wv": [0.0] * n_dn + [1.0], "thr": 0.5, "hold": 0}
sp = scene_fly.SceneFlyPolicy(brain=sb, readout_fn=lambda r: {}, readout=ro,
                              **fly_params.policy_kwargs(fly_params.load(HERE / "params" / "sectors_linear.json")))
a, src = sp.act(st)
vs = scene_fly.SCENE["VS"]["left"][0]
check("scene fly drives the self-motion (VS) cells and a looming population",
      sb.last.get(vs, 0) > 0 and any(sb.last.get(int(f), 0) > 0 for f in neurons.LC4_LOOMING_LEFT + neurons.LC4_LOOMING_RIGHT),
      f"VS {sb.last.get(vs)}")
check("scene fly readout bias -> right + climb, credited brain",
      a["right"] and a["jump"] and set(src.values()) == {"brain"}, f"{a} {src}")

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
