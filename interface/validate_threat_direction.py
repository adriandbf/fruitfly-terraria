"""Can the brain's own outputs tell which SIDE a threat is on?

Prerequisite for a brain-driven flee: running away needs a direction, and
the decoder only sees descending-neuron rates. This drives the threat
channel (encoder.py -> LC4) with a hostile on the left or right at several
distances, over several seeds, and scores each candidate direction
estimator PER 100ms SLICE (the decoder's real operating condition: one
decision per slice, not an average over seconds).

Result when first run (288 per-slice samples; first 3 slices of each trial
skipped as cold-start ramp):
  DNa02 only (L - R):   100% correct when it decides, decides in 86% of slices
  Giant Fiber (R - L):   92% correct, decides 88% (81% for left-side threats)
  3*DNa02 + GF:          99% correct, decides 97%
DNa02 is a clean, mirror-symmetric, CROSSED signal for threat: hostile on the
right -> DNa02_L fires; hostile on the left -> DNa02_R fires. That is the same
"stimulus side" code the approach channel uses -- which means the decoder's
current rule (DNa02_L dominant -> move right) walks TOWARD a threat, correct
for approach and backwards for fleeing.

Caveats: simulated brain with synthetic states (not yet live); the side
information enters via the encoder's ipsi/contra lateralization, the brain
then transforms it; DNa02 rates are small (~10Hz = about one spike per
slice) so single-slice decisions are noisy in coverage, not in sign.

Usage: python interface/validate_threat_direction.py [--seeds 8]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brain"))
sys.path.insert(0, str(ROOT / "brain" / "Drosophila_brain_model"))
sys.path.insert(0, str(ROOT / "interface"))

import pandas as pd
from model import default_params as params
from fast_sim import FastBrain
import neurons
import encoder

DISTANCES = (100, 250, 400)
SKIP_SLICES = 3
N_SLICES = 9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    repo = ROOT / "brain" / "Drosophila_brain_model"
    df_comp = pd.read_csv(repo / "Completeness_783.csv", index_col=0)
    df_con = pd.read_parquet(repo / "Connectivity_783.parquet")
    brain = FastBrain(df_comp, df_con, params, dt_ms=1.0, exc_indices=[], seed=0)
    brain.stimulate([], [], duration_ms=10)

    data = json.loads((ROOT / "brain" / "selected_neurons.json").read_text())
    by_side = lambda key, side: [int(r["root_id"]) for r in data["descending"][key] if r["side"] == side]
    gf_l, gf_r = by_side("DNp01_giant_fiber_jump", "left"), by_side("DNp01_giant_fiber_jump", "right")
    mean = lambda r, ids: sum(brain.rate_for(r, f) for f in ids) / len(ids)

    # sample: (true side, DNa02 evidence, GF evidence); positive => "threat is on the RIGHT"
    samples = []
    for side, dist, seed in itertools.product(("right", "left"), DISTANCES, range(args.seeds)):
        brain.reset_state(seed=seed)
        px = 1000.0
        hx = px + (dist if side == "right" else -dist)
        state = {"in_world": True, "player": {"x": px, "y": 0, "direction": 1}, "players": [],
                 "npcs": [{"x": hx, "y": 0, "hostile": True}]}
        ids, rates = encoder.encode(state)
        for i in range(N_SLICES):
            r = brain.stimulate(ids, rates, duration_ms=100)
            if i >= SKIP_SLICES:
                samples.append((side,
                                mean(r, neurons.DNA02_STEERING_LEFT) - mean(r, neurons.DNA02_STEERING_RIGHT),
                                mean(r, gf_r) - mean(r, gf_l)))

    def score(fn, subset):
        decided = [(s, fn(d, g)) for s, d, g in subset if fn(d, g) != 0]
        ok = sum(1 for s, v in decided if (v > 0) == (s == "right"))
        return (100 * ok / len(decided) if decided else float("nan"),
                100 * len(decided) / len(subset))

    estimators = {
        "DNa02 only": lambda d, g: d,
        "GF only": lambda d, g: g,
        "3*DNa02 + GF": lambda d, g: 3 * d + g,
    }
    print(f"{len(samples)} per-slice samples\n")
    print(f"{'estimator':<16}{'subset':<18}{'correct %':>10}{'decided %':>11}")
    for name, fn in estimators.items():
        for label, sub in (("all", samples),
                           ("threat on RIGHT", [x for x in samples if x[0] == "right"]),
                           ("threat on LEFT", [x for x in samples if x[0] == "left"])):
            a, c = score(fn, sub)
            print(f"{name:<16}{label:<18}{a:10.0f}{c:11.0f}")
        print()


if __name__ == "__main__":
    main()
