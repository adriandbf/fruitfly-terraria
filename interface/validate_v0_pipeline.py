"""Phase 4: documents the actual, observed behavior of the v0 encoder/
decoder pipeline against a few canonical synthetic game states. This is
NOT a pass/fail correctness test -- there's no ground truth for "what
should a hand-built v0 mapping do" -- it's a permanent, reproducible record
of what the real connectome does with these specific real neurons, since
some of it turned out to be genuinely surprising (see the asymmetry note
below) and worth being able to reproduce rather than just remembering.

This script only exercises the brain/encoder/decoder side with synthetic
states; interface/run_fly.py is the live version. Approach and the LC4-
based threat/jump pathway have both since been confirmed live too -- see
CLAUDE.md's status for the live test results.

Usage: python interface/validate_v0_pipeline.py
"""
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
import decoder

NAMES = {
    720575940604737708: "DNa02_R", 720575940629327659: "DNa02_L",
    720575940631082808: "MDN_1L", 720575940610236514: "MDN_1R",
    720575940616026939: "MDN_2L", 720575940640331472: "MDN_2R",
    720575940632499757: "DNp01_R(GF)", 720575940622838154: "DNp01_L(GF)",
    720575940628352850: "DNb05_L", 720575940629461442: "DNb05_R",
}


def build_brain():
    repo = ROOT / "brain" / "Drosophila_brain_model"
    df_comp = pd.read_csv(repo / "Completeness_783.csv", index_col=0)
    df_con = pd.read_parquet(repo / "Connectivity_783.parquet")
    return FastBrain(df_comp, df_con, params, dt_ms=1.0, exc_indices=[], seed=0)


def run_scenario(label, state, n_slices=10):
    brain = build_brain()
    brain.stimulate([], [], duration_ms=10)  # JIT warm-up
    drives = encoder.compute_drives(state)
    ids, rates = encoder.encode(state)
    print(f"\n=== {label} ===")
    print(f"drives: {drives}  (driving {len(ids)} neurons)")

    last_action = None
    for k in range(n_slices):
        spike_rates = brain.stimulate(ids, rates, duration_ms=100)
        readout = {f: brain.rate_for(spike_rates, f) for f in neurons.ALL_DESCENDING}
        last_action = decoder.decode(readout, state["player"]["direction"])
        active_total = int((spike_rates > 0).sum())
        nonzero_dn = {NAMES[f]: round(v, 1) for f, v in readout.items() if v > 0}
        print(f"  slice {k}: active_total={active_total:5d}  DN={nonzero_dn}  action={last_action}")
    return last_action


def main():
    base_player = {"x": 1000, "y": 500, "direction": 1}

    run_scenario("approach: human to the RIGHT",
                 {"in_world": True, "player": base_player,
                  "players": [{"x": 1100, "y": 500}], "npcs": []})

    run_scenario("approach: human to the LEFT",
                 {"in_world": True, "player": base_player,
                  "players": [{"x": 900, "y": 500}], "npcs": []})

    run_scenario("threat: hostile close, to the RIGHT",
                 {"in_world": True, "player": base_player, "players": [],
                  "npcs": [{"x": 1050, "y": 500, "hostile": True}]})

    run_scenario("idle: nothing nearby",
                 {"in_world": True, "player": base_player, "players": [], "npcs": []})

    print("""
=== summary ===
Mechanically working: real annotation-based neurons drive and read out
correctly, no crashes, valid actions every slice (see git history for a
real bug this caught: FlyWire IDs reaching FastBrain's internal array
indexing raw, before stimulate() translated them).

NOT yet correct, and deliberately left that way for Phase 5: the left and
right sugar/water GRN populations recruit wildly different amounts of
downstream activity in this real, individually-reconstructed connectome
(right: ~7-8k neurons active; left: ~460, flat, never ramping up over 10
slices) -- so "approach: human to the LEFT" above likely won't show the
mirror-image steering of the RIGHT case. This isn't a code bug (rates are
verified correct on both sides); it's a genuine asymmetry in this
particular brain's real wiring. CLAUDE.md's Phase 5 explicitly plans for
this: comparing against a random mapping, a shuffled connectome, and
ablations is exactly how you'd tell a real, usable pathway from one that's
this lopsided by chance/reconstruction quirks -- that calibration belongs
there, not as more hand-tuned constants here.
""")


if __name__ == "__main__":
    main()
