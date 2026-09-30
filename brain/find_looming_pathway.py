"""Phase 4 diagnostic, threat pathway: the bitter GRN -> descending pathway
is confirmed dead (brain/find_reachable_motor_neurons.py: only ~50-90
neurons active total, no meaningful descending/motor hit, confirmed live
in real combat). Real fly threat/escape circuits are visual, not
gustatory -- specific looming-detector neurons in the lobula columnar /
visual projection class (LC4, LC6, LPLC1, LPLC2, etc.) are well
documented in the literature to drive the Giant Fiber (DNp01) and other
escape descending neurons directly. Same Poisson-kick-to-v activation
mechanism used for sugar/bitter (mimicking optogenetic activation, not
realistic visual transduction -- consistent with how every sensory
channel in this project is driven).

For each candidate looming-detector type, drives its full population
(both sides at once) at max rate for several sustained slices, then
cross-references every neuron with nonzero spike rate against the real
annotation table's descending/motor categories, same method as
find_reachable_motor_neurons.py.

Usage: python brain/find_looming_pathway.py
"""
import sys
from pathlib import Path

repo = Path(__file__).parent / "Drosophila_brain_model"
sys.path.insert(0, str(repo))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from model import default_params as params
from fast_sim import FastBrain

ANNOTATIONS_TSV = "/private/tmp/claude-501/-Users-foolish-projects-fruitfly-terraria/7018cd34-309d-46c1-9061-7f823c2ebf71/scratchpad/flywire_annotations.tsv"
CANDIDATES = ["LC4", "LC6", "LPLC1", "LPLC2", "LC16", "LC12"]
N_SLICES = 20  # 2s sustained
MAX_RATE = 150.0


def build_brain():
    df_comp = pd.read_csv(repo / "Completeness_783.csv", index_col=0)
    df_con = pd.read_parquet(repo / "Connectivity_783.parquet")
    return FastBrain(df_comp, df_con, params, dt_ms=1.0, exc_indices=[], seed=0), df_comp


def sustained_rates(brain, drive_ids):
    brain.stimulate([], [], duration_ms=10)  # JIT warm-up
    rates_hz = [MAX_RATE] * len(drive_ids)
    rates = None
    for _ in range(N_SLICES):
        rates = brain.stimulate(drive_ids, rates_hz, duration_ms=100)
    return rates


def main():
    ann = pd.read_csv(ANNOTATIONS_TSV, sep="\t", low_memory=False)
    motor_like = ann[ann["super_class"].isin(["descending", "motor"])]
    motor_ids = dict(zip(motor_like["root_id"], motor_like.to_dict("records")))

    for name in CANDIDATES:
        drive_ids = ann[ann["cell_type"] == name]["root_id"].tolist()
        brain, df_comp = build_brain()
        present_ids = [i for i in drive_ids if i in set(df_comp.index)]
        rates = sustained_rates(brain, present_ids)
        active_total = int((rates > 0).sum())
        print(f"\n=== {name} ({len(present_ids)} neurons driven at {MAX_RATE}Hz, "
              f"{N_SLICES * 100}ms sustained) ===")
        print(f"total active neurons: {active_total}")

        hits = []
        for idx, rate in enumerate(rates):
            if rate <= 0:
                continue
            fid = df_comp.index[idx]
            if fid in motor_ids:
                row = motor_ids[fid]
                hits.append((rate, fid, row["super_class"], row["cell_class"],
                             row["cell_type"], row["hemibrain_type"], row["side"]))

        if not hits:
            print("no descending/motor neurons found in the active set")
            continue

        hits.sort(reverse=True)
        print(f"found {len(hits)} descending/motor neuron(s), top 15:")
        for rate, fid, sc, cc, ct, hb, side in hits[:15]:
            print(f"  {rate:6.1f} Hz  {fid}  {sc}/{cc}  cell_type={ct}  "
                  f"hemibrain_type={hb}  side={side}")


if __name__ == "__main__":
    main()
