"""Phase 4 diagnostic: neither the sugar->{DNp09,DNa02,MDN,DNp01} nor the
bitter->same pathway produced any decoder activity, confirmed both offline
(validate_v0_pipeline.py) and live (a real combat test where the character
took damage from 100 to 77 HP and never reacted). Before escalating to
Phase 5's full CMA-ES machinery, check whether these sensory channels reach
ANY real, named descending/motor neuron at all -- not just the four
originally picked -- so v0 has a chance at a visible reaction.

For each channel (sugar left, sugar right, bitter left, bitter right),
drives it at max rate for several sustained slices, then cross-references
every neuron with nonzero spike rate against the real annotation table's
super_class in {descending, motor}, reporting whichever are found, sorted
by rate.

Usage: python brain/find_reachable_motor_neurons.py
"""
import sys
from pathlib import Path

repo = Path(__file__).parent / "Drosophila_brain_model"
sys.path.insert(0, str(repo))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from model import default_params as params
from fast_sim import FastBrain
import neurons

ANNOTATIONS_TSV = "/private/tmp/claude-501/-Users-foolish-projects-fruitfly-terraria/7018cd34-309d-46c1-9061-7f823c2ebf71/scratchpad/flywire_annotations.tsv"
N_SLICES = 20  # 2s sustained, well past the ramp-up seen in earlier tests
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

    channels = {
        "sugar_left": neurons.SUGAR_WATER_GRN_LEFT,
        "sugar_right": neurons.SUGAR_WATER_GRN_RIGHT,
        "bitter_left": neurons.BITTER_GRN_LEFT,
        "bitter_right": neurons.BITTER_GRN_RIGHT,
    }

    for name, drive_ids in channels.items():
        brain, df_comp = build_brain()
        rates = sustained_rates(brain, drive_ids)
        active_total = int((rates > 0).sum())
        print(f"\n=== {name} ({len(drive_ids)} neurons driven at {MAX_RATE}Hz, "
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
        print(f"found {len(hits)} descending/motor neuron(s):")
        for rate, fid, sc, cc, ct, hb, side in hits[:30]:
            print(f"  {rate:6.1f} Hz  {fid}  {sc}/{cc}  cell_type={ct}  "
                  f"hemibrain_type={hb}  side={side}")


if __name__ == "__main__":
    main()
