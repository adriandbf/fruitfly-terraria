"""Check brain/np_sim.py (pure numpy) against fast_sim (numba) results:
re-runs conditions from a find_vertical_readout.py CSV with the same
protocol (reset seed 0, 3 warm-up slices, 10 measured slices at 150 Hz)
and compares descending-neuron rates. First run (2026-10-09): LC4, LPLC4
and LC6 matched to the Hz on every focus DN; correlation over all firing
DN types 0.993-0.997.

Usage: python brain/validate_np_sim.py brain/results/vertical_readout_<stamp>.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from np_sim import NpBrain  # noqa: E402

ref = pd.read_csv(sys.argv[1])
ann = pd.read_csv(HERE / "data" / "flywire_annotations.tsv.gz", sep="\t", low_memory=False,
                  usecols=["root_id", "super_class", "cell_type", "hemibrain_type", "side"])
b = NpBrain(HERE / "data" / "connectome_783.npz")
dn = ann[(ann.super_class == "descending") & ann.root_id.isin(b.flyid2idx.keys())]
dn_idx = np.array([b.flyid2idx[int(r)] for r in dn.root_id])
dn_type = dn.cell_type.fillna(dn.hemibrain_type).fillna("?").values
for cond, typ in [("LC4_both", "LC4"), ("LPLC4_both", "LPLC4"), ("LC6_both", "LC6")]:
    ids = ann[ann.cell_type == typ].root_id.tolist()
    b.reset_state(seed=0)
    for _ in range(3):
        b.stimulate(ids, [150] * len(ids))
    acc = sum(b.stimulate(ids, [150] * len(ids))[dn_idx] for _ in range(10)) / 10
    mine = pd.DataFrame({"t": dn_type, "r": acc}).groupby("t").r.mean()
    theirs = ref[ref.condition == cond].groupby("dn_type").rate_hz.mean()
    both = pd.concat([mine.rename("np"), theirs.rename("ref")], axis=1).fillna(0)
    both = both[(both.np > 0) | (both.ref > 0)]
    print(f"{cond}: DNp01 {mine.get('DNp01', 0):.0f}/{theirs.get('DNp01', 0):.0f}  "
          f"DNp07 {mine.get('DNp07', 0):.0f}/{theirs.get('DNp07', 0):.0f}  "
          f"corr over {len(both)} DN types {np.corrcoef(both.np, both.ref)[0, 1]:.3f}")
