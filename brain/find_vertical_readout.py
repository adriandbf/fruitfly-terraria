"""Stage M3 discovery: can the connectome itself drive CLIMB vs DIVE?

The fly's left/right already comes from the brain (DNa02, a crossed
"which side is the threat" code) and the jump from the Giant Fiber (DNp01).
Up/down in the Moon Lord fight currently comes from a hand rule. This
sweep looks for a brain-side vertical readout.

Literature anchor (the hypothesis this tests, not an assumption):
  - takeoff/escape: looming -> LC4/LPLC2 -> Giant Fiber DNp01 (von Reyn et
    al. 2014; Ache et al. 2019) -> in game: climb
  - landing: in flying flies, looming drives LANDING via descending
    neurons DNp07 and DNp10 (Ache et al. 2019, "State-dependent decoupling
    of sensory and motor circuits...") -> in game: dive
So the question is whether any visual looming/feature population reaches
DNp07/DNp10 in this connectome, and whether some input drives them more
than the Giant Fiber (that difference is what would make "dive" a real
brain decision instead of a hand rule).

Method (same as find_looming_pathway.py, the approach that has worked):
drive each candidate visual population at a fixed Poisson rate (an
optogenetic-style drive, not realistic vision), let it settle, then record
the rate of EVERY descending neuron. Inputs: each lobula columnar /
lobula plate type, split by side; plus a dorsal/ventral split of LC4 and
LPLC2 by soma height -- a WEAK proxy for "threat above vs below" (somata
are not laid out retinotopically), included only to see if it separates
anything at all.

Writes brain/results/vertical_readout_<stamp>.csv (every descending neuron
with a nonzero rate in any condition, plus the focus DNs always) and
prints the focus table. Runtime: ~1-2 min on the M2 Pro (one brain build,
~40 conditions x 1 s simulated).

Usage: python brain/find_vertical_readout.py [--rate 150] [--seconds 1.0]
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
repo = HERE / "Drosophila_brain_model"
sys.path.insert(0, str(repo))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from model import default_params as params
from fast_sim import FastBrain

ANNOTATIONS = HERE / "data" / "flywire_annotations.tsv.gz"
INPUT_TYPES = ["LC4", "LPLC2", "LC6", "LPLC1", "LPLC4", "LC16", "LC12", "LC11", "LC17",
               "LC9", "LC22", "LC21", "LC26", "LC24", "LC15", "LC18"]
SPLIT_TYPES = ["LC4", "LPLC2"]  # also driven as dorsal/ventral halves (soma height proxy)
FOCUS_DNS = ["DNp01", "DNp07", "DNp10", "DNp02", "DNp04", "DNp05", "DNp06", "DNp11",
             "DNp15", "DNp03", "DNa02", "DNb05", "DNa04", "DNb01", "MDN"]
WARMUP_SLICES = 3


def load_annotations():
    cols = ["root_id", "super_class", "cell_type", "hemibrain_type", "side", "soma_y"]
    return pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False, usecols=cols)


def conditions(ann):
    out = [("baseline", "none", "-", [])]
    for t in INPUT_TYPES:
        pop = ann[ann.cell_type == t]
        if pop.empty:
            continue
        for side in ("left", "right"):
            ids = pop[pop.side == side].root_id.tolist()
            if ids:
                out.append((f"{t}_{side}", t, side, ids))
        out.append((f"{t}_both", t, "both", pop.root_id.tolist()))
        if t in SPLIT_TYPES:
            ys = pop.soma_y.dropna()
            if len(ys) >= 4:
                mid = ys.median()  # FlyWire y grows downward (ventral)
                out.append((f"{t}_dorsal_half", t, "dorsal", pop[pop.soma_y < mid].root_id.tolist()))
                out.append((f"{t}_ventral_half", t, "ventral", pop[pop.soma_y >= mid].root_id.tolist()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=150.0, help="Poisson drive per input neuron (Hz)")
    ap.add_argument("--seconds", type=float, default=1.0, help="measured time per condition (after warm-up)")
    args = ap.parse_args()

    ann = load_annotations()
    print("Building brain (v783)...")
    t0 = time.time()
    df_comp = pd.read_csv(repo / "Completeness_783.csv", index_col=0)
    df_con = pd.read_parquet(repo / "Connectivity_783.parquet")
    brain = FastBrain(df_comp, df_con, params, dt_ms=1.0, exc_indices=[], seed=0)
    brain.stimulate([], [], duration_ms=10)
    print(f"  built in {time.time() - t0:.1f}s")

    dn = ann[ann.super_class == "descending"].copy()
    dn["idx"] = dn.root_id.map(brain.flyid2idx)
    dn = dn.dropna(subset=["idx"])
    dn["idx"] = dn.idx.astype(int)
    dn_type = dn.cell_type.fillna(dn.hemibrain_type).fillna("?")
    dn_idx = dn.idx.to_numpy()

    conds = conditions(ann)
    n_measure = max(1, int(round(args.seconds * 10)))
    rows = []
    t0 = time.time()
    for name, itype, side, ids in conds:
        brain.reset_state(seed=0)
        rates_hz = [args.rate] * len(ids)
        for _ in range(WARMUP_SLICES):
            brain.stimulate(ids, rates_hz, duration_ms=100)
        acc = np.zeros(len(dn_idx))
        total_active = 0
        for _ in range(n_measure):
            r = brain.stimulate(ids, rates_hz, duration_ms=100)
            acc += r[dn_idx]
            total_active = max(total_active, int((r > 0).sum()))
        acc /= n_measure
        for k, (rid, t, s) in enumerate(zip(dn.root_id, dn_type, dn.side)):
            if acc[k] > 0 or t in FOCUS_DNS:
                rows.append({"condition": name, "input_type": itype, "input_side": side, "n_driven": len(ids),
                             "active_neurons": total_active, "dn_root_id": rid, "dn_type": t, "dn_side": s,
                             "rate_hz": round(float(acc[k]), 2)})
        print(f"  {name:22s} driven {len(ids):4d}  active {total_active:6d}  "
              f"DNs firing {int((acc > 0).sum()):4d}")
    print(f"  sweep took {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"vertical_readout_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(HERE.parent)} ({len(df)} rows)")

    # Focus table: mean rate per focus DN type (both sides averaged) per condition
    focus = df[df.dn_type.isin(FOCUS_DNS)]
    table = focus.pivot_table(index="condition", columns="dn_type", values="rate_hz", aggfunc="mean").fillna(0)
    table = table.reindex(columns=[c for c in FOCUS_DNS if c in table.columns])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\nFocus DNs, mean rate (Hz) per condition:")
    print(table.round(1).to_string())

    # Landing vs takeoff contrast: where do DNp07/DNp10 beat the Giant Fiber?
    if {"DNp01"}.issubset(table.columns):
        land = table[[c for c in ("DNp07", "DNp10") if c in table.columns]].max(axis=1) if any(
            c in table.columns for c in ("DNp07", "DNp10")) else 0
        contrast = (land - table["DNp01"]).sort_values(ascending=False)
        print("\nLanding (max of DNp07/DNp10) minus takeoff (DNp01), Hz -- positive = a 'dive' candidate input:")
        print(contrast.round(1).head(12).to_string())

    # Strongest DN types overall, for anything we didn't think to look for
    top = (df[df.condition != "baseline"].groupby("dn_type").rate_hz.max().sort_values(ascending=False).head(20))
    print("\nTop 20 descending types by peak rate across all conditions:")
    print(top.round(1).to_string())


if __name__ == "__main__":
    main()
