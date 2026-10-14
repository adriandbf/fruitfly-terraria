"""One place to build "the fly's brain", in either engine and with the
null-model controls.

Engines (both expose stimulate(ids, rates_hz, duration_ms) -> per-neuron
rates array, rate_for(rates, flyid), reset_state(seed=None)):
  fast -- brain/fast_sim.FastBrain (numba; the live engine, ~20 ms per
          100 ms slice). Needs the model repo's parquet + pandas.
  np   -- brain/np_sim.NpBrain (pure numpy, ~300 ms per slice); runs
          anywhere brain/data/connectome_783.npz exists (offline training).

Null models (controls, applied the same way to either engine):
  none     -- the real connectome
  shuffled -- every synapse's target permuted (out-degree and weight
              distribution kept)
  ablated  -- the looming inputs' (LC4 + LPLC4) outgoing synapses zeroed
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for sub in ("brain", "brain/Drosophila_brain_model"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

import neurons  # noqa: E402

NPZ = ROOT / "brain" / "data" / "connectome_783.npz"
NULLS = ("none", "shuffled", "ablated")


def _engine(kind, seed=0):
    if kind == "np":
        from np_sim import NpBrain
        return NpBrain(NPZ, seed=seed)
    if kind == "fast":
        import pandas as pd
        from model import default_params
        from fast_sim import FastBrain
        repo = ROOT / "brain" / "Drosophila_brain_model"
        df_comp = pd.read_csv(repo / "Completeness_783.csv", index_col=0)
        df_con = pd.read_parquet(repo / "Connectivity_783.parquet")
        b = FastBrain(df_comp, df_con, default_params, 1.0, exc_indices=[], seed=seed)
        b.stimulate([], [], duration_ms=10)  # JIT warm-up
        return b
    raise ValueError(f"unknown engine '{kind}'")


def apply_null(brain, null, seed=0):
    if null == "none":
        return brain
    if null == "shuffled":
        brain.post_idx = np.random.default_rng(seed).permutation(brain.post_idx)
    elif null == "ablated":
        for fid in list(neurons.LC4_LOOMING) + list(neurons.LPLC4_LOOMING):
            i = brain.flyid2idx.get(int(fid))
            if i is not None:
                brain.post_w[brain.indptr[i]:brain.indptr[i + 1]] = 0.0
    else:
        raise ValueError(f"unknown null model '{null}'")
    return brain


def make_brain(engine="np", null="none", seed=0):
    """-> (brain, readout_fn); readout_fn maps a rates array to {flyid: Hz}
    over neurons.ALL_DESCENDING."""
    brain = apply_null(_engine(engine, seed), null, seed)
    return brain, (lambda rates: {fid: brain.rate_for(rates, fid) for fid in neurons.ALL_DESCENDING})
