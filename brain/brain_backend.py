"""One place to build "the fly's brain" for offline training, in any of
three engines and with any of the null-model controls.

Engines (all expose stimulate(ids, rates_hz, duration_ms) -> rates array,
rate_for(rates, flyid), reset_state(seed=None)):
  fast  -- brain/fast_sim.FastBrain (numba; the Mac's live engine, ~20 ms
           per 100 ms slice). Needs the model repo's parquet + pandas.
  np    -- brain/np_sim.NpBrain (pure numpy, ~300 ms per slice); runs
           anywhere brain/data/connectome_783.npz exists.
  table -- brain/response_table.TableBrain: a lookup table of descending-
           neuron rates measured from one of the engines above on a grid
           of looming inputs. ~1000x faster; an approximation, so anything
           it finds must be re-checked on a real engine (and then live).

Null models (Phase 5 baselines, applied the same way to every engine):
  none           -- the real connectome and the claimed neurons
  shuffled       -- every synapse's target permuted (out-degree and weight
                    distribution kept; training/baselines.py's model)
  ablated        -- the looming inputs' (LC4 + LPLC4) outgoing synapses zeroed
  random_mapping -- same population sizes, but the input and readout
                    neurons are random ones (patches brain/neurons.py
                    module-wide, so call this once per process)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for sub in ("brain", "brain/Drosophila_brain_model", "interface"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

import neurons  # noqa: E402

NPZ = ROOT / "brain" / "data" / "connectome_783.npz"
NULLS = ("none", "shuffled", "ablated", "random_mapping")

# Every neurons.py attribute the Moon Lord encoder/decoder/FlyPolicy read.
INPUT_ATTRS = ["LC4_LOOMING_LEFT", "LC4_LOOMING_RIGHT", "LPLC4_LOOMING_LEFT", "LPLC4_LOOMING_RIGHT",
               "SUGAR_WATER_GRN_LEFT", "SUGAR_WATER_GRN_RIGHT"]
OUTPUT_ATTRS = ["DNA02_STEERING_LEFT", "DNA02_STEERING_RIGHT", "DNP01_JUMP_LEFT", "DNP01_JUMP_RIGHT",
                "DNP07_LANDING_LEFT", "DNP07_LANDING_RIGHT", "DNB05_MOVE_LEFT", "DNB05_MOVE_RIGHT",
                "MDN_BACKWARD"]


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


def apply_random_mapping(all_ids, seed=0):
    """Swap every input/readout population in brain/neurons.py for random
    neurons of the same size (module-wide; call once per process)."""
    rng = np.random.default_rng(seed)
    attrs = INPUT_ATTRS + OUTPUT_ATTRS
    sizes = [len(getattr(neurons, a)) for a in attrs]
    sample = [int(x) for x in rng.choice(np.asarray(all_ids), size=sum(sizes), replace=False)]
    i = 0
    for a, n in zip(attrs, sizes):
        setattr(neurons, a, sample[i:i + n])
        i += n
    for base in ("DNA02_STEERING", "DNP01_JUMP", "DNP07_LANDING", "DNB05_MOVE",
                 "LC4_LOOMING", "LPLC4_LOOMING", "SUGAR_WATER_GRN"):
        setattr(neurons, base, getattr(neurons, base + "_LEFT") + getattr(neurons, base + "_RIGHT"))
    neurons.ALL_DESCENDING = (neurons.DNA02_STEERING + neurons.MDN_BACKWARD + neurons.DNP01_JUMP
                              + neurons.DNB05_MOVE + neurons.DNP07_LANDING)


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
    elif null == "random_mapping":
        apply_random_mapping(brain.ids if hasattr(brain, "ids") else list(brain.flyid2idx), seed)
    else:
        raise ValueError(f"unknown null model '{null}'")
    return brain


def readout_fn_for(brain):
    """rates array -> {flyid: Hz} over neurons.ALL_DESCENDING (read at call
    time, so a random mapping applied later is honoured)."""
    return lambda rates: {fid: brain.rate_for(rates, fid) for fid in neurons.ALL_DESCENDING}


def make_brain(engine="np", null="none", seed=0, table_path=None):
    """-> (brain, readout_fn), ready for FlyPolicy(brain=..., readout_fn=...)."""
    if engine == "table":
        from response_table import TableBrain
        brain = TableBrain(table_path, seed=seed)
        if brain.null != null:
            raise ValueError(f"table {table_path} was measured with null='{brain.null}', not '{null}'")
        if null == "random_mapping":
            apply_random_mapping(brain.all_ids, brain.null_seed)
        return brain, brain.readout
    brain = apply_null(_engine(engine, seed), null, seed)
    return brain, readout_fn_for(brain)
