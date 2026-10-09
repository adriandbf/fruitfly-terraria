"""One-time export of the v783 connectome to plain numpy (.npz), so the
model can also be loaded where pyarrow/numba aren't installed (e.g. the
cloud workspace Claude runs analysis in). Not used by the live loop.

Writes brain/data/connectome_783.npz with:
  ids    -- FlyWire root ids, index-aligned with Completeness_783.csv
  pre    -- presynaptic neuron index per synapse row
  post   -- postsynaptic neuron index per synapse row
  w      -- 'Excitatory x Connectivity' (signed synapse count; multiply by
            params['w_syn'] for the model's weight)
Usage: python brain/export_connectome_npz.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
repo = HERE / "Drosophila_brain_model"
df_comp = pd.read_csv(repo / "Completeness_783.csv", index_col=0)
df_con = pd.read_parquet(repo / "Connectivity_783.parquet")
out = HERE / "data" / "connectome_783.npz"
out.parent.mkdir(exist_ok=True)
np.savez_compressed(out,
                    ids=df_comp.index.to_numpy(dtype=np.int64),
                    pre=df_con["Presynaptic_Index"].to_numpy(dtype=np.int32),
                    post=df_con["Postsynaptic_Index"].to_numpy(dtype=np.int32),
                    w=df_con["Excitatory x Connectivity"].to_numpy(dtype=np.float32))
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(df_con)} synapse rows, {len(df_comp)} neurons)")
