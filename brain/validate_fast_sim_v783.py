"""Phase 4: re-validate fast_sim.py (speed + correctness vs Brian) on the
v783 connectome, before building the encoder/decoder on top of it.

v783 is used from here on (not v630, which Phase 0-3 validated against)
because the real FlyWire annotation table only ships v783 root_ids -- see
neurons.py's docstring. v783 is a ~9% bigger graph (138,639 neurons /
15.09M synapses vs 127,400 / 14.7M), so this reconfirms both conclusions
still hold rather than assuming they carry over.

Drive: the sugar/water GRNs from neurons.py (real annotation, same role as
the original v630 "sugar neuron" example) at 150Hz, matching model.py's
default r_poi.

Usage: python brain/validate_fast_sim_v783.py [dt_ms]
"""
import sys, time
from pathlib import Path

repo = Path(__file__).parent / "Drosophila_brain_model"
sys.path.insert(0, str(repo))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from brian2 import Network, ms, defaultclock
from model import create_model, poi, default_params as params
from fast_sim import FastBrain
import neurons

dt_ms = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
DURATION_MS = 1000

comp = repo / "Completeness_783.csv"
conn = repo / "Connectivity_783.parquet"
df_comp = pd.read_csv(comp, index_col=0)
df_con = pd.read_parquet(conn)
flyid2idx = {fid: i for i, fid in enumerate(df_comp.index)}

drive_ids = neurons.SUGAR_WATER_GRN
exc = [flyid2idx[f] for f in drive_ids if f in flyid2idx]
print(f"driving {len(exc)}/{len(drive_ids)} sugar/water GRNs (rest not in this completeness set)")

print(f"\n=== Brian (runtime, dt={dt_ms}ms), {DURATION_MS}ms, v783 ===")
defaultclock.dt = dt_ms * ms
t0 = time.time()
neu, syn, spk = create_model(comp, conn, params)
pois, neu = poi(neu, exc, [], params)
net = Network(neu, syn, spk, *pois)
build_t = time.time() - t0
t0 = time.time()
net.run(DURATION_MS * ms)
brian_wall = time.time() - t0
brian_counts = pd.Series(spk.count[:], index=df_comp.index)
print(f"build: {build_t:.1f}s  run wall: {brian_wall:.1f}s  total spikes: {spk.num_spikes:,}  "
      f"neurons fired: {(brian_counts > 0).sum():,}")
print("top 10:\n" + brian_counts.sort_values(ascending=False).head(10).to_string())

print(f"\n=== FastBrain (numba, dt={dt_ms}ms), {DURATION_MS}ms, v783 ===")
t0 = time.time()
brain = FastBrain(df_comp, df_con, params, dt_ms, exc, seed=1)
build_t = time.time() - t0
brain.run_ms(10)  # JIT warm-up, excluded from the reported speed
t0 = time.time()
times, fneurons = brain.run_ms(DURATION_MS - 10)
fast_wall = time.time() - t0
fast_counts = pd.Series(np.bincount(fneurons, minlength=len(df_comp)), index=df_comp.index)
print(f"build: {build_t:.1f}s  run wall (warm, {DURATION_MS-10}ms sim): {fast_wall:.1f}s  "
      f"total spikes: {len(fneurons):,}  neurons fired: {(fast_counts > 0).sum():,}")
print("top 10:\n" + fast_counts.sort_values(ascending=False).head(10).to_string())
per_100ms = fast_wall / (DURATION_MS - 10) * 100 * 1000
print(f"-> {per_100ms:.1f} ms per 100ms slice (warm)")

print(f"\n=== comparison ===")
brian_active = set(brian_counts[brian_counts > 0].index)
fast_active = set(fast_counts[fast_counts > 0].index)
overlap = brian_active & fast_active
print(f"active-neuron overlap: {len(overlap)} / brian={len(brian_active)} fast={len(fast_active)} "
      f"({100*len(overlap)/max(len(brian_active),1):.0f}%)")
print(f"total spikes ratio (fast/brian): {len(fneurons) / spk.num_spikes:.2f}")
