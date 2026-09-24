"""Phase 3: validate fast_sim.FastBrain against Brian's own output.

Independent Poisson streams mean exact spike-for-spike matching isn't
expected (or meaningful) between the two -- what matters is that the
aggregate behavior agrees: similar total spike volume, a similar number of
neurons ever firing, comparable MN9 (readout) activity, and substantial
overlap in WHICH neurons dominate. This is deliberately not a tight
statistical test (no repeated-trial confidence interval) -- it's a sanity
check that the reimplementation isn't wildly wrong, given the size of the
network and the time budget for this pass.

Usage: python brain/validate_fast_sim.py [dt_ms]
"""
import sys, time
from pathlib import Path

repo = Path(__file__).parent / "Drosophila_brain_model"
sys.path.insert(0, str(repo))

import numpy as np
import pandas as pd
from brian2 import Network, ms, defaultclock
from model import create_model, poi, default_params as params

sys.path.insert(0, str(Path(__file__).parent))
from fast_sim import FastBrain

SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = 720575940660219265

dt_ms = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1

comp = repo / "2023_03_23_completeness_630_final.csv"
conn = repo / "2023_03_23_connectivity_630_final.parquet"
df_comp = pd.read_csv(comp, index_col=0)
df_con = pd.read_parquet(conn)
flyid2idx = {fid: i for i, fid in enumerate(df_comp.index)}
exc = [flyid2idx[f] for f in SUGAR]
mn9_idx = flyid2idx[MN9]

DURATION_MS = 1000

print(f"=== Brian (runtime, dt={dt_ms}ms), {DURATION_MS}ms ===")
defaultclock.dt = dt_ms * ms
neu, syn, spk = create_model(comp, conn, params)
pois, neu = poi(neu, exc, [], params)
net = Network(neu, syn, spk, *pois)
t0 = time.time()
net.run(DURATION_MS * ms)
brian_wall = time.time() - t0
brian_counts = pd.Series(spk.count[:], index=df_comp.index)
print(f"wall time: {brian_wall:.1f}s  total spikes: {spk.num_spikes:,}  "
      f"neurons fired: {(brian_counts > 0).sum():,}  MN9 spikes: {brian_counts[MN9]}")
print("top 10:\n" + brian_counts.sort_values(ascending=False).head(10).to_string())

print(f"\n=== FastBrain (numba, dt={dt_ms}ms), {DURATION_MS}ms ===")
brain = FastBrain(df_comp, df_con, params, dt_ms, exc, seed=1)
t0 = time.time()
times, neurons = brain.run_ms(DURATION_MS)
fast_wall = time.time() - t0
fast_counts = pd.Series(np.bincount(neurons, minlength=len(df_comp)), index=df_comp.index)
print(f"wall time: {fast_wall:.1f}s  total spikes: {len(neurons):,}  "
      f"neurons fired: {(fast_counts > 0).sum():,}  MN9 spikes: {fast_counts.iloc[mn9_idx]}")
print("top 10:\n" + fast_counts.sort_values(ascending=False).head(10).to_string())

print(f"\n=== comparison ===")
print(f"speedup: {brian_wall / fast_wall:.1f}x")
brian_active = set(brian_counts[brian_counts > 0].index)
fast_active = set(fast_counts[fast_counts > 0].index)
overlap = brian_active & fast_active
print(f"active-neuron overlap: {len(overlap)} / brian={len(brian_active)} fast={len(fast_active)} "
      f"({100*len(overlap)/max(len(brian_active),1):.0f}% of Brian's active set also active in FastBrain)")
print(f"total spikes ratio (fast/brian): {len(neurons) / spk.num_spikes:.2f}")
