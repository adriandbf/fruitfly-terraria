"""Phase 3: does coarsening dt also speed up RUNTIME (Cython) mode?

Standalone mode got fast with a coarser dt (bench_dt_sweep.py), but it
can't take new input between slices, so it can never be the live loop by
itself. Runtime mode (net.run() called repeatedly, like smoke_test.py) can
take new input each call — this checks whether it gets the same dt
speedup, which would settle Phase 3 without a custom simulator.

Usage: python brain/bench_runtime_dt.py DT_MS [N_SLICES]
"""
import sys, time
from pathlib import Path

repo = Path(__file__).parent / "Drosophila_brain_model"
sys.path.insert(0, str(repo))

import pandas as pd
from brian2 import Network, ms, defaultclock, prefs
from model import create_model, poi, default_params as params

SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]

dt_ms = float(sys.argv[1])
n_slices = int(sys.argv[2]) if len(sys.argv) > 2 else 6
defaultclock.dt = dt_ms * ms

comp = repo / "2023_03_23_completeness_630_final.csv"
conn = repo / "2023_03_23_connectivity_630_final.parquet"
df_comp = pd.read_csv(comp, index_col=0)
flyid2idx = {fid: i for i, fid in enumerate(df_comp.index)}
exc = [flyid2idx[f] for f in SUGAR]

t0 = time.time()
neu, syn, spk = create_model(comp, conn, params)
pois, neu = poi(neu, exc, [], params)
net = Network(neu, syn, spk, *pois)
print(f"dt={dt_ms}ms  codegen target: {prefs.codegen.target}  build time: {time.time()-t0:.1f}s")

times = []
for k in range(n_slices):
    t = time.time()
    net.run(100 * ms)
    dt_wall = time.time() - t
    times.append(dt_wall)
    print(f"  slice {k}: {dt_wall*1000:8.1f} ms wall for 100 ms simulated")

steady = times[1:]
avg = sum(steady) / len(steady)
print(f"steady-state: {avg*1000:.1f} ms per 100 ms slice -> "
      f"{'OK for 10 Hz' if avg < 0.1 else 'still too slow for 10 Hz'}")
