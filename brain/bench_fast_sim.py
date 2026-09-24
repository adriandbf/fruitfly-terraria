"""Phase 3: steady-state speed of FastBrain, JIT warm-up excluded.

numba compiles _run_loop on its first call, which is a one-time cost (paid
once at process startup, not per control-loop decision) but would badly
distort a naive per-slice timing average. This mirrors smoke_test.py's
"slice 0 is warm-up" approach.

Usage: python brain/bench_fast_sim.py DT_MS [N_SLICES]
"""
import sys, time
from pathlib import Path

repo = Path(__file__).parent / "Drosophila_brain_model"
sys.path.insert(0, str(repo))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from model import default_params as params
from fast_sim import FastBrain

SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]

dt_ms = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
n_slices = int(sys.argv[2]) if len(sys.argv) > 2 else 8

comp = repo / "2023_03_23_completeness_630_final.csv"
conn = repo / "2023_03_23_connectivity_630_final.parquet"
df_comp = pd.read_csv(comp, index_col=0)
df_con = pd.read_parquet(conn)
flyid2idx = {fid: i for i, fid in enumerate(df_comp.index)}
exc = [flyid2idx[f] for f in SUGAR]

t0 = time.time()
brain = FastBrain(df_comp, df_con, params, dt_ms, exc, seed=0)
print(f"dt={dt_ms}ms  build time: {time.time()-t0:.2f}s")

times = []
for k in range(n_slices):
    t = time.time()
    _, neurons = brain.run_ms(100)
    dt_wall = time.time() - t
    times.append(dt_wall)
    tag = " (JIT warm-up)" if k == 0 else ""
    print(f"slice {k}: {dt_wall*1000:8.2f} ms wall for 100 ms simulated "
          f"(spikes this slice: {len(neurons):,}){tag}")

steady = times[1:]
avg = sum(steady) / len(steady)
print(f"\nsteady-state (excluding JIT warm-up): {avg*1000:.2f} ms per 100 ms slice -> "
      f"{'OK for 10 Hz' if avg < 0.1 else 'TOO SLOW for 10 Hz'}")
