"""Phase 3 speed experiment: does a coarser timestep get us under 100ms?

Brian's default dt=0.1ms means every one of the 127,400 neurons gets a
state update 1000 times per 100ms slice, even though only ~380 ever spike.
That dense per-timestep cost is the likely bottleneck (not the sparse
event delivery, which Brian already handles via spike queues). Before
writing a custom event-driven simulator, check the cheap lever: does
raising dt buy real speed, and does it still produce a similar firing
pattern (same neurons dominating) as the dt=0.1ms result in
smoke_test.py's log?

Uses standalone mode (fastest, no OpenMP — see bench_standalone.py) at a
few dt values and reports wall time per 100ms slice plus the top-10 by
spike count, to eyeball whether coarsening dt changes WHICH neurons fire,
not just how fast.

Usage: python brain/bench_dt_sweep.py DT_MS [N_SLICES]
Example: python brain/bench_dt_sweep.py 0.5 10
"""
import sys, time, shutil
from pathlib import Path

repo = Path(__file__).parent / "Drosophila_brain_model"
sys.path.insert(0, str(repo))

import pandas as pd
from brian2 import set_device, device, prefs, ms, defaultclock, Network
from model import create_model, poi, default_params as params

SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = 720575940660219265

dt_ms = float(sys.argv[1])
n_slices = int(sys.argv[2]) if len(sys.argv) > 2 else 10

build_dir = Path(__file__).parent / f"standalone_build_dt{dt_ms}"
shutil.rmtree(build_dir, ignore_errors=True)

flags = ["-w", "-O3", "-ffast-math", "-fno-finite-math-only"]
prefs.codegen.cpp.extra_compile_args_gcc = flags
set_device("cpp_standalone", directory=str(build_dir), build_on_run=False)
defaultclock.dt = dt_ms * ms

comp = repo / "2023_03_23_completeness_630_final.csv"
conn = repo / "2023_03_23_connectivity_630_final.parquet"
df_comp = pd.read_csv(comp, index_col=0)
flyid2idx = {fid: i for i, fid in enumerate(df_comp.index)}
exc = [flyid2idx[f] for f in SUGAR]

neu, syn, spk = create_model(comp, conn, params)
pois, neu = poi(neu, exc, [], params)
net = Network(neu, syn, spk, *pois)
net.run(n_slices * 100 * ms)

t0 = time.time()
device.build(directory=str(build_dir), compile=True, run=False, debug=False)
t_compile = time.time() - t0
t0 = time.time()
device.run(directory=str(build_dir))
t_total = time.time() - t0

sim = device._last_run_time
per_slice = sim / n_slices * 1000
counts = pd.Series(spk.count[:], index=df_comp.index).sort_values(ascending=False)
print(f"\ndt={dt_ms}ms  compile={t_compile:.0f}s  process_total={t_total:.1f}s  "
      f"sim_loop={sim:.2f}s -> {per_slice:.1f} ms per 100 ms slice "
      f"({'OK' if per_slice < 100 else 'still too slow'} for 10 Hz)")
print(f"total spikes: {spk.num_spikes:,}  neurons that fired: {(counts > 0).sum():,}  "
      f"MN9 spikes: {counts.get(MN9, 'n/a')}")
print("top 10 by spike count:")
print(counts.head(10).to_string())
