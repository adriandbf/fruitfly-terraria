"""Phase 3 speed benchmark: Brian 2 C++ standalone mode with OpenMP.

Usage: python brain/bench_standalone.py THREADS [N_SLICES]

Standalone mode compiles the whole simulation into one binary, so this only
measures throughput. It cannot change inputs between slices; if it is fast
enough we still have to solve the closed-loop problem separately.
Each thread count needs its own build (the thread count is compiled in).
"""
import sys, time, shutil
from pathlib import Path

repo = Path(__file__).parent / "Drosophila_brain_model"
sys.path.insert(0, str(repo))

import pandas as pd
from brian2 import set_device, device, prefs, ms, defaultclock
from model import create_model, poi, default_params as params

SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]

threads = int(sys.argv[1])
n_slices = int(sys.argv[2]) if len(sys.argv) > 2 else 10

build_dir = Path(__file__).parent / f"standalone_build_{threads}"
shutil.rmtree(build_dir, ignore_errors=True)

omp = "/opt/homebrew/opt/libomp"  # Apple clang has no bundled OpenMP runtime
# Setting extra_compile_args_gcc replaces Brian's defaults, so restate the optimisation flags
# (minus -march=native, which this clang rejects for apple-m*).
flags = ["-w", "-O3", "-ffast-math", "-fno-finite-math-only"]
if threads > 0:
    # Needs CXX=brain/cxx_omp.sh: Brian passes a bare -fopenmp, which Apple clang rejects.
    prefs.devices.cpp_standalone.openmp_threads = threads
    flags += [f"-I{omp}/include"]
    prefs.codegen.cpp.extra_link_args = [f"-L{omp}/lib", "-lomp"]
prefs.codegen.cpp.extra_compile_args_gcc = flags
set_device("cpp_standalone", directory=str(build_dir), build_on_run=False)

comp = repo / "2023_03_23_completeness_630_final.csv"
conn = repo / "2023_03_23_connectivity_630_final.parquet"
df_comp = pd.read_csv(comp, index_col=0)
flyid2idx = {fid: i for i, fid in enumerate(df_comp.index)}
exc = [flyid2idx[f] for f in SUGAR]

neu, syn, spk = create_model(comp, conn, params)
pois, neu = poi(neu, exc, [], params)

from brian2 import Network
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
print(f"threads={threads} compile={t_compile:.0f}s process_total={t_total:.1f}s "
      f"sim_loop={sim:.2f}s -> {per_slice:.0f} ms per 100 ms slice "
      f"({spk.num_spikes:,} spikes, {(pd.Series(spk.count[:]) > 0).sum()} neurons fired)")
