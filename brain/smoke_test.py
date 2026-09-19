"""Phase 0 smoke test for the Shiu et al. fly brain model (FlyWire v630).

Builds the network once, drives the 21 sugar-sensing neurons from the upstream
example, and runs several 100 ms slices, timing each. The per-slice time is
the number Phase 3 depends on: it must be well under 100 ms of wall clock
for a 10 Hz control loop.
"""
import sys, time, platform
from pathlib import Path

repo = Path(__file__).parent / "Drosophila_brain_model"
sys.path.insert(0, str(repo))

import pandas as pd
import brian2
from brian2 import Network, ms, Hz, prefs
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

print(f"python {platform.python_version()} on {platform.platform()}")
print(f"brian2 {brian2.__version__}, codegen target: {prefs.codegen.target}")

comp = repo / "2023_03_23_completeness_630_final.csv"
conn = repo / "2023_03_23_connectivity_630_final.parquet"

t0 = time.time()
df_comp = pd.read_csv(comp, index_col=0)
flyid2idx = {fid: i for i, fid in enumerate(df_comp.index)}
exc = [flyid2idx[f] for f in SUGAR]
neu, syn, spk = create_model(comp, conn, params)
pois, neu = poi(neu, exc, [], params)
net = Network(neu, syn, spk, *pois)
print(f"neurons: {len(neu):,}  synapses: {len(syn):,}  build time: {time.time()-t0:.1f} s")

SLICE = 100 * ms
times = []
for k in range(6):
    t = time.time()
    net.run(SLICE)
    dt = time.time() - t
    times.append(dt)
    print(f"slice {k}: {dt*1000:8.0f} ms wall for 100 ms simulated  (spikes so far: {spk.num_spikes:,})")

steady = times[1:]  # slice 0 includes compile / warm-up
avg = sum(steady) / len(steady)
print(f"\nsteady-state: {avg*1000:.0f} ms wall per 100 ms slice "
      f"-> {'OK for 10 Hz loop' if avg < 0.1 else 'TOO SLOW for 10 Hz, see plan Phase 3 fallbacks'}")

# Which neurons fired most? (sanity check: sugar neurons and downstream feeding circuits should lead)
counts = pd.Series(spk.count[:], index=df_comp.index).sort_values(ascending=False)
print("\ntop 10 neurons by spike count:")
print(counts.head(10).to_string())
print(f"\nneurons that fired at least once: {(counts > 0).sum():,}")
print(f"MN9 spike count: {counts.get(MN9, 'n/a')}")
