"""Phase 3: deterministic, single-synapse validation of fast_sim's update math.

The full-network validation (validate_fast_sim.py) showed a real discrepancy
(29% more total spikes, a non-driven neuron exceeding its driven neighbors'
firing rate) that independent-Poisson-stream noise alone shouldn't explain.
This isolates the core per-neuron analytic update from all that noise: feed
Brian and FastBrain's update loop the exact same hand-specified spike input
(via SpikeGeneratorGroup on the Brian side), with one synapse of the model's
real weight and delay, and compare spike timing exactly -- no randomness
anywhere, so any mismatch is a real bug, not statistics.
"""
import sys
from pathlib import Path

repo = Path(__file__).parent / "Drosophila_brain_model"
sys.path.insert(0, str(repo))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from brian2 import (NeuronGroup, Synapses, SpikeGeneratorGroup, SpikeMonitor,
                     StateMonitor, Network, ms, mV, defaultclock)
from model import default_params as params

DT_MS = 0.1
defaultclock.dt = DT_MS * ms

# --- Brian: 1 driver (forced spikes) -> 1 receiver, single real synapse ---
driver_spike_times = np.array([1.0, 3.0, 5.0, 5.2, 20.0]) * ms
driver = SpikeGeneratorGroup(1, [0, 0, 0, 0, 0], driver_spike_times)

receiver = NeuronGroup(
    1, model=params['eqs'], method='linear',
    threshold=params['eq_th'], reset=params['eq_rst'],
    refractory='rfc', namespace=params,
)
receiver.v = params['v_0']
receiver.g = 0
receiver.rfc = params['t_rfc']

syn = Synapses(driver, receiver, 'w : volt', on_pre='g += w', delay=params['t_dly'])
syn.connect(i=0, j=0)
syn.w = params['w_syn']  # single unit of connectivity, matching the real model's per-edge weight formula

v_mon = StateMonitor(receiver, ['v', 'g'], record=True)
spk_mon = SpikeMonitor(receiver)
net = Network(driver, receiver, syn, v_mon, spk_mon)
net.run(30 * ms)

brian_spike_times = np.array(spk_mon.t / ms)
print("=== Brian ===")
print("receiver spike times (ms):", brian_spike_times)
print("v trace at t=0,1,1.5,2,3,5,6,10,20,21ms:")
sample_ts = [0, 1, 1.5, 2, 3, 5, 6, 10, 20, 21]
for st in sample_ts:
    idx = int(round(st / DT_MS))
    if idx < v_mon.v.shape[1]:
        print(f"  t={st:5.1f}ms  v={float(v_mon.v[0, idx]/mV):8.3f}mV  g={float(v_mon.g[0, idx]/mV):8.3f}mV")

# --- FastBrain: same scenario, hand-built (no full network machinery needed) ---
print("\n=== FastBrain-equivalent (manual, same math as fast_sim.py) ===")
import math
v0 = float(params['v_0'] / mV)
v_rst = float(params['v_rst'] / mV)
v_th = float(params['v_th'] / mV)
t_mbr = float(params['t_mbr'] / ms)
tau = float(params['tau'] / ms)
t_rfc = float(params['t_rfc'] / ms)
t_dly = float(params['t_dly'] / ms)
w_syn = float(params['w_syn'] / mV)

a = math.exp(-DT_MS / t_mbr)
b = math.exp(-DT_MS / tau)
diff = 1.0 / t_mbr - 1.0 / tau
c = (b - a) / (t_mbr * diff)
refractory_steps = max(1, round(t_rfc / DT_MS))
delay_steps = max(1, round(t_dly / DT_MS))

n_steps = round(30 / DT_MS)
driver_steps = set(round(float(t) / DT_MS) for t in driver_spike_times / ms)
delay_buffer = np.zeros(n_steps + delay_steps + 2)
v = v0
g = 0.0
refractory_until = -1
spikes = []
trace = {}
for k in range(n_steps):
    g += delay_buffer[k]
    if k in driver_steps:
        delay_buffer[k + delay_steps] += w_syn
    if refractory_until < k:
        v = v0 + (v - v0) * a + g * c
        g = g * b
    t_ms = k * DT_MS
    if any(abs(t_ms - st) < DT_MS / 2 for st in sample_ts):
        trace[round(t_ms, 3)] = (v, g)
    if refractory_until < k and v > v_th:
        spikes.append(t_ms)
        v = v_rst
        g = 0.0
        refractory_until = k + refractory_steps

print("receiver spike times (ms):", spikes)
for st in sample_ts:
    for t_ms, (vv, gg) in trace.items():
        if abs(t_ms - st) < DT_MS / 2:
            print(f"  t={st:5.1f}ms  v={vv:8.3f}mV  g={gg:8.3f}mV")
            break
