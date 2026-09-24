"""Phase 3: does a direct v-kick (like PoissonInput's target_var='v', which
fast_sim.py's Poisson mechanism mimics) still apply while the neuron is
refractory in Brian? fast_sim currently applies it UNCONDITIONALLY,
regardless of refractory state -- if Brian actually suppresses it, that's
a real bug and a plausible cause of the ~29% excess spike count seen in
validate_fast_sim.py, since a Poisson-driven neuron gets ~150 such kicks/
second and is refractory a meaningful fraction of the time.

Uses a direct v-targeting synapse (on_pre='v += w'), same mechanism as
PoissonInput, to force an instant first spike, then delivers a second kick
timed to land inside the ensuing refractory window.
"""
import sys
from pathlib import Path

repo = Path(__file__).parent / "Drosophila_brain_model"
sys.path.insert(0, str(repo))

import numpy as np
from brian2 import (NeuronGroup, Synapses, SpikeGeneratorGroup, SpikeMonitor,
                     StateMonitor, Network, ms, mV, defaultclock)
from model import default_params as params

DT_MS = 0.1
defaultclock.dt = DT_MS * ms
t_rfc_ms = float(params['t_rfc'] / ms)

# kick 1 at t=1.0ms forces an instant spike (direct v += big weight);
# kick 2 at t=2.0ms lands inside the refractory window that follows
driver = SpikeGeneratorGroup(1, [0, 0], [1.0, 2.0] * ms)

receiver = NeuronGroup(
    1, model=params['eqs'], method='linear',
    threshold=params['eq_th'], reset=params['eq_rst'],
    refractory='rfc', namespace=params,
)
receiver.v = params['v_0']
receiver.g = 0
receiver.rfc = params['t_rfc']

big_weight = params['w_syn'] * params['f_poi']  # same magnitude as the real Poisson kick
syn = Synapses(driver, receiver, 'w : volt', on_pre='v += w', delay=0 * ms)  # direct-to-v, like PoissonInput
syn.connect(i=0, j=0)
syn.w = big_weight

v_mon = StateMonitor(receiver, ['v', 'g'], record=True)
spk_mon = SpikeMonitor(receiver)
net = Network(driver, receiver, syn, v_mon, spk_mon)
net.run(6 * ms)

spike_times = np.array(spk_mon.t / ms)
print("spike times (ms):", spike_times)
print(f"t_rfc = {t_rfc_ms}ms")
if len(spike_times):
    print(f"refractory window after first spike: [{spike_times[0]:.2f}, {spike_times[0]+t_rfc_ms:.2f}]ms")
print("(kick 2 fires at t=2.0ms -- inside that window if the first spike was at/near t=1.0ms)")

print("\nt(ms)   v(mV)      g(mV)")
for k in range(v_mon.v.shape[1]):
    t_ms = round(k * DT_MS, 4)
    if 0.9 <= t_ms <= 3.5:
        print(f"{t_ms:5.1f}  {float(v_mon.v[0,k]/mV):9.4f}  {float(v_mon.g[0,k]/mV):9.4f}")

print("\nIf v jumps by ~68.75mV again right at t=2.0/2.1ms despite being refractory,")
print("Brian applies direct v-kicks even during refractory -> fast_sim's unconditional")
print("kick is correct. If v does NOT jump (stays at v_rst through the window), Brian")
print("suppresses it -> fast_sim needs a refractory guard on the Poisson kick.")
