"""Phase 3: custom LIF simulator, reimplementing model.py's dynamics for speed.

Why this exists: Brian's runtime (Cython) mode has a large FIXED per-call
overhead per net.run() -- around 320ms regardless of dt (see
an early benchmark) -- which rules it out for a 100ms-slice, 10Hz control
loop. Brian's standalone C++ mode is fast (early benchmark: 15-150ms
depending on dt) but compiles one fixed simulation and can't take new
input between slices, so it can't drive a live loop either.

This reimplements the same equations directly, JIT-compiled with numba to
get native speed with normal per-call Python interop. The key realization
from the standalone benchmark: the dense per-neuron state update (127,400
neurons) is cheap once compiled -- it's not what made standalone fast at
coarse dt vs. fine dt scale so differently from runtime mode; that gap was
Python dispatch overhead, not raw compute. So this uses a dense state
update (matching Brian's own 'linear' analytic integration exactly, not an
approximation) plus sparse, spike-triggered synapse delivery -- the same
algorithmic shape Brian already uses internally, just compiled without
Brian's runtime scheduling overhead.

Equations, from Drosophila_brain_model/model.py's default_params:
    dv/dt = (v_0 - v + g) / t_mbr   (unless refractory)
    dg/dt = -g / tau                 (unless refractory)
    spike when v > v_th (never while refractory) -> v = v_rst, g = 0,
        refractory for t_rfc
    synapse: on presynaptic spike, g += w on each postsynaptic neuron,
        after a FIXED delay t_dly (one global delay, not per-synapse --
        confirmed from model.py: Synapses(..., delay=params['t_dly']))
    "exc"/sugar Poisson drive: Poisson events at rate r_poi, each adding
        w_syn * f_poi directly to v, undelayed (PoissonInput has no delay
        argument in model.py) -- confirmed to be a strong enough kick
        (~69mV against a 7mV v_rst-to-v_th gap) that one event essentially
        forces an immediate spike

The reset string's "w = 0" (in model.py's eq_rst) is dead code: verified by
inspecting the compiled NeuronGroup that 'w' isn't a declared neuron
variable (it's the Synapses object's per-edge weight name, a coincidental
reuse) and running it produces no error or warning -- Brian silently
treats it as an unused local. Omitted here.

NOT modeled (unused by default_params, add if a later phase needs them):
a second Poisson input class (r_poi2) and neuron silencing.

See validate_fast_sim_v783.py for the comparison against Brian's own output.
"""
import math
import numpy as np
import numba
from numba import njit


class FastBrain:
    def __init__(self, df_comp, df_con, params, dt_ms, exc_indices, seed=0):
        """
        df_comp, df_con: same dataframes create_model() takes (completeness
            and connectivity), loaded once by the caller.
        params: the same default_params dict from model.py (Brian Quantity
            values; converted to plain floats in mV/ms here).
        dt_ms: fixed timestep in ms. Coarser = faster but less precise
            refractory/delay timing resolution (see the early dt benchmarks for
            the speed/dt tradeoff measured on the standalone build).
        exc_indices: neuron INDICES (0-based, into df_comp's row order,
            NOT FlyWire IDs) that receive the Poisson "sugar" drive for
            run_ms() -- same role as `exc` in model.py's poi(). This is
            lower-level plumbing for the benchmark/validation scripts,
            which already do their own id->index translation. The public,
            Phase 4-facing API is stimulate(), below, which takes real
            FlyWire IDs and translates internally -- don't pass FlyWire IDs
            here directly, they're ~18-digit numbers that would silently
            read/write far out of bounds (numba doesn't bounds-check
            array indexing; this crashed the process with a segfault the
            first time encoder.py's real FlyWire IDs reached this code
            through the old, single "neuron_ids means indices" contract).
        """
        self.n = len(df_comp)
        self.dt = dt_ms
        self.rng = np.random.default_rng(seed)
        self.flyid2idx = {int(fid): i for i, fid in enumerate(df_comp.index)}

        # --- scalar params, stripped of Brian units (mV, ms) ---
        v0 = float(params['v_0'] / 1e-3)          # volts -> mV
        v_rst = float(params['v_rst'] / 1e-3)
        v_th = float(params['v_th'] / 1e-3)
        t_mbr = float(params['t_mbr'] * 1000)     # seconds -> ms
        tau = float(params['tau'] * 1000)
        t_rfc = float(params['t_rfc'] * 1000)
        t_dly = float(params['t_dly'] * 1000)
        w_syn = float(params['w_syn'] / 1e-3)
        r_poi = float(params['r_poi'])            # Hz
        f_poi = float(params['f_poi'])

        self.v0, self.v_rst, self.v_th = v0, v_rst, v_th
        self.poisson_weight = w_syn * f_poi
        self.poisson_rate_hz = r_poi

        # --- precompute the exact analytic per-dt update coefficients ---
        # Same closed-form solution Brian's method='linear' computes for
        # this 2-state linear system (derived by hand, see module docstring
        # for the equations): with u = v - v0,
        #   g(t)  = g0 * b
        #   u(t)  = u0 * a + g0 * c
        # where a = exp(-dt/t_mbr), b = exp(-dt/tau),
        #       c = (b - a) / (t_mbr * (1/t_mbr - 1/tau))
        a = math.exp(-dt_ms / t_mbr)
        b = math.exp(-dt_ms / tau)
        diff = 1.0 / t_mbr - 1.0 / tau
        c = (b - a) / (t_mbr * diff)
        self.decay_v, self.decay_g, self.coupling_vg = a, b, c

        self.refractory_steps = max(1, round(t_rfc / dt_ms))
        self.delay_steps = max(1, round(t_dly / dt_ms))

        # --- state ---
        self.v = np.full(self.n, v0, dtype=np.float64)
        self.g = np.zeros(self.n, dtype=np.float64)
        self.refractory_until_step = np.full(self.n, -1, dtype=np.int64)
        self.step = 0

        self.exc_indices = np.asarray(exc_indices, dtype=np.int64)

        # ring buffer of pending g deltas, one dense length-n slot per
        # future step within the delay window
        self.delay_buffer = np.zeros((self.delay_steps + 1, self.n), dtype=np.float64)

        # --- sparse connectivity, CSR (grouped by presynaptic neuron) ---
        i_pre = df_con['Presynaptic_Index'].to_numpy()
        i_post = df_con['Postsynaptic_Index'].to_numpy()
        w = (df_con['Excitatory x Connectivity'].to_numpy() * w_syn).astype(np.float64)

        order = np.argsort(i_pre, kind='stable')
        i_pre_sorted = i_pre[order]
        self.post_idx = i_post[order].astype(np.int64)
        self.post_w = w[order]
        self.indptr = np.searchsorted(i_pre_sorted, np.arange(self.n + 1)).astype(np.int64)

        # spike log for this call: (step, neuron_index) pairs
        self._spike_steps = []
        self._spike_neurons = []

        self._v0 = v0  # kept for reset_state()

    def reset_state(self, seed=None):
        """Reset neural dynamics to the resting state (v=v0, g=0, no
        refractory, empty delay buffer, step counter back to 0) WITHOUT
        rebuilding the connectivity structure (CSR arrays) -- that's the
        expensive one-time part, and Phase 5's training loop needs to run
        many short episodes, each starting from a clean brain. seed=None
        keeps the current RNG stream going (still reproducible run-to-run
        if the whole training script itself is seeded); pass an int to
        make this specific reset's Poisson draws reproducible on their own."""
        self.v.fill(self._v0)
        self.g.fill(0.0)
        self.refractory_until_step.fill(-1)
        self.delay_buffer.fill(0.0)
        self.step = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def run_ms(self, duration_ms):
        """Advance the simulation by duration_ms, driving the fixed sugar
        neurons from __init__ at r_poi, and return spikes fired during this
        call as (times_ms, neuron_indices) arrays. Used by the validation/
        benchmark scripts, which need a stimulus matching model.py's own
        smoke test exactly. For the general encoder-facing API, see
        stimulate()."""
        poisson_p = np.full(len(self.exc_indices), self.poisson_rate_hz * (self.dt / 1000.0))
        return self._advance(duration_ms, self.exc_indices, poisson_p)

    def stimulate(self, neuron_ids, rates_hz, duration_ms=100):
        """Phase 3/4's target API (README.md): drive `neuron_ids` -- real
        FlyWire root IDs, e.g. from brain/neurons.py -- with independent
        Poisson processes at the given `rates_hz` (same mechanism as the
        model's sugar drive: each event is a direct kick to v, not g) for
        duration_ms, then return per-neuron spike RATES in Hz for every
        neuron in the network, index-aligned with the completeness
        dataframe this FastBrain was built from (use self.flyid2idx, or
        rate_for(), to look up a specific neuron's rate by FlyWire ID).

        neuron_ids and rates_hz can be empty (no external drive this
        slice) and rates_hz can differ per neuron, unlike run_ms's fixed
        single rate. A FlyWire ID not present in this connectome is
        silently skipped (paired rate dropped with it), not an error.
        """
        indices, kept_rates = [], []
        for fid, rate in zip(neuron_ids, rates_hz):
            idx = self.flyid2idx.get(int(fid))
            if idx is not None:
                indices.append(idx)
                kept_rates.append(rate)
        indices = np.asarray(indices, dtype=np.int64)
        poisson_p = np.asarray(kept_rates, dtype=np.float64) * (self.dt / 1000.0)
        _, spike_neurons = self._advance(duration_ms, indices, poisson_p)
        counts = np.bincount(spike_neurons, minlength=self.n)
        return counts / (duration_ms / 1000.0)

    def rate_for(self, rates, flyid):
        """Look up one neuron's rate (Hz) from stimulate()'s returned array
        by its real FlyWire ID. 0.0 if that ID isn't in this connectome."""
        idx = self.flyid2idx.get(int(flyid))
        return float(rates[idx]) if idx is not None else 0.0

    def _advance(self, duration_ms, drive_indices, drive_poisson_p):
        n_steps = round(duration_ms / self.dt)
        spike_steps, spike_neurons = _run_loop(
            n_steps, self.step, self.n,
            self.v, self.g, self.refractory_until_step,
            self.decay_v, self.decay_g, self.coupling_vg, self.v0, self.v_rst, self.v_th,
            self.refractory_steps, self.delay_steps, self.delay_buffer,
            self.indptr, self.post_idx, self.post_w,
            drive_indices, drive_poisson_p, self.poisson_weight,
            self.rng.random((n_steps, len(drive_indices))),
        )
        self.step += n_steps
        times_ms = (np.asarray(spike_steps, dtype=np.float64)) * self.dt
        return times_ms, np.asarray(spike_neurons, dtype=np.int64)


@njit(cache=True)
def _run_loop(n_steps, step0, n, v, g, refractory_until_step,
              decay_v, decay_g, coupling_vg, v0, v_rst, v_th,
              refractory_steps, delay_steps, delay_buffer,
              indptr, post_idx, post_w,
              exc_indices, poisson_p, poisson_weight, poisson_rand):
    spike_steps = []
    spike_neurons = []
    buflen = delay_buffer.shape[0]

    for local_step in range(n_steps):
        k = step0 + local_step
        slot = k % buflen

        # 1. deliver synaptic deltas scheduled for this step, then clear the slot.
        # Dropped (not deferred) for refractory targets -- confirmed against Brian
        # (an early refractory check) that it suppresses ANY modification of v/g
        # during refractory, not just their own continuous integration; an
        # incoming delta during refractory just vanishes, it doesn't queue.
        for i in range(n):
            if refractory_until_step[i] < k:
                g[i] += delay_buffer[slot, i]
        delay_buffer[slot, :] = 0.0

        # 2. Poisson "sugar" drive: direct kick to v, undelayed. Same
        # refractory suppression as above -- this was the actual bug found
        # by the Brian comparison's 29% excess spike count: applying this
        # unconditionally let a wrongly-inflated v persist into the next
        # non-refractory step, one step after refractory technically ends.
        for j in range(exc_indices.shape[0]):
            idx = exc_indices[j]
            if refractory_until_step[idx] < k and poisson_rand[local_step, j] < poisson_p[j]:
                v[idx] += poisson_weight

        # 3. continuous update for non-refractory neurons (exact analytic step)
        for i in range(n):
            if refractory_until_step[i] < k:
                v[i] = v0 + (v[i] - v0) * decay_v + g[i] * coupling_vg
                g[i] = g[i] * decay_g

        # 4. threshold check + reset + schedule downstream delivery
        for i in range(n):
            if refractory_until_step[i] < k and v[i] > v_th:
                spike_steps.append(k)
                spike_neurons.append(i)
                v[i] = v_rst
                g[i] = 0.0
                refractory_until_step[i] = k + refractory_steps

                target_slot = (k + delay_steps) % buflen
                for e in range(indptr[i], indptr[i + 1]):
                    delay_buffer[target_slot, post_idx[e]] += post_w[e]

    return spike_steps, spike_neurons
