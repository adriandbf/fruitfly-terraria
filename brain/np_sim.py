"""Pure-numpy port of fast_sim.FastBrain, for running the connectome where
numba/pyarrow aren't installed (e.g. a minimal analysis environment).
Same equations, same order of operations per 1 ms step as fast_sim's
_run_loop (deliver delayed synaptic input -> Poisson kicks -> exact
analytic membrane/conductance step -> threshold, reset, schedule
delivery), all with fast_sim's refractory rules. Vectorised per step
instead of per neuron, so it's slower than numba but fine for offline
sweeps (~seconds per simulated second).

Loads the plain-numpy export from brain/export_connectome_npz.py, so it
needs no pandas/pyarrow. Parameters are model.py's default_params,
written out as plain numbers (mV, ms) to avoid importing Brian.

Not bit-identical to fast_sim (different RNG draw order), so compare
statistically -- see validate_np_sim.py.
"""
import math

import numpy as np

# model.py default_params, in mV / ms / Hz
V0, V_RST, V_TH = -52.0, -52.0, -45.0
T_MBR, TAU, T_RFC, T_DLY = 20.0, 5.0, 2.2, 1.8
W_SYN, F_POI = 0.275, 250.0


class NpBrain:
    def __init__(self, npz_path, dt_ms=1.0, seed=0):
        d = np.load(npz_path)
        self.ids = d["ids"]
        self.n = len(self.ids)
        self.flyid2idx = {int(f): i for i, f in enumerate(self.ids)}
        self.dt = dt_ms
        self.rng = np.random.default_rng(seed)
        pre, post = d["pre"].astype(np.int64), d["post"].astype(np.int64)
        w = d["w"].astype(np.float64) * W_SYN
        order = np.argsort(pre, kind="stable")
        self.post_idx, self.post_w = post[order], w[order]
        self.indptr = np.searchsorted(pre[order], np.arange(self.n + 1))

        a = math.exp(-dt_ms / T_MBR)
        b = math.exp(-dt_ms / TAU)
        c = (b - a) / (T_MBR * (1.0 / T_MBR - 1.0 / TAU))
        self.decay_v, self.decay_g, self.coupling = a, b, c
        self.refractory_steps = max(1, round(T_RFC / dt_ms))
        self.delay_steps = max(1, round(T_DLY / dt_ms))
        self.poisson_weight = W_SYN * F_POI
        self.reset_state()

    def reset_state(self, seed=None):
        self.v = np.full(self.n, V0)
        self.g = np.zeros(self.n)
        self.refr_until = np.full(self.n, -1, dtype=np.int64)
        self.buf = np.zeros((self.delay_steps + 1, self.n))
        self.step = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def stimulate(self, neuron_ids, rates_hz, duration_ms=100):
        """Same contract as FastBrain.stimulate: Poisson-drive the given
        FlyWire ids, return per-neuron spike rates (Hz) for this slice."""
        idx, p = [], []
        for fid, r in zip(neuron_ids, rates_hz):
            i = self.flyid2idx.get(int(fid))
            if i is not None:
                idx.append(i)
                p.append(r * self.dt / 1000.0)
        idx = np.asarray(idx, dtype=np.int64)
        p = np.asarray(p)
        counts = np.zeros(self.n)
        buflen = self.buf.shape[0]
        n_steps = round(duration_ms / self.dt)
        for _ in range(n_steps):
            k = self.step
            slot = k % buflen
            free = self.refr_until < k
            self.g[free] += self.buf[slot, free]
            self.buf[slot, :] = 0.0
            if len(idx):
                kick = (self.rng.random(len(idx)) < p) & free[idx]
                self.v[idx[kick]] += self.poisson_weight
            self.v[free] = V0 + (self.v[free] - V0) * self.decay_v + self.g[free] * self.coupling
            self.g[free] *= self.decay_g
            spk = np.flatnonzero(free & (self.v > V_TH))
            if len(spk):
                counts[spk] += 1
                self.v[spk] = V_RST
                self.g[spk] = 0.0
                self.refr_until[spk] = k + self.refractory_steps
                starts, ends = self.indptr[spk], self.indptr[spk + 1]
                lens = ends - starts
                if lens.sum():
                    edges = np.repeat(ends - lens.cumsum(), lens) + np.arange(lens.sum())
                    target = (k + self.delay_steps) % buflen
                    self.buf[target] += np.bincount(self.post_idx[edges], weights=self.post_w[edges],
                                                    minlength=self.n)
            self.step += 1
        return counts / (duration_ms / 1000.0)

    def rate_for(self, rates, flyid):
        i = self.flyid2idx.get(int(flyid))
        return float(rates[i]) if i is not None else 0.0
