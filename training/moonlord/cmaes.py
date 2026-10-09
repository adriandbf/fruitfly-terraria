"""Minimal CMA-ES (Hansen's standard (mu/mu_w, lambda) algorithm with rank-one
and rank-mu updates), used when the `cma` package isn't installed. Same
ask/tell shape as cma.CMAEvolutionStrategy, box-bounded to [0, 1] by
clipping the evaluated point (the unclipped sample still drives the update,
plus a small penalty for distance outside the box)."""
import math

import numpy as np


class CMAES:
    def __init__(self, x0, sigma, popsize=None, seed=0):
        self.n = n = len(x0)
        self.mean = np.asarray(x0, dtype=float)
        self.sigma = float(sigma)
        self.lam = popsize or 4 + int(3 * math.log(n))
        self.mu = self.lam // 2
        w = math.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / (self.w ** 2).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.C = np.eye(n)
        self.B, self.D = np.eye(n), np.ones(n)
        self.rng = np.random.default_rng(seed)
        self.gen = 0
        self._samples = None

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        y = z @ (self.B * self.D).T
        self._samples = self.mean + self.sigma * y
        return [np.clip(x, 0.0, 1.0) for x in self._samples]

    @staticmethod
    def penalty(x):
        return float(((x - np.clip(x, 0, 1)) ** 2).sum())

    def tell(self, fitness):
        """fitness: list, lower is better, in ask() order."""
        xs = self._samples
        f = np.asarray(fitness, dtype=float) + np.array([self.penalty(x) for x in xs])
        idx = np.argsort(f)[:self.mu]
        old = self.mean
        sel = xs[idx]
        self.mean = self.w @ sel
        y = (self.mean - old) / self.sigma
        inv_sqrt = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * inv_sqrt @ y
        self.gen += 1
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chin
                < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        ys = (sel - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (not hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (ys.T * self.w) @ ys)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1))
        self.sigma = min(self.sigma, 1.0)
        self.C = (self.C + self.C.T) / 2
        d2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(d2, 1e-20))
