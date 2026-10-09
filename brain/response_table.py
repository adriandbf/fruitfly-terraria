"""Brain response table: the connectome's descending-neuron output measured
on a grid of looming inputs, so offline tuning can run ~1000x faster.

In the Moon Lord fight the fly's brain only ever receives four inputs:
LC4 left/right (a threat level or below) and LPLC4 left/right (a threat
from above); the approach (sugar) channel is silent because there's no
other player. Each 100 ms decision is therefore a function of four rates
plus the brain's leftover state and Poisson noise. This script runs a real
engine (np_sim here, fast_sim on the Mac) at every point of a 4-D grid of
those rates -- each point warmed up first, then K separate 100 ms slices
recorded, so the table keeps the brain's own trial-to-trial noise -- and
TableBrain plays it back: multilinear interpolation between grid points,
one recorded sample drawn at random per decision.

What it leaves out: any memory longer than the warm-up (the LIF membrane
time constant is 20 ms, so a 100 ms slice is mostly settled) and the
exact Poisson draws. That's why it's only used to SEARCH; candidates are
re-scored on a real engine (train_fly_cma.py --rescore) and then live.

Build:     python brain/response_table.py build --engine np --null none --workers 2
Validate:  python brain/response_table.py validate --table brain/data/response_table_np_none.npz --points 60
"""
import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import neurons  # noqa: E402

CHANNELS = ["LC4_LOOMING_LEFT", "LC4_LOOMING_RIGHT", "LPLC4_LOOMING_LEFT", "LPLC4_LOOMING_RIGHT"]
LEVELS = [0.0, 15.0, 40.0, 80.0, 130.0, 200.0, 300.0]


def _ids_rates(point):
    ids, rates = [], []
    for attr, r in zip(CHANNELS, point):
        if r > 0:
            pop = getattr(neurons, attr)
            ids += list(pop)
            rates += [r] * len(pop)
    return ids, rates


def _worker(args):
    engine, null, seed, points, warm_ms, k = args
    from brain_backend import make_brain
    brain, _ = make_brain(engine, null, seed=seed)
    dn = list(neurons.ALL_DESCENDING)
    out = []
    for idx, point in points:
        ids, rates = _ids_rates(point)
        if warm_ms:
            brain.stimulate(ids, rates, duration_ms=warm_ms)
        samples = []
        for _ in range(k):
            r = brain.stimulate(ids, rates, duration_ms=100)
            samples.append([brain.rate_for(r, f) for f in dn])
        out.append((idx, samples))
    return out


def build(engine, null, out, levels=LEVELS, k=3, warm_ms=50, workers=2, seed=0):
    from brain_backend import make_brain  # noqa: F401  (fail early on a missing engine)
    if null == "random_mapping":
        from brain_backend import apply_random_mapping
        d = np.load(HERE / "data" / "connectome_783.npz")
        apply_random_mapping(d["ids"], seed)
    L = len(levels)
    grid = list(itertools.product(range(L), repeat=4))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(grid))  # random order: each point follows an unrelated one, like live
    pts = [(grid[i], tuple(levels[j] for j in grid[i])) for i in order]
    chunks = [pts[w::workers] for w in range(workers)]
    # Split further so progress shows up.
    jobs = []
    for w, ch in enumerate(chunks):
        for s in range(0, len(ch), 200):
            jobs.append((engine, null, seed, ch[s:s + 200], warm_ms, k))
    dn = list(neurons.ALL_DESCENDING)
    table = np.zeros((k, L, L, L, L, len(dn)), dtype=np.float32)
    t0, done = time.time(), 0
    # One worker process per job keeps each engine's memory bounded; the
    # engine build (~2 s np, ~1 min fast) is small next to 200 points.
    with mp.get_context("spawn").Pool(workers) as pool:
        for res in pool.imap_unordered(_worker, jobs):
            for idx, samples in res:
                table[(slice(None),) + idx] = np.asarray(samples, dtype=np.float32)
            done += len(res)
            print(f"  {done}/{len(grid)} points  [{time.time() - t0:.0f}s]", flush=True)
    meta = {"engine": engine, "null": null, "null_seed": seed, "levels": list(levels), "k": k,
            "warm_ms": warm_ms, "channels": CHANNELS, "dn_ids": [int(x) for x in dn],
            "built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0)}
    np.savez_compressed(out, table=table, meta=json.dumps(meta))
    print(f"wrote {out}")
    return out


class TableBrain:
    """Drop-in for FastBrain/NpBrain inside FlyPolicy (stimulate/rate_for/
    reset_state), backed by a response table."""

    def __init__(self, path, seed=0):
        d = np.load(path)
        self.table = d["table"]
        self.meta = json.loads(str(d["meta"]))
        self.levels = np.asarray(self.meta["levels"], dtype=np.float64)
        self.dn_ids = self.meta["dn_ids"]
        self.null, self.null_seed = self.meta["null"], self.meta["null_seed"]
        self.all_ids = None
        if self.null == "random_mapping":
            self.all_ids = np.load(HERE / "data" / "connectome_783.npz")["ids"]
        self.col = {f: i for i, f in enumerate(self.dn_ids)}
        self.rng = np.random.default_rng(seed)

    def reset_state(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _channel_rates(self, ids, rates):
        given = dict(zip((int(i) for i in ids), rates))
        out = []
        for attr in CHANNELS:
            pop = getattr(neurons, attr)
            out.append(sum(given.get(int(f), 0.0) for f in pop) / max(len(pop), 1))
        known = set().union(*(set(int(f) for f in getattr(neurons, a)) for a in CHANNELS))
        stray = [i for i, r in given.items() if r > 0 and i not in known]
        if stray:
            raise ValueError(f"TableBrain only models the looming channels; {len(stray)} other inputs driven "
                             "(approach/waypoint channel?) -- use a real engine")
        return out

    def interpolate(self, point, k=None):
        lv = self.levels
        lo, frac = [], []
        for r in point:
            r = min(max(r, lv[0]), lv[-1])
            i = min(int(np.searchsorted(lv, r, side="right")) - 1, len(lv) - 2)
            lo.append(i)
            frac.append((r - lv[i]) / (lv[i + 1] - lv[i]))
        k = self.rng.integers(self.table.shape[0]) if k is None else k
        acc = np.zeros(self.table.shape[-1])
        for corner in itertools.product((0, 1), repeat=4):
            w = 1.0
            for c, f in zip(corner, frac):
                w *= f if c else 1.0 - f
            if w:
                acc += w * self.table[(k,) + tuple(i + c for i, c in zip(lo, corner))]
        return acc

    def stimulate(self, ids, rates, duration_ms=100):
        return self.interpolate(self._channel_rates(ids, rates))

    def rate_for(self, rates, flyid):
        i = self.col.get(int(flyid))
        return float(rates[i]) if i is not None else 0.0

    def readout(self, rates):
        return {fid: self.rate_for(rates, fid) for fid in neurons.ALL_DESCENDING}


def validate(table_path, points=60, engine=None, seed=1):
    """Off-grid check: random input points, real engine vs table, on the
    quantities FlyPolicy actually decides from."""
    tb = TableBrain(table_path, seed=seed)
    from brain_backend import make_brain
    brain, _ = make_brain(engine or tb.meta["engine"], tb.null, seed=seed)
    rng = np.random.default_rng(seed)
    n = neurons
    mean = lambda get, ids: float(np.mean([get(f) for f in ids]))

    def summary(get):
        return {"gf": max(get(f) for f in n.DNP01_JUMP),
                "dnp07": max(mean(get, n.DNP07_LANDING_LEFT), mean(get, n.DNP07_LANDING_RIGHT)),
                "dnp07_side": np.sign(mean(get, n.DNP07_LANDING_RIGHT) - mean(get, n.DNP07_LANDING_LEFT)),
                "dna02": mean(get, n.DNA02_STEERING_LEFT) - mean(get, n.DNA02_STEERING_RIGHT)}

    rows = []
    for _ in range(points):
        # Realistic inputs: one LC4 side and maybe one LPLC4 side, contra a fraction.
        def side_pair():
            if rng.random() < 0.35:
                return [0.0, 0.0]
            r, c = rng.uniform(0, 300), rng.uniform(0, 0.6)
            return [r, r * c] if rng.random() < 0.5 else [r * c, r]
        point = side_pair() + side_pair()
        ids, rates = _ids_rates(point)
        brain.stimulate(ids, rates, duration_ms=50)
        real = brain.stimulate(ids, rates, duration_ms=100)
        again = brain.stimulate(ids, rates, duration_ms=100)  # noise floor: the engine against itself
        tab = tb.stimulate(ids, rates)
        rows.append((point, summary(lambda f: brain.rate_for(real, f)), summary(lambda f: tb.rate_for(tab, f)),
                     summary(lambda f: brain.rate_for(again, f))))
    def compare(a_rows, b_rows):
        g = lambda rs, k: np.array([r[k] for r in rs])
        gf_r, gf_t = g(a_rows, "gf"), g(b_rows, "gf")
        d7_r, d7_t = g(a_rows, "dnp07"), g(b_rows, "dnp07")
        a_r, a_t = g(a_rows, "dna02"), g(b_rows, "dna02")
        corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float("nan")
        side_mask = (np.abs(a_r) > 5) | (np.abs(a_t) > 5)
        return {"gf_corr": round(corr(gf_r, gf_t), 3), "gf_mae_hz": round(float(np.abs(gf_r - gf_t).mean()), 1),
                "dnp07_corr": round(corr(d7_r, d7_t), 3), "dnp07_mae_hz": round(float(np.abs(d7_r - d7_t).mean()), 1),
                "dna02_corr": round(corr(a_r, a_t), 3),
                "flee_on30_agree": round(float(np.mean((gf_r >= 30) == (gf_t >= 30))), 3),
                "dive40_agree": round(float(np.mean(((d7_r >= 40) & (d7_r > gf_r)) == ((d7_t >= 40) & (d7_t > gf_t)))), 3),
                "dna02_side_agree": round(float(np.mean(np.sign(a_r[side_mask]) == np.sign(a_t[side_mask])))
                                          if side_mask.any() else float("nan"), 3)}
    real_rows = [r for _, r, _, _ in rows]
    res = {"points": points, "table_vs_engine": compare(real_rows, [t for _, _, t, _ in rows]),
           "engine_vs_itself": compare(real_rows, [a for _, _, _, a in rows])}
    print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--engine", default="np", choices=["np", "fast"])
    b.add_argument("--null", default="none", choices=["none", "shuffled", "ablated", "random_mapping"])
    b.add_argument("--k", type=int, default=3, help="100 ms samples kept per grid point")
    b.add_argument("--warm-ms", type=int, default=50)
    b.add_argument("--workers", type=int, default=2)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default=None)
    v = sub.add_parser("validate")
    v.add_argument("--table", required=True)
    v.add_argument("--points", type=int, default=60)
    v.add_argument("--engine", default=None, choices=["np", "fast"])
    args = ap.parse_args()
    if args.cmd == "build":
        out = args.out or str(HERE / "data" / f"response_table_{args.engine}_{args.null}.npz")
        build(args.engine, args.null, out, k=args.k, warm_ms=args.warm_ms, workers=args.workers, seed=args.seed)
    else:
        validate(args.table, args.points, args.engine)


if __name__ == "__main__":
    main()
