"""Scene fly v3: the v3 senses (scene3.py: what is approaching, from where,
how soon) through the fly's brain, and a linear readout of 117 identified
descending neurons trained to reproduce the v3 teacher's drives
(train_senses3.py). The teacher reads the same senses directly; the fly only
gets them as visual-population firing, through the connectome.

  python training/moonlord/scene_fly3.py collect --engine np --null none --fights 10
  python training/moonlord/scene_fly3.py dagger <fly fight logs...> --out logs/moonlord/scene3_dagger.npz --append
  python training/moonlord/scene_fly3.py fit --data logs/moonlord/scene3_np_none.npz
  python training/moonlord/scene_fly3.py eval --params training/moonlord/params/scene3_np_none.json --fights 6
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT / "brain", ROOT / "interface"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import scene3  # noqa: E402

SCENE = json.loads((ROOT / "brain" / "scene_neurons.json").read_text())
ALL_DNS = SCENE["all_dns"]
TEACHER = HERE / "params" / "senses3_teacher.json"


def populations():
    import neurons
    pops = {("LC4", -1): neurons.LC4_LOOMING_LEFT, ("LC4", 1): neurons.LC4_LOOMING_RIGHT,
            ("LPLC4", -1): neurons.LPLC4_LOOMING_LEFT, ("LPLC4", 1): neurons.LPLC4_LOOMING_RIGHT}
    for t in ("LC6", "LC11", "LC12", "LPLC2", "LC22", "VS", "HS"):
        pops[(t, -1)], pops[(t, 1)] = SCENE[t]["left"], SCENE[t]["right"]
    return pops


def arena():
    import fly_params
    return fly_params.arena_px()


class SceneFly3Policy(scene3.V3Policy):
    """The fly: senses -> brain (100 ms) -> readout DNs -> linear drives ->
    the same move rule as the teacher (hold / override / climb-dive).
    Credited "brain"."""
    name = "fly"

    def __init__(self, readout, brain=None, readout_fn=None, arena_px=None):
        n = len(scene3.CHANNELS) + 3
        super().__init__([0.0] * (2 * n) + [readout["thr"], readout["hold"] / 20.0, readout["override"] / 2.0],
                         arena_px or arena())
        self.readout = readout
        self.brain = brain
        self.pops = populations()
        self.dn_ids = readout.get("dn_ids") or [d["id"] for d in SCENE["readout_dns_v3"]]
        self.last_spikes = None

    def _ensure_brain(self):
        if self.brain is None:
            sys.path.insert(0, str(ROOT / "brain" / "Drosophila_brain_model"))
            import run_fly  # noqa: E402  (Mac: numba FastBrain)
            self.brain = run_fly.build_brain()
            self.brain.stimulate([], [], duration_ms=10)

    def brain_step(self, state):
        self._ensure_brain()
        ids, rates = [], []
        for k, r in scene3.scene_rates_v3(state, self.arena).items():
            if r > 0:
                ids += list(self.pops[k])
                rates += [r] * len(self.pops[k])
        self.last_spikes = self.brain.stimulate(ids, rates, duration_ms=100)
        return np.array([self.brain.rate_for(self.last_spikes, f) for f in self.dn_ids])

    def drives(self, state):
        f = np.concatenate([self.brain_step(state) / 100.0, [1.0]])
        return float(f @ np.asarray(self.readout["wh"])), float(f @ np.asarray(self.readout["wv"]))

    def act(self, state):
        a, src = super().act(state)
        return a, {k: "brain" for k in src}


def _teacher():
    return scene3.V3Policy(json.loads(Path(_OPTS.get("teacher", TEACHER)).read_text())["weights"], arena())


_OPTS = {}   # per-worker options (teacher file, master) set by _set_opts


def _set_opts(opts):
    _OPTS.update(opts)


def _fake(seed, randomize):
    from fake_bridge import FakeMoonLordBridge
    from train_fly_cma import randomized
    cfg = {"seed": seed, **randomized({"difficulty": "master"} if _OPTS.get("master") else {}, seed, randomize)}
    return FakeMoonLordBridge(cfg)


def _record(brain, st, X):
    fly = _record.fly
    fly.brain_step(st)
    X.append(np.array([brain.rate_for(fly.last_spikes, f) for f in ALL_DNS]))


def _collect_job(job):
    engine, null, seeds, randomize, logs = job
    import harness
    from brain_backend import make_brain
    from fake_bridge import FakeMoonLordBridge
    from train_fly_cma import randomized
    brain, _ = make_brain(engine, null, seed=0)
    teacher = _teacher()
    _record.fly = SceneFly3Policy({"wh": [], "wv": [], "thr": 0, "hold": 0, "override": 0}, brain=brain)
    X, H, V = [], [], []

    def label(st):
        p = st.get("player") or {}
        if not st.get("in_world") or p.get("dead") or "cx" not in p:
            return
        h, v = teacher.drives(st)
        _record(brain, st, X)
        H.append(h)
        V.append(v)

    if logs:   # DAgger: the fly's own logged states
        for path in logs:
            brain.reset_state(seed=0)
            for line in open(path):
                rec = json.loads(line)
                if rec.get("phase") == "play" and rec.get("state"):
                    label(rec["state"])
    else:      # the teacher's own fights in the fake
        lo = json.loads((HERE / "loadout.json").read_text())
        cfg = harness.config_from_loadout(lo, "fight", "balanced")
        cfg["max_seconds"] = {"fight": 180.0}

        class Recorder(scene3.V3Policy):
            def act(self, state):
                label(state)
                return super().act(state)

        for seed in seeds:
            brain.reset_state(seed=seed)
            harness.run_episode(_fake(seed, randomize), Recorder(teacher.w, teacher.arena), cfg, seed)
    return np.array(X), np.array(H), np.array(V)


def _run_jobs(jobs, workers):
    with mp.get_context("spawn").Pool(workers, _set_opts, (dict(_OPTS),)) as pool:
        res = [r for r in pool.map(_collect_job, jobs) if len(r[0])]
    return (np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res]),
            np.concatenate([r[2] for r in res]))


def collect(args):
    seeds = list(range(3000, 3000 + args.fights))
    t0 = time.time()
    X, H, V = _run_jobs([(args.engine, args.null, seeds[i::args.workers], args.randomize, None)
                         for i in range(args.workers)], args.workers)
    out = ROOT / "logs" / "moonlord" / f"scene3_{args.engine}_{args.null}{'_' + Path(args.teacher).stem if Path(args.teacher) != TEACHER else ''}.npz"
    np.savez(out, X=X, H=H, V=V, n_teacher=len(X))
    print(f"{len(X)} ticks [{time.time() - t0:.0f}s] -> {out.relative_to(ROOT)}")


def dagger(args):
    paths = [str(Path(p).resolve()) for p in args.logs]
    t0 = time.time()
    X, H, V = _run_jobs([(args.engine, args.null, [], 0.0, paths[i::args.workers]) for i in range(args.workers)],
                        args.workers)
    out = Path(args.out)
    n_teacher = len(X)
    if args.append and out.exists():
        d = np.load(out)
        n_teacher = int(d["n_teacher"])
        X, H, V = np.concatenate([d["X"], X]), np.concatenate([d["H"], H]), np.concatenate([d["V"], V])
    np.savez(out, X=X, H=H, V=V, n_teacher=n_teacher)
    print(f"{len(X)} ticks ({len(X) - n_teacher} from the fly's own fights) [{time.time() - t0:.0f}s] -> {out}")


def fit(args):
    d = np.load(args.data)
    X, H, V = d["X"], d["H"], d["V"]
    n_teacher = int(d["n_teacher"]) if "n_teacher" in d else len(X)
    want = {x["id"] for x in SCENE["readout_dns_v3"]}
    cols = [i for i, f in enumerate(ALL_DNS) if int(f) in want]
    F = np.column_stack([X[:, cols] / 100.0, np.ones(len(X))])
    n = len(F)
    idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    sw = np.where(np.arange(n) >= n_teacher, args.dagger_weight, 1.0)
    Fw = F[tr] * sw[tr, None]
    A = Fw.T @ F[tr] + args.ridge * np.eye(F.shape[1])
    wh, wv = np.linalg.solve(A, Fw.T @ H[tr]), np.linalg.solve(A, Fw.T @ V[tr])
    r2 = lambda w, y: float(1 - np.mean((F[te] @ w - y[te]) ** 2) / np.var(y[te]))
    print(f"readout {len(cols)} DNs, active {(X[:, cols].mean(0) > 1).sum()}; held-out R^2 sideways {r2(wh, H):.3f}, "
          f"vertical {r2(wv, V):.3f}")
    tw = json.loads(Path(args.teacher).read_text())["weights"]
    n2 = 2 * (len(scene3.CHANNELS) + 3)
    out = (Path(args.out) if args.out else HERE / "params" / f"{Path(args.data).stem}.json").resolve()
    out.write_text(json.dumps({"readout": {"wh": wh.tolist(), "wv": wv.tolist(), "thr": abs(tw[n2]),
                                           "hold": abs(tw[n2 + 1]) * 20, "override": abs(tw[n2 + 2]) * 2,
                                           "dn_ids": [int(ALL_DNS[i]) for i in cols]},
                               "meta": {"what": "scene fly v3 (scene_fly3.py)", "data": Path(args.data).name,
                                        "teacher": Path(args.teacher).name,
                                        "r2_sideways": r2(wh, H), "r2_vertical": r2(wv, V),
                                        "finished": time.strftime("%Y-%m-%d %H:%M:%S")}}, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


def _eval_job(job):
    engine, null, params_path, seeds, randomize = job
    import harness
    from brain_backend import make_brain
    from fake_bridge import FakeMoonLordBridge
    from train_fly_cma import randomized
    brain, _ = make_brain(engine, null, seed=0)
    ro = json.loads(Path(params_path).read_text())["readout"]
    lo = json.loads((HERE / "loadout.json").read_text())
    cfg = harness.config_from_loadout(lo, "fight", "balanced")
    cfg["max_seconds"] = {"fight": 180.0}
    cfg["max_seconds"] = {"fight": _OPTS.get("fight_seconds", 180.0)}
    rows = []
    for seed in seeds:
        brain.reset_state(seed=seed)
        b = _fake(seed, randomize)
        r = harness.run_episode(b, SceneFly3Policy(ro, brain=brain), cfg, seed)
        mins = max(r["seconds"], 1.0) / 60.0
        rows.append((r["seconds"], 1 - r["boss_parts_frac_left"], b.damage_by.get("eye", 0) / mins,
                     (sum(b.damage_by.values()) - b.damage_by.get("eye", 0)) / mins, 60 * r["hits"] / max(r["seconds"], 1),
                     b.damage_by.get("deathray", 0) / mins, sum(b.damage_by.values()) / mins))
    return rows


def evaluate(args):
    seeds = list(range(5000, 5000 + args.fights))
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, _set_opts, (dict(_OPTS),)) as pool:
        rows = [r for part in pool.map(_eval_job, [(args.engine, args.null, args.params, seeds[i::args.workers],
                                                     args.randomize) for i in range(args.workers)]) for r in part]
    secs = sorted(r[0] for r in rows)
    print(f"{Path(args.params).stem} on {args.engine}/{args.null}{' MASTER' if _OPTS.get('master') else ''}: "
          f"median survival {secs[len(secs) // 2]:.1f}s, boss damage {100 * np.mean([r[1] for r in rows]):.1f}%, "
          f"eye dmg/min {np.mean([r[2] for r in rows]):.0f}, other dmg/min {np.mean([r[3] for r in rows]):.0f}, "
          f"hits/min {np.mean([r[4] for r in rows]):.1f}, deathray dmg/min {np.mean([r[5] for r in rows]):.0f}, "
          f"total dmg/min {np.mean([r[6] for r in rows]):.0f}  ({len(rows)} fights, {time.time() - t0:.0f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "eval", "dagger"):
        s = sub.add_parser(name)
        s.add_argument("--engine", default="np", choices=["fast", "np"])
        s.add_argument("--null", default="none", choices=["none", "shuffled", "ablated"])
        s.add_argument("--workers", type=int, default=2)
        if name != "dagger":
            s.add_argument("--fights", type=int, default=10)
            s.add_argument("--randomize", type=float, default=0.15)
        if name == "eval":
            s.add_argument("--params", required=True)
            s.add_argument("--fight-seconds", type=float, default=180.0)
        s.add_argument("--teacher", default=str(TEACHER))
        s.add_argument("--master", action="store_true", help="the Master-mode fake (fake_bridge MASTER profile)")
        if name == "dagger":
            s.add_argument("logs", nargs="+")
            s.add_argument("--out", required=True)
            s.add_argument("--append", action="store_true")
    f = sub.add_parser("fit")
    f.add_argument("--data", required=True)
    f.add_argument("--ridge", type=float, default=1.0)
    f.add_argument("--dagger-weight", type=float, default=4.0)
    f.add_argument("--teacher", default=str(TEACHER), help="the teacher whose hold/override/threshold the fly uses")
    f.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd != "fit":
        _OPTS.update(teacher=args.teacher, master=args.master, fight_seconds=getattr(args, "fight_seconds", 180.0))
    {"collect": collect, "dagger": dagger, "fit": fit, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
