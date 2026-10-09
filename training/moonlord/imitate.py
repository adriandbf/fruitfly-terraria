"""Imitation warm start: tune the fly's interface knobs so that, shown the
exact situations from logged LIVE fights, the fly presses what the teacher
pressed (the human demos and the scripted dodger's better fights).

How: replay each teacher fight's logged game states, tick by tick, through
FlyPolicy (encoder -> brain -> decoder, the real pipeline) and compare its
keys with the teacher's on two heads:
  horizontal: left / right / neither
  vertical:   climb (jump or up) / dive (down) / neither
Score = mean of the two heads' BALANCED accuracies (per-class recall
averaged over classes), so "always fly left" can't score well just because
teachers fly left a lot. CMA-ES (same knobs as train_fly_cma.py) maximises
it; the result is a params file to start train_fly_cma.py from (--init).

Limits worth knowing: the replay is open-loop -- the fly sees where the
TEACHER went, not where it would have gone -- so a high score means "reacts
like the teacher in the teacher's situations", not "survives". That's why
this is only a warm start for the closed-loop CMA-ES and live fights.
Teachers' "up" (the fast climb the fly never presses) counts as climb.

Usage:
  python training/moonlord/imitate.py --engine table --generations 25
  python training/moonlord/imitate.py --engine fast --workers 8 --teachers human
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

import fly_params  # noqa: E402

LOG_DIR = ROOT / "logs" / "moonlord"
_W = {}


def teacher_label(action):
    h = -1 if action.get("left") and not action.get("right") else (1 if action.get("right") and not action.get("left") else 0)
    v = 1 if (action.get("jump") or action.get("up")) else (-1 if action.get("down") else 0)
    return h, v


def find_teacher_logs(teachers, min_seconds, results=LOG_DIR / "results.jsonl"):
    """Teacher fights from results.jsonl: chosen policies, live only,
    survived at least min_seconds."""
    out = []
    for line in results.read_text().splitlines():
        r = json.loads(line)
        if r.get("policy") in teachers and not r.get("simulated") and r.get("seconds", 0) >= min_seconds \
                and r.get("log") and (ROOT / r["log"]).exists():
            out.append(ROOT / r["log"])
    return out


def load_episode(path, stride=1):
    """[(state, (h, v))] for the play ticks of one logged fight."""
    ticks = []
    for line in open(path):
        rec = json.loads(line)
        if rec.get("phase") == "play" and rec.get("state") and rec.get("action"):
            ticks.append((rec["state"], teacher_label(rec["action"])))
    return ticks[::stride]


def _init_worker(engine, null, table, brain_seed, episodes):
    from brain_backend import make_brain
    brain, readout = make_brain(engine, null, seed=brain_seed, table_path=table)
    _W.update(brain=brain, readout=readout, episodes=episodes)


def _replay(job):
    """(unit, episode index) -> list of (fly h, fly v, teacher h, teacher v)."""
    unit, ei = job
    from policies import FlyPolicy
    params = fly_params.unit_to_params(unit)
    brain = _W["brain"]
    brain.reset_state(seed=ei)
    pol = FlyPolicy(brain=brain, readout_fn=_W["readout"], **fly_params.policy_kwargs(params))
    pol.reset()
    out = []
    for state, (th, tv) in _W["episodes"][ei]:
        a, _ = pol.act(state)
        fh, fv = teacher_label(a)
        out.append((fh, fv, th, tv))
    return out


def balanced_accuracy(pred, true):
    classes = sorted(set(true))
    recalls = [np.mean([p == c for p, t in zip(pred, true) if t == c]) for c in classes]
    return float(np.mean(recalls)) if recalls else 0.0


def score(rows):
    fh, fv, th, tv = zip(*rows)
    bh, bv = balanced_accuracy(fh, th), balanced_accuracy(fv, tv)
    return {"score": (bh + bv) / 2, "horizontal": bh, "vertical": bv,
            "raw_h": float(np.mean(np.array(fh) == np.array(th))),
            "raw_v": float(np.mean(np.array(fv) == np.array(tv)))}


class Evaluator:
    def __init__(self, args, episodes):
        init = (args.engine, args.null, args.table, args.brain_seed, episodes)
        self.n = len(episodes)
        if args.workers > 1:
            self.pool = mp.get_context("spawn").Pool(args.workers, _init_worker, init)
            self.map = self.pool.map
        else:
            _init_worker(*init)
            self.pool, self.map = None, lambda f, xs: [f(x) for x in xs]

    def scores(self, units, idx):
        jobs = [(list(map(float, u)), i) for u in units for i in idx]
        res = self.map(_replay, jobs)
        out = []
        for c in range(len(units)):
            rows = [r for chunk in res[c * len(idx):(c + 1) * len(idx)] for r in chunk]
            out.append(score(rows))
        return out

    def close(self):
        if self.pool:
            self.pool.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="fast", choices=["fast", "np", "table"])
    ap.add_argument("--null", default="none", choices=["none", "shuffled", "ablated", "random_mapping"])
    ap.add_argument("--table", default=None)
    ap.add_argument("--teachers", default="human,scripted")
    ap.add_argument("--min-seconds", type=float, default=35.0, help="only teacher fights that lasted this long")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth tick (faster on slow engines)")
    ap.add_argument("--holdout-every", type=int, default=4, help="every Nth teacher fight is held out")
    ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--brain-seed", type=int, default=0)
    ap.add_argument("--name", default="imitate")
    args = ap.parse_args(argv)
    if args.engine == "table" and not args.table:
        args.table = str(ROOT / "brain" / "data" / f"response_table_np_{args.null}.npz")

    paths = find_teacher_logs(args.teachers.split(","), args.min_seconds)
    episodes = [load_episode(p, args.stride) for p in paths]
    keep = [i for i, e in enumerate(episodes) if e]
    paths, episodes = [paths[i] for i in keep], [episodes[i] for i in keep]
    held = [i for i in range(len(episodes)) if i % args.holdout_every == args.holdout_every - 1]
    train = [i for i in range(len(episodes)) if i not in held]
    print(f"{len(episodes)} teacher fights ({sum(map(len, episodes))} ticks): {len(train)} train, {len(held)} held out")
    ev = Evaluator(args, episodes)
    t0 = time.time()
    d = fly_params.default_unit()
    base = ev.scores([d], train)[0]
    print(f"defaults: score {base['score']:.3f} (h {base['horizontal']:.3f}, v {base['vertical']:.3f})  "
          f"[{time.time() - t0:.0f}s]", flush=True)
    try:
        import cma
        es = cma.CMAEvolutionStrategy(list(d), args.sigma, {"popsize": args.popsize, "bounds": [0.0, 1.0],
                                                            "seed": args.seed + 1, "verbose": -9})
        tell = lambda xs, f: es.tell(xs, f)
    except ImportError:
        from cmaes import CMAES
        es = CMAES(d, args.sigma, popsize=args.popsize, seed=args.seed)
        tell = lambda xs, f: es.tell(f)
    best = {"score": base["score"], "unit": list(d), "gen": -1}
    for gen in range(args.generations):
        xs = es.ask()
        sc = ev.scores(xs, train)
        tell(xs, [-s["score"] for s in sc])
        i = int(np.argmax([s["score"] for s in sc]))
        if sc[i]["score"] > best["score"]:
            best = {"score": sc[i]["score"], "unit": [float(v) for v in xs[i]], "gen": gen, **sc[i]}
        print(f"gen {gen:3d}  best {sc[i]['score']:.3f} (h {sc[i]['horizontal']:.3f}, v {sc[i]['vertical']:.3f})"
              f"  best so far {best['score']:.3f}  [{time.time() - t0:.0f}s]", flush=True)
    hd = ev.scores([d, best["unit"]], held) if held else [None, None]
    ev.close()
    if held:
        print(f"held-out: defaults {hd[0]['score']:.3f} -> imitation {hd[1]['score']:.3f} "
              f"(h {hd[1]['horizontal']:.3f}, v {hd[1]['vertical']:.3f})")
    out = HERE / "params" / f"{args.name}.json"
    fly_params.save(out, fly_params.unit_to_params(best["unit"]), method="imitation", engine=args.engine,
                    null=args.null, teachers=args.teachers, min_seconds=args.min_seconds,
                    teacher_logs=[str(p.relative_to(ROOT)) for p in paths], held_out=held,
                    train_defaults=base, train_best={k: v for k, v in best.items() if k != "unit"},
                    heldout_defaults=hd[0], heldout_best=hd[1], generations=args.generations,
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
