"""CMA-ES over the fly's interface knobs (fly_params.PARAM_SPEC: encoder,
decoder and FlyPolicy thresholds -- never the connectome) against the
offline Moon Lord fight (fake_bridge.py, calibrated to live fights),
with the real brain in the loop.

Score per candidate = mean boss damage fraction over a FIXED set of fight
seeds (same seeds for every candidate, so differences are the knobs, not
luck): 1 - (eye-aware boss HP left / total), the same progress measure as
report.py. The best candidate is then scored on HELD-OUT seeds it never
trained on, next to the untuned defaults, and saved as a params file
harness.py --params-file can load for live fights.

Brain engines (brain/brain_backend.py):
  --engine fast   numba FastBrain, the live engine (the Mac; recommended
                  for real runs, e.g. overnight)
  --engine np     pure-numpy engine (anywhere; ~15x slower than fast)
  --engine table  response-table surrogate (brain/response_table.py;
                  ~1000x faster, approximate -- rescore its winner with
                  --rescore on a real engine before trusting it)
Null models for the baselines: --null shuffled | ablated | random_mapping
(re-tuned with the same budget, as the plan requires).

Examples:
  # overnight on the Mac, all cores
  python training/moonlord/train_fly_cma.py --engine fast --workers 8 \
      --generations 40 --popsize 16 --train-seeds 8 --name fly_cma
  # the same budget for a null brain
  python training/moonlord/train_fly_cma.py --engine fast --null shuffled --name shuffled_cma ...
  # warm start from imitation
  python training/moonlord/train_fly_cma.py --init training/moonlord/params/imitate.json ...
  # re-score saved params on a real engine (no search)
  python training/moonlord/train_fly_cma.py --rescore training/moonlord/params/fly_cma.json --engine fast
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

PARAMS_DIR = HERE / "params"
LOG_DIR = ROOT / "logs" / "moonlord"
TRAIN_SEED0, HELDOUT_SEED0 = 1000, 5000
_FF = HERE / "fake_fit.json"
FAKE_FIT_STAMP = json.loads(_FF.read_text()).get("fitted") if _FF.exists() else None

_W = {}  # per-worker state: brain, readout, cfg, fake overrides


# Domain randomization: every training fight gets its own boss, each of these
# scaled by U(1-r, 1+r) (seeded by the fight seed, so it's the same boss for
# every candidate). Stops the search from leaning on one exact fake detail
# (2026-10-09: it parked the fly beside the fake's motionless hand).
RANDOMIZE_KEYS = ["boss_drift", "hand_speed", "hand_in", "hand_out", "eye_turn", "eye_speed", "eye_period",
                  "bolt_speed", "bolt_period", "bolt_lead", "sphere_speed", "sphere_period", "contact_radius",
                  "dps_near", "regen_per_s"]


def randomized(fake_cfg, seed, r):
    if not r:
        return fake_cfg
    import random
    from fake_bridge import DEFAULT_CONFIG, FITTED, MASTER, MASTER_FIT
    # 2026-10-12 fix: jitter around the profile the fake will actually use. Before, a
    # Master fight was jittered around the EXPERT values, and since explicit config wins,
    # the Master attack settings (eyes, bolts, spheres, contact) never applied in training.
    master = {**MASTER, **MASTER_FIT} if fake_cfg.get("difficulty") == "master" else {}
    base = {**DEFAULT_CONFIG, **FITTED, **master, **fake_cfg}
    rng = random.Random(seed * 7919 + 13)
    out = dict(fake_cfg)
    for k in RANDOMIZE_KEYS:
        v = base[k] * rng.uniform(1 - r, 1 + r)
        out[k] = int(round(v)) if isinstance(base[k], int) else v
    out["hand_in"] = min(out["hand_in"], out["hand_out"] - 50)
    return out


def _init_worker(engine, null, table, brain_seed, fake_cfg, whip_strategy, max_seconds, vertical, randomize=0.0,
                 extra_policy=None):
    import harness
    from brain_backend import make_brain
    brain, readout = make_brain(engine, null, seed=brain_seed, table_path=table)
    loadout = json.loads((HERE / "loadout.json").read_text())
    cfg = harness.config_from_loadout(loadout, "fight", whip_strategy)
    cfg["max_seconds"] = {"fight": max_seconds}
    _W.update(brain=brain, readout=readout, cfg=cfg, fake=fake_cfg or {}, vertical=vertical, randomize=randomize,
              extra=extra_policy or {})


def _episode(job):
    """(unit vector, fight seed) -> (progress, seconds, outcome)."""
    unit, seed = job
    import harness
    from fake_bridge import FakeMoonLordBridge
    from policies import FlyPolicy
    params = fly_params.unit_to_params(unit)
    brain = _W["brain"]
    brain.reset_state(seed=seed)
    pol = FlyPolicy(brain=brain, readout_fn=_W["readout"],
                    **{**fly_params.policy_kwargs(params, _W["vertical"]), **_W["extra"]})
    bridge = FakeMoonLordBridge({"seed": seed, **randomized(_W["fake"], seed, _W["randomize"])})
    row = harness.run_episode(bridge, pol, _W["cfg"], seed)
    frac = row.get("boss_parts_frac_left")
    return (1.0 - frac if frac is not None else 0.0), row["seconds"], row["outcome"]


class Evaluator:
    def __init__(self, args):
        init = (args.engine, args.null, args.table, args.brain_seed, _fake_cfg(args),
                args.whip_strategy, args.max_seconds, args.vertical, args.randomize, _extra(args))
        if args.workers > 1:
            self.pool = mp.get_context("spawn").Pool(args.workers, _init_worker, init)
            self.map = self.pool.map
        else:
            _init_worker(*init)
            self.pool, self.map = None, lambda f, xs: [f(x) for x in xs]

    def scores(self, units, seeds):
        """-> per-candidate dicts {progress, seconds, wins} (means over seeds)."""
        jobs = [(list(map(float, u)), s) for u in units for s in seeds]
        res = self.map(_episode, jobs)
        out = []
        for i in range(len(units)):
            chunk = res[i * len(seeds):(i + 1) * len(seeds)]
            out.append({"progress": float(np.mean([r[0] for r in chunk])),
                        "seconds": float(np.mean([r[1] for r in chunk])),
                        "wins": sum(r[2] == "win" for r in chunk),
                        "per_seed": [round(r[0], 4) for r in chunk]})
        return out

    def close(self):
        if self.pool:
            self.pool.close()


def _extra(args):
    """FlyPolicy switches that aren't searched over (structural variants)."""
    return {"multi_threat": True} if args.multi_threat else {}


def _rel(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def _fake_cfg(args):
    return json.loads(args.fake_config) if args.fake_config else {}


def _default_table(args):
    if args.engine == "table" and not args.table:
        args.table = str(ROOT / "brain" / "data" / f"response_table_np_{args.null}.npz")


def rescore(args):
    data = fly_params.load(args.rescore)
    units = {"defaults": fly_params.default_unit(), Path(args.rescore).stem: fly_params.params_to_unit(data)}
    ev = Evaluator(args)
    seeds = list(range(HELDOUT_SEED0, HELDOUT_SEED0 + args.heldout_seeds))
    t0 = time.time()
    res = dict(zip(units, ev.scores(list(units.values()), seeds)))
    ev.close()
    for k, r in res.items():
        print(f"{k:24s} held-out progress {100 * r['progress']:5.1f}%  survival {r['seconds']:5.1f}s  "
              f"wins {r['wins']}/{len(seeds)}   [{args.engine}/{args.null}]")
    print(f"[{time.time() - t0:.0f}s]")
    return res


def train(args):
    name = args.name or f"cma_{args.engine}_{args.null}"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    train_seeds = list(range(TRAIN_SEED0, TRAIN_SEED0 + args.train_seeds))
    held_seeds = list(range(HELDOUT_SEED0, HELDOUT_SEED0 + args.heldout_seeds))
    x0 = fly_params.params_to_unit(fly_params.load(args.init)) if args.init else fly_params.default_unit()
    ev = Evaluator(args)
    t0 = time.time()
    base = ev.scores([fly_params.default_unit()] + ([x0] if args.init else []), train_seeds)
    print(f"defaults: train progress {100 * base[0]['progress']:.1f}%  survival {base[0]['seconds']:.1f}s"
          + (f" | init: {100 * base[1]['progress']:.1f}%" if args.init else "") + f"  [{time.time() - t0:.0f}s]",
          flush=True)
    try:
        import cma
        es = cma.CMAEvolutionStrategy(list(x0), args.sigma, {"popsize": args.popsize, "bounds": [0.0, 1.0],
                                                             "seed": args.seed + 1, "verbose": -9})
        ask, tell = es.ask, lambda xs, f: es.tell(xs, f)
        backend = "cma"
    except ImportError:
        from cmaes import CMAES
        es = CMAES(x0, args.sigma, popsize=args.popsize, seed=args.seed)
        ask, tell = es.ask, lambda xs, f: es.tell(f)
        backend = "builtin"
    print(f"CMA-ES ({backend}), {len(x0)} knobs, popsize {args.popsize}, {len(train_seeds)} train seeds, "
          f"engine {args.engine}, null {args.null}", flush=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = LOG_DIR / f"cma_{stamp}_{name}.jsonl"
    best = {"progress": base[-1]["progress"], "unit": list(x0), "gen": -1}
    for gen in range(args.generations):
        xs = ask()
        sc = ev.scores(xs, train_seeds)
        tell(xs, [-s["progress"] for s in sc])
        i = int(np.argmax([s["progress"] for s in sc]))
        if sc[i]["progress"] > best["progress"]:
            best = {"progress": sc[i]["progress"], "unit": [float(v) for v in xs[i]], "gen": gen,
                    "seconds": sc[i]["seconds"]}
        with open(hist_path, "a") as f:
            f.write(json.dumps({"gen": gen, "best": sc[i], "mean_progress": float(np.mean([s["progress"] for s in sc])),
                                "unit": [float(v) for v in xs[i]], "elapsed_s": round(time.time() - t0)}) + "\n")
        print(f"gen {gen:3d}  best {100 * sc[i]['progress']:5.1f}%  mean {100 * np.mean([s['progress'] for s in sc]):5.1f}%"
              f"  (best so far {100 * best['progress']:5.1f}% @gen {best['gen']})  [{time.time() - t0:.0f}s]", flush=True)

    # Held-out check: defaults vs best (and the CMA mean, often more robust than one lucky sample).
    mean_unit = list(np.clip(getattr(es, "mean", None) if hasattr(es, "mean") else es.result.xfavorite, 0, 1))
    cands = {"defaults": fly_params.default_unit(), "best": best["unit"], "mean": mean_unit}
    held = dict(zip(cands, ev.scores(list(cands.values()), held_seeds)))
    ev.close()
    for k, r in held.items():
        print(f"held-out {k:9s} progress {100 * r['progress']:5.1f}%  survival {r['seconds']:5.1f}s  wins {r['wins']}")
    pick = "best" if held["best"]["progress"] >= held["mean"]["progress"] else "mean"
    params = fly_params.unit_to_params(cands[pick])
    if args.vertical != "brain":
        params["policy"]["vertical"] = args.vertical
    params["policy"].update(_extra(args))
    out = PARAMS_DIR / f"{name}.json"
    fly_params.save(out, params, engine=args.engine, null=args.null, table=args.table, picked=pick,
                    train_seeds=len(train_seeds), heldout_seeds=len(held_seeds),
                    generations=args.generations, popsize=args.popsize, fake_config=_fake_cfg(args),
                    whip_strategy=args.whip_strategy, max_seconds=args.max_seconds, randomize=args.randomize,
                    fake_fit=FAKE_FIT_STAMP,
                    train_progress_defaults=base[0]["progress"], train_progress_best=best["progress"],
                    heldout={k: {kk: v[kk] for kk in ("progress", "seconds", "wins")} for k, v in held.items()},
                    history=_rel(hist_path), finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                    seconds=round(time.time() - t0))
    print(f"wrote {_rel(out)} (picked {pick})")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="fast", choices=["fast", "np", "table"])
    ap.add_argument("--null", default="none", choices=["none", "shuffled", "ablated", "random_mapping"])
    ap.add_argument("--table", default=None, help="response table for --engine table "
                                                  "(default brain/data/response_table_np_<null>.npz)")
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--train-seeds", type=int, default=8)
    ap.add_argument("--heldout-seeds", type=int, default=16)
    ap.add_argument("--max-seconds", type=float, default=120.0, help="fight length cap (game seconds)")
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--brain-seed", type=int, default=0)
    ap.add_argument("--vertical", default="brain", choices=["brain", "rule"])
    ap.add_argument("--whip-strategy", default="balanced", choices=["nearest", "balanced"])
    ap.add_argument("--fake-config", default=None, help='JSON overrides for fake_bridge, e.g. \'{"minion_dps": 2000}\'')
    ap.add_argument("--multi-threat", action="store_true", help="fly sees every threat at once (FlyPolicy multi_threat)")
    ap.add_argument("--randomize", type=float, default=0.15,
                    help="per-fight boss variation, +-fraction (0 = the exact fitted fake)")
    ap.add_argument("--init", default=None, help="params file to start from (e.g. imitate.py's)")
    ap.add_argument("--rescore", default=None, help="score this params file vs defaults on held-out seeds, no search")
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)
    _default_table(args)
    return rescore(args) if args.rescore else train(args)


if __name__ == "__main__":
    main()
