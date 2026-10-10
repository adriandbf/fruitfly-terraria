"""CMA-ES for the v3-senses teacher (scene3.V3Policy) on the fake fight, with
per-fight boss randomization. The teacher sees exactly the fly's v3 senses
(no positions, no paths) -- its moves are what those senses make possible;
the fly's brain then learns to reproduce them (scene_fly.py, v3 mode).

  python training/moonlord/train_senses3.py --generations 40
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

_W = {}


def _init(randomize, penalty=0.5):
    import harness
    import fly_params
    lo = json.loads((HERE / "loadout.json").read_text())
    cfg = harness.config_from_loadout(lo, "fight", "balanced")
    cfg["max_seconds"] = {"fight": 240.0}
    _W.update(cfg=cfg, randomize=randomize, arena=fly_params.arena_px(lo), penalty=penalty)


# Where live data says the fake is trustworthy (exposure in 120+ live fights):
# under 1100 px from the core and more than 15 tiles above the arena floor (the start counts). Teachers
# trained without this fled 2000 px away or hugged the floor -- both looked
# great offline and failed live (sectors_v4: 121 s offline, 32 s live).
ENVELOPE = {"dist": (0.0, 1100.0), "height_tiles": (15.0, 250.0)}


class _Track(scene3.V3Policy):
    def act(self, state):
        p = state.get("player") or {}
        ep = state.get("episode") or {}
        core = next((n for n in state.get("npcs", []) if n.get("type") == 398), None)
        if core is not None and "cx" in p and ep.get("arena"):
            import math
            d = math.hypot(p["cx"] - core["cx"], p["cy"] - core["cy"])
            h = (ep["arena"]["y"] + 42 - p["cy"]) / 16.0
            self.n += 1
            self.out += not (ENVELOPE["dist"][0] <= d <= ENVELOPE["dist"][1]
                             and ENVELOPE["height_tiles"][0] <= h <= ENVELOPE["height_tiles"][1])
        return super().act(state)


def _episode(job):
    w, seed = job
    import harness
    from fake_bridge import FakeMoonLordBridge
    from train_fly_cma import randomized
    pol = _Track(w, _W["arena"])
    pol.n = pol.out = 0
    r = harness.run_episode(FakeMoonLordBridge({"seed": seed, **randomized({}, seed, _W["randomize"])}),
                            pol, _W["cfg"], seed)
    left = r["boss_parts_frac_left"]
    prog = 1.0 - (left if left is not None else 1.0)
    outside = pol.out / max(pol.n, 1)
    # score: boss damage, minus a penalty for time spent where the fake is unvalidated
    return prog - _W["penalty"] * outside, r["seconds"], prog, outside


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--popsize", type=int, default=20)
    ap.add_argument("--train-seeds", type=int, default=12)
    ap.add_argument("--heldout-seeds", type=int, default=30)
    ap.add_argument("--randomize", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default="senses3_teacher")
    ap.add_argument("--init", default=None)
    ap.add_argument("--scale", type=float, default=2.0, help="weights range +-scale")
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--penalty", type=float, default=0.5, help="score penalty per unit fraction of time outside the live-validated envelope")
    args = ap.parse_args(argv)
    from cmaes import CMAES
    train = list(range(1000, 1000 + args.train_seeds))
    held = list(range(5000, 5000 + args.heldout_seeds))
    sc_ = args.scale
    x0 = np.full(scene3.N_W, 0.5)
    if args.init:
        w0 = np.asarray(json.loads(Path(args.init).read_text())["weights"], dtype=float)
        x0 = np.clip((w0 / sc_ + 1) / 2, 0, 1)
    es = CMAES(x0, args.sigma, popsize=args.popsize, seed=2)
    best = (-1.0, None)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, _init, (args.randomize, args.penalty)) as pool:
        for g in range(args.generations):
            ws = [list(map(float, (u * 2 - 1) * sc_)) for u in es.ask()]
            res = pool.map(_episode, [(w, s) for w in ws for s in train])
            sc = [float(np.mean([r[0] for r in res[i * len(train):(i + 1) * len(train)]])) for i in range(len(ws))]
            es.tell([-x for x in sc])
            i = int(np.argmax(sc))
            if sc[i] > best[0]:
                best = (sc[i], ws[i])
            print(f"gen {g:2d} best {100 * sc[i]:5.1f}%  mean {100 * np.mean(sc):5.1f}%  [{time.time() - t0:.0f}s]",
                  flush=True)
        h = pool.map(_episode, [(best[1], s) for s in held])
    prog, secs = float(np.mean([r[2] for r in h])), float(np.median([r[1] for r in h]))
    outside = float(np.mean([r[3] for r in h]))
    print(f"held-out: boss damage {100 * prog:.1f}%  median survival {secs:.1f}s  outside envelope {100 * outside:.0f}%")
    out = HERE / "params" / f"{args.name}.json"
    out.write_text(json.dumps({"weights": best[1], "meta": {
        "what": "v3-senses teacher (scene3.V3Policy), NOT the fly", "train_progress": best[0],
        "heldout_progress": prog, "heldout_median_s": secs, "randomize": args.randomize,
        "heldout_outside_envelope": outside, "penalty": args.penalty, "envelope": ENVELOPE,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S")}}, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
