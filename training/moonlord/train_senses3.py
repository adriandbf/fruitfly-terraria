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


def _init(randomize, penalty=0.5, objective="boss", eye_weight=1.0, master=False, fight_s=240.0,
          weights=None, range_px=0.0, range_penalty=0.0):
    import harness
    import fly_params
    lo = json.loads((HERE / "loadout.json").read_text())
    cfg = harness.config_from_loadout(lo, "fight", "balanced")
    cfg["max_seconds"] = {"fight": fight_s}
    cfg.pop("loadout_items", None)
    _W.update(cfg=cfg, randomize=randomize, arena=fly_params.arena_px(lo), penalty=penalty,
              objective=objective, eye_weight=eye_weight, master=master,
              weights=weights or {"eye": eye_weight}, range_px=range_px, range_penalty=range_penalty)


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
            self.far += bool(_W.get("range_px")) and d > _W["range_px"]
        return super().act(state)


def _episode(job):
    w, seed = job
    import harness
    from fake_bridge import FakeMoonLordBridge
    from train_fly_cma import randomized
    pol = _Track(w, _W["arena"])
    pol.n = pol.out = pol.far = 0
    fake = {"seed": seed, **randomized({}, seed, _W["randomize"])}
    if _W["master"]:
        fake["difficulty"] = "master"
    b = FakeMoonLordBridge(fake)
    r = harness.run_episode(b, pol, _W["cfg"], seed)
    left = r["boss_parts_frac_left"]
    prog = 1.0 - (left if left is not None else 1.0)
    outside = pol.out / max(pol.n, 1)
    mins = max(r["seconds"], 1.0) / 60.0
    eye = b.damage_by.get("eye", 0) / mins
    other = (sum(b.damage_by.values()) - b.damage_by.get("eye", 0)) / mins
    ray = b.damage_by.get("deathray", 0) / mins
    far = pol.far / max(pol.n, 1)
    if _W["objective"] == "hits":
        # 2026-10-12 (user: minimise hits, most damaging attack first): damage taken per
        # minute, the current target attack weighted up (weights per source, default 1);
        # a little boss damage so it keeps fighting; the envelope penalty so it stays where
        # the fake is valid; a range penalty for time beyond minion range (the eye-dodging
        # fly drifted to ~950 px from the core live and the boss died slower).
        dmg = sum(_W["weights"].get(k, 1.0) * v for k, v in b.damage_by.items()) / mins
        score = (-dmg / 1000.0 + 0.5 * prog - _W["penalty"] * outside - _W["range_penalty"] * far)
    else:
        # score: boss damage, minus a penalty for time spent where the fake is unvalidated
        score = prog - _W["penalty"] * outside
    return score, r["seconds"], prog, outside, eye, other, ray, far


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
    ap.add_argument("--objective", default="boss", choices=["boss", "hits"],
                    help="boss = boss damage dealt; hits = damage TAKEN per minute (eye-weighted), plus a little boss damage")
    ap.add_argument("--eye-weight", type=float, default=1.0, help="with --objective hits: weight on homing-eye damage")
    ap.add_argument("--master", action="store_true", help="fight the Master-mode fake (fake_bridge MASTER profile)")
    ap.add_argument("--weights", default="", help="with --objective hits: per-source damage weights, e.g. deathray=3,eye=1 "
                                                  "(sources: eye bolt sphere deathray tongue contact); overrides --eye-weight")
    ap.add_argument("--range-px", type=float, default=0.0, help="distance from the core beyond which minions fall off")
    ap.add_argument("--range-penalty", type=float, default=0.0, help="score penalty per unit fraction of time beyond --range-px")
    ap.add_argument("--fight-seconds", type=float, default=240.0)
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
    best = (-float("inf"), None)
    t0 = time.time()
    wts = {k: float(v) for k, v in (kv.split("=") for kv in args.weights.split(",") if kv)} or None
    with mp.get_context("spawn").Pool(args.workers, _init, (args.randomize, args.penalty, args.objective,
                                                               args.eye_weight, args.master, args.fight_seconds,
                                                               wts, args.range_px, args.range_penalty)) as pool:
        if args.init:   # where the warm start begins, on the held-out seeds
            h0 = pool.map(_episode, [(list(map(float, w0)), s) for s in held])
            print(f"start: eye dmg/min {np.mean([r[4] for r in h0]):.0f}  other dmg/min {np.mean([r[5] for r in h0]):.0f}  "
                  f"deathray dmg/min {np.mean([r[6] for r in h0]):.0f}  beyond range {100 * np.mean([r[7] for r in h0]):.0f}%  "
                  f"boss {100 * np.mean([r[2] for r in h0]):.1f}%  outside {100 * np.mean([r[3] for r in h0]):.0f}%", flush=True)
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
        # Pick on HELD-OUT fights between the best-of-run candidate (chosen on noisy
        # training fights) and the CMA mean (2026-10-12: the best-of-run overfit).
        mean_w = list(map(float, (np.clip(es.mean, 0, 1) * 2 - 1) * sc_))
        cands = [best[1], mean_w]
        hs = [pool.map(_episode, [(w, s) for s in held]) for w in cands]
        pick = int(np.argmax([np.mean([r[0] for r in h]) for h in hs]))
        print(f"held-out score: best-of-run {np.mean([r[0] for r in hs[0]]):.3f}, CMA mean {np.mean([r[0] for r in hs[1]]):.3f}"
              f" -> {'best-of-run' if pick == 0 else 'CMA mean'}")
        best, h = (best[0], cands[pick]), hs[pick]
    prog, secs = float(np.mean([r[2] for r in h])), float(np.median([r[1] for r in h]))
    outside = float(np.mean([r[3] for r in h]))
    eye, other = float(np.mean([r[4] for r in h])), float(np.mean([r[5] for r in h]))
    ray, far = float(np.mean([r[6] for r in h])), float(np.mean([r[7] for r in h]))
    print(f"held-out: boss damage {100 * prog:.1f}%  median survival {secs:.1f}s  outside envelope {100 * outside:.0f}%  "
          f"eye dmg/min {eye:.0f}  other dmg/min {other:.0f}  deathray dmg/min {ray:.0f}  beyond range {100 * far:.0f}%")
    out = HERE / "params" / f"{args.name}.json"
    out.write_text(json.dumps({"weights": best[1], "meta": {
        "what": "v3-senses teacher (scene3.V3Policy), NOT the fly", "train_progress": best[0],
        "heldout_progress": prog, "heldout_median_s": secs, "randomize": args.randomize,
        "heldout_outside_envelope": outside, "penalty": args.penalty, "envelope": ENVELOPE,
        "objective": args.objective, "eye_weight": args.eye_weight, "master": args.master,
        "heldout_eye_dmg_per_min": eye, "heldout_other_dmg_per_min": other,
        "heldout_deathray_dmg_per_min": ray, "heldout_beyond_range": far, "weights": wts,
        "range_px": args.range_px, "range_penalty": args.range_penalty,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S")}}, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
