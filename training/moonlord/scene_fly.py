"""Scene fly (v2 senses + learned readout), 2026-10-09.

Why: every policy that only gets the fly's old inputs (looming summed into
left/right, level/above) tops out around 34% boss damage offline, the fly
included; an 8-direction looming controller (SectorPolicy) reached 47 s /
48% damage LIVE. So the fly needs richer senses -- but they have to go
THROUGH the brain to count.

Senses (four visual input types per eye, each a real FlyWire population):
  LC4    big things approaching level/below (boss parts, the core from afar,
         floor and walls)
  LPLC4  anything approaching from above, and the ceiling
  LC11   small moving objects level/below (projectiles) -- LC11 is the
         small-object detector
  LPLC2  imminent collisions (time-to-contact under ttc_imminent ticks)
Different input types drive different descending neurons in this
connectome (brain/scene_neurons.json, from a 45-type sweep): LC4 ->
DNp04/DNp02/DNp11, LPLC4 -> DNb05/DNp07, LC11 -> DNg100, LPLC2 -> DNp103...

Readout: a LINEAR map from 69 of those descending neurons' rates (+ vertical
speed, + bias) to a sideways drive and a vertical drive, with
SectorPolicy's commitment rule (a side, once chosen, is held for a while).
The weights are learned by imitation: replay the live-proven SectorPolicy's
fights, feed the fly's senses through the brain, and regress its drives on
the descending-neuron rates (ridge). Nothing from the teacher reaches the
fly at fight time -- only its own brain's output.

Control (planned in the Moon Lord plan): the same pipeline on the shuffled
connectome. If a scrambled brain imitates as well, the wiring isn't doing
the work.

  python training/moonlord/scene_fly.py collect --fights 10 --engine np --null none
  python training/moonlord/scene_fly.py fit --data logs/moonlord/scene_np_none.npz
  python training/moonlord/scene_fly.py eval --params training/moonlord/params/scene_np_none.json --fights 8
"""
import argparse
import json
import math
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
import policies  # noqa: E402

SCENE = json.loads((ROOT / "brain" / "scene_neurons.json").read_text())
TYPES = ("LC4", "LPLC4", "LC11", "LPLC2")
TEACHER = HERE / "params" / "sectors_linear.json"
ALL_DNS = SCENE["all_dns"]   # collect records every descending neuron; fit picks the readout set


def populations():
    import neurons
    return {("LC4", -1): neurons.LC4_LOOMING_LEFT, ("LC4", 1): neurons.LC4_LOOMING_RIGHT,
            ("LPLC4", -1): neurons.LPLC4_LOOMING_LEFT, ("LPLC4", 1): neurons.LPLC4_LOOMING_RIGHT,
            ("LC11", -1): SCENE["LC11"]["left"], ("LC11", 1): SCENE["LC11"]["right"],
            ("LPLC2", -1): SCENE["LPLC2"]["left"], ("LPLC2", 1): SCENE["LPLC2"]["right"],
            ("VS", -1): SCENE["VS"]["left"], ("VS", 1): SCENE["VS"]["right"],
            ("HS", -1): SCENE["HS"]["left"], ("HS", 1): SCENE["HS"]["right"]}


def scene_rates(pol, state, rng_big=900.0, rng_small=700.0, ttc_imminent=25.0, mx=150.0, sumw=0.5):
    """{(type, side): Hz} for the four input types, side -1 left / +1 right.
    pol supplies the attention helpers (threats, core keep-away, surfaces)."""
    p = state["player"]
    pcx, pcy = p["cx"], p["cy"]
    acc = {(t, s): [] for t in TYPES for s in (-1, 1)}

    def add(t, side, r):
        if r <= 0:
            return
        for s in ((-1, 1) if side == 0 else (side,)):
            acc[(t, s)].append(r)

    thr = {**pol.encoder.DEFAULT_PARAMS, **(pol.enc_params or {})}["threat_max_range_px"]
    for th, ttc in pol._threats(state):
        x, y = th["x"], th["y"]
        d = math.hypot(x - pcx, y - pcy)
        side = 1 if x >= pcx else -1
        small = th.get("from_projectile") is not None
        if not small:
            d = max(1.0, d - 40.0)
        above = y < pcy - pol.ABOVE_PX and th.get("approaching", True)
        r = max(0.0, 1.0 - d / (rng_small if small else rng_big))
        add("LPLC4" if above else ("LC11" if small else "LC4"), side, r)
        if ttc < ttc_imminent:
            add("LPLC2", side, 1.0 - ttc / ttc_imminent)
    cp = pol._core_pseudo(state)
    if cp:
        c = next(n for n in state["npcs"] if n.get("type") == 398)
        add("LPLC4" if c["cy"] < pcy - pol.ABOVE_PX else "LC4", 1 if c["cx"] >= pcx else -1,
            max(0.0, 1.0 - cp[1] / thr))
    for ps, eff, slot in pol._surfaces(state):
        dx = ps["x"] - p["x"]
        add("LPLC4" if slot == "above" else "LC4", 0 if abs(dx) < 1 else (1 if dx > 0 else -1),
            max(0.0, 1.0 - eff / thr))
    out = {}
    for k, rs in acc.items():
        top = max(rs) if rs else 0.0
        out[k] = min(mx, mx * (top + sumw * (sum(rs) - top))) if rs else 0.0
    # Self-motion (optic flow) on a baseline, since rates can't go negative:
    # VS cells (vertical flow) fire more the faster the fly climbs; HS cells
    # (horizontal flow) of each eye fire more for motion toward that side.
    vx, vy = p.get("vx", 0.0), p.get("vy", 0.0)
    out[("VS", -1)] = out[("VS", 1)] = min(250.0, max(0.0, 100.0 - 12.0 * vy))
    out[("HS", -1)] = min(250.0, max(0.0, 100.0 - 12.0 * vx))
    out[("HS", 1)] = min(250.0, max(0.0, 100.0 + 12.0 * vx))
    return out


class SceneFlyPolicy(policies.FlyPolicy):
    """The fly with v2 senses and a learned linear readout (see module doc).
    All keys it presses are credited "brain"."""
    name = "fly"

    def __init__(self, readout=None, **kw):
        super().__init__(**kw)
        self.pops = populations()
        self.readout = readout          # {"wh": [...], "wv": [...], "thr": f, "hold": f, "dn_ids": [...]}
        self.dn_ids = (readout or {}).get("dn_ids") or [d["id"] for d in SCENE["readout_dns"]]
        self.last_spikes = None
        self.dir, self.hold = 1, 0
        self.last_dn_vec = None
        self.last_scene = None

    def reset(self):
        super().reset()
        self.dir, self.hold = 1, 0

    def brain_step(self, state):
        """Senses -> brain (100 ms) -> the readout DNs' rates."""
        self._ensure_brain()
        sr = scene_rates(self, state)
        self.last_scene = sr
        ids, rates = [], []
        for k, r in sr.items():
            if r > 0:
                ids += list(self.pops[k])
                rates += [r] * len(self.pops[k])
        spikes = self.brain.stimulate(ids, rates, duration_ms=100)
        self.last_spikes = spikes
        vec = np.array([self.brain.rate_for(spikes, f) for f in self.dn_ids])
        self.last_dn_vec = vec
        return vec

    def act(self, state):
        p = state.get("player") or {}
        if not state.get("in_world") or p.get("dead") or "cx" not in p:
            return dict(policies.RELEASE), {}
        vec = self.brain_step(state)
        extra = [p.get("vy", 0.0) / 10.0] if self.readout.get("use_vy") else []
        f = np.concatenate([vec / 100.0, extra, [1.0]])
        h = float(f @ np.asarray(self.readout["wh"]))
        v = float(f @ np.asarray(self.readout["wv"]))
        if self.hold > 0:
            self.hold -= 1
        elif abs(h) > self.readout["thr"]:
            self.dir = 1 if h > 0 else -1
            self.hold = int(self.readout["hold"])
        vert = {"jump": True} if v > 0.3 else ({"down": True} if v < -0.3 else {"up": True})
        action = {**policies.RELEASE, "left": self.dir < 0, "right": self.dir > 0, **vert}
        return action, policies._sources(action, "brain")


# ---- imitation data ------------------------------------------------------
def _teacher():
    data = fly_params.load(TEACHER)
    return data, {**fly_params.policy_kwargs(data), "weights": data["weights"]}


def _collect_job(job):
    engine, null, seeds, randomize = job
    import harness
    from brain_backend import make_brain
    from fake_bridge import FakeMoonLordBridge
    from train_fly_cma import randomized
    brain, _ = make_brain(engine, null, seed=0)
    data, tkw = _teacher()
    lo = json.loads((HERE / "loadout.json").read_text())
    cfg = harness.config_from_loadout(lo, "fight", "balanced")
    cfg["max_seconds"] = {"fight": 180.0}
    X, H, V = [], [], []

    class Recorder(policies.SectorPolicy):
        """The teacher, also feeding the fly's senses through the brain each tick."""
        def __init__(self, **kw):
            super().__init__(**kw)
            self.fly = SceneFlyPolicy(brain=brain, readout_fn=lambda r: {}, **fly_params.policy_kwargs(data))

        def act(self, state):
            a, s = super().act(state)
            p = state.get("player") or {}
            if state.get("in_world") and not p.get("dead") and "cx" in p:
                f = self.sectors(state) + [p.get("vy", 0.0) / 10.0, 1.0]
                self.fly.brain_step(state)
                sp = self.fly.last_spikes
                X.append(np.array([brain.rate_for(sp, fid) for fid in ALL_DNS] + [p.get("vy", 0.0)]))
                H.append(sum(x * w for x, w in zip(f, self.w[0:10])))
                V.append(sum(x * w for x, w in zip(f, self.w[10:20])))
            return a, s

    for seed in seeds:
        brain.reset_state(seed=seed)
        harness.run_episode(FakeMoonLordBridge({"seed": seed, **randomized({}, seed, randomize)}),
                            Recorder(**tkw), cfg, seed)
    return np.array(X), np.array(H), np.array(V)


def collect(args):
    seeds = list(range(3000, 3000 + args.fights))
    jobs = [(args.engine, args.null, seeds[i::args.workers], args.randomize) for i in range(args.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(_collect_job, jobs)
    X = np.concatenate([r[0] for r in res])
    H = np.concatenate([r[1] for r in res])
    V = np.concatenate([r[2] for r in res])
    out = ROOT / "logs" / "moonlord" / f"scene_{args.engine}_{args.null}.npz"
    np.savez(out, X=X, H=H, V=V)
    print(f"{len(X)} ticks from {args.fights} teacher fights [{time.time() - t0:.0f}s] -> {out.relative_to(ROOT)}")


def fit(args):
    d = np.load(args.data)
    X, H, V = d["X"], d["H"], d["V"]
    dn = np.asarray(ALL_DNS)
    if args.select == "scene":   # the 69 DNs the real brain's sweep picked (brain/scene_neurons.json)
        want = {x["id"] for x in SCENE["readout_dns"]}
        cols = [i for i, f in enumerate(dn) if int(f) in want]
    else:                        # the k most variable DNs in THIS brain's own data (fair to null brains)
        cols = list(np.argsort(-X[:, :-1].var(0))[:args.k])
    print(f"readout: {len(cols)} DNs ({args.select}); active (>1 Hz mean) {(X[:, cols].mean(0) > 1).sum()}")
    dn_ids = [int(dn[i]) for i in cols]
    # Readout = descending neurons only (+ bias). The character's vertical speed
    # reaches the brain through the VS/HS self-motion senses, not the readout.
    F = (np.column_stack([X[:, cols] / 100.0, X[:, -1] / 10.0, np.ones(len(X))]) if args.use_vy
         else np.column_stack([X[:, cols] / 100.0, np.ones(len(X))]))
    n = len(F)
    idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    lam = args.ridge
    A = F[tr].T @ F[tr] + lam * np.eye(F.shape[1])
    wh = np.linalg.solve(A, F[tr].T @ H[tr])
    wv = np.linalg.solve(A, F[tr].T @ V[tr])

    def r2(w, y):
        pred = F[te] @ w
        return 1 - np.mean((pred - y[te]) ** 2) / np.var(y[te])

    def agree(w, y, thr):
        pred = F[te] @ w
        return float(np.mean(np.sign(pred) == np.sign(y[te]))), float(np.mean((pred > thr) == (y[te] > thr)))
    teacher = fly_params.load(TEACHER)["weights"]
    thr, hold = abs(teacher[20]), abs(teacher[21]) * 30
    print(f"held-out ticks: R^2 sideways {r2(wh, H):.3f}, vertical {r2(wv, V):.3f}; "
          f"sign agreement sideways {agree(wh, H, 0)[0]:.3f}, climb(v>0.3) agreement {agree(wv, V, 0.3)[1]:.3f}")
    stem = Path(args.data).stem + ("" if args.select == "variance" else "_scene")
    out = HERE / "params" / f"{stem}.json"
    base = fly_params.load(TEACHER)
    out.write_text(json.dumps({"encoder": base["encoder"], "decoder": base["decoder"], "policy": base["policy"],
                               "readout": {"wh": wh.tolist(), "wv": wv.tolist(), "thr": thr, "hold": hold,
                                           "dn_ids": dn_ids, "use_vy": bool(args.use_vy)},
                               "meta": {"what": "scene fly: v2 senses -> brain -> linear readout learned by "
                                                "imitating SectorPolicy", "data": str(Path(args.data).name),
                                        "r2_sideways": r2(wh, H), "r2_vertical": r2(wv, V), "ridge": lam,
                                        "finished": time.strftime("%Y-%m-%d %H:%M:%S")}}, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


def _eval_job(job):
    engine, null, params_path, seeds, randomize = job
    import harness
    from brain_backend import make_brain
    from fake_bridge import FakeMoonLordBridge
    from train_fly_cma import randomized
    brain, _ = make_brain(engine, null, seed=0)
    data = fly_params.load(params_path)
    lo = json.loads((HERE / "loadout.json").read_text())
    cfg = harness.config_from_loadout(lo, "fight", "balanced")
    cfg["max_seconds"] = {"fight": 180.0}
    rows = []
    for seed in seeds:
        brain.reset_state(seed=seed)
        pol = SceneFlyPolicy(brain=brain, readout_fn=lambda r: {}, readout=data["readout"],
                             **fly_params.policy_kwargs(data))
        r = harness.run_episode(FakeMoonLordBridge({"seed": seed, **randomized({}, seed, randomize)}), pol, cfg, seed)
        rows.append((r["seconds"], 1 - r["boss_parts_frac_left"]))
    return rows


def evaluate(args):
    seeds = list(range(5000, 5000 + args.fights))
    jobs = [(args.engine, args.null, args.params, seeds[i::args.workers], args.randomize) for i in range(args.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = [r for part in pool.map(_eval_job, jobs) for r in part]
    secs = sorted(r[0] for r in rows)
    print(f"{Path(args.params).stem} on {args.engine}/{args.null}: median survival {secs[len(secs) // 2]:.1f}s, "
          f"boss damage {100 * np.mean([r[1] for r in rows]):.1f}%  ({len(rows)} fights, {time.time() - t0:.0f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--engine", default="fast", choices=["fast", "np"])
        s.add_argument("--null", default="none", choices=["none", "shuffled", "ablated", "random_mapping"])
        s.add_argument("--fights", type=int, default=10)
        s.add_argument("--workers", type=int, default=2)
        s.add_argument("--randomize", type=float, default=0.15)
        if name == "eval":
            s.add_argument("--params", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--data", required=True)
    f.add_argument("--ridge", type=float, default=1.0)
    f.add_argument("--select", default="variance", choices=["variance", "scene"])
    f.add_argument("--k", type=int, default=69)
    f.add_argument("--use-vy", action="store_true", help="also give the readout the vertical speed directly (not the fly)")
    args = ap.parse_args(argv)
    {"collect": collect, "fit": fit, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
