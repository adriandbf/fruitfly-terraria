"""Policies for the Moon Lord harness (Goal 2). Every policy has the same
shape:

    policy.reset()                       # start of an episode
    action, sources = policy.act(state)  # once per 10 Hz tick

`action` is a set_action dict (left/right/jump/use/up/down). `sources` maps
each PRESSED key to who decided it -- "brain" (the fly), "script" (a
hand-written policy), "random". harness.py totals these into the per-episode
credit ledger, so a result can always say how much of the behaviour was the
fly's.

Baselines from the plan: IdlePolicy (the build standing still),
RandomPolicy, ScriptedDodger (the hand-written ceiling). FlyPolicy is the
real thing: encoder -> FastBrain -> decoder.
"""
import math
import random
import sys
from pathlib import Path

KEYS = ("left", "right", "jump", "use", "up", "down")
RELEASE = {k: False for k in KEYS}

PHANTASMAL_DEATHRAY = 455
PHANTASMAL_EYE = 452


def _sources(action, who):
    return {k: who for k in KEYS if action.get(k)}


class IdlePolicy:
    name = "idle"

    def reset(self):
        pass

    def act(self, state):
        return dict(RELEASE), {}


class RandomPolicy:
    """Random key combinations, each held for a random 1-8 ticks (so it
    actually travels instead of jittering in place)."""
    name = "random"

    def __init__(self, seed=0):
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)
        self.hold, self.current = 0, dict(RELEASE)

    def act(self, state):
        if self.hold <= 0:
            h = self.rng.choice((-1, 0, 1))
            v = self.rng.choice(("up", None, "down"))
            self.current = {**RELEASE, "left": h < 0, "right": h > 0,
                            "jump": v == "up", "up": v == "up", "down": v == "down"}
            self.hold = self.rng.randint(1, 8)
        self.hold -= 1
        return dict(self.current), _sources(self.current, "random")


class ScriptedDodger:
    """Hand-written dodger, the 'ceiling' baseline. Each tick it tries the 9
    combinations of (left / none / right) x (ascend / none / descend),
    rolls each one forward HORIZON ticks with simple kinematics, moves every
    projectile in a straight line (the deathray as a fixed beam), and picks
    the cheapest: projectile overlap (weighted by damage), getting close to
    the boss's core, and straying from the arena centre.

    The kinematics constants match fake_bridge.DEFAULT_CONFIG; they are a
    guess for the real game and should be refit from live logs."""
    name = "scripted"

    HORIZON = 36   # ticks (0.6 s) of lookahead
    STEP = 3       # ticks per lookahead sample
    MARGIN = 14.0  # px of extra clearance wanted around each projectile

    def __init__(self, run_speed=6.0, wing_ascent=5.0, gravity=0.4, max_fall=10.0,
                 boss_keep_away=350.0, arena_half_width=1300.0, arena_height=2900.0, beam_width=40.0):
        self.run_speed, self.wing_ascent = run_speed, wing_ascent
        self.gravity, self.max_fall = gravity, max_fall
        self.boss_keep_away, self.arena_half_width = boss_keep_away, arena_half_width
        self.arena_height = arena_height
        self.beam_width = beam_width
        self.reset()

    def reset(self):
        self.prev = (0, None)
        self.home = self.floor = None
        self.ray_prev = None  # last seen deathray angle, to estimate its sweep rate

    def _rollout(self, p, h, v):
        x, y = p["cx"], p["cy"]
        vx, vy = p.get("vx", 0.0), p.get("vy", 0.0)
        wing = p.get("wing_time", 0) or 0
        infinite = p.get("mount_active", False)
        pts = []
        for t in range(1, self.HORIZON + 1):
            vx += max(-0.6, min(0.6, h * self.run_speed - vx))
            if v == "up" and (wing > 0 or infinite):
                vy = max(-self.wing_ascent, vy - 1.0)
                wing -= 1
            elif infinite:
                vy += max(-0.5, min(0.5, (self.wing_ascent if v == "down" else 0.0) - vy))
            else:
                vy = min(self.max_fall, vy + self.gravity * (2.0 if v == "down" else 1.0))
            x += vx
            y += vy
            if p.get("on_ground") and y > p["cy"] and v != "up":
                y, vy = p["cy"], 0.0  # can't fall through the floor we're standing on
            pts.append((t, x, y))
        return pts

    def _eye_paths(self, eyes, pts, turn=0.03):
        """Turn-limited homing eyes steered at the rolled-out player, per tick."""
        paths = []
        for pr in eyes:
            ex, ey = pr["cx"], pr["cy"]
            speed = math.hypot(pr["vx"], pr["vy"]) or 1.0
            ang = math.atan2(pr["vy"], pr["vx"])
            path = []
            for _, x, y in pts:
                want = math.atan2(y - ey, x - ex)
                d = (want - ang + math.pi) % (2 * math.pi) - math.pi
                ang += max(-turn, min(turn, d))
                ex += speed * math.cos(ang)
                ey += speed * math.sin(ang)
                path.append((ex, ey))
            paths.append((pr, path))
        return paths

    def act(self, state):
        p = state.get("player") or {}
        if not state.get("in_world") or p.get("dead") or "cx" not in p:
            return dict(RELEASE), {}
        projs = state.get("projectiles", [])
        boss = next((n for n in state.get("npcs", []) if n.get("boss") or n.get("moon_lord_part")), None)
        hw, hh = (p.get("width") or 20) / 2, (p.get("height") or 42) / 2
        # Stay near the arena mark; with no mark, near where the episode began.
        arena = (state.get("episode") or {}).get("arena")
        if self.home is None:
            self.home = arena["x"] + hw if arena else p["cx"]
            # Arena floor = the mark's feet; its top is arena_height above.
            self.floor = (arena["y"] + 2 * hh) if arena else p["cy"] + hh

        # The deathray sweeps; estimate its angular speed (rad/tick) from the
        # angle seen last tick (10 Hz = 6 game ticks apart).
        ray_omega = 0.0
        ray = next((pr for pr in projs if pr["type"] == PHANTASMAL_DEATHRAY), None)
        if ray is not None:
            ang = math.atan2(ray["vy"], ray["vx"])
            if self.ray_prev is not None:
                ray_omega = ((ang - self.ray_prev + math.pi) % (2 * math.pi) - math.pi) / 6.0
            self.ray_prev = ang
        else:
            self.ray_prev = None

        best, best_cost = None, None
        for h in (-1, 0, 1):
            for v in ("up", None, "down"):
                cost = 0.0
                pts = self._rollout(p, h, v)
                eyes = self._eye_paths([pr for pr in projs if pr["type"] == PHANTASMAL_EYE], pts)
                for t, x, y in pts[self.STEP - 1::self.STEP]:
                    w = 1.0 - 0.5 * t / self.HORIZON  # sooner threats matter more
                    for pr, path in eyes:
                        qx, qy = path[t - 1]
                        r = (pr.get("width") or 16) / 2
                        gx = abs(qx - x) - (hw + r + self.MARGIN)
                        gy = abs(qy - y) - (hh + r + self.MARGIN)
                        if gx < 0 and gy < 0:
                            cost += w * (pr.get("damage", 50) or 50) * (1 + min(-gx, -gy) / 20.0)
                    for pr in projs:
                        if pr["type"] == PHANTASMAL_EYE:
                            continue
                        dmg = pr.get("damage", 50) or 50
                        if pr["type"] == PHANTASMAL_DEATHRAY:
                            a_t = math.atan2(pr["vy"], pr["vx"]) + ray_omega * t
                            ux, uy = math.cos(a_t), math.sin(a_t)
                            n = 1.0
                            rx, ry = x - pr["cx"], y - pr["cy"]
                            along = (rx * ux + ry * uy) / n
                            perp = abs(-rx * uy + ry * ux) / n
                            if along > 0:
                                gap = perp - (self.beam_width / 2 + hw + self.MARGIN)
                                if gap < 0:
                                    cost += w * dmg * (1 - gap / 40.0)
                            continue
                        qx, qy = pr["cx"] + pr["vx"] * t, pr["cy"] + pr["vy"] * t
                        r = (pr.get("width") or 16) / 2
                        gx = abs(qx - x) - (hw + r + self.MARGIN)
                        gy = abs(qy - y) - (hh + r + self.MARGIN)
                        if gx < 0 and gy < 0:
                            cost += w * dmg * (1 + min(-gx, -gy) / 20.0)
                    if boss:
                        d = math.hypot(x - boss.get("cx", boss["x"]), y - boss.get("cy", boss["y"]))
                        if d < self.boss_keep_away:
                            cost += w * 2.0 * (self.boss_keep_away - d)
                    off = abs(x - self.home) - self.arena_half_width
                    if off > 0:
                        cost += w * off
                    above = (self.floor - self.arena_height) - y  # >0 once above the arena's top row
                    if above > 0:
                        cost += w * above
                cost += 0.5 * (v == "up")                 # save wing time
                cost += 0.2 * ((h, v) != self.prev)       # mild preference not to dither
                if best_cost is None or cost < best_cost:
                    best, best_cost = (h, v), cost
        self.prev = best
        h, v = best
        action = {**RELEASE, "left": h < 0, "right": h > 0,
                  "jump": v == "up", "up": v == "up", "down": v == "down"}
        return action, _sources(action, "script")


class FlyPolicy:
    """The fly: encoder -> FastBrain -> decoder, exactly the run_fly.py
    pipeline. Stage M0/M2 adapter: hostile PROJECTILES that are closing in
    are presented to the encoder as hostile 'NPCs', so the existing
    looming (LC4) channel sees them. That adapter is a hand-written choice
    and is logged as such; the real M2 work replaces it with a proper
    projectile encoding (visual-field position + approach speed).

    Not driven by the brain yet: up/down (no vertical readout found; stage
    M3) and `use` (minions attack on their own; holding use would just
    re-cast the summon staff). Both are held False and NOT credited.

    brain/encoder/decoder can be injected for tests; by default the brain is
    built on first use (needs the Mac's venv and connectome data)."""
    name = "fly"

    def __init__(self, enc_params=None, dec_params=None, brain=None, readout_fn=None,
                 threat_horizon_ticks=90):
        root = Path(__file__).resolve().parents[2]
        for sub in ("brain", "interface"):
            if str(root / sub) not in sys.path:
                sys.path.insert(0, str(root / sub))
        import encoder
        import decoder
        self.encoder, self.decoder = encoder, decoder
        self.enc_params, self.dec_params = enc_params, dec_params
        self.brain, self.readout_fn = brain, readout_fn
        self.threat_horizon = threat_horizon_ticks
        self.flee = decoder.StatefulDecoder(dec_params)
        self.last_drives = None

    def _ensure_brain(self):
        if self.brain is None:
            root = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(root / "brain" / "Drosophila_brain_model"))
            sys.path.insert(0, str(root / "interface"))
            import run_fly  # noqa: E402  (Mac-only: pandas/numba/connectome)
            self.brain = run_fly.build_brain()
            self.brain.stimulate([], [], duration_ms=10)  # JIT warm-up
            self.readout_fn = lambda rates: run_fly.rates_for_readout(self.brain, rates)

    def reset(self):
        self.flee.reset()
        if self.brain is not None and hasattr(self.brain, "reset_state"):
            self.brain.reset_state()

    def threats_as_npcs(self, state):
        """Closing hostile projectiles (arriving within threat_horizon
        ticks) as pseudo-NPC dicts the existing encoder understands."""
        p = state.get("player") or {}
        pcx, pcy = p.get("cx", p.get("x", 0.0)), p.get("cy", p.get("y", 0.0))
        out = []
        for pr in state.get("projectiles", []):
            rx, ry = pcx - pr["cx"], pcy - pr["cy"]
            dist = math.hypot(rx, ry) or 1.0
            if pr["type"] == PHANTASMAL_DEATHRAY:
                closing = 1.0  # a beam is "arriving" everywhere along it
            else:
                closing = (pr["vx"] * rx + pr["vy"] * ry) / dist  # px/tick toward the player
            if closing <= 0:
                continue
            if pr["type"] != PHANTASMAL_DEATHRAY and dist / closing > self.threat_horizon:
                continue
            out.append({"x": pr["cx"], "y": pr["cy"], "hostile": True, "from_projectile": pr["type"]})
        return out

    def act(self, state):
        p = state.get("player") or {}
        if not state.get("in_world") or p.get("dead"):
            return dict(RELEASE), {}
        self._ensure_brain()
        seen = dict(state)
        seen["npcs"] = list(state.get("npcs", [])) + self.threats_as_npcs(state)
        self.last_drives = self.encoder.compute_drives(seen, self.enc_params)
        ids, rates = self.encoder.encode(seen, self.enc_params)
        spikes = self.brain.stimulate(ids, rates, duration_ms=100)
        readout = self.readout_fn(spikes)
        a = self.flee.decode(readout, p.get("direction", 1))
        action = {**RELEASE, "left": bool(a.get("left")), "right": bool(a.get("right")),
                  "jump": bool(a.get("jump"))}
        return action, _sources(action, "brain")


def make_policy(name, seed=0, **kw):
    if name == "idle":
        return IdlePolicy()
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "scripted":
        return ScriptedDodger(**kw)
    if name == "fly":
        return FlyPolicy(**kw)
    raise ValueError(f"unknown policy '{name}'")
