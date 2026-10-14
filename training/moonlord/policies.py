"""Policies for the Moon Lord harness (Goal 2). Every policy has the same
shape:

    policy.reset()                       # start of an episode
    action, sources = policy.act(state)  # once per 10 Hz tick

`action` is a set_action dict (left/right/jump/use/up/down). `sources` maps
each PRESSED key to who decided it -- "brain" (the fly), "script" (a
hand-written policy), "random". harness.py totals these into the per-episode
credit ledger, so a result can always say how much of the behaviour was the
fly's.

Baselines: IdlePolicy (the build standing still), RandomPolicy and
ScriptedDodger (a hand-written dodger), plus HumanPolicy for recording your
own play. The fly itself is scene_fly3.SceneFly3Policy.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import movement  # noqa: E402

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
            self.current = {**RELEASE, **movement.keys_for(h, self.rng.choice(movement.MODES))}
            self.hold = self.rng.randint(1, 8)
        self.hold -= 1
        return dict(self.current), _sources(self.current, "random")


class ScriptedDodger:
    """Hand-written dodger, the 'ceiling' baseline. Each tick it tries the 12
    combinations of (left / none / right) x (fast climb / climb / glide / dive),
    rolls each forward HORIZON ticks with the probe-fitted movement model
    (movement.py), moves every projectile in a straight line (homing eyes
    steered, the deathray swept at its measured rate), and picks the
    cheapest. Costs: projectile overlap (by damage), touching any Moon Lord
    part, leaving a ring around the core (so it circles the boss instead of
    standing inside it -- live, idle/fly sat ~140 px from the core), leaving
    the arena, a climb that would coast past the arena top, and standing
    still horizontally."""
    name = "scripted"

    HORIZON = 36   # ticks (0.6 s) of lookahead
    STEP = 3       # ticks per lookahead sample
    MARGIN = 14.0  # px of extra clearance wanted around each projectile

    # ring/keep-away: (450, 900) px / 250 px. A retune to (350, 700) / 150 looked far
    # better OFFLINE (median 109 s vs 43 s) but halved survival LIVE (20.4 s vs
    # 40.5 s median, 5 fights each, 2026-10-09; more head/body contact hits), and
    # the fake still prefers it even with head contact added -- so the fake is
    # not trusted for this knob; the live result wins.
    def __init__(self, boss_keep_away=250.0, ring=(450.0, 900.0), arena_half_width=1300.0,
                 arena_height=2900.0, beam_width=40.0, dither_cost=0.2, row_spacing_px=384.0, rows=12):
        self.boss_keep_away, self.ring = boss_keep_away, ring
        # Cost of changing (direction, vertical mode) from the last tick. 0.2
        # (default) let it flip direction 1.9x/s live; the human sweeps for
        # seconds at a time (0.27 flips/s, 2026-10-09 large world).
        self.dither_cost = dither_cost
        self.row_spacing_px, self.rows = row_spacing_px, rows
        self.arena_half_width, self.arena_height = arena_half_width, arena_height
        self.beam_width = beam_width
        self.reset()

    def reset(self):
        self.prev = (0, "down")
        self.home = self.floor = None
        self.ray_prev = None  # last seen deathray angle, to estimate its sweep rate

    def _rollout(self, p, h, mode):
        x, y = p["cx"], p["cy"]
        vx, vy = p.get("vx", 0.0), p.get("vy", 0.0)
        ground = bool(p.get("on_ground"))
        pts = []
        for t in range(1, self.HORIZON + 1):
            vx, vy = movement.step(vx, vy, h, mode, ground)
            ground = False
            feet0 = y + 21
            x += vx
            y += vy
            if y > self.floor - 21:  # arena floor
                y, vy = self.floor - 21, 0.0
            elif vy > 0 and self.row_spacing_px:
                # Planter rows (live): falling with down held passes through but
                # the row stops the fall (vy -> ~1); any other mode lands on it.
                r = int((self.floor - feet0) // self.row_spacing_px)
                row = self.floor - r * self.row_spacing_px
                if 1 <= r <= self.rows and feet0 <= row < y + 21:
                    if mode == "down":
                        vy = min(vy, 1.0)
                    else:
                        y, vy, ground = row - 21, 0.0, True
            pts.append((t, x, y))
        return pts, vy

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
        hw, hh = (p.get("width") or 20) / 2, (p.get("height") or 42) / 2
        arena = (state.get("episode") or {}).get("arena")
        if self.home is None:
            self.home = arena["x"] + hw if arena else p["cx"]
            self.floor = (arena["y"] + 2 * hh) if arena else p["cy"] + hh
        top = self.floor - self.arena_height
        bosses = [n for n in state.get("npcs", []) if n.get("boss") or n.get("moon_lord_part")]
        core = next((n for n in bosses if n.get("type") == 398), None)

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
            for mode in movement.MODES:
                pts, vy_end = self._rollout(p, h, mode)
                eyes = self._eye_paths([pr for pr in projs if pr["type"] == PHANTASMAL_EYE], pts)
                cost = 0.0
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
                            rx, ry = x - pr["cx"], y - pr["cy"]
                            along = rx * ux + ry * uy
                            perp = abs(-rx * uy + ry * ux)
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
                    for boss in bosses:
                        d = math.hypot(x - boss.get("cx", boss["x"]), y - boss.get("cy", boss["y"]))
                        if d < self.boss_keep_away:
                            cost += w * 2.0 * (self.boss_keep_away - d)
                    if core is not None:
                        d = math.hypot(x - core["cx"], y - core["cy"])
                        lo, hi = self.ring
                        if d < lo:
                            cost += w * 0.3 * (lo - d)
                        elif d > hi:
                            cost += w * 0.1 * (d - hi)
                    off = abs(x - self.home) - self.arena_half_width
                    if off > 0:
                        cost += w * off
                    above = top - y
                    if above > 0:
                        cost += w * above
                # A climb keeps coasting long after letting go: charge where it would end up.
                apex = pts[-1][2] - movement.coast_rise(vy_end)
                if apex < top:
                    cost += 2.0 * (top - apex)
                cost += 1.0 * (h == 0)                     # keep moving sideways (the human demos almost never stop)
                cost += self.dither_cost * ((h, mode) != self.prev)  # preference not to dither
                if best_cost is None or cost < best_cost:
                    best, best_cost = (h, mode), cost
        self.prev = best
        action = {**RELEASE, **movement.keys_for(*best)}
        return action, _sources(action, "script")


class HumanPolicy:
    """You, playing. Used with the mod's passthrough mode: the harness sends
    no movement, your keyboard drives the character, and this reads back the
    controls you actually held (get_state.debug) so the demo is logged in the
    same format as every other policy, credited "human"."""
    name = "human"
    passthrough = True

    def reset(self):
        pass

    def act(self, state):
        d = state.get("debug") or {}
        action = {**RELEASE, "left": bool(d.get("control_left")), "right": bool(d.get("control_right")),
                  "jump": bool(d.get("control_jump")), "up": bool(d.get("control_up")),
                  "down": bool(d.get("control_down")), "use": bool(d.get("control_use_item"))}
        return action, _sources(action, "human")


def make_policy(name, seed=0, **kw):
    if name == "idle":
        return IdlePolicy()
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "scripted":
        return ScriptedDodger(**kw)
    if name == "human":
        return HumanPolicy()
    raise ValueError(f"unknown policy '{name}'")
