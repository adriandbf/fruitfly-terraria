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
                 arena_height=2900.0, beam_width=40.0):
        self.boss_keep_away, self.ring = boss_keep_away, ring
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
            x += vx
            y += vy
            if y > self.floor - 21:  # arena floor (planter rows are passable; the bottom row isn't)
                y, vy = self.floor - 21, 0.0
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
                cost += 0.2 * ((h, mode) != self.prev)     # mild preference not to dither
                if best_cost is None or cost < best_cost:
                    best, best_cost = (h, mode), cost
        self.prev = best
        action = {**RELEASE, **movement.keys_for(*best)}
        return action, _sources(action, "script")


class FlyPolicy:
    """The fly: encoder -> FastBrain -> decoder, the run_fly.py pipeline, plus
    two hand-written Moon Lord layers (logged, and credited separately):

    1. ATTENTION (sensory side): the encoder reacts to the single NEAREST
       hostile, which in this fight is almost always the boss's core sitting
       ~130 px under the player (live 2026-10-09), so incoming projectiles
       barely registered. Instead the fly is shown the single MOST URGENT
       threat: the projectile or Moon Lord part (core, hands, head, True
       Eyes) with the shortest time to contact. It still reaches the brain
       only through the looming (LC4) channel; the brain decides the rest.

    2. FLEE BY FLYING (motor side): flying sideways is >2x faster than
       running (probe: ~8 vs 3.5 px/tick). While the decoder's flee reflex is
       active and has a direction from the brain, this layer keeps the fly
       airborne in the sawtooth the human demos use: hold jump (climb) until
       rising fast, then dive (down) until falling fast, repeat. Left/right
       still come from the brain (DNa02); jump/down here come from the rule
       and are credited "rule", not "brain". Outside a flee nothing changes.

    Not driven by the brain: up (fast climb) and `use` (the whip assist).

    brain/readout_fn can be injected for tests; by default the brain is
    built on first use (needs the Mac's venv and connectome data)."""
    name = "fly"

    CLIMB_UNTIL_VY = -9.0   # px/tick: climbing this fast -> start the dive
    DIVE_UNTIL_VY = 6.0     # px/tick: falling this fast -> climb again

    ABOVE_PX = 60.0         # a threat this far above the fly's centre counts as "from above"

    def __init__(self, enc_params=None, dec_params=None, brain=None, readout_fn=None,
                 threat_horizon_ticks=90, fly_flee=True, vertical="brain",
                 commit_ticks=0, dive_on_hz=40.0, waypoint=False, waypoint_px=600.0,
                 above_px=None):
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
        if above_px is not None:
            self.ABOVE_PX = float(above_px)
        self.fly_flee = fly_flee
        # vertical: "brain" = climb/dive from the connectome (LC4 -> Giant Fiber
        # = climb, LPLC4 -> DNp07 landing = dive; M3, 2026-10-09); "rule" = the
        # hand-written sawtooth; "off" = neither (decoder only).
        self.vertical = vertical if fly_flee else "off"
        import neurons
        self.neurons = neurons
        self.last_vertical = None
        # (1) direction commitment: once a flee direction is chosen, keep it for
        # commit_ticks decisions unless the brain's evidence has reversed for
        # that long too (live: the core sits right under the fly, so DNa02
        # flipped tick to tick and the fly jittered in place). 0 = off.
        self.commit_ticks = commit_ticks
        # (3) landing (DNp07) rate needed to dive (live: >=40 Hz on 71% of ticks)
        self.dive_on_hz = dive_on_hz
        # (2) waypoint: show the approach (sugar) channel a pseudo "player" at a
        # safe spot waypoint_px from the boss core, on the side the fly is on,
        # so the brain's own approach steering pulls it away from the boss when
        # it isn't fleeing. Hand-picked target; brain-driven steering.
        self.waypoint, self.waypoint_px = waypoint, waypoint_px
        self._commit_dir, self._commit_left, self._against = 0, 0, 0
        self.flee = decoder.StatefulDecoder(dec_params)
        self.last_drives = None
        self.last_threat = None
        self._diving = False

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
        self._diving = False
        self._commit_dir, self._commit_left, self._against = 0, 0, 0
        if self.brain is not None and hasattr(self.brain, "reset_state"):
            self.brain.reset_state()

    def threats_as_npcs(self, state):
        """Closing hostile projectiles (arriving within threat_horizon
        ticks) as pseudo-NPC dicts the existing encoder understands."""
        return [t for t, _ in self._threats(state) if t.get("from_projectile") is not None]

    def _threats(self, state):
        """(pseudo-NPC, ticks to contact) for every closing projectile and
        every Moon Lord part. Parts count even when not closing: the boss
        chases at the fly's own speed, so being close is the danger."""
        p = state.get("player") or {}
        pcx, pcy = p.get("cx", p.get("x", 0.0)), p.get("cy", p.get("y", 0.0))
        pvx, pvy = p.get("vx", 0.0), p.get("vy", 0.0)
        out = []
        for pr in state.get("projectiles", []):
            rx, ry = pcx - pr["cx"], pcy - pr["cy"]
            dist = math.hypot(rx, ry) or 1.0
            if pr["type"] == PHANTASMAL_DEATHRAY:
                closing = 1.0  # a beam is "arriving" everywhere along it
            else:
                closing = ((pr["vx"] - pvx) * rx + (pr["vy"] - pvy) * ry) / dist
            if closing <= 0:
                continue
            ttc = dist / closing
            if pr["type"] != PHANTASMAL_DEATHRAY and ttc > self.threat_horizon:
                continue
            out.append(({"x": pr["cx"], "y": pr["cy"], "hostile": True, "from_projectile": pr["type"]}, ttc))
        for n in state.get("npcs", []):
            if not (n.get("moon_lord_part") or n.get("type") == 400):
                continue
            cx, cy = n.get("cx", n["x"]), n.get("cy", n["y"])
            rx, ry = pcx - cx, pcy - cy
            dist = max(1.0, math.hypot(rx, ry) - 40.0)
            true_closing = ((n.get("vx", 0.0) - pvx) * rx + (n.get("vy", 0.0) - pvy) * ry) / (math.hypot(rx, ry) or 1.0)
            closing = max(1.0, true_closing)
            out.append(({"x": cx, "y": cy, "hostile": True, "from_npc": n.get("type"),
                         "approaching": true_closing > 0.5}, dist / closing))
        return out

    def most_urgent(self, state):
        threats = self._threats(state)
        return min(threats, key=lambda t: t[1])[0] if threats else None

    def _commit(self, want):
        """Direction commitment: keep the committed side for commit_ticks
        decisions; switch only after the brain has wanted the other side for
        commit_ticks decisions in a row (or the commitment has run out)."""
        if self._commit_left > 0 and want != self._commit_dir:
            self._against += 1
            if self._against < self.commit_ticks:
                self._commit_left -= 1
                return {"left": self._commit_dir < 0, "right": self._commit_dir > 0}
        if want != self._commit_dir or self._commit_left <= 0:
            self._commit_dir, self._commit_left = want, self.commit_ticks
        self._against = 0
        self._commit_left -= 1
        return {"left": want < 0, "right": want > 0}

    def urgent_by_elevation(self, state):
        """(most urgent threat from ABOVE, most urgent threat level/below)."""
        p = state.get("player") or {}
        pcy = p.get("cy", p.get("y", 0.0))
        # Only things actually approaching count as looming from above: the
        # boss's head hangs ~270 px overhead all fight, and treating it as a
        # threat made LPLC4 fire constantly (first brain-in-the-loop run:
        # 100% dive). Projectiles are only listed when closing anyway.
        above = [t for t in self._threats(state)
                 if t[0]["y"] < pcy - self.ABOVE_PX and t[0].get("approaching", True)]
        level = [t for t in self._threats(state) if t[0]["y"] >= pcy - self.ABOVE_PX]
        pick = lambda ts: min(ts, key=lambda t: t[1])[0] if ts else None
        return pick(above), pick(level)

    def _lplc4_drive(self, state, threat):
        """LPLC4 ids/rates for a threat from above: same distance->rate and
        left/right split as the encoder's LC4 channel."""
        if threat is None:
            return [], []
        enc, n = self.encoder, self.neurons
        prm = {**enc.DEFAULT_PARAMS, **(self.enc_params or {})}
        p = state["player"]
        dist = math.hypot(threat["x"] - p.get("cx", p["x"]), threat["y"] - p.get("cy", p["y"]))
        rate = enc._range_rate(dist, prm["threat_max_range_px"], prm["max_rate_hz"])
        side = "right" if threat["x"] >= p.get("cx", p["x"]) else "left"
        return enc._lateralized(rate, side, n.LPLC4_LOOMING_LEFT, n.LPLC4_LOOMING_RIGHT, prm["contra_fraction"])

    def act(self, state):
        p = state.get("player") or {}
        if not state.get("in_world") or p.get("dead"):
            return dict(RELEASE), {}
        self._ensure_brain()
        seen = dict(state)
        extra_ids, extra_rates = [], []
        if self.vertical == "brain":
            above, level = self.urgent_by_elevation(state)
            self.last_threat = {"above": above, "level": level}
            seen["npcs"] = [level] if level else []          # LC4 channel: level/below threat
            extra_ids, extra_rates = self._lplc4_drive(state, above)  # LPLC4 channel: threat from above
        else:
            threat = self.most_urgent(state)
            self.last_threat = threat
            seen["npcs"] = [threat] if threat else []   # attention: one threat, the most urgent
        if self.waypoint:
            core = next((n for n in state.get("npcs", []) if n.get("type") == 398), None)
            if core is not None:
                side = 1.0 if p.get("cx", p["x"]) >= core.get("cx", core["x"]) else -1.0
                seen["players"] = [{"x": core.get("cx", core["x"]) + side * self.waypoint_px,
                                    "y": p.get("y", 0.0), "name": "waypoint"}]
        self.last_drives = self.encoder.compute_drives(seen, self.enc_params)
        ids, rates = self.encoder.encode(seen, self.enc_params)
        spikes = self.brain.stimulate(list(ids) + list(extra_ids), list(rates) + list(extra_rates), duration_ms=100)
        readout = self.readout_fn(spikes)
        a = self.flee.decode(readout, p.get("direction", 1))
        action = {**RELEASE, "left": bool(a.get("left")), "right": bool(a.get("right")),
                  "jump": bool(a.get("jump"))}
        sources = _sources(action, "brain")

        fleeing = self.flee.last_mode == "flee" and self.flee.last_flee_dir != 0
        if self.commit_ticks and (action.get("left") or action.get("right")):
            action = {**action, **self._commit(1 if action.get("right") else -1)}
        if self.vertical == "brain":
            n = self.neurons
            mean = lambda ids: sum(readout.get(f, 0.0) for f in ids) / max(len(ids), 1)
            land_l, land_r = mean(n.DNP07_LANDING_LEFT), mean(n.DNP07_LANDING_RIGHT)
            gf = mean(n.DNP01_JUMP)
            landing = max(land_l, land_r)
            if landing >= self.dive_on_hz and landing > gf:
                # Dive: DNp07 is ipsilateral (LPLC4_left -> DNp07_L), so go AWAY
                # from the side whose DNp07 fires harder -- unless a committed
                # direction is running, which keeps its course.
                go_right = land_l > land_r
                if self.commit_ticks and self._commit_left > 0 and self._commit_dir:
                    go_right = self._commit_dir > 0
                action = {**RELEASE, "left": not go_right, "right": go_right, "down": True}
                self.last_vertical = "dive"
            elif fleeing:
                action = {**action, "jump": True}   # takeoff wins: keep climbing
                self.last_vertical = "climb"
            else:
                self.last_vertical = None
            self.last_dn = {"gf": round(gf, 1), "dnp07_l": round(land_l, 1), "dnp07_r": round(land_r, 1)}
            return action, _sources(action, "brain")
        if self.vertical == "rule" and fleeing:
            vy = p.get("vy", 0.0)
            if self._diving and vy >= self.DIVE_UNTIL_VY:
                self._diving = False
            elif not self._diving and vy <= self.CLIMB_UNTIL_VY:
                self._diving = True
            action["jump"], action["down"] = (False, True) if self._diving else (True, False)
            sources = {k: v for k, v in sources.items() if k in ("left", "right")}
            sources["down" if self._diving else "jump"] = "rule"
        else:
            self._diving = False
        return action, sources


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
    if name == "fly":
        return FlyPolicy(**kw)
    if name == "human":
        return HumanPolicy()
    raise ValueError(f"unknown policy '{name}'")
