"""Fly senses v3 (2026-10-10): see WHAT is approaching, from WHERE, and HOW SOON.

Why: live, the v2 scene fly's senses registered the homing eyes ~1 s before
they hit (72% of eye hits had >= 0.5 s of warning), but it changed its move
in the last 0.5 s before only 4 of 25 eye hits; bolts gave only ~0.2 s of
warning because they were coded by distance, not by time to contact. The
user's call: the fly should observe what's coming and dodge it.

Every channel is a real FlyWire visual population per eye (left/right =
which side of the character the thing is on), each a different feature
detector driving its own descending neurons in the np_sim sweep:
  LC4    big things (boss parts) level with the fly       -> DNp04/DNp02/DNp11
  LPLC4  anything approaching from ABOVE                   -> DNb05/DNp07
  LC6    big things BELOW (and the floor)                  -> DNp70/DNp01/DNp103
  LC11   small objects (projectiles) level                 -> DNg100
  LC12   small objects BELOW                               -> DNpe045/DNge054
  LPLC2  anything about to hit (time to contact < 0.5 s)   -> DNp103/DNp04
  LC22   the deathray beam sweeping close                  -> DNp26/DNp05
  VS/HS  self-motion (optic flow), as in v2
Rates code TIME TO CONTACT (closing speed matters: a fast bolt far away can
be more urgent than a slow sphere close by), as looming detectors do, plus a
proximity floor for big things that hover close without closing (the boss
chases at the fly's own speed).
"""
import math

PHANTASMAL_DEATHRAY = 455
LOOM = ("LC4", "LPLC4", "LC6", "LC11", "LC12", "LPLC2", "LC22")
CHANNELS = [(t, s) for t in LOOM for s in (-1, 1)]

TTC_SMALL = 60.0      # game ticks: a projectile 1 s from contact starts to register
TTC_BIG = 90.0
TTC_IMMINENT = 30.0   # LPLC2: under 0.5 s
NEAR_BIG = 500.0      # px: proximity floor for boss parts
NEAR_SMALL = 200.0    # px: proximity floor for projectiles hovering close without closing
BEAM_PX = 260.0       # deathray: perpendicular distance at which LC22 starts
BEAM_RAD = 0.6        # ... or angular gap (from the beam's origin) at which it starts (v3.1)
MX = 150.0            # Hz


def _ttc(rx, ry, rvx, rvy):
    """Time to contact (ticks) of something at (rx, ry) relative to the fly,
    moving at (rvx, rvy) relative to it; inf if not closing."""
    d = math.hypot(rx, ry) or 1.0
    closing = -(rx * rvx + ry * rvy) / d
    return d / closing if closing > 0.05 else float("inf"), d


def scene_rates_v3(state, arena=None, above_px=60.0, sumw=0.5):
    """{(type, side): Hz} for LOOM channels + ("VS"/"HS", side) self-motion."""
    p = state["player"]
    pcx, pcy = p["cx"], p["cy"]
    pvx, pvy = p.get("vx", 0.0), p.get("vy", 0.0)
    acc = {k: [] for k in CHANNELS}

    def add(t, side, r):
        if r > 0:
            acc[(t, side)].append(min(1.0, r))

    for pr in state.get("projectiles", []):
        rx, ry = pr["cx"] - pcx, pr["cy"] - pcy
        side = 1 if rx >= 0 else -1
        if pr["type"] == PHANTASMAL_DEATHRAY:
            ux, uy = pr["vx"], pr["vy"]
            n = math.hypot(ux, uy) or 1.0
            ux, uy = ux / n, uy / n
            qx, qy = pcx - pr["cx"], pcy - pr["cy"]
            along = qx * ux + qy * uy
            if along > 0:
                perp = abs(-qx * uy + qy * ux)
                # side the beam is on, relative to the fly
                bx = pr["cx"] + ux * along
                # v3.1 (2026-10-12): also by ANGULAR gap from the beam's origin. Live, LC22 was
                # silent 0.3 s before 18 of 49 deathray hits: a beam sweeping in at ~0.0116
                # rad/tick from ~1 rad away crosses the old 260 px band in ~0.4 s. Within
                # BEAM_RAD of the fly's bearing it now ramps up (~0.9 s of warning).
                gap = abs(math.atan2(-qx * uy + qy * ux, along))
                add("LC22", 1 if bx >= pcx else -1, max(1.0 - perp / BEAM_PX, 1.0 - gap / BEAM_RAD))
            continue
        ttc, d = _ttc(rx, ry, pr.get("vx", 0.0) - pvx, pr.get("vy", 0.0) - pvy)
        r = max(1.0 - ttc / TTC_SMALL, 1.0 - d / NEAR_SMALL)   # time to contact, plus a close-by floor
        if ry < -above_px:
            add("LPLC4", side, r)
        elif ry > above_px:
            add("LC12", side, r)
        else:
            add("LC11", side, r)
        if ttc < TTC_IMMINENT:
            add("LPLC2", side, 1.0 - ttc / TTC_IMMINENT)
    for n in state.get("npcs", []):
        if not (n.get("moon_lord_part") or n.get("type") == 400):
            continue
        rx, ry = n.get("cx", n["x"]) - pcx, n.get("cy", n["y"]) - pcy
        side = 1 if rx >= 0 else -1
        ttc, d = _ttc(rx, ry, n.get("vx", 0.0) - pvx, n.get("vy", 0.0) - pvy)
        d = max(1.0, d - 40.0)
        r = max(1.0 - ttc / TTC_BIG, 1.0 - d / NEAR_BIG)
        if ry < -above_px:
            add("LPLC4", side, r)
        elif ry > above_px:
            add("LC6", side, r)
        else:
            add("LC4", side, r)
        if ttc < TTC_IMMINENT:
            add("LPLC2", side, 1.0 - ttc / TTC_IMMINENT)
    # Arena surfaces: the ceiling looms from above, the floor from below, walls level.
    ep = state.get("episode") or {}
    mark = ep.get("arena")
    if mark and arena:
        a = arena
        feet = mark["y"] + (p.get("height") or 42)
        top = feet - a["height_px"]
        if ep.get("full_gravity_y") is not None:
            top = max(top, ep["full_gravity_y"])
        centre = mark["x"] + (p.get("width") or 20) / 2
        for dist, vel, t in ((pcy - top, -pvy, "LPLC4"), (feet - pcy, pvy, "LC6")):
            ttc = dist / vel if vel > 0.05 else float("inf")
            r = max(1.0 - ttc / TTC_BIG, 1.0 - dist / 250.0)
            add(t, -1, r)
            add(t, 1, r)
        for dist, vel, side in ((pcx - (centre - a["half_width_px"]), -pvx, -1),
                                ((centre + a["half_width_px"]) - pcx, pvx, 1)):
            ttc = dist / vel if vel > 0.05 else float("inf")
            add("LC4", side, max(1.0 - ttc / TTC_BIG, 1.0 - dist / 250.0))
    out = {}
    for k, rs in acc.items():
        if rs:
            top = max(rs)
            out[k] = min(MX, MX * (top + sumw * (sum(rs) - top)))
        else:
            out[k] = 0.0
    out[("VS", -1)] = out[("VS", 1)] = min(250.0, max(0.0, 100.0 - 12.0 * pvy))
    out[("HS", -1)] = min(250.0, max(0.0, 100.0 - 12.0 * pvx))
    out[("HS", 1)] = min(250.0, max(0.0, 100.0 + 12.0 * pvx))
    return out


def features(rates):
    """Rates -> teacher feature vector (looming channels /150, self-motion centred)."""
    f = [rates[k] / MX for k in CHANNELS]
    f += [(rates[("VS", 1)] - 100.0) / 120.0, (rates[("HS", 1)] - rates[("HS", -1)]) / 240.0]
    return f


N_W = 2 * (len(CHANNELS) + 3) + 3   # h and v weights over features + bias, thr, hold, override


class V3Policy:
    """Linear map from the v3 senses to moves -- the TEACHER used to train the
    fly's readout (credited "script"; it reads the senses directly, no brain).
    Same move rule the fly uses: a sideways direction is held for `hold`
    decisions once chosen, unless the drive the other way exceeds `override`
    (a fast dodge); vertical: climb (v > 0.3), dive (v < -0.3), else glide."""
    name = "senses3"

    def __init__(self, weights, arena=None):
        self.w = [float(x) for x in weights]
        self.arena = arena
        self.reset()

    def reset(self):
        self.dir, self.hold = 1, 0

    def drives(self, state):
        f = features(scene_rates_v3(state, self.arena)) + [1.0]
        n = len(f)
        h = sum(a * b for a, b in zip(f, self.w[0:n]))
        v = sum(a * b for a, b in zip(f, self.w[n:2 * n]))
        return h, v

    def decide(self, h, v):
        n2 = 2 * (len(CHANNELS) + 3)
        thr, hold, override = abs(self.w[n2]), abs(self.w[n2 + 1]) * 20, abs(self.w[n2 + 2]) * 2
        want = 1 if h > 0 else -1
        if self.hold > 0 and not (want != self.dir and abs(h) > override):
            self.hold -= 1
        elif abs(h) > thr:
            if want != self.dir:
                self.hold = int(hold)
            self.dir = want
        vert = {"jump": True} if v > 0.3 else ({"down": True} if v < -0.3 else {"up": True})
        return {"left": self.dir < 0, "right": self.dir > 0, "jump": False, "up": False, "down": False,
                "use": False, **vert}

    def act(self, state):
        p = state.get("player") or {}
        if not state.get("in_world") or p.get("dead") or "cx" not in p:
            return {k: False for k in ("left", "right", "jump", "use", "up", "down")}, {}
        a = self.decide(*self.drives(state))
        return a, {k: "script" for k, v in a.items() if v}
