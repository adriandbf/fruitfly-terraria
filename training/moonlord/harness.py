"""Goal 2 (Moon Lord) episode harness: runs fights (or dodge-gym drills)
back to back with nobody at the keyboard, and scores each one.

One episode:
  1. wait until the player is alive (the game's own respawn after a death)
  2. reset_episode: despawn hostiles and hostile projectiles, heal, clear
     debuffs, remove items on the ground, move to the arena mark, zero the
     mod's counters; then the inventory is wiped and set back to the loadout
  3. SETUP (scripted, logged as "setup", never credited to any policy):
     select each summon staff and hold `use` until the minion slots fill
     then select the whip slot (if the loadout has one)
  4. fight mode: spawn the Moon Lord and wait for it to appear
     gym mode:   no boss; volleys of Moon Lord projectiles are fired at the
                 player on a seeded schedule (stage M2's dodge gym)
  5. 10 Hz loop: get_state -> policy.act -> set_action, every tick logged.
     With the loadout's whip_assist on, a scripted assist holds `use` aimed
     at the nearest hittable Moon Lord part (credited "assist", not "brain")
  6. end on: boss killed (win), player dead (loss), boss gone without a
     kill (despawn), or the time cap (timeout; in gym mode that's survival)

Each episode appends a summary line to logs/moonlord/results.jsonl
(results_fake.jsonl for --fake runs), with
the outcome, survival time, damage taken and the credit ledger (how many
ticks each source -- brain / script / random -- pressed a key).

Usage (Mac, mod loaded, in a world with any fresh character):
  python training/moonlord/harness.py --setup-character       # once: loadout.json gear + max stats
  python training/moonlord/harness.py --build-arena           # once per world: platforms + arena mark
  python training/moonlord/harness.py --mark-arena            # re-mark after a game restart
  python training/moonlord/harness.py --policy idle --episodes 5
  python training/moonlord/harness.py --policy fly --episodes 20 --keep-focus
  python training/moonlord/harness.py --policy scripted --mode gym --episodes 50
  python training/moonlord/harness.py --fake --policy scripted --episodes 10   # offline, no game
  python training/moonlord/harness.py --policy human                 # record YOU fighting (you steer)
  python training/moonlord/harness.py --policy human --mode free --max-seconds 60   # movement demo, no boss
"""
import argparse
import json
import re
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "bridge"))

from policies import KEYS, RELEASE, make_policy  # noqa: E402
import movement  # noqa: E402

TICK_HZ = 10
TICK_S = 1.0 / TICK_HZ
LOG_DIR = ROOT / "logs" / "moonlord"
LOADOUT = HERE / "loadout.json"
MOON_LORD_CORE = 398
PHANTASMAL_BOLT, PHANTASMAL_SPHERE, PHANTASMAL_EYE = 462, 454, 452

DEFAULTS = {
    "mode": "fight",
    "max_seconds": {"fight": 300.0, "gym": 20.0, "free": 120.0},
    "countdown_s": 0,            # seconds of warning before the boss spawns (human demos)
    "summon_slots": [],          # hotbar slots holding summon staffs, e.g. [0, 1]
    "heal_at_hp": 300,           # heal assist: quick-heal at or below this HP (when not potion-sick)
    "restock_potions": 10,       # healing potions put in the inventory at every reset
    "heal_potion": "SuperHealingPotion",
    "whip_slot": None,           # hotbar slot of the whip
    "whip_assist": False,        # scripted whip swings at the Moon Lord (credited "assist", never "brain")
    "whip_strategy": "nearest",  # nearest | balanced (hands to hold_frac, head, hands, core)
    "whip_hold_frac": 0.15,
    "buffs": [],                 # buff names re-applied at every reset (potions; furniture buffs come from the arena)
    "buff_seconds": 900,
    "summon_hold_ticks": 120,    # up to 12 s of `use` per slot; stops as soon as the minion slots are full
    "arena": None,               # loadout arena dims, for the leash (half_width_tiles, rows, row_spacing_tiles)
    "boss_type": MOON_LORD_CORE,
    "boss_appear_timeout_s": 10.0,
    "respawn_timeout_s": 90.0,
    # Dodge gym (mode=gym)
    "gym_period_s": 1.0,         # one volley per second
    "gym_volley": 3,             # projectiles per volley
    "gym_distance": 600.0,       # px from the player where a volley starts
    "gym_speed": 10.0,           # px/tick
    "gym_spread": 0.15,          # radians of aim error per projectile
    "gym_damage": 50,
    # Bolts only by default: they fly straight. Spheres and eyes normally
    # take orders from the Moon Lord's AI (aim, release, homing), so spawned
    # on their own in the real game they may hover or vanish -- try them
    # live before relying on them (--gym-types 462,454,452).
    "gym_types": [PHANTASMAL_BOLT],
}


class HarnessError(RuntimeError):
    pass


def connect(fake=False, fake_config=None):
    if fake:
        from fake_bridge import FakeMoonLordBridge
        return FakeMoonLordBridge(fake_config)
    from dumb_bot import Bridge
    return Bridge()


def _ok(resp, what):
    if not resp.get("ok"):
        raise HarnessError(f"{what} failed: {resp.get('error')}")
    return resp


class Clock:
    """Paces the loop at 10 Hz against the real game; a no-op for the fake
    bridge (its time only moves on set_action)."""

    def __init__(self, simulated):
        self.simulated = simulated
        self.t0 = time.monotonic()

    def start_tick(self):
        self.t0 = time.monotonic()

    def end_tick(self):
        if not self.simulated:
            time.sleep(max(0.0, TICK_S - (time.monotonic() - self.t0)))

    def wait(self, seconds):
        if not self.simulated:
            time.sleep(seconds)


def wait_alive(bridge, clock, timeout_s):
    """Block until the player is alive. With the fake bridge, time only
    moves on set_action, so step it with released keys."""
    waited = 0.0
    while True:
        st = bridge.request({"cmd": "get_state"})
        if not st.get("in_world"):
            raise HarnessError("not in a world (load the arena world first)")
        p = st["player"]
        if not (p.get("dead") or p.get("ghost")):
            return st
        if waited > timeout_s:
            raise HarnessError(f"player still dead after {timeout_s}s")
        if getattr(bridge, "simulated", False):
            bridge.request({"cmd": "set_action", **RELEASE})
            waited += TICK_S
        else:
            time.sleep(0.5)
            waited += 0.5


def summon_minions(bridge, clock, slots, hold_ticks, log):
    """Scripted setup: cast each summon staff until minion slots are full.
    Returns the number of setup ticks used."""
    ticks = 0
    for slot in slots:
        _ok(bridge.request({"cmd": "select_slot", "slot": slot}), f"select_slot {slot}")
        for _ in range(hold_ticks):
            clock.start_tick()
            st = bridge.request({"cmd": "get_state"})
            p = st.get("player") or {}
            if p.get("max_minions") and p.get("minions", 0) >= p["max_minions"]:
                break
            bridge.request({"cmd": "set_action", **RELEASE, "use": True})
            log({"phase": "setup", "slot": slot, "minions": p.get("minions")})
            ticks += 1
            clock.end_tick()
        bridge.request({"cmd": "set_action", **RELEASE})
    return ticks


def leash(state, action, arena_cfg, margin_px=48.0):
    """Arena leash, a scripted assist applied the same way to every policy:
    drops `up`/`jump` at the arena's top row, and `left`/`right` at its side
    edges, so nothing flies off to the world's ceiling (seen live: random
    keys climbed 176 tiles every fight). Returns (action, changed keys)."""
    ep = state.get("episode") or {}
    mark, p = ep.get("arena"), state.get("player") or {}
    if not mark or not arena_cfg or "cx" not in p:
        return action, []
    half = arena_cfg.get("half_width_tiles", 100) * 16.0
    height = arena_cfg.get("rows", 4) * arena_cfg.get("row_spacing_tiles", 12) * 16.0
    # The world's own height limit can sit below the arena's top row (live:
    # nothing rose more than 176 tiles above the floor of the 192-tile arena),
    # so the leash also stops at max_rise_tiles.
    if arena_cfg.get("max_rise_tiles"):
        height = min(height, arena_cfg["max_rise_tiles"] * 16.0)
    feet = mark["y"] + (p.get("height") or 42)
    centre_x = mark["x"] + (p.get("width") or 20) / 2
    changed = []
    out = dict(action)
    top = feet - height + margin_px
    # Never into reduced gravity: above full_gravity_y (reported by the mod)
    # gravity fades toward space, slowing falls (user, small world 2026-10-09).
    if ep.get("full_gravity_y") is not None:
        top = max(top, ep["full_gravity_y"] + margin_px)
    # A fast climb coasts upward for seconds after letting go (probe: up to
    # 18.7 px/tick vs ~0.1 px/tick^2 of gravity), so judge by where the climb
    # would END: if that's past the top, turn the climb into a hover.
    apex = p["cy"] - movement.coast_rise(p.get("vy", 0.0))
    if p["cy"] < top:
        for k in ("up", "jump"):
            if out.get(k):
                out[k] = False
                changed.append(k)
    elif apex < top:
        # There is no hover (probe v2): jump alone keeps climbing, so stop both.
        for k in ("up", "jump"):
            if out.get(k):
                out[k] = False
                changed.append(k)
    if p["cx"] < centre_x - half + margin_px and out.get("left"):
        out["left"] = False
        changed.append("left")
    if p["cx"] > centre_x + half - margin_px and out.get("right"):
        out["right"] = False
        changed.append("right")
    return out, changed


MOON_LORD_HEAD, MOON_LORD_HAND, TRUE_EYE = 396, 397, 400


def moon_lord_progress(state):
    """How much of the Moon Lord is left. A destroyed eye (head or hand)
    reports FULL hp again and spawns a True Eye of Cthulhu (type 400), so
    raw part hp undercounts progress (live 2026-10-09: a human fight with all
    three eyes destroyed read as 97% left). With N True Eyes alive, the N
    eye-parts reading the most hp are the destroyed ones and count as 0.
    Returns {hp_left, hp_total, eyes_destroyed} or None if no Moon Lord."""
    npcs = state.get("npcs", [])
    eyes = [n for n in npcs if n.get("type") in (MOON_LORD_HEAD, MOON_LORD_HAND)]
    core = next((n for n in npcs if n.get("type") == MOON_LORD_CORE), None)
    if core is None and not eyes:
        return None
    destroyed = min(len(eyes), sum(1 for n in npcs if n.get("type") == TRUE_EYE))
    alive = sorted(eyes, key=lambda n: n.get("hp", 0))[:len(eyes) - destroyed]
    total = (core or {}).get("max_hp", 0) + sum(n.get("max_hp", 0) for n in eyes)
    left = (core or {}).get("hp", 0) + sum(n.get("hp", 0) for n in alive)
    return {"hp_left": left, "hp_total": total, "eyes_destroyed": destroyed}


def _alive_eyes(state):
    """Head/hand eye parts not yet destroyed (destroyed ones read FULL hp and
    have a True Eye; see moon_lord_progress)."""
    npcs = state.get("npcs", [])
    eyes = [n for n in npcs if n.get("type") in (MOON_LORD_HEAD, MOON_LORD_HAND)]
    destroyed = min(len(eyes), sum(1 for n in npcs if n.get("type") == TRUE_EYE))
    return sorted(eyes, key=lambda n: n.get("hp", 0))[:len(eyes) - destroyed]


def whip_target(state, strategy="nearest", hold_frac=0.15):
    """The Moon Lord part a whip swing should aim at. Minions attack the
    whip-tagged enemy, so this chooses where all the summon damage goes.
      nearest:  the nearest part that can take damage (the original assist)
      balanced: the player's strategy (user, 2026-10-09) -- bring the HAND
                eyes down to hold_frac of their hp but don't destroy them
                (a destroyed eye becomes a True Eye and the damage coming at
                you jumps), destroy the HEAD eye, then finish the hands, then
                the core. Falls back down the list when a part can't be hit.
    Returns (x, y) world pixels or None."""
    p = state.get("player") or {}
    px, py = p.get("cx", p.get("x", 0.0)), p.get("cy", p.get("y", 0.0))
    parts = [n for n in state.get("npcs", []) if n.get("moon_lord_part") or n.get("boss")]
    if not parts:
        return None
    hittable = [n for n in parts if not n.get("dont_take_damage")] or parts
    xy = lambda n: (n.get("cx", n["x"]), n.get("cy", n["y"]))
    nearest = min(hittable, key=lambda n: (xy(n)[0] - px) ** 2 + (xy(n)[1] - py) ** 2)
    if strategy != "balanced":
        return xy(nearest)
    alive = _alive_eyes(state)
    frac = lambda n: n.get("hp", 0) / max(n.get("max_hp", 1), 1)
    hands = sorted([n for n in alive if n.get("type") == MOON_LORD_HAND], key=frac, reverse=True)
    head = [n for n in alive if n.get("type") == MOON_LORD_HEAD]
    core = [n for n in parts if n.get("type") == MOON_LORD_CORE]
    order = ([h for h in hands if frac(h) > hold_frac] + head
             + sorted(hands, key=frac) + core)
    for n in order:
        if not n.get("dont_take_damage"):
            return xy(n)
    return xy(nearest)


def spawn_boss(bridge, clock, boss_type, timeout_s):
    _ok(bridge.request({"cmd": "spawn_npc", "type": boss_type}), "spawn_npc")
    waited = 0.0
    while waited <= timeout_s:
        st = bridge.request({"cmd": "get_state"})
        if any(n.get("type") == boss_type for n in st.get("npcs", [])):
            return st
        clock.wait(0.2)
        waited += 0.2
        if getattr(bridge, "simulated", False):
            bridge.request({"cmd": "set_action", **RELEASE})
    raise HarnessError(f"boss type {boss_type} did not appear within {timeout_s}s")


def gym_volley(bridge, rng, cfg):
    """Fire one volley of projectiles at the player from a random direction.
    Uses dx/dy (offsets from the player's centre), so it needs no state."""
    ang = rng.uniform(0, 2 * math.pi)
    d = cfg["gym_distance"]
    sx, sy = d * math.cos(ang), d * math.sin(ang)
    for _ in range(cfg["gym_volley"]):
        aim = math.atan2(-sy, -sx) + rng.uniform(-cfg["gym_spread"], cfg["gym_spread"])
        t = rng.choice(cfg["gym_types"])
        _ok(bridge.request({"cmd": "spawn_projectile", "type": t, "dx": sx, "dy": sy,
                            "vx": cfg["gym_speed"] * math.cos(aim), "vy": cfg["gym_speed"] * math.sin(aim),
                            "damage": cfg["gym_damage"]}), "spawn_projectile")


def _trim_state(st):
    """Logged state without the tile grid (useless in a sky arena, large)."""
    return {k: v for k, v in st.items() if k != "tiles"}


def nurse_step(bridge, st, cfg, ns, log):
    """Nurse healing protocol (user design 2026-10-11), a scripted assist like
    the potion heal: when HP < hp_below and potion sickness is on, use the
    Magic Mirror (back to the bed under the arena), get the Nurse's heal, then
    fly straight up the shaft. Returns None when it isn't driving this tick,
    else 1 on the tick a heal lands (0 otherwise)."""
    n = cfg["nurse"]
    p = st.get("player") or {}
    house = (st.get("episode") or {}).get("nurse_house")
    phase = ns.get("phase")

    def back_to_whip(cooldown):
        # Failed or finished trip: whip back in hand, and no new trip for a while
        # (live 2026-10-11: a failed mirror retriggered the next tick, over and over).
        ns.clear()
        ns["cooldown"] = cooldown
        ns["fails"] = fails
        if cfg.get("whip_slot") is not None:
            bridge.request({"cmd": "select_slot", "slot": cfg["whip_slot"]})

    fails = ns.get("fails", 0)
    if phase is None:
        if ns.get("cooldown", 0) > 0:
            ns["cooldown"] -= 1
            return None
        # Only go home when home is the bed (spawn set there), the Nurse is alive,
        # and the protocol hasn't failed twice this fight.
        spawn_ok = house and house.get("spawn_at_bed", house.get("spawn_x", -1) >= 0)
        if (spawn_ok and house.get("nurse_alive") and fails < n.get("max_failed_trips", 2)
                and 0 < p.get("hp", 0) < n.get("hp_below", 200)
                and p.get("potion_delay", 0) > 0 and not p.get("dead")):
            bridge.request({"cmd": "select_slot", "slot": n.get("mirror_slot", 8)})
            ns.update(phase="mirror", t=0, fails=fails)
            phase = "mirror"
        else:
            return None
    ns["t"] = ns.get("t", 0) + 1
    if phase == "mirror":
        # Arrived = within 20 tiles of the house (not "moved far": the world spawn
        # can be right next to the arena).
        hx, hy = house["x"] * 16, house["y"] * 16
        home = math.hypot(p.get("cx", 0) - hx, p.get("cy", 0) - hy) < 20 * 16
        if home:
            ns.update(phase="heal", t=0)
            phase = "heal"
        elif ns["t"] > n.get("mirror_timeout_ticks", 40):
            fails += 1
            log({"phase": "nurse_fail", "why": "mirror did not bring the player home", "fails": fails})
            back_to_whip(n.get("retry_cooldown_ticks", 100))
            return None
        else:
            ns["last_action"] = {**RELEASE, "use": True}
            bridge.request({"cmd": "set_action", **RELEASE, "use": True})
            return 0
    if phase == "heal":
        r = bridge.request({"cmd": "nurse_heal"})
        ns.update(phase="exit", t=0, heal=r)
        healed = 1 if r.get("ok") and r.get("healed", 0) > 0 else 0
        if not r.get("ok"):
            fails += 1
            ns["fails"] = fails
        log({"phase": "nurse_heal", "result": r})
        if cfg.get("whip_slot") is not None:
            bridge.request({"cmd": "select_slot", "slot": cfg["whip_slot"]})
        ns["last_action"] = {**RELEASE, "jump": True, "up": True}
        bridge.request({"cmd": "set_action", **RELEASE, "jump": True, "up": True})
        return healed
    if phase == "exit":
        if ns["t"] > n.get("exit_ticks", 25):
            back_to_whip(0)
            return None
        ns["last_action"] = {**RELEASE, "jump": True, "up": True}
        bridge.request({"cmd": "set_action", **RELEASE, "jump": True, "up": True})
        return 0
    ns.clear()
    return None


def run_episode(bridge, policy, cfg, episode_seed, log_file=None):
    simulated = getattr(bridge, "simulated", False)
    clock = Clock(simulated)
    mode = cfg["mode"]
    max_s = cfg["max_seconds"][mode] if isinstance(cfg["max_seconds"], dict) else cfg["max_seconds"]
    rng = random.Random(episode_seed)

    def log(rec):
        if log_file is not None:
            log_file.write(json.dumps({"t": time.time(), **rec}) + "\n")

    bridge.request({"cmd": "set_action", **RELEASE})
    wait_alive(bridge, clock, cfg["respawn_timeout_s"])
    _ok(bridge.request({"cmd": "reset_episode"}), "reset_episode")
    if cfg.get("loadout_items"):
        # Inventory back to exactly the loadout every fight (user, 2026-10-11): wipe
        # it, then hotbar, inventory (potions, platinum), armour and accessories.
        _ok(bridge.request({"cmd": "equip_loadout", "reset_inventory": True, **cfg["loadout_items"]}),
            "reset inventory")
    elif cfg.get("restock_potions"):
        _ok(bridge.request({"cmd": "equip_loadout",
                            "inventory": [{"name": cfg["heal_potion"], "stack": cfg["restock_potions"]}]}),
            "restock potions")
    if cfg.get("buffs"):
        _ok(bridge.request({"cmd": "apply_buffs", "names": cfg["buffs"], "seconds": cfg["buff_seconds"]}),
            "apply_buffs")
    nurse = cfg.get("nurse") or {}
    if nurse.get("enabled"):
        # Nurse protocol: keep the money stocked (slot after the potions) and the
        # Nurse alive in her box, spawn on the bed.
        _ok(bridge.request({"cmd": "equip_loadout", "inventory": [
            {"name": cfg["heal_potion"], "stack": cfg.get("restock_potions") or 10},
            {"name": "PlatinumCoin", "stack": nurse.get("platinum", 999)}]}), "restock coins")
        house = nurse.get("house") or {}
        r = bridge.request({"cmd": "nurse_ready", **({"x": house["x"], "y": house["y"]} if house else {})})
        if not r.get("ok"):
            print(f"  nurse_ready: {r.get('error')} -- protocol inactive this fight "
                  "(rebuild the house: --build-nurse-house)", flush=True)
    setup_ticks = summon_minions(bridge, clock, cfg["summon_slots"], cfg["summon_hold_ticks"], log)
    if cfg.get("whip_slot") is not None:
        _ok(bridge.request({"cmd": "select_slot", "slot": cfg["whip_slot"]}), "select whip slot")
    human = getattr(policy, "passthrough", False)
    _ok(bridge.request({"cmd": "set_passthrough", "enabled": human}), "set_passthrough")
    if cfg.get("countdown_s"):
        for n in range(int(cfg["countdown_s"]), 0, -1):
            print(f"  {'boss' if mode == 'fight' else 'recording'} in {n}...", flush=True)
            clock.wait(1.0)
        if not getattr(bridge, "simulated", False):
            print("  go!", flush=True)
    if mode == "fight":
        spawn_boss(bridge, clock, cfg["boss_type"], cfg["boss_appear_timeout_s"])
    policy.reset()

    ledger = {"none": 0}
    heals = 0  # heal-assist potions drunk (kept out of the key ledger)
    nstate, nurse_trips = {}, 0   # Nurse protocol state (nurse_step)
    tick, outcome = 0, None
    wall_start = time.monotonic()
    start_ticks = None
    next_volley = 0.0
    boss_seen = mode != "fight"
    while outcome is None:
        clock.start_tick()
        st = bridge.request({"cmd": "get_state"})
        if not st.get("in_world"):
            raise HarnessError("left the world mid-episode")
        ep = st.get("episode") or {}
        if start_ticks is None:
            start_ticks = ep.get("ticks", 0)
        elapsed = (ep.get("ticks", 0) - start_ticks) / 60.0 if "ticks" in ep else tick * TICK_S
        p = st["player"]
        boss_alive = any(n.get("type") == cfg["boss_type"] for n in st.get("npcs", []))
        boss_seen = boss_seen or boss_alive

        if mode == "fight" and ep.get("boss_killed"):
            outcome = "win"
        elif p.get("dead") or p.get("ghost"):
            outcome = "loss"
        elif mode == "fight" and boss_seen and not boss_alive:
            outcome = "despawn"
        elif elapsed >= max_s:
            outcome = "timeout" if mode == "fight" else "survived"
        if outcome is not None:
            log({"phase": "end", "tick": tick, "outcome": outcome, "state": _trim_state(st)})
            break

        if mode == "gym" and elapsed >= next_volley:
            gym_volley(bridge, rng, cfg)
            next_volley += cfg["gym_period_s"]

        if nurse.get("enabled"):
            done = nurse_step(bridge, st, cfg, nstate, log)
            if done is not None:   # the protocol drove this tick
                ledger["nurse"] = ledger.get("nurse", 0) + 1
                nurse_trips += done
                log({"phase": "play", "tick": tick, "elapsed_s": elapsed, "action": nstate.get("last_action"),
                     "sources": {"nurse": "assist"}, "state": _trim_state(st), "nurse": nstate.get("phase")})
                tick += 1
                clock.end_tick()
                continue
        action, sources = policy.act(st)
        leashed = []
        if not human:  # a human demo is never overridden (passthrough ignores movement anyway)
            action, leashed = leash(st, action, cfg.get("arena"))
        if leashed:
            sources = {k: v for k, v in sources.items() if action.get(k)}
            ledger["leash"] = ledger.get("leash", 0) + 1
        payload = {k: bool(action.get(k)) for k in KEYS}
        # Whip assist: scripted, NOT the fly. Swings at the boss whenever
        # one is up; the policy's own `use` is ignored while it's on.
        if cfg.get("whip_assist") and cfg.get("whip_slot") is not None:
            target = whip_target(st, cfg.get("whip_strategy", "nearest"), cfg.get("whip_hold_frac", 0.15))
            if target is not None:
                payload.update(use=True, aim_x=target[0], aim_y=target[1])
                sources = {**{k: v for k, v in sources.items() if k != "use"}, "use": "assist"}
                action = {**action, "use": True}
        bridge.request({"cmd": "set_action", **payload})
        # Heal assist: scripted, the same for every policy (the game's Quick Heal).
        if (cfg.get("heal_at_hp") and 0 < p.get("hp", 0) <= cfg["heal_at_hp"]
                and not p.get("potion_delay") and p.get("heal_potions", 1) > 0):
            if bridge.request({"cmd": "quick_heal"}).get("healed"):
                heals += 1
        who = set(sources.values())
        if not who:
            ledger["none"] += 1
        for w in who:
            ledger[w] = ledger.get(w, 0) + 1
        log({"phase": "play", "tick": tick, "elapsed_s": elapsed, "action": action, "sources": sources,
             "state": _trim_state(st),
             "drives": getattr(policy, "last_drives", None),
             "vertical": getattr(policy, "last_vertical", None), "dn": getattr(policy, "last_dn", None)})
        tick += 1
        clock.end_tick()

    bridge.request({"cmd": "set_action", **RELEASE})
    if human:
        bridge.request({"cmd": "set_passthrough", "enabled": False})
    ep = st.get("episode") or {}
    boss = next((n for n in st.get("npcs", []) if n.get("type") == cfg["boss_type"]), None)
    progress = moon_lord_progress(st)
    parts_hp, parts_max = (progress["hp_left"], progress["hp_total"]) if progress else (0, 0)
    parts = bool(progress)
    return {
        "policy": policy.name, "mode": mode, "seed": episode_seed, "outcome": outcome,
        "win": outcome == "win", "survived": outcome in ("win", "survived", "timeout"),
        "seconds": round(elapsed, 2), "ticks": tick, "setup_ticks": setup_ticks,
        "hits": ep.get("hits"), "damage_taken": ep.get("damage_taken"), "heals": heals,
        "nurse_trips": nurse_trips,
        "damage_per_min": round(60.0 * (ep.get("damage_taken") or 0) / max(elapsed, 1e-6), 1),
        "boss_hp_left": boss.get("hp") if boss else (0 if outcome == "win" else None),
        "boss_parts_hp": parts_hp if parts else (0 if outcome == "win" else None),
        "boss_parts_frac_left": round(parts_hp / parts_max, 3) if parts_max else (0.0 if outcome == "win" else None),
        "eyes_destroyed": progress["eyes_destroyed"] if progress else (3 if outcome == "win" else None),
        "ledger": ledger, "simulated": simulated,
        # Game seconds per wall-clock second (1.0 = full speed). Live fights
        # slowed to ~0.35 once the whip started swinging (2026-10-08).
        "game_speed": None if simulated else round(elapsed / max(time.monotonic() - wall_start, 1e-6), 2),
    }


def summarize(rows):
    n = len(rows)
    if not n:
        return "no episodes"
    wins = sum(r["win"] for r in rows)
    surv = sum(r["survived"] for r in rows)
    secs = sorted(r["seconds"] for r in rows)
    dmg = [r["damage_per_min"] for r in rows]
    return (f"{n} episodes: win {wins}/{n}, survived {surv}/{n}, "
            f"median time {secs[n // 2]:.1f}s, mean damage/min {sum(dmg) / n:.0f}, "
            f"mean boss HP left {_mean_frac(rows)}")


def _mean_frac(rows):
    fr = [r["boss_parts_frac_left"] for r in rows if r.get("boss_parts_frac_left") is not None]
    return f"{100 * sum(fr) / len(fr):.0f}%" if fr else "n/a"


MARK_FILE = LOG_DIR / "arena_mark.json"        # legacy single-world file (still read as a fallback)
NURSE_FILE = LOG_DIR / "nurse_house.json"       # legacy


def _world_id(ep):
    """What identifies the world: its name, size and surface height."""
    return {"world_name": ep.get("world_name"), "world_tiles": ep.get("world_tiles"),
            "world_surface_y": ep.get("world_surface_y")}


def _same_world(saved, ep):
    keys = ("world_tiles", "world_surface_y") + (("world_name",) if saved.get("world_name") else ())
    return all(saved.get(k) == _world_id(ep).get(k) for k in keys)


def _per_world(stem, ep):
    """logs/moonlord/<stem>_<world name>.json: one file per world, so switching
    worlds (Normal test world vs the Expert world) keeps both."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", ep.get("world_name") or "").strip("_")
    return LOG_DIR / (f"{stem}_{name}.json" if name else f"{stem}.json")


def save_arena_mark(bridge):
    """Remember the arena mark on disk: the mod keeps it only in memory, so a
    mod rebuild or game restart forgets it (live 2026-10-09: the scripted
    batch after a rebuild ran with no mark, so no leash and no reset-to-arena)."""
    ep = bridge.request({"cmd": "get_episode"}).get("episode") or {}
    if ep.get("arena") and ep.get("world_tiles"):
        f = _per_world("arena_mark", ep)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"arena": ep["arena"], **_world_id(ep)}))


def restore_arena_mark(bridge):
    """Put the saved mark back if the game has none and it's the same world.
    Returns a short status line."""
    ep = bridge.request({"cmd": "get_episode"}).get("episode") or {}
    where = f" in world '{ep.get('world_name')}'" if ep.get("world_name") else ""
    if ep.get("arena"):
        return f"arena mark: set{where}"
    saved = None
    for f in (_per_world("arena_mark", ep), MARK_FILE):
        if f.exists():
            d = json.loads(f.read_text())
            if _same_world(d, ep):
                saved, src = d, f
                break
    if saved is None:
        return f"arena mark: NONE{where} (stand on the arena floor and run --mark-arena, or --build-arena --here)"
    a = saved["arena"]
    _ok(bridge.request({"cmd": "set_arena", "x": a["x"], "y": a["y"]}), "set_arena")
    return f"arena mark: restored from {src.name}{where} ({a['x']:.0f}, {a['y']:.0f})"


def config_from_loadout(loadout, mode="fight", whip_strategy=None, buffs=True, heal=True, whip_assist=True):
    """Episode config for run_episode from loadout.json (shared by main()
    and the offline trainers, so they fight exactly the same fight)."""
    cfg = json.loads(json.dumps(DEFAULTS))
    cfg["mode"] = mode
    cfg["summon_slots"] = list(loadout.get("summon_slots", []))
    cfg["arena"] = loadout.get("arena")
    for k in ("heal_at_hp", "restock_potions", "heal_potion"):
        if k in loadout:
            cfg[k] = loadout[k]
    if not heal:
        cfg["heal_at_hp"] = 0
    cfg["whip_slot"] = loadout.get("whip_slot")
    cfg["whip_strategy"] = whip_strategy or loadout.get("whip_strategy", "nearest")
    cfg["whip_assist"] = bool(loadout.get("whip_assist")) and whip_assist
    items = {k: loadout[k] for k in ("armor", "accessories", "hotbar", "inventory") if k in loadout}
    if items:
        cfg["loadout_items"] = items
    if buffs:
        cfg["buffs"] = list(loadout.get("buffs", []))
        cfg["buff_seconds"] = loadout.get("buff_seconds", cfg["buff_seconds"])
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="idle", choices=["idle", "random", "scripted", "fly", "human", "sectors", "senses3"])
    ap.add_argument("--mode", default="fight", choices=["fight", "gym", "free"],
                    help="fight = Moon Lord; gym = projectile volleys; free = nothing spawns (movement demos)")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="first episode seed; episode i uses seed+i")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--summon-slots", default="", help="hotbar slots of summon staffs, e.g. 0,1")
    ap.add_argument("--keep-focus", action="store_true",
                    help="ask the mod to keep feeding input while the game window is unfocused (untested)")
    ap.add_argument("--mark-arena", action="store_true", help="set the arena to where the player stands, then exit")
    ap.add_argument("--setup-character", action="store_true",
                    help="give the loaded character the loadout.json gear and max stats, then exit")
    ap.add_argument("--build-nurse-house", action="store_true",
                    help="build the Nurse house under the arena (loadout nurse.depth_tiles / offset_x_tiles)")
    ap.add_argument("--nurse", action="store_true",
                    help="Nurse healing protocol on (Magic Mirror home + Nurse heal when HP < nurse.hp_below and potion-sick)")
    ap.add_argument("--here", action="store_true",
                    help="with --build-arena: build where the player stands, ignoring any existing arena mark")
    ap.add_argument("--build-arena", action="store_true",
                    help="place loadout.json's platform rows around the player and mark the arena, then exit")
    ap.add_argument("--clear-tombstones", action="store_true",
                    help="remove every gravestone in the world, then exit (resets also clear them near the arena)")
    ap.add_argument("--loadout", default=str(LOADOUT), help="loadout file (default training/moonlord/loadout.json)")
    ap.add_argument("--no-fly-flee", action="store_true",
                    help="fly policy without the flee-by-flying rule (ablation: brain/decoder movement only)")
    ap.add_argument("--fly-vertical", choices=["brain", "rule", "off"], default=None,
                    help="fly climb/dive source: brain (LC4->Giant Fiber climb, LPLC4->DNp07 dive; default), "
                         "rule (hand sawtooth), off")
    ap.add_argument("--fly-commit", type=int, default=0, help="fly: keep a flee direction for N decisions (0 = off)")
    ap.add_argument("--fly-dive-hz", type=float, default=40.0, help="fly: landing (DNp07) rate needed to dive")
    ap.add_argument("--fly-waypoint", action="store_true",
                    help="fly: approach channel pulled toward a safe spot 600 px from the core")
    ap.add_argument("--no-heal", action="store_true", help="don't drink healing potions (scripted assist)")
    ap.add_argument("--whip-strategy", choices=["nearest", "balanced"], default=None,
                    help="which Moon Lord part the whip assist tags (minions follow the tag)")
    ap.add_argument("--no-whip-assist", action="store_true", help="don't swing the whip (scripted assist)")
    ap.add_argument("--no-loadout-buffs", action="store_true", help="don't re-apply the loadout's buffs each episode")
    ap.add_argument("--params-file", default=None, help="tuned params JSON for --policy fly (encoder/decoder/policy blocks, from train_fly_cma.py or imitate.py)")
    ap.add_argument("--gym-types", default=None, help="projectile type ids for gym volleys, e.g. 462,454,452")
    ap.add_argument("--fake", action="store_true", help="use the offline fake bridge instead of the game")
    ap.add_argument("--fly-null", default="none", choices=["none", "shuffled", "ablated"],
                    help="scene fly: run it on a null brain (control)")
    ap.add_argument("--dodger-ring", default=None, help="scripted: min,max px from the core to circle at (default 450,900)")
    ap.add_argument("--dodger-keep-away", type=float, default=None, help="scripted: px to keep from every boss part (default 250)")
    ap.add_argument("--dodger-dither", type=float, default=None, help="scripted: cost of changing move (default 0.2)")
    ap.add_argument("--no-log", action="store_true", help="don't write per-tick logs (summaries still written)")
    args = ap.parse_args(argv)

    bridge = connect(fake=args.fake, fake_config={"seed": args.seed})
    loadout = json.loads(Path(args.loadout).read_text())
    if args.mark_arena:
        print(_ok(bridge.request({"cmd": "set_arena"}), "set_arena"))
        save_arena_mark(bridge)
        return []
    if args.clear_tombstones:
        print(_ok(bridge.request({"cmd": "clear_tombstones"}), "clear_tombstones"))
        return []
    if args.build_nurse_house:
        nl = loadout.get("nurse") or {}
        r = bridge.request({"cmd": "build_nurse_house", "depth_tiles": nl.get("depth_tiles", 40),
                            "offset_x_tiles": nl.get("offset_x_tiles", 0)})
        print(json.dumps(r, indent=1))
        if r.get("ok") or r.get("house"):
            ep = bridge.request({"cmd": "get_episode"}).get("episode") or {}
            nf = _per_world("nurse_house", ep)
            nf.parent.mkdir(parents=True, exist_ok=True)
            nf.write_text(json.dumps({"house": r["house"], **_world_id(ep)}))
        bridge.close()
        return []
    if args.setup_character or args.build_arena:
        if args.setup_character:
            req = {k: loadout.get(k) for k in ("armor", "accessories", "hotbar", "inventory", "max_stats")}
            print(bridge.request({"cmd": "equip_loadout", **req}))
        if args.build_arena:
            # Rebuild around the existing arena mark if there is one, else where you stand.
            mark = None if args.here else (bridge.request({"cmd": "get_episode"}).get("episode") or {}).get("arena")
            where = {"x": mark["x"], "y": mark["y"]} if mark else {}
            print(bridge.request({"cmd": "build_arena", **loadout.get("arena", {}), **where}))
            save_arena_mark(bridge)
        bridge.close()
        return []

    if not args.fake:
        print(restore_arena_mark(bridge))
    cfg = config_from_loadout(loadout, args.mode, args.whip_strategy,
                              buffs=not args.no_loadout_buffs, heal=not args.no_heal,
                              whip_assist=not args.no_whip_assist)
    if args.nurse:
        nurse = dict(loadout.get("nurse") or {})
        saved = None
        if not args.fake:
            ep = bridge.request({"cmd": "get_episode"}).get("episode") or {}
            for nf in (_per_world("nurse_house", ep), NURSE_FILE):
                if nf.exists() and _same_world(json.loads(nf.read_text()), ep):
                    saved = json.loads(nf.read_text())
                    break
        if saved:
            nurse["house"] = saved.get("house")
        cfg["nurse"] = {**nurse, "enabled": True}
    if args.max_seconds:
        cfg["max_seconds"] = {args.mode: args.max_seconds}
    if args.summon_slots:
        cfg["summon_slots"] = [int(s) for s in args.summon_slots.split(",") if s.strip()]
    if args.policy == "human":
        cfg["countdown_s"] = 3  # time to put your hands on the keyboard
    if args.gym_types:
        cfg["gym_types"] = [int(s) for s in args.gym_types.split(",") if s.strip()]
    if args.keep_focus:
        _ok(bridge.request({"cmd": "set_keep_focus", "enabled": True}), "set_keep_focus")

    kw = {}
    if args.policy == "fly" and args.params_file:
        data = json.loads(Path(args.params_file).read_text())
        kw = {"enc_params": data.get("encoder"), "dec_params": data.get("decoder")}
    if args.policy == "fly" and args.no_fly_flee:
        kw["fly_flee"] = False
    if args.policy == "fly" and args.fly_vertical:
        kw["vertical"] = args.fly_vertical
    if args.policy == "fly":
        kw.update(commit_ticks=args.fly_commit, dive_on_hz=args.fly_dive_hz, waypoint=args.fly_waypoint)
        import fly_params
        kw["arena"] = fly_params.arena_px(loadout)
    if args.policy == "fly" and args.params_file:
        # Tuned FlyPolicy knobs (train_fly_cma.py / imitate.py) win over the CLI defaults.
        kw.update(data.get("policy") or {})
    if args.policy == "sectors":
        # Information-ceiling probe (not the fly): weights + FlyPolicy attention knobs from a params file.
        data = json.loads(Path(args.params_file or HERE / "params" / "sectors_linear.json").read_text())
        import fly_params
        kw = {**fly_params.policy_kwargs(data), "weights": data["weights"]}
    if args.policy == "scripted":
        # Keep the scripted dodger inside the built arena (90% of its half-width).
        a = loadout.get("arena", {})
        rise = min(a.get("rows", 4) * a.get("row_spacing_tiles", 12), a.get("max_rise_tiles") or 10 ** 6)
        kw = {"arena_half_width": 0.9 * 16 * a.get("half_width_tiles", 90), "arena_height": 0.9 * 16 * rise}
        if args.dodger_ring:
            kw["ring"] = tuple(float(v) for v in args.dodger_ring.split(","))
        if args.dodger_keep_away is not None:
            kw["boss_keep_away"] = args.dodger_keep_away
        if args.dodger_dither is not None:
            kw["dither_cost"] = args.dodger_dither
    if args.policy == "senses3":
        # v3-senses teacher (scene3.V3Policy): reads the fly's v3 senses directly, no brain. Not the fly.
        import scene3
        import fly_params
        data = json.loads(Path(args.params_file or HERE / "params" / "senses3_teacher.json").read_text())
        policy = scene3.V3Policy(data["weights"], fly_params.arena_px(loadout))
    elif (args.policy == "fly" and args.params_file
          and "override" in (json.loads(Path(args.params_file).read_text()).get("readout") or {})):
        # Scene fly v3 (scene_fly3.py)
        from scene_fly3 import SceneFly3Policy
        brain_kw = {}
        if args.fly_null != "none":
            from brain_backend import make_brain
            b, _ = make_brain("fast", args.fly_null, seed=0)
            brain_kw = {"brain": b}
        policy = SceneFly3Policy(json.loads(Path(args.params_file).read_text())["readout"], **brain_kw)
    elif args.policy == "fly" and args.params_file and "readout" in json.loads(Path(args.params_file).read_text()):
        # Scene fly (v2 senses + learned readout, scene_fly.py): same brain, richer input types.
        from scene_fly import SceneFlyPolicy
        import fly_params
        data = json.loads(Path(args.params_file).read_text())
        brain_kw = {}
        if args.fly_null != "none":
            # Null-brain control live (same senses and readout, scrambled/ablated brain).
            from brain_backend import make_brain
            b, rf = make_brain("fast", args.fly_null, seed=0)
            brain_kw = {"brain": b, "readout_fn": rf}
        policy = SceneFlyPolicy(readout=data["readout"], **brain_kw, **fly_params.policy_kwargs(data))
    else:
        policy = make_policy(args.policy, seed=args.seed, **kw)

    if hasattr(policy, "_ensure_brain") and not args.fake:
        print("Building the brain (once, before the first fight)...", flush=True)
        policy._ensure_brain()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    try:
        for i in range(args.episodes):
            seed = args.seed + i
            log_path = LOG_DIR / f"{stamp}_{args.policy}_{args.mode}_ep{i:03d}.jsonl"
            if args.no_log:
                row = run_episode(bridge, policy, cfg, seed)
            else:
                with open(log_path, "w") as f:
                    row = run_episode(bridge, policy, cfg, seed, f)
                row["log"] = str(log_path.relative_to(ROOT))
            row["run"] = stamp
            row["whip_strategy"] = cfg["whip_strategy"]
            row["nurse"] = bool(cfg.get("nurse"))
            if args.policy == "scripted" and (args.dodger_ring or args.dodger_keep_away is not None
                                              or args.dodger_dither is not None):
                row["variant"] = (f"ring{args.dodger_ring or '450,900'}_keep{args.dodger_keep_away or 250:g}"
                                  f"_dither{args.dodger_dither if args.dodger_dither is not None else 0.2:g}")
            if args.policy == "fly" and args.no_fly_flee:
                row["variant"] = "no_fly_flee"
            elif args.policy == "fly" and args.params_file:
                row["variant"] = ("tuned_" + Path(args.params_file).stem
                                  + ("" if args.fly_null == "none" else f"_brain_{args.fly_null}"))
            elif args.policy == "fly":
                row["variant"] = (f"vertical_{args.fly_vertical or 'brain'}"
                                  + (f"_commit{args.fly_commit}" if args.fly_commit else "")
                                  + (f"_dive{args.fly_dive_hz:g}" if args.fly_dive_hz != 40.0 else "")
                                  + ("_waypoint" if args.fly_waypoint else ""))
            rows.append(row)
            # Fake-bridge runs go to their own file so they never mix with real fights.
            with open(LOG_DIR / ("results_fake.jsonl" if args.fake else "results.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"ep {i:3d} seed {seed}: {row['outcome']:8s} {row['seconds']:6.1f}s  "
                  f"hits {row['hits']}  dmg {row['damage_taken']}  ledger {row['ledger']}")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        try:
            bridge.request({"cmd": "set_action", **RELEASE})
            if args.keep_focus:
                bridge.request({"cmd": "set_keep_focus", "enabled": False})
        except Exception:
            pass
        bridge.close()
    print(summarize(rows))
    return rows


if __name__ == "__main__":
    main()
