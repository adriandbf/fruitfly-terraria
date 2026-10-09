"""Goal 2 (Moon Lord) episode harness: runs fights (or dodge-gym drills)
back to back with nobody at the keyboard, and scores each one.

One episode:
  1. wait until the player is alive (the game's own respawn after a death)
  2. reset_episode: despawn hostiles and hostile projectiles, heal, clear
     debuffs, move to the arena mark, zero the mod's counters
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


def whip_target(state):
    """The Moon Lord part a whip swing should aim at: the nearest part that
    can currently take damage (eyes before the core opens), else the
    nearest part. Returns (x, y) world pixels or None."""
    p = state.get("player") or {}
    px, py = p.get("cx", p.get("x", 0.0)), p.get("cy", p.get("y", 0.0))
    parts = [n for n in state.get("npcs", []) if n.get("moon_lord_part") or n.get("boss")]
    if not parts:
        return None
    hittable = [n for n in parts if not n.get("dont_take_damage")] or parts
    best = min(hittable, key=lambda n: (n.get("cx", n["x"]) - px) ** 2 + (n.get("cy", n["y"]) - py) ** 2)
    return best.get("cx", best["x"]), best.get("cy", best["y"])


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
    if cfg.get("restock_potions"):
        _ok(bridge.request({"cmd": "equip_loadout",
                            "inventory": [{"name": cfg["heal_potion"], "stack": cfg["restock_potions"]}]}),
            "restock potions")
    if cfg.get("buffs"):
        _ok(bridge.request({"cmd": "apply_buffs", "names": cfg["buffs"], "seconds": cfg["buff_seconds"]}),
            "apply_buffs")
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
            target = whip_target(st)
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
             "drives": getattr(policy, "last_drives", None)})
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="idle", choices=["idle", "random", "scripted", "fly", "human"])
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
    ap.add_argument("--build-arena", action="store_true",
                    help="place loadout.json's platform rows around the player and mark the arena, then exit")
    ap.add_argument("--clear-tombstones", action="store_true",
                    help="remove every gravestone in the world, then exit (resets also clear them near the arena)")
    ap.add_argument("--loadout", default=str(LOADOUT), help="loadout file (default training/moonlord/loadout.json)")
    ap.add_argument("--no-fly-flee", action="store_true",
                    help="fly policy without the flee-by-flying rule (ablation: brain/decoder movement only)")
    ap.add_argument("--no-heal", action="store_true", help="don't drink healing potions (scripted assist)")
    ap.add_argument("--no-whip-assist", action="store_true", help="don't swing the whip (scripted assist)")
    ap.add_argument("--no-loadout-buffs", action="store_true", help="don't re-apply the loadout's buffs each episode")
    ap.add_argument("--params-file", default=None, help="tuned encoder/decoder params for --policy fly")
    ap.add_argument("--gym-types", default=None, help="projectile type ids for gym volleys, e.g. 462,454,452")
    ap.add_argument("--fake", action="store_true", help="use the offline fake bridge instead of the game")
    ap.add_argument("--no-log", action="store_true", help="don't write per-tick logs (summaries still written)")
    args = ap.parse_args(argv)

    bridge = connect(fake=args.fake, fake_config={"seed": args.seed})
    loadout = json.loads(Path(args.loadout).read_text())
    if args.mark_arena:
        print(_ok(bridge.request({"cmd": "set_arena"}), "set_arena"))
        return []
    if args.clear_tombstones:
        print(_ok(bridge.request({"cmd": "clear_tombstones"}), "clear_tombstones"))
        return []
    if args.setup_character or args.build_arena:
        if args.setup_character:
            req = {k: loadout.get(k) for k in ("armor", "accessories", "hotbar", "inventory", "max_stats")}
            print(bridge.request({"cmd": "equip_loadout", **req}))
        if args.build_arena:
            # Rebuild around the existing arena mark if there is one, else where you stand.
            mark = (bridge.request({"cmd": "get_episode"}).get("episode") or {}).get("arena")
            where = {"x": mark["x"], "y": mark["y"]} if mark else {}
            print(bridge.request({"cmd": "build_arena", **loadout.get("arena", {}), **where}))
        bridge.close()
        return []

    cfg = json.loads(json.dumps(DEFAULTS))
    cfg["mode"] = args.mode
    if args.max_seconds:
        cfg["max_seconds"] = {args.mode: args.max_seconds}
    cfg["summon_slots"] = ([int(s) for s in args.summon_slots.split(",") if s.strip()]
                           if args.summon_slots else list(loadout.get("summon_slots", [])))
    if args.policy == "human":
        cfg["countdown_s"] = 3  # time to put your hands on the keyboard
    cfg["arena"] = loadout.get("arena")
    for k in ("heal_at_hp", "restock_potions", "heal_potion"):
        if k in loadout:
            cfg[k] = loadout[k]
    if args.no_heal:
        cfg["heal_at_hp"] = 0
    cfg["whip_slot"] = loadout.get("whip_slot")
    cfg["whip_assist"] = bool(loadout.get("whip_assist")) and not args.no_whip_assist
    if not args.no_loadout_buffs:
        cfg["buffs"] = list(loadout.get("buffs", []))
        cfg["buff_seconds"] = loadout.get("buff_seconds", cfg["buff_seconds"])
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
    if args.policy == "scripted":
        # Keep the scripted dodger inside the built arena (90% of its half-width).
        a = loadout.get("arena", {})
        rise = min(a.get("rows", 4) * a.get("row_spacing_tiles", 12), a.get("max_rise_tiles") or 10 ** 6)
        kw = {"arena_half_width": 0.9 * 16 * a.get("half_width_tiles", 90), "arena_height": 0.9 * 16 * rise}
    policy = make_policy(args.policy, seed=args.seed, **kw)

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
            if args.policy == "fly" and args.no_fly_flee:
                row["variant"] = "no_fly_flee"
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
