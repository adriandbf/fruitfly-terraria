"""Offline stand-in for the FruitFlyBridge mod's Goal 2 (Moon Lord) API.

Speaks the same request/response protocol as the real mod (get_state,
set_action, reset_episode, spawn_npc, spawn_projectile, select_slot,
set_arena, set_keep_focus, get_episode) so harness.py and every policy run
unchanged against either. Two uses:
  1. Testing the harness and policies without the game.
  2. The start of the offline dodge gym (stage M2): tuning needs hundreds of
     episodes, far too slow against the live game.

It is a CARTOON of the fight, not a model of it. Every number below
(speeds, damage, attack timing, boss HP, minion DPS) is an assumption in
Terraria's own units (pixels, 60 ticks/s), chosen to be roughly the right
order of magnitude. Calibrate against logged live fights before trusting
any result from it. Not modelled: phases, the hands' and head's separate
HP and vulnerability, leeches, platforms, knockback, defense, debuffs,
buffs, and real minion targeting.

Time only moves on set_action: each call advances SUBSTEPS game ticks
(6 ticks = one 10 Hz decision), so episodes run as fast as Python can go.
"""
import json
import math
import random
from pathlib import Path

import movement

TICKS_PER_S = 60
PART_OFFSET = {"hand_l": (-400.0, -150.0), "hand_r": (400.0, -150.0), "head": (0.0, -400.0), "core": (0.0, 0.0)}
PART_TYPE = {"hand_l": 397, "hand_r": 397, "head": 396, "core": 398}
TRUE_EYE = 400
SUBSTEPS = 6  # game ticks per set_action (10 Hz decisions)

# Projectile type ids (real Terraria ProjectileID values).
PHANTASMAL_EYE = 452
PHANTASMAL_SPHERE = 454
PHANTASMAL_DEATHRAY = 455
PHANTASMAL_BOLT = 462
MOON_LEECH = 456   # the tongue: live, launched from ~180 px above the core every ~17 s, slightly homing,
                   # 0 damage itself; a catch drags the player to the core (attributed hits ~1 per 26 s)
# NPC type ids (real NPCID values).
MOON_LORD_CORE = 398

# Calibrated 2026-10-08 against the first live idle fight (Expert world,
# Hallowed summoner, 9 minions; logs/moonlord/20261008_005154_idle_fight_ep000.jsonl,
# 18 s, one fight -- treat every number as a rough first fit):
#   boss parts 67.5k head + 2 x 37.5k hands + 75k core = 217.5k HP
#   minions (no whip) took 1658 HP/s off the parts
#   damage per hit after defense: sphere ~126, bolt ~71, eye ~66, deathray ~78
#   new projectiles/s: spheres ~1.3, eyes ~0.9, bolts ~0.55; bolt speed 8,
#   eye ~9, spheres hover (~4) then launch (up to ~16)
#   deathray: first ~17 s in, sweeps ~0.0116 rad/tick from the head (~270 px up)
#   player regen ~10 HP/s (potions + campfire + heart lantern); 500 max HP
DEFAULT_CONFIG = {
    "seed": 0,
    "arena_half_width": 6400.0,   # walls this far either side of the start x (= the 400-tile arena)
    "floor_y": 0.0,               # world y of the floor (y grows downward)
    "rows": 12, "row_spacing_px": 384.0,   # planter-box rows above the floor (large-world arena)
    "row_pass_vy": 1.0,                   # fall speed left after dropping through a row (live median ~1)
    "ceiling_px": 4000.0,         # world top this far above the floor (large world: normal gravity ends ~3900 px up)
    "player_hp": 600,                # 500 + Lifeforce 20% (live max_hp 600)
    "regen_per_s": 3.5,              # live 2026-10-09: ~3.5 HP/s between hits (was 10)
    "player_w": 20, "player_h": 42,
    "wing_time_max": 150,         # measured; movement itself comes from movement.py (probe-fitted)
    "infinite_flight": False,     # flying mount: up/down steer, never runs out
    "immune_ticks": 40,           # Terraria's standard post-hit immunity
    "respawn_ticks": 300,
    # Boss
    "boss_hp": 217500,            # total of the parts below
    # Per-part model (2026-10-09): live HP of each part; the core can't be hurt
    # until all three eyes are destroyed; each destroyed eye becomes a True Eye
    # of Cthulhu that keeps shooting. Live (10 fights with True Eyes): damage
    # taken 14.5/s with none -> 26.0/s with one (+80%), mostly extra Phantasmal
    # Eyes; a destroyed part keeps its own attacks here. Minion damage goes to
    # the whip-tagged part (the one nearest the aim point) or else the nearest
    # hittable part, like summon targeting in the game.
    "part_hp": {"head": 67500, "hand_l": 37500, "hand_r": 37500, "core": 75000},
    "true_eye_period": 100, "true_eye_shots": 3,   # homing Phantasmal Eyes per volley, per True Eye
    "minion_dps": 1658.0,         # measured, no whip (Hallowed gear; used only when dps_near is None)
    # v2 (2026-10-09 evening, 70 small-world + 15 large-world live fights, Spooky
    # gear): boss damage falls off with the player's distance from the core
    # (whip reach / minion travel): idle ~3500 HP/s at ~130 px, scripted ~2900
    # at ~400 px, human ~2200 at ~660 px. Total minion+whip DPS = dps_near -
    # dps_per_px * distance, floored at dps_min; None = the old flat model.
    "dps_near": 3700.0, "dps_per_px": 2.2, "dps_min": 0.0,   # live 2026-10-10: ~0 beyond ~1500 px (minions/whip out of reach)
    # Live 2026-10-09 (random/scripted/human fights): the boss CHASES -- its core
    # moves ~8 px/tick (as fast as the player flies) and sits ~130 px BELOW
    # the player; the head is ~400 px above the core, hands ~400 px to each side.
    "boss_height": -130.0,        # core this far above the player (negative = below)
    "teleport_px": 2350.0,        # core teleports onto a player this far away (live logs)
    "boss_drift": 8.0,            # px/tick chase speed of the core: a VECTOR speed (live: |v| ~7.9 in every
                                  # direction), not 8 per axis -- the old per-axis clamp made the boss 11 px/tick
                                  # on diagonals, so climbing/diving away never paid off offline
    # Hands (live, 25k samples): ~120 px above the core; the hand on the
    # player's side pulls IN to ~330 px from the core once the player is past
    # ~150 px on that side, and the far hand swings OUT to ~700 px.
    "hand_dy": -120.0, "hand_in": 330.0, "hand_out": 700.0, "hand_speed": 12.0,
    "contact_radius": 45.0, "contact_damage": 100,   # core/hand hitboxes are 46x66 px (live); contact is a top hit source
    # Attacks
    "bolt_period": 330, "bolt_damage": 90, "bolt_speed": 8.0, "bolt_count": 3,
    "sphere_period": 280, "sphere_damage": 140, "sphere_speed": 10.0, "sphere_count": 6,
    "eye_period": 190, "eye_damage": 85, "eye_speed": 9.0, "eye_turn": 0.03, "eye_life": 240,
    # v2 aim model (fitted to live hit rates by calibrate_fake.py / fit_fake.py)
    "bolt_lead": 0.0, "bolt_spread": 0.12, "bolt_life": 600,
    "sphere_lead": 0.0, "sphere_spread": 0.35, "sphere_life": 600,
    # Moon Leech (tongue), v3 2026-10-09 from 70+ live fights: catch -> pulled toward the core
    # for leech_grab_ticks (no control), then bitten for leech_damage.
    "leech_period": 1040, "leech_speed": 9.0, "leech_turn": 0.02, "leech_life": 310,
    "leech_grab_ticks": 40, "leech_pull": 14.0, "leech_damage": 100,
    "ray_period": 900, "ray_first": 1020, "ray_damage": 100, "ray_sweep": 1.4,
    "ray_duration": 120, "ray_width": 40.0, "ray_length": 2400.0,
    "boss_attacks": True,
}

# Attack knobs fitted to live fights by fit_fake.py, if present (overrides
# the defaults above; explicit config still wins).
_FIT_PATH = Path(__file__).resolve().parent / "fake_fit.json"
FITTED = json.loads(_FIT_PATH.read_text())["config"] if _FIT_PATH.exists() else {}

SIZES = {MOON_LEECH: 16, PHANTASMAL_BOLT: 8, PHANTASMAL_SPHERE: 40, PHANTASMAL_EYE: 14, PHANTASMAL_DEATHRAY: 36}  # measured hitboxes
SOURCE = {PHANTASMAL_EYE: "eye", PHANTASMAL_BOLT: "bolt", PHANTASMAL_SPHERE: "sphere",
          PHANTASMAL_DEATHRAY: "deathray", MOON_LEECH: "tongue"}

# Master mode (live 2026-10-11, "The Violent Heart", 10 fly + 5 idle fights;
# hit_sources.py): damage per hit MEASURED live, by source -- homing eye 125,
# bolt 101, deathray 278, sphere 156, tongue 123, body contact ~120 -- used
# directly. Moon Lord parts have 1.275x the Expert HP (core 95,625 vs 75,000).
# Minion damage vs distance is unchanged from Expert; attack timing is assumed
# unchanged (Master runs the Expert AI).
MASTER = {"eye_damage": 125, "bolt_damage": 101, "ray_damage": 278, "sphere_damage": 156,
          "leech_damage": 123, "contact_damage": 120,
          # Homing eyes rebuilt from the live Master tracks (1,353 eye tracks): they leave the
          # HANDS (not the head), fly at ~8-14 px/tick, turn gently (median ~0.01 rad/tick, p90
          # ~0.04), overshoot and loop back. Speed 11 +-30%, turn 0.03, a volley every 160 ticks
          # gives the senses3 teacher 5.7 eye hits/min offline vs the fly's 4.9 live.
          "eye_origin": "hands", "eye_speed": 11.0, "eye_speed_spread": 0.3, "eye_turn": 0.03,
          "eye_period": 160, "eye_life": 300}
MASTER_HP = 1.275

DEFAULT_DAMAGE = {PHANTASMAL_BOLT: 71, PHANTASMAL_SPHERE: 126, PHANTASMAL_EYE: 66, PHANTASMAL_DEATHRAY: 78}


class _Proj:
    __slots__ = ("type", "x", "y", "vx", "vy", "size", "damage", "life", "homing", "angle", "omega")

    def __init__(self, type_, x, y, vx, vy, damage, life=600, homing=0.0):
        self.type, self.x, self.y, self.vx, self.vy = type_, x, y, vx, vy
        self.size = SIZES.get(type_, 16)
        self.damage, self.life, self.homing = damage, life, homing
        self.angle = self.omega = 0.0


class FakeMoonLordBridge:
    simulated = True

    def __init__(self, config=None):
        self.cfg = {**DEFAULT_CONFIG, **FITTED, **(config or {})}
        if self.cfg.get("difficulty") == "master":
            for k, v in MASTER.items():
                if k not in (config or {}):
                    self.cfg[k] = v
            if "part_hp" not in (config or {}):
                self.cfg["part_hp"] = {k: v * MASTER_HP for k, v in self.cfg["part_hp"].items()}
                self.cfg["boss_hp"] = sum(self.cfg["part_hp"].values())
        self.rng = random.Random(self.cfg["seed"])
        c = self.cfg
        self.spawn_x, self.floor = 0.0, c["floor_y"]
        self.arena = None  # set below, once the player exists
        self.keep_focus = False
        self.passthrough = False
        self.human_keys = {}  # tests: keys a pretend human holds in passthrough mode
        self.action = {}
        self.selected = 0
        self.hotbar = {0: "StardustDragonStaff", 1: "EmpressBlade", 2: "RainbowWhip", 8: "MagicMirror"}
        self._mirror_hold = 0
        self.whip_bonus = 1.25  # assumed minion DPS multiplier while the whip keeps tagging
        self.max_minions = 10  # measured: 9 with Hallowed, 10 with Spooky (live 2026-10-09)
        self.minions = 0
        self._summon_hold = 0
        self.px, self.py = self.spawn_x, self.floor - c["player_h"]  # top-left
        # Marked arena = the spawn point, like a live world after --build-arena
        # (so the leash and FlyPolicy's looming surfaces see the same box offline).
        self.arena = (self.px, self.py) if c.get("mark_arena", True) else None
        self.vx = self.vy = 0.0
        self.hp, self.dead, self.dead_ticks = c["player_hp"], False, 0
        self.immune = 0
        self.wing_time = c["wing_time_max"]
        self.potions, self.potion_delay = 10, 0
        self.projs = []
        self.boss = None
        self._reset_counters()

    # ---- protocol ---------------------------------------------------------
    def request(self, payload):
        cmd = payload.get("cmd")
        fn = getattr(self, "_cmd_" + str(cmd), None)
        if fn is None:
            return {"ok": False, "error": f"unknown cmd '{cmd}'"}
        return fn(payload)

    def release(self):
        self.action = {}

    def close(self):
        pass

    def _cmd_ping(self, _):
        return {"ok": True, "pong": True}

    def _cmd_set_passthrough(self, p):
        self.passthrough = bool(p.get("enabled", True))
        return {"ok": True, "passthrough": self.passthrough}

    def _cmd_set_keep_focus(self, p):
        self.keep_focus = bool(p.get("enabled", True))
        return {"ok": True, "keep_focus": self.keep_focus}

    def _cmd_get_episode(self, _):
        return {"ok": True, "episode": self._episode()}

    # ---- Nurse protocol (NurseBay.cs) -------------------------------------
    def _cmd_build_nurse_house(self, p):
        # house_depth_tiles (config) wins: the hand-built v5 house sits right under the floor (6).
        self.house = (self.spawn_x, self.floor + 16 * self.cfg.get("house_depth_tiles", p.get("depth_tiles", 40)))
        self.nurse_hp = 250
        return {"ok": True, "house": {"x": int(self.house[0] // 16), "y": int(self.house[1] // 16)}}

    def _cmd_nurse_ready(self, p):
        if getattr(self, "house", None) is None:
            if p.get("x") is None:
                return {"ok": False, "error": "no nurse house"}
            self.house = (p["x"] * 16.0, p["y"] * 16.0)
        self.nurse_hp = 250
        return {"ok": True, "hp": 250}

    # House layout as NurseBay.cs v4: one room, bed (spawn) 7 tiles left of the
    # right wall column, Nurse at 3 left; talking range = Player.tileRangeX/Y (5/4 tiles).
    def _nurse_pos(self):
        return self.house[0] - 2.5 * 16, self.house[1] - 21

    def _cmd_nurse_talk(self, p):
        if getattr(self, "house", None) is None:
            return {"ok": False, "error": "no Nurse"}
        pcx, pcy = self._centre()
        nx, ny = self._nurse_pos()
        if abs(nx - pcx) > 5 * 16 + 9 or abs(ny - pcy) > 4 * 16 + 21:
            self.talking = False
            return {"ok": False, "error": "out of talking range", "dx": nx - pcx, "dy": ny - pcy}
        self.talking = True
        return {"ok": True, "talking": True, "dx": nx - pcx}

    def _cmd_nurse_heal(self, p):
        if getattr(self, "house", None) is None:
            return {"ok": False, "error": "no Nurse"}
        pcx, pcy = self._centre()
        nx, ny = self._nurse_pos()
        if not getattr(self, "talking", False) or abs(nx - pcx) > 5 * 16 + 9 or abs(ny - pcy) > 4 * 16 + 21:
            return {"ok": False, "error": "not talking to the Nurse (walk over, nurse_talk first)"}
        self.talking = False
        missing = self.cfg["player_hp"] - self.hp
        self.hp = self.cfg["player_hp"]
        return {"ok": True, "healed": int(missing), "price": int(missing) * 200}

    def _nurse_json(self):
        if getattr(self, "house", None) is None:
            return None
        return {"x": int(self.house[0] // 16), "y": int(self.house[1] // 16), "nurse_alive": True,
                "nurse_hp": getattr(self, "nurse_hp", 250),
                "nurse_cx": self._nurse_pos()[0], "nurse_cy": self._nurse_pos()[1],
                "shaft_top_y": int(self.floor // 16),
                # live 2026-10-11: an invalid house left the spawn at the world spawn
                "spawn_at_bed": self.cfg.get("nurse_report_spawn", True)}

    def _cmd_set_arena(self, p):
        self.arena = (p.get("x", self.px), p.get("y", self.py))
        return {"ok": True, "arena": {"x": self.arena[0], "y": self.arena[1]}}

    def _cmd_quick_heal(self, _):
        before = self.hp
        sick = self.potion_delay > 0
        if not sick and not self.dead and self.potions > 0:
            self.hp = min(self.cfg["player_hp"], self.hp + 200)
            self.potions -= 1
            self.potion_delay = 3600  # 60 s potion sickness
        return {"ok": True, "hp_before": before, "hp_after": self.hp, "potion_sick": sick,
                "healed": self.hp > before}

    def _cmd_equip_loadout(self, p):
        for entry in p.get("inventory") or []:
            if "Heal" in str(entry.get("name")):
                self.potions = int(entry.get("stack", 1))
        if p.get("hotbar"):
            self.hotbar = {i: (n[0] if isinstance(n, list) else n) for i, n in enumerate(p["hotbar"]) if n}
        return {"ok": True, "unknown_items": []}

    def _cmd_apply_buffs(self, p):
        return {"ok": True, "applied": len(p.get("names") or []), "unknown_buffs": []}

    def _cmd_clear_tombstones(self, _):
        return {"ok": True, "tombstone_tiles_removed": 0}

    def _cmd_build_arena(self, p):
        self.arena = (p.get("x", self.px), p.get("y", self.py))
        return {"ok": True, "floor_tiles_placed": 0, "campfires": [], "lanterns": [],
                "arena": {"x": self.arena[0], "y": self.arena[1]}}

    def _cmd_select_slot(self, p):
        slot = int(p.get("slot", 0))
        if not 0 <= slot <= 9:
            return {"ok": False, "error": "slot must be 0-9"}
        self.selected = slot
        return {"ok": True, "slot": slot, "item_type": 0, "item_name": self.hotbar.get(slot, "")}

    def _cmd_reset_episode(self, _):
        if self.dead:
            return {"ok": False, "error": "player is dead; wait for respawn"}
        n_npc = 1 if self.boss else 0
        n_proj = len(self.projs)
        self.boss, self.projs = None, []
        self.hp = self.cfg["player_hp"]
        if self.arena:
            self.px, self.py = self.arena
        self.vx = self.vy = 0.0
        self.immune = 0
        self.action = {}
        self._reset_counters()
        return {"ok": True, "npcs_removed": n_npc, "projectiles_removed": n_proj,
                "moved_to_arena": self.arena is not None}

    def _cmd_spawn_npc(self, p):
        t = int(p.get("type", MOON_LORD_CORE))
        if t != MOON_LORD_CORE:
            return {"ok": False, "error": "fake bridge only models the Moon Lord"}
        cx = self.px + self.cfg["player_w"] / 2
        self.boss = {"cx": cx, "cy": self.py - self.cfg["boss_height"], "ray": None, "vx": 0.0, "vy": 0.0,
                     "hand_dx": {"hand_l": -self.cfg["hand_out"], "hand_r": self.cfg["hand_out"]},
                     "parts": dict(self.cfg["part_hp"]), "destroyed": []}
        return {"ok": True, "type": t}

    def _cmd_spawn_projectile(self, p):
        t = int(p.get("type", PHANTASMAL_BOLT))
        pcx, pcy = self._centre()
        x = p["x"] if "x" in p else pcx + p.get("dx", 0.0)
        y = p["y"] if "y" in p else pcy + p.get("dy", 0.0)
        homing = self.cfg["eye_turn"] if t == PHANTASMAL_EYE else 0.0
        self.projs.append(_Proj(t, x, y, p.get("vx", 0.0), p.get("vy", 0.0),
                                p.get("damage", DEFAULT_DAMAGE.get(t, 60)),
                                life=self.cfg["eye_life"] if t == PHANTASMAL_EYE else 600, homing=homing))
        return {"ok": True, "index": len(self.projs) - 1, "type": t}

    def _cmd_set_action(self, p):
        self.action = {k: bool(p.get(k, False)) for k in ("left", "right", "jump", "use", "up", "down")}
        if self.passthrough:  # like the mod: movement from the "human", only use from the bridge
            self.action = {**{k: bool(self.human_keys.get(k)) for k in ("left", "right", "jump", "up", "down")},
                           "use": self.action["use"]}
        self.aim = (p["aim_x"], p["aim_y"]) if p.get("aim_x") is not None else None
        for _ in range(SUBSTEPS):
            self._tick()
        return {"ok": True}

    def _cmd_get_state(self, _):
        c = self.cfg
        pcx, pcy = self._centre()
        npcs = []
        if self.boss:
            b = self.boss
            for name, (dx, dy) in self._offsets().items():
                dead = name in b["destroyed"]
                hp = c["part_hp"][name] if dead else b["parts"][name]  # like the game: a destroyed eye reads full hp
                npcs.append({"type": PART_TYPE[name], "name": name, "x": b["cx"] + dx - 23, "y": b["cy"] + dy - 33,
                             "cx": b["cx"] + dx, "cy": b["cy"] + dy, "vx": b["vx"], "vy": b["vy"], "width": 46, "height": 66,
                             "hp": int(hp), "max_hp": c["part_hp"][name], "hostile": True, "boss": name == "core",
                             "moon_lord_part": True, "dont_take_damage": dead or not self._hittable(name)})
            for name in b["destroyed"]:
                dx, dy = self._offsets()[name]
                npcs.append({"type": TRUE_EYE, "name": "True Eye of Cthulhu", "x": b["cx"] + dx, "y": b["cy"] + dy - 60,
                             "cx": b["cx"] + dx, "cy": b["cy"] + dy - 60, "vx": 0.0, "vy": 0.0, "width": 30,
                             "height": 30, "hp": 200, "max_hp": 200, "hostile": True, "boss": False,
                             "moon_lord_part": False, "dont_take_damage": True})
        projs = []
        for pr in self.projs:
            projs.append({"type": pr.type, "cx": pr.x, "cy": pr.y, "vx": pr.vx, "vy": pr.vy,
                          "width": pr.size, "height": pr.size, "damage": pr.damage})
        return {
            "ok": True, "in_world": True,
            "player": {"x": self.px, "y": self.py, "cx": pcx, "cy": pcy, "direction": 1 if self.vx >= 0 else -1,
                       "hp": max(0, int(self.hp)), "max_hp": c["player_hp"], "mana": 200, "max_mana": 200,
                       "on_ground": self._on_ground(), "dead": self.dead, "ghost": False,
                       "vx": self.vx, "vy": self.vy, "width": c["player_w"], "height": c["player_h"],
                       "wing_time": self.wing_time, "gravity": 0.4, "wing_time_max": c["wing_time_max"],
                       "mount_active": c["infinite_flight"], "immune": self.immune > 0,
                       "minions": self.minions, "max_minions": self.max_minions,
                       "selected_item": self.selected,
                       "potion_delay": self.potion_delay, "heal_potions": self.potions},
            "npcs": npcs, "players": [], "projectiles": projs,
            "tiles": {"origin_tx": 0, "origin_ty": 0, "width": 0, "height": 0, "types": [], "solid": []},
            "episode": self._episode(),
            "debug": {"has_focus": True, "simulated": True,
                      **{f"control_{k}": bool(self.human_keys.get(k)) for k in ("left", "right", "jump", "up", "down")},
                      "control_use_item": bool(self.action.get("use"))},
        }

    # ---- simulation -------------------------------------------------------
    def _reset_counters(self):
        self.ticks = self.hits = self.damage_taken = self.deaths = 0
        self.damage_by = {}
        self.boss_killed, self.boss_kill_tick = False, -1
        self.grabbed = 0
        self.on_row = False

    def _episode(self):
        return {"ticks": self.ticks, "seconds": self.ticks / TICKS_PER_S, "hits": self.hits,
                "damage_taken": self.damage_taken, "deaths": self.deaths, "damage_by": dict(self.damage_by),
                "boss_killed": self.boss_killed, "boss_kill_tick": self.boss_kill_tick,
                "keep_focus": self.keep_focus,
                "arena": {"x": self.arena[0], "y": self.arena[1]} if self.arena else None,
                "full_gravity_y": self.floor - self.cfg["ceiling_px"],
                "nurse_house": self._nurse_json()}

    def _centre(self):
        return self.px + self.cfg["player_w"] / 2, self.py + self.cfg["player_h"] / 2

    def _in_house(self):
        """Below the arena floor = in the Nurse house / its shaft (NurseBay.cs v3):
        fly room and shaft span 10 tiles left of the divider column."""
        return getattr(self, "house", None) is not None and self.py + self.cfg["player_h"] > self.floor + 1

    def _floor_here(self):
        return self.house[1] if self._in_house() else self.floor

    def _on_ground(self):
        return self.py + self.cfg["player_h"] >= self._floor_here() - 0.01 or getattr(self, "on_row", False)

    def _row_lines(self):
        c = self.cfg
        return [self.floor - r * c["row_spacing_px"] for r in range(1, c["rows"] + 1)] if c.get("rows") else []

    def _tick(self):
        c = self.cfg
        self.ticks += 1
        if self.dead:
            self.dead_ticks += 1
            self.projs = []
            if self.dead_ticks >= c["respawn_ticks"]:
                self.dead, self.dead_ticks = False, 0
                self.hp = c["player_hp"]
                self.px, self.py = self.spawn_x, self.floor - c["player_h"]
                self.vx = self.vy = 0.0
                self.minions = 0  # minions vanish on death, as in the game
            return
        if self.hp < c["player_hp"]:
            self.hp = min(c["player_hp"], self.hp + c["regen_per_s"] / TICKS_PER_S)
        self._move_player()
        self._summon()
        if self.boss:
            self._boss_tick()
        self._projectiles_tick()
        if self.immune > 0:
            self.immune -= 1
        if self.potion_delay > 0:
            self.potion_delay -= 1

    def _move_player(self):
        c, a = self.cfg, self.action
        if getattr(self, "grabbed", 0) > 0 and self.boss:   # caught by the tongue
            self.grabbed -= 1
            pcx, pcy = self._centre()
            dx, dy = self.boss["cx"] - pcx, self.boss["cy"] - pcy
            d = math.hypot(dx, dy) or 1.0
            step = min(c["leech_pull"], d)
            self.vx, self.vy = dx / d * step, dy / d * step
            self.px += self.vx
            self.py += self.vy
            if self.grabbed == 0:
                self._hurt(c["leech_damage"], "tongue")
            return
        h = (1 if a.get("right") else 0) - (1 if a.get("left") else 0)
        self.vx, self.vy = movement.step(self.vx, self.vy, h, movement.mode_of(a), self._on_ground())
        if self.on_row and (a.get("down") or self.vy < 0):
            self.on_row = False            # drop through / take off
        feet0 = self.py + c["player_h"]
        self.px += self.vx
        self.py += self.vy
        if self.on_row:
            self.py, self.vy = feet0 - c["player_h"], 0.0
        elif self.vy > 0:
            feet1 = self.py + c["player_h"]
            for row in self._row_lines():
                if feet0 <= row < feet1:
                    # Planter rows (live 2026-10-09): falling with down held passes
                    # through but the row stops the fall (vy 10 -> ~1); without
                    # down you land on it.
                    if a.get("down"):
                        self.vy = min(self.vy, c["row_pass_vy"])
                    else:
                        self.py, self.vy, self.on_row = row - c["player_h"], 0.0, True
                    break
        if self._in_house():
            left, right = self.house[0] - 10 * 16, self.house[0] - c["player_w"]
        else:
            left, right = self.spawn_x - c["arena_half_width"], self.spawn_x + c["arena_half_width"] - c["player_w"]
        if not left <= self.px <= right:
            self.px, self.vx = max(left, min(right, self.px)), 0.0  # walls stop you dead
        floor = self._floor_here()
        if self.py + c["player_h"] >= floor:
            self.py = floor - c["player_h"]
            self.vy = min(self.vy, 0.0)
        top = self.floor - c["ceiling_px"]
        if self.py < top:
            self.py, self.vy = top, 0.0  # world height limit

    def _summon(self):
        # Magic Mirror: holding use for 90 ticks teleports to the spawn bed (the Nurse house).
        if self.hotbar.get(self.selected) == "MagicMirror":
            self._mirror_hold = self._mirror_hold + 1 if self.action.get("use") else 0
            if self._mirror_hold >= 90 and getattr(self, "house", None) is not None:
                land_x = self.house[0] - 7 * 16 + getattr(self, "mirror_miss_px", 0.0)   # the bed; test hook: elsewhere
                land_y = self.floor if getattr(self, "mirror_miss_px", 0.0) else self.house[1]   # a miss = world spawn up top
                self.px, self.py = land_x - self.cfg["player_w"] / 2, land_y - self.cfg["player_h"]
                self.vx = self.vy = 0.0
                self._mirror_hold = 0
            return
        # Holding `use` with a summon staff selected adds a minion every 30 ticks.
        if self.action.get("use") and self.selected in self.hotbar and not self._whip_selected():
            self._summon_hold += 1
            if self._summon_hold % 6 == 0 and self.minions < self.max_minions:
                self.minions += 1
        else:
            self._summon_hold = 0

    def _whip_selected(self):
        return "Whip" in str(self.hotbar.get(self.selected, ""))

    def _boss_tick(self):
        c, b = self.cfg, self.boss
        pcx, pcy = self._centre()
        tx, ty = pcx - b["cx"], (pcy - c["boss_height"]) - b["cy"]
        d = math.hypot(tx, ty)
        if d > c["teleport_px"]:
            # Live (28 cases in the logs): with the player ~2,300-2,400 px away the
            # core jumps onto them.
            b["cx"], b["cy"] = pcx, pcy - c["boss_height"]
            tx = ty = d = 0.0
        step = min(c["boss_drift"], d)
        b["vx"], b["vy"] = (tx / d * step, ty / d * step) if d > 1e-9 else (0.0, 0.0)
        b["cx"] += b["vx"]
        b["cy"] += b["vy"]
        for nm, sign in (("hand_l", -1.0), ("hand_r", 1.0)):
            side = (pcx - b["cx"]) * sign          # how far the player is out on this hand's side
            f = min(1.0, max(0.0, (side + 300.0) / 450.0))   # -300 px -> 0 (out), +150 px -> 1 (in)
            want = sign * (c["hand_out"] + f * (c["hand_in"] - c["hand_out"]))
            cur = b["hand_dx"][nm]
            b["hand_dx"][nm] = cur + max(-c["hand_speed"], min(c["hand_speed"], want - cur))
        if self.minions > 0:
            whipping = self.action.get("use") and self._whip_selected()
            whip = self.whip_bonus if whipping else 1.0
            target = self._minion_target(pcx, pcy, self.aim if whipping else None)
            if target:
                if c.get("dps_near") is not None:  # v2: distance-dependent, whip included
                    dist = math.hypot(pcx - b["cx"], pcy - b["cy"])
                    dps = max(c["dps_min"], c["dps_near"] - c["dps_per_px"] * dist) * (1.0 if whipping else 1 / 1.25)
                else:
                    dps = c["minion_dps"] * whip
                b["parts"][target] -= dps * self.minions / max(self.max_minions, 1) / TICKS_PER_S
                if b["parts"][target] <= 0:
                    if target == "core":
                        self.boss, self.boss_killed, self.boss_kill_tick = None, True, self.ticks
                        self.projs = [p for p in self.projs if p.type != PHANTASMAL_DEATHRAY]
                        return
                    b["parts"][target] = 0
                    b["destroyed"].append(target)
        # Contact: core, both hands and the head (live: head contact was a top
        # hit source for the scripted dodger once it kept farther from the core).
        for bx, by in [(b["cx"] + dx, b["cy"] + dy) for dx, dy in self._offsets().values()]:
            if math.hypot(pcx - bx, pcy - by) < c["contact_radius"]:
                self._hurt(c["contact_damage"], "contact")
        if not c["boss_attacks"]:
            return
        t = self.ticks
        head = (b["cx"], b["cy"] - 400.0)
        off = self._offsets()
        hands = tuple((b["cx"] + off[n][0], b["cy"] + off[n][1]) for n in ("hand_l", "hand_r"))
        alive_hands, head_alive = hands, True  # destroyed parts keep attacking (see part_hp note)
        for nm in b["destroyed"]:
            if t % c["true_eye_period"] == 0:
                dx, dy = off[nm]
                for _ in range(c["true_eye_shots"]):
                    sp = c["eye_speed"] * self.rng.uniform(1 - c.get("eye_speed_spread", 0.0), 1 + c.get("eye_speed_spread", 0.0))
                    self._aimed(PHANTASMAL_EYE, b["cx"] + dx, b["cy"] + dy - 60, sp, c["eye_damage"],
                                spread=0.8, homing=c["eye_turn"], life=c["eye_life"])
        if alive_hands and t % c["bolt_period"] == 0:
            hx, hy = self.rng.choice(alive_hands)
            for _ in range(c["bolt_count"]):
                self._aimed(PHANTASMAL_BOLT, hx, hy, c["bolt_speed"], c["bolt_damage"], spread=c["bolt_spread"],
                            life=c["bolt_life"], lead=c["bolt_lead"])
        if alive_hands and t % c["sphere_period"] == 0:
            hx, hy = self.rng.choice(alive_hands)
            for _ in range(c["sphere_count"]):
                self._aimed(PHANTASMAL_SPHERE, hx, hy, c["sphere_speed"], c["sphere_damage"], spread=c["sphere_spread"],
                            life=c["sphere_life"], lead=c["sphere_lead"])
        if c["leech_period"] and t % c["leech_period"] == c["leech_period"] // 2:
            self._aimed(MOON_LEECH, b["cx"], b["cy"] - 180.0, c["leech_speed"], 0, spread=0.1,
                        homing=c["leech_turn"], life=c["leech_life"])
        if head_alive and t % c["eye_period"] == 0:
            for _ in range(3):
                # live Master logs: eyes leave the HANDS (and True Eyes), not the head
                if c.get("eye_origin") == "hands" and alive_hands:
                    ox, oy = self.rng.choice(alive_hands)
                else:
                    ox, oy = head[0] + self.rng.uniform(-200, 200), head[1]
                sp = c["eye_speed"] * self.rng.uniform(1 - c.get("eye_speed_spread", 0.0), 1 + c.get("eye_speed_spread", 0.0))
                self._aimed(PHANTASMAL_EYE, ox, oy, sp, c["eye_damage"], spread=0.8, homing=c["eye_turn"], life=c["eye_life"])
        if head_alive and t >= c["ray_first"] and (t - c["ray_first"]) % c["ray_period"] == 0:
            hy = b["cy"] - 400.0
            ang = math.atan2(pcy - hy, pcx - b["cx"]) - c["ray_sweep"] / 2 * self.rng.choice((-1, 1))
            ray = _Proj(PHANTASMAL_DEATHRAY, b["cx"], b["cy"] - 400.0, math.cos(ang), math.sin(ang), c["ray_damage"],
                        life=c["ray_duration"])
            ray.angle = ang
            ray.omega = c["ray_sweep"] / c["ray_duration"] * (1 if (ang < math.atan2(pcy - hy, pcx - b["cx"])) else -1)
            self.projs.append(ray)

    def _hittable(self, name):
        b = self.boss
        if name in b["destroyed"]:
            return False
        if name == "core":
            return len(b["destroyed"]) >= 3
        return True

    def _minion_target(self, pcx, pcy, aim):
        """Part the minions hit: the hittable part nearest the whip's aim
        point (whip tag), else the hittable part nearest the player."""
        b = self.boss
        off = self._offsets()
        cand = [n for n in off if self._hittable(n)]
        if not cand:
            return None
        ox, oy = aim if aim else (pcx, pcy)
        return min(cand, key=lambda n: (b["cx"] + off[n][0] - ox) ** 2 + (b["cy"] + off[n][1] - oy) ** 2)

    def _offsets(self):
        """Each part's offset from the core right now (hands move, see hand_*)."""
        b, c = self.boss, self.cfg
        hd = b.get("hand_dx") or {"hand_l": PART_OFFSET["hand_l"][0], "hand_r": PART_OFFSET["hand_r"][0]}
        return {"hand_l": (hd["hand_l"], c["hand_dy"]), "hand_r": (hd["hand_r"], c["hand_dy"]),
                "head": PART_OFFSET["head"], "core": PART_OFFSET["core"]}

    def _aimed(self, type_, x, y, speed, damage, spread=0.0, homing=0.0, life=600, lead=0.0):
        """Fire at the player; lead in [0, 1] blends aiming at where the
        player is now (0) with where they'll be at impact if they keep their
        velocity (1)."""
        pcx, pcy = self._centre()
        if lead:
            t = math.hypot(pcx - x, pcy - y) / max(speed, 1e-6)
            for _ in range(2):  # refine the intercept time once
                fx, fy = pcx + self.vx * t, pcy + self.vy * t
                t = math.hypot(fx - x, fy - y) / max(speed, 1e-6)
            pcx, pcy = pcx + lead * self.vx * t, pcy + lead * self.vy * t
        ang = math.atan2(pcy - y, pcx - x) + self.rng.uniform(-spread, spread)
        self.projs.append(_Proj(type_, x, y, speed * math.cos(ang), speed * math.sin(ang), damage,
                                life=life, homing=homing))

    def _projectiles_tick(self):
        c = self.cfg
        pcx, pcy = self._centre()
        hw, hh = c["player_w"] / 2, c["player_h"] / 2
        keep = []
        for p in self.projs:
            p.life -= 1
            if p.type == PHANTASMAL_DEATHRAY:
                if not self.boss:
                    continue
                p.x, p.y = self.boss["cx"], self.boss["cy"] - 400.0
                p.angle += p.omega
                p.vx, p.vy = math.cos(p.angle), math.sin(p.angle)
                # distance from player centre to the beam segment
                rx, ry = pcx - p.x, pcy - p.y
                along = rx * p.vx + ry * p.vy
                perp = abs(-rx * p.vy + ry * p.vx)
                if 0 <= along <= c["ray_length"] and perp <= c["ray_width"] / 2 + hw:
                    self._hurt(p.damage, SOURCE.get(p.type, "other"))
            else:
                if p.homing:
                    speed = math.hypot(p.vx, p.vy) or 1.0
                    cur = math.atan2(p.vy, p.vx)
                    want = math.atan2(pcy - p.y, pcx - p.x)
                    d = (want - cur + math.pi) % (2 * math.pi) - math.pi
                    cur += max(-p.homing, min(p.homing, d))
                    p.vx, p.vy = speed * math.cos(cur), speed * math.sin(cur)
                p.x += p.vx
                p.y += p.vy
                r = p.size / 2
                if abs(p.x - pcx) <= hw + r and abs(p.y - pcy) <= hh + r:
                    if p.type == MOON_LEECH:
                        if not getattr(self, "grabbed", 0) and not self.dead:
                            self.grabbed = self.cfg["leech_grab_ticks"]
                        continue
                    if self._hurt(p.damage, SOURCE.get(p.type, "other")):
                        continue  # projectile is used up by the hit
            if p.life > 0 and abs(p.x - pcx) < 4000 and abs(p.y - pcy) < 4000:
                keep.append(p)
        self.projs = keep

    def _hurt(self, damage, source="other"):
        if self.immune > 0 or self.dead:
            return False
        self.hp -= damage
        self.hits += 1
        self.damage_taken += damage
        self.damage_by[source] = self.damage_by.get(source, 0) + damage
        self.immune = self.cfg["immune_ticks"]
        if self.hp <= 0:
            self.dead, self.dead_ticks = True, 0
            self.deaths += 1
        return True
