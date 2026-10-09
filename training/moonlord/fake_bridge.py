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
import math
import random

import movement

TICKS_PER_S = 60
SUBSTEPS = 6  # game ticks per set_action (10 Hz decisions)

# Projectile type ids (real Terraria ProjectileID values).
PHANTASMAL_EYE = 452
PHANTASMAL_SPHERE = 454
PHANTASMAL_DEATHRAY = 455
PHANTASMAL_BOLT = 462
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
    "ceiling_px": 4000.0,         # world top this far above the floor (beyond the 3072 px arena)
    "player_hp": 500,
    "regen_per_s": 10.0,
    "player_w": 20, "player_h": 42,
    "wing_time_max": 150,         # measured; movement itself comes from movement.py (probe-fitted)
    "infinite_flight": False,     # flying mount: up/down steer, never runs out
    "immune_ticks": 40,           # Terraria's standard post-hit immunity
    "respawn_ticks": 300,
    # Boss
    "boss_hp": 217500,
    "minion_dps": 1658.0,         # measured, no whip
    # Live 2026-10-09 (random/scripted/human fights): the boss CHASES -- its core
    # moves ~8 px/tick (as fast as the player flies) and sits ~130 px BELOW
    # the player; the head is ~400 px above the core, hands ~400 px to each side.
    "boss_height": -130.0,        # core this far above the player (negative = below)
    "boss_drift": 8.0,            # px/tick max chase speed of the core
    "contact_radius": 45.0, "contact_damage": 80,   # core/hand hitboxes are 46x66 px (live); contact is a top hit source
    # Attacks
    "bolt_period": 330, "bolt_damage": 71, "bolt_speed": 8.0, "bolt_count": 3,
    "sphere_period": 280, "sphere_damage": 126, "sphere_speed": 10.0, "sphere_count": 6,
    "eye_period": 190, "eye_damage": 66, "eye_speed": 9.0, "eye_turn": 0.03, "eye_life": 240,
    "ray_period": 900, "ray_first": 1020, "ray_damage": 78, "ray_sweep": 1.4,
    "ray_duration": 120, "ray_width": 40.0, "ray_length": 2400.0,
    "boss_attacks": True,
}

SIZES = {PHANTASMAL_BOLT: 8, PHANTASMAL_SPHERE: 40, PHANTASMAL_EYE: 14, PHANTASMAL_DEATHRAY: 36}  # measured hitboxes
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
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.rng = random.Random(self.cfg["seed"])
        c = self.cfg
        self.spawn_x, self.floor = 0.0, c["floor_y"]
        self.arena = None
        self.keep_focus = False
        self.passthrough = False
        self.human_keys = {}  # tests: keys a pretend human holds in passthrough mode
        self.action = {}
        self.selected = 0
        self.hotbar = {0: "StardustDragonStaff", 1: "EmpressBlade", 2: "RainbowWhip"}
        self.whip_bonus = 1.25  # assumed minion DPS multiplier while the whip keeps tagging
        self.max_minions = 9  # measured with the Hallowed loadout + buffs
        self.minions = 0
        self._summon_hold = 0
        self.px, self.py = self.spawn_x, self.floor - c["player_h"]  # top-left
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
        self.boss = {"cx": cx, "cy": self.py - self.cfg["boss_height"], "hp": self.cfg["boss_hp"],
                     "ray": None}
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
            npcs.append({"type": MOON_LORD_CORE, "name": "Moon Lord", "x": b["cx"] - 60, "y": b["cy"] - 60,
                         "cx": b["cx"], "cy": b["cy"], "vx": 0.0, "vy": 0.0, "width": 120, "height": 120,
                         "hp": int(b["hp"]), "max_hp": c["boss_hp"], "hostile": True, "boss": True,
                         "moon_lord_part": True, "dont_take_damage": False})
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
                       "wing_time": self.wing_time, "wing_time_max": c["wing_time_max"],
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
        self.boss_killed, self.boss_kill_tick = False, -1

    def _episode(self):
        return {"ticks": self.ticks, "seconds": self.ticks / TICKS_PER_S, "hits": self.hits,
                "damage_taken": self.damage_taken, "deaths": self.deaths,
                "boss_killed": self.boss_killed, "boss_kill_tick": self.boss_kill_tick,
                "keep_focus": self.keep_focus,
                "arena": {"x": self.arena[0], "y": self.arena[1]} if self.arena else None}

    def _centre(self):
        return self.px + self.cfg["player_w"] / 2, self.py + self.cfg["player_h"] / 2

    def _on_ground(self):
        return self.py + self.cfg["player_h"] >= self.floor - 0.01

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
        h = (1 if a.get("right") else 0) - (1 if a.get("left") else 0)
        self.vx, self.vy = movement.step(self.vx, self.vy, h, movement.mode_of(a), self._on_ground())
        self.px += self.vx
        self.py += self.vy
        left, right = self.spawn_x - c["arena_half_width"], self.spawn_x + c["arena_half_width"] - c["player_w"]
        if not left <= self.px <= right:
            self.px, self.vx = max(left, min(right, self.px)), 0.0  # walls stop you dead
        if self.py + c["player_h"] >= self.floor:
            self.py = self.floor - c["player_h"]
            self.vy = min(self.vy, 0.0)
        top = self.floor - c["ceiling_px"]
        if self.py < top:
            self.py, self.vy = top, 0.0  # world height limit

    def _summon(self):
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
        b["cx"] += max(-c["boss_drift"], min(c["boss_drift"], pcx - b["cx"]))
        b["cy"] += max(-c["boss_drift"], min(c["boss_drift"], (pcy - c["boss_height"]) - b["cy"]))
        if self.minions > 0:
            whip = self.whip_bonus if (self.action.get("use") and self._whip_selected()) else 1.0
            b["hp"] -= c["minion_dps"] * whip * self.minions / max(self.max_minions, 1) / TICKS_PER_S
        if b["hp"] <= 0:
            self.boss, self.boss_killed, self.boss_kill_tick = None, True, self.ticks
            self.projs = [p for p in self.projs if p.type != PHANTASMAL_DEATHRAY]
            return
        for bx, by in ((b["cx"], b["cy"]), (b["cx"] - 400.0, b["cy"] - 150.0), (b["cx"] + 400.0, b["cy"] - 150.0)):
            if math.hypot(pcx - bx, pcy - by) < c["contact_radius"]:
                self._hurt(c["contact_damage"])
        if not c["boss_attacks"]:
            return
        t = self.ticks
        head = (b["cx"], b["cy"] - 400.0)
        hands = ((b["cx"] - 400.0, b["cy"] - 150.0), (b["cx"] + 400.0, b["cy"] - 150.0))
        if t % c["bolt_period"] == 0:
            hx, hy = self.rng.choice(hands)
            for _ in range(c["bolt_count"]):
                self._aimed(PHANTASMAL_BOLT, hx, hy, c["bolt_speed"], c["bolt_damage"], spread=0.12)
        if t % c["sphere_period"] == 0:
            hx, hy = self.rng.choice(hands)
            for _ in range(c["sphere_count"]):
                self._aimed(PHANTASMAL_SPHERE, hx, hy, c["sphere_speed"], c["sphere_damage"], spread=0.35)
        if t % c["eye_period"] == 0:
            for _ in range(3):
                self._aimed(PHANTASMAL_EYE, head[0] + self.rng.uniform(-200, 200), head[1],
                            c["eye_speed"], c["eye_damage"], spread=0.8, homing=c["eye_turn"], life=c["eye_life"])
        if t >= c["ray_first"] and (t - c["ray_first"]) % c["ray_period"] == 0:
            hy = b["cy"] - 400.0
            ang = math.atan2(pcy - hy, pcx - b["cx"]) - c["ray_sweep"] / 2 * self.rng.choice((-1, 1))
            ray = _Proj(PHANTASMAL_DEATHRAY, b["cx"], b["cy"] - 400.0, math.cos(ang), math.sin(ang), c["ray_damage"],
                        life=c["ray_duration"])
            ray.angle = ang
            ray.omega = c["ray_sweep"] / c["ray_duration"] * (1 if (ang < math.atan2(pcy - hy, pcx - b["cx"])) else -1)
            self.projs.append(ray)

    def _aimed(self, type_, x, y, speed, damage, spread=0.0, homing=0.0, life=600):
        pcx, pcy = self._centre()
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
                    self._hurt(p.damage)
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
                    if self._hurt(p.damage):
                        continue  # projectile is used up by the hit
            if p.life > 0 and abs(p.x - pcx) < 4000 and abs(p.y - pcy) < 4000:
                keep.append(p)
        self.projs = keep

    def _hurt(self, damage):
        if self.immune > 0 or self.dead:
            return False
        self.hp -= damage
        self.hits += 1
        self.damage_taken += damage
        self.immune = self.cfg["immune_ticks"]
        if self.hp <= 0:
            self.dead, self.dead_ticks = True, 0
            self.deaths += 1
        return True
