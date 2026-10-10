"""Offline checks for the Moon Lord harness (Goal 2, stage M0). No game,
no brain: runs the harness against fake_bridge.py, and FlyPolicy with a
stub brain. Run: python training/moonlord/test_harness.py"""
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
import policies  # noqa: E402
from fake_bridge import FakeMoonLordBridge  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def cfg(**over):
    c = {**harness.DEFAULTS, "max_seconds": dict(harness.DEFAULTS["max_seconds"])}
    c.update(over)
    return c


def episodes(policy, n, conf, fake_over=None):
    rows = []
    for s in range(n):
        b = FakeMoonLordBridge({"seed": s, **(fake_over or {})})
        rows.append(harness.run_episode(b, policy, conf, s))
    return rows


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


# 1. The fake speaks the same JSON as the mod: every key GameState.cs emits
#    for player / npcs / projectiles is present in the fake's get_state.
gs = (HERE.parents[1] / "bridge/mod/FruitFlyBridge/GameState.cs").read_text()
mod_keys = set(re.findall(r'\["([a-z_]+)"\]\s*=', gs))
b = FakeMoonLordBridge()
b.request({"cmd": "spawn_npc"})
b.request({"cmd": "spawn_projectile", "dx": 300, "dy": 0, "vx": -5})
st = b.request({"cmd": "get_state"})
fake_keys = set(st) | set(st["player"]) | set(st["npcs"][0]) | set(st["projectiles"][0])
debug_only = {"has_focus", "control_left", "control_right", "velocity_x", "is_operating_another_entity",
              "frozen", "stoned", "cc_immune", "game_menu", "local_player_null", "my_player_index",
              "net_mode", "origin_tx", "origin_ty", "types", "solid", "name",
              # item-use diagnostics (debug block only)
              "control_use_item", "release_use_item", "delay_use_item", "item_animation", "mouse_interface",
              "held_item_type", "held_item_name", "mouse_world_x", "mouse_world_y", "whip_projectiles",
              "game_is_active", "game_paused", "control_jump", "control_up", "control_down"}
missing = sorted(mod_keys - fake_keys - debug_only)
check("fake get_state has every key the mod emits", not missing, f"missing {missing}")

# 2. Idle loses; the scripted dodger lasts clearly longer.
conf = cfg(summon_slots=[0, 1])
idle = episodes(policies.IdlePolicy(), 6, conf)
# Checks the dodging machinery, not the tuning. Live (2026-10-09) the default
# dodger lasts ~2.1x idle (38.6 s vs 18.3 s); the fitted fake gives ~1.9x.
scr = episodes(policies.ScriptedDodger(), 6, conf)
check("idle never wins a fight", not any(r["win"] for r in idle))
check("scripted survives > 1.6x longer than idle (median)",
      median(r["seconds"] for r in scr) > 1.6 * median(r["seconds"] for r in idle),
      f"{median(r['seconds'] for r in scr)}s vs {median(r['seconds'] for r in idle)}s")

# 3. Credit ledger adds up and names the right source.
check("idle ledger is all 'none'", all(set(r["ledger"]) == {"none"} for r in idle))
check("ledger ticks sum to episode ticks", all(sum(r["ledger"].values()) == r["ticks"] for r in idle + scr))
check("scripted ledger credits 'script'", all(r["ledger"].get("script", 0) > 0 for r in scr))

# 4. Setup summons minions, and setup is not counted as play.
b = FakeMoonLordBridge()
row = harness.run_episode(b, policies.IdlePolicy(), cfg(summon_slots=[0, 1], max_seconds={"fight": 1.0}), 0)
check("setup fills minion slots", b.minions == b.max_minions, f"{b.minions}/{b.max_minions}")
check("setup ticks reported separately", row["setup_ticks"] > 0 and sum(row["ledger"].values()) == row["ticks"])

# 5. A winnable fight ends as a win (boss killed by minions, no attacks).
rows = episodes(policies.IdlePolicy(), 1, cfg(summon_slots=[0, 1]),
                {"boss_hp": 3000, "boss_attacks": False, "contact_damage": 0})
check("win is detected", rows[0]["outcome"] == "win" and rows[0]["boss_hp_left"] == 0, rows[0]["outcome"])

# 6. Back-to-back episodes on one bridge: the second waits for the respawn
#    after a death, and starts clean.
b = FakeMoonLordBridge({"seed": 3})
r1 = harness.run_episode(b, policies.IdlePolicy(), cfg(), 3)
r2 = harness.run_episode(b, policies.IdlePolicy(), cfg(), 4)
check("episode after a death starts clean", r1["outcome"] == "loss" and r2["seconds"] > 1 and r2["hits"] > 0,
      f"{r1['outcome']} then {r2['outcome']} {r2['seconds']}s")

# 7. Gym mode: volleys arrive, idle takes damage, the dodger takes less.
gconf = cfg(mode="gym")
gi = episodes(policies.IdlePolicy(), 6, gconf, {"boss_attacks": False})
gs_ = episodes(policies.ScriptedDodger(), 6, gconf, {"boss_attacks": False})
mean = lambda rows: sum(r["damage_taken"] for r in rows) / len(rows)
check("gym fires projectiles that hit an idle player", mean(gi) > 0, f"{mean(gi):.0f} dmg")
check("scripted takes less gym damage than idle", mean(gs_) < mean(gi), f"{mean(gs_):.0f} vs {mean(gi):.0f}")

# 8. FlyPolicy plumbing with a stub brain (no connectome needed).
sys.path.insert(0, str(HERE.parents[1] / "brain"))
import neurons  # noqa: E402


class StubBrain:
    def __init__(self):
        self.calls = []

    def stimulate(self, ids, rates, duration_ms):
        self.calls.append((list(ids), list(rates)))
        return None

    def reset_state(self):
        pass


stub = StubBrain()
# Giant Fiber firing + DNa02 lean: what a looming threat produces.
fake_rates = {f: 0.0 for f in neurons.ALL_DESCENDING}
for f in neurons.DNP01_JUMP:
    fake_rates[f] = 120.0
for f in neurons.DNA02_STEERING_LEFT:
    fake_rates[f] = 80.0
fly = policies.FlyPolicy(brain=stub, readout_fn=lambda _spikes: fake_rates, vertical="rule")
fly.reset()
p = {"cx": 0.0, "cy": 0.0, "x": -10.0, "y": -21.0, "direction": 1, "dead": False}
state = {"in_world": True, "player": p, "npcs": [], "players": [], "projectiles": [
    {"type": 462, "cx": 300.0, "cy": 0.0, "vx": -10.0, "vy": 0.0, "width": 10, "height": 10, "damage": 50},  # incoming
    {"type": 462, "cx": -300.0, "cy": 0.0, "vx": -10.0, "vy": 0.0, "width": 10, "height": 10, "damage": 50},  # leaving
]}
seen = fly.threats_as_npcs(state)
check("only the incoming projectile is a threat", len(seen) == 1 and seen[0]["x"] == 300.0, str(seen))
action, sources = fly.act(state)
lc4 = set(neurons.LC4_LOOMING_LEFT) | set(neurons.LC4_LOOMING_RIGHT)
ids, rates = stub.calls[-1]
check("the threat drives LC4", any(i in lc4 and r > 0 for i, r in zip(ids, rates)))
check("fleeing fly: direction from the brain, flight from the rule, no up/use",
      sources.get("left") == "brain" and sources.get("jump") == "rule"
      and not action["up"] and not action["use"], f"{action} {sources}")

# 8b. Attention: an incoming bolt about to hit beats the closer but slow core.
core_state = {"in_world": True, "player": dict(p), "players": [], "projectiles": [
    {"type": 462, "cx": 200.0, "cy": 0.0, "vx": -10.0, "vy": 0.0, "width": 8, "height": 8, "damage": 70}],
    "npcs": [{"type": 398, "x": -23.0, "y": 97.0, "cx": 0.0, "cy": 130.0, "vx": 0.0, "vy": 0.0,
              "moon_lord_part": True, "hostile": True}]}
urgent = fly.most_urgent(core_state)
check("attention picks the bolt (20 ticks away) over the core", urgent and urgent.get("from_projectile") == 462, str(urgent))

# 8c. Sawtooth: climbing fast switches to a dive; falling fast switches back.
fly2 = policies.FlyPolicy(brain=StubBrain(), readout_fn=lambda _s: fake_rates, vertical="rule")
fly2.reset()
st_up = {**state, "player": {**p, "vy": -12.0}}
a1, _ = fly2.act(st_up)
st_down = {**state, "player": {**p, "vy": 7.0}}
a2, _ = fly2.act(st_down)
check("sawtooth: dive after a fast climb, climb after a fast fall",
      a1["down"] and not a1["jump"] and a2["jump"] and not a2["down"], f"{a1} {a2}")
fly3 = policies.FlyPolicy(brain=StubBrain(), readout_fn=lambda _s: fake_rates, fly_flee=False)
fly3.reset()
a3, s3 = fly3.act(state)
check("fly_flee=False leaves the decoder's own action, all credited 'brain'",
      all(v == "brain" for v in s3.values()) and not a3["down"], f"{a3} {s3}")

# 8d. Vertical from the brain: landing (DNp07) beats the Giant Fiber -> dive away
#     from the stronger DNp07 side; Giant Fiber alone -> climb. Threats split by height.
land_rates = {f: 0.0 for f in neurons.ALL_DESCENDING}
for f in neurons.DNP07_LANDING_LEFT:
    land_rates[f] = 150.0
for f in neurons.DNP01_JUMP:
    land_rates[f] = 40.0
flyb = policies.FlyPolicy(brain=StubBrain(), readout_fn=lambda _s: land_rates)
flyb.reset()
a, srcs = flyb.act(state)
check("brain vertical: DNp07_L > Giant Fiber -> dive to the right",
      a["down"] and a["right"] and not a["jump"] and set(srcs.values()) == {"brain"}, f"{a} {srcs}")
flyc = policies.FlyPolicy(brain=StubBrain(), readout_fn=lambda _s: fake_rates)
flyc.reset()
a, srcs = flyc.act(state)
check("brain vertical: Giant Fiber only -> climb (jump held), credited brain",
      a["jump"] and not a["down"] and set(srcs.values()) == {"brain"}, f"{a} {srcs}")
st_ab = {**state, "projectiles": [
    {"type": 462, "cx": 0.0, "cy": -300.0, "vx": 0.0, "vy": 10.0, "width": 8, "height": 8, "damage": 70},
    {"type": 462, "cx": 300.0, "cy": 0.0, "vx": -10.0, "vy": 0.0, "width": 8, "height": 8, "damage": 70}]}
above, level = flyc.urgent_by_elevation(st_ab)
check("threats split by height (above -> LPLC4, level -> LC4)",
      above and above["y"] == -300.0 and level and level["y"] == 0.0, f"{above} {level}")
lp_ids, lp_rates = flyc._lplc4_drive(st_ab, above)
check("LPLC4 channel is driven for a threat from above",
      set(lp_ids) <= set(neurons.LPLC4_LOOMING) and max(lp_rates) > 0, f"{len(lp_ids)} ids")

# 8e. Direction commitment: a brief reversal is ignored, a sustained one is taken.
flyk = policies.FlyPolicy(brain=StubBrain(), readout_fn=lambda _s: fake_rates, commit_ticks=4)
flyk.reset()
seq = [flyk._commit(w)["right"] for w in (1, -1, -1, 1, -1, -1, -1, -1, -1)]
check("commitment: holds right through a 2-tick reversal, switches after 4",
      seq[:4] == [True, True, True, True] and seq[-1] is False, str(seq))

# 9. Whip assist: swings at the boss with aim, credited "assist", and the
#    fly/scripted policies are never credited for `use`.
class Spy(FakeMoonLordBridge):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.sent = []

    def _cmd_set_action(self, p):
        self.sent.append(dict(p))
        return super()._cmd_set_action(p)


spy = Spy({"seed": 1})
row = harness.run_episode(spy, policies.ScriptedDodger(),
                          cfg(summon_slots=[0], whip_slot=2, whip_assist=True, max_seconds={"fight": 5.0}), 1)
swings = [a for a in spy.sent if a.get("use") and a.get("aim_x") is not None]
check("whip assist swings with an aim point", len(swings) > 10, f"{len(swings)} aimed swings")
check("whip swings are credited 'assist'", row["ledger"].get("assist", 0) > 0, str(row["ledger"]))
check("whip slot is selected after summoning", spy.selected == 2)
t = harness.whip_target({"player": {"cx": 0, "cy": 0}, "npcs": [
    {"x": 0, "y": 0, "cx": 10, "cy": 0, "moon_lord_part": True, "dont_take_damage": True},
    {"x": 0, "y": 0, "cx": 500, "cy": 0, "moon_lord_part": True, "dont_take_damage": False}]})
check("whip aims at a hittable part over a closer invulnerable one", t == (500, 0), str(t))
b = FakeMoonLordBridge({"seed": 0, "boss_attacks": False, "contact_damage": 0, "boss_hp": 20000})
r_whip = harness.run_episode(b, policies.IdlePolicy(), cfg(summon_slots=[0], whip_slot=2, whip_assist=True), 0)
b = FakeMoonLordBridge({"seed": 0, "boss_attacks": False, "contact_damage": 0, "boss_hp": 20000})
r_none = harness.run_episode(b, policies.IdlePolicy(), cfg(summon_slots=[0]), 0)
check("whip shortens the (fake) kill time", r_whip["seconds"] < r_none["seconds"],
      f"{r_whip['seconds']}s vs {r_none['seconds']}s")

# 11. Regression: the scripted dodger with an arena mark (crashed live on tick 1).
b = FakeMoonLordBridge({"seed": 5})
b.request({"cmd": "set_arena"})
row = harness.run_episode(b, policies.ScriptedDodger(arena_half_width=5760, arena_height=2760),
                          cfg(summon_slots=[0], max_seconds={"fight": 5.0}), 5)
check("scripted dodger runs with an arena mark", row["ticks"] > 10, row["outcome"])

# 12. Leash: random keys can't climb above the arena or leave its sides.
b = FakeMoonLordBridge({"seed": 6, "boss_attacks": False, "contact_damage": 0})
b.request({"cmd": "set_arena"})
small = {"half_width_tiles": 20, "rows": 4, "row_spacing_tiles": 12}
row = harness.run_episode(b, policies.RandomPolicy(seed=6), cfg(arena=small, max_seconds={"fight": 60.0}), 6)
height = 4 * 12 * 16
check("leash keeps random keys under the arena top", b.py > b.arena[1] - height - 200,
      f"top reached {b.arena[1] - b.py:.0f}px above the mark, arena {height}px")
check("leash keeps random keys inside the sides", abs(b.px - b.arena[0]) < 20 * 16 + 200, f"x off {b.px - b.arena[0]:.0f}px")
check("leash interventions are counted", row["ledger"].get("leash", 0) > 0, str(row["ledger"]))

# 13. Setup keeps casting until the minion slots are full.
b = FakeMoonLordBridge()
harness.run_episode(b, policies.IdlePolicy(), cfg(summon_slots=[1], max_seconds={"fight": 1.0}), 0)
check("setup fills all 9 minion slots", b.minions == b.max_minions, f"{b.minions}/{b.max_minions}")

# 14. Heal assist: drinks at <= 300 HP, once per potion sickness, after a restock.
b = FakeMoonLordBridge({"seed": 2})
b.potions = 0
row = harness.run_episode(b, policies.IdlePolicy(), cfg(summon_slots=[1], heal_at_hp=300, restock_potions=10), 2)
check("restock + heal assist drink a potion", row["heals"] >= 1 and b.potions == 10 - row["heals"],
      f"heals {row['heals']}, potions left {b.potions}")
b = FakeMoonLordBridge({"seed": 2})
row = harness.run_episode(b, policies.IdlePolicy(), cfg(summon_slots=[1], heal_at_hp=0), 2)
check("heal_at_hp=0 turns the assist off", row["heals"] == 0)

# 15. Predictive leash: a fast climb is turned into a hover before the top.
st = {"episode": {"arena": {"x": 0.0, "y": 0.0}},
      "player": {"cx": 10.0, "cy": -2200.0, "vy": -18.0, "width": 20, "height": 42}}
out, changed = harness.leash(st, {"jump": True, "up": True}, {"half_width_tiles": 100, "rows": 16,
                                                              "row_spacing_tiles": 12, "max_rise_tiles": 160})
check("leash stops a climb that would coast past the top",
      not out["jump"] and not out["up"] and set(changed) == {"up", "jump"}, f"{out} {changed}")

# 15b. The leash never climbs into reduced (space) gravity.
st = {"episode": {"arena": {"x": 0.0, "y": 0.0}, "full_gravity_y": -1000.0},
      "player": {"cx": 10.0, "cy": -980.0, "vy": 0.0, "width": 20, "height": 42}}
out, changed = harness.leash(st, {"jump": True}, {"half_width_tiles": 100, "rows": 12,
                                                   "row_spacing_tiles": 24, "max_rise_tiles": 280})
check("leash stops climbing at the full-gravity line", not out["jump"] and changed == ["jump"], f"{out} {changed}")

# 16. Human demos: passthrough on during the episode, off after; the human's
#     keys are logged and credited "human"; free mode spawns nothing.
b = FakeMoonLordBridge({"seed": 9})
b.human_keys = {"right": True, "jump": True}
row = harness.run_episode(b, policies.HumanPolicy(), cfg(mode="free", max_seconds={"free": 2.0}), 9)
check("human demo logs the human's keys", row["ledger"].get("human", 0) > 0 and b.vx > 0, str(row["ledger"]))
check("passthrough is switched off after the demo", b.passthrough is False)
check("free mode spawns no boss", b.boss is None and row["outcome"] == "survived", row["outcome"])

# 10. loadout.json parses and every slot entry has a usable shape.
import json  # noqa: E402
lo = json.loads((HERE / "loadout.json").read_text())
shape_ok = all(isinstance(x, (str, int, list)) or x is None for x in lo["armor"] + lo["accessories"] + lo["hotbar"])
check("loadout.json entries are names / ids / alternative lists", shape_ok and len(lo["armor"]) == 3)

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
