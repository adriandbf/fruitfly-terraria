# Phase 2: bridge (bot-client mod)

A tModLoader mod, `FruitFlyBridge`, that runs on the Mac's client. It exposes
a local, loopback-only JSON-over-TCP server so the Python control loop can
read game state and hold keys, without either side needing to know the
other's internals. This is the "small JSON-over-socket API" from CLAUDE.md,
so the bridge (this mod, or later HeadlessTerrariaClient) can be swapped
without touching the brain/encoder/decoder code.

Runs only on the Mac, under the Mac's own Steam account — see the Phase 2
identity decision in CLAUDE.md. The PC's server and the human's client on
the PC never need this mod (`side = Client` in build.txt).

## Status: working

Confirmed end to end on the Mac against tModLoader v2026.07.3.0: builds
clean, `get_state` returns real position/health/NPC/tile data while
standing in `FruitFlyWorld`, and `set_action` actually moves the character
(held "right" via the socket walked it right in-game). `use` (attacking/
item use) hasn't been separately verified yet — only left/right/jump.

Getting movement working took three attempts, mostly around one hook
(`ModPlayer.ProcessTriggers`): the first version set `Player.controlX`
unconditionally every tick, which compiled fine but silently blocked ALL
input, including the real keyboard, because it kept forcing every key to
"released" whenever no bridge client was connected (the default state).
The fix for that got bundled with an incorrect switch to writing the
`TriggersSet` parameter instead of `Player.controlX`, which turned out to
be backwards — decompiling the actual game code
(`Terraria.GameInput.TriggersSet.CopyInto`) showed the engine copies
`TriggersSet` onto `Player.controlLeft`/`Right`/`Jump`/`controlUseItem`
*before* calling our hook, so writing back to `TriggersSet` there is a
no-op for movement. `BridgePlayer.cs` now sets `Player.controlX` directly,
gated on `BridgeSystem.IsClientConnected`, which is the combination that
actually works.

## Install on the Mac

tModLoader looks for mod source under
`~/Library/Application Support/Terraria/tModLoader/ModSources/` on macOS.
Symlink this folder in so editing in the repo is what gets built:

```bash
ln -s "$(pwd)/bridge/mod/FruitFlyBridge" \
  ~/"Library/Application Support/Terraria/tModLoader/ModSources/FruitFlyBridge"
```

Then in tModLoader: **Workshop > Mod Sources**, find `FruitFlyBridge`, and
click **Build + Reload**. Fix any compile errors it reports — expect at
least one round of this on the first build. Once it builds, enable it in
**Mods** and reload.

`obj/` and `bin/` that the build creates inside the symlinked folder are
git-ignored.

## Protocol

Newline-delimited JSON over TCP, `127.0.0.1:7778`, one request/response
per line. Never listens on a non-loopback address.

**`{"cmd": "ping"}`** → `{"ok": true, "pong": true}`

**`{"cmd": "get_state"}`** →
```json
{
  "ok": true,
  "in_world": true,
  "player": {"x": 1234.0, "y": 678.0, "direction": 1, "hp": 80, "max_hp": 100,
             "mana": 20, "max_mana": 20, "on_ground": true,
             "dead": false, "ghost": false},
  "npcs": [{"type": 1, "name": "Zombie", "x": 1300.0, "y": 670.0, "hp": 250, "hostile": true}],
  "players": [{"name": "Steve", "x": 1400.0, "y": 660.0, "hp": 90, "max_hp": 100}],
  "tiles": {"origin_tx": 70, "origin_ty": 40, "width": 31, "height": 17,
            "types": [-1, 0, 0, "..."], "solid": [false, true, true, "..."]}
}
```
If not currently in a world: `{"ok": true, "in_world": false}`.

`players` lists every OTHER active player in the world (not the bot's own
character, and never NPCs) — added for Phase 4, since the encoder needs the
human's position for a "walk toward the human" drive and `npcs` never
contains other players. Untested against a real build (no rebuild/live
test done yet as of this addition — see CLAUDE.md's Phase 4 status).

NPCs are limited to those within 800px (~50 tiles) of the player. The tile
grid is a flat row-major array centred on the player, ±15 tiles horizontally
and ±8 vertically; `-1` marks out-of-bounds or air.

**`{"cmd": "set_action", "left": bool, "right": bool, "jump": bool, "use": bool}`**
→ `{"ok": true}`. Keys are *held* — the mod keeps applying the most recent
action every game tick until a new one arrives, matching CLAUDE.md's "keys
held between decisions" control loop. There is no separate "release" command;
send all-false to release everything.

Unknown `cmd` → `{"ok": false, "error": "..."}`.

## Testing it

Once the mod is built, enabled, and you've joined `FruitFlyWorld` on the Mac:

```bash
python bridge/smoke_test_client.py
```

It pings the mod, prints one state snapshot, then holds right for a second
so you should see the character walk right in-game.

## Scripted dumb policy: `dumb_bot.py`

`bridge/dumb_bot.py` is the Phase 2 "dumb bot": polls `get_state` and calls
`set_action` at 10 Hz, logging every tick to `logs/dumb_bot_<timestamp>.jsonl`
(git-ignored). No brain involved — that's Phase 4+.

Policy, in priority order: **flee** a nearby hostile if health drops below
30% of max, else **attack** the nearest hostile within range, else
**patrol** back and forth. All three, plus the tile-based auto-jump, are
confirmed working against the live game (see logs above for an example run
where the character fought, dropped to 0 HP under `flee`, died, respawned,
and the bot picked back up in `attack` without crashing).

```bash
python bridge/dumb_bot.py              # runs until Ctrl+C
python bridge/dumb_bot.py --seconds 20 # fixed duration, for smoke tests
```

Dead/ghost handling: `get_state`'s `player` now carries `dead` and `ghost`
booleans (vanilla `Player.Update` branches on both before any control
processing, so while either is true no input does anything). Originally a
known gap — a live test where the character died to a zombie swarm showed
the bot happily issuing actions and logging for the rest of the run.
`dumb_bot.py` releases all keys while dead; `interface/run_fly.py` also
stops stepping the brain, then resets neural state on respawn so the new
life doesn't inherit frozen mid-panic activity. The Python side is tested
offline against a fake bridge; the C# fields need a rebuild and a live
death to confirm end to end.

## Next steps (not yet built)

- Logging is per-run only; nothing yet stitches multiple runs together or
  visualizes them.
- Remove the temporary `debug` block in `GameState.BuildSnapshot()`'s
  `in_world: false` response — it was added to diagnose a since-resolved
  false negative and can go once nobody's hit that again.
- Also temporary: a second `debug` block in the normal (`in_world: true`)
  response, added to diagnose a held `right` action producing zero
  movement in a live multiplayer test even with manual keyboard control
  confirmed working — surfaces `Main.hasFocus`, the actual applied
  `controlLeft`/`controlRight`/`velocity.X`, and known movement-blocking
  flags (`isOperatingAnotherEntity`, mount, frozen/stoned/CCed). Remove
  once that's root-caused.
- This whole policy is a placeholder to prove the loop, not a real
  behavior spec — Phase 4 replaces it with the actual encoder/decoder
  around the fly brain.

## Goal 2 (Moon Lord) additions

Added for the Moon Lord harness (`training/moonlord/`). None of this has
been compiled or run live yet.

**`set_action`** also takes `up` and `down` (wings, flying mounts,
platforms) and an optional `aim_x`/`aim_y` (world pixels). The aim moves
the cursor after input is read, so item use (the whip) swings that way;
it assumes zoom 1×. Omitted keys are released. Actions are now applied from
both `ProcessTriggers` and `SetControls`.

**`get_state`** adds:
- `player`: `cx`/`cy` (centre), `vx`/`vy`, `width`/`height`, `wing_time`/`wing_time_max`, `mount_active`, `immune`, `minions`/`max_minions`, `selected_item`.
- `npcs`: bosses and every Moon Lord part are always included, whatever the distance. Each NPC adds `cx`/`cy`, `vx`/`vy`, `width`/`height`, `max_hp`, `boss`, `moon_lord_part`, `dont_take_damage`. `x`/`y` are still the top-left corner.
- `projectiles`: hostile projectiles within 1600px, each with `type`, `cx`/`cy`, `vx`/`vy`, `width`/`height`, `damage`. For the deathray (type 455), `vx`/`vy` is the beam direction.
- `episode`: `ticks`/`seconds` since the last reset, `hits`, `damage_taken`, `deaths`, `boss_killed`, `boss_kill_tick`, `keep_focus`, `arena`.

**New commands.** All except the first two run on the game thread. Each
one waits up to 3 s, then returns `{"ok": false, "error": "game thread did not run..."}`.

| Command | Effect |
| --- | --- |
| `get_episode` | The `episode` block on its own |
| `set_keep_focus {enabled}` | While connected, force `Main.hasFocus = true` after input handling, so held keys apply with the window in the background (untested) |
| `set_arena {x?, y?}` | Where `reset_episode` puts the player; defaults to the current position |
| `reset_episode` | Despawn hostile NPCs and projectiles, heal, clear debuffs, move to the arena, release keys, zero counters. Refused while dead |
| `spawn_npc {type?}` | `NPC.SpawnOnPlayer`; default 398 (Moon Lord core) |
| `spawn_projectile {type?, x?/y? or dx?/dy?, vx, vy, damage?, knockback?}` | Hostile projectile for the dodge gym; default type 462 (Phantasmal Bolt) |
| `select_slot {slot}` | Select hotbar slot 0–9 (to cast summon staffs) |
| `equip_loadout {armor, accessories, hotbar, inventory, max_stats}` | Put items on the loaded character. Each item is an internal name (`ItemID.Search`), a numeric id, or a list of alternatives; `max_stats` consumes every Life Crystal, Life Fruit and Mana Crystal and turns on the Demon Heart slot. Unknown names come back in `unknown_items` |
| `apply_buffs {names, seconds}` | `AddBuff` by `BuffID` name. Unknown names come back in `unknown_buffs` |
| `build_arena {x?, y?, half_width_tiles, rows, row_spacing_tiles, furniture_spacing_tiles, floor_item, campfire_item, lantern_item, hanger_block_item, replace_platforms}` | Planter-box rows centred on `x`/`y` (default: the player), campfires on the lowest row, heart lanterns hanging from a block set into the second row; wooden platforms in the rows are replaced. Uses each item's own `createTile`/`placeStyle`. Reports every campfire and lantern, then marks the arena |
