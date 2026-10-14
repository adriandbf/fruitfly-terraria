# FruitFlyBridge: the game side

A client-side tModLoader mod (`mod/FruitFlyBridge/`) that lets Python read
the game and hold keys. It runs a loopback-only JSON-over-TCP server on
`127.0.0.1:7778`: one JSON request per line, one JSON response per line.
`client.py` is the Python client (`Bridge().request({"cmd": "get_state"})`).

It also runs the Moon Lord fight harness's game-side jobs: reset an
episode, equip the loadout, build the arena and the Nurse house, swing the
whip assist, keep input flowing while the window is unfocused.

## Install

Link this folder into tModLoader's mod sources so the repo copy is what gets built:

- **macOS**
  ```bash
  ln -s "$(pwd)/bridge/mod/FruitFlyBridge" \
    ~/"Library/Application Support/Terraria/tModLoader/ModSources/FruitFlyBridge"
  ```
- **Windows** (PowerShell; `setup_pc.ps1` does this for you)
  ```powershell
  New-Item -ItemType Junction -Target "$PWD\bridge\mod\FruitFlyBridge" `
    -Path "$([Environment]::GetFolderPath('MyDocuments'))\My Games\Terraria\tModLoader\ModSources\FruitFlyBridge"
  ```

Then in tModLoader: **Workshop > Develop Mods** (Mod Sources), find `FruitFlyBridge`,
**Build + Reload**, and make sure it is enabled under **Mods**. Built against
tModLoader v2026.07.3.0 (Terraria 1.4.4 content). `bin/` and `obj/` are git-ignored.

## How keys are applied

`BridgePlayer.cs` writes `Player.controlLeft/Right/Jump/...` directly (in
`ProcessTriggers` and `SetControls`), only while a bridge client is
connected, so the real keyboard works normally otherwise. Keys are held
until the next `set_action`; omitted keys are released. A held jump gives
one jump (the engine needs a released frame between jumps).

## Protocol

| Command | Effect |
| --- | --- |
| `ping` | `{"ok": true, "pong": true}` |
| `get_state` | Player, NPCs, hostile projectiles, a tile grid around the player, the episode counters (below) |
| `set_action {left, right, jump, use, up, down, aim_x?, aim_y?}` | Hold these keys; aim (world pixels) points item use, e.g. the whip |

Unknown commands return `{"ok": false, "error": "..."}`. While not in a
world, `get_state` returns `{"ok": true, "in_world": false}`.

The aim moves the cursor after input is read and assumes game zoom 1×.

**`get_state`** fields beyond position and health:
- `player`: `cx`/`cy` (centre), `vx`/`vy`, `width`/`height`, `wing_time`/`wing_time_max`, `mount_active`, `immune`, `minions`/`max_minions`, `selected_item`.
- `npcs`: bosses and every Moon Lord part are always included, whatever the distance. Each NPC adds `cx`/`cy`, `vx`/`vy`, `width`/`height`, `max_hp`, `boss`, `moon_lord_part`, `dont_take_damage`. `x`/`y` are still the top-left corner.
- `projectiles`: hostile projectiles within 1600px, each with `type`, `cx`/`cy`, `vx`/`vy`, `width`/`height`, `damage`. For the deathray (type 455), `vx`/`vy` is the beam direction.
- `episode`: `ticks`/`seconds` since the last reset, `hits`, `damage_taken`, `deaths`, `boss_killed`, `boss_kill_tick`, `keep_focus`, `arena`.

**Harness commands.** All except `get_episode` and `set_keep_focus` run on the game thread. Each
one waits up to 3 s, then returns `{"ok": false, "error": "game thread did not run..."}`.

| Command | Effect |
| --- | --- |
| `get_episode` | The `episode` block on its own |
| `set_keep_focus {enabled}` | While connected, force `Main.hasFocus = true` after input handling, so held keys apply with the window in the background  |
| `set_arena {x?, y?}` | Where `reset_episode` puts the player; defaults to the current position |
| `reset_episode` | Despawn hostile NPCs and projectiles, heal, clear debuffs, move to the arena, release keys, zero counters. Refused while dead |
| `spawn_npc {type?}` | `NPC.SpawnOnPlayer`; default 398 (Moon Lord core) |
| `spawn_projectile {type?, x?/y? or dx?/dy?, vx, vy, damage?, knockback?}` | Hostile projectile (testing); default type 462 (Phantasmal Bolt) |
| `select_slot {slot}` | Select hotbar slot 0–9 (to cast summon staffs) |
| `equip_loadout {armor, accessories, hotbar, inventory, max_stats}` | Put items on the loaded character. Each item is an internal name (`ItemID.Search`), a numeric id, or a list of alternatives; `max_stats` consumes every Life Crystal, Life Fruit and Mana Crystal and turns on the Demon Heart slot. Unknown names come back in `unknown_items` |
| `apply_buffs {names, seconds}` | `AddBuff` by `BuffID` name. Unknown names come back in `unknown_buffs` |
| `build_arena {x?, y?, half_width_tiles, rows, row_spacing_tiles, furniture_spacing_tiles, floor_item, campfire_item, lantern_item, hanger_block_item, replace_platforms}` | Planter-box rows centred on `x`/`y` (default: the player), campfires on the lowest row, heart lanterns hanging from a block set into the second row; wooden platforms in the rows are replaced. Uses each item's own `createTile`/`placeStyle`. Reports every campfire and lantern, then marks the arena |
| `clear_tombstones` | Remove every gravestone in the world |
| `quick_heal` | The game's Quick Heal (the heal assist); reports `healed` |
| `set_passthrough {enabled}` | Keyboard drives the character; the bridge only adds the whip assist (human demos) |
| `scan_nurse_house {dx0, dy0, dx1, dy1}` | Record the tiles, walls and furniture in a box around the arena mark (the hand-built Nurse house) |
| `build_nurse_house {blueprint?, ...}` | Stamp the scanned house back under the arena; knocks down earlier ones |
| `nurse_ready`, `nurse_talk`, `nurse_heal` | The Nurse protocol: keep her in her room, open her chat, buy the heal |
