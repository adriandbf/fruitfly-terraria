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

## Status: scaffolded, UNTESTED

This was written without a working tModLoader build to compile against, so
treat every field/method name below as "probably right, unverified." Build
it and fix whatever the compiler flags — see comments in the .cs files for
the specific things most likely to have drifted (TriggersSet vs. Player
control fields, NPC/tile field names).

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
             "mana": 20, "max_mana": 20, "on_ground": true},
  "npcs": [{"type": 1, "name": "Zombie", "x": 1300.0, "y": 670.0, "hp": 250, "hostile": true}],
  "tiles": {"origin_tx": 70, "origin_ty": 40, "width": 31, "height": 17,
            "types": [-1, 0, 0, "..."], "solid": [false, true, true, "..."]}
}
```
If not currently in a world: `{"ok": true, "in_world": false}`.

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

## Next steps (not yet built)

- A scripted "dumb" policy (Python) that polls `get_state` and calls
  `set_action` at 10 Hz, per Phase 2.
- Logging every state/action pair to `logs/` (git-ignored) so runs can be
  replayed.
- An `attack`/item-use target, since `use` currently just holds the mouse
  button with whatever's in the active hotbar slot — fine for a dumb bot,
  not for "attack nearby enemies" later.
