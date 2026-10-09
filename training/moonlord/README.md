# Goal 2: Moon Lord harness (stage M0)

Runs Moon Lord fights, or dodge-gym drills, back to back with nobody at
the keyboard. It scores every episode and records which source pressed
each key: the fly, a script, or a random baseline.

| File | What it is |
| --- | --- |
| `harness.py` | Episode runner and CLI: reset, summon minions, spawn the boss, 10 Hz loop, score |
| `policies.py` | `idle`, `random`, `scripted` (hand-written dodger, the ceiling), `fly` (encoder → brain → decoder) |
| `fake_bridge.py` | Offline cartoon of the fight, speaking the mod's protocol (tests now, offline dodge gym later) |
| `test_harness.py` | Offline checks, no game and no brain needed: `python training/moonlord/test_harness.py` |

The mod side is `bridge/mod/FruitFlyBridge/Harness.cs`, plus additions to
`GameState.cs`, `BridgeSystem.cs` and `BridgePlayer.cs`. The protocol is
in `bridge/README.md`.

## Status

- Offline: all 15 checks in `test_harness.py` pass.
- Not compiled yet. No .NET SDK was available where this was written. Hook
  signatures were checked against the tModLoader v2026.07.3.0 source, and
  member names against `tModLoader.dll`. The first **Build + Reload** is the
  real compile, so expect to fix a typo or two.
- Not tried live yet: every new command, `--keep-focus`, summoning
  minions, and the fly against real projectiles.

## First live session (on the Mac)

1. **Build the mod.** tModLoader → Workshop → Mod Sources → FruitFlyBridge →
   **Build + Reload**.
2. **Character and world. No downloads needed.** In tModLoader, create a
   new character (Classic, softcore) and a new small world. Make the world
   Expert if you can: Expert unlocks the sixth accessory slot (Demon
   Heart), and raising the difficulty later means a new world anyway. Enter
   the world, then run:
   `python training/moonlord/harness.py --setup-character`
   This applies `loadout.json`, a pre-Moon Lord summoner (tModLoader is on
   Terraria 1.4.4, so this sticks to 1.4.4 items):
   - Hallowed armour (summoner hood).
   - Empress Wings and the Soaring Insignia.
   - Papyrus Scarab, Necromantic Scroll, Pygmy Necklace, Ankh Shield (and
     a Worm Scarf in Master).
   - Stardust Dragon Staff (slot 0, fills every minion slot), Terraprisma
     (slot 1, spare) and the Kaleidoscope whip (slot 2).
   - Maxed life and mana.

   If the mod reports `unknown_items`, fix that entry in `loadout.json` and
   rerun. Each entry can list alternatives, e.g. `["RainbowWhip", 4914]`.
   Save and exit once so the gear sticks to the character.
3. **Arena.** Stand on a flat open spot on the surface, then run:
   `python training/moonlord/harness.py --build-arena`
   If an arena mark already exists, it builds there instead and swaps any
   old wooden-platform rows for planter boxes. It places:
   - 16 rows of 800-tile Daybloom Planter Boxes, 12 tiles apart, so the
     arena is about 800 tiles wide and 192 tall.
   - Campfires every 60 tiles on rows 1, 6, 11 and 16.
   - Heart lanterns every 60 tiles, hanging from Gray Bricks set into rows
     3, 8 and 13.

   Their buff range is about 170×125 tiles, so the whole arena is covered.
   All of these sizes are settings in `loadout.json`. Near the ground the
   rows only fill empty tiles, so hills and trees leave gaps.
   The output lists every campfire and lantern with `ok: true/false`. If a
   campfire won't stand on planter boxes, it gets a 3-brick pad and a
   retry. The arena mark is lost when the game restarts: stand on the spot
   and run `--mark-arena` again.
4. **Smoke test.** Run one idle fight with minions:
   `python training/moonlord/harness.py --policy idle` (summon slots and buffs come from `loadout.json`)
   Check that the boss spawns, minions get cast, the episode ends on death
   or a kill, and a line is added to `logs/moonlord/results.jsonl`.
5. **Focus test.** Run `--policy random --keep-focus`, then click another
   app. If the character keeps moving, focus is fixed. If it freezes or the
   harness reports "game thread did not run the command", focus is not
   fixed; note what happened.
6. **Calibrate (M1).** Run 5–10 fights each of `idle` and `scripted`.
   Raise the difficulty if idle ever wins.

Then: `--policy fly`, and the dodge gym with `--mode gym` (bolts by
default; spheres and eyes may misbehave when spawned without the boss).

## Live results so far

- 3 idle fights (2026-10-08): all lost, at 14–18 s. Minions took Moon Lord
  from 217.5k HP to about 190k (roughly 1,660 HP/s). Each hit did 66–131
  damage after defense. At that rate a full kill would take about 2.2 min
  of survival.
- The whip assist held `use` the whole time, but no swing happened.
  Possible causes: Terraria's use-delay gate never cleared because `use`
  was held, and the aim was being overwritten by input zoom handling. Both
  are patched (`BridgePlayer.Apply`, `HarnessSystem.PreUpdatePlayers`), but
  the patch is unconfirmed. `get_state.debug` now reports the item-use
  state (`whip_projectiles`, `item_animation`, `delay_use_item`...).
  `report.py` checks those fields in the logs.
- `fake_bridge.py` has been recalibrated to these fights. Offline, idle now
  dies at about 15 s (live: 14–18 s) and the scripted dodger wins about
  2 fights in 6.

- Batch 1 (2026-10-08, 5 fights each): idle 0/9 wins (median 15.3 s),
  random 0/6 (19.0 s), fly 0/5 (17.4 s, with 90% of boss HP left).
  The whip swung on about 96% of ticks.
- Bugs found and fixed:
  - The scripted dodger crashed on its first tick whenever an arena mark
    existed. It has now been replayed on all 5,146 logged states without
    an error.
  - Setup stopped casting at 6 of 9 minions. It now keeps casting until
    the slots are full.
  - Random keys flew 176 tiles up to the world ceiling. An arena leash
    (a scripted assist, applied the same way to every policy and counted
    as `leash` in the ledger) now blocks up/jump at the top row and
    left/right at the side edges.
- Terraprisma is now the summon cast at setup (slot 1).
- In single player the game pauses when its window is unfocused, so
  `--keep-focus` now also forces FNA's `Game.IsActive` back on. Untested.

## Recording your own play

`python training/moonlord/harness.py --policy human` starts a normal fight:
reset, potions, minions and buffs, then a 3-second countdown, then the
Moon Lord. You steer with your own keyboard; the mod is in passthrough mode
and only adds the whip assist and heal assist. Every tick is logged with
the keys you held, in the same format as every other policy and credited
`human`, so `report.py` shows you as a baseline. Useful flags:

- `--no-whip-assist`: swing the whip yourself.
- `--no-heal`: drink potions yourself (H).
- `--mode free --max-seconds 60`: no boss, just record a movement
  technique.
- `--episodes 3`: record several attempts in a row.

## Next live batch

`bash training/moonlord/run_batch.sh` runs, unattended:
1. 5 fights each of idle, random, scripted and fly.
2. A `--keep-focus` test.
3. `report.py` at the end.

Keep the Terraria window in front, except during the last fight (the focus
test). `N=10 bash training/moonlord/run_batch.sh` runs 10 fights per policy.

## Known limits

- **Whip assist:** with `whip_assist` on (the default in `loadout.json`), a
  script holds the whip aimed at the nearest Moon Lord part that can take
  damage. The ledger credits it as `assist`, never as `brain`. Turn it off
  with `--no-whip-assist`. Aiming assumes the game zoom is 1×. The ledger
  counts ticks per source, so with the assist on the totals add up to more
  than the episode's ticks.

- The fly's projectile sense is a stopgap: incoming projectiles are shown
  to the existing LC4 channel as if they were hostile NPCs. Replacing it
  with a proper encoding is stage M2.
- The fly has no up/down yet (stage M3) and doesn't press `use`. Minions
  do the damage.
- `fake_bridge.py` numbers are guesses. Refit them from live logs before
  tuning against it.
