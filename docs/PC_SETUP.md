# Recording the demo on a Windows PC

The fly's brain and the game share one machine: the brain takes about 20 ms of
CPU per 100 ms tick on the Mac, and the game runs alongside it. A stronger PC
gives the game headroom for higher graphics settings while recording. This
guide moves everything over, then records one fight with the brain viewer
beside the game.

What the PC needs that git doesn't carry: the arena world, the character,
the saved arena mark and Nurse house for that world, and the FlyWire
annotation table. `scripts/pack_for_pc.sh` bundles them on the Mac.

## 1. On the Mac: pack the bundle

```bash
cd ~/projects/fruitfly-terraria
bash scripts/pack_for_pc.sh                 # the Master world "The Violent Heart"
# or: bash scripts/pack_for_pc.sh "Another World Name"
```

This writes `fruitfly_pc_bundle.zip` (git-ignored). Move it to the PC with a
USB stick, a cloud drive or similar. Also push the repo so the PC can clone it.

## 2. On the PC: install the tools

- **Steam, Terraria and tModLoader**, signed in to the Steam account that owns them. Launch tModLoader once so it creates its folders, then quit.
- **Git for Windows**: https://git-scm.com
- **Python 3.12** from https://python.org. Tick *Add python.exe to PATH* in the installer.
- Access to the private repo: GitHub Desktop, or `git` with a personal access token.

## 3. On the PC: clone and run the setup script

In PowerShell:

```powershell
cd $HOME\Documents
git clone https://github.com/adriandbf/fruitfly-terraria.git
cd fruitfly-terraria
git checkout master-mode        # or main, once master-mode is merged
powershell -ExecutionPolicy Bypass -File setup_pc.ps1 -Bundle C:\path\to\fruitfly_pc_bundle.zip
```

`setup_pc.ps1`:
1. creates `.venv` and installs `requirements.txt` (numpy, numba, pandas, pyarrow, Brian 2, joblib),
2. clones the Shiu et al. brain model into `brain\Drosophila_brain_model` (about 370 MB),
3. copies the world and character into `Documents\My Games\Terraria\tModLoader\Worlds` and `Players`, and the arena mark, Nurse house and annotations into the repo,
4. links `bridge\mod\FruitFlyBridge` into tModLoader's `ModSources` (a directory junction, no admin rights needed),
5. runs the offline tests.

It's safe to run again; it skips what's already done. Add `-Force` to overwrite a world or character with the same name.

## 4. On the PC: build the mod

Start tModLoader. Go to **Workshop > Develop Mods**, find **FruitFlyBridge**, then **Build + Reload**. Check it's enabled under **Workshop > Manage Mods**.

The mod was written against tModLoader **v2026.07.3.0**. If Steam has updated tModLoader since then and the build fails, send me the error.

## 5. Game settings for the recording

- **Zoom 100%**, and keep it there. The whip assist aims assuming zoom 1x.
- Resolution: whatever you record at (e.g. 1920x1080 or 2560x1440). Use windowed or borderless so you can reach the terminal.
- Graphics: lighting *Color*, quality *High*, frame skip *Off*, background on.
- Single player, **The Violent Heart** world, the character from the bundle.

## 6. A dry run

In PowerShell, in the repo, with the game in the world:

```powershell
.venv\Scripts\python.exe demo.py
```

(`python` below means `.venv\Scripts\python.exe`.)

The first run takes 10-20 s to build the brain and lay out the viewer. Then:

1. Open **http://localhost:8765** in a browser: the brain appears, waiting for the fly.
2. The character is reset into the arena and the minions are summoned.
3. The terminal says *Start your recording, then press Enter*. Press Enter in the terminal; after a 3-second countdown the Moon Lord spawns. The game keeps receiving the fly's keys while the terminal has focus (`--keep-focus`).
4. When the fight ends, the keys are released and nothing resets. The viewer shows the result until you press Enter again.

Check that the game holds full speed during the fight. Each fight's line in `logs\moonlord\results.jsonl` has `game_speed`, which should be close to 1.0. If it drops, lower the graphics settings.

## 7. Record

OBS Studio works well:

- **Scene:** a *Game Capture* (or *Window Capture*) of tModLoader, plus a **Browser Source** at `http://localhost:8765/?bg=transparent`. Inside OBS the browser source renders the brain directly, so no browser window is needed. Size the browser source to the space you give it (a 960x1080 strip next to a 1920x1080 game on a 2880x1080 canvas works); the viewer adapts. Add `&hud=0` for the brain alone, or `&rotate=0` to stop the slow turn.
- Start recording, then press Enter in the terminal.
- If the fly loses, run `python demo.py` again. The default fly won 3 of its 10 live Master fights, so expect to need a few takes.

Every run also saves the brain's activity to `logs\moonlord\<run>_brain.jsonl.gz`. To re-record the brain alone later, at any size, from a take you liked:

```powershell
python viz\replay.py logs\moonlord\<run>_brain.jsonl.gz
```

Playback follows the live run's clock from the first fight tick (the Moon Lord appearing), so it lines up with the game video from that moment.

## Troubleshooting

- **`connection refused` on 7778:** the mod isn't loaded, or you aren't in a world.
- **"not building: no arena mark" / the character isn't put in the arena:** the world name or size doesn't match the bundled `arena_mark_*.json`. Stand on the arena floor and run `python training\moonlord\harness.py --mark-arena`.
- **No Nurse trips:** run `python training\moonlord\harness.py --build-nurse-house` once in this world.
- **Starting from a new world instead of the bundle:** make a large Master world, then run `--setup-character`, `--build-arena --here` (on open surface ground) and `--build-nurse-house` (main README, Setup step 4).
- **Viewer says "waiting for the fly":** the fight hasn't started yet, or `demo.py` was run with `--no-viz`.
