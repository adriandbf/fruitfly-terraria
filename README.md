# Fruit Fly Plays Terraria

A simulated fruit-fly brain fights Terraria's final boss, the Moon Lord, on Master mode, and can win.

The brain is the whole-brain model of *Drosophila melanogaster* from Shiu et al. (Nature, 2024): **138,639 leaky integrate-and-fire neurons and 15 million synapses**, wired exactly as in the FlyWire connectome (v783). What the fly sees in the game is delivered to real fly visual neurons, and its moves are read from real descending neurons, the cells that carry commands from a fly's brain to its body. The wiring is never changed. Only the interface around it is designed and trained: how the game becomes sensory input, and how descending-neuron firing becomes keys.

![The brain viewer: every neuron of the fly brain at its real position, lit as it fires](docs/brain_viewer.jpg)
*The brain viewer (`viz/`): all 138,639 neurons at their cell-body positions, glowing as they spike, with the fly's eyes on the left and its moves on the right.*

## Results

Moon Lord fights in the live game, a summoner in a purpose-built arena with a Nurse house underneath. 10 fights per row:

| Fly | Difficulty | Wins | Hits taken / min |
|---|---|---|---|
| Expert-trained fly | Expert, no Nurse | 0 / 10 | ~14 |
| Expert-trained fly (`scene3_np_none`) | Expert + Nurse | **9 / 10** | 14.1 |
| Expert-trained fly | **Master** + Nurse | 1 / 10 | 13.9 |
| Trained to dodge the homing eyes | Master + Nurse | 3 / 10 | 11.1 |
| + deathray-beam sense (`scene3_master_ray`, the default) | Master + Nurse | **3 / 10** | **9.8** |
| All attacks weighted (`scene3_master_all`) | Master + Nurse | 2 / 10 | 12.6 |

The same build standing still lost all 5 of its Master fights, taking about 29 hits a minute.

**Does the wiring matter?** The control brain is the same model with every synapse's target shuffled (same neurons, same senses, same readout, trained with the same pipeline). In offline Master fights its descending neurons barely respond (2 of 117 fire), the readout explains 13% of the teacher's moves instead of 92%, and it takes 2,294 damage a minute against 1,203 for the real connectome. On Expert, the shuffled brain barely moved and lost like a character standing still.

## How it works

```
game (tModLoader mod) --JSON over TCP--> senses --> fly brain, 100 ms --> 117 descending neurons --> readout --> keys
```

- **Senses** (`training/moonlord/scene3.py`). Every channel is a real FlyWire visual population, one per eye, and each codes *time to contact*, as looming detectors in a real fly do:
  LC4 (big things level with the fly), LPLC4 (approaching from above), LC6 (big things below), LC11 and LC12 (small objects, i.e. projectiles), LPLC2 (about to hit), LC22 (the sweeping deathray), and VS / HS (the fly's own motion).
- **Brain** (`brain/fast_sim.py`). A numba reimplementation of the Brian 2 model, about 20 ms of compute per 100 ms of brain time, validated against Brian 2. `brain/np_sim.py` is a pure-numpy version for offline training.
- **Readout** (`training/moonlord/scene_fly3.py`). A linear map from 117 identified descending neurons to "sideways" and "climb / dive", learned by imitating a teacher that sees the same senses, then refined on the fly's own fights (DAgger). See [training/moonlord/README.md](training/moonlord/README.md).
- **What is scripted, and credited as such in every log:** the whip swing (minions follow its tag and do the damage, as for a human summoner), healing potions, an arena leash, and trips to the Nurse. The brain moves the character on 84% of fight ticks and flies itself back out of the Nurse room.

## Watch it: one demo fight

With the game running in the arena world:

```bash
python demo.py
```

It builds the brain, opens the brain viewer at **http://localhost:8765**, puts the character in the arena and summons the minions. It then waits for you to press Enter, so you can start recording, and runs one Moon Lord fight with the best fly. When the fight ends it stops; nothing is reset. The brain's activity is also saved (`logs/moonlord/<run>_brain.jsonl.gz`), and `python viz/replay.py <file>` plays it back later.

Viewer keys: **H** hides the overlay, **R** toggles rotation, **E** toggles the wiring, drag to turn. For OBS, add it as a Browser Source at `http://localhost:8765/?bg=transparent` (`&hud=0` for the brain alone).

Recording on another machine (e.g. a stronger Windows PC): [docs/PC_SETUP.md](docs/PC_SETUP.md).

## Setup

1. **Terraria + tModLoader** (v2026.07.3.0, Terraria 1.4.4 content), and the mod in `bridge/` (link it into ModSources, **Build + Reload**; see [bridge/README.md](bridge/README.md)).
2. **Python 3.11** with `pip install -r requirements.txt`.
3. **Brain model and data** (not in this repo):
   - `git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model brain/Drosophila_brain_model` (the connectome tables)
   - the FlyWire neuron annotations in `brain/data/flywire_annotations.tsv.gz` (used for neuron identities and the viewer's positions)
   - optional, for offline training: `python brain/export_connectome_npz.py`
4. **World and character**, once, in a large Master world:
   ```bash
   python training/moonlord/harness.py --setup-character        # loadout.json gear, max stats
   python training/moonlord/harness.py --build-arena --here     # stand on open surface ground first
   python training/moonlord/harness.py --build-nurse-house      # the hand-built house, under the arena
   ```

## Running and training

```bash
python training/moonlord/harness.py --policy fly --nurse --whip-strategy balanced --episodes 10 --keep-focus
python training/moonlord/hit_sources.py "logs/moonlord/<run>_fly_fight_ep*.jsonl"
python training/moonlord/report.py
python training/moonlord/test_harness.py && python training/moonlord/test_training.py    # offline, no game
```

Other policies: `idle`, `random`, `scripted`, `human`. `--fake` runs the offline fight instead of the game (`--fake-master --engine np` works without numba). Training is described in [training/moonlord/README.md](training/moonlord/README.md).

## Repository layout

| Path | What |
|---|---|
| `demo.py` | One recorded demo fight |
| `brain/` | Brain simulators (`fast_sim.py`, `np_sim.py`), engines and controls (`brain_backend.py`), named neurons (`neurons.py`, `scene_neurons.json`), the vertical-readout sweep |
| `bridge/` | The tModLoader mod and its Python client |
| `training/moonlord/` | Fight harness, senses, the fly, training, the offline fight, analysis |
| `viz/` | The brain viewer, its live feed and replay |
| `docs/` | PC setup guide, images |

## History

1. **Speed.** Brian 2 was too slow for live play, so `fast_sim.py` reimplements the model; against Brian 2 it agrees on which neurons are active (98% overlap) and on motor-neuron spike counts.
2. **Bridge.** A client-side mod exposing game state and keys over local TCP.
3. **First goal: approach and flee.** Sugar-receptor neurons drove walking toward a player; LC4 looming neurons drove the Giant Fiber escape. Three null brains (random mapping, shuffled connectome, ablation) all scored zero.
4. **Moon Lord.** Arena, loadout and harness in the mod; an offline fight fitted to live logs; senses that grew from left/right looming to direction- and time-to-contact-resolved visual populations; the Nurse; Expert wins; then Master, cutting hits taken one attack at a time.

The code of the earlier stages is in the git history at tag `archive/pre-cleanup`.

## Credits

Brain model: Shiu, P. K. et al., *A Drosophila computational brain model reveals sensorimotor processing*, Nature (2024), https://github.com/philshiu/Drosophila_brain_model (MIT). Connectome and annotations: FlyWire (Dorkenwald et al.; Schlegel et al., Nature 2024). Terraria by Re-Logic; tModLoader by the tModLoader team.
