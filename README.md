# Fruit Fly Plays Terraria

A simulated fruit-fly brain plays Terraria. The brain is the whole-brain leaky integrate-and-fire model of *Drosophila* (Shiu et al.), wired from the FlyWire connectome (v783, ~127k neurons, ~15M synapses). Game state goes in through real fly sensory neurons; key presses come out of real descending neurons. The connectome's wiring is never changed: only the interface (what the fly sees, how its neuron firing becomes keys) is designed and trained.

**Current goal: the fly fights the Moon Lord**, the final boss, in a single-player arena, fight after fight, getting better between batches.

## Results so far (Moon Lord, live game)

| Setup | Fly wins | Median fight | Hits taken / min |
|---|---|---|---|
| Expert, no Nurse | 0 / 10 | 38 s | ~14 |
| Normal, no Nurse | 4 / 10 | 78 s | ~16 |
| Expert + Nurse (hand-built house) | **9 / 10** | 130 s | 14.1 |
| **Master** + Nurse (hand-built house) | 1 / 10 | 59 s | 13.9 |
| **Master** + Nurse, eye-dodging fly | 3 / 10 | 131 s | 11.1 |
| **Master** + Nurse, eye + beam-sense fly (scene3_master_ray) | **3 / 10** | 99 s | **9.8** |

Same setup, other controllers: scripted dodger 3/5 wins (15.1 hits/min); standing still 1/5 (31.9 hits/min). A control brain with a **shuffled connectome** and the same readout neurons barely moves and loses like standing still, so the dodging depends on the real wiring.

## Next major steps
1. **Master mode**, the hardest difficulty (same arena, Nurse house and loadout; Master unlocks the 7th accessory slot the loadout already uses).
2. **Minimise the hits the fly takes** each fight, and with them the Nurse trips. Work through the Moon Lord's attacks one at a time, most damaging first. First target done: homing eyes (22% of the fly's damage on Master) -- retrained to dodge them, eye damage per minute fell 67% live. New Master ranking: deathray 18% (biggest single hits), bolts 14%, tongue 13%, spheres 11%, body contact 27% across all parts, homing eyes 8%.
3. **Give the fly more autonomy.** Hand every scripted decision to the fly's brain, one at a time, keeping scripts only as fallbacks. Today the brain moves the character on 84% of fight ticks and pilots its own climb out of the Nurse room (25/25 trips on Master, ~0.4 s each). Still scripted: the Nurse trips (15.5% of ticks), the arena leash (edits 12.8% of the fly's ticks), the whip (swing, aim, target), healing potions. Minions stay with the game's AI, as for a human summoner.

## How the fly works
```
game state --(mod: JSON over TCP)--> senses --> fly brain (100 ms) --> descending neurons --> keys --> game
```
- **Senses** (`training/moonlord/scene3.py`): what is approaching, from which side and how soon, as firing of real FlyWire visual populations per eye: LC4 / LPLC4 / LC6 (big things level / above / below), LC11 / LC12 (small objects), LPLC2 (about to hit), LC22 (the deathray beam), VS / HS (self-motion).
- **Brain**: `brain/fast_sim.py` (numba, on the Mac, ~20 ms per 100 ms slice) or `brain/np_sim.py` (numpy, anywhere). Same equations as the Brian 2 model, validated against it.
- **Readout** (`training/moonlord/scene_fly3.py`): a linear map from 117 identified descending neurons to left/right and climb/dive, trained to imitate a teacher that reads the same senses (`train_senses3.py`), refined on the fly's own fights (DAgger).
- **Scripted helpers**, credited separately in every log (`brain` / `assist` / `leash` / `nurse`): whip swings, potion heals, the arena leash, the Nurse trip.

## Setup
- Terraria + tModLoader (v2026.07.3.0, Terraria 1.4.4 content) on the Mac; Python venv with `brain/requirements.lock.txt`.
- The brain model and connectome data are not in this repo: clone https://github.com/philshiu/Drosophila_brain_model into `brain/Drosophila_brain_model/` (`setup_phase0.sh`); FlyWire annotations go in `brain/data/`.
- Mod: symlink `bridge/mod/FruitFlyBridge` into tModLoader's `ModSources` and Build + Reload (`bridge/README.md`).

## Running a Moon Lord batch
In a **large** world, in the game, with the mod enabled:
```
python training/moonlord/harness.py --build-arena --here        # stand on open surface ground first
python training/moonlord/harness.py --setup-character
python training/moonlord/harness.py --build-nurse-house          # the hand-designed house (nurse_blueprint.json)
python training/moonlord/harness.py --policy fly --params-file training/moonlord/params/scene3_np_none.json --nurse --whip-strategy balanced --episodes 10 --keep-focus
python training/moonlord/hit_sources.py "logs/moonlord/<run>_fly_fight_ep*.jsonl"
```
Other policies: `idle`, `random`, `scripted`, `human` (record your own play). Designing the Nurse house: `--builder-kit` (pickaxe, hamaxe, blocks), build it under the arena, then `--scan-nurse-house`. Offline checks: `training/moonlord/test_harness.py`, `test_training.py` (no game needed; `fake_bridge.py` is an offline cartoon of the fight, calibrated to live logs).

## Repository layout
| Path | What |
|---|---|
| `brain/` | brain simulators, neuron picks (`neurons.py`, `scene_neurons.json`), connectome sweeps |
| `bridge/mod/FruitFlyBridge/` | the tModLoader mod: game state, key control, fight harness, arena, Nurse house |
| `interface/` | encoder / decoder / live loop from the first goal (approach and flee) |
| `training/moonlord/` | Moon Lord harness, policies, senses, training, offline fight, analysis |
| `training/` | Phase 5 tuning and baselines (first goal) |
| `server/` | dedicated server notes (first goal) |

## History
1. **Environment and brain speed**: the model ran too slowly in Brian 2 for live play, so `fast_sim.py` reimplements it (validated against Brian 2: 98% active-neuron overlap, motor-neuron spike counts matched).
2. **Bridge**: a client-side mod exposing game state and keys over local TCP.
3. **Approach and flee** (first goal): sugar receptor neurons drive walking toward a player; bitter receptors turned out to be a dead end for threats, while LC4 looming neurons drive the Giant Fiber escape. Phase 5 tuning plus three null baselines (random mapping, shuffled connectome, ablation) all scored zero, so the chosen neurons do real work.
4. **Moon Lord** (current): arena, loadout and harness in the mod; offline fight calibrated to live logs; the fly's senses grew from left/right looming to direction- and time-to-contact-resolved visual populations; Nurse healing protocol; Expert wins with the Nurse.

## Credits
Brain model: Shiu, P. K. et al., *A Drosophila computational brain model reveals sensorimotor processing* (Nature, 2024), https://github.com/philshiu/Drosophila_brain_model (MIT). Connectome and annotations: FlyWire (Dorkenwald et al., Schlegel et al., Nature 2024).
