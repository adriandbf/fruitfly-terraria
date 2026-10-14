# The Moon Lord fight: harness, training and analysis

Everything that runs, trains and scores the fly's Moon Lord fights.

## Files

| File | What it is |
| --- | --- |
| `harness.py` | Runs fights back to back and scores them: reset, minions, Moon Lord, 10 Hz loop, outcome. Also the one-off setup commands (character, arena, Nurse house) |
| `scene3.py` | The fly's **senses**: game state to firing rates of real FlyWire visual populations, per eye. Also `V3Policy`, the **teacher**, which reads those senses directly |
| `scene_fly3.py` | The **fly**: senses into the brain, 117 descending neurons out, linear readout to keys. Also collects training data, runs DAgger, fits and evaluates the readout |
| `train_senses3.py` | CMA-ES search for the teacher's weights on the offline fight |
| `fake_bridge.py` | The offline fight: a cartoon of the Moon Lord speaking the mod's protocol, fitted to live logs (`fake_fit.json` for Expert, `fake_fit_master.json` for Master) |
| `fit_master.py` | Fits the Master offline fight's attack knobs to live damage by source |
| `calibrate_master.py` | Offline Master fights for any policy, compared with live numbers |
| `policies.py` | Baselines: `idle`, `random`, `scripted` (hand-written dodger), `human` (you) |
| `movement.py` | Character movement model fitted to probes of the real game (run, flight, gravity) |
| `hit_sources.py` | What hit the fly: damage per attack from per-tick logs |
| `report.py` | Results table from `logs/moonlord/results.jsonl` |
| `loadout.json` | The character, the arena, the buffs and the Nurse protocol settings |
| `nurse_blueprint.json` | The hand-built Nurse house (scanned in game) |
| `params/` | Trained teachers (`senses3_*`) and fly readouts (`scene3_*`) |
| `test_harness.py`, `test_training.py` | Offline checks (no game, no brain) |

## A fight, tick by tick

Every 100 ms of game time: `get_state` from the mod, then the fly's senses
are computed, the brain is run for 100 ms with those sensory neurons
driven, the descending neurons' firing goes through the readout, and the
resulting keys are sent with `set_action`. Each key is credited to whoever
pressed it (`brain`, or a scripted `assist` / `leash` / `nurse`), and every
tick is logged to `logs/moonlord/<run>_fly_fight_epNNN.jsonl`.

Scripted helpers, the same for every policy and always credited separately:
- **Whip assist:** swings the whip at a Moon Lord part. The minions (the
  summoner's damage) follow the whip's tag; their aim is the game's own AI.
- **Heal assist:** the game's Quick Heal at 300 HP or below.
- **Leash:** stops the character leaving the arena box.
- **Nurse protocol** (`--nurse`): when low and potion-sick, Magic Mirror
  home, buy a heal from the Nurse, then the fly flies itself back out.

## How the fly was trained

The connectome is never changed. What is trained is the readout from
descending neurons to movement:

1. **Teacher.** `train_senses3.py` searches the weights of `V3Policy`, a
   linear policy that sees exactly the fly's senses, on hundreds of
   offline fights with randomized bosses (`--master --objective hits`
   minimizes damage taken per minute, weighted by attack).
2. **Imitation.** `scene_fly3.py collect` runs the teacher's offline fights
   through the brain and records the descending neurons' rates; `fit`
   learns a ridge-regression readout from those rates to the teacher's
   sideways and vertical drives.
3. **DAgger.** After a live batch, `scene_fly3.py dagger` relabels the
   states the fly actually visited with the teacher's choice and refits.
4. **Live check.** 10 Master fights with `harness.py --policy fly --nurse`,
   then `hit_sources.py` and `report.py`.

The control is the same pipeline on a **shuffled connectome**
(`--null shuffled`, every synapse's target permuted): only 2 of the 117
readout neurons respond, the readout can explain 13% of the teacher's
moves (92% for the real wiring), and that fly takes about twice the damage.

```bash
python training/moonlord/train_senses3.py --master --objective hits --name senses3_mine
python training/moonlord/scene_fly3.py collect --master --teacher training/moonlord/params/senses3_mine.json --fights 20
python training/moonlord/scene_fly3.py fit --data logs/moonlord/scene3_np_none_senses3_mine.npz --teacher training/moonlord/params/senses3_mine.json
python training/moonlord/scene_fly3.py eval --master --params training/moonlord/params/scene3_np_none_senses3_mine.json --fights 12
python training/moonlord/scene_fly3.py dagger logs/moonlord/<run>_fly_fight_ep*.jsonl --out <data>.npz --append
```

## The params that matter

| File | What |
| --- | --- |
| `scene3_master_ray.json` | **The default fly.** Master, eye and deathray stage: 9.8 hits/min live, 3/10 wins |
| `scene3_master_all.json` | Master, all attacks weighted: 12.6 hits/min, 2/10 wins, fights closer to the boss |
| `scene3_master_eyes.json` | Master, homing-eye stage: 11.1 hits/min, 3/10 wins |
| `scene3_np_none.json` | Expert fly: 9/10 wins with the Nurse |
| `scene3_np_shuffled.json` | The shuffled-connectome control's readout |
| `senses3_*.json` | The teachers the readouts above imitate |

## Recording your own play

`python training/moonlord/harness.py --policy human` runs a normal fight
with a 3-second countdown; you steer, and your keys are logged and credited
`human` like any other policy. `--mode free --max-seconds 60` records
movement with no boss.
