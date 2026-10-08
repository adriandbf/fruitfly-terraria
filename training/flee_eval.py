"""Brain-in-the-loop check of the flee reflex against a hostile that CHASES.

training/arena.py's hostile is inert, so it can't test fleeing. This adds a
1D chaser and compares, with the REAL encoder + FastBrain + decoder in the
loop:
  legacy: encoder with fear-suppresses-appetite OFF + the stateless decode()
  new:    encoder default (suppression ON) + decoder.StatefulDecoder
across scenarios with a hostile alone and with a human present on the same
or the opposite side, and reports how often the character is caught.

Simplifications, stated so the numbers aren't over-read:
  - 1D, flat, no jumping/terrain/knockback/damage: "caught" just means the
    hostile got within CONTACT_PX of the character.
  - HOSTILE_SPEED is an ASSUMPTION (zombie/slime speed from memory, not
    measured in this game). The player's 150px/s is the arena's own figure.
    --speeds lets you sweep it; 140 is nearly the player's own speed.
  - It can't say anything about a SWARM (the live fight that killed the
    character had 6 zombies), only about one chaser.

Usage: python training/flee_eval.py [--speeds 80] [--seeds 3] [--steps 120]
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brain"))
sys.path.insert(0, str(ROOT / "brain" / "Drosophila_brain_model"))
sys.path.insert(0, str(ROOT / "interface"))
sys.path.insert(0, str(ROOT / "training"))

import pandas as pd
from model import default_params as brain_params
from fast_sim import FastBrain
import neurons
import encoder
import decoder
from arena import Arena, WALK_SPEED_PX_PER_S, TICK_S

CONTACT_PX = 24.0
START_DIST_PX = 220.0
HUMAN_DIST_PX = 300.0


class ChaseArena(Arena):
    def __init__(self, rng, hostile_speed):
        super().__init__(rng)
        self.hostile_speed = hostile_speed

    def setup(self, hostile_side, human_side):
        self.player_x, self.player_direction = 0.0, 1
        self.hostile_x = START_DIST_PX if hostile_side == "right" else -START_DIST_PX
        self.hostile_hp = 25.0
        self.human_x = None if human_side is None else (HUMAN_DIST_PX if human_side == "right" else -HUMAN_DIST_PX)
        return self.get_state()

    def get_state(self):
        st = super().get_state()
        if self.human_x is None:
            st["players"] = []
        return st

    def step(self, action):
        super().step(action)
        gap = self.player_x - self.hostile_x
        move = min(abs(gap), self.hostile_speed * TICK_S)
        self.hostile_x += move if gap > 0 else -move
        return self.get_state()

    def gap(self):
        return abs(self.player_x - self.hostile_x)


def run_episode(brain, arena, policy, hostile_side, human_side, seed, steps):
    brain.reset_state(seed=seed)
    state = arena.setup(hostile_side, human_side)
    enc_params = {"approach_suppression_hz": 0.0} if policy == "legacy" else None
    stateful = decoder.StatefulDecoder() if policy == "new" else None
    first_contact, contact_ticks = None, 0
    for t in range(steps):
        ids, rates_hz = encoder.encode(state, enc_params)
        r = brain.stimulate(ids, rates_hz, duration_ms=100)
        readout = {f: brain.rate_for(r, f) for f in neurons.ALL_DESCENDING}
        if stateful:
            action = stateful.decode(readout, state["player"]["direction"])
        else:
            action = decoder.decode(readout, state["player"]["direction"])
        state = arena.step(action)
        if arena.gap() < CONTACT_PX:
            contact_ticks += 1
            if first_contact is None:
                first_contact = t
    return first_contact, contact_ticks, arena.gap()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speeds", type=float, nargs="+", default=[80.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=120)
    args = ap.parse_args()

    repo = ROOT / "brain" / "Drosophila_brain_model"
    df_comp = pd.read_csv(repo / "Completeness_783.csv", index_col=0)
    df_con = pd.read_parquet(repo / "Connectivity_783.parquet")
    brain = FastBrain(df_comp, df_con, brain_params, 1.0, exc_indices=[], seed=0)
    brain.stimulate([], [], duration_ms=10)

    scenarios = [
        ("hostile alone, from the left", "left", None),
        ("hostile alone, from the right", "right", None),
        ("hostile left, human right (opposite sides)", "left", "right"),
        ("hostile right, human left (opposite sides)", "right", "left"),
        ("hostile right, human right (same side)", "right", "right"),
        ("hostile left, human left (same side)", "left", "left"),
    ]
    secs = args.steps * TICK_S
    for speed in args.speeds:
        print(f"\n===== hostile speed {speed:.0f}px/s (player {WALK_SPEED_PX_PER_S:.0f}px/s), "
              f"{secs:.0f}s episodes, {args.seeds} seeds =====")
        print(f"{'scenario':<46}{'policy':<8}{'never caught':>13}{'mean first contact':>20}{'final gap px':>14}")
        totals = {"legacy": [0, 0], "new": [0, 0]}
        for name, hs, hu in scenarios:
            for policy in ("legacy", "new"):
                results = []
                for seed in range(args.seeds):
                    arena = ChaseArena(random.Random(seed), speed)
                    results.append(run_episode(brain, arena, policy, hs, hu, seed, args.steps))
                safe = sum(1 for fc, _, _ in results if fc is None)
                times = [(fc if fc is not None else args.steps) * TICK_S for fc, _, _ in results]
                gap = sum(g for _, _, g in results) / len(results)
                totals[policy][0] += safe
                totals[policy][1] += len(results)
                print(f"{name:<46}{policy:<8}{f'{safe}/{len(results)}':>13}{sum(times)/len(times):>19.1f}s{gap:>14.0f}")
        print(f"\noverall never-caught:  legacy {totals['legacy'][0]}/{totals['legacy'][1]}   "
              f"new {totals['new'][0]}/{totals['new'][1]}")


if __name__ == "__main__":
    main()
