"""Phase 5: CMA-ES over interface/encoder.py's and interface/decoder.py's
numeric parameters, against training/arena.py's simulated approach task.

Scope: tunes the 9 numeric knobs (4 encoder, 5 decoder) -- NOT which
neurons are used. Neuron selection is Phase 4's job (brain/neurons.py) and
the ablation/random-mapping/shuffled-connectome comparisons (see
train_baselines.py) are how you'd tell whether the specific neurons matter,
separately from how well their thresholds are tuned.

Reward, per episode: 1 - mean(distance_to_human)/initial_distance, over a
fixed 2s (20-tick) episode. 1.0 = stayed essentially on top of the human
the whole time; 0.0 = made no net progress; negative = walked away net.
Evaluated over a FIXED set of episode scenarios (same for every candidate,
generated once with a fixed seed) so CMA-ES compares candidates fairly --
letting the episode set vary per-candidate would make "who got lucky"
indistinguishable from "who's actually better".

Usage:
    python training/train_cma.py [--generations 15] [--popsize 8]
        [--episodes 6] [--steps 20] [--seed 0]
Writes training/best_params.json at the end.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brain"))
sys.path.insert(0, str(ROOT / "brain" / "Drosophila_brain_model"))
sys.path.insert(0, str(ROOT / "interface"))
sys.path.insert(0, str(ROOT / "training"))

import random
import numpy as np
import pandas as pd
import cma

from model import default_params as brain_params
from fast_sim import FastBrain
import encoder
import decoder
import neurons
from arena import Arena

DT_MS = 1.0

# (module, key, lower, upper) -- CMA-ES searches in normalized [0,1]^9
# space; these bounds map a unit value back to the real parameter.
PARAM_SPEC = [
    ("encoder", "max_rate_hz", 50.0, 300.0),
    ("encoder", "approach_max_range_px", 300.0, 4000.0),
    ("encoder", "threat_max_range_px", 100.0, 2000.0),
    ("encoder", "contra_fraction", 0.0, 1.0),
    ("decoder", "forward_threshold_hz", 0.0, 100.0),
    ("decoder", "backward_threshold_hz", 0.0, 100.0),
    ("decoder", "steer_margin_hz", 0.0, 50.0),
    ("decoder", "jump_threshold_hz", 0.0, 100.0),
    ("decoder", "use_threshold_hz", 0.0, 100.0),
]


def unit_to_params(unit_vec):
    """[0,1]^9 -> ({encoder params}, {decoder params}), clipped to bounds."""
    enc, dec = {}, {}
    for (module, key, lo, hi), u in zip(PARAM_SPEC, unit_vec):
        u = min(1.0, max(0.0, u))
        value = lo + u * (hi - lo)
        (enc if module == "encoder" else dec)[key] = value
    return enc, dec


def defaults_as_unit():
    """The current hand-picked defaults, expressed in the same normalized
    space -- used as CMA-ES's starting point and as the "before tuning"
    baseline to compare against."""
    unit = []
    for module, key, lo, hi in PARAM_SPEC:
        default = (encoder.DEFAULT_PARAMS if module == "encoder" else decoder.DEFAULT_PARAMS)[key]
        unit.append((default - lo) / (hi - lo))
    return unit


def build_brain():
    repo = ROOT / "brain" / "Drosophila_brain_model"
    df_comp = pd.read_csv(repo / "Completeness_783.csv", index_col=0)
    df_con = pd.read_parquet(repo / "Connectivity_783.parquet")
    return FastBrain(df_comp, df_con, brain_params, DT_MS, exc_indices=[], seed=0)


def make_episode_scenarios(n_episodes, seed):
    """Fixed, diverse set of (human_side, hostile_present, episode_seed)
    tuples, reused for every candidate this whole run."""
    rng = random.Random(seed)
    scenarios = []
    for i in range(n_episodes):
        side = "right" if i % 2 == 0 else "left"  # exactly balanced
        hostile = rng.random() < 0.5
        scenarios.append((side, hostile, rng.randint(0, 2**31 - 1)))
    return scenarios


def run_episode(brain, enc_params, dec_params, human_side, hostile_present, seed, n_steps):
    """One fixed-length episode; returns the normalized reward."""
    arena = Arena(random.Random(seed))
    state = arena.reset(human_side=human_side, hostile_present=hostile_present)
    initial_dist = arena.distance_to_human()
    brain.reset_state(seed=seed)

    total_dist = 0.0
    for _ in range(n_steps):
        neuron_ids, rates_hz = encoder.encode(state, enc_params)
        spike_rates = brain.stimulate(neuron_ids, rates_hz, duration_ms=100)
        readout = {fid: brain.rate_for(spike_rates, fid) for fid in neurons.ALL_DESCENDING}
        action = decoder.decode(readout, state["player"]["direction"], dec_params)
        state = arena.step(action)
        total_dist += arena.distance_to_human()

    mean_dist = total_dist / n_steps
    return 1.0 - mean_dist / initial_dist


def evaluate(brain, unit_vec, scenarios, n_steps):
    enc_params, dec_params = unit_to_params(unit_vec)
    rewards = [run_episode(brain, enc_params, dec_params, side, hostile, seed, n_steps)
               for side, hostile, seed in scenarios]
    return sum(rewards) / len(rewards)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", type=int, default=15)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Building brain (v783)...")
    brain = build_brain()
    brain.stimulate([], [], duration_ms=10)  # JIT warm-up

    scenarios = make_episode_scenarios(args.episodes, args.seed)
    print(f"Fixed episode set ({len(scenarios)}): "
          f"{[(s, h) for s, h, _ in scenarios]}")

    x0 = defaults_as_unit()
    baseline_reward = evaluate(brain, x0, scenarios, args.steps)
    print(f"\nBaseline (current hand-picked defaults) mean reward: {baseline_reward:.3f}\n")

    es = cma.CMAEvolutionStrategy(
        x0, 0.25,
        {"popsize": args.popsize, "bounds": [0.0, 1.0], "seed": args.seed, "verbose": -9},
    )

    t0 = time.time()
    gen = 0
    best_reward, best_unit = baseline_reward, x0
    while not es.stop() and gen < args.generations:
        candidates = es.ask()
        fitnesses = [-evaluate(brain, c, scenarios, args.steps) for c in candidates]  # cma minimizes
        es.tell(candidates, fitnesses)
        gen_best_reward = -min(fitnesses)
        gen_mean_reward = -sum(fitnesses) / len(fitnesses)
        if gen_best_reward > best_reward:
            best_reward = gen_best_reward
            best_unit = candidates[fitnesses.index(min(fitnesses))]
        print(f"gen {gen:3d}  best={gen_best_reward:+.3f}  mean={gen_mean_reward:+.3f}  "
              f"(best so far: {best_reward:+.3f})  [{time.time()-t0:.0f}s elapsed]")
        gen += 1

    enc_params, dec_params = unit_to_params(best_unit)
    print(f"\n=== done in {time.time()-t0:.0f}s ===")
    print(f"baseline reward: {baseline_reward:+.3f}")
    print(f"best reward:     {best_reward:+.3f}")
    print(f"encoder params: {json.dumps(enc_params, indent=2)}")
    print(f"decoder params: {json.dumps(dec_params, indent=2)}")

    # Break down by side, to quantify the known left/right asymmetry
    # (brain/neurons.py's docstring) rather than let it hide in an average.
    for side in ["right", "left"]:
        side_scenarios = [(s, h, seed) for s, h, seed in scenarios if s == side]
        if side_scenarios:
            r = evaluate(brain, best_unit, side_scenarios, args.steps)
            print(f"  reward with human on the {side}: {r:+.3f}")

    out_path = ROOT / "training" / "best_params.json"
    out_path.write_text(json.dumps({
        "encoder": enc_params, "decoder": dec_params,
        "baseline_reward": baseline_reward, "best_reward": best_reward,
        "generations": gen, "popsize": args.popsize,
        "episodes": args.episodes, "steps": args.steps, "seed": args.seed,
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
