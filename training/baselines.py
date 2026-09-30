"""Phase 5: the three required baselines (CLAUDE.md), comparing the
claimed sugar/water GRN -> DNa02/DNb05 approach pathway against:
  (a) random mapping -- same population sizes, randomly sampled neurons
  (b) degree-preserving shuffled connectome -- same neurons, randomized wiring
  (c) ablation -- the claimed sensory neurons' outgoing synapses zeroed

All three reuse train_cma.py's evaluate()/run_episode()/Arena machinery
unchanged, over the SAME fixed episode set, so every condition is compared
fairly. Evaluated at both the hand-picked defaults and today's CMA-ES-
tuned parameters (training/best_params.json) -- NOT a from-scratch CMA-ES
run per baseline (that would be ~3x the ~9.5min original run); a baseline
tuned for itself might do better than shown here. That's an honest gap,
flagged rather than hidden -- rerun with --tune-each if you want the fuller
(much slower) version.

Design notes on what each baseline actually does, and why:
  (a) Random mapping samples uniformly from the WHOLE connectome (no
      category constraint), matching each role's exact population size
      (e.g. 67 "sugar_left"-equivalent neurons, chosen at random). Applied
      by temporarily monkey-patching brain/neurons.py's module attributes
      that encoder.py/decoder.py read -- see neuron_set_override().
  (b) Shuffled connectome: FastBrain's CSR structure keeps out-degree
      exact (indptr unchanged) by construction; post_idx (which neuron
      each edge targets) is globally permuted, keeping post_w (weights)
      in their original per-edge slots. This exactly preserves each
      neuron's out-degree and the overall weight distribution, and
      approximately (not exactly) preserves the in-degree distribution --
      a standard, efficient configuration-model-style null model, not a
      literal double-edge-swap (which would be far too slow on 15M edges
      for this session).
  (c) Ablation zeroes the claimed SENSORY neurons' (sugar/water + bitter
      GRN) outgoing synapse weights -- the standard "silencing" operation,
      matching model.py's own silence() utility. This is the meaningful
      ablation for THIS pipeline: decoder.py reads DNa02/DNb05's own spike
      rate directly, so silencing THEIR outgoing synapses would trivially
      do nothing (their own rate is unaffected by what they can no longer
      influence); silencing the sensory neurons that drive them tests
      whether they're actually a necessary conduit, not a bystander next
      to some other coincidental source of activity. Poisson stimulation
      bypasses synapses entirely (a direct v-kick, per fast_sim.py), so an
      ablated GRN still "fires" locally -- it just can no longer excite
      anything downstream, which is the actual test.

Usage: python training/baselines.py [--episodes 6] [--steps 20] [--seed 0]
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brain"))
sys.path.insert(0, str(ROOT / "brain" / "Drosophila_brain_model"))
sys.path.insert(0, str(ROOT / "interface"))
sys.path.insert(0, str(ROOT / "training"))

import numpy as np
import pandas as pd

from model import default_params as brain_params
from fast_sim import FastBrain
import neurons
from train_cma import (DT_MS, make_episode_scenarios, evaluate, defaults_as_unit,
                        unit_to_params, PARAM_SPEC)

NEURON_SET_ATTRS = [
    "SUGAR_WATER_GRN_LEFT", "SUGAR_WATER_GRN_RIGHT",
    "BITTER_GRN_LEFT", "BITTER_GRN_RIGHT",
    "DNA02_STEERING_LEFT", "DNA02_STEERING_RIGHT",
    "MDN_BACKWARD", "DNP01_JUMP", "DNB05_MOVE",
]


@contextlib.contextmanager
def neuron_set_override(mapping):
    """Temporarily monkey-patch brain/neurons.py's module attributes that
    encoder.py/decoder.py actually read, restoring the real ones on exit.
    Also patches ALL_DESCENDING (decoder.py's "use" ensemble fallback) to
    the union of the substituted descending-side lists, so it stays
    consistent with whatever neuron set is active."""
    originals = {attr: getattr(neurons, attr) for attr in NEURON_SET_ATTRS}
    original_all = neurons.ALL_DESCENDING
    try:
        for attr, ids in mapping.items():
            setattr(neurons, attr, ids)
        neurons.ALL_DESCENDING = (
            neurons.DNA02_STEERING_LEFT + neurons.DNA02_STEERING_RIGHT
            + neurons.MDN_BACKWARD + neurons.DNP01_JUMP + neurons.DNB05_MOVE
        )
        yield
    finally:
        for attr, ids in originals.items():
            setattr(neurons, attr, ids)
        neurons.ALL_DESCENDING = original_all


def random_neuron_mapping(all_flyids, rng):
    """Same population size per role as the real, claimed selection --
    just randomly sampled from the whole connectome, no category
    constraint."""
    sizes = {attr: len(getattr(neurons, attr)) for attr in NEURON_SET_ATTRS}
    total = sum(sizes.values())
    sample = rng.choice(all_flyids, size=total, replace=False)
    mapping, i = {}, 0
    for attr, n in sizes.items():
        mapping[attr] = [int(x) for x in sample[i:i + n]]
        i += n
    return mapping


def build_brain(df_comp, df_con):
    return FastBrain(df_comp, df_con, brain_params, DT_MS, exc_indices=[], seed=0)


def build_shuffled_brain(df_comp, df_con, seed):
    """Same neurons, same out-degree per neuron (indptr untouched) and
    weight distribution (post_w untouched, same per-edge slots) -- only
    WHICH neuron each edge targets is randomized. See module docstring for
    why this is an approximation, not an exact double-edge-swap."""
    brain = build_brain(df_comp, df_con)
    rng = np.random.default_rng(seed)
    brain.post_idx = rng.permutation(brain.post_idx)
    return brain


def build_ablated_brain(df_comp, df_con, flyid2idx_source_ids):
    """Real connectome, but the given presynaptic FlyWire IDs' outgoing
    synapse weights are zeroed (model.py's own silence() convention)."""
    brain = build_brain(df_comp, df_con)
    for fid in flyid2idx_source_ids:
        idx = brain.flyid2idx.get(int(fid))
        if idx is not None:
            brain.post_w[brain.indptr[idx]:brain.indptr[idx + 1]] = 0.0
    return brain


def eval_both_param_sets(brain, scenarios, steps, default_unit, tuned_unit):
    default_reward = evaluate(brain, default_unit, scenarios, steps)
    tuned_reward = evaluate(brain, tuned_unit, scenarios, steps)
    return default_reward, tuned_reward


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    repo = ROOT / "brain" / "Drosophila_brain_model"
    df_comp = pd.read_csv(repo / "Completeness_783.csv", index_col=0)
    df_con = pd.read_parquet(repo / "Connectivity_783.parquet")
    all_flyids = df_comp.index.to_numpy()

    scenarios = make_episode_scenarios(args.episodes, args.seed)
    default_unit = defaults_as_unit()
    tuned_path = ROOT / "training" / "best_params.json"
    if tuned_path.exists():
        tuned_data = json.loads(tuned_path.read_text())
        tuned_unit = []
        for module, key, lo, hi in PARAM_SPEC:
            v = (tuned_data["encoder"] if module == "encoder" else tuned_data["decoder"])[key]
            tuned_unit.append((v - lo) / (hi - lo))
    else:
        print("No training/best_params.json found -- run train_cma.py first. Using defaults for 'tuned' too.")
        tuned_unit = default_unit

    results = {}

    print("=== claimed (real neurons, real connectome) ===")
    brain = build_brain(df_comp, df_con)
    brain.stimulate([], [], duration_ms=10)
    results["claimed"] = eval_both_param_sets(brain, scenarios, args.steps, default_unit, tuned_unit)
    print(f"  default={results['claimed'][0]:+.3f}  tuned={results['claimed'][1]:+.3f}")

    print("\n=== (a) random mapping (same sizes, random neurons) ===")
    rng = np.random.default_rng(args.seed)
    mapping = random_neuron_mapping(all_flyids, rng)
    with neuron_set_override(mapping):
        brain_random = build_brain(df_comp, df_con)
        brain_random.stimulate([], [], duration_ms=10)
        results["random"] = eval_both_param_sets(brain_random, scenarios, args.steps, default_unit, tuned_unit)
    print(f"  default={results['random'][0]:+.3f}  tuned={results['random'][1]:+.3f}")

    print("\n=== (b) degree-preserving shuffled connectome ===")
    brain_shuffled = build_shuffled_brain(df_comp, df_con, args.seed)
    brain_shuffled.stimulate([], [], duration_ms=10)
    results["shuffled"] = eval_both_param_sets(brain_shuffled, scenarios, args.steps, default_unit, tuned_unit)
    print(f"  default={results['shuffled'][0]:+.3f}  tuned={results['shuffled'][1]:+.3f}")

    print("\n=== (c) ablation (claimed sensory neurons' outgoing synapses zeroed) ===")
    ablated_ids = (neurons.SUGAR_WATER_GRN_LEFT + neurons.SUGAR_WATER_GRN_RIGHT
                   + neurons.BITTER_GRN_LEFT + neurons.BITTER_GRN_RIGHT)
    brain_ablated = build_ablated_brain(df_comp, df_con, ablated_ids)
    brain_ablated.stimulate([], [], duration_ms=10)
    results["ablated"] = eval_both_param_sets(brain_ablated, scenarios, args.steps, default_unit, tuned_unit)
    print(f"  default={results['ablated'][0]:+.3f}  tuned={results['ablated'][1]:+.3f}")

    print("\n=== summary ===")
    print(f"{'condition':<12}  {'default':>8}  {'tuned':>8}")
    for name, (d, t) in results.items():
        print(f"{name:<12}  {d:+8.3f}  {t:+8.3f}")

    out_path = ROOT / "training" / "baseline_results.json"
    out_path.write_text(json.dumps(
        {name: {"default": d, "tuned": t} for name, (d, t) in results.items()},
        indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
