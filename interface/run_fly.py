"""Phase 4: the actual fly-brain control loop, replacing Phase 2's scripted
dumb_bot.py. Ties together:

    bridge.get_state() -> encoder.encode() -> FastBrain.stimulate()
        -> decoder.decode() -> bridge.set_action()

at 10 Hz (100ms of simulated brain per decision, per CLAUDE.md's control
loop), logging every tick to logs/. UNTESTED end to end against the live
game as of this writing -- the brain side (encoder/decoder/FastBrain) has
its own offline checks (see brain/validate_fast_sim_v783.py and this
file's __main__ smoke test), but a live run needs the bridge mod rebuilt
first (GameState.cs gained a "players" field this pass) and a human in the
world to walk toward.

Uses the v783 connectome (not v630, which Phases 0-3 validated against) --
see brain/neurons.py's docstring for why.

Usage: python interface/run_fly.py [--seconds N] [--params-file PATH]
--params-file loads a Phase 5 training/train_cma.py output (e.g.
training/best_params.json) to use CMA-ES-tuned parameters instead of
encoder.py/decoder.py's hand-picked defaults.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "brain"))
sys.path.insert(0, str(ROOT / "brain" / "Drosophila_brain_model"))
sys.path.insert(0, str(ROOT / "bridge"))

import pandas as pd
from model import default_params as params
from fast_sim import FastBrain
import neurons
import encoder
import decoder
from dumb_bot import Bridge  # reuse the tested socket client from Phase 2

TICK_HZ = 10
TICK_S = 1.0 / TICK_HZ
DT_MS = 1.0  # see brain/bench_fast_sim.py / validate_fast_sim_v783.py for why

LOG_DIR = ROOT / "logs"

RELEASE = {"left": False, "right": False, "jump": False, "use": False}

# The handful of named neurons worth logging per tick for interpretability
# (not the full ~138k-neuron rate array -- see module docstring's sibling
# note in CLAUDE.md-style logging convention: this is a deliberate, small,
# purposeful subset, not "log everything").
READOUT_GROUPS = {
    "DNb05_forward": neurons.DNB05_MOVE,
    "DNa02_left": neurons.DNA02_STEERING_LEFT,
    "DNa02_right": neurons.DNA02_STEERING_RIGHT,
    "MDN_backward": neurons.MDN_BACKWARD,
    "DNp01_jump": neurons.DNP01_JUMP,
}


def build_brain():
    repo = ROOT / "brain" / "Drosophila_brain_model"
    df_comp = pd.read_csv(repo / "Completeness_783.csv", index_col=0)
    df_con = pd.read_parquet(repo / "Connectivity_783.parquet")
    return FastBrain(df_comp, df_con, params, DT_MS, exc_indices=[], seed=0)


def rates_for_readout(brain, rates):
    """Full per-neuron rates array -> {flywire_id: rate} for just the
    descending neurons the decoder (and our own logging) cares about."""
    return {fid: brain.rate_for(rates, fid) for fid in neurons.ALL_DESCENDING}


def run(duration_s=None, enc_params=None, dec_params=None):
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run_fly_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    print(f"Building brain (v783, dt={DT_MS}ms)...")
    brain = build_brain()
    brain.stimulate([], [], duration_ms=10)  # JIT warm-up, not logged
    print(f"Logging to {log_path}")

    tick = 0
    start = time.monotonic()

    with open(log_path, "w") as log_file:
        while duration_s is None or time.monotonic() - start < duration_s:
            try:
                bridge = Bridge()
            except OSError as e:
                print(f"Can't reach the bridge ({e}); is the mod loaded? Retrying in 2s.")
                time.sleep(2)
                continue

            try:
                while duration_s is None or time.monotonic() - start < duration_s:
                    loop_start = time.monotonic()

                    state = bridge.request({"cmd": "get_state"})
                    drives = encoder.compute_drives(state, enc_params)
                    neuron_ids, rates_hz = encoder.encode(state, enc_params)
                    spike_rates = brain.stimulate(neuron_ids, rates_hz, duration_ms=100)
                    readout = rates_for_readout(brain, spike_rates)

                    player_dir = (state.get("player") or {}).get("direction", 1)
                    action = decoder.decode(readout, player_dir, dec_params)
                    bridge.request({"cmd": "set_action", **action})

                    log_file.write(json.dumps({
                        "tick": tick, "t": time.time(), "state": state,
                        "encoder_drive": drives,
                        "decoder_rates": {name: sum(readout.get(f, 0.0) for f in ids) / max(len(ids), 1)
                                          for name, ids in READOUT_GROUPS.items()},
                        "action": action,
                    }) + "\n")
                    log_file.flush()

                    if tick % TICK_HZ == 0:  # once a second
                        hp = (state.get("player") or {}).get("hp")
                        print(f"tick {tick:6d}  in_world={state.get('in_world')}  hp={hp}  action={action}")

                    tick += 1
                    elapsed = time.monotonic() - loop_start
                    if elapsed > TICK_S:
                        print(f"  (slow tick: {elapsed*1000:.0f}ms, over the {TICK_S*1000:.0f}ms budget)")
                    time.sleep(max(0.0, TICK_S - elapsed))
            except (ConnectionError, OSError) as e:
                print(f"Bridge connection lost ({e}); reconnecting... (brain state kept)")
            finally:
                bridge.release()
                bridge.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=None,
                         help="run for a fixed duration instead of until Ctrl+C")
    parser.add_argument("--params-file", type=str, default=None,
                         help="path to a Phase 5 training/train_cma.py output "
                              "(e.g. training/best_params.json) to use tuned "
                              "encoder/decoder parameters instead of the hand-picked "
                              "defaults in encoder.py/decoder.py")
    args = parser.parse_args()

    enc_params, dec_params = None, None
    if args.params_file:
        data = json.loads(Path(args.params_file).read_text())
        enc_params, dec_params = data["encoder"], data["decoder"]
        print(f"Using tuned params from {args.params_file} "
              f"(trained reward: {data.get('best_reward', '?')})")

    try:
        run(duration_s=args.seconds, enc_params=enc_params, dec_params=dec_params)
    except KeyboardInterrupt:
        print("\nStopping, releasing keys...")


if __name__ == "__main__":
    main()
