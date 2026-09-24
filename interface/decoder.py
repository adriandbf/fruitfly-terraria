"""Phase 4: motor decoder v0. Descending-neuron spike rates (from
FastBrain.stimulate's output) -> a game action dict, per the real,
individually-identified descending neurons picked in brain/neurons.py:

  DNp09 (forward walk), DNa02 (steering), MDN (backward walk), DNp01 ==
  Giant Fiber (jump). No literature-established "attack" descending neuron
  was found, so "use" falls back to overall descending-ensemble activity --
  flagged below as the weakest-grounded part of this v0 pass.

Priority, each step: retreat (MDN) overrides steer/forward (DNp09/DNa02)
overrides idle. Jump and "use" are independent of the movement decision.
All thresholds are hand-picked defaults for v0, not tuned -- Phase 5's
CMA-ES loop is where that belongs; don't read anything into the specific
numbers beyond "plausible starting point".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "brain"))
import neurons

FORWARD_THRESHOLD_HZ = 20.0
BACKWARD_THRESHOLD_HZ = 20.0
STEER_MARGIN_HZ = 5.0     # min left/right DNa02 gap before committing to a side
JUMP_THRESHOLD_HZ = 30.0  # Giant Fiber: fast, decisive, high-threshold in real flies
USE_THRESHOLD_HZ = 15.0


def _mean_rate(rates_by_flyid, flyids):
    vals = [rates_by_flyid.get(f, 0.0) for f in flyids]
    return sum(vals) / len(vals) if vals else 0.0


def _max_rate(rates_by_flyid, flyids):
    vals = [rates_by_flyid.get(f, 0.0) for f in flyids]
    return max(vals) if vals else 0.0


def decode(rates_by_flyid, player_direction):
    """rates_by_flyid: {flywire_id: spike_rate_hz} for (at least) the
    descending neurons in brain/neurons.py -- missing entries treated as 0.
    player_direction: the character's current facing, 1 or -1 (from
    get_state's player.direction), since "forward"/"backward" are relative
    to heading, not absolute left/right.
    Returns {"left": bool, "right": bool, "jump": bool, "use": bool}.
    """
    forward_rate = _max_rate(rates_by_flyid, neurons.DNP09_FORWARD)
    backward_rate = _mean_rate(rates_by_flyid, neurons.MDN_BACKWARD)
    left_steer = _mean_rate(rates_by_flyid, neurons.DNA02_STEERING_LEFT)
    right_steer = _mean_rate(rates_by_flyid, neurons.DNA02_STEERING_RIGHT)
    jump_rate = _max_rate(rates_by_flyid, neurons.DNP01_JUMP)
    use_rate = _mean_rate(rates_by_flyid, neurons.ALL_DESCENDING)

    steer_gap = abs(right_steer - left_steer)

    # Originally gated steering behind forward_rate crossing its own
    # threshold ("DNp09 initiates, DNa02 steers"), on the assumption that's
    # how the two would divide labor here. A live check (see CLAUDE.md/
    # run_fly.py history) showed the real connectome doesn't route this
    # project's gustatory sensory drive through DNp09 at all -- it only
    # ever reached DNa02, so movement never triggered though a clear,
    # usable steering signal (DNa02 asymmetry) was present the whole time.
    # Each signal now independently competes to set direction, backward
    # only winning if it's the strongest of the three.
    direction = 0
    if (backward_rate > BACKWARD_THRESHOLD_HZ
            and backward_rate >= steer_gap and backward_rate >= forward_rate):
        direction = -player_direction  # retreat: opposite of current facing
    elif steer_gap > STEER_MARGIN_HZ:
        direction = 1 if right_steer > left_steer else -1
    elif forward_rate > FORWARD_THRESHOLD_HZ:
        direction = player_direction  # keep walking the way we're facing

    return {
        "left": direction < 0,
        "right": direction > 0,
        "jump": jump_rate > JUMP_THRESHOLD_HZ,
        "use": use_rate > USE_THRESHOLD_HZ,
    }
