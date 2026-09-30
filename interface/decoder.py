"""Phase 4: motor decoder v0. Descending-neuron spike rates (from
FastBrain.stimulate's output) -> a game action dict, per the real,
individually-identified descending neurons picked in brain/neurons.py:

  DNb05 (movement gate), DNa02 (steering), MDN (backward walk), DNp01 ==
  Giant Fiber (jump). No literature-established "attack" descending neuron
  was found, so "use" falls back to overall descending-ensemble activity --
  flagged below as the weakest-grounded part of this v0 pass.

Priority, each step: retreat (MDN) overrides steer/forward (DNb05/DNa02)
overrides idle. Jump and "use" are independent of the movement decision.
All thresholds are hand-picked defaults for v0, not tuned -- Phase 5's
CMA-ES loop is where that belongs; don't read anything into the specific
numbers beyond "plausible starting point".

Two things this v0 got wrong the first time, both caught by a live test
(character took real damage, never reacted at all) and a follow-up
diagnostic sweep (brain/find_reachable_motor_neurons.py) -- see
brain/neurons.py's docstring for the full account:
  - DNp09 (the original forward-walk pick) never actually fires from
    either sensory channel in this connectome -- swapped for DNb05, which
    empirically does, strongly.
  - The DNa02 left/right comparison had its sign backwards: DNa02_LEFT
    fires specifically when the target is on the RIGHT (confirmed twice,
    independently), the opposite of a naive ipsilateral-steering
    assumption. Fixed below.
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
    forward_rate = _max_rate(rates_by_flyid, neurons.DNB05_MOVE)
    backward_rate = _mean_rate(rates_by_flyid, neurons.MDN_BACKWARD)
    left_steer = _mean_rate(rates_by_flyid, neurons.DNA02_STEERING_LEFT)
    right_steer = _mean_rate(rates_by_flyid, neurons.DNA02_STEERING_RIGHT)
    jump_rate = _max_rate(rates_by_flyid, neurons.DNP01_JUMP)
    use_rate = _mean_rate(rates_by_flyid, neurons.ALL_DESCENDING)

    steer_gap = abs(right_steer - left_steer)

    # Each signal independently competes to set direction, backward only
    # winning if it's the strongest of the three (see module docstring for
    # why DNb05/the swapped DNa02 sign replaced the original DNp09/naive-
    # sign version).
    direction = 0
    if (backward_rate > BACKWARD_THRESHOLD_HZ
            and backward_rate >= steer_gap and backward_rate >= forward_rate):
        direction = -player_direction  # retreat: opposite of current facing
    elif steer_gap > STEER_MARGIN_HZ:
        # Sign is deliberately reversed vs. a naive "steer toward the more
        # active side" reading -- left_steer dominating empirically means
        # the target is on the RIGHT in this connectome (see docstring).
        direction = 1 if left_steer > right_steer else -1
    elif forward_rate > FORWARD_THRESHOLD_HZ:
        direction = player_direction  # keep walking the way we're facing

    return {
        "left": direction < 0,
        "right": direction > 0,
        "jump": jump_rate > JUMP_THRESHOLD_HZ,
        "use": use_rate > USE_THRESHOLD_HZ,
    }
