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

# Phase 5: these five are exactly what training/train_cma.py tunes via
# CMA-ES -- decode() takes an optional `params` dict (any missing key
# falls back to the default) so the training loop can pass a candidate
# vector without touching this module's globals. Called with no `params`,
# behavior is unchanged from before Phase 5.
DEFAULT_PARAMS = {
    "forward_threshold_hz": 20.0,
    "backward_threshold_hz": 20.0,
    "steer_margin_hz": 5.0,     # min left/right DNa02 gap before committing to a side
    "jump_threshold_hz": 30.0,  # Giant Fiber: fast, decisive, high-threshold in real flies
    "use_threshold_hz": 15.0,
    # StatefulDecoder (flee) only -- decode() ignores these:
    "flee_on_hz": 30.0,        # Giant Fiber rate that starts a flee
    "flee_off_hz": 15.0,       # ...and the lower rate that keeps it going (hysteresis)
    "flee_hold_ticks": 5,      # keep fleeing this long (0.5s) after it drops below
    "startle_hold_ticks": 1,   # startle-hop jump key hold: 1 tick ~ 6 frames ~ a ~3.8-tile hop
}


def _mean_rate(rates_by_flyid, flyids):
    vals = [rates_by_flyid.get(f, 0.0) for f in flyids]
    return sum(vals) / len(vals) if vals else 0.0


def _max_rate(rates_by_flyid, flyids):
    vals = [rates_by_flyid.get(f, 0.0) for f in flyids]
    return max(vals) if vals else 0.0


def decode(rates_by_flyid, player_direction, params=None):
    """rates_by_flyid: {flywire_id: spike_rate_hz} for (at least) the
    descending neurons in brain/neurons.py -- missing entries treated as 0.
    player_direction: the character's current facing, 1 or -1 (from
    get_state's player.direction), since "forward"/"backward" are relative
    to heading, not absolute left/right.
    params: optional overrides for any of DEFAULT_PARAMS.
    Returns {"left": bool, "right": bool, "jump": bool, "use": bool}.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

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
    if (backward_rate > p["backward_threshold_hz"]
            and backward_rate >= steer_gap and backward_rate >= forward_rate):
        direction = -player_direction  # retreat: opposite of current facing
    elif steer_gap > p["steer_margin_hz"]:
        # Sign is deliberately reversed vs. a naive "steer toward the more
        # active side" reading -- left_steer dominating empirically means
        # the target is on the RIGHT in this connectome (see docstring).
        direction = 1 if left_steer > right_steer else -1
    elif forward_rate > p["forward_threshold_hz"]:
        direction = player_direction  # keep walking the way we're facing

    return {
        "left": direction < 0,
        "right": direction > 0,
        "jump": jump_rate > p["jump_threshold_hz"],
        "use": use_rate > p["use_threshold_hz"],
    }


class StatefulDecoder:
    """decode() plus a flee reflex. Needs state across ticks, so it's a class;
    decode() stays stateless for the training arena and older callers.

    Design, chosen with the user: when a threat is near, do a one-time
    STARTLE HOP and RUN AWAY from it; fear beats approach.
      - Threat = Giant Fiber (DNp01) rate >= flee_on_hz, kept going while it
        stays >= flee_off_hz, then held flee_hold_ticks more (hysteresis +
        hold, because GF flickers around the threshold at range).
      - Direction comes from the BRAIN: DNa02 is a clean, mirror-symmetric,
        crossed code for "which side the stimulus is on" (hostile on the
        right -> DNa02_L fires): 100% correct when it fires, in 86% of
        slices (interface/validate_threat_direction.py). When DNa02 is
        silent the Giant Fiber's L-vs-R lean breaks the tie; when both are
        silent the last decided direction is held. Flee = the opposite way.
        Before any direction has ever been decided, the fly hops but doesn't
        run (a coin-flip run risks running INTO the threat).
      - The hop is ONE jump per threat onset, then the key is released: the
        engine only starts a new jump after a frame with jump NOT held
        (releaseJump), so a held key gives exactly one hop and pins the key.
        It re-arms once the threat is over.
      - Fear beats approach has two halves: the ENCODER silences the approach
        drive near a threat (so the shared DNa02 readout is threat-pure --
        see encoder.approach_suppression_hz), and here, while fleeing, the
        base approach/steering decision is ignored entirely.
    Outside a flee this is exactly decode() with jump forced off.
    """

    def __init__(self, params=None):
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.reset()

    def reset(self):
        self._active = False
        self._hold_left = 0
        self._hop_left = 0
        self._flee_dir = 0     # +1 run right / -1 run left / 0 not yet decided
        self.last_mode = "base"
        self.last_flee_dir = 0

    def decode(self, rates_by_flyid, player_direction):
        p = self.p
        gf = _max_rate(rates_by_flyid, neurons.DNP01_JUMP)

        if gf >= p["flee_on_hz"] or (self._active and gf >= p["flee_off_hz"]):
            active, self._hold_left = True, p["flee_hold_ticks"]
        elif self._hold_left > 0:
            active, self._hold_left = True, self._hold_left - 1
        else:
            active = False
        onset = active and not self._active
        self._active = active

        if not active:
            self._flee_dir, self._hop_left = 0, 0
            action = decode(rates_by_flyid, player_direction, p)
            action["jump"] = False
            self.last_mode, self.last_flee_dir = "base", 0
            return action

        if onset:
            self._hop_left = p["startle_hold_ticks"]

        # Which side is the threat on? DNa02 first, GF lean as tie-break.
        evidence = (_mean_rate(rates_by_flyid, neurons.DNA02_STEERING_LEFT)
                    - _mean_rate(rates_by_flyid, neurons.DNA02_STEERING_RIGHT))  # +: stimulus on the right
        if evidence == 0:
            evidence = (_mean_rate(rates_by_flyid, neurons.DNP01_JUMP_RIGHT)
                        - _mean_rate(rates_by_flyid, neurons.DNP01_JUMP_LEFT))
        if evidence != 0:
            self._flee_dir = -1 if evidence > 0 else 1   # run the OTHER way

        hop = self._hop_left > 0
        if hop:
            self._hop_left -= 1
        self.last_mode, self.last_flee_dir = "flee", self._flee_dir
        return {
            "left": self._flee_dir < 0,
            "right": self._flee_dir > 0,
            "jump": hop,
            "use": False,
        }
