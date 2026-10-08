"""Phase 4: sensory encoder v0. Game state (from bridge.get_state) -> which
sensory neurons to stimulate and at what rate, ready for
FastBrain.stimulate(neuron_ids, rates_hz, ...).

Two channels (see brain/neurons.py for the citations and how they were
picked):
  - SUGAR_WATER_GRN: "approach" drive, from the nearest other player (the
    human) -- reuses the exact appetitive mechanism already validated in
    Phase 0/3, repurposed as "walk toward the human" per the plan's goal.
  - LC4_LOOMING: "threat nearby" drive, from the nearest hostile NPC.
    Originally BITTER_GRN (gustatory/bitter) -- confirmed dead live (real
    combat, character took damage, never reacted; ~50-90 total neurons
    active, no descending/motor hit at any strength). Real fly threat/
    escape circuits are visual, not gustatory: LC4 is a lobula columnar
    looming-detector neuron, the canonically published primary visual
    input to the Giant Fiber escape circuit -- and empirically drives
    DNp01 (this project's jump target) strongly and bilaterally
    (brain/find_looming_pathway.py).

Both scale rate linearly with proximity (closer = higher Hz, 0 beyond
max_range). LATERALIZED by which side of the player the target is on,
splitting drive between each population's left/right sub-groups (full rate
to the target's side, a smaller fraction to the other) -- an early version
drove both sides identically, and an offline test with a synthetic game
state (human due east of the player) confirmed the resulting steering was
directionless: the network's left/right asymmetry came entirely from its
own internal wiring bias, not the input, so "left"/"right" in the output
had no relationship to which side the human was actually on. Real
gustatory GRNs aren't actually used by flies for
long-range spatial navigation this way (that's more a visual/plume-tracking
behavior) -- this is a hand-built v0 stand-in, using "side" from the real
annotation data (soma/nerve-entry side) as a proxy for "which half of the
world the target is in", not a claim about real fly gustatory navigation.

Phase 5: the four numeric knobs below (DEFAULT_PARAMS) are exactly what
training/train_cma.py tunes via CMA-ES -- compute_drives()/encode() take
an optional `params` dict (any missing key falls back to the default) so
the training loop can pass in a candidate parameter vector without
touching this module's globals. Called with no `params`, behavior is
unchanged from before Phase 5 (existing callers -- run_fly.py,
validate_v0_pipeline.py -- are unaffected).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "brain"))
import neurons

DEFAULT_PARAMS = {
    "max_rate_hz": 150.0,          # matches model.py's default r_poi
    "approach_max_range_px": 2000.0,  # ~125 tiles; 0Hz drive beyond this
    "threat_max_range_px": 800.0,     # matches the bridge mod's own NPC report radius
    # Contralateral (far) side still gets a fraction of the rate, rather
    # than a hard 0, to avoid a discontinuous on/off flip exactly at dx=0.
    "contra_fraction": 0.15,
    # Fear beats appetite, enforced HERE at the sensory stage (a hand-written
    # arbitration rule, not something the brain computes): approach drive is
    # scaled by max(0, 1 - threat_hz / this). 0 disables it. Why it can't be
    # done in the decoder: both channels drive the same steering neuron
    # (DNa02), and with a human AND a hostile present the approach signal
    # swamps the threat's weak one -- measured per 100ms slice, the threat-
    # side readout was only 77% correct unsuppressed (5/35 with the human on
    # the right and the hostile on the left: it would flee INTO the hostile)
    # versus 100% correct with approach suppressed. 50Hz of threat drive is
    # a hostile within ~530px at the default range.
    "approach_suppression_hz": 50.0,
}


def _nearest(entities, px, py):
    """Closest entity (dict with 'x'/'y') to (px, py), or None."""
    best, best_d2 = None, None
    for e in entities:
        dx, dy = e["x"] - px, e["y"] - py
        d2 = dx * dx + dy * dy
        if best_d2 is None or d2 < best_d2:
            best, best_d2 = e, d2
    return best, math.sqrt(best_d2) if best is not None else None


def _range_rate(distance, max_range, max_rate_hz):
    if distance is None or max_range <= 0:
        return 0.0
    frac = max(0.0, min(1.0, 1.0 - distance / max_range))
    return max_rate_hz * frac


def _side_of(entity, player_x):
    return "right" if entity["x"] >= player_x else "left"


def compute_drives(state, params=None):
    """The two named channels' rate and side -- the thing worth logging
    for interpretability. {"approach_hz", "approach_side", "threat_hz",
    "threat_side", "approach_scale"}; side is "left"/"right"/None, all 0/None
    if not in_world or nothing in range. approach_hz is AFTER suppression by
    a nearby threat; approach_scale is the factor applied (1.0 = none)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if not state.get("in_world"):
        return {"approach_hz": 0.0, "approach_side": None,
                "threat_hz": 0.0, "threat_side": None, "approach_scale": 1.0}

    player = state["player"]
    px, py = player["x"], player["y"]

    human, human_dist = _nearest(state.get("players", []), px, py)
    approach_rate = _range_rate(human_dist, p["approach_max_range_px"], p["max_rate_hz"])

    hostiles = [n for n in state.get("npcs", []) if n.get("hostile")]
    hostile, threat_dist = _nearest(hostiles, px, py)
    threat_rate = _range_rate(threat_dist, p["threat_max_range_px"], p["max_rate_hz"])

    # Fear beats appetite (see approach_suppression_hz above).
    supp = p["approach_suppression_hz"]
    approach_scale = max(0.0, 1.0 - threat_rate / supp) if supp > 0 else 1.0
    approach_rate *= approach_scale

    return {
        "approach_hz": approach_rate,
        "approach_side": _side_of(human, px) if human else None,
        "threat_hz": threat_rate,
        "threat_side": _side_of(hostile, px) if hostile else None,
        "approach_scale": approach_scale,
    }


def _lateralized(rate, side, left_ids, right_ids, contra_fraction):
    """Split `rate` between left_ids/right_ids: full rate to the side the
    target is on, contra_fraction to the other. side=None (nothing to
    drive toward) still returns both id lists, just all at 0 (rate is 0
    in that case anyway, from compute_drives)."""
    ipsi, contra = (right_ids, left_ids) if side == "right" else (left_ids, right_ids)
    ids = list(ipsi) + list(contra)
    rates = [rate] * len(ipsi) + [rate * contra_fraction] * len(contra)
    return ids, rates


def encode(state, params=None):
    """state: a get_state response dict. params: optional overrides for
    any of DEFAULT_PARAMS (Phase 5's CMA-ES loop passes candidate values).
    Returns (neuron_ids, rates_hz) lists, ready for FastBrain.stimulate().
    Empty lists (no drive) if not in_world."""
    if not state.get("in_world"):
        return [], []

    p = {**DEFAULT_PARAMS, **(params or {})}
    drives = compute_drives(state, p)
    approach_ids, approach_rates = _lateralized(
        drives["approach_hz"], drives["approach_side"],
        neurons.SUGAR_WATER_GRN_LEFT, neurons.SUGAR_WATER_GRN_RIGHT, p["contra_fraction"])
    threat_ids, threat_rates = _lateralized(
        drives["threat_hz"], drives["threat_side"],
        neurons.LC4_LOOMING_LEFT, neurons.LC4_LOOMING_RIGHT, p["contra_fraction"])
    return approach_ids + threat_ids, approach_rates + threat_rates
