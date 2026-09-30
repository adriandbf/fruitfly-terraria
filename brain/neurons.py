"""Named neuron sets for the encoder/decoder, picked from real FlyWire
annotations (CLAUDE.md Phase 4: "Pick motor neurons from FlyWire
annotations, not guesses"), not from this project's own guesswork.

Source: flyconnectome/flywire_annotations (Schlegel et al.),
supplemental_files/Supplemental_file1_neuron_annotations.tsv, main branch,
retrieved 2026-09-24. See brain/selected_neurons.json for the full rows
(cell_type, hemibrain_type, side, top_nt) this was built from -- that file
is the source of truth; this module just exposes it conveniently. The raw
30MB annotation table itself isn't committed (same "don't commit connectome
data" convention as brain/Drosophila_brain_model/), but the ~200 rows we
actually use are small enough to check in directly.

IMPORTANT: root_id here is FlyWire **v783**, not v630. FlyWire root IDs are
not stable across proofreading versions, and the annotation table only
ships v783 IDs -- so Phase 4 onward uses the v783 connectome files already
bundled in Drosophila_brain_model/ (Completeness_783.csv,
Connectivity_783.parquet), not the v630 files Phase 0-3 validated against.
Re-validated fast_sim.py against v783 specifically (see
validate_fast_sim_v783.py) before building on it: same conclusions held
(speed and spike-statistics agreement with Brian), just a ~9% bigger graph
(138,639 neurons / 15.09M synapses vs 127,400 / 14.7M).

Encoder side (sensory drive):
    SUGAR_WATER_GRN -- 129 neurons, gustatory/sugar-water. "Approach"
        signal: reuses the exact mechanism already validated in Phase 0/3
        (the original smoke test's "sugar neurons"), repurposed here as the
        drive for walking toward the human player.
    BITTER_GRN -- 65 neurons, gustatory/bitter. "Threat nearby" signal, for
        a hostile NPC close to the player -- bitter taste is a well
        established aversive channel in Drosophila.

Decoder side (motor readout). FlyWire is brain-only (no ventral nerve
cord), so there's no literal leg motor neuron in this connectome -- the
standard readout for connectome-based behavior control is DESCENDING
NEURONS, the brain neurons whose axons carry commands down to the VNC.
These four are specific, individually identified, well-characterized types
from the published literature (not "some descending neuron"):
    DNP09_FORWARD -- 2 neurons (1 per side). DNp09: initiates forward
        walking (Bidaye et al. 2020).
    DNA02_STEERING -- 2 neurons (1 per side). DNa02: steering control,
        asymmetric activity biases turning toward the more active side
        (Rayshubskiy et al. 2020).
    MDN_BACKWARD -- 4 neurons (2 per side). MDN ("Moonwalker Descending
        Neuron"): drives backward walking (Bidaye et al. 2014).
    DNP01_JUMP -- 2 neurons (1 per side). DNp01 == the Giant Fiber neuron
        (confirmed via hemibrain_type == "Giant Fiber" in the annotation
        table) -- the classic fast escape-jump command neuron.

No literature-established "attack nearby enemy" descending neuron was
identified for v0 -- decoder.py falls back to overall descending-ensemble
activity for that action, flagged there as the weakest-grounded part of
this pass, to revisit once real neurons.

UPDATE after the first live test (character took real damage, 100->77 HP,
and never reacted at all -- confirmed offline first in
interface/validate_v0_pipeline.py, then live): none of the four neurons
above ever actually fired from either sensory channel, sustained, at max
drive (brain/find_reachable_motor_neurons.py, a diagnostic sweep
cross-referencing every neuron that DOES light up against the real
annotation table's descending/motor categories, not just these four).
Findings from that sweep:
    DNB05_MOVE -- 2 neurons (1 per side). DNb05: NOT independently
        published for a specific behavior as far as this search found, but
        empirically the strongest, cleanest signal in this connectome for
        this project's approach channel -- fires strongly and bilaterally
        (200Hz left / 180Hz right) under sustained sugar_right drive, and
        not at all under sugar_left. Replaces DNP09_FORWARD as the "is
        there something to move toward" gate; DNp09 itself never fired
        from either sensory channel at all.
    DNA02_STEERING's sign was backwards, not its existence: DNa02_LEFT
        fires specifically when sugar_right (not sugar_left) is driven --
        confirmed in both the original un-lateralized test and this
        diagnostic. Since sugar_right is driven at full rate specifically
        when the target is on the right (encoder.py's lateralization),
        "DNa02_left active" empirically means "target is on the right",
        the opposite of what a naive ipsilateral-steering assumption would
        predict. Plausibly a crossed (contralateral) pathway from this
        particular sensory population to DNa02 in this individually
        reconstructed brain -- decoder.py now uses the sign the data
        actually show, not the textbook default.
    No usable substitute was found for MDN (backward) or DNp01 (jump) --
    neither ever appeared active under either sensory channel in the
    sweep. Left as-is (real, cited neurons that just don't fire from
    these particular inputs yet); the bitter/threat channel overall
    remains much weaker than sugar/approach in this connectome (~50-70
    total neurons active vs. thousands), so a stronger threat-side
    substitute may need a different channel entirely, not just a
    different neuron -- a Phase 5 question, not fixed here.
"""
import json
from pathlib import Path

_DATA = json.loads((Path(__file__).parent / "selected_neurons.json").read_text())


def _ids(section, key):
    return [int(row["root_id"]) for row in _DATA[section][key]]


def _ids_by_side(section, key, side):
    return [int(row["root_id"]) for row in _DATA[section][key] if row["side"] == side]


# --- sensory (encoder) ---
SUGAR_WATER_GRN = _ids("sensory", "sugar_water_grn")
SUGAR_WATER_GRN_LEFT = _ids_by_side("sensory", "sugar_water_grn", "left")
SUGAR_WATER_GRN_RIGHT = _ids_by_side("sensory", "sugar_water_grn", "right")

BITTER_GRN = _ids("sensory", "bitter_grn")
BITTER_GRN_LEFT = _ids_by_side("sensory", "bitter_grn", "left")
BITTER_GRN_RIGHT = _ids_by_side("sensory", "bitter_grn", "right")

# --- descending (decoder) ---
DNP09_FORWARD = _ids("descending", "DNp09_forward_walk")
DNP09_FORWARD_LEFT = _ids_by_side("descending", "DNp09_forward_walk", "left")
DNP09_FORWARD_RIGHT = _ids_by_side("descending", "DNp09_forward_walk", "right")

DNA02_STEERING = _ids("descending", "DNa02_steering")
DNA02_STEERING_LEFT = _ids_by_side("descending", "DNa02_steering", "left")
DNA02_STEERING_RIGHT = _ids_by_side("descending", "DNa02_steering", "right")

MDN_BACKWARD = _ids("descending", "MDN_backward_walk")

DNP01_JUMP = _ids("descending", "DNp01_giant_fiber_jump")

DNB05_MOVE = _ids("descending", "DNb05_forward_v2")
DNB05_MOVE_LEFT = _ids_by_side("descending", "DNb05_forward_v2", "left")
DNB05_MOVE_RIGHT = _ids_by_side("descending", "DNb05_forward_v2", "right")

ALL_DESCENDING = DNP09_FORWARD + DNA02_STEERING + MDN_BACKWARD + DNP01_JUMP + DNB05_MOVE
