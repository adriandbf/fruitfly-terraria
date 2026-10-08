"""Offline tests for the flee reflex: decoder.StatefulDecoder and the
encoder's fear-suppresses-appetite rule. Synthetic neuron rates, so each
behavior is pinned down on its own (the brain-in-the-loop check is
training/flee_eval.py).

Run: python interface/test_flee.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "brain"))
import neurons
import encoder
from decoder import StatefulDecoder, decode

passed = 0


def check(name, cond):
    global passed
    assert cond, f"FAIL: {name}"
    passed += 1
    print(f"  ok  {name}")


def rates(gf_l=0.0, gf_r=0.0, dna_l=0.0, dna_r=0.0, dnb=0.0):
    r = {}
    for f in neurons.DNP01_JUMP_LEFT: r[f] = gf_l
    for f in neurons.DNP01_JUMP_RIGHT: r[f] = gf_r
    for f in neurons.DNA02_STEERING_LEFT: r[f] = dna_l
    for f in neurons.DNA02_STEERING_RIGHT: r[f] = dna_r
    for f in neurons.DNB05_MOVE: r[f] = dnb
    return r


def moving(a):
    return "right" if a["right"] else "left" if a["left"] else "none"


print("outside a flee")
d = StatefulDecoder()
a = d.decode(rates(dna_l=40, dnb=200), 1)
check("no threat: identical to decode() (approach: DNa02_L dominant -> right)", moving(a) == "right" and d.last_mode == "base")
check("no threat: jump forced off", a["jump"] is False)
a = d.decode(rates(gf_l=20, gf_r=20), 1)  # below flee_on_hz
check("Giant Fiber below the on-threshold does not start a flee", d.last_mode == "base")

print("fleeing, direction from the brain")
d = StatefulDecoder()
a = d.decode(rates(gf_l=60, gf_r=90, dna_l=10), 1)  # DNa02_L fires => threat on the RIGHT
check("threat on the right (DNa02_L): runs LEFT", moving(a) == "left" and d.last_mode == "flee")
d = StatefulDecoder()
a = d.decode(rates(gf_l=90, gf_r=60, dna_r=10), 1)  # DNa02_R fires => threat on the LEFT
check("threat on the left (DNa02_R): runs RIGHT", moving(a) == "right")
d = StatefulDecoder()
a = d.decode(rates(gf_l=60, gf_r=90, dna_l=40, dnb=200), -1)
check("flee ignores the approach decision (same DNa02 would mean 'go right' to approach)",
      moving(a) == "left")
check("no attacking while fleeing", a["use"] is False)

d = StatefulDecoder()
a = d.decode(rates(gf_l=30, gf_r=90), 1)  # DNa02 silent: GF lean breaks the tie (R > L => threat right)
check("DNa02 silent: Giant Fiber lean breaks the tie (threat right -> run left)", moving(a) == "left")

print("holding direction when evidence is silent")
d = StatefulDecoder()
d.decode(rates(gf_l=60, gf_r=90, dna_l=10), 1)
a = d.decode(rates(gf_l=70, gf_r=70), 1)  # everything silent but GF level
check("both direction signals silent: holds the last decided direction", moving(a) == "left")
d = StatefulDecoder()
a = d.decode(rates(gf_l=70, gf_r=70), 1)
check("no direction ever decided: hops but does NOT run (a coin-flip risks running into it)",
      moving(a) == "none" and a["jump"] is True)

print("the startle hop")
d = StatefulDecoder()
seq = [d.decode(rates(gf_l=60, gf_r=90, dna_l=10), 1)["jump"] for _ in range(5)]
check("one hop on onset, then released for the rest of the threat", seq == [True, False, False, False, False])
d = StatefulDecoder(dict(startle_hold_ticks=2))
seq = [d.decode(rates(gf_l=60, gf_r=90, dna_l=10), 1)["jump"] for _ in range(4)]
check("startle_hold_ticks is honored", seq == [True, True, False, False])

print("hysteresis, hold and re-arming")
d = StatefulDecoder()
d.decode(rates(gf_l=60, gf_r=90, dna_l=10), 1)
a = d.decode(rates(gf_l=20, gf_r=20, dna_l=10), 1)  # between off(15) and on(30), already active
check("while active, GF between the thresholds keeps fleeing (hysteresis)", d.last_mode == "flee")
d = StatefulDecoder()
d.decode(rates(gf_l=60, gf_r=90, dna_l=10), 1)
modes = []
for _ in range(8):
    d.decode(rates(), 1)
    modes.append(d.last_mode)
check("after GF drops, keeps fleeing for the hold (5 ticks) then returns to base",
      modes == ["flee"] * 5 + ["base"] * 3)
a = d.decode(rates(gf_l=60, gf_r=90, dna_l=10), 1)
check("a new threat after it ended hops again (re-armed)", a["jump"] is True)
d = StatefulDecoder()
d.decode(rates(gf_l=60, gf_r=90, dna_l=10), 1)
d.reset()
check("reset() clears the flee state", d.last_mode == "base" and d._flee_dir == 0 and not d._active)

print("encoder: fear beats appetite")
base = {"in_world": True, "player": {"x": 1000, "y": 0, "direction": 1},
        "players": [{"x": 1300, "y": 0}], "npcs": []}
near = dict(base, npcs=[{"x": 1100, "y": 0, "hostile": True}])   # 100px
far = dict(base, npcs=[{"x": 1700, "y": 0, "hostile": True}])    # 700px
none_ = encoder.compute_drives(base)
check("no threat: approach untouched (scale 1.0)", none_["approach_scale"] == 1.0 and none_["approach_hz"] > 0)
dn = encoder.compute_drives(near)
check("hostile at 100px: approach drive silenced entirely", dn["approach_hz"] == 0.0 and dn["approach_scale"] == 0.0)
df = encoder.compute_drives(far)
check("hostile at 700px: approach only partly suppressed",
      0.0 < df["approach_scale"] < 1.0 and 0.0 < df["approach_hz"] < none_["approach_hz"])
off = encoder.compute_drives(near, {"approach_suppression_hz": 0.0})
check("approach_suppression_hz=0 disables the rule", off["approach_scale"] == 1.0 and off["approach_hz"] > 0)
ids, rs = encoder.encode(near)
sugar = set(neurons.SUGAR_WATER_GRN)
check("encode(): sugar neurons are driven at 0Hz when a hostile is close",
      all(r == 0.0 for i, r in zip(ids, rs) if i in sugar))
check("encode(): LC4 threat channel is still driven",
      any(r > 0 for i, r in zip(ids, rs) if i in set(neurons.LC4_LOOMING)))

print(f"\n{passed} checks passed")
