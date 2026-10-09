"""The fly's tunable interface knobs for the Moon Lord fight, in one place.

Shared by train_fly_cma.py (CMA-ES against the fake fight) and imitate.py
(imitation warm start from logged live fights). Only the INTERFACE is
tuned -- how game state becomes sensory rates (encoder), how descending-
neuron rates become keys (decoder), and FlyPolicy's few thresholds. The
connectome's weights are never touched, and neither is which neurons are
used (brain/neurons.py).

A params file has three blocks, all optional:
    {"encoder": {...}, "decoder": {...}, "policy": {...}}
harness.py --params-file reads all three (policy = FlyPolicy kwargs).

Search space: CMA-ES works in [0,1]^N; PARAM_SPEC maps each unit value to
its real range. Integer knobs are rounded.
"""
import json
from pathlib import Path

# (block, key, low, high, integer?)
PARAM_SPEC = [
    ("encoder", "max_rate_hz", 50.0, 300.0, False),
    ("encoder", "threat_max_range_px", 200.0, 2000.0, False),
    ("encoder", "contra_fraction", 0.0, 1.0, False),
    ("decoder", "steer_margin_hz", 0.0, 30.0, False),
    ("decoder", "forward_threshold_hz", 0.0, 100.0, False),
    ("decoder", "backward_threshold_hz", 0.0, 100.0, False),
    ("decoder", "jump_threshold_hz", 0.0, 100.0, False),
    ("decoder", "flee_on_hz", 5.0, 80.0, False),
    ("decoder", "flee_off_hz", 2.0, 80.0, False),    # clipped to <= flee_on_hz
    ("decoder", "flee_hold_ticks", 0, 15, True),
    ("policy", "threat_horizon_ticks", 20, 200, True),
    ("policy", "commit_ticks", 0, 30, True),
    ("policy", "dive_on_hz", 10.0, 150.0, False),
    ("policy", "above_px", 0.0, 300.0, False),
    ("policy", "keep_away_px", 0.0, 1500.0, False),   # 0 = off (the core looms only within threat range)
]

# Hand-picked starting point (the live "fly brain" setup before tuning):
# the module defaults, plus FlyPolicy's defaults.
POLICY_DEFAULTS = {"threat_horizon_ticks": 90, "commit_ticks": 0, "dive_on_hz": 40.0, "above_px": 60.0, "keep_away_px": 0.0}


def _defaults():
    import encoder
    import decoder
    return {"encoder": encoder.DEFAULT_PARAMS, "decoder": decoder.DEFAULT_PARAMS, "policy": POLICY_DEFAULTS}


def unit_to_params(unit):
    out = {"encoder": {}, "decoder": {}, "policy": {}}
    for (block, key, lo, hi, is_int), u in zip(PARAM_SPEC, unit):
        u = min(1.0, max(0.0, float(u)))
        v = lo + u * (hi - lo)
        out[block][key] = int(round(v)) if is_int else round(v, 4)
    d = out["decoder"]
    d["flee_off_hz"] = min(d["flee_off_hz"], d["flee_on_hz"])
    return out


def params_to_unit(params):
    """Params dict (missing keys = defaults) -> unit vector."""
    defaults = _defaults()
    unit = []
    for block, key, lo, hi, _ in PARAM_SPEC:
        v = (params.get(block) or {}).get(key, defaults[block][key])
        unit.append(min(1.0, max(0.0, (float(v) - lo) / (hi - lo))))
    return unit


def default_unit():
    return params_to_unit({})


def policy_kwargs(params, vertical="brain"):
    """Params dict -> FlyPolicy(**kwargs)."""
    return {"enc_params": dict(params.get("encoder") or {}), "dec_params": dict(params.get("decoder") or {}),
            "vertical": vertical, **(params.get("policy") or {})}


def load(path):
    return json.loads(Path(path).read_text())


def save(path, params, **meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({**params, **({"meta": meta} if meta else {})}, indent=2) + "\n")
