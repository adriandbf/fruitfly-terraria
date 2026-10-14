"""The arena box the fly's senses use (scene3: the walls, floor and ceiling
loom like any other big thing), from loadout.json."""
import json
from pathlib import Path


def arena_px(loadout=None):
    """Half-width and climbable height (rows x spacing, capped by max_rise) in px."""
    if loadout is None:
        loadout = json.loads((Path(__file__).resolve().parent / "loadout.json").read_text())
    a = loadout.get("arena") or {}
    rise = min(a.get("rows", 4) * a.get("row_spacing_tiles", 12), a.get("max_rise_tiles") or 10 ** 6)
    return {"half_width_px": 16.0 * a.get("half_width_tiles", 100), "height_px": 16.0 * rise}
