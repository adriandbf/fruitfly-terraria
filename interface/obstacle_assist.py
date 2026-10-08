"""Obstacle-ahead jump assist. NOT part of the fly's cognition.

Every other action in this project comes out of decoder.py reading the
fly brain's descending-neuron rates. This module is the exception: it
looks at the tile grid from get_state and adds a jump when the character
is walking into a step it can clear. That is a scripted assist layered on
top of the brain (ported and generalized from Phase 2's dumb_bot.py
heuristic, which only handled 1-tile steps), so:

  - it is a separate module, applied after decode() by run_fly.py, never
    inside the decoder;
  - run_fly.py logs `jump_source` ("brain" / "assist") every tick, so a
    jump the assist added is never credited to the fly;
  - run_fly.py takes --no-obstacle-assist to turn it off entirely.

A more faithful alternative, not built: make "obstacle ahead" a sensory
input (e.g. drive a looming-detector channel like LC4 by proximity) and
let the Giant Fiber jump emerge from the brain. Looming is arguably what a
wall is. That's a research task, not a port, and would also need its own
validation that walls and enemies being the same channel is acceptable.

Geometry, from the real tile grid (31x17, centered on the player's center
tile at (cx, cy) = (15, 8), seen in logs): the character's body occupies
rows cy-1..cy+1 and stands on the floor at row cy+2. An obstacle "ahead"
is the column at cx+/-1; its height is the run of solid tiles counted up
from row cy+1 (the lowest body row). The step is only clearable if that
height is <= max_jump_tiles AND there are `headroom_tiles` clear rows
above its top for the body to land into (the character is ~2.6 tiles
tall, so it needs a 3-tile gap).

max_jump_tiles=3 is a CONSERVATIVE GUESS, not a measurement -- nobody has
measured this character's actual jump height in this project. If steps of
that height turn out clearable (or not), change it from data. Likewise
hold_ticks=3 (~300ms at 10Hz) is meant to cover the ~15-frame (~250ms)
variable-height jump window in vanilla Terraria: releasing jump early cuts
the ascent short, which would drop the character back against the step.
"""

DEFAULT_PARAMS = {
    "max_jump_tiles": 3,
    "headroom_tiles": 3,
    "hold_ticks": 3,
}


class ObstacleAssist:
    def __init__(self, params=None):
        p = {**DEFAULT_PARAMS, **(params or {})}
        self.max_jump_tiles = p["max_jump_tiles"]
        self.headroom_tiles = p["headroom_tiles"]
        self.hold_ticks = p["hold_ticks"]
        self._hold = 0  # remaining ticks to keep jump held after a trigger

    def reset(self):
        self._hold = 0

    def apply(self, state, action):
        """Returns (action, assisted). `action` is not mutated. assisted is
        True only when THIS call is what made jump True -- if the brain
        already asked for a jump, it gets the credit and assisted is False.
        """
        action = dict(action)
        if action.get("jump"):
            self._hold = 0
            return action, False

        if self._hold > 0:
            self._hold -= 1
            action["jump"] = True
            return action, True

        if self._should_jump(state, action):
            self._hold = self.hold_ticks - 1
            action["jump"] = True
            return action, True

        return action, False

    def obstacle_height(self, tiles, direction):
        """Height in tiles of the solid run in the column directly ahead
        (direction +1 right / -1 left), counted up from the lowest body
        row, or 0 if the way is clear. Public so tests/diagnostics can
        inspect it."""
        w, h = tiles["width"], tiles["height"]
        solid = tiles["solid"]
        cx, cy = w // 2, h // 2
        col = cx + direction
        if not (0 <= col < w):
            return 0
        height = 0
        row = cy + 1
        while row >= 0 and solid[row * w + col]:
            height += 1
            row -= 1
        return height

    def _should_jump(self, state, action):
        if action.get("left") == action.get("right"):
            return False  # not walking (or both held, which decode() never does)
        player = state.get("player") or {}
        if not player.get("on_ground", True):
            return False  # can't start a jump in the air
        tiles = state.get("tiles")
        if not tiles or tiles.get("width", 0) < 3:
            return False

        direction = 1 if action.get("right") else -1
        height = self.obstacle_height(tiles, direction)
        if height == 0 or height > self.max_jump_tiles:
            return False

        w, h = tiles["width"], tiles["height"]
        solid = tiles["solid"]
        cx, cy = w // 2, h // 2
        col = cx + direction
        top_row = cy + 2 - height  # topmost solid row: run is cy+1 up to here
        for k in range(1, self.headroom_tiles + 1):
            row = top_row - k
            if row >= 0 and solid[row * w + col]:
                return False  # ceiling too low to land on top
        return True
