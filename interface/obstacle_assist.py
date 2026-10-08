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

max_jump_tiles=5 is derived, not guessed -- from the game's own code
(decompiled tModLoader v2026.07.3.0, Terraria.Player): jumpHeight=15 frames
of jumpSpeed=5.01 px/frame while jump is held, then gravity 0.4/frame, so
the apex is ~15*5.01 + 5.01^2/(2*0.4) = ~106px = ~6.7 tiles. A step must be
<= ~6 tiles to clear; 5 leaves a tile of margin. Live evidence is partial
and noisy (interface/probe_step_clearing.py: 2/4/5-tall pillars cleared;
a "cleared 7" is physically impossible and was an artifact of natural
terrain), so this is a physics-based bound with loose live support, not a
tight measurement.

Two engine facts the first draft of this module got wrong:
  - A new jump only starts after `releaseJump`, which the game sets only on
    a frame where jump is NOT held (Player.Update: `else { jump = 0;
    releaseJump = true; }`). Holding jump continuously gives exactly one
    jump. So after the hold we force one jump=False tick (cooldown) before
    any re-trigger; without it a persistent obstacle re-triggered on the
    very next tick, jump stayed held forever, and a second step (or a
    second attempt) would never fire.
  - The jump only rises as far as the ceiling above the CHARACTER allows,
    so the player's own column must be clear for `height` rows above the
    head. The real 5-tall wall that stalled a live run had a cave ceiling
    two rows above the head; the old 3-tile cap declined it by luck.
Walking onto a 1-tile ledge needs no jump at all (Collision.StepUp), so
height-1 triggers are harmless but redundant.

hold_ticks=3 (~300ms at 10Hz) covers the 15-frame (~250ms) variable-height
window: releasing jump early cuts the ascent short.
"""

DEFAULT_PARAMS = {
    "max_jump_tiles": 5,
    "headroom_tiles": 3,
    "hold_ticks": 3,
}


class ObstacleAssist:
    def __init__(self, params=None):
        p = {**DEFAULT_PARAMS, **(params or {})}
        self.max_jump_tiles = p["max_jump_tiles"]
        self.headroom_tiles = p["headroom_tiles"]
        self.hold_ticks = p["hold_ticks"]
        self._hold = 0      # remaining ticks to keep jump held after a trigger
        self._cooldown = 0  # forced jump=False ticks after a hold (releaseJump)

    def reset(self):
        self._hold = 0
        self._cooldown = 0

    def apply(self, state, action):
        """Returns (action, assisted). `action` is not mutated. assisted is
        True only when THIS call is what made jump True -- if the brain
        already asked for a jump, it gets the credit and assisted is False.
        """
        action = dict(action)
        if action.get("jump"):
            self._hold = 0
            self._cooldown = 0
            return action, False

        if self._hold > 0:
            self._hold -= 1
            if self._hold == 0:
                self._cooldown = 1  # next tick must release jump
            action["jump"] = True
            return action, True

        if self._cooldown > 0:
            self._cooldown -= 1
            return action, False

        if self._should_jump(state, action):
            self._hold = self.hold_ticks - 1
            if self._hold == 0:
                self._cooldown = 1
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
        # The jump can only rise as far as the ceiling over the CHARACTER
        # allows: rows cy-2 up to cy-1-height (the body occupies cy-1..cy+1)
        # in the player's own column must be clear.
        for k in range(2, height + 2):
            row = cy - k
            if row >= 0 and solid[row * w + cx]:
                return False  # would hit its head before reaching the top
        return True
