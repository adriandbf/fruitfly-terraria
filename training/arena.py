"""Phase 5: a fast, simulated test arena for CMA-ES training.

CMA-ES needs hundreds to thousands of evaluations -- completely
impractical against the live game (can't run thousands of episodes with a
human present). This is a lightweight, Python-only environment that
speaks the same get_state-shaped dicts as bridge/mod/FruitFlyBridge, so
interface/encoder.py and interface/decoder.py run UNCHANGED against it --
training tunes only their numeric parameters, never the game-interface
code itself.

Scope for this training pass, deliberately narrow: only the "approach"
behavior is scored, since it's the only one confirmed to produce any
decoder response at all (see CLAUDE.md's Phase 4 status -- the threat/
bitter pathway is confirmed dead live, and jump/DNp01 never fires from
either sensory channel). The arena still includes a hostile NPC some of
the time (so the threat channel gets exercised, in case a future neuron
swap makes it do something), but reward() only scores approach. Extend it
once a working threat pathway exists.

Physics are deliberately minimal: 1D horizontal movement only (no
gravity/tiles/jumping -- approach is purely a left/right decision, and
tiles aren't read by encoder.py/decoder.py at all, only Phase 2's retired
dumb_bot.py used them for auto-jump). "left"/"right" set a constant
velocity; "on_ground" is always True; the tile grid is a flat, all-clear
placeholder.
"""

WALK_SPEED_PX_PER_S = 150.0   # roughly Terraria's own walk speed
TICK_S = 0.1                  # 10 Hz control loop, matching CLAUDE.md


class Arena:
    def __init__(self, rng):
        """rng: a random.Random (or numpy Generator with .uniform/.choice) --
        pass one explicitly, don't rely on module-global randomness, so
        training can control reproducibility precisely."""
        self.rng = rng
        self.player_x = 0.0
        self.player_direction = 1
        self.human_x = 0.0
        self.hostile_x = None
        self.hostile_hp = 0.0

    def reset(self, human_side=None, hostile_present=None):
        """Randomize a fresh episode.
        human_side: force 'left'/'right' of the player, or None for random
            -- used to separately measure reward on the connectome's
            effective vs. weak side (see brain/neurons.py's docstring).
        hostile_present: force True/False, or None for random.
        Returns the initial get_state()-shaped dict.
        """
        self.player_x = 0.0
        self.player_direction = self.rng.choice([-1, 1])

        side = human_side or self.rng.choice(["left", "right"])
        dist = self.rng.uniform(200, 1500)
        self.human_x = self.player_x + (dist if side == "right" else -dist)

        want_hostile = (self.rng.random() < 0.5) if hostile_present is None else hostile_present
        if want_hostile:
            hdist = self.rng.uniform(100, 700)
            hside = self.rng.choice([-1, 1])
            self.hostile_x = self.player_x + hside * hdist
            self.hostile_hp = 25.0
        else:
            self.hostile_x = None

        return self.get_state()

    def get_state(self):
        players = [{"name": "human", "x": self.human_x, "y": 0.0, "hp": 100, "max_hp": 100}]
        npcs = []
        if self.hostile_x is not None:
            npcs.append({"type": 1, "name": "TestSlime", "x": self.hostile_x, "y": 0.0,
                         "hp": self.hostile_hp, "hostile": True})
        return {
            "ok": True,
            "in_world": True,
            "player": {
                "x": self.player_x, "y": 0.0, "direction": self.player_direction,
                "hp": 100, "max_hp": 100, "mana": 20, "max_mana": 20, "on_ground": True,
            },
            "npcs": npcs,
            "players": players,
            "tiles": {"origin_tx": 0, "origin_ty": 0, "width": 1, "height": 1,
                      "types": [-1], "solid": [False]},
        }

    def step(self, action):
        """action: {"left","right","jump","use"} bool dict, as decoder.py
        returns. Advances TICK_S of simulated time, returns the new state."""
        if action.get("right") and not action.get("left"):
            self.player_x += WALK_SPEED_PX_PER_S * TICK_S
            self.player_direction = 1
        elif action.get("left") and not action.get("right"):
            self.player_x -= WALK_SPEED_PX_PER_S * TICK_S
            self.player_direction = -1
        # both or neither held: no movement, facing unchanged (matches
        # decoder.py, which never sets both left and right at once)
        return self.get_state()

    def distance_to_human(self):
        return abs(self.human_x - self.player_x)
