"""Phase 2: scripted "dumb" bot.

No brain involved yet — that's Phase 4+. This is a deterministic, scripted
policy that proves out the read-state -> decide -> send-action loop at
10 Hz, per README.md: "Bridge with a dumb bot: read state (position,
health, nearby NPCs/tiles), send actions (left, right, jump, use item),
scripted policy, logged at 10 Hz."

Policy, in priority order: flee a nearby hostile if health is critically
low, else chase and attack the nearest hostile NPC if one's close by, else
patrol back and forth. Auto-jumps over a one-tile obstacle ahead, using the
tile grid from get_state. All thresholds below are guesses, not tuned —
nothing here is meant to be a good fly-brain stand-in, just something that
reacts to the world.

Added after a first live test where the attack-only policy let health drop
to 6/100 without disengaging — there was no self-preservation at all.

Usage:
    python bridge/dumb_bot.py            # runs until Ctrl+C
    python bridge/dumb_bot.py --seconds 10   # runs for a fixed time, for smoke tests
"""
import argparse
import json
import socket
import time
from datetime import datetime
from pathlib import Path

HOST, PORT = "127.0.0.1", 7778
TICK_HZ = 10
TICK_S = 1.0 / TICK_HZ

ATTACK_RANGE = 400   # px: start chasing a hostile NPC within this 2D distance
ATTACK_VBAND = 200   # px: ignore hostiles more than this far above/below (different floor)
USE_RANGE = 60       # px: close enough horizontally to stop closing and attack
PATROL_TICKS = 30    # ~3s per leg when nothing else to react to

FLEE_HP_FRACTION = 0.3   # retreat instead of engaging once hp/max_hp drops below this
FLEE_RANGE = 600         # px: watch for a threat to flee a bit further out than we'd chase

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

RELEASE = {"left": False, "right": False, "jump": False, "use": False}


class DumbPolicy:
    """Deterministic and not-smart on purpose — see module docstring."""

    def __init__(self):
        self.patrol_dir = 1  # 1 = right, -1 = left
        self.patrol_countdown = PATROL_TICKS
        self.last_mode = "idle"  # for logging only, not sent to the game

    def decide(self, state):
        if not state.get("in_world"):
            self.last_mode = "not_in_world"
            return dict(RELEASE)

        player = state["player"]
        if player.get("dead") or player.get("ghost"):
            self.last_mode = "dead"
            return dict(RELEASE)
        npcs = state.get("npcs", [])
        low_health = player["max_hp"] > 0 and player["hp"] / player["max_hp"] < FLEE_HP_FRACTION

        if low_health:
            threat = self._nearest_hostile(npcs, player, FLEE_RANGE)
            if threat is not None:
                self.last_mode = "flee"
                action = self._flee(threat, player)
                action["jump"] = self._wants_to_jump(state, action)
                return action
            # hurt but nothing nearby to flee from — fall through as normal

        target = self._nearest_hostile(npcs, player, ATTACK_RANGE)
        if target is not None:
            self.last_mode = "attack"
            dx = target["x"] - player["x"]
            direction = 1 if dx >= 0 else -1
            close_enough = abs(dx) < USE_RANGE
            action = {
                "left": direction < 0 and not close_enough,
                "right": direction > 0 and not close_enough,
                "jump": False,
                "use": close_enough,
            }
        else:
            self.last_mode = "patrol"
            action = self._patrol()

        action["jump"] = self._wants_to_jump(state, action)
        return action

    @staticmethod
    def _flee(threat, player):
        """Move directly away from the threat. No "close enough to stop"
        condition — unlike attacking, fleeing keeps going while the threat
        that triggered it is still around."""
        dx = threat["x"] - player["x"]
        direction = -1 if dx >= 0 else 1
        return {
            "left": direction < 0,
            "right": direction > 0,
            "jump": False,
            "use": False,
        }

    def _patrol(self):
        self.patrol_countdown -= 1
        if self.patrol_countdown <= 0:
            self.patrol_dir *= -1
            self.patrol_countdown = PATROL_TICKS
        return {
            "left": self.patrol_dir < 0,
            "right": self.patrol_dir > 0,
            "jump": False,
            "use": False,
        }

    @staticmethod
    def _nearest_hostile(npcs, player, range_px):
        candidates = []
        for n in npcs:
            if not n.get("hostile"):
                continue
            dx = n["x"] - player["x"]
            dy = n["y"] - player["y"]
            if abs(dy) > ATTACK_VBAND:
                continue
            dist2 = dx * dx + dy * dy
            if dist2 <= range_px * range_px:
                candidates.append((dist2, n))
        if not candidates:
            return None
        return min(candidates, key=lambda pair: pair[0])[1]

    @staticmethod
    def _wants_to_jump(state, action):
        """Heuristic auto-jump: a solid tile at foot level directly ahead,
        with clearance one tile above it. Approximate — the tile grid is
        tile-granularity and this ignores the player's exact hitbox."""
        if not (action["left"] or action["right"]):
            return False
        tiles = state.get("tiles")
        if not tiles:
            return False

        width, height = tiles["width"], tiles["height"]
        solid = tiles["solid"]
        cx, cy = width // 2, height // 2  # grid is centred on the player
        ahead_x = cx + (1 if action["right"] else -1)
        if not (0 <= ahead_x < width):
            return False

        def is_solid(gx, gy):
            if not (0 <= gx < width and 0 <= gy < height):
                return False
            return bool(solid[gy * width + gx])

        foot_y = cy + 1  # approximating foot level as one tile below centre
        return is_solid(ahead_x, foot_y) and not is_solid(ahead_x, foot_y - 1)


class Bridge:
    def __init__(self):
        self.sock = socket.create_connection((HOST, PORT), timeout=5)
        self.reader = self.sock.makefile()

    def request(self, payload):
        self.sock.sendall((json.dumps(payload) + "\n").encode())
        line = self.reader.readline()
        if not line:
            raise ConnectionError("bridge closed the connection")
        return json.loads(line)

    def release(self):
        try:
            self.request({"cmd": "set_action", **RELEASE})
        except Exception:
            pass

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def run(duration_s=None):
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"dumb_bot_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    print(f"Logging to {log_path}")

    policy = DumbPolicy()
    tick = 0
    start = time.monotonic()

    with open(log_path, "w") as log_file:
        while duration_s is None or time.monotonic() - start < duration_s:
            try:
                bridge = Bridge()
            except OSError as e:
                print(f"Can't reach the bridge ({e}); is the mod loaded? Retrying in 2s.")
                time.sleep(2)
                continue

            try:
                while duration_s is None or time.monotonic() - start < duration_s:
                    loop_start = time.monotonic()

                    state = bridge.request({"cmd": "get_state"})
                    action = policy.decide(state)
                    bridge.request({"cmd": "set_action", **action})

                    log_file.write(json.dumps({
                        "tick": tick, "t": time.time(), "state": state,
                        "action": action, "mode": policy.last_mode,
                    }) + "\n")
                    log_file.flush()

                    if tick % TICK_HZ == 0:  # once a second
                        hp = (state.get("player") or {}).get("hp")
                        print(f"tick {tick:6d}  mode={policy.last_mode:12s}  hp={hp}  action={action}")

                    tick += 1
                    time.sleep(max(0.0, TICK_S - (time.monotonic() - loop_start)))
            except (ConnectionError, OSError) as e:
                print(f"Bridge connection lost ({e}); reconnecting...")
            finally:
                bridge.release()
                bridge.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=None,
                         help="run for a fixed duration instead of until Ctrl+C")
    args = parser.parse_args()

    try:
        run(duration_s=args.seconds)
    except KeyboardInterrupt:
        print("\nStopping, releasing keys...")


if __name__ == "__main__":
    main()
