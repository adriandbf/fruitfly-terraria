"""Character movement model, fitted to probe_movement.py on the real game
(Hallowed summoner, Empress Wings + Soaring Insignia, full game speed).
Shared by fake_bridge.py, the scripted dodger and the leash. Per game tick
(60/s), pixels. Probe v3 (2026-10-09, every test from the ground) is the
clean one; v1/v2 were spoiled by the world ceiling.

  ground run: top ~3.5 px/tick, ~0.4 s to reach it
  airborne sideways: up to ~8 px/tick (accel ~0.3) while flying; flying is
      >2x faster sideways than running
  jump+up ("up"): -6.4 hop, then climbs at ~0.5 px/tick per tick to -18.7
  jump only ("jump"): hop, then climbs at ~0.13 px/tick per tick, reaching
      -18.7 too after ~2.5 s. down+jump is the same: there is NO hover
  up only in the air ("glide"): slow fall, ~0.1 px/tick per tick
  nothing / down ("down"): normal gravity ~0.4 px/tick per tick, max fall 10;
      after letting go of a climb at -15, it coasts ~18 tiles higher.
      Down drops through planter rows (nothing lands on them).
Human demo (2026-10-09): airborne 93-98% of the time, constant sideways
movement (~7 px/tick), alternating diagonal climbs (side+jump) and dives
(side+down), ~340-455 px from the core.
"""
RUN_SPEED = 3.5
RUN_ACCEL = 1.0 / 6
AIR_SPEED = 8.0
AIR_ACCEL = 0.3
JUMP_VY = -6.4
CLIMB_ACCEL = 0.5            # jump+up
CLIMB_MAX = 18.7
SOFT_CLIMB_ACCEL = 0.13      # jump only (and down+jump)
GRAVITY = 0.4                # nothing / down
GLIDE_GRAVITY = 0.1          # up only
MAX_FALL = 10.0

MODES = ("up", "jump", "glide", "down")


def keys_for(h, mode):
    """(h in -1/0/1, mode) -> set_action keys."""
    return {"left": h < 0, "right": h > 0, "jump": mode in ("up", "jump"),
            "up": mode in ("up", "glide"), "down": mode == "down", "use": False}


def mode_of(action):
    if action.get("jump") and action.get("up"):
        return "up"
    if action.get("jump"):
        return "jump"
    if action.get("up"):
        return "glide"
    return "down"  # nothing and down fall alike (down also drops through planter rows)


def step(vx, vy, h, mode, on_ground):
    """One game tick. Returns (vx, vy)."""
    speed, accel = (RUN_SPEED, RUN_ACCEL) if on_ground else (AIR_SPEED, AIR_ACCEL)
    target = h * speed
    if vx < target:
        vx = min(target, vx + accel)
    elif vx > target:
        vx = max(target, vx - accel)
    if on_ground and mode in ("up", "jump"):
        vy = JUMP_VY
    elif mode == "up":
        vy = max(-CLIMB_MAX, vy - CLIMB_ACCEL)
    elif mode == "jump":
        vy = max(-CLIMB_MAX, vy - SOFT_CLIMB_ACCEL)
    elif mode == "glide":
        vy = min(MAX_FALL, vy + (GRAVITY if vy < 0 else GLIDE_GRAVITY))
    else:
        vy = min(MAX_FALL, vy + GRAVITY)
    return vx, vy


def coast_rise(vy):
    """Extra height (px) still gained after letting go while moving up at vy<0
    (probe v3: -15.2 coasted ~290 px, i.e. normal gravity)."""
    return vy * vy / (2 * GRAVITY) if vy < 0 else 0.0
