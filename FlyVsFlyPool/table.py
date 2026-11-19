"""table.py - pool table geometry: where to aim, and which shots are possible.

Units are metres and degrees (pooltool's units). x runs across the table,
y along it. An aim angle of 0 points along +x, 90 along +y.
"""

import math

import numpy as np
import pooltool as pt

# ----------------------------------------------------------------------------
# Table measurements (read from pooltool's default 7-foot pool table)
# ----------------------------------------------------------------------------
TABLE = pt.Table.default()
WIDTH, LENGTH = TABLE.w, TABLE.l
POCKETS = {pid: np.array(p.center[:2]) for pid, p in TABLE.pockets.items()}  # pocket centres
R = pt.Ball.create("cue", xy=(0.1, 0.1)).params.R                         # ball radius
CUE = "cue"
MAX_CUT = 70.0      # cuts thinner than this many degrees are treated as impossible


# ----------------------------------------------------------------------------
# Small geometry helpers
# ----------------------------------------------------------------------------
def unit(v):
    """Direction of a vector, with length 1."""
    return np.asarray(v, dtype=float) / np.linalg.norm(v)


def ghost_ball(ball, pocket):
    """Where the cue ball must be when it hits `ball` to send it into `pocket`."""
    return ball - 2 * R * unit(pocket - ball)


def cut_angle(cue, ball, pocket):
    """How thin the cut is, in degrees. 0 = dead straight, 90 = impossible."""
    a = unit(ghost_ball(ball, pocket) - cue)     # cue ball's direction
    b = unit(pocket - ball)                      # object ball's direction
    return math.degrees(math.acos(float(np.clip(a @ b, -1, 1))))


def distance_to_line(p, a, b):
    """Shortest distance from point p to the line segment a-b."""
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1) if ab.any() else 0
    return float(np.linalg.norm(p - (a + t * ab)))


def aim_angle(start, end):
    """Aim direction in degrees from point `start` towards point `end`."""
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 360


# ----------------------------------------------------------------------------
# best_pocket: the easiest clear pocket for a ball (or None)
# ----------------------------------------------------------------------------
def best_pocket(positions, ball_id):
    cue, ball = positions[CUE], positions[ball_id]
    others = [xy for k, xy in positions.items() if k not in (CUE, ball_id)]
    best, best_cost = None, math.inf
    for pid, pocket in POCKETS.items():
        cut = cut_angle(cue, ball, pocket)
        if cut > MAX_CUT:
            continue                                   # too thin
        ghost = ghost_ball(ball, pocket)
        # Skip the pocket if another ball is in the way of either path.
        if any(distance_to_line(o, cue, ghost) < 2 * R or
               distance_to_line(o, ball, pocket) < 2 * R for o in others):
            continue
        # Longer shots and thinner cuts cost more. Cheapest wins.
        cost = (np.linalg.norm(ghost - cue) + np.linalg.norm(pocket - ball)) \
            / math.cos(math.radians(cut)) ** 2
        if cost < best_cost:
            best, best_cost = pid, cost
    return best


# ----------------------------------------------------------------------------
# shot_numbers: the 3 numbers (0 = bad, 1 = good) the fly is shown for a shot
# ----------------------------------------------------------------------------
def shot_numbers(positions, ball_id, pocket_id):
    cue, ball, pocket = positions[CUE], positions[ball_id], POCKETS[pocket_id]
    near_pocket = np.clip(1 - np.linalg.norm(pocket - ball) / 1.0, 0, 1)   # 1 = right by the pocket
    straight = max(0.0, math.cos(math.radians(cut_angle(cue, ball, pocket))))  # 1 = dead straight
    near_cue = np.clip(1 - np.linalg.norm(ball - cue) / 2.0, 0, 1)        # 1 = right by the cue ball
    return (float(near_pocket), float(straight), float(near_cue))


# ----------------------------------------------------------------------------
# shot_speed: how hard to hit (a rule of thumb, not physics)
# ----------------------------------------------------------------------------
def shot_speed(positions, ball_id, pocket_id):
    cue, ball, pocket = positions[CUE], positions[ball_id], POCKETS[pocket_id]
    # A cut passes on less speed, so hit harder (but don't divide by tiny numbers).
    transfer = max(math.cos(math.radians(cut_angle(cue, ball, pocket))), 0.35)
    speed = 0.9 + 0.5 * np.linalg.norm(ball - cue) + 1.3 * np.linalg.norm(pocket - ball) / transfer
    return float(min(speed, 4.5))     # metres per second, capped


# ----------------------------------------------------------------------------
# free_spot: can the cue ball be put down here?
# ----------------------------------------------------------------------------
def free_spot(positions, xy):
    margin = R + 0.01
    on_cloth = margin < xy[0] < WIDTH - margin and margin < xy[1] < LENGTH - margin
    clear = all(np.linalg.norm(xy - p) > 2.2 * R for k, p in positions.items() if k != CUE)
    return on_cloth and clear


# ----------------------------------------------------------------------------
# ball_in_hand_spot: after a foul, put the cue ball behind a ball for a straight pot
# ----------------------------------------------------------------------------
def ball_in_hand_spot(positions, ball_id):
    ball = positions[ball_id]
    for pid in sorted(POCKETS, key=lambda p: np.linalg.norm(POCKETS[p] - ball)):  # nearest pocket first
        for distance_back in (0.25, 0.4, 0.15):
            xy = ghost_ball(ball, POCKETS[pid]) - distance_back * unit(POCKETS[pid] - ball)
            if free_spot(positions, xy) and best_pocket({**positions, CUE: xy}, ball_id):
                return xy
    return None


# ----------------------------------------------------------------------------
# list_options: every shot the fly can choose from this turn
# ----------------------------------------------------------------------------
def list_options(positions, legal_balls, ball_in_hand):
    """One option per legal ball: which ball, which pocket, where the cue ball
    is, the aim, the speed, and the 3 numbers the fly is shown."""
    options = []
    for ball_id in legal_balls:
        pos = positions
        # Ball in hand: move the cue ball to a good spot for this ball first.
        if ball_in_hand:
            spot = ball_in_hand_spot(positions, ball_id)
            if spot is None:
                continue
            pos = {**positions, CUE: spot}
        pocket = best_pocket(pos, ball_id)
        if pocket is None:
            continue
        ghost = ghost_ball(pos[ball_id], POCKETS[pocket])
        options.append({"ball": ball_id, "pocket": pocket, "cue_xy": pos[CUE], "safety": False,
                        "aim": aim_angle(pos[CUE], ghost),
                        "speed": shot_speed(pos, ball_id, pocket),
                        "numbers": shot_numbers(pos, ball_id, pocket)})
    if options:
        return options

    # No clear pot anywhere: offer "safeties" - just hit a legal ball full on.
    cue = positions[CUE]
    for ball_id in legal_balls:
        pocket = min(POCKETS, key=lambda p: np.linalg.norm(POCKETS[p] - positions[ball_id]))
        options.append({"ball": ball_id, "pocket": pocket, "cue_xy": cue, "safety": True,
                        "aim": aim_angle(cue, positions[ball_id]), "speed": 2.0,
                        "numbers": shot_numbers(positions, ball_id, pocket)})
    return options


# ----------------------------------------------------------------------------
# break_shot: the opening shot (fixed, not chosen by the fly)
# ----------------------------------------------------------------------------
def break_shot(positions, rng):
    """Hit the front ball of the rack hard, with a tiny random wobble."""
    front_ball = min((k for k in positions if k != CUE), key=lambda k: positions[k][1])
    aim = aim_angle(positions[CUE], positions[front_ball]) + rng.uniform(-0.4, 0.4)
    return {"ball": front_ball, "pocket": None, "aim": aim % 360, "speed": 7.0}
