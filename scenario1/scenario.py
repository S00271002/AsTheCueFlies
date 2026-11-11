"""Scenario 1 - choose the best shot: the table, the options, the physics
and the taste rules.

The cue ball sits on the head string. Ball 1 sits near a corner pocket (easy
pot); ball 2 sits in the middle of the table, far from every pocket (hard
pot). The fly chooses which ball to go for; aiming and speed are automatic.

Taste rules:
  * sugar  - either ball potted (cue ball stays on the table)
  * bitter - the cue ball missed the ball it was aiming for, or that ball
             never came within FAR metres of a pocket. Always mild and short.
  * none   - a scratch, or close but not potted
"""

import math

import numpy as np
import pooltool as pt

from brain import WATER

KEY = "scenario1_choose_shot"
TITLE = "Scenario 1 - choose the best shot"

FAR = 0.30            # metres; never got this close to a pocket -> bitter
SUGAR_HZ = 200.0      # full-strength sugar (the rate used in the paper)
SUGAR_MS = 50.0
BITTER_HZ = 40.0      # mild: low rate, short pulse
BITTER_MS = 30.0
BITTER_REWARD = -0.3  # learning signal for bitter
LOOK_HZ = 150.0
LOOK_MS = 100.0


# -- physics ----------------------------------------------------------------

POCKETED = 4  # pooltool motion state for a ball in a pocket
CUE = "cue"


def table_geometry():
    table = pt.Table.default()
    pockets = {pid: np.array(p.center[:2], dtype=float) for pid, p in table.pockets.items()}
    radius = pt.Ball.create(CUE, xy=(0.1, 0.1)).params.R
    return table.w, table.l, pockets, radius


def build_system(positions):
    balls = {bid: pt.Ball.create(bid, xy=tuple(xy)) for bid, xy in positions.items()}
    return pt.System(table=pt.Table.default(), balls=balls, cue=pt.Cue(cue_ball_id=CUE))


def simulate(shot, with_tracks=True):
    """shot: dict with positions, V0, phi, theta, a, b, target.

    Returns (system, result) where result describes the outcome.
    """
    system = build_system(shot["positions"])
    system.cue.set_state(V0=shot["V0"], phi=shot["phi"], theta=shot.get("theta", 0.0),
                         a=shot.get("a", 0.0), b=shot.get("b", 0.0))
    pt.simulate(system, inplace=True)
    if with_tracks:
        try:
            pt.continuize(system, dt=0.01, inplace=True)
        except Exception:
            pass
    return system, measure(system, shot)


def tracks(system):
    """Ball paths: {ball_id: (times, xy, motion_states)}."""
    out = {}
    for bid, ball in system.balls.items():
        data = None
        try:
            vec = ball.history_cts.vectorize()
            if vec is not None and len(vec[2]):
                rvw, s, t = vec
                data = (np.asarray(t, float), np.asarray(rvw)[:, 0, :2], np.asarray(s, int))
        except Exception:
            data = None
        if data is None:
            states = list(ball.history.states) or [ball.state]
            data = (np.array([st.t for st in states], float),
                    np.array([st.rvw[0][:2] for st in states], float),
                    np.array([st.s for st in states], int))
        out[bid] = data
    return out


def _event_kind(event):
    kind = getattr(event, "event_type", "")
    return str(getattr(kind, "value", kind)).lower()


def measure(system, shot):
    _, _, pockets, _ = table_geometry()
    target = shot["target"]
    start = {k: np.array(v, float) for k, v in shot["positions"].items()}

    potted = [bid for bid, b in system.balls.items() if b.state.s == POCKETED]
    final = {bid: np.array(b.state.rvw[0][:2], float) for bid, b in system.balls.items()}

    # Which ball did the cue ball touch first?
    first_hit = ""
    for event in sorted(system.events, key=lambda e: e.time):
        ids = tuple(getattr(event, "ids", ()))
        if "ball_ball" in _event_kind(event) and CUE in ids:
            first_hit = [i for i in ids if i != CUE][0]
            break
    target_moved = (target in potted
                    or np.linalg.norm(final[target] - start[target]) > 0.002)
    contacted = bool(first_hit == target or (first_hit == "" and target_moved)
                     or target_moved)

    # Closest the target got to any pocket during the shot.
    t, xy, _ = tracks(system).get(target, (None, np.array([start[target], final[target]]), None))
    xy = np.vstack([start[target], xy, final[target]])
    best_pid, best_d = None, math.inf
    for pid, c in pockets.items():
        d = float(np.min(np.linalg.norm(xy - c, axis=1)))
        if d < best_d:
            best_pid, best_d = pid, d
    if target in potted:
        best_d = 0.0
        best_pid = min(pockets, key=lambda k: np.linalg.norm(pockets[k] - final[target]))
    start_dist = min(float(np.linalg.norm(start[target] - c)) for c in pockets.values())
    final_dist = 0.0 if target in potted else min(
        float(np.linalg.norm(final[target] - c)) for c in pockets.values())

    return {
        "potted": potted,
        "target_potted": target in potted,
        "objects_potted": [b for b in potted if b != CUE],
        "scratch": CUE in potted,
        "first_hit": first_hit,
        "contacted_target": contacted,
        "target_closest_pocket": best_pid,
        "target_min_dist": best_d,
        "target_start_dist": start_dist,
        "target_final_dist": final_dist,
        "final": {k: v.tolist() for k, v in final.items()},
        "duration": float(max((e.time for e in system.events), default=0.0)),
    }


# -- aiming helpers ---------------------------------------------------------

def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def ghost_phi(cue_xy, target_xy, pocket_xy, radius):
    """Direction (degrees) to send the cue ball to pot target into pocket."""
    ghost = np.asarray(target_xy) - 2 * radius * unit(np.asarray(pocket_xy) - target_xy)
    d = ghost - cue_xy
    return math.degrees(math.atan2(d[1], d[0])) % 360


def cut_angle(cue_xy, target_xy, pocket_xy, radius):
    ghost = np.asarray(target_xy) - 2 * radius * unit(np.asarray(pocket_xy) - target_xy)
    a = unit(ghost - cue_xy)
    b = unit(np.asarray(pocket_xy) - target_xy)
    return math.degrees(math.acos(float(np.clip(a @ b, -1, 1))))


def seg_dist(p, a, b):
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


# -- the scenario -----------------------------------------------------------

def _rotate(v, degrees):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class Scenario:
    key = ""
    title = ""
    sensory_ids = []
    aim_noise_deg = 0.0      # random aiming error, like a real player
    speed_noise = 0.02       # +-2 % random speed error
    best_option = None       # known best option, if there is one

    def __init__(self):
        self.w, self.l, self.pockets, self.R = table_geometry()
        self.positions = {k: np.array(v, float) for k, v in self.layout().items()}
        self.options = self.make_options()

    # to override
    def layout(self): ...
    def make_options(self): ...
    def shot_for(self, option): ...
    def judge(self, shot, result): ...

    def labels(self):
        return [o["label"] for o in self.options]

    def make_shot(self, option_index, rng):
        shot = self.shot_for(self.options[option_index])
        noise = float(rng.normal(0, self.aim_noise_deg)) if self.aim_noise_deg else 0.0
        speed_factor = float(1 + rng.normal(0, self.speed_noise)) if self.speed_noise else 1.0
        shot["aim_noise"] = noise
        shot["phi"] = (shot["phi"] + noise) % 360
        shot["V0"] = float(shot["V0"] * speed_factor)
        shot["positions"] = {k: [float(v[0]), float(v[1])] for k, v in self.positions.items()}
        return shot

    def _bitter_or_none(self, result):
        if not result["contacted_target"]:
            return ("bitter", 1.0, "missed the target ball")
        if result["target_min_dist"] > FAR:
            return ("bitter", 1.0, f"target never came within {FAR:.2f} m of a pocket")
        return ("none", 0.0, "close but not potted")

    def best_pocket(self, ball_id):
        """Easiest pocket for a ball: short, small cut, clear path."""
        cue, ball = self.positions[CUE], self.positions[ball_id]
        others = [v for k, v in self.positions.items() if k not in (CUE, ball_id)]
        best, best_cost = None, math.inf
        for pid, pxy in self.pockets.items():
            cut = cut_angle(cue, ball, pxy, self.R)
            if cut > 70:
                continue
            ghost = ball - 2 * self.R * unit(pxy - ball)
            if any(seg_dist(o, cue, ghost) < 2 * self.R or seg_dist(o, ball, pxy) < 2 * self.R
                   for o in others):
                continue
            cost = (np.linalg.norm(ghost - cue) + np.linalg.norm(pxy - ball)) / math.cos(math.radians(cut)) ** 2
            if cost < best_cost:
                best, best_cost = pid, cost
        return best

    def pot_speed(self, ball_id, pocket):
        cue, ball, pxy = self.positions[CUE], self.positions[ball_id], self.pockets[pocket]
        cut = cut_angle(cue, ball, pxy, self.R)
        d_cue = np.linalg.norm(ball - cue)
        d_obj = np.linalg.norm(pxy - ball)
        transfer = max(math.cos(math.radians(cut)), 0.35)
        return float(min(0.9 + 0.5 * d_cue + 1.3 * d_obj / transfer, 4.5))


class ChooseShot(Scenario):
    """Scenario 1: cue ball on the head string; ball 1 sits near a corner
    pocket (easy pot), ball 2 sits in the middle of the table, far from every
    pocket (hard pot). The fly chooses which ball to go for; aiming and speed
    are automatic. Potting either ball earns sugar."""

    key = "scenario1_choose_shot"
    title = "Scenario 1 - choose the best shot"
    sensory_ids = WATER
    aim_noise_deg = 0.5
    best_option = 0

    def layout(self):
        cue = np.array([self.w / 2, self.l / 4])           # head string
        pocket = self.pockets["rt"]
        towards_cue = _rotate(unit(cue - pocket), -4)      # nearly straight in
        near = pocket + 0.22 * towards_cue
        far = np.array([self.w * 0.5, self.l * 0.63])      # middle of the table
        return {CUE: cue, "1": near, "2": far}

    def make_options(self):
        opts = []
        for bid, desc in (("1", "ball 1 (near a pocket)"), ("2", "ball 2 (far from pockets)")):
            pocket = self.best_pocket(bid)
            opts.append({"label": f"pot {desc}", "target": bid, "pocket": pocket})
        return opts

    def shot_for(self, o):
        cue, ball = self.positions[CUE], self.positions[o["target"]]
        return {"target": o["target"], "pocket": o["pocket"],
                "phi": ghost_phi(cue, ball, self.pockets[o["pocket"]], self.R),
                "V0": self.pot_speed(o["target"], o["pocket"]), "theta": 0.0, "a": 0.0, "b": 0.0}

    def judge(self, shot, r):
        if r["scratch"]:
            return ("none", 0.0, "scratch: cue ball went in")
        if r["objects_potted"]:
            return ("sugar", 1.0, "potted " + ", ".join(f"ball {b}" for b in r["objects_potted"]))
        return self._bitter_or_none(r)


def taste_stimulus(taste, strength):
    """Rate (Hz) and duration (ms) of the taste pulse given to the brain."""
    if taste == "sugar":
        return SUGAR_HZ * strength, SUGAR_MS
    if taste == "bitter":
        return BITTER_HZ, BITTER_MS
    return 0.0, 0.0


def reward_signal(taste, strength):
    if taste == "sugar":
        return float(strength)
    if taste == "bitter":
        return BITTER_REWARD
    return 0.0
