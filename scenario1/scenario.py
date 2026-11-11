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
        return ("bitter", 0.5, "close but not potted")

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
    """Scenario 1: the cue ball sits on the head string and two balls are
    placed at random on the table, in new positions every shot. The fly is
    shown how each ball lies (how close it is to a pocket, how straight the
    pot is, how far it is from the cue ball) and chooses which ball to go
    for; aiming and speed are automatic. Potting either ball earns sugar."""

    key = "scenario1_choose_shot"
    title = "Scenario 1 - choose the best shot"
    sensory_ids = WATER
    aim_noise_deg = 0.5
    best_option = None  # changes every shot

    def layout(self):
        return self.random_layout(np.random.default_rng(0))

    def random_layout(self, rng):
        """Cue ball on the head string; balls 1 and 2 anywhere, as long as
        they do not touch and each one has a pocket it can go into."""
        cue = np.array([self.w / 2, self.l / 4])
        margin = self.R + 0.03
        for _ in range(1000):
            balls = {CUE: cue}
            for bid in ("1", "2"):
                balls[bid] = np.array([rng.uniform(margin, self.w - margin),
                                       rng.uniform(margin, self.l - margin)])
            points = list(balls.values())
            if min(np.linalg.norm(a - b) for i, a in enumerate(points)
                   for b in points[i + 1:]) < 4 * self.R:
                continue
            self.positions = balls
            if all(self.best_pocket(bid) is not None for bid in ("1", "2")):
                return balls
        raise RuntimeError("Could not find a playable layout")

    def new_layout(self, rng):
        """Called before every shot: new ball positions and options."""
        self.positions = self.random_layout(rng)
        self.options = self.make_options()

    def make_options(self):
        return [{"label": f"pot ball {bid}", "target": bid, "pocket": self.best_pocket(bid)}
                for bid in ("1", "2")]

    def ball_features(self, bid):
        """How a ball lies, each from 0 (bad) to 1 (good)."""
        cue, ball = self.positions[CUE], self.positions[bid]
        pocket = self.pockets[self.best_pocket(bid)]
        return {
            "near_pocket": float(np.clip(1 - np.linalg.norm(pocket - ball) / 1.0, 0, 1)),
            "straight": float(math.cos(math.radians(cut_angle(cue, ball, pocket, self.R)))),
            "near_cue": float(np.clip(1 - np.linalg.norm(ball - cue) / 2.0, 0, 1)),
        }

    def look_stimuli(self, bid):
        """What the fly sees when it looks at one ball: 3 water neurons per
        feature, firing at 20 Hz (bad) to 200 Hz (good). The same 9 neurons
        are used for both balls; the fly looks at them one at a time."""
        features = self.ball_features(bid)
        return [(WATER[j * 3:j * 3 + 3], 20 + 180 * features[name])
                for j, name in enumerate(("near_pocket", "straight", "near_cue"))]

    def shot_for(self, o):
        cue, ball = self.positions[CUE], self.positions[o["target"]]
        return {"target": o["target"], "pocket": o["pocket"],
                "phi": ghost_phi(cue, ball, self.pockets[o["pocket"]], self.R),
                "V0": self.pot_speed(o["target"], o["pocket"]), "theta": 0.0, "a": 0.0, "b": 0.0}

        def judge(self, shot, r):
        if r["scratch"]:
            return ("bitter", 1.0, "scratch: cue ball went in")
        if r["target_potted"] and r["target_closest_pocket"] == shot["pocket"]:
            return ("sugar", 1.0, f"potted ball {shot['target']} in {shot['pocket']}")
        if r["target_potted"]:
            return ("none", 0.0, f"ball {shot['target']} went in {r['target_closest_pocket']}, "
                                 f"not the called pocket {shot['pocket']}")
        if r["objects_potted"]:
            return ("none", 0.0, "potted " + ", ".join(f"ball {b}" for b in r["objects_potted"])
                                 + f" instead of ball {shot['target']}")
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
