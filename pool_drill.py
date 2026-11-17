"""A repeatable one-ball pool drill for testing shot rewards."""

import argparse
from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite

import pooltool as pt

SOFT_SPEED = 1.0
FIRM_SPEED = 2.0
LAYOUTS = ("short", "long")
TRAIN_OFFSETS = (-0.024, -0.012, 0.0, 0.012, 0.024)
EVAL_OFFSETS = (-0.018, -0.006, 0.006, 0.018)


def drill_geometry(table, layout, offset=0.0):
    if layout == "short":
        y = table.pockets["lc"].center[1]
        return (0.85 + offset, y), (0.45 + offset, y), "lc"
    if layout == "long":
        return None, (0.7, 1.6 + offset), "lb"
    raise ValueError(f"Unknown drill layout: {layout}")


def pocket_distance(table, layout, offset):
    _, (x, y), pocket = drill_geometry(table, layout, offset)
    px, py = table.pockets[pocket].center[:2]
    return hypot(px - x, py - y)


@dataclass(frozen=True)
class DrillResult:
    potted: bool
    scratched: bool

    @property
    def reward(self) -> int:
        return int(self.potted and not self.scratched)


def simulate_drill_shot(
    speed: float, *, layout: str = "short", offset: float = 0.0
) -> tuple[pt.System, DrillResult]:
    """Reset the layout, take a straight shot, and score its pocket events."""
    if not isfinite(speed) or speed <= 0:
        raise ValueError("Cue speed must be a finite positive number in m/s.")

    table = pt.Table.default()
    cue_xy, object_xy, pocket = drill_geometry(table, layout, offset)
    pocket_x, pocket_y = table.pockets[pocket].center[:2]
    dx, dy = pocket_x - object_xy[0], pocket_y - object_xy[1]
    if cue_xy is None:
        distance = hypot(dx, dy)
        cue_xy = (object_xy[0] - 0.19 * dx / distance, object_xy[1] - 0.19 * dy / distance)
    angle = degrees(atan2(dy, dx)) % 360

    shot = pt.System(
        table=table,
        balls={
            "cue": pt.Ball.create("cue", xy=cue_xy),
            "1": pt.Ball.create("1", xy=object_xy),
        },
        cue=pt.Cue(cue_ball_id="cue"),
    )
    shot.strike(phi=angle, V0=speed, a=0, b=1, theta=0)
    pt.simulate(shot, inplace=True)

    pocketed_balls = {
        event.ids[0]
        for event in shot.events
        if event.event_type == pt.EventType.BALL_POCKET
    }
    result = DrillResult(
        potted="1" in pocketed_balls, scratched="cue" in pocketed_balls
    )
    return shot, result


def run_drill_shot(speed: float, *, layout: str = "short", show: bool = False) -> DrillResult:
    shot, result = simulate_drill_shot(speed, layout=layout)
    if show:
        pt.show(shot)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("speed", type=float, help="cue speed in m/s")
    parser.add_argument("--layout", choices=LAYOUTS, default="short")
    parser.add_argument("--show", action="store_true", help="open the shot replay")
    args = parser.parse_args()
    result = run_drill_shot(args.speed, layout=args.layout, show=args.show)
    print(
        f"Potted: {result.potted}; scratched: {result.scratched}; reward: {result.reward}"
    )
