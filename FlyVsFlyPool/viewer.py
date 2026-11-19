"""viewer.py - the window you watch the game in (drawn with matplotlib).

On the left: the table from above. All 16 balls (stripes get a white band),
a dashed line showing where the shot's aimed, and a red ring round the
pocket that was called.

On the right: the scoreboard, what just happened, and a bar chart of how
hard the current fly's brain lit up for each shot it could've taken. The
coloured bar is the one it went for.
"""

import matplotlib.pyplot as plt
import numpy as np
import pooltool as pt
from matplotlib.patches import Circle, Rectangle

import table as tb

# ----------------------------------------------------------------------------
# Colours
# ----------------------------------------------------------------------------
CLOTH, RAIL = "#2e7d4f", "#6b3e1f"
BALL_COLOURS = {"1": "#f2c200", "2": "#1f4fbf", "3": "#d22020", "4": "#5b2a86",
                "5": "#f07b14", "6": "#1d7a3a", "7": "#7a1f1f", "8": "#111111"}
for n in range(9, 16):                       # stripes 9-15 borrow the colours of 1-7
    BALL_COLOURS[str(n)] = BALL_COLOURS[str(n - 8)]
BALL_COLOURS["cue"] = "#f8f8f0"
FLY_COLOURS = {"Fly A": "#c0392b", "Fly B": "#2471a3"}   # Fly A is red, Fly B is blue
FRAME_DT = 0.01                              # one animation frame per 0.01 s of game time


# ============================================================================
# Viewer: the whole window
# ============================================================================
class Viewer:
    # ------------------------------------------------------------------------
    # __init__: open the window and draw everything that never moves
    # ------------------------------------------------------------------------
    def __init__(self, speed=1.5):
        self.speed = speed
        self.closed = False
        plt.ion()                            # "interactive mode" so the game keeps going while the window's up
        self.fig = plt.figure(figsize=(12, 9.5))
        self.fig.canvas.manager.set_window_title("Fly vs Fly - 8-ball")
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "closed", True))

        # Split the window into 3 bits: table (left), text (top right), bar chart (bottom right).
        self.ax = self.fig.add_axes([0.01, 0.02, 0.46, 0.96])
        self.text_ax = self.fig.add_axes([0.5, 0.52, 0.48, 0.46])
        self.bar_ax = self.fig.add_axes([0.58, 0.06, 0.38, 0.40])
        self.text_ax.axis("off")

        # Draw the table: wooden edge, green cloth, the break line, black pockets.
        ax, rail = self.ax, 0.06
        ax.add_patch(Rectangle((-rail, -rail), tb.WIDTH + 2 * rail, tb.LENGTH + 2 * rail, color=RAIL))
        ax.add_patch(Rectangle((0, 0), tb.WIDTH, tb.LENGTH, color=CLOTH))
        ax.plot([0, tb.WIDTH], [tb.LENGTH / 4] * 2, color="white", alpha=0.25, lw=1)
        for pocket in tb.TABLE.pockets.values():
            ax.add_patch(Circle(pocket.center[:2], pocket.radius, color="black"))

        # Build every ball out of a few pieces: the ball itself, a coloured band
        # (stripes only), a little white circle with the number on, and a faint
        # trail that shows where it's been.
        self.balls = {}
        for ball_id, colour in BALL_COLOURS.items():
            striped = ball_id.isdigit() and int(ball_id) >= 9
            disc = ax.add_patch(Circle((0, 0), tb.R, facecolor="white" if striped else colour,
                                       edgecolor="black", lw=0.6, zorder=5))
            band = ax.add_patch(Rectangle((0, 0), 2 * tb.R, 1.2 * tb.R, facecolor=colour,
                                          lw=0, zorder=6)) if striped else None
            spot = ax.add_patch(Circle((0, 0), 0.45 * tb.R, facecolor="white", lw=0, zorder=7)) \
                if ball_id != "cue" else None
            number = ax.text(0, 0, ball_id, ha="center", va="center", fontsize=5.5,
                             fontweight="bold", zorder=8) if ball_id != "cue" else None
            trail = ax.plot([], [], color=colour if ball_id != "8" else "#999999",
                            alpha=0.35, lw=1.2, zorder=3)[0]
            self.balls[ball_id] = (disc, band, spot, number, trail)

        # The dashed aim line, and the red ring for the called pocket.
        self.aim_line = ax.plot([], [], "--", color="white", alpha=0.75, lw=1, zorder=4)[0]
        self.pocket_ring = ax.add_patch(Circle((0, 0), 0.075, fill=False, edgecolor="red",
                                               lw=2, zorder=4, visible=False))

        # Keep the table the right shape and hide the graph axes.
        ax.set_aspect("equal")
        ax.set_xlim(-rail - 0.1, tb.WIDTH + rail + 0.1)
        ax.set_ylim(-rail - 0.05, tb.LENGTH + rail + 0.05)
        ax.axis("off")

        # The text panel, plus the big "X wins!" banner for the end.
        self.text = self.text_ax.text(0, 1, "", va="top", fontsize=10, family="monospace",
                                      transform=self.text_ax.transAxes)
        self.banner = self.fig.text(0.24, 0.5, "", ha="center", fontsize=26, fontweight="bold",
                                    color="white", bbox=dict(facecolor="black", alpha=0.7, lw=0))
        self.pause(0.1)

    # ------------------------------------------------------------------------
    # Little window helpers
    # ------------------------------------------------------------------------
    def is_open(self):
        # True until you close the window.
        return not self.closed and plt.fignum_exists(self.fig.number)

    def pause(self, seconds):
        # Redraw and wait a moment (this is also what keeps the window responsive).
        if self.is_open():
            plt.pause(max(seconds, 0.001))

    def keep_open(self):
        # Game's over - leave the window up until you close it yourself.
        plt.ioff()
        if self.is_open():
            plt.show()

    # ------------------------------------------------------------------------
    # move_ball: move one ball (and all its bits) to a new spot
    # ------------------------------------------------------------------------
    def move_ball(self, ball_id, xy, visible):
        disc, band, spot, number, _ = self.balls[ball_id]
        disc.center = xy
        disc.set_visible(visible)
        if band is not None:
            band.set_xy((xy[0] - tb.R, xy[1] - 0.6 * tb.R))
            band.set_clip_path(disc)          # trim the band so it only shows inside the ball
            band.set_visible(visible)
        if spot is not None:
            spot.center = xy
            spot.set_visible(visible)
            number.set_position(xy)
            number.set_visible(visible)

    # ------------------------------------------------------------------------
    # draw_text: the scoreboard and the "what just happened" bit
    # ------------------------------------------------------------------------
    def draw_text(self, s):
        lines = [f"Shot {s['shot']}", ""]
        for name in ("Fly A", "Fly B"):
            arrow = ">" if name == s["fly"] else " "      # arrow points at whoever's shooting
            lines.append(f"{arrow} {name}  {s['groups'][name]:<11} potted {s['score'].get(name, 0)}")
        lines += ["", f"{s['fly']}: {s['label']}"]
        if s["chosen"] is not None:
            lines += [f"  {len(s['options'])} options shown to its brain",
                      f"  chosen one fired MN9 {s['mn9'][s['chosen']]}x, "
                      f"downstream {s['downstream'][s['chosen']]} spikes"]
        lines += [f"  speed {s['speed']:.1f} m/s", "",
                  f"Potted: {', '.join(s['potted']) or 'nothing'}", f"Result: {s['result']}"]
        self.text.set_text("\n".join(lines))
        self.text.set_color(FLY_COLOURS[s["fly"]])          # text goes red for A, blue for B

    # ------------------------------------------------------------------------
    # draw_bars: the bar chart of how much the brain liked each option
    # ------------------------------------------------------------------------
    def draw_bars(self, s):
        ax = self.bar_ax
        ax.clear()
        if s["chosen"] is None:                    # the break - no brain involved
            ax.set_title("The break (a fixed shot, not chosen by the fly)", fontsize=9)
            ax.axis("off")
            return
        ax.axis("on")
        labels = [("safety " if o["safety"] else "") + f"{o['ball']}->{o['pocket']}"
                  for o in s["options"]]
        colours = ["#bbbbbb"] * len(labels)
        colours[s["chosen"]] = FLY_COLOURS[s["fly"]]   # colour in the one it picked
        rows = np.arange(len(labels))
        ax.barh(rows, s["downstream"], color=colours)
        for i, m in enumerate(s["mn9"]):               # stick a star on anything that made MN9 fire
            if m > 0:
                ax.text(s["downstream"][i], i, f"  * MN9 {m}", va="center", fontsize=7)
        ax.set_yticks(rows, labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("downstream spikes in 100 ms", fontsize=8)
        ax.set_title(f"{s['fly']}'s brain response to each option", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)

    # ------------------------------------------------------------------------
    # show_shot: play back a shot that pooltool has already worked out
    # ------------------------------------------------------------------------
    def show_shot(self, system, s):
        if not self.is_open():
            return
        self.draw_text(s)
        self.draw_bars(s)

        # Draw the aim line (cue ball to where it should hit) and ring the called pocket.
        start = s["start_positions"]
        if s["pocket"] is not None:
            ghost = tb.ghost_ball(start[s["ball"]], tb.POCKETS[s["pocket"]])
            self.aim_line.set_data([start["cue"][0], ghost[0]], [start["cue"][1], ghost[1]])
            self.pocket_ring.center = tb.POCKETS[s["pocket"]]
            self.pocket_ring.set_visible(True)
        else:                                  # the break has no called pocket
            self.aim_line.set_data([], [])
            self.pocket_ring.set_visible(False)

        # pooltool only remembers where balls were at collisions, so ask it to
        # fill in a position every 0.01 s so we get a smooth animation.
        pt.continuize(system, dt=FRAME_DT, inplace=True)
        paths, states = {}, {}
        for ball_id, ball in system.balls.items():
            rvw, state, _ = ball.history_cts.vectorize()
            paths[ball_id], states[ball_id] = rvw[:, 0, :2], state
        n_frames = min(len(p) for p in paths.values())
        for parts in self.balls.values():
            parts[4].set_data([], [])               # wipe the trails from the last shot

        # Play it back at roughly 30 frames a second (skipping frames to keep up).
        step = max(1, int(round(self.speed / (30 * FRAME_DT))))
        for i in list(range(0, n_frames, step)) + [n_frames - 1]:
            if not self.is_open():
                return
            for ball_id in paths:
                in_pocket = states[ball_id][i] == pt.constants.pocketed
                self.move_ball(ball_id, paths[ball_id][i], visible=not in_pocket)   # potted balls vanish
                path = paths[ball_id][: i + 1]
                if np.ptp(path, axis=0).max() > 1e-4:   # only draw trails for balls that actually moved
                    self.balls[ball_id][4].set_data(path[:, 0], path[:, 1])
            self.pause(step * FRAME_DT / self.speed)

        # Game over? Show who won. Otherwise hang on the final position for a sec.
        if s["game_over"]:
            self.banner.set_text(f"{s['winner'] or 'Nobody'} wins!")
            self.pause(3.0)
        else:
            self.pause(0.8 / self.speed)
# ============================================================================
# end of Viewer
# ============================================================================
