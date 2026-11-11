"""Live window for watch.py.

Left:   the table, animated from above (the shot the fly chose).
Right:  the fly's choice (probability of each option, before -> after),
        the brain (look phase, then the taste pulse: orange = sugar,
        purple = bitter), and the learning curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

SENSE = "#59a14f"
BRAIN = "#4e79a7"
SUGAR = "#f28e2b"
BITTER = "#9c27b0"
BALL_COLOURS = {"cue": "white", "1": "gold", "2": "#1f5fbf"}
TASTE_TEXT = {"sugar": ("SUGAR", SUGAR), "bitter": ("mild bitter", BITTER),
              "none": ("no taste", "grey")}

def describe_shot(shot):
    """The pooltool cue settings for a shot, as text."""
    a, b = shot["a"], shot["b"]
    spin = "back spin" if b < 0 else "top spin" if b > 0 else "centre ball"
    side = "none" if a == 0 else ("right" if a > 0 else "left")
    return (f"POOLTOOL SHOT\n"
            f"Target   ball {shot['target']} -> pocket {shot.get('pocket') or '-'}\n"
            f"Power    V0    = {shot['V0']:.2f} m/s\n"
            f"Aim      phi   = {shot['phi']:.1f} deg (error {shot['aim_noise']:+.2f})\n"
            f"Spin     b     = {b:+.2f} ({spin})\n"
            f"Side     a     = {a:+.2f} ({side})\n"
            f"Cue tilt theta = {shot['theta']:.0f} deg")
class TrainingDisplay:
    def __init__(self, scenario, speed=1.0, history=()):
        self.sc = scenario
        self.speed = max(speed, 0.1)
        self.success_hist = [r.get("success") == "True" for r in history if not r.get("error")]
        self.taste_hist = [r.get("taste", "") for r in history if not r.get("error")]
        self.pbest_hist = [float(r["p_best_after"]) for r in history
                           if not r.get("error") and r.get("p_best_after")]

        plt.ion()
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title(scenario.title)
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1, 2.2], height_ratios=[1.1, 1.4, 1],
                                   hspace=0.55, wspace=0.15)
        self.ax_table = self.fig.add_subplot(gs[:2, 0])
        self.ax_shot = self.fig.add_subplot(gs[2, 0])
        self.ax_shot.axis("off")
        self.shot_info = self.ax_shot.text(0.0, 1.0, "", va="top", ha="left", fontsize=8,
                                           family="monospace", transform=self.ax_shot.transAxes)
        self.ax_choice = self.fig.add_subplot(gs[0, 1])
        self.ax_brain = self.fig.add_subplot(gs[1, 1])
        self.ax_learn = self.fig.add_subplot(gs[2, 1])
        self.headline = self.fig.suptitle("", fontsize=12)
        self._draw_table()
        self._draw_learning()
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "_closed", True))
        self._closed = False
        self._pause(0.1)

    # -- helpers ------------------------------------------------------------
    def is_open(self):
        return not self._closed and plt.fignum_exists(self.fig.number)

    def _pause(self, seconds):
        if self.is_open():
            plt.pause(max(seconds / self.speed, 0.001))

    def close(self):
        plt.ioff()
        if self.is_open():
            self.headline.set_text(self.headline.get_text() + "   (stopped - close this window)")
            plt.show()

    def _draw_table(self):
        ax, sc = self.ax_table, self.sc
        ax.clear()
        ax.add_patch(Rectangle((0, 0), sc.w, sc.l, color="#2e7d4f"))
        for pid, c in sc.pockets.items():
            ax.add_patch(Circle(c, 0.055, color="black", zorder=2))
        ax.plot([0, sc.w], [sc.l / 4] * 2, color="white", lw=0.6, alpha=0.5)  # head string
        self.balls = {}
        for bid, xy in sc.positions.items():
            patch = Circle(xy, sc.R, color=BALL_COLOURS.get(bid, "#c0392b"), ec="black",
                           lw=0.6, zorder=5)
            ax.add_patch(patch)
            self.balls[bid] = patch
        self.aim_line, = ax.plot([], [], color="white", ls="--", lw=1, zorder=4)
        self.pocket_ring = Circle((0, 0), 0.075, fill=False, color="yellow", lw=2, zorder=3,
                                  visible=False)
        ax.add_patch(self.pocket_ring)
        ax.set_xlim(-0.08, sc.w + 0.08)
        ax.set_ylim(-0.08, sc.l + 0.08)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(self.sc.title, fontsize=10)

    def _reset_balls(self):
        for bid, patch in self.balls.items():
            patch.center = tuple(self.sc.positions[bid])
            patch.set_visible(True)

    # -- one shot -------------------------------------------------------------
    def show_shot(self, row, v):
        if not self.is_open() or not v:
            return
        n = row["shot_number"]
        self._reset_balls()

        # 1. Look: the brain responds to the scene.
        self.headline.set_text(f"Shot {n}: the fly looks at the table...")
        self._draw_choice(v["p_before"], None)
        self._animate_brain(v, phase="look")

        # 2. Choice.
        self._draw_choice(v["p_before"], row["option_index"])
        self.headline.set_text(f"Shot {n}: the fly chooses  →  {row['option_label']}")
        shot = v["shot"]
        self.shot_info.set_text(describe_shot(shot))
        cue = np.array(self.sc.positions["cue"])
        phi = np.radians(shot["phi"])
        self.aim_line.set_data([cue[0], cue[0] + 0.5 * np.cos(phi)],
                               [cue[1], cue[1] + 0.5 * np.sin(phi)])
        if shot.get("pocket"):
            self.pocket_ring.center = tuple(self.sc.pockets[shot["pocket"]])
            self.pocket_ring.set_visible(True)
        else:
            self.pocket_ring.set_visible(False)
        self._pause(0.8)

        # 3. The shot.
        self.aim_line.set_data([], [])
        self._animate_shot(v["tracks"])

        # 4. Taste.
        text, colour = TASTE_TEXT[row["taste"]]
        self.headline.set_text(f"Shot {n}: {row['taste_reason']}  →  {text}")
        self.headline.set_color(colour if row["taste"] != "none" else "black")
        self._animate_brain(v, phase="taste", taste=row["taste"])

        # 5. Learning.
        self._draw_choice(v["p_after"], row["option_index"], before=v["p_before"])
        self.success_hist.append(bool(row["success"]))
        self.taste_hist.append(row["taste"])
        if row["p_best_after"] != "":
            self.pbest_hist.append(float(row["p_best_after"]))
        self._draw_learning()
        self._pause(1.2)
        self.headline.set_color("black")

    def _draw_choice(self, p, chosen, before=None):
        ax = self.ax_choice
        ax.clear()
        labels = self.sc.labels()
        y = np.arange(len(labels))
        colours = ["#cccccc"] * len(labels)
        if self.sc.best_option is not None:
            colours[self.sc.best_option] = "#a8d5a8"
        if chosen is not None:
            colours[chosen] = "#f4a3a5"
        ax.barh(y, p, color=colours, height=0.8)
        for k, label in enumerate(labels):
            ax.text(0.01, k, f"{label}   {p[k]:.0%}", va="center", fontsize=8 if len(labels) <= 3 else 7)
        if before is not None:
            ax.scatter(before, y, marker="|", color="black", s=80, zorder=3,
                       label="before this shot")
            ax.legend(loc="lower right", fontsize=7, frameon=False)
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Chance the fly picks each option", fontsize=8)
        title = "Fly's choice (red = chosen"
        title += ", green = best known option)" if self.sc.best_option is not None else ")"
        ax.set_title(title, fontsize=9, loc="left")

    def _animate_brain(self, v, phase, taste="none"):
        ax = self.ax_brain
        brain = v["brain"]
        look, tst = v["look"], v["taste"]
        gap = 5.0
        if phase == "look":
            ax.clear()
            sensory = set(brain.sensory)
            order = {}
            for t, n in zip(look.times_ms, look.neurons):
                order.setdefault(int(n), t)
            special = set(brain.sugar) | set(brain.bitter)
            rows = sorted(order, key=lambda n: (n not in sensory, order[n]))
            extra = [n for n in set(int(n) for n in tst.neurons) if n not in order]
            rows += sorted(extra, key=lambda n: (n not in special, n))
            self._row = {n: i for i, n in enumerate(rows)}
            self._nrows = max(len(rows), 1)
            ax.set_xlim(0, look.duration_ms + gap + max(tst.duration_ms, 30))
            ax.set_ylim(-1, self._nrows)
            ax.set_ylabel("Active neurons", fontsize=8)
            ax.set_xlabel("Brain time (ms): look at the table  |  taste", fontsize=8)
            ax.axvline(look.duration_ms + gap / 2, color="black", lw=0.8)
            ax.set_title("Fly brain (green = sensory input, blue = rest of brain, "
                         "orange = sugar, purple = bitter)", fontsize=9, loc="left")
            colours = np.where(np.isin(look.neurons, list(sensory)), SENSE, BRAIN)
            times, rows_y = look.times_ms, np.array([self._row[int(n)] for n in look.neurons])
            steps = 8
            for k in range(1, steps + 1):
                cut = look.duration_ms * k / steps
                m = times <= cut
                if k == 1:
                    self._look_dots = ax.scatter(times[m], rows_y[m], s=2, c=colours[m])
                else:
                    self._look_dots.set_offsets(np.column_stack([times[m], rows_y[m]]))
                    self._look_dots.set_color(colours[m])
                self._pause(0.08)
        else:
            offset = look.duration_ms + gap
            sugar, bitter = set(brain.sugar), set(brain.bitter)
            colours = np.array([SUGAR if int(n) in sugar else BITTER if int(n) in bitter else BRAIN
                                for n in tst.neurons])
            if len(tst.neurons):
                rows_y = np.array([self._row.get(int(n), 0) for n in tst.neurons])
                ax.scatter(tst.times_ms + offset, rows_y, s=2, c=colours)
            text, colour = TASTE_TEXT[taste]
            ax.text(offset + 2, self._nrows * 0.95, text, color=colour, fontsize=10,
                    va="top", fontweight="bold")
            self._pause(0.6)

    def _animate_shot(self, tracks):
        end = max(float(t[-1]) for t, _, _ in tracks.values()) if tracks else 0.0
        frames = int(np.clip(end * 25, 10, 160))
        for k in range(frames + 1):
            now = end * k / frames
            for bid, (t, xy, s) in tracks.items():
                patch = self.balls.get(bid)
                if patch is None or len(t) == 0:
                    continue
                i = min(np.searchsorted(t, now, side="right") - 1, len(t) - 1)
                i = max(i, 0)
                if len(t) > 1 and i < len(t) - 1:
                    f = (now - t[i]) / max(t[i + 1] - t[i], 1e-9)
                    pos = xy[i] + f * (xy[i + 1] - xy[i])
                else:
                    pos = xy[i]
                patch.center = tuple(pos)
                patch.set_visible(not (len(s) and s[i] == 4))
            self._pause(1 / 30)
            if not self.is_open():
                return

    def _draw_learning(self):
        ax = self.ax_learn
        ax.clear()
        n = len(self.success_hist)
        x = np.arange(1, n + 1)
        if n:
            s = np.array(self.success_hist, float)
            window = 20
            rolling = np.convolve(s, np.ones(window), "full")[:n] / np.minimum(x, window)
            ax.plot(x, rolling, color=SUGAR, label=f"success rate (last {window} shots)")
            tastes = np.array(self.taste_hist)
            ax.scatter(x[tastes == "sugar"], np.full((tastes == "sugar").sum(), 1.04),
                       s=6, color=SUGAR, marker="|")
            ax.scatter(x[tastes == "bitter"], np.full((tastes == "bitter").sum(), -0.04),
                       s=6, color=BITTER, marker="|")
        if self.pbest_hist:
            xb = np.arange(n - len(self.pbest_hist) + 1, n + 1)
            ax.plot(xb, self.pbest_hist, color="#2e7d4f", label="chance of picking the best option")
        ax.set_ylim(-0.08, 1.08)
        ax.set_xlim(0, max(n, 10) + 1)
        ax.set_xlabel("Shot number (all sessions)", fontsize=8)
        ax.set_title("Learning (ticks: orange = sugar, purple = bitter)", fontsize=9, loc="left")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="center right", fontsize=7, frameon=False)
