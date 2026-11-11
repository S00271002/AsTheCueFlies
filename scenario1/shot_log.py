"""Every shot, written to CSV the moment it finishes.

Written to scenario1/runs/shots.csv
Each row is flushed and synced to disk straight away, so even if a run is
stopped or crashes, every completed shot is recorded. Failed shots are
recorded too, with the error message.
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

COLUMNS = [
    # who / when
    "shot_number", "run_id", "timestamp", "mode", "scenario", "seed",
    # brain: look phase
    "look_stimulus_hz", "look_duration_ms", "look_total_spikes", "look_active_neurons",
    # decision
    "option_index", "option_label", "best_option_index", "p_chosen",
    "p_best_before", "p_best_after", "probabilities_json",
    # the shot
    "target_ball", "pocket", "start_positions_json", "V0", "phi_deg",
    "aim_noise_deg", "theta_deg", "spin_a", "spin_b",
    # what happened
    "contacted_target", "first_ball_hit", "balls_potted", "target_potted",
    "cue_scratch", "target_closest_pocket", "target_min_dist_m",
    "target_start_dist_m", "target_final_dist_m", "final_positions_json",
    "shot_duration_s", "success",
    # taste and learning
    "taste", "taste_reason", "taste_strength", "taste_rate_hz", "taste_duration_ms",
    "reward_signal", "baseline_before", "advantage",
    "taste_sugar_spikes", "taste_bitter_spikes", "taste_mn9_spikes", "taste_total_spikes",
    "rolling_success_50", "shots_learned",
    # housekeeping
    "seconds_taken", "error",
]


class ShotLog:
    def __init__(self, folder, scenario_key, mode, config):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "shots.csv"
        self.scenario = scenario_key
        self.mode = mode
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

        if self.path.exists():
            with open(self.path, newline="") as f:
                header = next(csv.reader(f), [])
            if header != COLUMNS:
                # Keep the old file and start a new one with the new columns.
                old = self.path.with_name(f"shots_old_columns_{self.run_id}.csv")
                os.replace(self.path, old)
                print(f"Column layout changed; previous log kept as {old.name}")

        self.history = self._read_history()
        new_file = not self.path.exists()
        self.file = open(self.path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=COLUMNS)
        if new_file:
            self.writer.writeheader()
            self._sync()

        with open(self.folder / "runs.jsonl", "a") as f:
            f.write(json.dumps({"run_id": self.run_id, "mode": mode,
                                "started": datetime.now().isoformat(timespec="seconds"),
                                "first_shot_number": self.next_shot_number(),
                                **config}) + "\n")

    def _read_history(self):
        rows = []
        if self.path.exists():
            with open(self.path, newline="") as f:
                for row in csv.DictReader(f):
                    rows.append(row)
        return rows

    def next_shot_number(self):
        numbers = [int(r["shot_number"]) for r in self.history if r.get("shot_number", "").isdigit()]
        return (max(numbers) + 1) if numbers else 1

    def recent_successes(self, n=50):
        rows = [r for r in self.history if not r.get("error")][-n:]
        return [r.get("success") == "True" for r in rows]

    def write(self, row):
        row = {k: row.get(k, "") for k in COLUMNS}
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")
        row["run_id"] = self.run_id
        row["mode"] = self.mode
        row["scenario"] = self.scenario
        self.writer.writerow(row)
        self._sync()
        self.history.append({k: str(v) for k, v in row.items()})

    def _sync(self):
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.file.close()


class RunLock:
    """Stops two programs training scenario 1 at the same time."""

    def __init__(self, folder):
        self.path = Path(folder) / "training.lock"

    def __enter__(self):
        if self.path.exists():
            try:
                pid = int(self.path.read_text().strip())
                os.kill(pid, 0)
                raise SystemExit(
                    f"This scenario is already being trained (process {pid}). "
                    "Stop that window first.")
            except (ValueError, ProcessLookupError, PermissionError):
                pass  # stale lock from a closed run
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except OSError:
            pass
