"""The training loop used by watch.py and train.py.

For each shot:
  1. Look   - stimulate the scenario's sensory neurons; record the brain.
  2. Choose - the decision layer turns the brain's activity into a choice.
  3. Shoot  - pooltool simulates the shot.
  4. Taste  - sugar for success, a mild bitter pulse for a bad miss, or
              nothing; the taste neurons are stimulated and the brain's
              response is recorded.
  5. Learn  - the decision layer updates from the taste.
  6. Log    - the shot is written to scenario1/runs/shots.csv.
"""

import argparse
import json
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

import scenario as ps
from decision import DecisionLayer
from shot_log import RunLock, ShotLog

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def parse_args(mode, description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--shots", type=int, default=0,
                    help="stop after this many shots (default: run until stopped)")
    ap.add_argument("--seed", type=int, default=0, help="base random seed")
    ap.add_argument("--learning-rate", type=float, default=0.5)
    ap.add_argument("--fresh", action="store_true",
                    help="start a new fly (keeps the old weights file as a backup)")
    if mode == "watch":
        ap.add_argument("--speed", type=float, default=1.0,
                        help="animation speed (2 = twice as fast)")
    return ap.parse_args()


def run(mode, args):
    scenario = ps.ChooseShot()
    scenario_key = ps.KEY
    folder = RUNS
    folder.mkdir(parents=True, exist_ok=True)
    weights_path = folder / "fly_decision_weights.npz"

    with RunLock(folder):
        if args.fresh and weights_path.exists():
            backup = weights_path.with_name(f"fly_decision_weights_backup_{time.strftime('%Y%m%d-%H%M%S')}.npz")
            weights_path.replace(backup)
            print(f"Previous weights kept as {backup.name}")

        from brain import TrainingBrain
        print(f"\n{scenario.title}")
        print("Loading the fly brain...")
        brain = TrainingBrain(scenario.sensory_ids)

        decision, resumed = DecisionLayer.load_or_new(
            weights_path, len(scenario.options), brain.size, learning_rate=args.learning_rate)
        print("Continuing the trained fly" if resumed else "Starting a new, untrained fly",
              f"({decision.shots_learned} shots learned so far)")

        config = {
            "scenario_title": scenario.title, "options": scenario.labels(),
            "best_option": scenario.best_option, "learning_rate": decision.learning_rate,
            "epsilon": decision.epsilon, "baseline_rate": decision.baseline_rate,
            "aim_noise_deg": scenario.aim_noise_deg, "speed_noise": scenario.speed_noise,
            "look_hz": ps.LOOK_HZ, "look_ms": ps.LOOK_MS, "sugar_hz": ps.SUGAR_HZ,
            "sugar_ms": ps.SUGAR_MS, "bitter_hz": ps.BITTER_HZ, "bitter_ms": ps.BITTER_MS,
            "bitter_reward": ps.BITTER_REWARD, "far_m": ps.FAR, "seed": args.seed,
            "layout": {k: v.tolist() for k, v in scenario.positions.items()},
        }
        log = ShotLog(folder, scenario_key, mode, config)
        recent = deque(log.recent_successes(50), maxlen=50)
        print(f"Logging every shot to {log.path.relative_to(HERE.parent)}")

        display = None
        if mode == "watch":
            from display import TrainingDisplay
            display = TrainingDisplay(scenario, speed=args.speed, history=log.history)

        save_every = 1 if mode == "watch" else 10
        done = 0
        failures = 0
        try:
            while not args.shots or done < args.shots:
                if display is not None and not display.is_open():
                    break
                shot_number = log.next_shot_number()
                row, visuals = play_one(scenario, brain, decision, shot_number,
                                        args.seed, recent, record=display is not None)
                log.write(row)
                done += 1
                if row["error"]:
                    failures += 1
                    print(f"Shot {shot_number}: ERROR {row['error']}")
                    if failures >= 5:
                        print("Five failed shots in a row; stopping.")
                        break
                    continue
                failures = 0
                print_shot(row)
                if decision.shots_learned % save_every == 0:
                    decision.save(weights_path)
                if display is not None:
                    display.show_shot(row, visuals)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            decision.save(weights_path)
            log.close()
            print(f"Saved the fly's learning ({decision.shots_learned} shots) and "
                  f"{done} new shot(s) to the log.")
            if display is not None:
                display.close()


def play_one(scenario, brain, decision, shot_number, base_seed, recent, record):
    started = time.perf_counter()
    seed = base_seed * 1_000_003 + shot_number
    rng = np.random.default_rng(seed)
    row = {"shot_number": shot_number, "seed": seed, "error": ""}
    visuals = {}
    try:
        # 1. Look
        look = brain.simulate([(brain.sensory, ps.LOOK_HZ)], ps.LOOK_MS,
                              seed=seed, record=record, reset=True)
        x_hat = decision.normalise(look.counts)

        # 2. Choose
        p, drive = decision.probabilities(x_hat)
        choice = decision.choose(p, rng)
        best = scenario.best_option

        # 3. Shoot
        shot = scenario.make_shot(choice, rng)
        system, result = ps.simulate(shot, with_tracks=True)

        # 4. Taste
        taste, strength, reason = scenario.judge(shot, result)
        rate, duration = ps.taste_stimulus(taste, strength)
        neurons = brain.sugar if taste == "sugar" else brain.bitter if taste == "bitter" else []
        taste_activity = brain.simulate([(neurons, rate)], duration or 30.0,
                                        record=record, reset=False)

        # 5. Learn
        reward = ps.reward_signal(taste, strength)
        baseline_before = decision.baseline
        advantage = decision.learn(x_hat, p, choice, reward)
        p_after, _ = decision.probabilities(x_hat)

        success = taste == "sugar" and strength >= 0.999
        recent.append(success)

        row.update({
            "look_stimulus_hz": ps.LOOK_HZ, "look_duration_ms": ps.LOOK_MS,
            "look_total_spikes": look.total(),
            "look_active_neurons": int((look.counts > 0).sum()),
            "option_index": choice, "option_label": scenario.options[choice]["label"],
            "best_option_index": "" if best is None else best,
            "p_chosen": round(float(p[choice]), 4),
            "p_best_before": "" if best is None else round(float(p[best]), 4),
            "p_best_after": "" if best is None else round(float(p_after[best]), 4),
            "probabilities_json": json.dumps([round(float(v), 4) for v in p_after]),
            "target_ball": shot["target"], "pocket": shot.get("pocket", ""),
            "start_positions_json": json.dumps(shot["positions"]),
            "V0": round(shot["V0"], 4), "phi_deg": round(shot["phi"], 4),
            "aim_noise_deg": round(shot["aim_noise"], 4), "theta_deg": shot["theta"],
            "spin_a": shot["a"], "spin_b": shot["b"],
            "contacted_target": result["contacted_target"],
            "first_ball_hit": result["first_hit"],
            "balls_potted": " ".join(result["potted"]),
            "target_potted": result["target_potted"], "cue_scratch": result["scratch"],
            "target_closest_pocket": result["target_closest_pocket"],
            "target_min_dist_m": round(result["target_min_dist"], 4),
            "target_start_dist_m": round(result["target_start_dist"], 4),
            "target_final_dist_m": round(result["target_final_dist"], 4),
            "final_positions_json": json.dumps({k: [round(c, 4) for c in v]
                                                for k, v in result["final"].items()}),
            "shot_duration_s": round(result["duration"], 3),
            "success": success,
            "taste": taste, "taste_reason": reason, "taste_strength": round(strength, 4),
            "taste_rate_hz": round(rate, 2), "taste_duration_ms": duration,
            "reward_signal": round(reward, 4), "baseline_before": round(baseline_before, 4),
            "advantage": round(advantage, 4),
            "taste_sugar_spikes": taste_activity.total(brain.sugar),
            "taste_bitter_spikes": taste_activity.total(brain.bitter),
            "taste_mn9_spikes": taste_activity.total(brain.mn9),
            "taste_total_spikes": taste_activity.total(),
            "rolling_success_50": round(sum(recent) / len(recent), 4),
            "shots_learned": decision.shots_learned,
        })
        if record:
            visuals = {"look": look, "taste": taste_activity, "p_before": p,
                       "p_after": p_after, "tracks": ps.tracks(system), "brain": brain,
                       "result": result, "shot": shot}
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    row["seconds_taken"] = round(time.perf_counter() - started, 2)
    return row, visuals


def print_shot(row):
    taste = {"sugar": "SUGAR ", "bitter": "bitter", "none": "  -   "}[row["taste"]]
    best = f"  p(best) {row['p_best_after']:.2f}" if row["p_best_after"] != "" else ""
    print(f"#{row['shot_number']:<5} {taste} {row['option_label']:<34} "
          f"{row['taste_reason']:<48} success(50) {row['rolling_success_50']:.0%}{best}")
