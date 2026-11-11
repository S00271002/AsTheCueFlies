"""Train the fly on Scenario 1 (choose the best shot), without graphics.

Runs until stopped (Ctrl+C or close the window). Every shot is logged to
scenario1/runs/shots.csv and the fly's learning is saved to
scenario1/runs/fly_decision_weights.npz, so the next run (watch or train)
carries on from where this one stopped.

    python scenario1/train.py               # run until stopped
    python scenario1/train.py --shots 500   # stop after 500 shots
    python scenario1/train.py --fresh       # start a new, untrained fly
"""
from loop import parse_args, run

if __name__ == "__main__":
    run("train", parse_args("train", __doc__))
