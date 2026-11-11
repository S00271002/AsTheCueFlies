"""Watch the fly play Scenario 1 (choose the best shot) in a loop.

Every shot still trains the fly and is logged to scenario1/runs/shots.csv.
Close the window or press Ctrl+C to stop.

    python scenario1/watch.py               # keep going
    python scenario1/watch.py --speed 2     # faster animation
    python scenario1/watch.py --shots 20    # stop after 20 shots
"""
from loop import parse_args, run

if __name__ == "__main__":
    run("watch", parse_args("watch", __doc__))
