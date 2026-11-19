"""main.py - two fly brains go head to head at 8-ball. Start here!

    python main.py               # play a game and watch it
    python main.py --speed 3     # same, but 3x faster
    python main.py --no-watch    # no window, just text in the terminal
    python main.py --seed 7      # different rack, different brain noise

No training, no rewards. Each fly just goes for whatever shot its brain
reacts to most (the details are in brain.py). pooltool handles the physics
and the 8-ball rules, so the flies can't cheat.
"""

import argparse

import numpy as np
import pooltool as pt
from pooltool.ruleset.datatypes import BallInHandOptions
from pooltool.ruleset.eight_ball import BallGroup

import table as tb
from brain import Fly, load_connectome

MAX_SHOTS = 250     # if a game drags on this long we call it (most balls potted wins)


# ----------------------------------------------------------------------------
# balls_on_table: where every ball is that hasn't been potted yet
# ----------------------------------------------------------------------------
def balls_on_table(system):
    # Gives back {ball id: (x, y)}, skipping anything sitting in a pocket.
    return {ball_id: np.array(ball.state.rvw[0][:2])
            for ball_id, ball in system.balls.items()
            if ball.state.s != pt.constants.pocketed}


# ----------------------------------------------------------------------------
# group_name: turn pooltool's ball groups into words for the scoreboard
# ----------------------------------------------------------------------------
def group_name(balls):
    names = {BallGroup.SOLIDS: "solids", BallGroup.STRIPES: "stripes",
             BallGroup.EIGHT: "the 8", BallGroup.UNDECIDED: "open table"}
    return names[BallGroup.get(tuple(balls))]


# ============================================================================
# play_game: the actual game, from the break until somebody wins
# ============================================================================
def play_game(flies, seed, viewer):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)                      # pooltool's rack uses numpy's random, so seed that too

    # ---- Set the table: rack the balls, cue ball on the line at the bottom ----
    system = pt.System(table=pt.Table.default(),
                       balls=pt.get_rack(pt.GameType.EIGHTBALL, pt.Table.default()),
                       cue=pt.Cue(cue_ball_id="cue"))
    pt.ruleset.respot(system, "cue", tb.WIDTH / 2, tb.LENGTH / 4)

    # ---- Grab pooltool's 8-ball rules (Fly A breaks) ----
    game = pt.get_ruleset(pt.GameType.EIGHTBALL)(players=[pt.Player("Fly A"), pt.Player("Fly B")])
    groups = {"Fly A": "open table", "Fly B": "open table"}   # who's on solids/stripes (for the screen)

    shot = 0
    while True:
        shot += 1
        fly = flies[game.active_player.name]      # whose go is it? Fly A or Fly B
        rules = game.shot_constraints             # what the rules say they're allowed to do
        positions = balls_on_table(system)
        ball_in_hand = rules.ball_in_hand != BallInHandOptions.NONE   # did the other fly just foul?

        # ---- 1. Work out the shot ----
        if shot == 1:
            # The break. This one's fixed - the fly doesn't get a say.
            choice = tb.break_shot(positions, rng)
            options, chosen, mn9, downstream = [], None, [], []
            label = "break"
        else:
            # List every legal shot and let the fly's brain pick one.
            options = tb.list_options(positions, [b for b in rules.hittable if b in positions],
                                      ball_in_hand)
            chosen, mn9, downstream = fly.choose(options)
            choice = options[chosen]
            if ball_in_hand:                      # other fly fouled, so put the cue ball where we want it
                pt.ruleset.respot(system, "cue", *choice["cue_xy"])
            rules.ball_call, rules.pocket_call = choice["ball"], choice["pocket"]   # call it: "7 in the corner"
            label = ("safety on " if choice["safety"] else "") + \
                f"ball {choice['ball']} -> pocket {choice['pocket']}"

        # ---- 2. Take the shot (pooltool does all the physics) ----
        start_positions = balls_on_table(system)
        system.strike(V0=choice["speed"], phi=choice["aim"], theta=0.0, a=0.0, b=0.0)
        pt.simulate(system, inplace=True)

        # ---- 3. Let the rules decide: foul? whose turn? game over? ----
        game.process_and_advance(system)
        result = game.shot_info
        potted = [f"{e.ids[0]} in {e.ids[1]}" for e in system.events
                  if e.event_type == pt.EventType.BALL_POCKET]

        # Once someone's on solids or stripes, note it down for the scoreboard.
        next_group = group_name(game.shot_constraints.hittable)
        if not result.game_over and next_group in ("solids", "stripes"):
            other_group = "stripes" if next_group == "solids" else "solids"
            groups[game.active_player.name] = next_group
            groups[game.last_player.name] = other_group

        # ---- 4. Show what happened (terminal + window) ----
        result_text = "legal" if result.legal else f"FOUL: {result.reason}"
        print(f"[{shot:>3}] {fly.name}: {label:<28} potted: {', '.join(potted) or '-':<14} {result_text}")
        if viewer is not None:
            if not viewer.is_open():              # you closed the window, so we stop
                return
            viewer.show_shot(system, {
                "shot": shot, "fly": fly.name, "label": label, "ball": choice["ball"],
                "pocket": choice["pocket"], "speed": choice["speed"],
                "start_positions": start_positions, "options": options, "chosen": chosen,
                "mn9": mn9, "downstream": downstream, "potted": potted, "result": result_text,
                "groups": groups, "score": dict(result.score),
                "game_over": result.game_over,
                "winner": result.winner.name if result.winner else None})

        # ---- 5. Is it over? ----
        if result.game_over:
            print(f"*** {result.winner.name if result.winner else 'Nobody'} wins! ***")
            return
        if shot >= MAX_SHOTS:
            a, b = result.score.get("Fly A", 0), result.score.get("Fly B", 0)
            print(f"*** {MAX_SHOTS} shots: " + ("draw" if a == b else
                  ("Fly A" if a > b else "Fly B") + " wins on balls potted") + " ***")
            return
# ============================================================================
# end of play_game
# ============================================================================


# ----------------------------------------------------------------------------
# main: read the command-line options, load the brains, play
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Two fly brains play 8-ball.")
    parser.add_argument("--speed", type=float, default=1.5, help="playback speed (1 = real time)")
    parser.add_argument("--seed", type=int, default=0, help="changes the rack and the brain noise")
    parser.add_argument("--no-watch", action="store_true", help="no window, just text")
    args = parser.parse_args()

    # Load the brain wiring once - both flies use the same copy.
    print("Loading the fly brain...")
    connectome = load_connectome()

    # ======================== FLY A starts here ========================
    fly_a = Fly("Fly A", connectome, seed=[args.seed, 1])
    # ======================== FLY A ends here ==========================

    # ======================== FLY B starts here ========================
    fly_b = Fly("Fly B", connectome, seed=[args.seed, 2])
    # ======================== FLY B ends here ==========================
    # Same brain wiring, but different seeds, so each fly's brain noise is
    # its own. That's the only difference between them.

    flies = {"Fly A": fly_a, "Fly B": fly_b}

    # Open the window (unless you said --no-watch) and let them play.
    viewer = None
    if not args.no_watch:
        from viewer import Viewer
        viewer = Viewer(speed=args.speed)
    play_game(flies, args.seed, viewer)
    if viewer is not None:
        viewer.keep_open()                    # leave the final table up until you close it


if __name__ == "__main__":
    main()
