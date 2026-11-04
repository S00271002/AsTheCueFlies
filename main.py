import pooltool as pt

def play_shot(cue_speed=1.5):
    """Simulate and display the two-ball demo shot."""
    shot = pt.System(
        table=pt.Table.default(),
        balls={
            "cue": pt.Ball.create("cue", xy=(0.4, 0.4)),
            "1": pt.Ball.create("1", xy=(0.4, 0.8)),
        },
        cue=pt.Cue(cue_ball_id="cue"),
    )
    shot.cue.set_state(V0=cue_speed, phi=90, a=0, b=0, theta=0)
    pt.simulate(shot, inplace=True)
    pt.show(shot)


if __name__ == "__main__":
    play_shot()
