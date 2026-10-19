import pooltool as pt

table = pt.Table.default()

balls = {
    "cue": pt.Ball.create("cue", xy=(0.4, 0.4)),
    "1": pt.Ball.create("1", xy=(0.4, 0.8)),
}

cue = pt.Cue(cue_ball_id="cue")

shot = pt.System(table=table, balls=balls, cue=cue)

shot.cue.set_state(V0=1.5, phi=90, a=0, b=0, theta=0)

pt.simulate(shot, inplace=True)
pt.show(shot)
