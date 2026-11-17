This project will try and connect the brain of a fruit fly to a pool simulation. 

## Tools and technologies

- **Python** — main programming language.
- **pooltool** — simulates pool physics and displays shot replays.
- **Eon Systems fly-brain** — fruit-fly neural simulation based on the FlyWire connectome.
- **PyTorch with CUDA** — runs the neural simulation on an NVIDIA GPU.
- **NumPy and SciPy** — numerical calculations.
- **pandas and PyArrow** — read and analyse recorded neural activity.
- **Matplotlib** — plotting and visualising results.
- **WSL 2 with Ubuntu 22.04** — Linux environment for the fly simulation on Windows.
- **Miniconda / Conda** — manages the fly simulation’s Python environment.
- **Python venv** — isolates the Windows pooltool environment.
- **Git and GitHub** — version control and project hosting.
- **VS Code** — code editor.

## Environments

- **Pool simulation:** Windows, Python 3.11.9, `.venv`.
- **Fly simulation:** Ubuntu through WSL, Python 3.10, Conda environment `brain-fly`.

Package versions are recorded in `requirements.txt` and
`environment-brain.yml`.

## Run the demos

From the project folder in Ubuntu/WSL:

```bash
conda activate brain-fly
python main.py              # Fixed-speed pool shot
python fly_controller.py    # Six fly-controlled drill trials with sugar feedback
python view_spikes.py        # Replay the recorded sugar-stimulation spikes
```

## First training drill

`pool_drill.py` resets a two-ball layout each time and scores the shot from
pooltool's pocket events. A clean pot earns reward `1`; a miss or scratch
earns `0`. The pool-only script does not stimulate the brain.

```bash
python pool_drill.py 1.0 --layout short --show  # Clean pot
python pool_drill.py 2.0 --layout short         # Pot and scratch
python pool_drill.py 1.0 --layout long          # Miss
python pool_drill.py 2.0 --layout long          # Clean pot
```

Omit `--show` to run without a window.

The fly controller uses a separate cue before the shot, then reads spikes
from two fixed groups to choose a speed. A clean pot triggers a 100 ms sugar
pulse in the same trial. The neural state resets between shots. By default
the synaptic weights stay fixed, so this is a feedback demo.

```bash
python fly_controller.py --trials 6
python fly_controller.py --trials 3 --show
```

The controller prints the choice, shot outcome, and sugar feedback for each
trial. With `--show`, close the replay to continue to the next shot.

To try learning:

```bash
python fly_controller.py --learn --trials 12 --eval-trials 8
```

A clean pot triggers both the sugar pulse and a software reward signal that
strengthens active excitatory connections into the chosen readout group.
The decoder stays fixed. The before/after checks use the same random seeds,
with no sugar pulses or weight updates. Omit `--learn` for the frozen
control. Updated weights live in memory unless saved as a checkpoint; the
connectome files are untouched. This is an engineered learning rule, not a model of
biological sugar-driven plasticity.

To check whether one model can learn both layouts, use a balanced, shuffled
mix of short and long shots. Each layout activates a different input neuron,
but both use the same neural model and fixed decoder. Evaluation reports each
layout separately, with no sugar or updates.

```bash
python fly_controller.py --mixed --learn --trials 20 --eval-trials 8 --seed 42
python fly_controller.py --mixed         --trials 20 --eval-trials 8 --seed 42
```

Look for `P(firm)` to fall on the short layout and rise on the long one.
Picking one favourite speed for both layouts is not layout-dependent learning.

In the seed-42 run above, mean `P(firm)` moved from 0.382 to 0.263
on the short layout and from 0.671 to 0.773 on the long layout.
The frozen control's probabilities did not move. Both runs scored 6/8
clean pots on each layout before and after, so this run demonstrates
a change in choice probabilities, with no measured gain in clean pots.

## Save and resume learning

```bash
python fly_controller.py --mixed --learn --trials 20 --save fly-mixed.json
python fly_controller.py --mixed --learn --trials 20 --load fly-mixed.json --save fly-mixed.json
python fly_controller.py --mixed --trials 0 --eval-trials 8 --load fly-mixed.json
```

The checkpoint stores modified synapses, the fixed readout, the completed
trial count, and random generator states. Keep the same seed, drill mode,
speeds, and device type when loading. Resuming starts a new shuffled block
with the saved weights; it does not recreate the schedule of a longer
uninterrupted run. The final command evaluates without any training shots.

## Train on varied positions

Use `--varied` to train on ten starting positions and evaluate on eight
separate positions. The controller encodes object-ball distance to the
target pocket as the relative firing rates of two input neurons. Pooltool
still sets the aim; the fly chooses only between the two cue speeds.

```bash
python fly_controller.py --varied --learn --trials 100 --save-every 10 --save fly-varied.json
python fly_controller.py --varied --trials 0 --eval-trials 24 --load fly-varied.json
```

To continue, add `--load fly-varied.json` to the first command. Use a fresh
checkpoint for this curriculum; earlier fixed-layout checkpoints are
incompatible. If interrupted, resume from the last completed save. Compare
against a frozen run on the same seed (omit `--learn` and use a separate
checkpoint filename). Report short and long clean-pot rates separately.

To test whether learning on one layout interferes with the other, start
two fresh runs with the same seed. Each trains on only one layout and
evaluates both, using the usual held-out positions:

```bash
python -u fly_controller.py --varied --learn --train-layout short --trials 10 --eval-trials 24 --seed 42
python -u fly_controller.py --varied --learn --train-layout long  --trials 10 --eval-trials 24 --seed 42
```

On short shots, higher `P(firm)` is worse. On long shots, lower
`P(firm)` is worse. Compare each run's before and after results on
the layout it didn't train on.

## Demo notes

The pool-only scripts (`main.py` and `pool_drill.py`) don't need PyTorch.
The spike viewer needs a recorded run. Generate one from the project folder:

```bash
(cd external/fly-brain && python main.py --pytorch --experiment sugar --t_run 0.1 --n_run 1)
```

## External projects

- [pooltool](https://github.com/ekiefl/pooltool)
- [Eon Systems fly-brain](https://github.com/eonsystemspbc/fly-brain)

### First neural simulation

Ran the fly model on an NVIDIA RTX 4060 Laptop GPU.
During 100 ms of simulated time, 303 neurons produced 1,328 spikes.

The activity viewer replays recorded spikes:
- Orange: directly stimulated sugar-sensing neurons.
- Blue: other neurons firing in the network.

This demonstrates simulated neural activity, not learning or pool control.
