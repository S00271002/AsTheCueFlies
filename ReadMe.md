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
python fly_controller.py    # Same shot, with speed set by neural activity
python view_spikes.py        # Replay the recorded sugar-stimulation spikes
```

Only the fixed-speed demo also runs in the Windows pool-only `.venv`.
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
