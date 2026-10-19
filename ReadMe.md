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