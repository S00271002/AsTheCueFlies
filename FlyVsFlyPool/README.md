# Fly vs Fly Pool
DO NOT COMMIT .pkl FILES ADD TO GIT IGNORE THEY ARE GENERATED NEW EACH TIME YOU START 

Two copies of a simulated fruit-fly brain (the FlyWire connectome, run with
the Shiu et al. 2024 neuron model) play a game of 8-ball against each other.
pooltool does the physics and the rules. The game is drawn from above with
matplotlib.

**There is no training and no reward.** On each turn, the fly whose turn it
is gets shown every legal shot. It takes the one its brain reacts to most
strongly.

## Files

| File | What it does |
|---|---|
| `main.py` | Start here. Creates Fly A and Fly B and plays the game |
| `brain.py` | Loads the brain and lets a fly look at its options and choose |
| `table.py` | Table geometry: aiming, pockets, which shots are possible |
| `viewer.py` | The 2D matplotlib window |
| `fly_model.py` | The neuron model, from the Eon Systems fly-brain repo (GPL v2, see `LICENSE-fly-brain.txt`) |
| `data/` | `2025_Completeness_783.csv` and `2025_Connectivity_783.parquet` (the connectome) |

## Setup (Windows, PowerShell, in this folder)

```powershell
C:\Python313\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

The last line installs PyTorch with CUDA for an NVIDIA graphics card. Without
one, the brain runs on the CPU, which is much slower.

The first run builds `data/weight_coo.pkl` and `data/weight_csr.pkl`, so it is
slower. If PyTorch is upgraded and loading fails, delete those two files.

## Run

```powershell
.venv\Scripts\python.exe main.py               # play one game and watch it
.venv\Scripts\python.exe main.py --speed 3     # three times faster
.venv\Scripts\python.exe main.py --no-watch    # no window, just text
.venv\Scripts\python.exe main.py --seed 7      # a different rack and brain noise
```

## How a fly chooses a shot

1. `table.py` lists every legal shot. That's one per ball the fly is allowed
   to hit, each with its easiest pocket.
2. Each shot is described by three numbers from 0 (bad) to 1 (good): how near
   the ball is to the pocket, how straight the shot is, and how near the ball
   is to the cue ball.
3. Those numbers are turned into firing rates (20–200 Hz) on 18 of the fly's
   water-taste neurons, 6 neurons per number.
4. The brain runs for 100 ms, with one copy for each shot side by side, and
   its spikes are counted.
5. The fly takes the shot that made MN9 (the "drink" motor neuron) fire most.
   If MN9 is tied, as it usually is at 0, it takes the shot that made the most
   other neurons fire.

Fly A and Fly B share the same wiring. Each has its own random seed, so the
noise in their brains is different, and they can choose differently in the
same position.

The fly only chooses **which shot**. The aim and speed are worked out in
`table.py`, the break is a fixed hard shot, and the rules (fouls, ball in
hand, the 8 last) are pooltool's 8-ball rules. With ball in hand, the cue
ball is placed behind each ball for a straight pot, and the fly picks one.
A game that reaches 250 shots goes to whoever has potted more balls.
