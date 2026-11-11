"""The fly brain used during training, and the neurons it uses.

Each shot has two brain phases:
  1. Look:  the water-taste sensory neurons are stimulated (the fly "sees"
            the table) and the whole brain responds. Spike counts from this
            phase feed the decision layer.
  2. Taste: after the shot, sugar (reward) or a mild bitter pulse (failure)
            is given to the real taste neurons and the brain's response is
            recorded, including MN9, the motor neuron that starts feeding.

The connectome is never changed. Learning happens in the decision layer
(decision.py). Neuron IDs come from the Shiu et al. paper code in
external/fly-brain/code/paper-phil-drosophila.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FLY_FOLDER = Path(__file__).resolve().parent.parent / "external" / "fly-brain"
sys.path.insert(0, str(FLY_FOLDER / "code"))

import pyarrow  # noqa: F401  Import before torch to avoid the known library conflict.
import torch
from run_pytorch import DT, MODEL_PARAMS, TorchModel, get_hash_tables, get_weights

# Taste: the reward and the mild punishment.
SUGAR = [  # labellar sugar-sensing gustatory receptor neurons (right side)
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940639259967, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
BITTER = [  # labellar bitter-sensing gustatory receptor neurons
    720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093,
    720575940626287336, 720575940618600651, 720575940627692048, 720575940630195909,
    720575940646212996, 720575940610483162, 720575940645743412, 720575940627578156,
    720575940622298631, 720575940621008895, 720575940629146711, 720575940610259370,
    720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118,
    720575940604027168,
]
MN9 = 720575940660219265  # proboscis motor neuron: the fly's "start feeding" output

# "What the fly sees": the fly has no neurons for pool tables, so the
# scenario is signalled through real water-taste sensory neurons.
WATER = [
    720575940612950568, 720575940631898285, 720575940606002609, 720575940612579053,
    720575940622902535, 720575940616177458, 720575940660292225, 720575940622486922,
    720575940613786774, 720575940629852866, 720575940625861168, 720575940613996959,
    720575940617857694, 720575940644965399, 720575940625203504, 720575940630553415,
    720575940635172191, 720575940634796536,
]


@dataclass
class Activity:
    counts: np.ndarray                 # spikes per neuron
    duration_ms: float
    times_ms: np.ndarray = field(default_factory=lambda: np.empty(0))
    neurons: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    def total(self, indices=None):
        if indices is None:
            return int(self.counts.sum())
        return int(self.counts[indices].sum()) if len(indices) else 0


class TrainingBrain:
    def __init__(self, sensory_ids, device=None):
        data = FLY_FOLDER / "data"
        neuron_file = data / "2025_Completeness_783.csv"
        connection_file = data / "2025_Connectivity_783.parquet"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        index, _ = get_hash_tables(str(neuron_file))
        weights = get_weights(str(connection_file), str(neuron_file), str(data),
                              csr=True).to(device=self.device)
        self.size = weights.shape[0]
        print(f"Loaded connections for {self.size:,} neurons")

        def lookup(ids, name):
            found = [index[i] for i in ids if i in index]
            if len(found) < len(ids):
                print(f"Note: {len(ids) - len(found)} {name} neuron(s) not in this connectome version")
            return found

        self.sensory = lookup(sensory_ids, "sensory")
        self.sugar = lookup(SUGAR, "sugar")
        self.bitter = lookup(BITTER, "bitter")
        self.mn9 = lookup([MN9], "MN9")
        stimulated = sorted(set(self.sensory + self.sugar + self.bitter))

        self.model = TorchModel(
            batch=1, size=self.size, dt=DT, params=MODEL_PARAMS,
            weights=weights, exc_indices=stimulated, device=self.device,
        )
        self.model.eval()
        self.state = None
    def simulate(self, stimuli, duration_ms, seed=None, record=False, reset=True):
        """Run the brain. stimuli = [(neuron_indices, rate_hz), ...]."""
        if seed is not None:
            torch.manual_seed(int(seed))
        if reset or self.state is None:
            self.state = self.model.state_init()

        rates = torch.zeros(1, self.size, device=self.device)
        for indices, hz in stimuli:
            if len(indices) and hz > 0:
                rates[0, indices] = float(hz)

        counts = torch.zeros(self.size, device=self.device)
        times, fired_all = [], []
        steps = int(round(duration_ms / DT))
        with torch.no_grad():
            for step in range(steps):
                self.state = self.model(rates, *self.state)
                fired = self.state[2][0] > 0
                counts += fired.float()
                if record:
                    idx = torch.nonzero(fired).flatten()
                    if idx.numel():
                        fired_all.append(idx.cpu())
                        times.append(torch.full((idx.numel(),), step * DT))

        activity = Activity(counts=counts.cpu().numpy(), duration_ms=duration_ms)
        if record and fired_all:
            activity.neurons = torch.cat(fired_all).numpy()
            activity.times_ms = torch.cat(times).numpy()
        return activity
