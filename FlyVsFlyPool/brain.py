"""brain.py - the fly brain, and how a fly picks its shot.

Quick heads-up: nothing in here learns anything. No training, no rewards,
no sugar. Each fly is just a copy of the real fruit fly brain wiring
(FlyWire), run with the neuron model in fly_model.py. When it's a fly's
turn, we show its brain all the shots it could take and it goes with
whichever one gets its brain the most fired up.
"""

from pathlib import Path

import numpy as np
import pyarrow  # noqa: F401  (has to come before torch or they fight over a library)
import torch

from fly_model import DT, MODEL_PARAMS, TorchModel, get_hash_tables, get_weights

# ----------------------------------------------------------------------------
# Settings - the knobs you might want to fiddle with
# ----------------------------------------------------------------------------
DATA = Path(__file__).resolve().parent / "data"           # the data/ folder sitting next to this file
NEURON_FILE = DATA / "2025_Completeness_783.csv"          # every neuron in the brain
CONNECTION_FILE = DATA / "2025_Connectivity_783.parquet"  # every connection between them

LOOK_MS = 100.0                 # how long the fly gets to "think" about its options (ms)
LOW_HZ, HIGH_HZ = 20.0, 200.0   # a rubbish shot buzzes at 20 Hz, a great one at 200 Hz

# The fly doesn't have neurons for pool tables (shocker), so we borrow 18 of
# its water-tasting neurons and use them as its way of "seeing" a shot.
# These are FlyWire IDs, taken from the Shiu et al. 2024 paper's code.
WATER_NEURONS = [
    720575940612950568, 720575940631898285, 720575940606002609, 720575940612579053,
    720575940622902535, 720575940616177458, 720575940660292225, 720575940622486922,
    720575940613786774, 720575940629852866, 720575940625861168, 720575940613996959,
    720575940617857694, 720575940644965399, 720575940625203504, 720575940630553415,
    720575940635172191, 720575940634796536,
]

# MN9 is the neuron that makes the fly stick its tongue out to drink.
# If a shot makes this one fire, the fly basically wants it.
MN9 = 720575940660219265


# ----------------------------------------------------------------------------
# load_connectome: read the brain wiring off the disk
# We only do this once and both flies share it - no point loading it twice.
# ----------------------------------------------------------------------------
def load_connectome():
    # Use the graphics card if you've got one, otherwise the CPU (way slower).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Brain runs on: {device}")

    # Neurons have big 18-digit IDs; this turns them into simple row numbers.
    id_to_row, _ = get_hash_tables(str(NEURON_FILE))

    # The big table of who's connected to who, and how strongly.
    # First run builds a cache in data/ (slow), after that it just loads it.
    weights = get_weights(str(CONNECTION_FILE), str(NEURON_FILE), str(DATA)).to(device)
    print(f"Loaded the connectome: {weights.shape[0]:,} neurons")

    # Bundle up everything the flies need into one dict.
    return {
        "device": device,
        "weights": weights,
        "size": weights.shape[0],
        "inputs": [id_to_row[i] for i in WATER_NEURONS],
        "mn9": id_to_row[MN9],
    }


# ----------------------------------------------------------------------------
# shot_to_rates: turn a shot's 3 scores into how fast to buzz the 18 neurons
# ----------------------------------------------------------------------------
def shot_to_rates(numbers):
    # numbers is (near_pocket, straight, near_cue), each between 0 and 1.
    # Each score gets 6 neurons: 0 means 20 Hz, 1 means 200 Hz, and so on in between.
    rates = []
    for value in numbers:
        rates += [LOW_HZ + (HIGH_HZ - LOW_HZ) * value] * 6
    return rates


# ============================================================================
# Fly: one fly brain. We make two of these in main.py - Fly A and Fly B.
# ============================================================================
class Fly:
    def __init__(self, name, connectome, seed):
        # name: "Fly A" or "Fly B"
        # connectome: the shared brain wiring from load_connectome()
        # seed: each fly gets its own, so their brains are noisy in different ways
        self.name = name
        self.c = connectome
        self.rng = np.random.default_rng(seed)

    # ---- look: run the brain for a bit and count up the spikes ------------
    def look(self, rates):
        # rates: one row of 18 buzz rates per shot option.
        # We run a separate copy of the brain for every option, all at once,
        # and get back how many times every single neuron fired for each one.
        c = self.c
        n_options = len(rates)

        # Spin up a brain with one copy per option (they all share the wiring).
        model = TorchModel(batch=n_options, size=c["size"], dt=DT, params=MODEL_PARAMS,
                           weights=c["weights"], exc_indices=c["inputs"], device=c["device"])

        # Everything starts at 0 Hz, except our 18 water neurons get poked.
        input_rates = torch.zeros(n_options, c["size"], device=c["device"])
        input_rates[:, c["inputs"]] = torch.tensor(np.array(rates), dtype=torch.float32,
                                                   device=c["device"])

        # The pokes are a bit random - this is the fly's own dice to roll.
        gen = torch.Generator(device=c["device"])
        gen.manual_seed(int(self.rng.integers(2**31)))

        # Start the brain from resting, then tick it forward 0.1 ms at a time
        # (100 ms = 1,000 ticks), keeping a tally of who fired.
        state = model.state_init()
        spike_counts = torch.zeros(n_options, c["size"], device=c["device"])
        with torch.no_grad():
            for _ in range(int(LOOK_MS / DT)):
                state = model(input_rates, *state, generator=gen)
                spike_counts += state[2]          # state[2] = who fired on this tick
        return spike_counts.cpu().numpy()

    # ---- choose: go with whatever got the brain most excited ---------------
    def choose(self, options):
        # Show the brain every option and pick one.
        # Hands back: which option it picked, plus the spike numbers (for the screen).
        rates = [shot_to_rates(o["numbers"]) for o in options]
        counts = self.look(rates)

        # How many times the "drink this" neuron fired for each option.
        mn9 = counts[:, self.c["mn9"]].astype(int)
        # How many spikes happened in the rest of the brain (not counting
        # the 18 neurons we were poking ourselves - that'd be cheating).
        downstream = (counts.sum(axis=1) - counts[:, self.c["inputs"]].sum(axis=1)).astype(int)

        # Most MN9 wins. Tie? Most downstream spikes wins. Still a tie? Coin flip.
        tie_breaker = self.rng.random(len(options))
        best = int(np.lexsort((tie_breaker, downstream, mn9))[-1])
        return best, mn9, downstream
# ============================================================================
# end of Fly
# ============================================================================
