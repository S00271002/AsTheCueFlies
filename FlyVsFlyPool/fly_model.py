"""fly_model.py - the simulated fruit-fly brain (the maths of every neuron).

WHERE THIS CODE COMES FROM
    This file is a trimmed copy of `code/run_pytorch.py` from the Eon Systems
    fly-brain repository (https://github.com/eonsystemspbc/fly-brain), which
    is licensed under the GNU GPL v2 (see LICENSE-fly-brain.txt in this
    folder). Only the parts this project needs are kept: the neuron model and
    the two functions that load the connectome. The benchmarking code, and its
    `import benchmark` line, are removed. The maths is unchanged; the comments
    are new.

    That repository re-implements, in PyTorch, the whole-brain model published
    by Shiu et al. (Nature, 2024), which runs on the FlyWire connectome: a map
    of roughly 139,000 neurons and the synapses between them in one adult
    female fruit fly (Drosophila melanogaster).

WHAT THE MODEL IS
    Every neuron is a "leaky integrate-and-fire" (LIF) neuron:
      * It has a voltage v that sits at a resting value (-52 mV).
      * Input from other neurons pushes v up or down.
      * v constantly "leaks" back towards rest (time constant 20 ms).
      * If v crosses a threshold (-45 mV) the neuron fires a spike, v is reset,
        and the neuron cannot fire again for a short refractory period (2.2 ms).

    A spike does not hit the next neuron instantly. It arrives after a delay
    (1.8 ms) and then acts through an "alpha-function" synapse, a conductance
    that rises and decays with a 5 ms time constant.

    Which neuron talks to which, and how strongly, is the connectome: a sparse
    weight matrix W where W[post, pre] = (number of synapses from `pre` to
    `post`) x (+1 if `pre` is excitatory, -1 if inhibitory).

    Sensory input is given as Poisson spike trains: each stimulated neuron
    receives random input spikes at a chosen average rate (e.g. 150 Hz).

    Time moves in fixed steps of DT = 0.1 ms, so 100 ms of brain time is
    1,000 steps. Every step updates all ~139,000 neurons at once with tensor
    operations, which is why a GPU helps so much.

BATCHES
    Every tensor has a first dimension called `batch`. Each batch row is a
    completely separate copy of the brain, with its own voltages and spikes,
    sharing only the weights. This project uses batch = 2 so the fly can "look
    at" both balls at the same time instead of one after the other.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401  Must be imported before torch (avoids a libarrow clash).
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model parameters (the same values as the Brian2 model in Shiu et al.)
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    'tauSyn': 5.0,        # ms   - how fast a synaptic conductance decays
    'tDelay': 1.8,        # ms   - how long a spike takes to reach the next neuron
    'v0': -52.0,          # mV   - starting voltage
    'vReset': -52.0,      # mV   - voltage straight after a spike
    'vRest': -52.0,       # mV   - voltage the neuron leaks back towards
    'vThreshold': -45.0,  # mV   - voltage at which the neuron fires
    'tauMem': 20.0,       # ms   - how fast the voltage leaks back to rest
    'tRefrac': 2.2,       # ms   - "recovery" time after a spike
    'scalePoisson': 250,  # size of one external (sensory) input spike
    'wScale': 0.275,      # mV per synapse: converts synapse counts into voltage
}

DT = 0.1  # Simulation time step in ms (matches Brian2's defaultclock.dt).


# ---------------------------------------------------------------------------
# 1. Sensory input: random (Poisson) spikes at a chosen rate
# ---------------------------------------------------------------------------
class PoissonSpikeGenerator(nn.Module):
    """Generates one time step of Poisson-distributed input spikes.

    A Poisson spike train at rate r Hz fires in a time step of DT ms with
    probability r * DT / 1000. For 150 Hz and DT = 0.1 ms that is 0.015, so
    about 15 spikes per 100 ms. `torch.bernoulli` makes that coin flip for
    every neuron at once; neurons with rate 0 never fire. Each input spike
    is multiplied by `scale` (250) so one input spike is a strong kick.
    """

    def __init__(self, dt, scale, device='cpu'):
        super().__init__()
        self.prob_scale = dt / 1000.0   # converts Hz into "probability per step"
        self.scale = scale              # size of each input spike (250)
        self.device = device            # "cuda" or "cpu"

    def forward(self, rates, generator=None):
        # rates * prob_scale = chance of an input spike this step, per neuron.
        # torch.bernoulli flips one biased coin per neuron (1 = spike).
        # `generator` is a seeded random number generator, so a run can be
        # repeated exactly.
        return torch.bernoulli(rates * self.prob_scale, generator=generator) * self.scale


# ---------------------------------------------------------------------------
# 2. Synapses: delayed, smoothly decaying input from other neurons
# ---------------------------------------------------------------------------
class AlphaSynapse(nn.Module):
    """Synaptic conductance with a transmission delay.

    `delay_buffer` is a queue holding the input that will arrive over the
    next 1.8 ms (18 steps). Each step the oldest entry is added to the
    conductance and the newest input joins the back of the queue.

    The conductance decays each step by the factor (1 - DT / tauSyn), which
    is the step-by-step (Euler) version of exponential decay with a 5 ms
    time constant. `refrac` is 1 for neurons that are ready and 0 for
    neurons still in their refractory period, so input is ignored while a
    neuron recovers.
    """

    def __init__(self, batch, size, dt, params, device='cpu'):
        super().__init__()
        self.time_factor = dt / params['tauSyn']      # 0.1 / 5 = 0.02 decay per step
        self.steps_delay = int(params['tDelay'] / dt)  # 1.8 / 0.1 = 18 steps of delay
        self.size = size                               # number of neurons
        self.device = device
        self.batch = batch                             # number of independent brain copies

    def state_init(self):
        # No synaptic input anywhere, and nothing waiting in the delay queue.
        # Shapes: conductance (batch, size); delay_buffer (batch, 19, size).
        conductance = torch.zeros(self.batch, self.size, device=self.device)
        delay_buffer = torch.zeros(
            self.batch, self.steps_delay + 1, self.size, device=self.device
        )
        return conductance, delay_buffer

    def forward(self, input_, conductance, delay_buffer, refrac):
        # Decay the old conductance and add the input that has just arrived
        # (the front of the queue), but only for neurons that are not
        # refractory.
        conductance_new = (
            conductance * (1 - self.time_factor) + delay_buffer[:, 0, :] * refrac
        )
        # Move the queue forward one step and put this step's input at the back.
        delay_buffer = torch.roll(delay_buffer, shifts=-1, dims=1)
        delay_buffer[:, -1, :] = input_
        return conductance_new, delay_buffer


# ---------------------------------------------------------------------------
# 3. Neurons: leaky integrate-and-fire
# ---------------------------------------------------------------------------
class LIFNeuron(nn.Module):
    """Leaky integrate-and-fire neuron.

    Each step:
        v <- v + (direct sensory kick)
        v <- v + (DT / tauMem) * (g - (v - vRest))     # leak + synaptic input
        spike if v > vThreshold, then v <- vReset

    The ATan class below gives a "surrogate gradient". It only matters when
    training a network by back-propagation, which this project never does
    (the brain is run inside torch.no_grad()). In the forward direction it
    is simply "1 if v is above threshold, else 0".
    """

    def __init__(self, batch, size, dt, params, device='cpu'):
        super().__init__()
        self.size = size
        self.dt = dt
        self.tau_mem = params['tauMem']
        self.v_reset = params['vReset']
        self.v_rest = params['vRest']
        self.v_threshold = params['vThreshold']
        self.v_0 = params['v0']
        self.time_factor = dt / self.tau_mem      # 0.1 / 20 = 0.005: share of the gap to rest closed per step
        self.spike_gradient = self.ATan.apply     # the "did it cross threshold?" function
        self.device = device
        self.batch = batch

    def state_init(self):
        # Every neuron starts at v0 (-52 mV) and has not spiked.
        v = torch.zeros(self.batch, self.size, device=self.device) + self.v_0
        spikes = torch.zeros(self.batch, self.size, device=self.device)
        return spikes, v

    def forward(self, conductance, voltage_stim, v):
        # Brian/GeNN equivalent:
        #   V += Vstim
        #   V += MemFactor * (G - (V - Vrest))
        v = v + voltage_stim                                          # direct sensory kick
        v = v + self.time_factor * (conductance - (v - self.v_rest))  # synaptic input minus leak

        spike = self.spike_gradient(v - self.v_threshold)  # 1.0 or 0.0 per neuron

        # Neurons that spiked are put back to vReset.
        reset = ((v - self.v_reset) * spike).detach()
        v = v - reset

        return spike, v

    @staticmethod
    class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v):
            spike = (v > 0).float()
            ctx.save_for_backward(v)
            return spike

        @staticmethod
        def backward(ctx, grad_output):
            (v,) = ctx.saved_tensors
            grad = 1 / (1 + (np.pi * v).pow_(2)) * grad_output
            return grad


# ---------------------------------------------------------------------------
# 4. Synapse + neuron + refractory period, combined
# ---------------------------------------------------------------------------
class AlphaLIF(nn.Module):
    """One layer of LIF neurons with alpha synapses and a refractory period.

    `refrac` counts the steps since each neuron last fired. A neuron is
    "ready" again once that count reaches 22 steps (2.2 ms).

    `exc_indices`: neurons we stimulate directly get a refractory period of
    0, as in the original model, so their synaptic input is never blocked.
    (The refractory period only gates synaptic input; a direct sensory kick
    of 250 x 0.275 = ~69 mV makes a neuron fire whatever its state.)
    """

    def __init__(
        self,
        batch,
        size,
        dt,
        params,
        exc_indices=None,
        device='cpu'
    ):
        super().__init__()
        self.size = size
        self.synapse = AlphaSynapse(batch, size, dt, params, device=device)
        self.neuron = LIFNeuron(batch, size, dt, params, device=device)
        base_refrac = int(round(params['tRefrac'] / dt))   # 2.2 / 0.1 = 22 steps

        # How many steps each neuron must wait after a spike (22 for most).

        self.refrac_steps = torch.full(
            (size,),
            base_refrac,
            dtype=torch.long,
            device=device,
        )

        if exc_indices is not None:
            self.refrac_steps[exc_indices] = 0

    def state_init(self):
        """A fresh brain: every neuron at rest, nothing in flight."""
        conductance, delay_buffer = self.synapse.state_init()
        spikes, v = self.neuron.state_init()
        # Start every neuron as "ready" (steps since last spike = the full wait).
        refrac = self.refrac_steps.unsqueeze(0).repeat(self.neuron.batch, 1).float()
        return conductance, delay_buffer, spikes, v, refrac

    def forward(self,
            recurrent_input,
            voltage_stim,
            conductance,
            delay_buffer,
            spikes,
            v,
            refrac):
        # Count steps since the last spike (back to 0 for neurons that just fired).
        refrac = torch.where(
            spikes > 0,
            torch.zeros_like(refrac),
            refrac + 1,
        )
        # Update the synapses. The last argument is 1.0 for neurons that have
        # waited long enough since their last spike, else 0.0.
        conductance_new, delay_buffer = self.synapse(
            recurrent_input,
            conductance,
            delay_buffer,
            (
                refrac >= self.refrac_steps.unsqueeze(0)
            ).float()
        )

        # Update the voltages and find which neurons fire this step.
        spikes, v_new = self.neuron(
            conductance,
            voltage_stim,
            v
        )
        # A neuron that fires has its synaptic conductance cleared.
        conductance_reset = (conductance_new * spikes).detach()
        conductance_new = conductance_new - conductance_reset
        return conductance_new, delay_buffer, spikes, v_new, refrac


# ---------------------------------------------------------------------------
# 5. The whole brain: sensory input + connectome + neurons
# ---------------------------------------------------------------------------
class TorchModel(nn.Module):
    """One time step of the whole brain.

    1. Sensory input: Poisson spikes for the neurons we stimulate.
    2. Recurrent input: last step's spikes (a 0/1 vector) multiplied by the
       connectome. For each neuron this adds up the signed synapse counts
       from every neuron that just fired. W is sparse (each neuron
       connects to only a tiny fraction of the others), so a sparse matrix
       product is used.
    3. Both are scaled by wScale (0.275 mV per synapse) and fed into AlphaLIF.
    """

    def __init__(
            self,
            batch,
            size,
            dt,
            params,
            weights,
            exc_indices=None,
            device='cpu'
        ):
        super().__init__()
        self.neurons = AlphaLIF(
            batch,
            size,
            dt,
            params,
            exc_indices=exc_indices,
            device=device
        )
        self.weights = weights                     # sparse connectome W[post, pre]
        self.poisson = PoissonSpikeGenerator(dt, params['scalePoisson'], device=device)
        self.scale = params['wScale']              # 0.275 mV per synapse

    def state_init(self):
        return self.neurons.state_init()

    def forward(self, rates, conductance, delay_buffer, spikes, v, refrac, generator=None):
        # 1. Random sensory input spikes for the stimulated neurons.
        poisson_spikes = self.poisson(
            rates,
            generator=generator
        )

        # Each input spike is a 250 x 0.275 = ~69 mV kick.
        voltage_stim = self.scale * poisson_spikes

        # 2. Recurrent input: last step's spikes (batch, size) times W
        # transposed. Entry j of the result is the signed synapse count from
        # all neurons that just fired onto neuron j.
        weighted_spikes = torch.matmul(
            spikes,
            self.weights.transpose(0, 1)
        )

        recurrent_input = self.scale * weighted_spikes   # synapse counts -> mV

        conductance, delay_buffer, spikes, v, refrac = self.neurons(
            recurrent_input,
            voltage_stim,
            conductance,
            delay_buffer,
            spikes,
            v,
            refrac,
        )
        return conductance, delay_buffer, spikes, v, refrac


# ---------------------------------------------------------------------------
# 6. Loading the connectome
# ---------------------------------------------------------------------------
def get_hash_tables(comp_path):
    """Map FlyWire neuron IDs (18-digit numbers) to tensor rows and back.

    The completeness CSV lists every neuron once; its row number is that
    neuron's index in every tensor above.
    """
    df_comp = pd.read_csv(comp_path, index_col=0)        # index = FlyWire IDs
    # enumerate gives (row number, FlyWire ID) pairs.
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    i2flyid = {j: i for i, j in flyid2i.items()}
    return flyid2i, i2flyid


def get_weights(conn_path, comp_path, wt_dir, csr=True):
    """Build the sparse connectome weight matrix W[post, pre].

    The connectivity parquet has one row per connected pair of neurons, with
    their indices and the signed synapse count ('Excitatory x Connectivity').
    Building the matrix takes a while, so the result is cached as
    weight_coo.pkl / weight_csr.pkl in `wt_dir` and reused on later runs.
    (Delete those two files if you upgrade PyTorch and loading fails.)

    COO = a list of (row, column, value) triples; CSR = the same data
    re-arranged by row, which is faster for the matrix product each step.
    """
    wt_dir = Path(wt_dir)
    coo_path = wt_dir / 'weight_coo.pkl'
    csr_path = wt_dir / 'weight_csr.pkl'

    data_conn = pd.read_parquet(conn_path)     # one row per connected pair of neurons
    data_name = pd.read_csv(comp_path)         # one row per neuron
    num_neurons = data_name.shape[0]           # the matrix is num_neurons x num_neurons

    # Use the cached matrix if there is one; otherwise build it and cache it.

    try:
        with open(coo_path, 'rb') as f:
            weight_coo = pickle.load(f)
    except FileNotFoundError:
        print('Weights not found, constructing COO weight matrix...')
        # Row = postsynaptic neuron, column = presynaptic neuron.
        idx = [
            data_conn['Postsynaptic_Index'].to_list(),
            data_conn['Presynaptic_Index'].to_list(),
        ]
        # Value = synapse count, made negative for inhibitory presynaptic neurons.
        val = data_conn['Excitatory x Connectivity'].to_list()
        weight_coo = torch.sparse_coo_tensor(
            idx, val, (num_neurons, num_neurons)
        ).to(torch.float32)
        with open(coo_path, 'wb') as f:
            pickle.dump(weight_coo, f)

    if csr:
        try:
            with open(csr_path, 'rb') as f:
                weight_csr = pickle.load(f)
        except FileNotFoundError:
            print('CSR weights not found, converting from COO...')
            weight_csr = weight_coo.to_sparse_csr()
            with open(csr_path, 'wb') as f:
                pickle.dump(weight_csr, f)
        return weight_csr
    else:
        return weight_coo
