"""Decision layer: how the fly chooses, and how sugar and bitter teach it.

There is one decision neuron per option (for example "go for ball 1" or
"back spin, hard"). Each decision neuron has a synapse from every neuron in
the fly brain. During the look phase, the brain's spike counts x (normalised)
drive the decision neurons:

    drive_k = sum_i W[k, i] * x_i

The fly picks an option at random, weighted by softmax(drive). There is
always a small chance (epsilon) of trying something new.

After the shot, the taste becomes a learning signal r:
    sugar of strength s  ->  r = +s        (0 < s <= 1)
    mild bitter          ->  r = -0.3
    no taste             ->  r = 0

The synapses onto the chosen decision neuron that were active are
strengthened when r beats the fly's running expectation (the baseline), and
weakened when it falls short. This is a reward-modulated, three-factor
learning rule (pre-synaptic activity x post-synaptic choice x reward); in
the fly, dopamine neurons in the mushroom body play this role.
"""

import os
import tempfile
from pathlib import Path

import numpy as np


class DecisionLayer:
    def __init__(self, n_options, n_neurons, learning_rate=0.5, epsilon=0.05,
                 baseline_rate=0.05):
        self.n_options = n_options
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.baseline_rate = baseline_rate
        self.W = np.zeros((n_options, n_neurons), dtype=np.float32)
        self.baseline = 0.0
        self.shots_learned = 0

    @staticmethod
    def normalise(counts):
        x = np.asarray(counts, dtype=np.float32)
        norm = float(np.linalg.norm(x))
        return x / norm if norm > 0 else x

    def probabilities(self, x_hat):
        drive = self.W @ x_hat
        z = np.exp(drive - drive.max())
        p = z / z.sum()
        p = (1 - self.epsilon) * p + self.epsilon / self.n_options
        return p, drive

    def choose(self, p, rng):
        return int(rng.choice(self.n_options, p=p))

    def learn(self, x_hat, p, chosen, reward):
        advantage = reward - self.baseline
        post = -p.copy()
        post[chosen] += 1.0
        self.W += (self.learning_rate * advantage) * np.outer(post, x_hat).astype(np.float32)
        self.baseline += self.baseline_rate * (reward - self.baseline)
        self.shots_learned += 1
        return advantage

    # -- saving ------------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = np.flatnonzero(np.any(self.W != 0, axis=0))
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
        os.close(fd)
        np.savez_compressed(
            tmp, cols=cols, values=self.W[:, cols], n_options=self.n_options,
            n_neurons=self.n_neurons, baseline=self.baseline,
            shots_learned=self.shots_learned, learning_rate=self.learning_rate,
            epsilon=self.epsilon, baseline_rate=self.baseline_rate,
        )
        os.replace(tmp, path)

    @classmethod
    def load_or_new(cls, path, n_options, n_neurons, **kwargs):
        path = Path(path)
        layer = cls(n_options, n_neurons, **kwargs)
        if not path.exists():
            return layer, False
        d = np.load(path)
        if int(d["n_options"]) != n_options or int(d["n_neurons"]) != n_neurons:
            raise SystemExit(
                f"{path} was saved for a different setup "
                f"({int(d['n_options'])} options, {int(d['n_neurons'])} neurons). "
                "Rename or move it to start fresh.")
        layer.W[:, d["cols"]] = d["values"]
        layer.baseline = float(d["baseline"])
        layer.shots_learned = int(d["shots_learned"])
        return layer, True
        

class BallValueLayer(DecisionLayer):
    """Decision layer for random layouts: the fly looks at each ball in turn
    and one shared set of synapses turns the brain's response to that ball
    into a "how good is this ball" drive. The fly picks a ball with
    softmax(drive). Because the synapses are shared, what it learns is about
    how a ball lies, not whether it is called ball 1 or ball 2.

    Learning: W += rate * (reward - baseline) * (x_chosen - sum_k p_k x_k)
    """

    def __init__(self, n_choices, n_neurons, **kwargs):
        super().__init__(1, n_neurons, **kwargs)
        self.n_choices = n_choices

    def probabilities(self, x_hats):
        drive = np.array([float(self.W[0] @ x) for x in x_hats])
        z = np.exp(drive - drive.max())
        p = z / z.sum()
        p = (1 - self.epsilon) * p + self.epsilon / self.n_choices
        return p, drive

    def choose(self, p, rng):
        return int(rng.choice(self.n_choices, p=p))

    def learn(self, x_hats, p, chosen, reward):
        advantage = reward - self.baseline
        expected = sum(pk * x for pk, x in zip(p, x_hats))
        self.W[0] += (self.learning_rate * advantage) * (x_hats[chosen] - expected).astype(np.float32)
        self.baseline += self.baseline_rate * (reward - self.baseline)
        self.shots_learned += 1
        return advantage

    @classmethod
    def load_or_new(cls, path, n_choices, n_neurons, **kwargs):
        layer, resumed = DecisionLayer.load_or_new(path, 1, n_neurons, **kwargs)
        shared = cls(n_choices, n_neurons, **kwargs)
        shared.W, shared.baseline, shared.shots_learned = layer.W, layer.baseline, layer.shots_learned
        return shared, resumed
