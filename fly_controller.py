"""Run the pool drill with optional reward-gated synaptic plasticity."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import sys
from tempfile import NamedTemporaryFile

import pyarrow  # Load before torch to avoid the upstream library conflict.
import torch

from pool_drill import (
    EVAL_OFFSETS, FIRM_SPEED, LAYOUTS, SOFT_SPEED, TRAIN_OFFSETS,
    pocket_distance, simulate_drill_shot,
)
import pooltool as pt

fly_folder = Path(__file__).resolve().parent / "external" / "fly-brain"
sys.path.insert(0, str(fly_folder / "code"))

from benchmark import get_experiment
from run_pytorch import DT, MODEL_PARAMS, TorchModel, get_hash_tables, get_weights


class FlyBrain:
    """Reuse the connectome across trials; reset neural state between shots."""

    def __init__(self, seed, mixed=False, varied=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.reward_rng = torch.Generator(device=self.device).manual_seed(seed + 1)
        data = fly_folder / "data"
        neurons = data / "2025_Completeness_783.csv"
        connections = data / "2025_Connectivity_783.parquet"
        index, _ = get_hash_tables(str(neurons))

        # Reuse the walking input as an arbitrary drill cue, separate from sugar.
        cue = get_experiment("p9")
        sugar = get_experiment("sugar")
        self.cue_indices = [index[n] for n in cue["neu_exc"]]
        self.sugar_indices = [index[n] for n in sugar["neu_exc"]]
        self.input_indices = self.cue_indices + self.sugar_indices

        weights = get_weights(str(connections), str(neurons), str(data), csr=True)
        weights = weights.to(self.device)
        self.size = weights.shape[0]
        self.model = TorchModel(
            batch=1, size=self.size, dt=DT, params=MODEL_PARAMS,
            weights=weights, exc_indices=self.input_indices, device=self.device,
        ).eval()
        if mixed:
            # Different input neurons signal the layouts. The decoder is shared.
            self.cue_rates = {
                layout: self.make_rates([neuron], 2 * cue["stim_rate"])
                for layout, neuron in zip(LAYOUTS, self.cue_indices)
            }
        else:
            self.cue_rates = {"short": self.make_rates(self.cue_indices, cue["stim_rate"])}
        self.sugar_rates = self.make_rates(self.sugar_indices, sugar["stim_rate"])
        self.varied = varied
        if varied:
            table = pt.Table.default()
            self.distance_rates = {}
            for layout in LAYOUTS:
                for offset in (*TRAIN_OFFSETS, *EVAL_OFFSETS):
                    distance = pocket_distance(table, layout, offset)
                    far = min(1.0, max(0.0, (distance - 0.45) / 1.4))
                    rates = self.make_rates([self.cue_indices[0]], 200 * (1 - far))
                    rates[:, self.cue_indices[1]] = 200 * far
                    self.distance_rates[layout, offset] = rates
            self.cue_rates = {layout: self.distance_rates[layout, 0.0] for layout in LAYOUTS}
        print(f"Using {self.device}; loaded {self.size:,} neurons.")

    def make_rates(self, indices, rate):
        rates = torch.zeros(1, self.size, device=self.device)
        rates[:, indices] = rate
        return rates

    @torch.no_grad()
    def advance(self, state, rates, duration_ms=100, *, generator=None):
        counts = torch.zeros(self.size, device=self.device)
        generator = self.rng if generator is None else generator
        for _ in range(round(duration_ms / DT)):
            state = self.model(rates, *state, generator=generator)
            counts += (state[2][0] > 0).float()
        return state, counts

    def calibrate(self):
        """Pick two fixed readout groups from a reward-free cue response."""
        counts = torch.zeros(self.size, device=self.device)
        for rates in self.cue_rates.values():
            _, response = self.advance(self.model.state_init(), rates)
            counts += response
        counts[self.input_indices] = 0
        ranked = counts.argsort(descending=True, stable=True)[:12]
        active = ranked[counts[ranked] > 0]
        if len(active) < 2:
            raise RuntimeError("The drill cue did not activate enough readout neurons.")
        # Alternate by firing rank to avoid putting all high-rate cells on one side.
        size = len(active) // 2 * 2
        return active[:size:2], active[1:size:2]


class ShotPlasticity:
    """Strengthen existing excitatory inputs to the rewarded readout group."""

    def __init__(self, weights, groups):
        rows = weights.crow_indices()
        self.values = weights.values()
        self.edges = []
        for group in groups:
            positions = torch.cat([
                torch.arange(rows[i].item(), rows[i + 1].item(), device=weights.device)
                for i in group.tolist()
            ])
            positions = positions[self.values[positions] > 0]
            sources = weights.col_indices()[positions]
            baseline = self.values[positions].clone()
            self.edges.append((positions, sources, baseline))

    @torch.no_grad()
    def reinforce(self, action, decision_counts):
        positions, sources, baseline = self.edges[action]
        # Trial-local trace: credit presynaptic firing before sugar arrived.
        eligible = (decision_counts[sources] / 5).clamp(max=1)
        previous = self.values[positions]
        updated = (previous + 0.25 * baseline * eligible).minimum(3 * baseline)
        self.values[positions] = updated
        return int((updated != previous).sum().item())

    def changes(self):
        changes = {}
        for positions, _, baseline in self.edges:
            current = self.values[positions]
            changed = current != baseline
            changes.update(zip(positions[changed].tolist(), current[changed].tolist()))
        return changes

    def restore(self, changes):
        pending = {int(position): float(value) for position, value in changes.items()}
        for positions, _, baseline in self.edges:
            keys = positions.tolist()
            selected = [i for i, position in enumerate(keys) if position in pending]
            values = self.values.new_tensor([pending.pop(keys[i]) for i in selected])
            original = baseline[selected]
            if not ((values >= original) & (values <= 3 * original)).all().item():
                raise ValueError("Checkpoint contains invalid synaptic strengths.")
            self.values[positions[selected]] = values
        if pending:
            raise ValueError("Checkpoint contains connections outside the readout groups.")


def connectome_fingerprint():
    digest = sha256()
    for name in ("2025_Completeness_783.csv", "2025_Connectivity_783.parquet"):
        with (fly_folder / "data" / name).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path, payload):
    # Replace the old checkpoint only once its replacement is fully written.
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        try:
            json.dump(payload, temp, allow_nan=False)
            temp.close()
            Path(temp.name).replace(path)
        finally:
            Path(temp.name).unlink(missing_ok=True)


def choose_action(brain, soft_group, firm_group, layout, offset=0.0):
    rates = (brain.distance_rates[layout, offset] if brain.varied else brain.cue_rates[layout])
    state, counts = brain.advance(brain.model.state_init(), rates)
    soft = int(counts[soft_group].sum().item())
    firm = int(counts[firm_group].sum().item())
    probability = 0.5 + 0.4 * (firm - soft) / max(1, firm + soft)
    draw = torch.rand((), device=brain.device, generator=brain.rng).item()
    return state, counts, soft, firm, probability, int(draw < probability)


def evaluate(brain, groups, layout, trials, seed):
    """Use matched random streams and no reward pulses or weight updates."""
    saved_rng = brain.rng.get_state()
    brain.rng.manual_seed(seed)
    rewards, probabilities = 0, 0.0
    try:
        for i in range(trials):
            offset = EVAL_OFFSETS[i % len(EVAL_OFFSETS)] if brain.varied else 0.0
            _, _, _, _, probability, action = choose_action(brain, *groups, layout, offset)
            _, result = simulate_drill_shot(
                FIRM_SPEED if action else SOFT_SPEED, layout=layout, offset=offset
            )
            probabilities += probability
            rewards += result.reward
    finally:
        brain.rng.set_state(saved_rng)
    return rewards, probabilities / trials


def run_trials(trials, seed, show, learn=False, eval_trials=0, mixed=False,
               load=None, save=None, varied=False, save_every=0, train_layout=None):
    fingerprint = connectome_fingerprint() if load or save else None
    checkpoint = json.loads(load.read_text(encoding="utf-8")) if load else None
    config = {"version": 1, "connectome": fingerprint, "mixed": mixed,
              "speeds": [SOFT_SPEED, FIRM_SPEED], "seed": seed}
    if varied:
        config["curriculum"] = "distance-v1"
    if checkpoint is not None and checkpoint["config"] != config:
        raise ValueError("Checkpoint requires the same connectome, drill mode, speeds, and seed.")

    brain = FlyBrain(seed, mixed=mixed, varied=varied)
    if checkpoint is None:
        groups = brain.calibrate()
        completed = 0
    else:
        if checkpoint["device"] != brain.device:
            raise ValueError("Checkpoint RNG requires the same device type (CPU or CUDA).")
        groups = tuple(torch.tensor(group, dtype=torch.long, device=brain.device)
                       for group in checkpoint["groups"])
        ids = [neuron for group in checkpoint["groups"] for neuron in group]
        if (len(groups) != 2 or not len(groups[0]) or len(groups[0]) != len(groups[1])
                or len(set(ids)) != len(ids)
                or any(i < 0 or i >= brain.size or i in brain.input_indices for i in ids)):
            raise ValueError("Checkpoint has invalid readout groups.")
        brain.rng.set_state(torch.tensor(checkpoint["rng"], dtype=torch.uint8))
        brain.reward_rng.set_state(torch.tensor(checkpoint["reward_rng"], dtype=torch.uint8))
        completed = checkpoint["completed"]
    print(f"Fixed readout: {len(groups[0])} neurons per action.", flush=True)
    plasticity = ShotPlasticity(brain.model.weights, groups) if learn or load or save else None
    if checkpoint is not None:
        plasticity.restore(checkpoint["changes"])
        print(f"Loaded {load}: {completed} previous trials.", flush=True)
    layouts = tuple(brain.cue_rates)

    def persist(total):
        save_checkpoint(save, {
            "config": config,
            "device": brain.device,
            "groups": [group.tolist() for group in groups],
            "changes": plasticity.changes(),
            "completed": total,
            "rng": brain.rng.get_state().tolist(),
            "reward_rng": brain.reward_rng.get_state().tolist(),
        })
        print(f"Saved {save}: {total} total trials.", flush=True)

    def report(phase):
        for index, layout in enumerate(layouts):
            pots, probability = evaluate(
                brain, groups, layout, eval_trials, seed + 10_000 + index
            )
            print(
                f"{phase} {layout} (no sugar): {pots}/{eval_trials} clean pots; "
                f"mean P(firm)={probability:.3f}",
                flush=True,
            )

    if eval_trials and trials:
        report("Before")
    total_reward = 0

    offsets = TRAIN_OFFSETS if varied else (0.0,)
    training_layouts = (train_layout,) if train_layout else layouts
    cases = [(layout, offset) for offset in offsets for layout in training_layouts]
    schedule = [cases[(completed + i) % len(cases)] for i in range(trials)]
    Random(seed + completed).shuffle(schedule)
    for trial, (layout, offset) in enumerate(schedule, start=completed + 1):
        state, counts, soft, firm, firm_probability, action = choose_action(
            brain, *groups, layout, offset
        )
        speed = FIRM_SPEED if action else SOFT_SPEED

        shot, result = simulate_drill_shot(speed, layout=layout, offset=offset)
        total_reward += result.reward
        print(
            f"Trial {trial} [{layout}, offset={offset:+.3f} m]: soft/firm spikes={soft}/{firm}, "
            f"P(firm)={firm_probability:.2f}, "
            f"speed={speed:.2f} m/s, potted={result.potted}, "
            f"scratched={result.scratched}, reward={result.reward}"
        )
        if result.reward:
            # Deliver sugar after the outcome, continuing this trial's neural state.
            _, reward_counts = brain.advance(
                state, brain.sugar_rates, generator=brain.reward_rng
            )
            sugar_spikes = int(reward_counts[brain.sugar_indices].sum().item())
            print(f"  Sugar pulse: {sugar_spikes} spikes in sugar-input neurons.")
            if learn:
                changed = plasticity.reinforce(action, counts)
                print(f"  Updated {changed} existing synapses.")
        else:
            print("  No sugar pulse.")

        if save_every and (trial - completed) % save_every == 0:
            persist(trial)
        if show:
            pt.show(shot)

    if trials:
        print(f"Clean pots: {total_reward}/{trials}. Plasticity: {'on' if learn else 'off'}.")
    if save:
        persist(completed + trials)
    if eval_trials:
        report("After")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true", help="replay each shot")
    parser.add_argument("--learn", action="store_true", help="enable synaptic updates")
    parser.add_argument("--mixed", action="store_true", help="mix short and long layouts")
    parser.add_argument("--varied", action="store_true", help="train on varied positions using distance cues")
    parser.add_argument("--train-layout", choices=LAYOUTS,
                        help="train on only this layout; still evaluate both (requires --varied)")
    parser.add_argument("--save", type=Path, help="save a JSON checkpoint")
    parser.add_argument("--save-every", type=int, default=0, help="checkpoint every N training shots")
    parser.add_argument("--load", type=Path, help="restore a JSON checkpoint")
    parser.add_argument("--eval-trials", type=int, default=0,
                        help="sugar-free trials per layout before and after training")
    args = parser.parse_args()
    if args.trials < 0 or (args.trials == 0 and args.eval_trials < 1):
        parser.error("--trials must be positive, or use --eval-trials with zero")
    if args.eval_trials < 0:
        parser.error("--eval-trials cannot be negative")
    if args.train_layout and (not args.varied or not args.trials):
        parser.error("--train-layout requires --varied and positive --trials")
    if (args.mixed or args.varied) and not args.train_layout and args.trials % 2:
        parser.error("--mixed requires an even --trials count for equal layout exposure")
    if args.save_every < 0 or (args.save_every and not args.save):
        parser.error("--save-every needs a positive interval and --save")
    run_trials(args.trials, args.seed, args.show, args.learn, args.eval_trials,
               args.mixed or args.varied, args.load, args.save, args.varied,
               args.save_every, args.train_layout)
