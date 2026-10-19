from pathlib import Path
import sys

project_folder = Path(__file__).resolve().parent
fly_folder = project_folder / "external" / "fly-brain"
sys.path.insert(0, str(fly_folder / "code"))

import pyarrow  # Import before torch to avoid the known library conflict.
import torch
import pooltool as pt

from benchmark import get_experiment
from run_pytorch import (
    DT,
    MODEL_PARAMS,
    TorchModel,
    get_hash_tables,
    get_weights,
)


def get_cue_speed():
    """Run the fly model and map downstream spike count to cue speed."""
    data_folder = fly_folder / "data"
    neuron_file = data_folder / "2025_Completeness_783.csv"
    connection_file = data_folder / "2025_Connectivity_783.parquet"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Fix the random seed so runs are easier to compare.
    torch.manual_seed(42)

    flyid_to_index, _ = get_hash_tables(str(neuron_file))

    weights = get_weights(
        str(connection_file),
        str(neuron_file),
        str(data_folder),
        csr=True,
    ).to(device=device)

    neuron_count = weights.shape[0]
    print(f"Loaded connections for {neuron_count:,} neurons")

    experiment = get_experiment("sugar")
    input_indices = [flyid_to_index[neuron_id] for neuron_id in experiment["neu_exc"]]

    print(f"Selected {len(input_indices)} input neurons")

    model = TorchModel(
        batch=1,
        size=neuron_count,
        dt=DT,
        params=MODEL_PARAMS,
        weights=weights,
        exc_indices=input_indices,
        device=device,
    )

    model.eval()
    state = model.state_init()

    rates = torch.zeros(1, neuron_count, device=device)
    rates[:, input_indices] = experiment["stim_rate"]

    spike_counts = torch.zeros(neuron_count, device=device)
    duration_ms = 100
    steps = round(duration_ms / DT)

    print(f"Running {duration_ms} ms of brain activity...")

    with torch.no_grad():
        for _ in range(steps):
            state = model(rates, *state)
            spikes = state[2]
            spike_counts += (spikes[0] > 0).float()

    downstream_counts = spike_counts.clone()
    downstream_counts[input_indices] = 0

    input_spikes = spike_counts[input_indices].sum().item()
    response_spikes = downstream_counts.sum().item()

    print(f"Total spikes: {int(spike_counts.sum().item())}")
    print(f"Active neurons: {int((spike_counts > 0).sum().item())}")
    print(f"Input spikes: {int(input_spikes)}")
    print(f"Other spikes: {int(response_spikes)}")

    # Engineered starting rule, not learned behaviour.
    activity_scale = 2000
    activity_level = min(response_spikes / activity_scale, 1.0)
    cue_speed = 0.5 + activity_level * 1.5

    print(f"Chosen cue speed: {cue_speed:.2f} m/s")
    return cue_speed


def play_shot(cue_speed):
    """Simulate a straight shot using the brain-derived cue speed."""
    table = pt.Table.default()

    balls = {
        "cue": pt.Ball.create("cue", xy=(0.4, 0.4)),
        "1": pt.Ball.create("1", xy=(0.4, 0.8)),
    }

    cue = pt.Cue(cue_ball_id="cue")
    shot = pt.System(table=table, balls=balls, cue=cue)

    shot.cue.set_state(
        V0=cue_speed,
        phi=90,
        a=0,
        b=0,
        theta=0,
    )

    pt.simulate(shot, inplace=True)
    pt.show(shot)


if __name__ == "__main__":
    cue_speed = get_cue_speed()
    play_shot(cue_speed)
