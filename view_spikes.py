from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

project_folder = Path(__file__).resolve().parent
spike_file = project_folder / "external/fly-brain/data/results/pytorch_t0.1s_n1.parquet"

spikes = pd.read_parquet(spike_file)

sugar_neuron_ids = {  # actual ids for sugar neurons in the fly brain
    720575940624963786,
    720575940630233916,
    720575940637568838,
    720575940638202345,
    720575940617000768,
    720575940630797113,
    720575940632889389,
    720575940621754367,
    720575940621502051,
    720575940640649691,
    720575940639332736,
    720575940616885538,
    720575940639198653,
    720575940639259967,
    720575940617937543,
    720575940632425919,
    720575940633143833,
    720575940612670570,
    720575940628853239,
    720575940629176663,
    720575940611875570,
}

spikes["is_input"] = spikes["flywire_id"].isin(sugar_neuron_ids)

fig, ax = plt.subplots()

ax.set_xlim(0, spikes["time_ms"].max() + 5)
ax.set_ylim(0, spikes["neuron_index"].max() + 1)
ax.set_xlabel("Simulation time (milliseconds)")
ax.set_ylabel("Neuron index")
ax.set_title("Fruit-fly neural activity")

dots = ax.scatter([], [], s=5)


def update(frame):
    start = frame * 5
    end = start + 5

    active = spikes[(spikes["time_ms"] >= start) & (spikes["time_ms"] < end)]

    dots.set_offsets(active[["time_ms", "neuron_index"]].to_numpy())
    colours = active["is_input"].map(
        {
            True: "orange",
            False: "blue",
        }
    )

    dots.set_color(colours.tolist())
    ax.set_title(f"Neural activity: {start}–{end} ms")

    return (dots,)


frame_count = int(spikes["time_ms"].max() // 5) + 1

animation = FuncAnimation(
    fig,
    update,
    frames=frame_count,
    interval=200,
    repeat=True,
)

plt.show()
