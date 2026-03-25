from pathlib import Path

import numpy as np

import scienceplots

from scripts.data import DATASETS, FIG_ROOT
from scripts.sim.sim import ExchangeParams, voltage_to_j
from scripts.sim.units import ns, Hz_to_rad, MHz
from scripts.util.load_data import load_by_uuid
from scripts.util.plot import plot_2d_heatmap, add_panel_label
from scripts.util.processing import fft_2d
from scripts.util.rethreshold import rethreshold_dataset

uuids = {
    "drive_map": DATASETS["drive_map"],
    "ramsey": DATASETS["ramsey_2d"],
}

operation_points = {
    # these are where we did the RB and matches the 2d map,
    # but we cannot directly put them in the map we use currently, because the 2d map has a different
    # we can try and find a better dataset for the 2d map, where we include the whole range
    "idle": (-13.9, -9.5),
    "x": (-14.2, -25.5), # 200 ns for x90, 10 ns ramps
    "z": (-10.5, -10.5), # 50 ns for z90, 10 ns ramps
}

datasets = {label: load_by_uuid(uuid) for label, uuid in uuids.items()}

_CMAP = "Blues"
_RASTERIZE = True


def make_figure_drive_map(ds, ax):
    rethres_ds = rethreshold_dataset(ds, "_m0_2", new_axis_name="rethresholded")

    plot_2d_heatmap(rethres_ds, x_name="J1_x90", y_name="J7_x90", z_name="rethresholded_state", ax=ax,
                    rasterized=_RASTERIZE,
                    cmap=_CMAP)

    # mark the operation points
    # for label, point in operation_points.items():
    #     ax.scatter(*point, label=label)

    ax.set_xlabel("$J_{1,2}$ [mV]")
    ax.set_ylabel("$J_{2,3}$ [mV]")
    ax.set_xticks([-16, -14, -12])


def make_figure_drive_map_sim(ax):
    v_range_12 = np.linspace(-0, -35, 81)
    v_range_23 = np.linspace(-12, -16, 81)
    duration = 200 * ns

    pop = do_simulation(v_range_12, v_range_23, duration)

    ax.pcolormesh(v_range_23, v_range_12, pop, rasterized=_RASTERIZE, cmap=_CMAP)
    ax.set_xlabel("$J_{1,2}$ [mV]")
    ax.set_ylabel("$J_{2,3}$ [mV]")
    ax.set_xticks([-16, -14, -12])


def do_simulation(v_range_12, v_range_23, duration):
    from scripts.sim import PulseSimulator

    b = [
        52.5 * MHz * Hz_to_rad,
        74 * MHz * Hz_to_rad,
        46.5 * MHz * Hz_to_rad,
    ]

    # crude estimate of the voltage-to-exchange conversion
    exch12 = ExchangeParams(scale=146758, exponent=-0.108851, offset=16.9625)
    exch23 = ExchangeParams(scale=2688.31, exponent=-0.577192, offset=-0.778441)
    j12_static = voltage_to_j(operation_points["idle"][1], exch12)
    j23_static = voltage_to_j(operation_points["idle"][0], exch23)

    sim = PulseSimulator(b=b, j12=j12_static, j23=j23_static)
    sim.set_exchange_params("J12", exch12)
    sim.set_exchange_params("J23", exch23)

    # we simulate a grid of voltage pulses and compute the return probability after 200 ns
    pop = np.zeros((len(v_range_12), len(v_range_23)))
    for i, v12 in enumerate(v_range_12):
        for j, v23 in enumerate(v_range_23):
            sim.clear_pulses()

            # we prepare the system in the first excited state
            evals, evecs = sim.static_hamiltonian.eigenstates()
            psi0 = evecs[1]

            sim.add_voltage_pulse("J12", v12, 0.0, duration)
            sim.add_voltage_pulse("J23", v23, 0.0, duration)
            states = sim.evolve(psi0, [0, duration])
            psi_f = states[-1]

            # we want the probability to leave the initial state, which is the first excited state
            pop[i, j] = 1 - abs(psi0.overlap(psi_f)) ** 2

    return pop


def make_figure_2d_ramsey(ds, ax):
    rethres_ds = rethreshold_dataset(ds, "_m0_2", new_axis_name="rethresholded")

    plot_2d_heatmap(rethres_ds, x_name="idle_time", y_name="J1", z_name="rethresholded_state", x_scale=1e-3, ax=ax,
                    rasterized=_RASTERIZE, cmap=_CMAP)

    ax.set_xlabel(r"$\tau$ [$\mathrm{\mu}$s]")
    ax.set_ylabel("$J_{1,2}$ [mV]")
    ax.set_xticks([0, 1, 2])


def make_figure_2d_ramsey_fft(ds, ax):
    rethres_ds = rethreshold_dataset(ds, "_m0_2", new_axis_name="rethresholded")
    fft_ds = fft_2d(rethres_ds, "rethresholded", fft_dim="idle_time", keep_dim="J1")

    plot_2d_heatmap(fft_ds, x_name="idle_time_freq", y_name="J1", z_name="rethresholded_fft_power", x_scale=1e3, ax=ax,
                    rasterized=_RASTERIZE, cmap=_CMAP)

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("$J_{1,2}$ [mV]")
    ax.set_xticks([0, 4, 8, 12])


def build_figure():
    fig = plt.figure(figsize=(4, 4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2,
                          height_ratios=[1.0, 1.0],
                          width_ratios=[1.0, 1.0],
                          )
    axs = gs.subplots()

    make_figure_drive_map(datasets["drive_map"], axs[0, 0])
    make_figure_drive_map_sim(axs[0, 1])

    make_figure_2d_ramsey(datasets["ramsey"], axs[1, 0])
    make_figure_2d_ramsey_fft(datasets["ramsey"], axs[1, 1])

    fig.canvas.draw()  # ensure axes are positioned before adding labels
    add_panel_label(axs[0, 0], "(a)")
    add_panel_label(axs[0, 1], "(b)")
    add_panel_label(axs[1, 0], "(c)")
    add_panel_label(axs[1, 1], "(d)")

    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    with plt.style.context(['science', 'nature', 'no-latex']):
        fig = build_figure()

        path = FIG_ROOT / Path(__file__).stem
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "main.pdf", backend="pdf")

        plt.show()
