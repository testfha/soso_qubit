import numpy as np
import scienceplots
import matplotlib.pyplot as plt

from collections import namedtuple
from pathlib import Path

from scripts.data import DATASETS, FIG_ROOT, IMAGES
from scripts.fitting.larmor import model as freq_rabi_model
from scripts.util.load_data import load_by_uuid
from scripts.util.plot import plot_1d_scatter, add_panel_label, make_image_panel
from scripts.util.rethreshold import rethreshold_dataset

DataCfg = namedtuple("DataCfg", ("uuid_larmor", "uuid_rabi", "invert", "label_left"))

data = {
    "q1 (1,2)": DataCfg(DATASETS["larmor_peak_q1_ro_1_2"], DATASETS["rabi_q1_ro_1_2"], False, False),
    "q2 (1,2)": DataCfg(DATASETS["larmor_peak_q2_ro_1_2"], DATASETS["rabi_q2_ro_1_2"], False, False),
    "q2 (2,3)": DataCfg(DATASETS["larmor_peak_q2_ro_2_3"], DATASETS["rabi_q2_ro_2_3"], True, True),
    "q3 (2,3)": DataCfg(DATASETS["larmor_peak_q3_ro_2_3"], DATASETS["rabi_q3_ro_2_3"], True, True),
}


# Shared color mapping for all panels
def get_label_colors():
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {label: cycle[i % len(cycle)] for i, label in enumerate(data.keys())}


def get_data_larmor():
    datasets = {}
    for label, data_cfg in data.items():
        ds = load_by_uuid(data_cfg.uuid_larmor)
        rethres_ds = rethreshold_dataset(
            ds, "_m0_2", new_axis_name="rethresholded", invert=data_cfg.invert
        )
        datasets[label] = rethres_ds
    return datasets


def do_fit_larmor(ds):
    x_name = next(filter(lambda x: "freq" in x, ds.coords))
    x = ds[x_name].values
    y = ds["_m0_2_fraction"].values
    return freq_rabi_model.fit(
        y,
        x=x,
        center_frequency=np.mean(x),
        rabi_frequency=1e6,
        amplitude=0.5,
        offset=0.0,
        angle=np.pi,
    )


def make_figure_larmor(datasets, ax, colors):
    y_name = "_m0_2_fraction"
    for label, ds in datasets.items():
        x_name = next(filter(lambda x: "freq" in x, ds.coords))
        res = do_fit_larmor(ds)

        center_frequency = float(res.params["center_frequency"])
        label_offset = float(res.params["rabi_frequency"]) * np.sqrt(2) / 2
        label_pos = center_frequency - label_offset if data[label].label_left else center_frequency + label_offset
        label_y = freq_rabi_model.eval(params=res.params, x=label_pos)

        xx = np.linspace(float(ds[x_name].min()), float(ds[x_name].max()), 100)
        yy = freq_rabi_model.eval(params=res.params, x=xx)

        ax.plot(
            xx * 1e-6,
            yy,
            ls="-",
            marker="none",
            zorder=0,
            color=colors[label],
        )
        ha = "right" if data[label].label_left else "left"
        additional_offset = -0.8 if data[label].label_left else 0.8
        ax.text(label_pos * 1e-6 + additional_offset, label_y, label, ha=ha, va="bottom", color=colors[label])

        plot_1d_scatter(
            ds,
            x_name=x_name,
            y_name=y_name,
            ax=ax,
            label=label,
            x_scale=1e-6,
            color=colors[label],
        )

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel(r"$P_{\uparrow}$")
    ax.set_ylim(bottom=0, top=1)


def get_data_rabi():
    datasets = {}
    for label, data_cfg in data.items():
        ds = load_by_uuid(data_cfg.uuid_rabi)
        datasets[label] = ds
    return datasets


def make_figure_rabi(datasets, axs, colors):
    x_name = "drive_time"
    y_name = "_m0_2_fraction"
    for (label, ds), ax in zip(datasets.items(), axs):
        plot_1d_scatter(
            ds,
            x_name=x_name,
            y_name=y_name,
            ax=ax,
            label=label,
            x_scale=1e-3,
            color=colors[label],
        )
        ax.set_ylim(bottom=0, top=1)
        ax.set_ylabel(r"$P_{\uparrow}$")

    for ax in axs[:-1]:
        ax.tick_params(labelbottom=False)

    axs[-1].set_xlabel(r"Drive duration [$\mathrm{\mu}$s]")


def build_figure():
    fig = plt.figure(figsize=(5, 3))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.2, 0.6],
        width_ratios=[1.6, 1.0],
    )

    ax_img = fig.add_subplot(gs[0, 0])

    subgs_b = gs[0, 1].subgridspec(4, 1, hspace=0.4)
    ax_b1 = fig.add_subplot(subgs_b[0, 0])
    sub_axs = [
        ax_b1,
        fig.add_subplot(subgs_b[1, 0], sharex=ax_b1),
        fig.add_subplot(subgs_b[2, 0], sharex=ax_b1),
        fig.add_subplot(subgs_b[3, 0], sharex=ax_b1),
    ]

    ax_bottom = fig.add_subplot(gs[1, :])

    colors = get_label_colors()

    make_image_panel(ax_img, IMAGES["device"])
    make_figure_rabi(get_data_rabi(), sub_axs, colors)
    make_figure_larmor(get_data_larmor(), ax_bottom, colors)

    fig.canvas.draw()
    fig.subplots_adjust(wspace=0.6, hspace=0.4)

    add_panel_label(ax_img, "(a)")
    add_panel_label(ax_b1, "(b)")
    add_panel_label(ax_bottom, "(c)")

    return fig


if __name__ == "__main__":
    with plt.style.context(["science", "scatter", "nature", "no-latex"]):
        fig = build_figure()

        path = FIG_ROOT / Path(__file__).stem
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "main.pdf", backend="pdf", dpi=300)

        plt.show()
