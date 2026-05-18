from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter


ROOT = Path("/Users/tivakhtel/SOSO-qubit/tess_code")
COLORS = {
    "infidelity": "#2f6fb0",
    "leakage": "#d97706",
    "x90": "#c43d2f",
    "z90": "#2f855a",
    "ink": "#202020",
    "guide": "#b8b8b8",
}


def prx_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (6.45, 3.05),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10.8,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "legend.frameon": False,
            "legend.handlelength": 2.4,
        }
    )


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def rows_to_array(rows: list[dict]) -> np.ndarray:
    data = np.array(
        [[row["tg_ns"], row["infidelity"], row["leakage"], row["dJ_MHz"]] for row in rows],
        dtype=float,
    )
    return data[np.argsort(data[:, 0])]


def setup_log_axis(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", top=True, right=True, pad=4)


def panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.02) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13.2,
        fontweight="bold",
        color=COLORS["ink"],
    )


def add_target_markers(
    ax: plt.Axes,
    point: dict,
    *,
    color: str,
    marker: str,
    text_offset: tuple[float, float],
) -> None:
    tg = float(point["tg_ns"])
    infidelity = float(point["infidelity"])
    leakage = float(point["leakage"])

    ax.axvline(tg, color=COLORS["guide"], lw=0.7, ls=(0, (1.2, 2.0)), zorder=0)
    ax.scatter(
        [tg, tg],
        [infidelity, leakage],
        s=34,
        marker=marker,
        facecolors="white",
        edgecolors=color,
        linewidths=1.1,
        zorder=6,
    )
    ax.annotate(
        point["label"],
        xy=(tg, leakage),
        xytext=text_offset,
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=9.4,
        color=color,
        bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.9},
    )


def draw_panel(
    ax: plt.Axes,
    data: np.ndarray,
    point: dict,
    *,
    title: str,
    point_color: str,
    point_marker: str,
    point_text_offset: tuple[float, float],
    ylabel: bool,
) -> None:
    dotted = (0, (0.45, 2.25))
    lw = 1.45

    ax.plot(data[:, 0], data[:, 1], color=COLORS["infidelity"], lw=lw, solid_capstyle="round")
    ax.plot(
        data[:, 0],
        data[:, 2],
        color=COLORS["leakage"],
        lw=lw,
        ls=dotted,
        dash_capstyle="round",
    )
    add_target_markers(ax, point, color=point_color, marker=point_marker, text_offset=point_text_offset)

    setup_log_axis(ax)
    ax.set_xlim(float(data[:, 0].min()), float(data[:, 0].max()))
    ax.set_xlabel(r"$t_g$ [ns]")
    if ylabel:
        ax.set_ylabel("Infidelity / leakage")
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(title, pad=6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "1q_revival_noise_panel.json")
    parser.add_argument("--output", type=Path, default=ROOT / "1q_revival_noise_panel_prx.pdf")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    prx_style()
    payload = load_payload(args.input)

    data23 = rows_to_array(payload["j23_scan"])
    data12 = rows_to_array(payload["j12_scan"])
    x90 = payload["x90"]
    z90 = payload["z90"]

    positive = np.concatenate(
        [
            data23[:, 1],
            data23[:, 2],
            data12[:, 1],
            data12[:, 2],
            np.array([x90["infidelity"], x90["leakage"], z90["infidelity"], z90["leakage"]], dtype=float),
        ]
    )
    positive = positive[positive > 0]
    ymin = 10 ** np.floor(np.log10(positive.min()))
    ymax = 10 ** np.ceil(np.log10(positive.max()))

    fig, axes = plt.subplots(1, 2, sharey=True)

    draw_panel(
        axes[0],
        data23,
        x90,
        title=r"$J_{23}$ pulse",
        point_color=COLORS["x90"],
        point_marker="o",
        point_text_offset=(-12, -16),
        ylabel=True,
    )
    draw_panel(
        axes[1],
        data12,
        z90,
        title=r"$J_{12}$ pulse",
        point_color=COLORS["z90"],
        point_marker="D",
        point_text_offset=(-8, -16),
        ylabel=False,
    )

    axes[0].set_ylim(ymin, ymax)

    panel_label(axes[0], "(a)", x=-0.18, y=1.03)
    panel_label(axes[1], "(b)", x=-0.08, y=1.03)

    legend_handles = [
        Line2D([], [], color=COLORS["infidelity"], lw=1.45, label="Infidelity"),
        Line2D([], [], color=COLORS["leakage"], lw=1.45, ls=(0, (0.45, 2.25)), label="Leakage"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.7,
    )

    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.82, wspace=0.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
