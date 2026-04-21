from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter


def load_xy(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    data = np.atleast_2d(data)
    if data.shape[1] != 2:
        raise ValueError(f"{path} must have exactly 2 columns, got shape {data.shape}")
    return data


def prx_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (3.15, 2.55),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.3,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True, type=Path)
    parser.add_argument("--noisy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clean-label", default="Clean")
    parser.add_argument("--noisy-label", default="Quasistatic noise")
    parser.add_argument("--panel-label", default="(a)")
    args = parser.parse_args()

    prx_style()

    clean = load_xy(args.clean)
    noisy = load_xy(args.noisy)

    x = clean[:, 0]
    y_clean = clean[:, 1]
    y_noisy = noisy[:, 1]

    fig, ax = plt.subplots()

    blue = "#1f4e79"
    orange = "#c35a1e"

    ax.plot(
        x,
        y_clean,
        color=blue,
        lw=1.9,
        label=args.clean_label,
        solid_capstyle="round",
        zorder=3,
    )

    ax.plot(
        noisy[:, 0],
        y_noisy,
        color=orange,
        lw=1.9,
        label=args.noisy_label,
        solid_capstyle="round",
        zorder=2,
    )

    positive = np.concatenate([y_clean, y_noisy])
    positive = positive[positive > 0]
    ymin = 10 ** np.floor(np.log10(positive.min()))
    ymax = 10 ** np.ceil(np.log10(positive.max()))

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(min(clean[:, 0].min(), noisy[:, 0].min()), max(clean[:, 0].max(), noisy[:, 0].max()))

    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.tick_params(which="both", top=True, right=True, pad=4)
    ax.set_xlabel(r"$t_g$ [ns]")
    ax.set_ylabel("Gate infidelity")

    ax.text(
        0.03,
        0.96,
        args.panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    ax.text(
        0.26,
        0.84,
        r"$ZZ^\prime$ gate",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02))

    fig.tight_layout(pad=0.28)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
