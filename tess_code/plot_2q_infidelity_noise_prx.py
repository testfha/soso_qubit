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
    order = np.argsort(data[:, 0])
    return data[order]


def prx_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (3.25, 2.7),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
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
            "legend.handlelength": 2.3,
        }
    )


def entangling_phase_over_pi(t_ns: float, delta: float) -> float:
    t_us = t_ns / 1000.0
    arg = max(1.0 - (delta * t_us) ** 2, 0.0)
    phi = 2.0 * np.pi * np.sqrt(arg)
    phi_mod = np.mod(phi, 2.0 * np.pi)
    phi_ent = min(phi_mod, 2.0 * np.pi - phi_mod)
    return phi_ent / np.pi


def canonical_phase_peak_time_ns(delta: float) -> float:
    return 1000.0 * np.sqrt(3.0) / (2.0 * abs(delta))


def annotate_phase(ax: plt.Axes, x_ns: float, y: float, delta: float, dx: float, dy_scale: float) -> None:
    phi_over_pi = entangling_phase_over_pi(x_ns, delta)
    ax.plot([x_ns], [y], marker="o", ms=3.8, color="black", zorder=5)
    ax.annotate(
        rf"${phi_over_pi:.2f}\pi$",
        xy=(x_ns, y),
        xytext=(x_ns + dx, y * dy_scale),
        textcoords="data",
        fontsize=8.3,
        ha="left" if dx > 0 else "right",
        va="center",
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "black"},
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.92},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qs", required=True, type=Path)
    parser.add_argument("--uncorr", required=True, type=Path)
    parser.add_argument("--halfcorr", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--delta", required=True, type=float, help="Signed Delta in inverse microseconds / MHz units used in Mathematica.")
    args = parser.parse_args()

    prx_style()

    qs = load_xy(args.qs)
    uncorr = load_xy(args.uncorr)
    halfcorr = load_xy(args.halfcorr)

    x_cut = canonical_phase_peak_time_ns(args.delta)
    qs = qs[qs[:, 0] <= x_cut]
    uncorr = uncorr[uncorr[:, 0] <= x_cut]
    halfcorr = halfcorr[halfcorr[:, 0] <= x_cut]

    x_all = np.concatenate([qs[:, 0], uncorr[:, 0], halfcorr[:, 0]])
    y_all = np.concatenate([qs[:, 1], uncorr[:, 1], halfcorr[:, 1]])
    positive = y_all[y_all > 0]
    ymin = 10 ** np.floor(np.log10(positive.min()))
    ymax = 10 ** np.ceil(np.log10(positive.max()))

    fig, ax = plt.subplots()

    ax.plot(qs[:, 0], qs[:, 1], color="#1f4e79", lw=1.9, label="Quasistatic", solid_capstyle="round")
    ax.plot(uncorr[:, 0], uncorr[:, 1], color="#c35a1e", lw=1.8, label=r"$1/f,\ c=0$", solid_capstyle="round")
    ax.plot(halfcorr[:, 0], halfcorr[:, 1], color="#5b8f29", lw=1.8, label=r"$1/f,\ c=0.5$", solid_capstyle="round")

    ax.set_yscale("log")
    ax.set_xlim(x_all.min(), x_all.max())
    ax.set_ylim(ymin, ymax)

    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.tick_params(which="both", top=True, right=True, pad=4)
    ax.set_xlabel(r"$t_g$ [ns]")
    ax.set_ylabel("Gate infidelity")

    ax.text(0.24, 0.86, r"$ZZ^\prime$ gate", transform=ax.transAxes, ha="left", va="top", fontsize=10)

    x_left = x_all.min()
    x_right = x_all.max()
    y_left = np.interp(x_left, qs[:, 0], qs[:, 1])
    y_right = np.interp(x_right, qs[:, 0], qs[:, 1])

    annotate_phase(ax, x_left, y_left, args.delta, dx=3.0, dy_scale=1.7)
    annotate_phase(ax, x_right, y_right, args.delta, dx=-3.0, dy_scale=1.7)

    ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.02))

    fig.tight_layout(pad=0.28)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
