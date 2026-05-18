from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
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


def trim_with_endpoint(data: np.ndarray, x_cut: float) -> np.ndarray:
    left = data[data[:, 0] <= x_cut]
    if len(left) == 0:
        return data[:1].copy()
    if np.isclose(left[-1, 0], x_cut):
        return left
    y_cut = np.interp(x_cut, data[:, 0], data[:, 1])
    return np.vstack([left, [x_cut, y_cut]])


def annotate_phase(ax: plt.Axes, x_ns: float, y: float, delta: float, dx: float, dy_scale: float) -> None:
    phi_over_pi = entangling_phase_over_pi(x_ns, delta)
    phase_label = r"$\pi$" if np.isclose(phi_over_pi, 1.0, atol=5e-3) else rf"${phi_over_pi:.2f}\pi$"
    ax.plot([x_ns], [y], marker="o", ms=3.8, color="black", zorder=5)
    ax.annotate(
        phase_label,
        xy=(x_ns, y),
        xytext=(x_ns + dx, y * dy_scale),
        textcoords="data",
        fontsize=11.3,
        ha="left" if dx > 0 else "right",
        va="center",
        arrowprops={"arrowstyle": "-", "lw": 0.75, "color": "black"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qs", required=True, type=Path)
    parser.add_argument("--uncorr", required=True, type=Path)
    parser.add_argument("--halfcorr", required=True, type=Path)
    parser.add_argument("--leakage-qs", required=True, type=Path)
    parser.add_argument("--leakage-uncorr", required=True, type=Path)
    parser.add_argument("--leakage-halfcorr", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--delta", required=True, type=float, help="Signed Delta in inverse microseconds / MHz units used in Mathematica.")
    args = parser.parse_args()

    prx_style()

    qs = load_xy(args.qs)
    uncorr = load_xy(args.uncorr)
    leakage_qs = load_xy(args.leakage_qs)
    leakage_uncorr = load_xy(args.leakage_uncorr)
    halfcorr = load_xy(args.halfcorr)
    leakage_halfcorr = load_xy(args.leakage_halfcorr)

    x_cut = canonical_phase_peak_time_ns(args.delta)
    qs = trim_with_endpoint(qs, x_cut)
    uncorr = trim_with_endpoint(uncorr, x_cut)
    leakage_qs = trim_with_endpoint(leakage_qs, x_cut)
    leakage_uncorr = trim_with_endpoint(leakage_uncorr, x_cut)
    halfcorr = trim_with_endpoint(halfcorr, x_cut)
    leakage_halfcorr = trim_with_endpoint(leakage_halfcorr, x_cut)

    x_all = np.concatenate([qs[:, 0], uncorr[:, 0]])
    y_all = np.concatenate([qs[:, 1], uncorr[:, 1]])
    positive = y_all[y_all > 0]
    ymin = 10 ** np.floor(np.log10(positive.min()))
    ymax = 10 ** np.ceil(np.log10(positive.max()))

    leakage_y_all = np.concatenate([leakage_qs[:, 1], leakage_uncorr[:, 1]])
    leakage_positive = leakage_y_all[leakage_y_all > 0]
    y_all_combined = np.concatenate([positive, leakage_positive])
    ymin = 10 ** np.floor(np.log10(y_all_combined.min()))
    ymax = 10 ** np.ceil(np.log10(y_all_combined.max()))

    fig, ax = plt.subplots()

    c_qs_inf = "#3f7fbe"
    c_qs_leak = "#4fa66b"
    c_1f_inf = "#d65a4a"
    c_1f_leak = "#e39a3b"
    ax.plot(qs[:, 0], qs[:, 1], color=c_qs_inf, lw=1.55, label="Quasistatic infidelity", solid_capstyle="round")
    ax.plot(
        leakage_qs[:, 0],
        leakage_qs[:, 1],
        color=c_qs_leak,
        lw=1.55,
        ls=(0, (0.35, 2.0)),
        dash_capstyle="round",
        label="Quasistatic leakage",
    )
    ax.plot(uncorr[:, 0], uncorr[:, 1], color=c_1f_inf, lw=1.45, label=r"$1/f$ infidelity", solid_capstyle="round")
    ax.plot(
        leakage_uncorr[:, 0],
        leakage_uncorr[:, 1],
        color=c_1f_leak,
        lw=1.45,
        ls=(0, (0.35, 2.0)),
        dash_capstyle="round",
        label=r"$1/f$ leakage",
    )

    ax.set_yscale("log")
    ax.set_xlim(x_all.min(), x_all.max())
    ax.set_ylim(ymin, ymax)

    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.tick_params(which="both", top=True, right=True, pad=4)
    ax.set_xlabel(r"$t_g$ [ns]")
    ax.set_ylabel("")

    ax.set_title(r"$ZZ^\prime$ gate", pad=6)

    x_left = x_all.min()
    x_right = x_all.max()
    y_left = np.interp(x_left, qs[:, 0], qs[:, 1])
    y_right = np.interp(x_right, qs[:, 0], qs[:, 1])

    annotate_phase(ax, x_left, y_left, args.delta, dx=11.0, dy_scale=1.82)
    annotate_phase(ax, x_right, y_right, args.delta, dx=-11.5, dy_scale=1.24)

    source_handles = [
        (
            Line2D([], [], color=c_qs_inf, lw=1.55),
            Line2D([], [], color=c_qs_leak, lw=1.55, ls=(0, (0.35, 2.0)), dash_capstyle="round"),
        ),
        (
            Line2D([], [], color=c_1f_inf, lw=1.45),
            Line2D([], [], color=c_1f_leak, lw=1.45, ls=(0, (0.35, 2.0)), dash_capstyle="round"),
        ),
    ]
    metric_handles = [
        Line2D([], [], color="black", lw=1.55, label="Infidelity"),
        Line2D([], [], color="black", lw=1.55, ls=(0, (0.35, 2.0)), label="Leakage", dash_capstyle="round"),
    ]
    leg_left = ax.legend(
        handles=source_handles,
        labels=["Quasistatic", r"$1/f$"],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        handlelength=2.6,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.7)},
    )
    ax.add_artist(leg_left)
    ax.legend(
        handles=metric_handles,
        loc="lower right",
        bbox_to_anchor=(0.82, 0.02),
        handlelength=2.6,
    )

    fig.tight_layout(pad=0.28)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
