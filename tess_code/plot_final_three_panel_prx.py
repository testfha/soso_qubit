from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

from plot_2q_gate_mechanism import draw_device_panel, draw_pulse, draw_revival_panel


ROOT = Path("/Users/tivakhtel/SOSO-qubit/tess_code")

COLORS = {
    "qs_inf": "#3f7fbe",
    "qs_leak": "#4fa66b",
    "onef_inf": "#d65a4a",
    "onef_leak": "#e39a3b",
    "ink": "#202020",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (11.6, 4.25),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.85,
            "axes.labelsize": 12.0,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.2,
            "ytick.minor.size": 2.2,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "legend.frameon": False,
            "legend.handlelength": 2.4,
        }
    )


def load_xy(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    data = np.atleast_2d(data)
    return data[np.argsort(data[:, 0])]


def load_scan(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    data = np.atleast_1d(data)
    return data[np.argsort(data["tPi_ns"])]


def canonical_phase_peak_time_ns(delta: float) -> float:
    return 1000.0 * np.sqrt(3.0) / (2.0 * abs(delta))


def trim_with_endpoint(data: np.ndarray, x_cut: float) -> np.ndarray:
    left = data[data[:, 0] <= x_cut]
    if len(left) == 0:
        return data[:1].copy()
    if np.isclose(left[-1, 0], x_cut):
        return left
    return np.vstack([left, [x_cut, np.interp(x_cut, data[:, 0], data[:, 1])]])


def entangling_phase_over_pi(t_ns: float, delta: float) -> float:
    t_us = t_ns / 1000.0
    arg = max(1.0 - (delta * t_us) ** 2, 0.0)
    phi = 2.0 * np.pi * np.sqrt(arg)
    phi_mod = np.mod(phi, 2.0 * np.pi)
    return min(phi_mod, 2.0 * np.pi - phi_mod) / np.pi


def annotate_phase(ax: plt.Axes, x_ns: float, y: float, delta: float, dx: float, dy_scale: float) -> None:
    phi_over_pi = entangling_phase_over_pi(x_ns, delta)
    label = r"$\pi$" if np.isclose(phi_over_pi, 1.0, atol=5e-3) else rf"${phi_over_pi:.2f}\pi$"
    ax.plot([x_ns], [y], marker="o", ms=3.0, color=COLORS["ink"], zorder=5)
    ax.annotate(
        label,
        xy=(x_ns, y),
        xytext=(x_ns + dx, y * dy_scale),
        fontsize=10.8,
        ha="left" if dx > 0 else "right",
        va="center",
        arrowprops={"arrowstyle": "-", "lw": 0.75, "color": COLORS["ink"]},
    )


def t_ns_from_delta(delta_mhz: np.ndarray | float) -> np.ndarray | float:
    delta = np.asarray(delta_mhz)
    return np.divide(
        1000.0 * np.sqrt(3.0),
        2.0 * delta,
        out=np.full_like(delta, np.inf, dtype=float),
        where=delta != 0,
    )


def delta_ticks_and_labels(delta_min: float, delta_max: float) -> tuple[np.ndarray, list[str]]:
    ticks = np.arange(2.0, 20.5, 1.0)
    ticks = ticks[(ticks >= delta_min - 1e-9) & (ticks <= delta_max + 1e-9)]
    labels = [f"{int(round(t))}" if int(round(t)) in {2, 4, 6, 10, 20} else "" for t in ticks]
    return ticks, labels


def setup_log_axis(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", top=True, right=True, pad=3)


def add_panel_label(ax: plt.Axes, label: str, x: float, y: float) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18.0,
        fontweight="bold",
        color=COLORS["ink"],
    )


def draw_curves(
    ax: plt.Axes,
    curves: list[tuple[np.ndarray, np.ndarray, str, str]],
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    title: str,
    ylabel: bool,
) -> None:
    dotted = (0, (0.55, 2.3))
    setup_log_axis(ax)
    for x, y, color, style_name in curves:
        lw = 1.38 if style_name == "solid" else 1.18
        ls = "-" if style_name == "solid" else dotted
        ax.plot(x, y, color=color, lw=lw, ls=ls, solid_capstyle="round", dash_capstyle="round")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, labelpad=3)
    if ylabel:
        ax.set_ylabel("Infidelity / leakage", labelpad=8)
    else:
        ax.tick_params(labelleft=False)
    ax.set_box_aspect(1.02)
    ax.text(0.05, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=13.2)


def add_legend(fig: plt.Figure) -> None:
    dotted = (0, (0.55, 2.3))
    handles = [
        Line2D([], [], color=COLORS["qs_inf"], lw=1.38, label="Quasistatic"),
        Line2D([], [], color=COLORS["qs_leak"], lw=1.18, ls=dotted, label=""),
        Line2D([], [], color=COLORS["onef_inf"], lw=1.38, label=r"$1/f$"),
        Line2D([], [], color=COLORS["onef_leak"], lw=1.18, ls=dotted, label=""),
        Line2D([], [], color=COLORS["ink"], lw=1.38, label="Infidelity"),
        Line2D([], [], color=COLORS["ink"], lw=1.18, ls=dotted, label="Leakage"),
    ]
    labels = ["Quasistatic", "", r"$1/f$", "", "Infidelity", "Leakage"]
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.77, 0.02),
        ncol=6,
        fontsize=10.2,
        handlelength=2.1,
        handletextpad=0.5,
        columnspacing=0.8,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "final_three_panel_prx.pdf")
    parser.add_argument("--delta", type=float, default=4.0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    style()

    x_cut = canonical_phase_peak_time_ns(args.delta)
    qs = trim_with_endpoint(load_xy(ROOT / "infidelity_qs.csv"), x_cut)
    onef = trim_with_endpoint(load_xy(ROOT / "infidelity_1f_uncorr.csv"), x_cut)
    qs_leak = trim_with_endpoint(load_xy(ROOT / "leakage_qs.csv"), x_cut)
    onef_leak = trim_with_endpoint(load_xy(ROOT / "leakage_1f_uncorr.csv"), x_cut)
    scan = load_scan(ROOT / "pi_gate_vs_delta_scan.csv")
    scan = scan[scan["Delta_g3_minus_g3prime"] >= 2.0]

    positive = np.concatenate(
        [
            qs[:, 1],
            onef[:, 1],
            qs_leak[:, 1],
            onef_leak[:, 1],
            scan["qs_infidelity"],
            scan["onef_infidelity"],
            scan["qs_leakage"],
            scan["onef_leakage"],
        ]
    )
    positive = positive[positive > 0]
    ylim = (10 ** np.floor(np.log10(positive.min())), 10 ** np.ceil(np.log10(positive.max())))

    fig = plt.figure()
    outer = GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[2.55, 1.0, 1.0],
        left=0.04,
        right=0.985,
        top=0.965,
        bottom=0.16,
        wspace=0.12,
    )

    left = outer[0].subgridspec(3, 1, height_ratios=[0.25, 1.56, 0.92], hspace=0.02)
    ax_pulse = fig.add_subplot(left[0])
    ax_device = fig.add_subplot(left[1])
    ax_revival = fig.add_subplot(left[2])
    ax_b = fig.add_subplot(outer[1])
    ax_c = fig.add_subplot(outer[2], sharey=ax_b)

    draw_pulse(ax_pulse)
    draw_device_panel(ax_device)
    draw_revival_panel(ax_revival)

    draw_curves(
        ax_b,
        [
            (qs[:, 0], qs[:, 1], COLORS["qs_inf"], "solid"),
            (qs_leak[:, 0], qs_leak[:, 1], COLORS["qs_leak"], "dotted"),
            (onef[:, 0], onef[:, 1], COLORS["onef_inf"], "solid"),
            (onef_leak[:, 0], onef_leak[:, 1], COLORS["onef_leak"], "dotted"),
        ],
        xlim=(min(qs[:, 0].min(), onef[:, 0].min()), max(qs[:, 0].max(), onef[:, 0].max())),
        ylim=ylim,
        xlabel=r"$t_g$ [ns]",
        title=r"fixed $\Delta = 4$ MHz",
        ylabel=True,
    )
    annotate_phase(ax_b, ax_b.get_xlim()[0], np.interp(ax_b.get_xlim()[0], qs[:, 0], qs[:, 1]), args.delta, 10.0, 1.75)
    annotate_phase(ax_b, ax_b.get_xlim()[1], np.interp(ax_b.get_xlim()[1], qs[:, 0], qs[:, 1]), args.delta, -9.0, 1.24)

    draw_curves(
        ax_c,
        [
            (scan["tPi_ns"], scan["qs_infidelity"], COLORS["qs_inf"], "solid"),
            (scan["tPi_ns"], scan["qs_leakage"], COLORS["qs_leak"], "dotted"),
            (scan["tPi_ns"], scan["onef_infidelity"], COLORS["onef_inf"], "solid"),
            (scan["tPi_ns"], scan["onef_leakage"], COLORS["onef_leak"], "dotted"),
        ],
        xlim=(scan["tPi_ns"].min(), scan["tPi_ns"].max()),
        ylim=ylim,
        xlabel=r"$t_1$ [ns]",
        title=r"$\phi_{\mathrm{ent}}=\pi$",
        ylabel=False,
    )

    secax = ax_c.twiny()
    secax.set_xlim(ax_c.get_xlim())
    secax.set_xlabel(r"$\Delta$ [MHz]", labelpad=6)
    ticks, labels = delta_ticks_and_labels(scan["Delta_g3_minus_g3prime"].min(), scan["Delta_g3_minus_g3prime"].max())
    secax.set_xticks(t_ns_from_delta(ticks))
    secax.set_xticklabels(labels)
    secax.tick_params(direction="in", pad=4, labelsize=9.8)
    for name in ("bottom", "left", "right"):
        secax.spines[name].set_visible(False)
    secax.spines["top"].set_linewidth(0.85)

    add_panel_label(ax_device, "(a)", x=-0.02, y=1.01)
    add_panel_label(ax_b, "(b)", x=-0.22, y=1.03)
    add_panel_label(ax_c, "(c)", x=-0.22, y=1.03)
    add_legend(fig)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
