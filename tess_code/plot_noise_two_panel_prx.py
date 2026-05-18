from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter


ROOT = Path("/Users/tivakhtel/SOSO-qubit/tess_code")


def prx_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (6.2, 3.35),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10.7,
            "axes.titlesize": 10.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.1,
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
            "legend.handlelength": 2.5,
        }
    )


def load_xy(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    data = np.atleast_2d(data)
    order = np.argsort(data[:, 0])
    return data[order]


def load_scan(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    data = np.atleast_1d(data)
    order = np.argsort(data["tPi_ns"])
    return data[order]


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


def entangling_phase_over_pi(t_ns: float, delta: float) -> float:
    t_us = t_ns / 1000.0
    arg = max(1.0 - (delta * t_us) ** 2, 0.0)
    phi = 2.0 * np.pi * np.sqrt(arg)
    phi_mod = np.mod(phi, 2.0 * np.pi)
    return min(phi_mod, 2.0 * np.pi - phi_mod) / np.pi


def annotate_phase(ax: plt.Axes, x_ns: float, y: float, delta: float, dx: float, dy_scale: float) -> None:
    phi_over_pi = entangling_phase_over_pi(x_ns, delta)
    label = r"$\pi$" if np.isclose(phi_over_pi, 1.0, atol=5e-3) else rf"${phi_over_pi:.2f}\pi$"
    ax.plot([x_ns], [y], marker="o", ms=3.5, color="black", zorder=5)
    ax.annotate(
        label,
        xy=(x_ns, y),
        xytext=(x_ns + dx, y * dy_scale),
        textcoords="data",
        fontsize=10.5,
        ha="left" if dx > 0 else "right",
        va="center",
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "black"},
    )


def t_ns_from_delta(delta_mhz: np.ndarray | float) -> np.ndarray | float:
    delta = np.asarray(delta_mhz)
    return np.divide(1000.0 * np.sqrt(3.0), 2.0 * delta, out=np.full_like(delta, np.inf, dtype=float), where=delta != 0)


def delta_ticks(delta_min: float, delta_max: float) -> np.ndarray:
    ticks = np.arange(2.0, 20.5, 1.0)
    ticks = ticks[(ticks >= delta_min - 1e-9) & (ticks <= delta_max + 1e-9)]
    return ticks if len(ticks) >= 2 else np.linspace(delta_min, delta_max, 3)


def delta_tick_label(tick: float) -> str:
    labeled_ticks = {2, 4, 6, 10, 20}
    tick_int = int(round(tick))
    return f"{tick_int:g}" if tick_int in labeled_ticks else ""


def setup_log_axis(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", top=True, right=True, pad=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "noise_two_panel_prx.pdf")
    parser.add_argument("--delta", type=float, default=4.0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    prx_style()

    qs = load_xy(ROOT / "infidelity_qs.csv")
    onef = load_xy(ROOT / "infidelity_1f_uncorr.csv")
    qs_leak = load_xy(ROOT / "leakage_qs.csv")
    onef_leak = load_xy(ROOT / "leakage_1f_uncorr.csv")

    x_cut = canonical_phase_peak_time_ns(args.delta)
    qs = trim_with_endpoint(qs, x_cut)
    onef = trim_with_endpoint(onef, x_cut)
    qs_leak = trim_with_endpoint(qs_leak, x_cut)
    onef_leak = trim_with_endpoint(onef_leak, x_cut)

    scan = load_scan(ROOT / "pi_gate_vs_delta_scan.csv")
    scan = scan[scan["Delta_g3_minus_g3prime"] >= 2.0]
    tpi = scan["tPi_ns"]

    c_qs_inf = "#3f7fbe"
    c_qs_leak = "#4fa66b"
    c_1f_inf = "#d65a4a"
    c_1f_leak = "#e39a3b"
    dotted = (0, (0.45, 2.4))
    lw = 1.35

    all_y = np.concatenate(
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
    positive = all_y[all_y > 0]
    ymin = 10 ** np.floor(np.log10(positive.min()))
    ymax = 10 ** np.ceil(np.log10(positive.max()))

    fig, axes = plt.subplots(1, 2, sharey=True)
    ax0, ax1 = axes

    ax0.plot(qs[:, 0], qs[:, 1], color=c_qs_inf, lw=lw, solid_capstyle="round")
    ax0.plot(qs_leak[:, 0], qs_leak[:, 1], color=c_qs_leak, lw=lw, ls=dotted, dash_capstyle="round")
    ax0.plot(onef[:, 0], onef[:, 1], color=c_1f_inf, lw=lw, solid_capstyle="round")
    ax0.plot(onef_leak[:, 0], onef_leak[:, 1], color=c_1f_leak, lw=lw, ls=dotted, dash_capstyle="round")
    ax0.set_xlim(min(qs[:, 0].min(), onef[:, 0].min()), max(qs[:, 0].max(), onef[:, 0].max()))
    ax0.set_ylim(ymin, ymax)
    ax0.set_xlabel(r"$t_g$ [ns]")
    ax0.set_ylabel("Infidelity / leakage")
    ax0.set_title(r"Fixed $\Delta=4$ MHz", pad=6)
    setup_log_axis(ax0)
    annotate_phase(ax0, ax0.get_xlim()[0], np.interp(ax0.get_xlim()[0], qs[:, 0], qs[:, 1]), args.delta, 10.0, 1.75)
    annotate_phase(ax0, ax0.get_xlim()[1], np.interp(ax0.get_xlim()[1], qs[:, 0], qs[:, 1]), args.delta, -10.5, 1.22)

    ax1.plot(tpi, scan["qs_infidelity"], color=c_qs_inf, lw=lw, solid_capstyle="round")
    ax1.plot(tpi, scan["qs_leakage"], color=c_qs_leak, lw=lw, ls=dotted, dash_capstyle="round")
    ax1.plot(tpi, scan["onef_infidelity"], color=c_1f_inf, lw=lw, solid_capstyle="round")
    ax1.plot(tpi, scan["onef_leakage"], color=c_1f_leak, lw=lw, ls=dotted, dash_capstyle="round")
    ax1.set_xlim(tpi.min(), tpi.max())
    ax1.set_xlabel(r"$t_\pi$ [ns]")
    ax1.set_title(r"$\phi_{\rm ent}=\pi$", pad=6)
    setup_log_axis(ax1)
    ax1.tick_params(labelleft=False)

    secax = ax1.twiny()
    secax.set_xlim(ax1.get_xlim())
    secax.set_xlabel(r"$\Delta$ [MHz]", labelpad=5)
    top_ticks = delta_ticks(scan["Delta_g3_minus_g3prime"].min(), scan["Delta_g3_minus_g3prime"].max())
    tick_positions = t_ns_from_delta(top_ticks)
    secax.set_xticks(tick_positions)
    secax.set_xticklabels([delta_tick_label(tick) for tick in top_ticks])
    secax.tick_params(direction="in", pad=3)

    source_handles = [
        (
            Line2D([], [], color=c_qs_inf, lw=lw),
            Line2D([], [], color=c_qs_leak, lw=lw, ls=dotted, dash_capstyle="round"),
        ),
        (
            Line2D([], [], color=c_1f_inf, lw=lw),
            Line2D([], [], color=c_1f_leak, lw=lw, ls=dotted, dash_capstyle="round"),
        ),
    ]
    metric_handles = [
        Line2D([], [], color="black", lw=lw, label="Infidelity"),
        Line2D([], [], color="black", lw=lw, ls=dotted, label="Leakage", dash_capstyle="round"),
    ]
    handles = [*source_handles, *metric_handles]
    labels = ["Quasistatic", r"$1/f$", "Infidelity", "Leakage"]
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        handlelength=2.7,
        columnspacing=1.6,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.7)},
    )

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.23, top=0.80, wspace=0.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
