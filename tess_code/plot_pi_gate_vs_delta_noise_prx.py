from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import LogLocator, NullFormatter


def prx_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (3.35, 2.75),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10.8,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 7.9,
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


def load_scan(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    data = np.atleast_1d(data)
    order = np.argsort(data["tPi_ns"])
    return data[order]


def delta_from_t_ns(t_ns: np.ndarray | float) -> np.ndarray | float:
    t = np.asarray(t_ns)
    return np.divide(1000.0 * np.sqrt(3.0), 2.0 * t, out=np.full_like(t, np.inf, dtype=float), where=t != 0)


def t_ns_from_delta(delta_mhz: np.ndarray | float) -> np.ndarray | float:
    delta = np.asarray(delta_mhz)
    return np.divide(1000.0 * np.sqrt(3.0), 2.0 * delta, out=np.full_like(delta, np.inf, dtype=float), where=delta != 0)


def nice_delta_ticks(delta_min: float, delta_max: float) -> np.ndarray:
    candidates = np.arange(2.0, 20.5, 1.0)
    ticks = candidates[(candidates >= delta_min - 1e-9) & (candidates <= delta_max + 1e-9)]
    if len(ticks) >= 2:
        return ticks
    return np.linspace(delta_min, delta_max, 3)


def delta_tick_label(tick: float) -> str:
    labeled_ticks = {2, 4, 6, 10, 20}
    tick_int = int(round(tick))
    return f"{tick_int:g}" if tick_int in labeled_ticks else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/Users/tivakhtel/SOSO-qubit/tess_code/pi_gate_vs_delta_scan.csv"),
        help="CSV exported from Mathematica with tPi_ns, delta, clean, QS, and 1/f columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/tivakhtel/SOSO-qubit/tess_code/pi_gate_vs_delta_noise_prx.pdf"),
    )
    parser.add_argument("--no-show", action="store_true", help="Save files without opening an interactive figure window.")
    args = parser.parse_args()

    prx_style()
    data = load_scan(args.input)

    data = data[data["Delta_g3_minus_g3prime"] >= 2.0]
    t = data["tPi_ns"]
    qs_inf = data["qs_infidelity"]
    qs_leak = data["qs_leakage"]
    onef_inf = data["onef_infidelity"]
    onef_leak = data["onef_leakage"]

    positive = np.concatenate([qs_inf, qs_leak, onef_inf, onef_leak])
    positive = positive[positive > 0]
    ymin = 10 ** np.floor(np.log10(positive.min()))
    ymax = 10 ** np.ceil(np.log10(positive.max()))

    fig, ax = plt.subplots()

    c_qs_inf = "#3f7fbe"
    c_qs_leak = "#4fa66b"
    c_1f_inf = "#d65a4a"
    c_1f_leak = "#e39a3b"
    dotted = (0, (0.45, 2.4))
    lw = 1.45

    ax.plot(t, qs_inf, color=c_qs_inf, lw=lw, solid_capstyle="round")
    ax.plot(t, qs_leak, color=c_qs_leak, lw=lw, ls=dotted, dash_capstyle="round")
    ax.plot(t, onef_inf, color=c_1f_inf, lw=lw, solid_capstyle="round")
    ax.plot(t, onef_leak, color=c_1f_leak, lw=lw, ls=dotted, dash_capstyle="round")

    ax.set_yscale("log")
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$t_\pi$ [ns]")
    ax.set_ylabel("Infidelity / leakage")
    ax.set_title(r"$ZZ^\prime$ gate, $\phi_{\rm ent}=\pi$", pad=7)

    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", top=False, right=True, pad=4)

    secax = ax.secondary_xaxis("top", functions=(delta_from_t_ns, t_ns_from_delta))
    secax.set_xlabel(r"$\Delta g_3$ [MHz]", labelpad=5)
    delta_vals = delta_from_t_ns(np.array([t.max(), t.min()]))
    top_ticks = nice_delta_ticks(delta_vals.min(), delta_vals.max())
    secax.set_xticks(top_ticks)
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
    leg_left = ax.legend(
        handles=source_handles,
        labels=["Quasistatic", r"$1/f$"],
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        handlelength=2.7,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.7)},
    )
    ax.add_artist(leg_left)
    ax.legend(
        handles=metric_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        handlelength=2.7,
    )

    fig.tight_layout(pad=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
