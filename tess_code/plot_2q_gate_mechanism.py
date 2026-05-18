from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from plot_two_qubit_soso import confinement


ROOT = Path("/Users/tivakhtel/SOSO-qubit/tess_code")

PALETTE = {
    "coral": "#E07A5F",
    "sage": "#81B29A",
    "navy": "#3D405B",
    "gold": "#F2CC8F",
    "amber": "#D8B26A",
    "slate": "#5C6B73",
    "sky": "#7FB3D3",
    "plum": "#9A8C98",
    "forest": "#4F7A68",
    "indigo": "#6175B7",
    "charcoal": "#2F2F2F",
    "cream": "#F8F5F1",
    "boxfill": "#EEF2E6",
}

PULSE_TICK_FS = 10.4
DEVICE_LABEL_FS = 12.6
DEVICE_TEXT_FS = 11.8
REVIVAL_LABEL_FS = 10.8
REVIVAL_TEXT_FS = 12.8
FORMULA_FS = 15.0


def style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (8.3, 4.05),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
        }
    )


def add_colored_dot(ax: plt.Axes, x: float, y: float, color: str, r: float = 0.38) -> None:
    ax.add_patch(
        Circle(
            (x, y),
            r,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
            alpha=0.28,
            zorder=8,
        )
    )
    ax.add_patch(
        Circle(
            (x + 0.05, y - 0.04),
            0.29,
            facecolor=color,
            edgecolor="none",
            alpha=0.16,
            zorder=9,
        )
    )
    ax.add_patch(
        Circle(
            (x - 0.10, y + 0.12),
            0.14,
            facecolor="white",
            edgecolor="none",
            alpha=0.50,
            zorder=10,
        )
    )


def add_spin_arrow(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    up: bool,
    color: str,
    label: str,
) -> None:
    if up:
        start = (x - 0.12, y - 0.72)
        end = (x + 0.16, y + 0.74)
    else:
        start = (x + 0.12, y + 0.72)
        end = (x - 0.16, y - 0.74)

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=3.0,
            color=color,
            zorder=15,
            capstyle="round",
            joinstyle="round",
        )
    )
    ax.text(
        x,
        y + 0.90,
        label,
        ha="center",
        va="bottom",
        fontsize=DEVICE_LABEL_FS,
        color=PALETTE["charcoal"],
    )


def add_exchange_arc(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    label: str,
    *,
    rad: float = -0.68,
    off: bool = False,
    label_dx: float = 0.0,
    label_dy: float = 0.72,
    fontsize: float = DEVICE_LABEL_FS,
) -> None:
    alpha = 0.45 if off else 1.0
    ax.annotate(
        "",
        xy=(x2, y),
        xytext=(x1, y),
        arrowprops=dict(
            arrowstyle="<->",
            lw=2.0,
            color=PALETTE["indigo"],
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=16,
        ),
        zorder=6,
    )
    ax.text(
        (x1 + x2) / 2 + label_dx,
        y + label_dy,
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=PALETTE["indigo"],
        alpha=alpha,
    )


def rounded_box(
    ax: plt.Axes,
    center: tuple[float, float],
    text: str,
    *,
    w: float,
    h: float,
    fontsize: float,
) -> None:
    x, y = center
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=PALETTE["boxfill"],
        edgecolor=PALETTE["slate"],
        linewidth=1.0,
        alpha=0.88,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=PALETTE["charcoal"],
        zorder=4,
    )


def draw_pulse(ax: plt.Axes) -> None:
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.035, 0.36)
    ax.axis("off")

    t = np.array([0.35, 0.35, 0.43, 0.43, 0.57, 0.57, 0.65, 0.65])
    j = np.array([0.0, 0.0, 0.00, 0.30, 0.30, 0.00, 0.0, 0.0])

    ax.fill_between(t, j, alpha=0.12, color=PALETTE["indigo"])
    ax.plot(t, j, lw=2.2, color=PALETTE["indigo"], solid_capstyle="round")
    ax.text(0.50, 0.19, r"$J_{\rm int}$", ha="center", va="center", fontsize=11.2, color=PALETTE["indigo"])
    ax.text(0.43, -0.010, r"$0$", ha="center", va="top", fontsize=PULSE_TICK_FS, color=PALETTE["charcoal"])
    ax.text(0.57, -0.010, r"$t_1$", ha="center", va="top", fontsize=PULSE_TICK_FS, color=PALETTE["charcoal"])


def draw_device_panel(ax: plt.Axes) -> None:
    yshift = 2.48
    centers = np.array([-5.55, -3.30, -1.05, 1.05, 3.30, 5.55])
    xlim = (-6.22, 6.22)
    x = np.linspace(xlim[0], xlim[1], 1200)
    y = confinement(x, yshift, centers)
    dot_y = 0.40 + yshift
    exchange_y = dot_y - 0.50

    ax.plot(x, y, color=PALETTE["plum"], lw=2.8, zorder=1)
    ax.fill_between(x, y, y.min() - 0.42, color=PALETTE["plum"], alpha=0.055, zorder=0)

    dot_colors = [
        PALETTE["coral"],
        PALETTE["sage"],
        PALETTE["sky"],
        PALETTE["sky"],
        PALETTE["sage"],
        PALETTE["coral"],
    ]
    dot_labels = [r"$g_1$", r"$g_2$", r"$g_3$", r"$g_3^\prime$", r"$g_2^\prime$", r"$g_1^\prime$"]
    spin_up = [False, False, True, True, False, False]

    for xd, color, label, up in zip(centers, dot_colors, dot_labels, spin_up):
        add_colored_dot(ax, xd, dot_y, color, r=0.40)
        add_spin_arrow(ax, xd, dot_y, up=up, color=color, label=label)

    add_exchange_arc(ax, centers[0] + 0.36, centers[1] - 0.36, exchange_y, r"$J_{12}$", label_dy=0.66)
    add_exchange_arc(
        ax,
        centers[1] + 0.36,
        centers[2] - 0.36,
        exchange_y,
        r"$J_{23}=0$",
        off=True,
        label_dx=-0.12,
        label_dy=0.56,
        fontsize=DEVICE_LABEL_FS - 0.4,
    )
    add_exchange_arc(
        ax,
        centers[3] + 0.36,
        centers[4] - 0.36,
        exchange_y,
        r"$J_{23}^\prime=0$",
        off=True,
        label_dx=0.12,
        label_dy=0.56,
        fontsize=DEVICE_LABEL_FS - 0.4,
    )
    add_exchange_arc(ax, centers[4] + 0.36, centers[5] - 0.36, exchange_y, r"$J_{12}^\prime$", label_dy=0.66)

    ax.annotate(
        "",
        xy=(centers[3] - 0.22, dot_y + 1.12),
        xytext=(centers[2] + 0.22, dot_y + 1.12),
        arrowprops=dict(
            arrowstyle="<->",
            lw=2.3,
            color=PALETTE["coral"],
            connectionstyle="arc3,rad=-0.42",
            mutation_scale=18,
        ),
        zorder=12,
    )
    ax.text(
        0.0,
        dot_y + 1.52,
        r"$J_{\rm int}$",
        ha="center",
        va="bottom",
        fontsize=DEVICE_LABEL_FS + 0.4,
        color=PALETTE["coral"],
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(0.35, 5.45)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def state_box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str) -> None:
    _ = color
    rounded_box(ax, xy, text, w=2.45, h=0.90, fontsize=REVIVAL_TEXT_FS - 0.5)


def curved_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], rad: float, color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=2.2,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=4,
        )
    )


def draw_revival_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.0)
    ax.axis("off")
    logic_color = PALETTE["indigo"]
    leak_color = PALETTE["amber"]
    arrow_color = PALETTE["plum"]
    yshift = 0.26

    state_box(ax, (2.65, 1.78 + yshift), r"$|01^\prime\rangle,\ |10^\prime\rangle$", logic_color)
    state_box(ax, (7.35, 1.78 + yshift), r"$|\bar 0\bar 1^\prime\rangle,\ |\bar 1\bar 0^\prime\rangle$", leak_color)
    ax.text(2.65, 2.43 + yshift, "logical", ha="center", va="bottom", fontsize=REVIVAL_LABEL_FS, color=logic_color)
    ax.text(7.35, 2.43 + yshift, "leakage", ha="center", va="bottom", fontsize=REVIVAL_LABEL_FS, color=leak_color)

    curved_arrow(ax, (4.15, 2.18 + yshift), (5.85, 2.18 + yshift), -0.28, arrow_color)
    curved_arrow(ax, (5.85, 1.34 + yshift), (4.15, 1.34 + yshift), -0.28, arrow_color)
    ax.text(
        5.0,
        0.12,
        r"population returns to the logical subspace at $t_1$",
        ha="center",
        va="center",
        fontsize=DEVICE_TEXT_FS - 0.4,
        color=PALETTE["charcoal"],
    )
    ax.text(
        5.0,
        1.76 + yshift,
        r"leakage revival",
        ha="center",
        va="center",
        fontsize=REVIVAL_TEXT_FS - 0.7,
        color=PALETTE["charcoal"],
    )
    ax.text(
        5.0,
        -1.10,
        r"$U_{\rm log}(t_1)=\mathrm{diag}\!\left(e^{-iJ_{\rm int}t_1/4},\,-e^{i(J_{\rm int}t_1/4+\Delta t_1/2)},\,-e^{i(J_{\rm int}t_1/4-\Delta t_1/2)},\,e^{-iJ_{\rm int}t_1/4}\right)$",
        ha="center",
        va="top",
        fontsize=FORMULA_FS - 0.5,
        color=PALETTE["charcoal"],
        clip_on=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "2q_gate_mechanism.pdf")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    style()
    fig = plt.figure()
    gs = GridSpec(3, 1, figure=fig, height_ratios=[0.11, 1.61, 0.74], hspace=0.025)

    draw_pulse(fig.add_subplot(gs[0]))
    draw_device_panel(fig.add_subplot(gs[1]))
    draw_revival_panel(fig.add_subplot(gs[2]))

    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", transparent=True)
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight", transparent=True)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
