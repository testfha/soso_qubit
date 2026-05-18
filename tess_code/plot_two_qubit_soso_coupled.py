from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from plot_qubit_passport import (
    PALETTE,
    QUBIT_LABEL_FS,
    add_dot,
    panel_label,
    prx_style,
)


LEFT_DOT_X = np.array([-4.20, -2.52, -0.84])
RIGHT_DOT_X = -LEFT_DOT_X[::-1]
ALL_DOT_X = np.concatenate([LEFT_DOT_X, RIGHT_DOT_X])
DOT_Y_OFFSET = 0.74


def six_dot_potential(x: np.ndarray, centers: np.ndarray = ALL_DOT_X) -> np.ndarray:
    width = 0.82
    depth = -1.68
    softness = 0.10
    wells = np.array([depth + ((x - center) / width) ** 2 for center in centers])
    base = -softness * np.log(np.exp(-wells / softness).sum(axis=0))
    central_barrier = 0.58 * np.exp(-(x / 0.40) ** 2)
    return base + central_barrier


def draw_global_potential(ax: plt.Axes) -> None:
    x = np.linspace(ALL_DOT_X[0] - 1.05, ALL_DOT_X[-1] + 1.05, 1600)
    y = six_dot_potential(x)
    ax.plot(x, y, color=PALETTE["plum"], lw=2.0, zorder=1)
    ax.fill_between(x, y, y.min() - 0.75, color=PALETTE["plum"], alpha=0.08, zorder=0)


def add_coupled_dot_ring(ax: plt.Axes, x: float, y: float) -> None:
    ax.add_patch(
        Circle(
            (x, y),
            0.52,
            facecolor="none",
            edgecolor=PALETTE["coral"],
            linewidth=1.6,
            alpha=0.95,
            zorder=14,
        )
    )


def add_bridge_exchange(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    label: str,
    *,
    color: str = PALETTE["coral"],
    rad: float = -0.34,
    label_offset: float = 0.10,
) -> None:
    arrow = FancyArrowPatch(
        (x1, y),
        (x2, y),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="<->",
        mutation_scale=16,
        linewidth=2.0,
        color=color,
        zorder=13,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(arrow)

    arc_lift = abs(rad) * (x2 - x1) * 0.42
    ax.text(
        0.5 * (x1 + x2),
        y + arc_lift + label_offset,
        label,
        ha="center",
        va="bottom",
        fontsize=QUBIT_LABEL_FS,
        color=color,
        zorder=14,
    )


def add_centered_spin(
    ax: plt.Axes,
    x: float,
    y: float,
    dx: float,
    dy: float,
    color: str,
    label: str,
    *,
    label_dy: float = 0.94,
    linewidth: float = 2.9,
    mutation_scale: float = 18,
) -> None:
    arrow = FancyArrowPatch(
        (x - dx, y - dy),
        (x + dx, y + dy),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        zorder=15,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(arrow)
    ax.text(
        x,
        y + label_dy,
        label,
        ha="center",
        va="bottom",
        fontsize=QUBIT_LABEL_FS,
        color=PALETTE["charcoal"],
        zorder=16,
    )


def add_arc_exchange(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    label: str,
    *,
    color: str = PALETTE["exchange"],
    rad: float = -0.72,
    linewidth: float = 1.9,
    mutation_scale: float = 15,
    label_offset: float = 0.34,
    alpha: float = 1.0,
) -> None:
    arrow = FancyArrowPatch(
        (x1, y),
        (x2, y),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="<->",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        alpha=alpha,
        zorder=12,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(arrow)

    arc_lift = abs(rad) * (x2 - x1) * 0.34
    ax.text(
        0.5 * (x1 + x2),
        y + arc_lift + label_offset,
        label,
        ha="center",
        va="bottom",
        fontsize=QUBIT_LABEL_FS,
        color=color,
        alpha=alpha,
        zorder=13,
    )


def draw_soso_qubit(
    ax: plt.Axes,
    dot_x: np.ndarray,
    dot_y: np.ndarray,
    *,
    suffix: str = "",
    mirrored: bool = False,
) -> None:
    if mirrored:
        spin_colors = [PALETTE["sky"], PALETTE["sage"], PALETTE["coral"]]
        spin_labels = [rf"$g_3{suffix}$", rf"$g_2{suffix}$", rf"$g_1{suffix}$"]
        spin_dirs = [(0.11, 0.70), (-0.11, -0.76), (-0.11, -0.74)]
    else:
        spin_colors = [PALETTE["coral"], PALETTE["sage"], PALETTE["sky"]]
        spin_labels = [rf"$g_1{suffix}$", rf"$g_2{suffix}$", rf"$g_3{suffix}$"]
        spin_dirs = [(-0.11, -0.74), (-0.11, -0.76), (0.11, 0.70)]

    for xd, yd, color in zip(dot_x, dot_y, spin_colors):
        add_dot(ax, xd, yd, r=0.38, color=color, alpha=0.30)

    for xd, yd, (dx, dy), color, label in zip(dot_x, dot_y, spin_dirs, spin_colors, spin_labels):
        add_centered_spin(
            ax,
            xd,
            yd,
            dx,
            dy,
            color,
            label,
            label_dy=0.92,
            linewidth=2.8,
            mutation_scale=18,
        )

    barrier_12 = 0.5 * (dot_x[0] + dot_x[1])
    barrier_23 = 0.5 * (dot_x[1] + dot_x[2])
    exchange_y_12 = six_dot_potential(np.array([barrier_12]))[0] + 0.18
    exchange_y_23 = six_dot_potential(np.array([barrier_23]))[0] + 0.18

    if mirrored:
        add_arc_exchange(
            ax,
            dot_x[0] + 0.40,
            dot_x[1] - 0.40,
            exchange_y_12,
            rf"$J_{{23}}{suffix}=0$",
            color=PALETTE["exchange"],
            rad=-0.68,
            linewidth=1.8,
            mutation_scale=14,
            label_offset=0.18,
            alpha=0.58,
        )
        add_arc_exchange(
            ax,
            dot_x[1] + 0.40,
            dot_x[2] - 0.40,
            exchange_y_23,
            rf"$J_{{12}}{suffix}$",
            color=PALETTE["exchange"],
            rad=-0.68,
            linewidth=1.8,
            mutation_scale=14,
            label_offset=0.18,
        )
    else:
        add_arc_exchange(
            ax,
            dot_x[0] + 0.40,
            dot_x[1] - 0.40,
            exchange_y_12,
            rf"$J_{{12}}{suffix}$",
            color=PALETTE["exchange"],
            rad=-0.68,
            linewidth=1.8,
            mutation_scale=14,
            label_offset=0.18,
        )
        add_arc_exchange(
            ax,
            dot_x[1] + 0.40,
            dot_x[2] - 0.40,
            exchange_y_23,
            rf"$J_{{23}}{suffix}=0$",
            color=PALETTE["exchange"],
            rad=-0.68,
            linewidth=1.8,
            mutation_scale=14,
            label_offset=0.18,
            alpha=0.58,
        )


def main() -> None:
    prx_style()

    fig, ax = plt.subplots(figsize=(7.8, 2.9))
    draw_global_potential(ax)

    dot_y = six_dot_potential(ALL_DOT_X) + DOT_Y_OFFSET
    left_dot_y = dot_y[:3]
    right_dot_y = dot_y[3:]

    draw_soso_qubit(ax, LEFT_DOT_X, left_dot_y)
    draw_soso_qubit(ax, RIGHT_DOT_X, right_dot_y, suffix="'", mirrored=True)

    coupled_y = max(left_dot_y[2], right_dot_y[0]) + 0.34
    add_coupled_dot_ring(ax, LEFT_DOT_X[2], left_dot_y[2])
    add_coupled_dot_ring(ax, RIGHT_DOT_X[0], right_dot_y[0])
    add_bridge_exchange(
        ax,
        LEFT_DOT_X[2] + 0.06,
        RIGHT_DOT_X[0] - 0.06,
        coupled_y,
        r"$J_{\mathrm{int}}$",
        color=PALETTE["coral"],
        rad=-0.30,
        label_offset=0.14,
    )

    panel_label(ax, "(a)", x=0.01, y=0.98)
    ax.set_xlim(-5.45, 5.45)
    ax.set_ylim(-2.65, 1.38)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.subplots_adjust(top=0.97, bottom=0.06, left=0.02, right=0.98)

    out = Path("/Users/tivakhtel/SOSO-qubit/tess_code/two_qubit_soso_coupled.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=400)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
