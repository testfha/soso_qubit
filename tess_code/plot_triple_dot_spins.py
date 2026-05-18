from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch


def style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (6.2, 2.8),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
        }
    )


def confinement(x: np.ndarray) -> np.ndarray:
    centers = np.array([-2.2, 0.0, 2.2])
    width = 0.95
    depth = -1.05
    softness = 0.18
    wells = np.array([depth + ((x - c) / width) ** 2 for c in centers])
    return -softness * np.log(np.sum(np.exp(-wells / softness), axis=0))


def add_dot(ax: plt.Axes, x: float, y: float, r: float = 0.38) -> None:
    ax.add_patch(
        Circle(
            (x, y),
            r,
            facecolor="#c8ddf4",
            edgecolor="#99b9d8",
            linewidth=0.9,
            alpha=0.92,
            zorder=3,
        )
    )
    ax.add_patch(
        Circle(
            (x + 0.06, y - 0.06),
            0.31,
            facecolor="#9fc3e6",
            edgecolor="none",
            alpha=0.28,
            zorder=3,
        )
    )
    ax.add_patch(
        Circle(
            (x - 0.12, y + 0.12),
            0.15,
            facecolor="white",
            edgecolor="none",
            alpha=0.62,
            zorder=4,
        )
    )


def add_spin(
    ax: plt.Axes,
    x: float,
    y: float,
    dx: float,
    dy: float,
    color: str,
    label: str,
    label_side: str = "left",
    label_dx: float | None = None,
    label_dy: float | None = None,
    linewidth: float = 2.4,
    mutation_scale: float = 16,
) -> None:
    arrow = FancyArrowPatch(
        (x - dx, y - dy),
        (x + dx, y + dy),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        zorder=6,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(arrow)
    x_text = x - 0.34 if label_side == "left" else x + 0.34
    y_text = y + 0.58 * dy
    ha = "right" if label_side == "left" else "left"
    if label_dx is not None:
        x_text = x + label_dx
        ha = "left"
    if label_dy is not None:
        y_text = y + label_dy
    ax.text(x_text, y_text, label, ha=ha, va="center", fontsize=16, color="#404040")


def add_exchange(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    label: str,
    linewidth: float = 1.9,
    label_fontsize: float = 17,
    mutation_scale: float = 28,
) -> None:
    exchange_color = "#6b5b95"
    ax.annotate(
        "",
        xy=(x2, y),
        xytext=(x1, y),
        arrowprops=dict(
            arrowstyle="<->",
            lw=linewidth,
            color=exchange_color,
            connectionstyle="arc3,rad=-0.72",
            mutation_scale=mutation_scale,
        ),
        zorder=5,
    )
    ax.text((x1 + x2) / 2, y + 0.78, label, ha="center", va="bottom", fontsize=label_fontsize, color=exchange_color)


def main() -> None:
    style()

    fig, ax = plt.subplots()
    yshift = -0.72

    x = np.linspace(-3.18, 3.18, 520)
    y = confinement(x) + yshift
    ax.plot(x, y, color="#6b5b95", lw=2.0, zorder=1)

    dot_x = [-2.2, 0.0, 2.2]
    dot_y = [0.42 + yshift, 0.42 + yshift, 0.42 + yshift]

    for xd, yd in zip(dot_x, dot_y):
        add_dot(ax, xd, yd)

    tilt = 0.13
    add_spin(ax, dot_x[0], dot_y[0], -tilt, -0.88, "#b24a3a", r"$g_1$", label_dx=-0.34, label_dy=0.92)
    add_spin(ax, dot_x[1], dot_y[1], -tilt, -0.92, "#4f7f48", r"$g_2$", label_dx=-0.34, label_dy=0.92)
    add_spin(ax, dot_x[2], dot_y[2], tilt, 0.74, "#355e8d", r"$g_3$", label_dx=-0.34, label_dy=0.92)

    add_exchange(ax, -1.75, -0.45, -0.1 + yshift, r"$J_{12}$")
    add_exchange(ax, 0.45, 1.75, -0.1 + yshift, r"$J_{23}$")

    ax.set_xlim(-3.45, 3.45)
    ax.set_ylim(-2.45, 0.95)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.03, right=0.97)
    out = Path("/Users/tivakhtel/SOSO-qubit/tess_code/triple_dot_spins.pdf")
    fig.savefig(out, bbox_inches="tight", transparent=True)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
