from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch


def style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (9.4, 3.1),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
        }
    )


def confinement(x: np.ndarray, y0: float, centers: np.ndarray) -> np.ndarray:
    width = 0.95
    depth = y0 - 1.05
    softness = 0.18
    wells = np.array([depth + ((x - c) / width) ** 2 for c in centers])
    return -softness * np.log(np.sum(np.exp(-wells / softness), axis=0))


def add_dot(ax: plt.Axes, x: float, y: float, r: float = 0.29) -> None:
    ax.add_patch(Circle((x, y), r, facecolor="#c8ddf4", edgecolor="#99b9d8", linewidth=0.9, alpha=0.92, zorder=3))
    ax.add_patch(Circle((x + 0.05, y - 0.05), 0.23, facecolor="#9fc3e6", edgecolor="none", alpha=0.28, zorder=3))
    ax.add_patch(Circle((x - 0.09, y + 0.09), 0.11, facecolor="white", edgecolor="none", alpha=0.62, zorder=4))


def add_spin(ax: plt.Axes, x: float, y: float, dx: float, dy: float, color: str, label: str, label_dx: float = -0.34) -> None:
    arrow = FancyArrowPatch(
        (x - dx, y - dy),
        (x + dx, y + dy),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.4,
        color=color,
        zorder=6,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(arrow)
    ax.text(x, y + abs(dy) + 0.16, label, ha="center", va="bottom", fontsize=10, color="#404040")


def add_exchange(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    label: str,
    rad: float = -0.72,
    *,
    off: bool = False,
) -> None:
    exchange_color = "#6b5b95"
    label_size = 10
    if off:
        ax.annotate(
            "",
            xy=(x2, y),
            xytext=(x1, y),
            arrowprops=dict(
                arrowstyle="<->",
                lw=1.9,
                color=exchange_color,
                connectionstyle=f"arc3,rad={rad}",
                alpha=0.42,
            ),
            zorder=5,
        )
    else:
        ax.annotate(
            "",
            xy=(x2, y),
            xytext=(x1, y),
            arrowprops=dict(
                arrowstyle="<->",
                lw=1.9,
                color=exchange_color,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=5,
        )
    ax.text(
        (x1 + x2) / 2,
        y + (0.78 if rad < 0 else -0.62),
        label,
        ha="center",
        va="bottom" if rad < 0 else "top",
        fontsize=label_size,
        color=exchange_color,
        alpha=0.75 if off else 1.0,
    )


def draw_vertical_exchange(ax: plt.Axes, x: float, y1: float, y2: float, label: str) -> None:
    exchange_color = "#6b5b95"
    ax.annotate(
        "",
        xy=(x, y2),
        xytext=(x, y1),
        arrowprops=dict(arrowstyle="<->", lw=1.9, color=exchange_color),
        zorder=5,
    )
    ax.text(x + 0.38, (y1 + y2) / 2, label, ha="left", va="center", fontsize=11, color=exchange_color)


def draw_qubit(ax: plt.Axes, xshift: float, yshift: float, suffix: str = "", mirror: bool = False) -> tuple[float, float]:
    centers = np.array([-2.2, 0.0, 2.2]) + xshift
    x = np.linspace(centers[0] - 1.0, centers[-1] + 1.0, 520)
    y = confinement(x, yshift, centers)
    ax.plot(x, y, color="#6b5b95", lw=2.0, zorder=1)

    dot_x = centers.tolist()
    dot_y = [0.34 + yshift, 0.34 + yshift, 0.34 + yshift]

    tilt = 0.13
    spin_specs = (
        [("#b24a3a", rf"$g_1{suffix}$"), ("#4f7f48", rf"$g_2{suffix}$"), ("#355e8d", rf"$g_3{suffix}$")]
        if not mirror
        else [("#8d4f7f", rf"$g_3{suffix}$"), ("#4d8a6b", rf"$g_2{suffix}$"), ("#c7684c", rf"$g_1{suffix}$")]
    )
    spin_vecs = (
        [(tilt, -0.88), (tilt, -0.92), (tilt, -0.74)]
        if not mirror
        else [(tilt, -0.74), (tilt, -0.92), (tilt, -0.88)]
    )
    shaft_shift = 0.2

    for (xd, yd), (dx, dy) in zip(zip(dot_x, dot_y), spin_vecs):
        norm = np.hypot(dx, dy)
        add_dot(ax, xd - shaft_shift * dx / norm, yd - shaft_shift * dy / norm)

    if mirror:
        add_spin(ax, dot_x[0], dot_y[0], spin_vecs[0][0], spin_vecs[0][1], spin_specs[0][0], spin_specs[0][1], label_dx=0.38)
        add_spin(ax, dot_x[1], dot_y[1], spin_vecs[1][0], spin_vecs[1][1], spin_specs[1][0], spin_specs[1][1], label_dx=0.38)
        add_spin(ax, dot_x[2], dot_y[2], spin_vecs[2][0], spin_vecs[2][1], spin_specs[2][0], spin_specs[2][1], label_dx=0.38)
    else:
        add_spin(ax, dot_x[0], dot_y[0], spin_vecs[0][0], spin_vecs[0][1], spin_specs[0][0], spin_specs[0][1], label_dx=-0.38)
        add_spin(ax, dot_x[1], dot_y[1], spin_vecs[1][0], spin_vecs[1][1], spin_specs[1][0], spin_specs[1][1], label_dx=-0.38)
        add_spin(ax, dot_x[2], dot_y[2], spin_vecs[2][0], spin_vecs[2][1], spin_specs[2][0], spin_specs[2][1], label_dx=-0.38)

    if mirror:
        add_exchange(
            ax,
            dot_x[0] + 0.45,
            dot_x[1] - 0.45,
            -0.1 + yshift,
            rf"$J_{{23}}{suffix}=0$",
            rad=-0.72,
            off=True,
        )
        add_exchange(ax, dot_x[1] + 0.45, dot_x[2] - 0.45, -0.1 + yshift, rf"$J_{{12}}{suffix}$", rad=-0.72)
    else:
        add_exchange(ax, dot_x[0] + 0.45, dot_x[1] - 0.45, -0.1 + yshift, rf"$J_{{12}}{suffix}$", rad=-0.72)
        add_exchange(
            ax,
            dot_x[1] + 0.45,
            dot_x[2] - 0.45,
            -0.1 + yshift,
            rf"$J_{{23}}{suffix}=0$",
            rad=-0.72,
            off=True,
        )

    jint_x = dot_x[0] if mirror else dot_x[2]
    jint_y = dot_y[0] if mirror else dot_y[2]
    return jint_x, jint_y


def main() -> None:
    style()

    fig, ax = plt.subplots()

    yshift = 2.85
    x3_left, y3_left = draw_qubit(ax, xshift=-3.8, yshift=yshift, suffix="", mirror=False)
    x3_right, y3_right = draw_qubit(ax, xshift=3.8, yshift=yshift, suffix=r"'", mirror=True)

    add_exchange(ax, x3_left + 0.45, x3_right - 0.45, 0.95 + yshift, r"$J_{\mathrm{int}}$", rad=-0.38)

    ax.set_xlim(-7.6, 7.6)
    ax.set_ylim(0.7, 5.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.03, right=0.97)
    out = Path("/Users/tivakhtel/SOSO-qubit/tess_code/two_qubit_soso.pdf")
    fig.savefig(out, bbox_inches="tight", transparent=True)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
