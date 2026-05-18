from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PALETTE = {
    "coral": "#E07A5F",
    "sage": "#81B29A",
    "navy": "#3D405B",
    "gold": "#F2CC8F",
    "slate": "#5C6B73",
    "sky": "#7FB3D3",
    "plum": "#9A8C98",
    "charcoal": "#2F2F2F",
    "cream": "#F8F5F1",
    "boxfill": "#EEF2E6",
    "exchange": "#6D5A9E",
}

FONT_FAMILY = "serif"
MATHTEXT_FONT = "cm"

PANEL_LABEL_FS = 11
TITLE_FS = 11
LABEL_FS = 10
TEXT_FS = 9.5
TICK_FS = 8.5
QUBIT_LABEL_FS = 10

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
ID2 = np.eye(2, dtype=complex)


def prx_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.65),
            "figure.dpi": 300,
            "figure.facecolor": "white",
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.02,
            "font.family": FONT_FAMILY,
            "font.size": 9,
            "mathtext.fontset": MATHTEXT_FONT,
            "axes.linewidth": 0.6,
            "axes.labelsize": LABEL_FS,
            "axes.titlesize": TITLE_FS,
            "axes.titleweight": "medium",
            "axes.labelpad": 3,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": PALETTE["slate"],
            "axes.labelcolor": PALETTE["charcoal"],
            "xtick.labelsize": TICK_FS,
            "ytick.labelsize": TICK_FS,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.top": True,
            "ytick.right": True,
            "xtick.color": PALETTE["slate"],
            "ytick.color": PALETTE["slate"],
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "legend.handlelength": 1.5,
        }
    )


def confinement(x):
    return 0.08 * (x**2 - 9) * np.cos(0.95 * x) - 1.2


def add_dot(ax, x, y, r=0.5, color="#7FB3D3", alpha=0.30):
    circle = plt.Circle(
        (x, y),
        r,
        fc=color,
        ec=color,
        lw=1.4,
        alpha=alpha,
        zorder=10,
    )
    ax.add_patch(circle)

    inner = plt.Circle(
        (x + 0.08 * r, y - 0.02 * r),
        0.72 * r,
        fc=color,
        ec="none",
        alpha=0.14,
        zorder=11,
    )
    ax.add_patch(inner)

    highlight = plt.Circle(
        (x - 0.28 * r, y + 0.28 * r),
        0.34 * r,
        fc="white",
        ec="none",
        alpha=0.38,
        zorder=12,
    )
    ax.add_patch(highlight)


def add_spin(
    ax,
    x,
    y,
    tilt,
    length,
    color,
    label,
    label_dx=0,
    label_dy=0,
    linewidth=2.7,
    mutation_scale=18,
):
    dx, dy = length * np.sin(tilt), length * np.cos(tilt)

    arrow = FancyArrowPatch(
        (x, y - 0.13),
        (x + dx, y + dy),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        lw=linewidth,
        color=color,
        zorder=15,
    )
    ax.add_patch(arrow)

    ax.text(
        x + label_dx,
        y + label_dy,
        label,
        fontsize=QUBIT_LABEL_FS,
        color=PALETTE["charcoal"],
        ha="center",
        va="bottom",
        zorder=20,
    )


def add_exchange(ax, x1, x2, y, label, **kw):
    color = kw.get("color", PALETTE["exchange"])
    lw = kw.get("linewidth", 1.8)
    ms = kw.get("mutation_scale", 14)
    fs = kw.get("label_fontsize", QUBIT_LABEL_FS)
    height = kw.get("height", 0.52)

    xm = 0.5 * (x1 + x2)

    left = FancyArrowPatch(
        (xm, y + height),
        (x1, y),
        connectionstyle="arc3,rad=0.25",
        arrowstyle="-|>",
        mutation_scale=ms,
        lw=lw,
        color=color,
        zorder=20,
    )

    right = FancyArrowPatch(
        (xm, y + height),
        (x2, y),
        connectionstyle="arc3,rad=-0.25",
        arrowstyle="-|>",
        mutation_scale=ms,
        lw=lw,
        color=color,
        zorder=20,
    )

    ax.add_patch(left)
    ax.add_patch(right)

    ax.text(
        xm,
        y + height + 0.16,
        label,
        ha="center",
        va="bottom",
        fontsize=fs,
        color=color,
        zorder=25,
    )


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


def j_idle(g1, g2, g3):
    return ((g2 - g1) ** 2 - (2 * g3 - g1 - g2) ** 2) / (2 * (2 * g3 - g1 - g2))


def mz_minus_half_indices():
    sz_tot = 0.5 * (
        kron3(SIGMA_Z, ID2, ID2)
        + kron3(ID2, SIGMA_Z, ID2)
        + kron3(ID2, ID2, SIGMA_Z)
    )
    return np.where(np.isclose(np.real(np.diag(sz_tot)), -0.5))[0]


def hamiltonian(j12, j23, g1, g2, g3):
    zeeman = 0.5 * (
        g1 * kron3(SIGMA_Z, ID2, ID2)
        + g2 * kron3(ID2, SIGMA_Z, ID2)
        + g3 * kron3(ID2, ID2, SIGMA_Z)
    )

    def exchange(sig_a, sig_b, pos):
        if pos == 12:
            return 0.25 * kron3(sig_a, sig_b, ID2)
        return 0.25 * kron3(ID2, sig_a, sig_b)

    exch12 = j12 * (
        exchange(SIGMA_X, SIGMA_X, 12)
        + exchange(SIGMA_Y, SIGMA_Y, 12)
        + exchange(SIGMA_Z, SIGMA_Z, 12)
    )

    exch23 = j23 * (
        exchange(SIGMA_X, SIGMA_X, 23)
        + exchange(SIGMA_Y, SIGMA_Y, 23)
        + exchange(SIGMA_Z, SIGMA_Z, 23)
    )

    return zeeman + exch12 + exch23


def compute_spectrum(j_vals, g1, g2, g3):
    idx = mz_minus_half_indices()
    spectrum = []

    for j12 in j_vals:
        ham = hamiltonian(j12, 0.0, g1, g2, g3)
        evals = np.sort(np.real(np.linalg.eigvalsh(ham[np.ix_(idx, idx)])))
        spectrum.append(evals)

    spectrum = np.array(spectrum)
    return spectrum - spectrum.min()


def panel_label(ax, label, x=-0.10, y=1.08):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FS,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PALETTE["charcoal"],
        clip_on=False,
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
    )


def rounded_box(ax, center, text, w=0.32, h=0.16, fs=TEXT_FS):
    x, y = center

    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.04",
        facecolor=PALETTE["boxfill"],
        edgecolor=PALETTE["slate"],
        linewidth=0.6,
        alpha=0.9,
        zorder=5,
    )
    ax.add_patch(patch)

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        zorder=10,
        color=PALETTE["charcoal"],
    )


def draw_qubit_schematic(ax):
    yshift = -0.58

    x = np.linspace(-3.0, 3.0, 400)
    y = confinement(x) + yshift

    ax.plot(x, y, color=PALETTE["plum"], lw=2.0, zorder=1)
    ax.fill_between(x, y, y.min() - 0.45, color=PALETTE["plum"], alpha=0.08)

    dot_x = [-1.95, 0.0, 1.95]
    dot_y = [0.34 + yshift] * 3

    spin_colors = [PALETTE["coral"], PALETTE["sage"], PALETTE["sky"]]
    spin_labels = [r"$g_1$", r"$g_2$", r"$g_3$"]

    for xd, yd, col in zip(dot_x, dot_y, spin_colors):
        add_dot(ax, xd, yd, r=0.38, color=col, alpha=0.30)

    spin_dirs = [
        (-0.12, -0.82),
        (-0.12, -0.84),
        (0.12, 0.72),
    ]

    label_positions = [
        (-0.24, 0.76),
        (-0.24, 0.76),
        (-0.24, 0.76),
    ]

    for i, (xd, yd) in enumerate(zip(dot_x, dot_y)):
        tilt, length = spin_dirs[i]
        label_dx, label_dy = label_positions[i]

        add_spin(
            ax,
            xd,
            yd,
            tilt,
            length,
            spin_colors[i],
            spin_labels[i],
            label_dx=label_dx,
            label_dy=label_dy,
            linewidth=2.7,
            mutation_scale=18,
        )

    barrier_12 = -0.975
    barrier_23 = 0.975
    half_width = 0.58

    add_exchange(
        ax,
        barrier_12 - half_width,
        barrier_12 + half_width,
        -0.18 + yshift,
        r"$J_{12}$",
        height=0.52,
        linewidth=1.7,
        label_fontsize=QUBIT_LABEL_FS,
        mutation_scale=13,
        color=PALETTE["exchange"],
    )

    add_exchange(
        ax,
        barrier_23 - half_width,
        barrier_23 + half_width,
        -0.18 + yshift,
        r"$J_{23}$",
        height=0.52,
        linewidth=1.7,
        label_fontsize=QUBIT_LABEL_FS,
        mutation_scale=13,
        color=PALETTE["exchange"],
    )

    ax.set_xlim(-3.05, 3.05)
    ax.set_ylim(-2.12, 0.82)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_spectrum(ax):
    g1, g2, g3 = 53.0, 74.0, 47.0

    j_id = j_idle(g1, g2, g3)
    j_vals = np.linspace(0, 1.6 * j_id, 400)
    spec = compute_spectrum(j_vals, g1, g2, g3)

    colors = [PALETTE["sage"], PALETTE["plum"], PALETTE["coral"]]
    labels = [r"$|0\rangle$", r"$|1\rangle$", r"leak"]

    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.plot(j_vals, spec[:, i], lw=1.5, color=color, label=label)

    y_id = 0.5 * (
        np.interp(j_id, j_vals, spec[:, 0])
        + np.interp(j_id, j_vals, spec[:, 1])
    )

    ax.scatter(
        [j_id],
        [y_id],
        s=50,
        facecolors="none",
        edgecolors=PALETTE["coral"],
        linewidths=1.5,
        zorder=10,
    )

    ax.annotate(
        "idle",
        (j_id, y_id),
        xytext=(j_id + 0.8, y_id + 2.5),
        fontsize=TEXT_FS,
        color=PALETTE["coral"],
        arrowprops=dict(
            arrowstyle="-",
            lw=0.5,
            color=PALETTE["coral"],
            alpha=0.6,
        ),
    )

    ax.set_xlabel(r"$J_{12}$ (MHz)")
    ax.set_ylabel(r"$E$ (MHz)", labelpad=4)

    ax.set_title(
        r"$m_z = -\frac{1}{2}$, $J_{23} = 0$",
        fontsize=TITLE_FS,
        pad=4,
    )

    ax.set_xlim(0, j_vals.max())
    ax.set_ylim(-0.5, spec.max() + 3)
    ax.legend(loc="upper left", fontsize=7, handlelength=1.2)


def draw_initialization(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.set_title("Initialization", fontsize=TITLE_FS, pad=8, fontweight="medium")

    rounded_box(
        ax,
        (0.22, 0.50),
        r"$|\downarrow\downarrow\uparrow\rangle$",
        w=0.36,
        h=0.22,
    )

    rounded_box(
        ax,
        (0.78, 0.50),
        r"$|0\rangle$",
        w=0.28,
        h=0.22,
    )

    ax.annotate(
        "",
        xy=(0.58, 0.50),
        xytext=(0.42, 0.50),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.8,
            color=PALETTE["plum"],
            mutation_scale=14,
        ),
    )

    ax.text(
        0.50,
        0.74,
        r"ramp $J_{12}$",
        ha="center",
        fontsize=TEXT_FS - 0.5,
        color=PALETTE["plum"],
    )

    ax.text(
        0.22,
        0.18,
        r"$J_{12,23}=0$",
        ha="center",
        fontsize=TEXT_FS - 1,
        color=PALETTE["slate"],
    )

    ax.text(
        0.78,
        0.18,
        r"$J_{12}=J_{12}^{\rm idle}$",
        ha="center",
        fontsize=TEXT_FS - 1,
        color=PALETTE["slate"],
    )


def draw_control(ax):
    ax.set_xlim(-0.08, 1.02)
    ax.set_ylim(-0.08, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")

    # Cleaner than repeating sigma labels in the title.
    ax.set_title("Single-qubit gates", fontsize=TITLE_FS, pad=14, fontweight="medium")

    origin = 0.14
    idle = np.array([origin, 0.36])

    ax.annotate(
        "",
        xy=(0.92, origin),
        xytext=(origin, origin),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.2,
            color=PALETTE["slate"],
            mutation_scale=12,
        ),
    )

    ax.annotate(
        "",
        xy=(origin, 0.88),
        xytext=(origin, origin),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.2,
            color=PALETTE["slate"],
            mutation_scale=12,
        ),
    )

    # sigma_z direction: shifted right, no collision with J12 axis.
    ax.annotate(
        "",
        xy=(idle[0] + 0.075, idle[1] + 0.34),
        xytext=tuple(idle),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=2.2,
            color=PALETTE["sage"],
            mutation_scale=14,
        ),
    )

    # sigma_x direction.
    ax.annotate(
        "",
        xy=(idle[0] + 0.40, idle[1] + 0.22),
        xytext=tuple(idle),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=2.2,
            color=PALETTE["sky"],
            mutation_scale=14,
        ),
    )

    ax.scatter(
        *idle,
        s=45,
        color=PALETTE["coral"],
        zorder=15,
        edgecolors="white",
        linewidths=0.5,
    )

    ax.text(
        idle[0] + 0.07,
        idle[1] - 0.075,
        "idle",
        fontsize=TEXT_FS - 0.5,
        color=PALETTE["coral"],
        ha="left",
        va="center",
    )

    ax.text(
        idle[0] + 0.14,
        idle[1] + 0.35,
        r"$\sigma_z$",
        fontsize=TEXT_FS,
        color=PALETTE["sage"],
        ha="left",
        va="center",
    )

    ax.text(
        idle[0] + 0.44,
        idle[1] + 0.24,
        r"$\sigma_x$",
        fontsize=TEXT_FS,
        color=PALETTE["sky"],
        ha="left",
        va="center",
    )

    ax.text(
        0.58,
        origin - 0.085,
        r"$J_{23}$",
        fontsize=LABEL_FS,
        ha="center",
        va="top",
        color=PALETTE["slate"],
    )

    ax.text(
        origin - 0.095,
        0.58,
        r"$J_{12}$",
        fontsize=LABEL_FS,
        ha="right",
        va="center",
        rotation=90,
        color=PALETTE["slate"],
    )


def draw_readout(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.set_title("Readout", fontsize=TITLE_FS, pad=8, fontweight="medium")

    rounded_box(
        ax,
        (0.22, 0.50),
        r"$|T_-^{12}\downarrow\rangle$",
        w=0.36,
        h=0.22,
        fs=TEXT_FS - 0.5,
    )

    rounded_box(
        ax,
        (0.78, 0.50),
        r"$|S^{12}\downarrow\rangle$",
        w=0.32,
        h=0.22,
        fs=TEXT_FS - 0.5,
    )

    ax.annotate(
        "",
        xy=(0.58, 0.50),
        xytext=(0.42, 0.50),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.8,
            color=PALETTE["plum"],
            mutation_scale=14,
        ),
    )

    ax.text(
        0.50,
        0.74,
        "PSB",
        ha="center",
        fontsize=TEXT_FS,
        color=PALETTE["plum"],
        fontweight="medium",
    )


def main():
    prx_style()

    fig = plt.figure(figsize=(7.2, 4.65), constrained_layout=False)

    gs = GridSpec(
        2,
        7,
        figure=fig,
        height_ratios=[1.10, 0.62],
        width_ratios=[1, 1, 1, 1, 0.12, 1.05, 1.05],
        wspace=0.72,
        hspace=0.80,
    )

    ax_qubit = fig.add_subplot(gs[0, 0:4])
    ax_spec = fig.add_subplot(gs[0, 5:7])

    ax_init = fig.add_subplot(gs[1, 0:2])
    ax_ctrl = fig.add_subplot(gs[1, 2:5])
    ax_read = fig.add_subplot(gs[1, 5:7])

    draw_qubit_schematic(ax_qubit)
    draw_spectrum(ax_spec)
    draw_initialization(ax_init)
    draw_control(ax_ctrl)
    draw_readout(ax_read)

    panel_label(ax_qubit, "(a)", x=-0.02, y=1.02)
    panel_label(ax_spec, "(b)", x=-0.14, y=1.08)

    panel_label(ax_init, "(c)", x=-0.13, y=1.08)
    panel_label(ax_ctrl, "(d)", x=-0.16, y=1.18)
    panel_label(ax_read, "(e)", x=-0.13, y=1.08)

    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.075,
        top=0.935,
        wspace=0.72,
        hspace=0.80,
    )

    out_dir = Path(__file__).resolve().parent
    fig.savefig(out_dir / "qubit_passport.pdf")
    fig.savefig(out_dir / "qubit_passport.png", dpi=400)

    plt.show()


if __name__ == "__main__":
    main()