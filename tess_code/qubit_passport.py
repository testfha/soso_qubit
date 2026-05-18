from __future__ import annotations

from itertools import permutations
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from matplotlib.patches import Circle

try:
    from plot_triple_dot_spins import add_dot, add_exchange, add_spin, confinement
except ImportError:
    def confinement(x):
        return 0.08 * (x**2 - 9) * np.cos(0.95 * x) - 1.2

    def add_dot(ax, x, y, r=0.5):
        circle = plt.Circle((x, y), r, fc="#D4E4F7", ec="#2C3E50", lw=1.8, zorder=10)
        ax.add_patch(circle)

    def add_spin(ax, x, y, tilt, length, color, label, label_dx=0, label_dy=0, **kw):
        dx, dy = length * np.sin(tilt), length * np.cos(tilt)
        arrow = FancyArrowPatch(
            (x, y - 0.15),
            (x + dx, y + dy),
            arrowstyle="-|>",
            mutation_scale=kw.get("mutation_scale", 18),
            lw=kw.get("linewidth", 2.5),
            color=color,
            zorder=15,
        )
        ax.add_patch(arrow)
        ax.text(
            x + label_dx,
            y + label_dy,
            label,
            fontsize=14,
            color=color,
            ha="center",
            va="bottom",
        )

    def add_exchange(ax, x1, x2, y, label, **kw):
        ax.annotate(
            "",
            xy=(x2, y),
            xytext=(x1, y),
            arrowprops=dict(
                arrowstyle="<->",
                lw=kw.get("linewidth", 2),
                color="#555",
                connectionstyle="arc3,rad=0.25",
                mutation_scale=kw.get("mutation_scale", 16),
            ),
        )
        ax.text(
            (x1 + x2) / 2,
            y + 0.35,
            label,
            ha="center",
            va="bottom",
            fontsize=kw.get("label_fontsize", 14),
            color="#333",
        )


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

FONT_FAMILY = "serif"
MATHTEXT_FONT = "cm"

PANEL_LABEL_FS = 5.8
TITLE_FS = 7.8
LABEL_FS = 7.0
TEXT_FS = 6.4
TICK_FS = 6.0
QUBIT_LABEL_FS = LABEL_FS

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
ID2 = np.eye(2, dtype=complex)


def prx_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (5.2, 2.55),
            "figure.dpi": 300,
            "figure.facecolor": "white",
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": FONT_FAMILY,
            "font.size": 6,
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
            "legend.fontsize": 5.6,
            "legend.frameon": False,
            "legend.handlelength": 1.5,
        }
    )


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


def j_idle(g1, g2, g3):
    return ((g2 - g1) ** 2 - (2 * g3 - g1 - g2) ** 2) / (2 * (2 * g3 - g1 - g2))


def mz_minus_half_indices():
    sz_tot = 0.5 * (
        kron3(SIGMA_Z, ID2, ID2) + kron3(ID2, SIGMA_Z, ID2) + kron3(ID2, ID2, SIGMA_Z)
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
    tracked_evals = []
    prev_vecs = None

    for j12 in j_vals:
        ham = hamiltonian(j12, 0.0, g1, g2, g3)
        sub = ham[np.ix_(idx, idx)]
        evals, evecs = np.linalg.eigh(sub)
        evals = np.real(evals)

        if prev_vecs is None:
            order = np.argsort(evals)
            evals = evals[order]
            evecs = evecs[:, order]
        else:
            overlaps = np.abs(prev_vecs.conj().T @ evecs) ** 2
            best_perm = max(permutations(range(len(evals))), key=lambda p: sum(overlaps[i, p[i]] for i in range(len(evals))))
            evals = evals[list(best_perm)]
            evecs = evecs[:, list(best_perm)]

        tracked_evals.append(evals)
        prev_vecs = evecs

    spectrum = np.array(tracked_evals)
    return spectrum - np.min(spectrum)


def panel_label(ax, label, x=-0.08, y=1.03):
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
        alpha=0.85,
        zorder=5,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=10)


def lower_title_badge(ax, text: str, *, y: float = 1.03) -> None:
    ax.text(
        0.5,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=QUBIT_LABEL_FS,
        fontweight="bold",
        color=PALETTE["charcoal"],
        zorder=20,
    )


def add_colored_dot(ax, x, y, color, r=0.42):
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


def add_centered_spin(
    ax,
    x: float,
    y: float,
    dx: float,
    dy: float,
    color: str,
    label: str,
    label_dx: float,
    label_dy: float,
    linewidth: float,
    mutation_scale: float,
) -> None:
    # Compensate for the single arrow head so the visible shaft stays centered on the dot.
    shift = 0.10 * np.sign(dy)
    add_spin(
        ax,
        x,
        y + shift,
        dx,
        dy,
        color,
        label,
        label_dx=label_dx,
        label_dy=label_dy,
        linewidth=linewidth,
        mutation_scale=mutation_scale,
    )


def draw_qubit_schematic(ax):
    yshift = -0.6
    x = np.linspace(-3.2, 3.2, 400)
    y = confinement(x) + yshift

    ax.plot(x, y, color=PALETTE["plum"], lw=2.0, zorder=1)
    ax.fill_between(x, y, y.min() - 0.5, color=PALETTE["plum"], alpha=0.08)

    dot_x = [-2.30, 0.0, 2.30]
    dot_y = [0.52 + yshift] * 3
    spin_colors = [PALETTE["coral"], PALETTE["sage"], PALETTE["sky"]]
    spin_labels = [r"$g_1$", r"$g_2$", r"$g_3$"]
    spin_dirs = [(-0.12, -0.90), (-0.12, -0.93), (0.12, 0.80)]

    for xd, yd, color in zip(dot_x, dot_y, spin_colors):
        add_colored_dot(ax, xd, yd, color, r=0.42)

    for i, (xd, yd) in enumerate(zip(dot_x, dot_y)):
        tilt, length = spin_dirs[i]
        add_centered_spin(
            ax,
            xd,
            yd,
            tilt,
            length,
            spin_colors[i],
            spin_labels[i],
            label_dx=-0.28,
            label_dy=0.75,
            linewidth=2.8,
            mutation_scale=12,
        )

    add_exchange(
        ax,
        -1.84,
        -0.36,
        -0.15 + yshift,
        r"$J_{12}$",
        linewidth=1.35,
        label_fontsize=QUBIT_LABEL_FS,
        mutation_scale=10,
    )
    add_exchange(
        ax,
        0.36,
        1.84,
        -0.15 + yshift,
        r"$J_{23}$",
        linewidth=1.35,
        label_fontsize=QUBIT_LABEL_FS,
        mutation_scale=10,
    )

    for txt in ax.texts:
        txt.set_fontsize(QUBIT_LABEL_FS)

    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-2.2, 0.8)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_spectrum(ax):
    g1, g2, g3 = 53.0, 74.0, 47.0
    j_id = j_idle(g1, g2, g3)
    j_vals = np.linspace(0, 1.6 * j_id, 400)
    spec = compute_spectrum(j_vals, g1, g2, g3)

    colors = [PALETTE["sage"], PALETTE["plum"], PALETTE["amber"]]
    labels = [r"$|0\rangle$", r"$|1\rangle$", r"leak"]

    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.plot(j_vals, spec[:, i], lw=1.5, color=color, label=label)

    y_id = 0.5 * (np.interp(j_id, j_vals, spec[:, 0]) + np.interp(j_id, j_vals, spec[:, 1]))
    ax.scatter(
        [j_id],
        [y_id],
        s=50,
        facecolors="none",
        edgecolors=PALETTE["charcoal"],
        linewidths=1.5,
        zorder=10,
    )
    ax.annotate(
        "idle",
        (j_id, y_id),
        xytext=(j_id + 0.8, y_id + 2.5),
        fontsize=QUBIT_LABEL_FS,
        color=PALETTE["charcoal"],
        arrowprops=dict(arrowstyle="-", lw=0.5, color=PALETTE["charcoal"], alpha=0.6),
    )

    ax.set_xlabel(r"$J_{12}$ (MHz)")
    ax.set_ylabel(r"$E$ (MHz)")
    ax.set_title(r"$m_z = -\frac{1}{2}$, $J_{23} = 0$", fontsize=QUBIT_LABEL_FS, pad=4)
    ax.set_xlim(0, j_vals.max())
    ax.set_ylim(-0.5, spec.max() + 3)
    ax.tick_params(labelsize=QUBIT_LABEL_FS)
    ax.xaxis.label.set_size(QUBIT_LABEL_FS)
    ax.yaxis.label.set_size(QUBIT_LABEL_FS)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.84), fontsize=QUBIT_LABEL_FS, handlelength=1.2)


def draw_initialization(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    lower_title_fs = QUBIT_LABEL_FS
    lower_text_fs = QUBIT_LABEL_FS
    lower_small_fs = QUBIT_LABEL_FS
    lower_title_badge(ax, "Initialization")

    rounded_box(ax, (0.22, 0.42), r"$|\downarrow\downarrow\uparrow\rangle$", w=0.36, h=0.22, fs=lower_text_fs)
    rounded_box(ax, (0.78, 0.42), r"$|0\rangle$", w=0.28, h=0.22, fs=lower_text_fs)

    ax.annotate(
        "",
        xy=(0.62, 0.42),
        xytext=(0.42, 0.42),
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color=PALETTE["plum"], mutation_scale=10),
    )
    ax.text(0.5, 0.64, r"ramp $J_{12}$", ha="center", fontsize=lower_text_fs - 1.0, color=PALETTE["plum"])

    ax.text(0.22, 0.08, r"$J_{12,23}=0$", ha="center", fontsize=lower_small_fs, color=PALETTE["slate"])
    ax.text(
        0.78,
        0.08,
        r"$J_{12}=J_{12}^{\rm idle}$",
        ha="center",
        fontsize=lower_small_fs,
        color=PALETTE["slate"],
    )


def draw_control(ax):
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.05, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    lower_title_fs = QUBIT_LABEL_FS
    lower_text_fs = QUBIT_LABEL_FS
    lower_label_fs = QUBIT_LABEL_FS
    lower_title_badge(ax, r"$\sigma_x,\sigma_z$ gates")

    origin_x = 0.08
    origin_y = 0.02
    idle = np.array([origin_x, 0.24])

    ax.annotate(
        "",
        xy=(0.98, origin_y),
        xytext=(origin_x, origin_y),
        arrowprops=dict(arrowstyle="-|>", lw=0.85, color=PALETTE["slate"], mutation_scale=5.5),
    )
    ax.annotate(
        "",
        xy=(origin_x, 0.88),
        xytext=(origin_x, origin_y),
        arrowprops=dict(arrowstyle="-|>", lw=0.85, color=PALETTE["slate"], mutation_scale=5.5),
    )

    ax.annotate(
        "",
        xy=(idle[0], idle[1] + 0.36),
        xytext=tuple(idle),
        arrowprops=dict(arrowstyle="-|>", lw=1.35, color=PALETTE["forest"], mutation_scale=6.5),
    )
    ax.annotate(
        "",
        xy=(idle[0] + 0.44, idle[1] + 0.26),
        xytext=tuple(idle),
        arrowprops=dict(arrowstyle="-|>", lw=1.35, color=PALETTE["indigo"], mutation_scale=6.5),
    )

    ax.scatter(*idle, s=28, color=PALETTE["charcoal"], zorder=15, edgecolors="white", linewidths=0.35)

    ax.text(idle[0] + 0.16, idle[1] - 0.06, "idle", fontsize=lower_text_fs - 0.2, color=PALETTE["charcoal"])
    ax.text(idle[0] + 0.04, idle[1] + 0.31, r"$\sigma_z$", fontsize=lower_text_fs, color=PALETTE["forest"])
    ax.text(idle[0] + 0.48, idle[1] + 0.21, r"$\sigma_x$", fontsize=lower_text_fs, color=PALETTE["indigo"])
    ax.text(0.68, origin_y - 0.15, r"$J_{23}$", fontsize=lower_label_fs, ha="center", color=PALETTE["slate"])
    ax.text(origin_x - 0.18, 0.70, r"$J_{12}$", fontsize=lower_label_fs, ha="center", rotation=90, color=PALETTE["slate"])


def draw_readout(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    lower_title_fs = QUBIT_LABEL_FS
    lower_text_fs = QUBIT_LABEL_FS
    lower_title_badge(ax, "Readout")

    rounded_box(ax, (0.22, 0.42), r"$|T_-^{12}\downarrow\rangle$", w=0.36, h=0.22, fs=lower_text_fs)
    rounded_box(ax, (0.78, 0.42), r"$|S^{12}\downarrow\rangle$", w=0.32, h=0.22, fs=lower_text_fs)

    ax.annotate(
        "",
        xy=(0.62, 0.42),
        xytext=(0.42, 0.42),
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color=PALETTE["plum"], mutation_scale=10),
    )
    ax.text(0.5, 0.64, "PSB", ha="center", fontsize=lower_text_fs - 1.0, color=PALETTE["plum"], fontweight="medium")


def main():
    prx_style()

    fig = plt.figure(constrained_layout=False)
    gs = GridSpec(2, 8, figure=fig, height_ratios=[0.52, 0.34], wspace=1.00, hspace=0.78)

    ax_qubit = fig.add_subplot(gs[0, 0:5])
    ax_spec = fig.add_subplot(gs[0, 5:8])
    ax_init = fig.add_subplot(gs[1, 0:3])
    ax_ctrl = fig.add_subplot(gs[1, 3:5])
    ax_read = fig.add_subplot(gs[1, 5:8])

    draw_qubit_schematic(ax_qubit)
    draw_spectrum(ax_spec)
    draw_initialization(ax_init)
    draw_control(ax_ctrl)
    draw_readout(ax_read)

    panel_label(ax_qubit, "(a)", x=-0.02, y=1.01)
    panel_label(ax_spec, "(b)", x=-0.12, y=1.05)
    panel_label(ax_init, "(c)", x=-0.07, y=1.02)
    panel_label(ax_ctrl, "(d)", x=-0.42, y=1.02)
    panel_label(ax_read, "(e)", x=-0.07, y=1.02)

    spec_pos = ax_spec.get_position()
    ax_spec.set_position([spec_pos.x0, spec_pos.y0, spec_pos.width + 0.02, spec_pos.height])

    out_dir = Path(__file__).resolve().parent
    fig.savefig(out_dir / "qubit_passport.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "qubit_passport.png", bbox_inches="tight", dpi=400)
    plt.show()


if __name__ == "__main__":
    main()
