from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch

# --- Минималистичная академическая палитра (PRX Compliant) ---
COLORS = {
    "ink": "#1A1A1A",
    "muted": "#626F7D",
    "sphere_fill": "#F8FAFC",
    "sphere_edge": "#5A6673",
    "sphere_back": "#BDC3C7",
    "traj_front": "#1F77B4",
    "traj_back": "#A9CCE3",
    "axis": "#D35400",
    "axis_soft": "#EDBB99",
    "start_point": "#C0392B",
}

def apply_prx_style() -> None:
    mpl.rcParams.update({
        "figure.figsize": (5.4, 3.2),  # Делаем фигуру более компактной и аккуратной
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.75,
    })

def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)

def rodrigues_rotate(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    vec = np.asarray(vec, dtype=float)
    return (
        vec * np.cos(angle)
        + np.cross(axis, vec) * np.sin(angle)
        + axis * np.dot(axis, vec) * (1.0 - np.cos(angle))
    )

def project_points(points: np.ndarray, twist_deg: float = -30.0, elev_deg: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    twist = np.deg2rad(twist_deg)
    elev = np.deg2rad(elev_deg)

    rz = np.array([
        [np.cos(twist), -np.sin(twist), 0.0],
        [np.sin(twist),  np.cos(twist), 0.0],
        [0.0,            0.0,           1.0]
    ])
    rx = np.array([
        [1.0, 0.0,           0.0],
        [0.0, np.cos(elev), -np.sin(elev)],
        [0.0, np.sin(elev),  np.cos(elev)]
    ])

    rotated = (rx @ (rz @ pts.T)).T
    return rotated[:, [0, 2]], rotated[:, 1]

def draw_split_curve(ax: plt.Axes, points: np.ndarray, front_args: dict, back_args: dict) -> None:
    xy, depth = project_points(points)
    
    xy_front = xy.copy()
    xy_front[depth < 0] = np.nan
    xy_back = xy.copy()
    xy_back[depth >= 0] = np.nan
    
    ax.plot(xy_back[:, 0], xy_back[:, 1], **back_args)
    ax.plot(xy_front[:, 0], xy_front[:, 1], **front_args)

def draw_bloch_panel(ax: plt.Axes) -> None:
    delta = 0.65
    j_int = 1.30  
    omega_axis = normalize(np.array([-j_int, 0.0, delta]))

    north = np.array([0.0, 0.0, 1.0])
    south = np.array([0.0, 0.0, -1.0])

    # 1. Задняя сетка сферы
    t = np.linspace(0.0, 2.0 * np.pi, 400)
    equator = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    meridian = np.column_stack([np.zeros_like(t), np.cos(t), np.sin(t)])
    
    grid_back = {"color": COLORS["sphere_back"], "lw": 0.6, "ls": (0, (3, 4)), "zorder": 1}
    grid_front = {"color": COLORS["sphere_edge"], "lw": 0.7, "alpha": 0.4, "zorder": 4}
    
    draw_split_curve(ax, equator, grid_front, grid_back)
    draw_split_curve(ax, meridian, grid_front, grid_back)

    # Вертикальная ось Z
    xy_z, _ = project_points(np.vstack([south, north]))
    ax.plot(xy_z[:, 0], xy_z[:, 1], color=COLORS["sphere_edge"], lw=0.7, ls="--", alpha=0.3, zorder=2)

    # 2. Матовое тело сферы
    ax.add_patch(Circle((0.0, 0.0), 1.0, facecolor=COLORS["sphere_fill"], edgecolor=COLORS["sphere_edge"], linewidth=1.0, zorder=3))
    ax.add_patch(Ellipse((-0.18, 0.22), 0.6, 0.35, angle=20, facecolor="white", alpha=0.4, zorder=3))

    # 3. Траектория
    phi = np.linspace(0.0, 2.0 * np.pi, 1000)
    trajectory = np.array([rodrigues_rotate(north, omega_axis, angle) for angle in phi])
    traj_xy, _ = project_points(trajectory)
    
    traj_front_style = {
        "color": COLORS["traj_front"], "lw": 2.2, "solid_capstyle": "round", "zorder": 6,
        "path_effects": [pe.Stroke(linewidth=3.5, foreground="white", alpha=0.9), pe.Normal()]
    }
    traj_back_style = {
        "color": COLORS["traj_back"], "lw": 1.4, "ls": (0, (2, 2)), "zorder": 5
    }
    draw_split_curve(ax, trajectory, traj_front_style, traj_back_style)

    # Изящные стрелки направления без грубых форм
    ax.add_patch(FancyArrowPatch(traj_xy[160], traj_xy[175], arrowstyle="-|>", mutation_scale=10.0, linewidth=1.5, color=COLORS["traj_front"], zorder=7))
    ax.add_patch(FancyArrowPatch(traj_xy[720], traj_xy[735], arrowstyle="-|>", mutation_scale=9.0, linewidth=1.2, color=COLORS["traj_back"], zorder=5))

    # 4. Вектор оси вращения \Omega
    omega_xy, _ = project_points(np.vstack([[0.0, 0.0, 0.0], 0.92 * omega_axis]))
    ax.plot(omega_xy[:, 0], omega_xy[:, 1], color=COLORS["axis_soft"], lw=1.2, ls=":", zorder=4)
    ax.add_patch(FancyArrowPatch(omega_xy[0], omega_xy[1], arrowstyle="-|>", mutation_scale=12.0, linewidth=1.8, color=COLORS["axis"], zorder=8))

    # 5. Маркеры критических точек
    north_xy, _ = project_points(north)
    leak_idx = int(np.argmin(trajectory[:, 2]))  
    leak_xy = traj_xy[leak_idx]

    ax.scatter([north_xy[0, 0]], [north_xy[0, 1]], s=36, color=COLORS["start_point"], edgecolors="white", linewidths=0.8, zorder=9)
    ax.scatter([leak_xy[0]], [leak_xy[1]], s=18, color=COLORS["traj_front"], edgecolors="white", linewidths=0.7, zorder=9)

    # 6. Текстовые аннотации (Чистка координат и наложений)
    ax.text(-1.15, 1.10, "(b)", fontsize=14.0, fontweight="bold", color=COLORS["ink"], ha="left", va="top")
    ax.text(north_xy[0, 0], north_xy[0, 1] + 0.14, r"$|01\rangle$", fontsize=13.5, color=COLORS["ink"], ha="center", va="bottom")
    
    south_xy, _ = project_points(south)
    ax.text(south_xy[0, 0], south_xy[0, 1] - 0.14, r"$|01\rangle_{\mathrm{L}}$", fontsize=13.5, color=COLORS["ink"], ha="center", va="top")
    
    ax.text(omega_xy[1, 0] - 0.05, omega_xy[1, 1] + 0.04, r"$\vec{\Omega}$", fontsize=13.0, color=COLORS["axis"], fontweight="bold", ha="right", va="bottom")
    
    # Убрали белую рамку, сделали чистый аккуратный текст рядом с точкой
    ax.text(leak_xy[0] + 0.08, leak_xy[1] - 0.04, r"$P_{\mathrm{leak}}^{\max}$", fontsize=10.5, color=COLORS["ink"], ha="left", va="center", zorder=8)
    
    # Сместили t=0 подальше влево от точки, чтобы избежать каши
    ax.text(north_xy[0, 0] - 0.14, north_xy[0, 1] - 0.08, r"$t=0, t_1$", fontsize=9.5, color=COLORS["muted"], ha="right", va="top")

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

def draw_text_block(ax: plt.Axes) -> None:
    ax.axis("off")
    # Оптимизированные отступы и размеры шрифтов для предотвращения обрезки текста
    rows = [
        (0.76, r"$\vec{\Omega} = (J_{\mathrm{int}},\, 0,\, \Delta)$", COLORS["axis"], 12.0),
        (0.53, r"$\Omega = \sqrt{\Delta^2 + J_{\mathrm{int}}^2}$", COLORS["ink"], 11.5),
        (0.30, r"$t_n = \frac{2\pi n}{\Omega}$", COLORS["ink"], 11.5),
        (0.06, r"closed loop $\Rightarrow$ zero residual leakage", COLORS["muted"], 10.0),
    ]
    for y, text, color, size in rows:
        ax.text(0.02, y, text, fontsize=size, color=color, ha="left", va="center")

def build_figure() -> plt.Figure:
    fig = plt.figure()
    gs = GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.0, 0.9],  # Более сбалансированное распределение места
        left=0.01, right=0.94,    # Оставляем запас справа для длинных предложений
        top=0.96, bottom=0.04,
        wspace=0.02
    )

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    draw_bloch_panel(ax_left)
    draw_text_block(ax_right)
    return fig

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "soso_bloch_sphere.pdf")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    apply_prx_style()
    fig = build_figure()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")

    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()