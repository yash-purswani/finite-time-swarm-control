#!/usr/bin/env python3
r"""Generate and export high-resolution single-panel 2D arena animation of the swarm maneuver.

Simulates the rotating-compressing star formation for arbitrary coupling gain m
and maneuver compression rate kappa, then renders and exports an arena-only animation
featuring:
  - Target stations q_i(t) and rotating-compressing star template outline
  - Physical agents x_i(t) with trajectory history trails
  - Proximity communication graph edges (||x_i - x_j|| < R_c)
  - Tracking displacement vectors (x_i to q_i)
  - Centroid reference path r(t)
  - On-screen Heads-Up Display (HUD) showing time, separation, diameter, error, and collision warnings.
  - Configurable agent spawning: random ring around stations (r in [0.3, 0.7] m)

Usage:
    python3 animate_swarm.py --m 2.5 --kappa 4.0 --t-end 5.0 --output results/safe_star.mp4
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finite_time.core.stations import build_star_template, star_stations
from finite_time.core.spawn import generate_random_ring_perturbation
from finite_time.core.simulation import simulate_star_maneuver


# =============================================================================
# DEFAULT CONFIGURATION (Can be edited directly here or via CLI)
# =============================================================================
DEFAULT_M = 2.5
DEFAULT_KAPPA = 4.0
DEFAULT_T_END = 5.0
DEFAULT_FPS = 30
DEFAULT_DPI = 120
DEFAULT_D_SAFE = 1.0
DEFAULT_R_C = 6.0
DEFAULT_R_MIN = 0.3
DEFAULT_R_MAX = 0.7
DEFAULT_SEED = 123
DEFAULT_OUTPUT = REPO_ROOT / "results" / "swarm_arena_animation.mp4"


def generate_animation(
    m: float = DEFAULT_M,
    kappa: float = DEFAULT_KAPPA,
    t_end: float = DEFAULT_T_END,
    d_safe: float = DEFAULT_D_SAFE,
    r_c: float = DEFAULT_R_C,
    r_min: float = DEFAULT_R_MIN,
    r_max: float = DEFAULT_R_MAX,
    seed: int = DEFAULT_SEED,
    output_path: Optional[Path] = None,
    fps: int = DEFAULT_FPS,
    dpi: int = DEFAULT_DPI,
    trail_length: int = 45,
) -> Path:
    """Run simulation and render single-panel 2D arena animation to MP4 or GIF."""
    if output_path is None:
        output_path = DEFAULT_OUTPUT
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    p0 = build_star_template(n=10)
    q0, _ = star_stations(0.0, kappa=kappa, p0=p0, a0=1.10, a_inf=0.75, omega=np.pi / 4.0)

    # 1. Determine Initial Perturbation delta0
    print(f"Spawning agents randomly in annular ring r in [{r_min:.2f}, {r_max:.2f}] m around stations (seed={seed})...")
    delta0, min_dx0, max_ell0 = generate_random_ring_perturbation(
        q0, r_min=r_min, r_max=r_max, d_safe=d_safe, r_c=r_c, seed=seed
    )
    print(f"Verified initial swarm state: min separation d_x(0) = {min_dx0:.4f} m > {d_safe} m, "
            f"diameter ell_x(0) = {max_ell0:.4f} m < {r_c} m.")

    # 2. Simulate Trajectories
    print(f"Simulating maneuver: m = {m:.2f}, kappa = {kappa:.2f}, t in [0, {t_end:.2f}] s...")
    res = simulate_star_maneuver(
        m=m,
        kappa=kappa,
        t_span=(0.0, t_end),
        delta0=delta0,
        d_safe=d_safe,
        r_c=r_c,
        rtol=1e-6,
        atol=1e-8,
        max_step=0.02,
    )

    t_all = res.t
    n_frames = int(t_end * fps)
    frame_indices = np.linspace(0, len(t_all) - 1, n_frames, dtype=int)

    n_agents = res.x.shape[1]
    star_closed = np.r_[np.arange(n_agents), 0]
    agent_colors = plt.get_cmap("tab10")(np.arange(n_agents))

    # Determine arena spatial bounds across all time with pleasant padding
    all_x = res.x[frame_indices].reshape(-1, 2)
    all_q = res.q[frame_indices].reshape(-1, 2)
    min_xy = np.minimum(all_x.min(axis=0), all_q.min(axis=0)) - 1.6
    max_xy = np.maximum(all_x.max(axis=0), all_q.max(axis=0)) + 1.6

    # 3. Set Up Single-Panel Figure
    fig, ax_arena = plt.subplots(figsize=(11, 8.5), dpi=dpi)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.92, bottom=0.08)

    ax_arena.set_xlim(min_xy[0], max_xy[0])
    ax_arena.set_ylim(min_xy[1], max_xy[1])
    ax_arena.set_aspect("equal", adjustable="box")
    ax_arena.set_xlabel("X Position [m]", fontsize=11, fontweight="bold")
    ax_arena.set_ylabel("Y Position [m]", fontsize=11, fontweight="bold")
    ax_arena.grid(True, linestyle=":", alpha=0.55)

    # Reference center track
    ref_center = np.mean(res.q, axis=1)
    ax_arena.plot(ref_center[:, 0], ref_center[:, 1], "k--", lw=1.3, alpha=0.5, label="Centroid Path $r(t)$")

    # Initial station & agent graphics
    q_init = res.q[frame_indices[0]]
    x_init = res.x[frame_indices[0]]

    star_stations_line, = ax_arena.plot(
        q_init[star_closed, 0], q_init[star_closed, 1], "--",
        color="#4b5563", lw=1.6, alpha=0.8, label="Station Star Contour"
    )
    stations_scatter = ax_arena.scatter(
        q_init[:, 0], q_init[:, 1], s=55, facecolors="none", edgecolors="#4b5563",
        linewidths=1.3, linestyle="--", zorder=3, label="Target Stations $q_i$"
    )
    agent_scatter = ax_arena.scatter(
        x_init[:, 0], x_init[:, 1], s=95, c=agent_colors, edgecolors="black",
        linewidths=0.9, zorder=5, label="Physical Agents $x_i$"
    )

    # Trails and error lines
    trails = [ax_arena.plot([], [], color=agent_colors[i], lw=1.3, alpha=0.65)[0] for i in range(n_agents)]
    err_lines = [ax_arena.plot([], [], color="#dc2626", ls=":", lw=1.1, alpha=0.85)[0] for i in range(n_agents)]

    # Dynamic proximity network edges (||x_i - x_j|| < R_c)
    pair_indices = [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]
    edge_lines = [ax_arena.plot([], [], color="#60a5fa", lw=1.0, alpha=0.45, zorder=2)[0]
                  for _ in range(len(pair_indices))]

    # On-screen Telemetry HUD (Heads-Up Display)
    hud_text = ax_arena.text(
        0.02, 0.98, "", transform=ax_arena.transAxes, va="top", fontsize=9.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", alpha=0.92, edgecolor="#cbd5e1", lw=1.2)
    )

    # Collision Warning Badge
    collision_banner = ax_arena.text(
        0.5, 0.05, "COLLISION DETECTED! (d_x < 1.0 m)", transform=ax_arena.transAxes,
        ha="center", va="center", fontsize=13, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#dc2626", alpha=0.95, edgecolor="darkred", lw=1.5),
        visible=False, zorder=10
    )

    spawn_label = f"Random Ring [{r_min:.2f}, {r_max:.2f}] m"
    ax_arena.set_title(
        f"Rotating-Compressing Star Formation Tracking ($m={m:.1f}$, $\\kappa={kappa:.1f}$)\n"
        f"Spawn: {spawn_label} | Outcome: {res.metrics['classification']} ({'Safe' if res.metrics['observed_safe'] else 'Collision'})",
        fontsize=11.5, fontweight="bold"
    )
    ax_arena.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    def update_frame(frame_num: int):
        idx = frame_indices[frame_num]
        t_val = t_all[idx]
        xi = res.x[idx]
        qi = res.q[idx]

        # 1. Update stations & agents
        star_stations_line.set_data(qi[star_closed, 0], qi[star_closed, 1])
        stations_scatter.set_offsets(qi)
        agent_scatter.set_offsets(xi)

        # 2. Update error displacement lines (x_i to q_i)
        for i in range(n_agents):
            err_lines[i].set_data([xi[i, 0], qi[i, 0]], [xi[i, 1], qi[i, 1]])

        # 3. Update trajectory history trails
        step_stride = max(1, len(t_all) // len(frame_indices))
        start_trail = max(0, idx - trail_length * step_stride)
        for i in range(n_agents):
            trails[i].set_data(res.x[start_trail:idx + 1:step_stride, i, 0],
                               res.x[start_trail:idx + 1:step_stride, i, 1])

        # 4. Update dynamic proximity communication graph
        cur_dists = np.linalg.norm(xi[:, None, :] - xi[None, :, :], axis=-1)
        for k, (i, j) in enumerate(pair_indices):
            if cur_dists[i, j] < r_c:
                edge_lines[k].set_data([xi[i, 0], xi[j, 0]], [xi[i, 1], xi[j, 1]])
            else:
                edge_lines[k].set_data([], [])

        # 5. Telemetry & Collision status
        cur_dx = res.d_x[idx]
        cur_ell = res.ell_x[idx]
        cur_err = res.delta_norm[idx]
        is_collision = bool(cur_dx < d_safe)

        collision_banner.set_visible(is_collision)
        status_label = "COLLISION VIOLATION" if is_collision else ("SETTLED" if cur_err < 1e-4 else "CONVERGING")

        hud_text.set_text(
            f"TELEMETRY HUD\n"
            f"---------------------------\n"
            f"Time        : {t_val:5.2f} / {t_end:.1f} s\n"
            f"Min Sep dx  : {cur_dx:5.3f} m  (safe > {d_safe:.1f})\n"
            f"Max Diam lx : {cur_ell:5.3f} m  (conn < {r_c:.1f})\n"
            f"Error ||d|| : {cur_err:5.4f}\n"
            f"Status      : {status_label}"
        )

        return (
            star_stations_line, stations_scatter, agent_scatter,
            *err_lines, *trails, *edge_lines,
            hud_text, collision_banner
        )

    print(f"Rendering {n_frames} frames ({fps} FPS)...")
    anim = FuncAnimation(fig, update_frame, frames=n_frames, blit=False)

    ext = output_path.suffix.lower()
    if ext == ".gif":
        print(f"Exporting to GIF via PillowWriter: {output_path}...")
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
    else:
        print(f"Exporting to MP4 via FFMpegWriter: {output_path}...")
        writer = FFMpegWriter(fps=fps, bitrate=2400, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        try:
            anim.save(output_path, writer=writer)
        except Exception as e:
            print(f"FFmpeg MP4 export failed ({e}). Falling back to GIF...")
            fallback_path = output_path.with_suffix(".gif")
            anim.save(fallback_path, writer=PillowWriter(fps=min(fps, 15)))
            output_path = fallback_path

    plt.close(fig)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Animation successfully generated: {output_path} ({size_mb:.2f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Simulate and animate rotating-compressing star formation tracking (Single-Panel Arena).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--m", type=float, default=DEFAULT_M, help="Coupling gain m")
    parser.add_argument("--kappa", type=float, default=DEFAULT_KAPPA, help="Compression rate kappa")
    parser.add_argument("--t-end", type=float, default=DEFAULT_T_END, help="Simulation time horizon (seconds)")
    parser.add_argument("--r-min", type=float, default=DEFAULT_R_MIN, help="Min radial offset for random spawn (m)")
    parser.add_argument("--r-max", type=float, default=DEFAULT_R_MAX, help="Max radial offset for random spawn (m)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for spawn reproducibility")
    parser.add_argument("--d-safe", type=float, default=DEFAULT_D_SAFE, help="Safety separation threshold (meters)")
    parser.add_argument("--r-c", type=float, default=DEFAULT_R_C, help="Communication radius threshold (meters)")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Render DPI")
    parser.add_argument("--output", type=Path, default=None, help="Output file path (.mp4 or .gif)")
    args = parser.parse_args()

    generate_animation(
        m=args.m,
        kappa=args.kappa,
        t_end=args.t_end,
        d_safe=args.d_safe,
        r_c=args.r_c,
        r_min=args.r_min,
        r_max=args.r_max,
        seed=args.seed,
        output_path=args.output,
        fps=args.fps,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
