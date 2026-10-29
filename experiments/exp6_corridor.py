#!/usr/bin/env python3
r"""Experiment 6: Narrow Corridor Traversal via Morphing (Ring -> Chevron -> Ring).

Demonstrates:
- Arbitrary formation morphing tracking along a straight path traversing a narrow corridor.
- Preservation of a spanning backbone 10-cycle graph $\mathcal{H}$.
- Compatibility with existing safe formation generation / corridor constraint techniques.
- Finite-time error convergence and verified safety margins.

Outputs:
- Figure (1x3 subplots): results/experiment_6/corridor_maneuver_1x3.pdf / .png
- Certificate plots (1x3 subplots): results/experiment_6/corridor_certificates.pdf / .png
- Optional animation: results/experiment_6/corridor_maneuver.gif
- Summary metrics: results/experiment_6/corridor_summary.json

Usage:
    python3 experiments/exp6_corridor.py [--output DIR] [--m GAIN] [--rerun] [--animate]
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finite_time.utils import write_json

# =============================================================================
# DEFAULT PARAMETERS (Matching paper: n=10, dsafe=1.0, Rc=6.0, beta=0.5)
# =============================================================================
DEFAULT_N = 10
DEFAULT_D = 2
DEFAULT_BETA = 0.5
DEFAULT_D_SAFE = 1.0
DEFAULT_R_C = 6.0
DEFAULT_EPS_W = 0.05
DEFAULT_NU = 1.0
DEFAULT_M = 10.0

DEFAULT_X_CORR_START = 12.0
DEFAULT_X_CORR_END = 30.0
DEFAULT_Y_WALL = 1.90           # Half-width (total corridor width = 3.80 m)
DEFAULT_WALL_HEIGHT = 4.20      # Thickness for visualization

DEFAULT_V_REF = 2.50            # Swarm reference speed [m/s]
DEFAULT_T_FINAL = 17.5          # Total maneuver duration [s]
DEFAULT_SEED = 42

# 10-Cycle Backbone Edges (Agent 1 to 10 in cycle)
BACKBONE_EDGES = [
    (0, 1),  # 1 - 2
    (1, 2),  # 2 - 3
    (2, 3),  # 3 - 4
    (3, 4),  # 4 - 5
    (4, 5),  # 5 - 6
    (5, 6),  # 6 - 7
    (6, 7),  # 7 - 8
    (7, 8),  # 8 - 9
    (8, 9),  # 9 - 10
    (9, 0)   # 10 - 1
]


def build_backbone_laplacian(n: int, edges: list) -> Tuple[np.ndarray, float]:
    L_H = np.zeros((n, n), dtype=float)
    for i, j in edges:
        L_H[i, i] += 1.0
        L_H[j, j] += 1.0
        L_H[i, j] -= 1.0
        L_H[j, i] -= 1.0
    eigvals = np.sort(eigh(L_H, eigvals_only=True))
    lambda_2 = float(eigvals[1])
    return L_H, lambda_2


def get_base_templates(n: int = DEFAULT_N) -> Tuple[np.ndarray, np.ndarray]:
    """Generate centered Ring (radius 3.6m) and Chevron (fits within corridor y in [-1.5, 1.5]m)."""
    R_ring = 3.60
    theta = np.zeros(n, dtype=float)
    for idx in range(n):
        agent_num = idx + 1
        theta[idx] = np.deg2rad((agent_num - 5) * 36.0)

    P_ring = np.column_stack([R_ring * np.cos(theta), R_ring * np.sin(theta)])

    P_chev_raw = np.array([
        [-1.20, -0.70],  # 1: Agent 1 (lower rear notch)
        [ 0.00, -1.50],  # 2: Agent 2 (lower rear wing tip)
        [ 2.00, -1.10],  # 3: Agent 3 (lower mid wing)
        [ 4.00, -0.70],  # 4: Agent 4 (lower front wing)
        [ 5.80,  0.00],  # 5: Agent 5 (Apex / Leader)
        [ 4.00,  0.70],  # 6: Agent 6 (upper front wing)
        [ 2.00,  1.10],  # 7: Agent 7 (upper mid wing)
        [ 0.00,  1.50],  # 8: Agent 8 (upper rear wing tip)
        [-1.20,  0.70],  # 9: Agent 9 (upper rear notch)
        [ 0.10,  0.00]   # 10: Agent 10 (center notch tip)
    ], dtype=float)

    P_ring_centered = P_ring - np.mean(P_ring, axis=0, keepdims=True)
    P_chev_centered = P_chev_raw - np.mean(P_chev_raw, axis=0, keepdims=True)
    return P_ring_centered, P_chev_centered


def quintic_step(tau: float) -> Tuple[float, float]:
    """C^2 smooth quintic step function on tau in [0, 1]."""
    tau_c = np.clip(tau, 0.0, 1.0)
    s = 10.0 * tau_c**3 - 15.0 * tau_c**4 + 6.0 * tau_c**5
    ds = (30.0 * tau_c**2 - 60.0 * tau_c**3 + 30.0 * tau_c**4) * (1.0 if 0.0 <= tau <= 1.0 else 0.0)
    return float(s), float(ds)


def morph_schedule(t: float) -> Tuple[float, float]:
    """alpha(t) in [0, 1]: 0 -> Ring, 1 -> Chevron."""
    t1_s, t1_e = 1.2, 2.8
    t2_s, t2_e = 13.5, 15.5

    if t < t1_s:
        return 0.0, 0.0
    elif t < t1_e:
        s, ds = quintic_step((t - t1_s) / (t1_e - t1_s))
        return s, ds / (t1_e - t1_s)
    elif t < t2_s:
        return 1.0, 0.0
    elif t < t2_e:
        s, ds = quintic_step((t - t2_s) / (t2_e - t2_s))
        return 1.0 - s, -ds / (t2_e - t2_s)
    else:
        return 0.0, 0.0


def make_station_fn(v_ref: float = DEFAULT_V_REF):
    p_ring, p_chev = get_base_templates(DEFAULT_N)
    dp_template = p_chev - p_ring

    def station_fn(t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r = np.array([v_ref * t, 0.0], dtype=float)
        dr = np.array([v_ref, 0.0], dtype=float)
        alpha, dalpha = morph_schedule(t)
        p = p_ring + alpha * dp_template
        dp = dalpha * dp_template
        q = r[None, :] + p
        dq = dr[None, :] + dp
        return q, dq, r, dr

    return station_fn


def phi_beta(z: np.ndarray, beta: float = DEFAULT_BETA) -> np.ndarray:
    return np.sign(z) * (np.abs(z)**beta)


def simulate_corridor(m: float = DEFAULT_M, seed: int = DEFAULT_SEED, t_final: float = DEFAULT_T_FINAL) -> Dict[str, Any]:
    n = DEFAULT_N
    d = DEFAULT_D
    beta = DEFAULT_BETA
    d_safe = DEFAULT_D_SAFE
    R_c = DEFAULT_R_C
    eps_w = DEFAULT_EPS_W
    nu = DEFAULT_NU
    w_under = 1.0 / (R_c**nu + eps_w)

    _, lambda_2_H = build_backbone_laplacian(n, BACKBONE_EDGES)
    gamma_H = 1.0 + m * w_under * lambda_2_H
    M_mat = m * np.eye(d)

    station_fn = make_station_fn(DEFAULT_V_REF)
    q0, _, _, _ = station_fn(0.0)

    # Initial perturbation
    np.random.seed(seed)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + 0.2
    radii = np.random.uniform(0.15, 0.22, size=n)
    delta0_mat = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    x0 = (q0 + delta0_mat).flatten()

    delta0_norm = float(np.linalg.norm(delta0_mat))
    delta0_mean = np.mean(delta0_mat, axis=0, keepdims=True)
    eta0_norm = float(np.linalg.norm(delta0_mat - delta0_mean))
    rho_0 = float(np.sqrt(2.0) * eta0_norm)
    T_deadline = float(np.log(1.0 + delta0_norm**(1.0 - beta)) / (1.0 - beta))

    def rhs(t, x_flat):
        x = x_flat.reshape((n, d))
        q, dq, _, _ = station_fn(t)
        delta = x - q

        diff_x = x[:, None, :] - x[None, :, :]
        dist_x = np.linalg.norm(diff_x, axis=-1)
        adj = ((dist_x < R_c) & (dist_x > 0)).astype(float)
        W = adj / (dist_x**nu + eps_w)
        L_w = np.diag(np.sum(W, axis=1)) - W

        coupling = (L_w @ delta) @ M_mat
        u = dq - delta - phi_beta(delta, beta) - coupling
        return u.flatten()

    t_eval = np.linspace(0.0, t_final, 1401)
    sol = solve_ivp(rhs, [0.0, t_final], x0, t_eval=t_eval, method='RK45', rtol=1e-7, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"Corridor simulation failed: {sol.message}")

    N_steps = len(t_eval)
    x_traj = sol.y.T.reshape((N_steps, n, d))
    q_traj = np.zeros_like(x_traj)
    r_traj = np.zeros((N_steps, d))

    delta_norm_t = np.zeros(N_steps)
    eta_norm_t = np.zeros(N_steps)
    R_bound_t = np.zeros(N_steps)
    rho_H_t = np.zeros(N_steps)

    d_q_t = np.zeros(N_steps)
    l_H_t = np.zeros(N_steps)
    s_margin_t = np.zeros(N_steps)
    d_x_t = np.zeros(N_steps)
    l_x_t = np.zeros(N_steps)

    for k, t in enumerate(t_eval):
        q, _, r, _ = station_fn(t)
        q_traj[k] = q
        r_traj[k] = r
        x = x_traj[k]

        delta = x - q
        eta = delta - np.mean(delta, axis=0, keepdims=True)
        delta_norm_t[k] = np.linalg.norm(delta)
        eta_norm_t[k] = np.linalg.norm(eta)

        inside = (1.0 + delta0_norm**(1.0 - beta)) * np.exp(-(1.0 - beta) * t) - 1.0
        R_bound_t[k] = (max(0.0, inside))**(1.0 / (1.0 - beta))
        rho_H_t[k] = min(np.sqrt(2.0) * R_bound_t[k], rho_0 * np.exp(-gamma_H * t))

        dist_q = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=-1)
        dist_x = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        i_upper, j_upper = np.triu_indices(n, k=1)
        d_q_t[k] = np.min(dist_q[i_upper, j_upper])
        d_x_t[k] = np.min(dist_x[i_upper, j_upper])

        l_H_t[k] = max(np.linalg.norm(q[i] - q[j]) for i, j in BACKBONE_EDGES)
        l_x_t[k] = max(np.linalg.norm(x[i] - x[j]) for i, j in BACKBONE_EDGES)

        s_margin_t[k] = min(d_q_t[k] - d_safe, R_c - l_H_t[k])

    mask_T = (t_eval > 1e-4) & (t_eval < T_deadline)
    gamma_req = float(np.max(np.log(rho_0 / s_margin_t[mask_T]) / t_eval[mask_T]))

    return {
        't_eval': t_eval,
        'x_traj': x_traj,
        'q_traj': q_traj,
        'r_traj': r_traj,
        'delta_norm_t': delta_norm_t,
        'eta_norm_t': eta_norm_t,
        'rho_H_t': rho_H_t,
        'd_q_t': d_q_t,
        'd_x_t': d_x_t,
        'l_H_t': l_H_t,
        'l_x_t': l_x_t,
        's_margin_t': s_margin_t,
        'delta0_norm': delta0_norm,
        'rho_0': rho_0,
        'T_deadline': T_deadline,
        'lambda_2_H': lambda_2_H,
        'gamma_req': gamma_req,
        'gamma_H': gamma_H,
    }


def plot_corridor_1x3(data: Dict[str, Any], output_path: Path, paper_copy_path: Optional[Path] = None) -> None:
    """Generate high-aspect-ratio 1x3 corridor maneuver snapshots with focused x-windows."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10.5,
        'legend.fontsize': 9
    })

    t_eval = data['t_eval']
    x_traj = data['x_traj']
    q_traj = data['q_traj']
    r_traj = data['r_traj']
    n = DEFAULT_N
    agent_colors = plt.cm.tab10(np.linspace(0, 1, n))

    snap_times = [1.2, 8.0, 16.2]
    snap_titles = [
        r"(a) Approach in Ring ($t = 1.2\,$s)",
        r"(b) Traversal in Chevron ($t = 8.0\,$s)",
        r"(c) Exit & Ring Recovery ($t = 16.2\,$s)"
    ]
    x_windows = [(-2.0, 15.0), (12.0, 29.0), (28.0, 45.0)]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), sharey=True)

    for ax, t_snap, title, xlim in zip(axes, snap_times, snap_titles, x_windows):
        idx = int(np.argmin(np.abs(t_eval - t_snap)))
        x_now = x_traj[idx]
        q_now = q_traj[idx]
        r_now = r_traj[idx]

        # Corridor Walls
        rect_top = patches.Rectangle(
            (DEFAULT_X_CORR_START, DEFAULT_Y_WALL),
            DEFAULT_X_CORR_END - DEFAULT_X_CORR_START, DEFAULT_WALL_HEIGHT,
            facecolor='#d95f02', alpha=0.22, edgecolor='#a63603', lw=1.5, hatch='///', zorder=2
        )
        rect_bot = patches.Rectangle(
            (DEFAULT_X_CORR_START, -DEFAULT_Y_WALL - DEFAULT_WALL_HEIGHT),
            DEFAULT_X_CORR_END - DEFAULT_X_CORR_START, DEFAULT_WALL_HEIGHT,
            facecolor='#d95f02', alpha=0.22, edgecolor='#a63603', lw=1.5, hatch='///', zorder=2
        )
        ax.add_patch(rect_top)
        ax.add_patch(rect_bot)

        if xlim[0] < DEFAULT_X_CORR_START < xlim[1]:
            ax.text(DEFAULT_X_CORR_START + 0.3, DEFAULT_Y_WALL + 0.4, 'Corridor Wall',
                    fontsize=8, color='#a63603', fontweight='bold', zorder=4)

        # Agent Trajectories
        for i in range(n):
            ax.plot(
                x_traj[:idx+1, i, 0], x_traj[:idx+1, i, 1],
                color=agent_colors[i], lw=1.1, alpha=0.55, zorder=1
            )

        ax.plot([0, DEFAULT_V_REF * DEFAULT_T_FINAL], [0, 0], 'k--', lw=1.0, alpha=0.4, label='Reference $r(t)$', zorder=1)

        # Backbone Edges
        edge_lines = [[x_now[i], x_now[j]] for i, j in BACKBONE_EDGES]
        lc = LineCollection(edge_lines, colors='#2b5c8f', linewidths=1.6, alpha=0.85, zorder=3, label=r'Backbone $\mathcal{H}$')
        ax.add_collection(lc)

        # Stations & Agents
        ax.scatter(
            q_now[:, 0], q_now[:, 1], s=65, facecolors='none', edgecolors='k',
            linewidths=1.2, zorder=4, label='Stations $q_i(t)$'
        )
        for i in range(n):
            ax.scatter(
                x_now[i, 0], x_now[i, 1], s=42, color=agent_colors[i],
                edgecolors='k', linewidths=0.6, zorder=5,
                label='Agents $x_i(t)$' if i == 0 else ''
            )
            ax.text(
                x_now[i, 0] + 0.28, x_now[i, 1] + 0.28, f'{i+1}',
                fontsize=8.5, fontweight='bold', color='#1f2d3d', zorder=6
            )

        ax.scatter(r_now[0], r_now[1], marker='+', s=85, color='k', linewidths=1.6, zorder=7, label='Centroid $c(t)$')

        ax.set_title(title, fontweight='bold', pad=8)
        ax.set_xlabel('$x$ [m]')
        ax.set_xlim(xlim)
        ax.set_ylim(-4.6, 4.6)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.4)

    axes[0].set_ylabel('$y$ [m]')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved 1x3 corridor maneuver figure to {output_path.with_suffix('.pdf')}")

    if paper_copy_path is not None:
        paper_copy_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_copy_path.with_suffix('.pdf'), bbox_inches='tight')
        fig.savefig(paper_copy_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"Copied figure to paper directory: {paper_copy_path.with_suffix('.pdf')}")

    plt.close(fig)


def plot_corridor_certificates(data: Dict[str, Any], output_path: Path) -> None:
    """Generate certificate verification plots (1x3 subplots)."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10.5,
        'legend.fontsize': 8.5
    })

    t_eval = data['t_eval']
    d_x_t = data['d_x_t']
    d_q_t = data['d_q_t']
    rho_H_t = data['rho_H_t']
    l_x_t = data['l_x_t']
    l_H_t = data['l_H_t']
    s_margin_t = data['s_margin_t']
    T_deadline = data['T_deadline']

    fig_cert, ax_c = plt.subplots(1, 3, figsize=(13.5, 3.4))

    # (a) Minimum pairwise distance
    ax_c[0].plot(t_eval, d_x_t, color='#1b9e77', lw=1.8, label='$d_x(t)$ (observed)')
    ax_c[0].plot(t_eval, d_q_t - rho_H_t, 'k--', lw=1.3, label=r'$d_q(t) - \rho_{\mathcal{H}}(t)$ (cert. bound)')
    ax_c[0].axhline(DEFAULT_D_SAFE, color='#d95f02', ls=':', lw=1.4, label=r'$d_{\mathrm{safe}} = 1.0\,$m')
    ax_c[0].set_title('(a) Min Pairwise Distance $d_x(t)$')
    ax_c[0].set_xlabel('Time $t$ [s]')
    ax_c[0].set_ylabel('Distance [m]')
    ax_c[0].set_ylim(0.5, 3.0)
    ax_c[0].legend(loc='lower right')
    ax_c[0].grid(True, ls=':', alpha=0.4)

    # (b) Maximum backbone edge length
    ax_c[1].plot(t_eval, l_x_t, color='#1b9e77', lw=1.8, label=r'$\ell_x(t)$ (observed)')
    ax_c[1].plot(t_eval, l_H_t + rho_H_t, 'k--', lw=1.3, label=r'$\ell_{\mathcal{H}}(t) + \rho_{\mathcal{H}}(t)$ (cert. bound)')
    ax_c[1].axhline(DEFAULT_R_C, color='#d95f02', ls=':', lw=1.4, label=r'$R_c = 6.0\,$m')
    ax_c[1].set_title(r'(b) Max Backbone Edge $\ell_x(t)$')
    ax_c[1].set_xlabel('Time $t$ [s]')
    ax_c[1].set_ylabel('Length [m]')
    ax_c[1].set_ylim(1.0, 6.5)
    ax_c[1].legend(loc='upper right')
    ax_c[1].grid(True, ls=':', alpha=0.4)

    # (c) Geometric margin s(t) vs Disagreement Tube rho_H(t)
    ax_c[2].plot(t_eval, s_margin_t, 'k-', lw=1.6, label='Geometric margin $s(t)$')
    ax_c[2].plot(t_eval, rho_H_t, color='#1b9e77', ls='--', lw=1.6, label=r'Disagreement tube $\rho_{\mathcal{H}}(t)$')
    ax_c[2].fill_between(t_eval, rho_H_t, s_margin_t, color='#1b9e77', alpha=0.18, label=r'Certified margin ($s > \rho_{\mathcal{H}}$)')
    ax_c[2].axvline(T_deadline, color='#7570b3', ls='-.', lw=1.2, label=fr'Deadline $T = {T_deadline:.2f}\,$s')
    ax_c[2].set_title(r'(c) Margin $s(t)$ vs. Tube $\rho_{\mathcal{H}}(t)$')
    ax_c[2].set_xlabel('Time $t$ [s]')
    ax_c[2].set_ylabel('Margin [m]')
    ax_c[2].set_xlim(0, 4.0)
    ax_c[2].set_ylim(0, 1.8)
    ax_c[2].legend(loc='upper right')
    ax_c[2].grid(True, ls=':', alpha=0.4)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_cert.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    fig_cert.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved certificate plots to {output_path.with_suffix('.pdf')}")
    plt.close(fig_cert)


def run_experiment_6(output_dir: Path, m: float = DEFAULT_M, rerun: bool = False, animate: bool = False) -> Dict[str, Any]:
    folder = output_dir / 'experiment_6'
    folder.mkdir(parents=True, exist_ok=True)
    summary_path = folder / 'corridor_summary.json'

    if not rerun and summary_path.exists():
        print(f"Loading cached Experiment 6 summary from {summary_path}")
        with open(summary_path, 'r') as f:
            import json
            return json.load(f)

    print("Running Experiment 6: Corridor Traversal via Morphing...")
    data = simulate_corridor(m=m)

    paper_folder = REPO_ROOT / 'ECC_paper' / 'figures'
    plot_corridor_1x3(data, folder / 'corridor_maneuver_1x3', paper_folder / 'corridor_maneuver_1x3')
    plot_corridor_certificates(data, folder / 'corridor_certificates')

    summary = {
        'name': 'corridor_maneuver',
        'n': DEFAULT_N,
        'm': m,
        'd_safe': DEFAULT_D_SAFE,
        'R_c': DEFAULT_R_C,
        'delta0_norm': float(data['delta0_norm']),
        'rho_0': float(data['rho_0']),
        'T_deadline': float(data['T_deadline']),
        'lambda_2_H': float(data['lambda_2_H']),
        'gamma_req': float(data['gamma_req']),
        'gamma_H': float(data['gamma_H']),
        'min_d_x': float(np.min(data['d_x_t'])),
        'max_l_x': float(np.max(data['l_x_t'])),
        'certified': bool(data['gamma_H'] > data['gamma_req']),
    }
    write_json(summary_path, summary)
    print(f"Experiment 6 complete. Summary saved to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 6: Corridor Traversal Maneuver")
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'results')
    parser.add_argument('--m', type=float, default=DEFAULT_M)
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--animate', action='store_true')
    args = parser.parse_args()

    run_experiment_6(output_dir=args.output, m=args.m, rerun=args.rerun, animate=args.animate)


if __name__ == '__main__':
    main()
