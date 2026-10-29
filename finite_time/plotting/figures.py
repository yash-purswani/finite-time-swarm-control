"""Figure generation functions matching publication styling."""
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import ConvexHull

from .style import use_paper_style, save_figure, COL_W, COL2_W, COLORS

REPO_ROOT = Path(__file__).resolve().parents[2]


def positive_masked_plot(ax, t, y, **kwargs):
    """Plot non-positive or sub-floor points safely by masking."""
    y = np.ma.masked_invalid(y)
    ax.plot(t, y, **kwargs)


# =============================================================================
# =============================================================================
# Figure 1: Morphing Formation Snapshots (Single Panel Clean Layout)
# =============================================================================
def plot_morphing_snapshots(arrays: Dict[str, Any], config: Dict[str, Any], folder: Path) -> None:
    """Plot multi-formation reconfiguring along a straight line trajectory in a single clean panel.

    Displays continuous agent tracks, Hungarian assigned stations, and highlighted
    target formation shapes with professional publication aesthetics.
    """
    use_paper_style()
    import matplotlib
    from scipy.spatial import ConvexHull

    raw_shapes = config.get('raw_shapes')
    names = config.get('formation_names', ['Circle', 'Wedge', 'Star', 'Arrow'])
    n = config['n']
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    t = arrays['t']
    X = arrays['X']
    q = arrays['q']
    r = q.mean(axis=1)

    fig, ax = plt.subplots(figsize=(COL2_W, 1.95))

    # 1. Reference path (straight line)
    ax.plot(r[:, 0], r[:, 1], color='#444444', ls='--', lw=0.75, alpha=0.5, zorder=1)

    # 2. Continuous agent trajectories
    for i in range(n):
        ax.plot(X[:, i, 0], X[:, i, 1], color=colors[i], lw=0.6, alpha=0.45, zorder=2)

    # 3. Highlight 4 formations at snapshot times
    for idx, (tq, name) in enumerate(zip(config['snapshot_times'], names)):
        t_idx = int(np.argmin(abs(t - tq)))
        q_snap = q[t_idx]
        x_snap = X[t_idx]
        r_snap = r[t_idx]

        # Template geometry outline
        if raw_shapes is not None and idx < len(raw_shapes):
            s_template = raw_shapes[idx]
            if name == 'Wedge':
                h = ConvexHull(s_template)
                outline = np.vstack([s_template[h.vertices], s_template[h.vertices[0]]]) + r_snap
            else:
                outline = np.vstack([s_template, s_template[0]]) + r_snap
        else:
            h = ConvexHull(q_snap)
            outline = np.vstack([q_snap[h.vertices], q_snap[h.vertices[0]]])

        # Clean, thin dotted formation contour
        ax.plot(outline[:, 0], outline[:, 1], color='#333333', ls=':', lw=0.75, alpha=0.65, zorder=3)

        # Subtle centroid cross marker
        ax.plot(r_snap[0], r_snap[1], '+', color='#111111', ms=4.0, mew=0.8, zorder=5)

        # Simple discrete markers: small open circle for stations, small solid dot for agents
        for i in range(n):
            col = colors[i]
            ax.plot(q_snap[i, 0], q_snap[i, 1], marker='o', ms=3.5, mfc='none', mec=col, mew=0.7, zorder=4)
            ax.plot(x_snap[i, 0], x_snap[i, 1], marker='o', ms=2.0, mfc=col, mec='none', zorder=5)

        # Professional clean annotation strictly within plot limits
        y_max_shape = outline[:, 1].max()
        ax.text(
            r_snap[0], y_max_shape + 0.16, rf"{name} ($t={tq:.1f}\,$s)",
            ha='center', va='bottom', fontsize=6.2, color='#222222',
            bbox=dict(boxstyle='square,pad=0.15', facecolor='#ffffff', edgecolor='#dddddd', lw=0.5, alpha=0.85),
            zorder=6
        )

    # Clean axes and bounding ensuring no labels clip the boundary
    x_min = X[..., 0].min() - 0.5
    x_max = X[..., 0].max() + 0.5
    y_min = -2.1
    y_max = 2.65

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    handles = [
        Line2D([], [], color='#444444', ls='--', lw=0.75, label='Reference $r(t)$'),
        Line2D([], [], marker='+', color='#111111', ls='', ms=4.0, mew=0.8, label='Centroid $r(t_k)$'),
        Line2D([], [], marker='o', markerfacecolor='none', markeredgecolor='#555555', ls='', ms=3.5, mew=0.7, label='Prescribed stations $q_i$'),
        Line2D([], [], marker='o', color='#555555', ls='', ms=2.0, label='Agent states $x_i$'),
        Line2D([], [], color='#888888', ls='-', lw=0.6, label='Trajectories'),
        Line2D([], [], color='#333333', ls=':', lw=0.75, label='Formation contour'),
    ]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=6, frameon=False, fontsize=6.2)

    save_figure(fig, folder, 'formation_snapshots')
    # Save copy to ECC_paper/figures as well
    acc_fig_dir = REPO_ROOT / 'ECC_paper' / 'figures'
    if acc_fig_dir.exists():
        save_figure(fig, acc_fig_dir, 'formation_snapshots')
    plt.close(fig)


# =============================================================================
# Figure 2: Convergence & Coupling Gain Evaluation (Exp 2)
# =============================================================================
def plot_gain_evaluation(runs: List[Tuple[float, Dict[str, Any], Dict[str, Any]]], folder: Path) -> None:
    """Plot 3-panel convergence figure across coupling gains m:
    (a) Station tracking ||delta||_2 vs universal bound R_bar and deadline T_bar (cut at 1e-4)
    (b) Disagreement ||eta||_2 vs rate bounds (cut at 1e-4)
    (c) Initial command ||u||_2 over the transient
    """
    use_paper_style()
    # Filter out m=2.5 if present
    filtered_runs = [r for r in runs if not np.isclose(r[0], 2.5)]
    if not filtered_runs:
        filtered_runs = runs

    fig, axes = plt.subplots(1, 3, figsize=(COL2_W, 2.5), constrained_layout=True)
    floor = 1e-4

    for m, ar, _ in filtered_runs:
        color = COLORS.get(m, '#009E73')
        t = ar['t']
        positive_masked_plot(axes[0], t, np.ma.masked_where(ar['delta_norm'] < floor, ar['delta_norm']),
                             color=color, label=f'$m={m:g}$')
        positive_masked_plot(axes[1], t, np.ma.masked_where(ar['eta_norm'] < floor, ar['eta_norm']),
                             color=color)
        positive_masked_plot(axes[1], t, np.ma.masked_where(ar['eta_bound'] < floor, ar['eta_bound']),
                             color=color, ls='--', lw=0.7)
        # Plot initial command on axes[2]
        mask = t <= 0.15
        axes[2].plot(t[mask], ar['u_norm'][mask], color=color, lw=1.0, label=f'$m={m:g}$')

    # Reference run for universal bound and deadline
    first_ar = filtered_runs[0][1]
    deadline = filtered_runs[0][2]['T_bar']
    positive_masked_plot(axes[0], first_ar['t'], np.ma.masked_where(first_ar['R_bar'] < floor, first_ar['R_bar']),
                         color='k', ls='--', lw=1.1, label=r'$\overline{\mathcal{R}}$')

    for ax in axes[:2]:
        ax.set_yscale('log')
        ax.set_ylim(floor, max(1.0, 1.5 * first_ar['R_bar'][0]))
        ax.axvline(deadline, color='k', ls=':', lw=0.8)

    axes[0].text(deadline - 0.04, 2e-4, r'$\overline{T}$', ha='right', fontsize=7)
    axes[0].legend(loc='upper right', ncol=2, fontsize=6)
    axes[1].legend(handles=[
        Line2D([], [], color='k', label='measured'),
        Line2D([], [], color='k', ls='--', label=r'bound: $\lambda_2(L_w)$')
    ], loc='upper right', fontsize=5.8)

    for ax in axes:
        ax.set_xlabel('$t$ [s]')
    for ax in axes[:2]:
        ax.set_xlim(0, min(first_ar['t'][-1], 1.15 * deadline))

    axes[0].set_ylabel(r'$\|\delta(t)\|_2$')
    axes[1].set_ylabel(r'$\|\eta(t)\|_2$')
    axes[2].set_ylabel(r'$\|u(t)\|_2$ [m/s]')
    axes[2].set_xlim(0, 0.15)

    axes[0].set_title('(a) Station tracking')
    axes[1].set_title('(b) Disagreement')
    axes[2].set_title(r'(c) Initial command $\|u\|_2$')

    save_figure(fig, folder, 'error_disagreement_control')
    acc_fig_dir = REPO_ROOT / 'ECC_paper' / 'figures'
    if acc_fig_dir.exists():
        save_figure(fig, acc_fig_dir, 'error_disagreement_control')


# =============================================================================
# Figure 3: Transient 2x3 Grid (Exp 3)
# =============================================================================
def plot_transient_2x3_grid(
    case_safe: Dict[str, Any],
    case_unsafe: Dict[str, Any],
    folder: Path,
    extra_paths: Optional[List[Path]] = None,
    transient_end: float = 2.5,
    d_safe: float = 1.0,
    r_c: float = 6.0,
    ylim_d: Tuple[float, float] = (0.7, 1.8),
    ylim_ell: Tuple[float, float] = (1.0, 6.5),
    ylim_margin: Tuple[float, float] = (-0.05, 1.60),
    filename: str = 'swarm_star_2x3_comparison'
) -> None:
    """Plot 2x3 grid comparing certified safe (Row 1) vs uncertified collision/breach (Row 2)."""
    use_paper_style()
    fig, axes = plt.subplots(2, 3, figsize=(COL2_W, 4.3), sharex=True, layout='none')

    cases = [case_safe, case_unsafe]
    row_colors = ['#009E73', '#e74c3c']

    for row, case in enumerate(cases):
        t = case['t']
        mask = t <= transient_end
        tz = t[mask]
        dx = case['d_x'][mask]
        ellx = case['ell_x'][mask]
        dq = case['d_q'][mask]
        ellq = case['ell_H'][mask]
        s = case['s'][mask]
        rho = case['rho_H'][mask]
        color = row_colors[row]
        certified = bool(case.get('certified', False))

        # Col 0: Minimum pair separation d_x(t)
        ax0 = axes[row, 0]
        ax0.plot(tz, dx, color=color, lw=1.0, label=r'$d_x(t)$ (observed)')
        ax0.plot(tz, dq - rho, color='k', ls='--', lw=1.0, label=r'$d_q(t) - \rho_{\mathcal{H}}(t)$')
        ax0.axhline(d_safe, color='#cc0000', ls=':', lw=1.0, label=f'$d_{{\\mathrm{{safe}}}} = {d_safe:g}$ m')

        # Red shading for mid-transient collision window in Row 2
        if row == 1:
            collision_mask = dx < d_safe
            if np.any(collision_mask):
                ax0.fill_between(tz, dx, d_safe, where=collision_mask, color='#e74c3c', alpha=0.35,
                                 label=r'collision breach ($d_x < d_{\mathrm{safe}}$)')

        ax0.set_ylabel("Distance [m]")
        ax0.set_ylim(*ylim_d)
        ax0.legend(loc='upper right', fontsize=5.8)

        # Col 1: Maximum pair separation ell_x(t)
        ax1 = axes[row, 1]
        ax1.plot(tz, ellx, color=color, lw=1.0, label=r'$\ell_x(t)$ (observed)')
        ax1.plot(tz, ellq + rho, color='k', ls='--', lw=1.0, label=r'$\ell_q(t) + \rho_{\mathcal{H}}(t)$')
        ax1.axhline(r_c, color='#cc0000', ls=':', lw=1.0, label=f'$R_c = {r_c:g}$ m')

        # Red shading for mid-transient connectivity violation in Row 2
        if row == 1:
            conn_mask = ellx > r_c
            if np.any(conn_mask):
                ax1.fill_between(tz, ellx, r_c, where=conn_mask, color='#e74c3c', alpha=0.35,
                                 label=r'connectivity breach ($\ell_x > R_c$)')

        ax1.set_ylabel("Length [m]")
        ax1.set_ylim(*ylim_ell)
        ax1.legend(loc='upper right', fontsize=5.8)

        # Col 2: Margin s(t) vs Distortion rho_H(t)
        ax2 = axes[row, 2]
        ax2.plot(tz, s, color='#0072B2', lw=1.0, label=r'$s(t)$')
        ax2.plot(tz, rho, color=color, ls='--', lw=1.0, label=r'$\rho_{\mathcal{H}}(t)$')

        if certified or np.all(s >= rho):
            ax2.fill_between(tz, rho, s, where=(s >= rho), color='#2ecc71', alpha=0.25,
                             label=r'certified ($s \geq \rho_{\mathcal{H}}$)')
        else:
            ax2.fill_between(tz, s, rho, where=(rho > s), color='#f39c12', alpha=0.35,
                             label=r'uncertified ($s < \rho_{\mathcal{H}}$)')
            ax2.fill_between(tz, rho, s, where=(s >= rho), color='#2ecc71', alpha=0.15)

        ax2.set_ylabel("Margin [m]")
        ax2.set_ylim(*ylim_margin)
        ax2.legend(loc='upper right', fontsize=5.8)

        for ax in axes[row]:
            ax.set_xlim(0, transient_end)
            t_bar = case.get('T_bar_univ', case.get('T_bar', None))
            if t_bar is not None and t_bar <= transient_end:
                ax.axvline(t_bar, color='#888888', ls=':', lw=0.7)

    axes[0, 0].set_title(r'(a) Minimum pairwise distance $d_x(t)$')
    axes[0, 1].set_title(r'(b) Maximum pairwise distance $\ell_x(t)$')
    axes[0, 2].set_title(r'(c) Geometric margin $s(t)$ vs. tube $\rho_{\mathcal{H}}(t)$')

    for ax in axes[1]:
        ax.set_xlabel(r'Time $t$ [s]')

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.92, hspace=0.25, wspace=0.28)
    save_figure(fig, folder, filename)

    if extra_paths:
        for p in extra_paths:
            p = Path(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p / f'{filename}.pdf', bbox_inches='tight')
            fig.savefig(p / f'{filename}.png', bbox_inches='tight', dpi=300)


# =============================================================================
# Figure 4: Parameter Map and Peak Control (Exp 4)
# =============================================================================
def plot_parameter_map_and_control(
    kappas: np.ndarray,
    ms: np.ndarray,
    classes: np.ndarray,
    rows: List[Dict[str, Any]],
    folder: Path,
    boundary: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> None:
    """Plot safety classification map with design boundary, and peak control heatmap."""
    use_paper_style()
    labels = [
        'Category A: Certified safe by Corollary 2',
        'Category B: Uncertified, observed safe',
        'Category C: Observed transient collision',
        'Category D: Infeasible geometry',
        'VALIDATION FAILURE'
    ]
    mapping = {key: i for i, key in enumerate(['A', 'B', 'C', 'D', 'VALIDATION_FAILURE'])}
    z = np.vectorize(mapping.get)(classes)
    colors = ['#009E73', '#56B4E9', '#D55E00', '#BDBDBD', '#CC00CC']

    # Map Panel
    fig, ax = plt.subplots(figsize=(COL_W, 2.85))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 5.5), 5)
    ax.pcolormesh(kappas, ms, z, cmap=cmap, norm=norm, shading='nearest', rasterized=False)

    if boundary is not None:
        boundary_k, boundary_m = boundary
        ax.plot(boundary_k, boundary_m, 'k--', lw=1.2, label=r'Design boundary $m_{\mathrm{req}}(\kappa)$')

    ax.scatter(np.tile(kappas, len(ms)), np.repeat(ms, len(kappas)), s=6, c='k', alpha=0.65)
    ax.set_title(f'{len(ms)} x {len(kappas)} sampled configurations', fontsize=7)
    ax.set_ylim(ms[0], ms[-1])
    ax.set_xlim(kappas[0], kappas[-1])
    ax.set_xlabel(r'Maneuver compression rate $\kappa$')
    ax.set_ylabel(r'Coupling gain $m$')

    handles = [
        Line2D([], [], marker='s', ls='', color=colors[i], label=labels[i])
        for i, key in enumerate(['A', 'B', 'C', 'D', 'VALIDATION_FAILURE']) if np.any(classes == key)
    ]
    if boundary is not None and np.any(np.isfinite(boundary[1])):
        handles.append(Line2D([], [], ls='--', color='k', label=r'Design boundary $m_{\mathrm{req}}(\kappa)$'))

    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize=5.8, frameon=False)
    ax.grid(False)
    save_figure(fig, folder, 'parameter_map')

    # Peak Control Panel
    peak = np.array([r['peak_control'] for r in rows]).reshape(len(ms), len(kappas))
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    image = ax.pcolormesh(kappas, ms, peak, cmap='viridis', shading='nearest', rasterized=False)
    if boundary is not None:
        ax.plot(boundary_k, boundary_m, 'w--', lw=0.9)
    ax.scatter(np.tile(kappas, len(ms)), np.repeat(ms, len(kappas)), s=5, c='w', alpha=0.6)
    ax.set_xlabel(r'Maneuver compression rate $\kappa$')
    ax.set_ylabel(r'Coupling gain $m$')
    fig.colorbar(image, ax=ax, label=r'Peak $\|u\|_2$ [m/s]')
    ax.grid(False)
    save_figure(fig, folder, 'parameter_control')
