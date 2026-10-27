"""Figure generation functions matching publication styling."""
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import ConvexHull

from .style import use_paper_style, save_figure, COL_W, COL2_W, COLORS


def positive_masked_plot(ax, t, y, **kwargs):
    """Plot non-positive or sub-floor points safely by masking."""
    y = np.ma.masked_invalid(y)
    ax.plot(t, y, **kwargs)


# =============================================================================
# =============================================================================
# Figure 1: Morphing Formation Snapshots
# =============================================================================
def plot_morphing_snapshots(arrays: Dict[str, Any], config: Dict[str, Any], folder: Path) -> None:
    """Plot multi-formation reconfiguring snapshots along the reference path matching standalone quality."""
    use_paper_style()
    import matplotlib
    from ..formations import hexagon, wedge, star, arrow
    from scipy.spatial import ConvexHull

    fig, axes = plt.subplots(1, 4, figsize=(COL2_W, 2.65), sharey=True)
    colors = plt.cm.tab10(np.arange(config['n']))
    names = ['Ring', 'Wedge', 'Star', 'Arrow']
    t = arrays['t']
    X = arrays['X']
    q = arrays['q']
    r = q.mean(axis=1)

    snapshot_indices = [int(np.argmin(abs(t - tq))) for tq in config['snapshot_times']]
    shown = np.concatenate([
        np.concatenate([q[index], X[np.searchsorted(t, max(0, tq - 1.5)):index + 1].reshape(-1, 2)])
        for index, tq in zip(snapshot_indices, config['snapshot_times'])
    ])
    y_limits = (shown[:, 1].min() - 0.4, shown[:, 1].max() + 0.4)

    L = config['radius']
    for ax, tq, name in zip(axes, config['snapshot_times'], names):
        index = int(np.argmin(abs(t - tq)))
        start = np.searchsorted(t, max(0, tq - 1.5))
        template = {
            'Ring': lambda: hexagon(config['n'], L),
            'Wedge': lambda: wedge(config['n'], 0.6 * L),
            'Star': lambda: star(config['n'], 1.1 * L),
            'Arrow': lambda: arrow(config['n'], L)
        }[name]()
        if name == 'Wedge':
            template = template[ConvexHull(template).vertices]
        outline = np.vstack([template, template[0]]) + r[index]
        ax.plot(outline[:, 0], outline[:, 1], color='#aaa', lw=0.6, alpha=0.6, zorder=0)

        for i, col in enumerate(colors):
            ax.plot(X[start:index + 1, i, 0], X[start:index + 1, i, 1], color=col, lw=0.65, alpha=0.7)
            ax.scatter(*q[index, i], s=38, facecolors='none', edgecolors=[col], lw=0.8, zorder=4)
            ax.scatter(*X[index, i], s=12, color=col, zorder=5)
            ax.annotate(str(i + 1), q[index, i], xytext=(3, 3), textcoords='offset points', fontsize=5, color=col)

        ax.plot(r[start:index + 1, 0], r[start:index + 1, 1], 'k--', lw=0.8)
        ax.plot(*r[index], '+', color='k', ms=6)

        local = np.concatenate([q[index], X[start:index + 1].reshape(-1, 2)])
        ax.set_xlim(local[:, 0].min() - 0.4, local[:, 0].max() + 0.4)
        ax.set_ylim(*y_limits)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{name}, $t={tq:g}$')
        ax.set_xlabel('$x$ (normalized)')

    axes[0].set_ylabel('$y$ (normalized)')
    fig.legend(
        handles=[
            Line2D([], [], marker='o', color='k', ls='', ms=3, label='agents'),
            Line2D([], [], marker='o', markerfacecolor='none', color='k', ls='', ms=5, label='assigned stations'),
            Line2D([], [], color='k', ls='--', label='reference trajectory')
        ],
        loc='outside upper center' if matplotlib.__version__ >= '3.7' else 'upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 1.11),
        frameon=False
    )
    save_figure(fig, folder, 'formation_snapshots')


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
    fig, axes = plt.subplots(1, 3, figsize=(COL2_W, 2.55))
    floor = 1e-4

    for m, ar, _ in runs:
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
    first_ar = runs[0][1]
    deadline = runs[0][2]['T_bar']
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
        Line2D([], [], color='k', ls='--', label=r'bound: integrated $\lambda_2(L_w)$')
    ], loc='upper right', fontsize=5.8)

    for ax in axes:
        ax.set_xlabel('$t$ [s]')
    for ax in axes[:2]:
        ax.set_xlim(0, min(first_ar['t'][-1], 1.15 * deadline))

    axes[0].set_ylabel(r'$\|\delta\|_2$')
    axes[1].set_ylabel(r'$\|\eta\|_2$')
    axes[2].set_ylabel(r'$\|u\|_2$ [m/s]')
    axes[2].set_xlim(0, 0.15)

    axes[0].set_title('(a) Station tracking')
    axes[1].set_title('(b) Disagreement')
    axes[2].set_title(r'(c) Initial command $\|u\|_2$')

    save_figure(fig, folder, 'error_disagreement_control')


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
    r_c: float = 6.0
) -> None:
    """Plot 2x3 grid comparing certified safe (Row 1) vs uncertified collision (Row 2)."""
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
        ax0.plot(tz, dx, color=color, lw=1.6, label=r'$d_x(t)$ (observed)')
        ax0.plot(tz, dq - rho, color='k', ls='--', lw=1.1, label=r'$d_q(t) - \rho_{\mathcal{H}}(t)$')
        ax0.axhline(d_safe, color='#cc0000', ls=':', lw=1.4, label=f'$d_{{\\mathrm{{safe}}}} = {d_safe:g}$ m')

        # Red shading for mid-transient collision window in Row 2
        if row == 1:
            collision_mask = dx < d_safe
            if np.any(collision_mask):
                ax0.fill_between(tz, dx, d_safe, where=collision_mask, color='#e74c3c', alpha=0.35,
                                 label=r'collision breach ($d_x < d_{\mathrm{safe}}$)')

        ax0.set_ylabel("Distance [m]")
        ax0.set_ylim(0.7, 1.8)
        ax0.legend(loc='upper right', fontsize=5.8)

        # Col 1: Maximum pair separation ell_x(t)
        ax1 = axes[row, 1]
        ax1.plot(tz, ellx, color=color, lw=1.6, label=r'$\ell_x(t)$ (observed)')
        ax1.plot(tz, ellq + rho, color='k', ls='--', lw=1.1, label=r'$\ell_q(t) + \rho_{\mathcal{H}}(t)$')
        ax1.axhline(r_c, color='#cc0000', ls=':', lw=1.4, label=f'$R_c = {r_c:g}$ m')
        ax1.set_ylabel("Length [m]")
        ax1.set_ylim(3.5, 6.3)
        ax1.legend(loc='upper right', fontsize=5.8)

        # Col 2: Margin s(t) vs Distortion rho_H(t)
        ax2 = axes[row, 2]
        ax2.plot(tz, s, color='#0072B2', lw=1.8, label=r'$s(t) = d_q - d_{\mathrm{safe}}$')
        ax2.plot(tz, rho, color=color, ls='--', lw=1.4, label=r'$\rho_{\mathcal{H}}(t)$')

        if certified or np.all(s >= rho):
            ax2.fill_between(tz, rho, s, where=(s >= rho), color='#2ecc71', alpha=0.25,
                             label=r'certified ($s \geq \rho_{\mathcal{H}}$)')
        else:
            ax2.fill_between(tz, s, rho, where=(rho > s), color='#f39c12', alpha=0.35,
                             label=r'uncertified ($s < \rho_{\mathcal{H}}$)')
            ax2.fill_between(tz, rho, s, where=(s >= rho), color='#2ecc71', alpha=0.15)

        ax2.set_ylabel("Margin [m]")
        ax2.set_ylim(-0.05, 0.65)
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
    save_figure(fig, folder, 'swarm_star_3x3_comparison')

    if extra_paths:
        for p in extra_paths:
            p = Path(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, bbox_inches='tight', dpi=300)


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
