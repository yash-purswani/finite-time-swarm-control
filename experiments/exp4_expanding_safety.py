#!/usr/bin/env python3
r"""Experiment 4: Expanding Star Formation - Transient Connectivity Preservation ($2 \times 3$ Grid).

Simulates rapid star expansion (\kappa = 8.0, a_0 = 1.70 -> a_\infty = 3.02) over t in [0, 2.5] s, comparing:
- Row 1: Certified Safe Case (m = 25.0, \gamma > \gamma_req, backbone links strictly preserved < R_c = 6.0 m)
- Row 2: Unsafe Mid-Transient Connectivity Breach (m = 0.0, \gamma < \gamma_req, link drop \ell_x > R_c = 6.0 m)

Outputs:
- Figure 4 (2x3 grid): results/experiment_4/swarm_star_expansion_2x3_comparison.pdf/png
  and ECC_paper/figures/swarm_star_expansion_2x3_comparison.pdf/png
- Table: results/experiment_4/star_expansion_cases_table.tex

Can be run standalone:
    python3 experiments/exp4_expanding_safety.py --m-safe 25.0 --m-unsafe 0.0 --kappa 8.0 --rerun
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finite_time.core.stations import build_star_template, star_stations, template_geometry_metrics, backbone_geometry_metrics
from finite_time.core.spawn import verify_initial_conditions
from finite_time.core.simulation import simulate_star_maneuver
from finite_time.core.theory import compute_required_rate, effective_decay_rate
from finite_time.plotting import plot_transient_2x3_grid
from finite_time.utils import save_simulation_run, write_star_cases_table

# =============================================================================
# DEFAULT EXPANSION PARAMETERS
# =============================================================================
DEFAULT_SAFE_M = 25.0
DEFAULT_UNSAFE_M = 0.0
DEFAULT_KAPPA = 8.0
DEFAULT_A0 = 1.70
DEFAULT_A_INF = 3.02
DEFAULT_TRANSIENT_END = 2.5
DEFAULT_TOTAL_T_END = 12.0
DEFAULT_D_SAFE = 1.0
DEFAULT_R_C = 6.0
DEFAULT_OMEGA = 0.7853981633974483  # pi / 4
DEFAULT_SOLVER = dict(method='DOP853', rtol=1e-6, atol=1e-8, max_step=0.02)


def generate_expansion_initial_perturbation(
    q0: np.ndarray,
    seed: int = 123,
    r_min: float = 0.02,
    r_max: float = 0.05,
    outward_edge: Tuple[int, int] = (0, 1),
    outward_push: float = 0.58
) -> np.ndarray:
    """Generate initial perturbation for rapid expansion test.

    Applies small annular ring noise around stations plus an outward stretch
    along the specified backbone edge to challenge network connectivity under rapid expansion.
    """
    rng = np.random.default_rng(seed)
    n = len(q0)
    r = rng.uniform(r_min, r_max, size=n)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    delta0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    i, j = outward_edge
    diff_ij = q0[j] - q0[i]
    uij = diff_ij / np.linalg.norm(diff_ij)
    # Pull agents i and j apart along edge (i, j)
    delta0[i] -= outward_push * uij
    delta0[j] += outward_push * uij
    return delta0


def run_experiment_4(
    output_dir: Optional[Path] = None,
    m_safe: float = DEFAULT_SAFE_M,
    m_unsafe: float = DEFAULT_UNSAFE_M,
    kappa: float = DEFAULT_KAPPA,
    a0: float = DEFAULT_A0,
    a_inf: float = DEFAULT_A_INF,
    transient_end: float = DEFAULT_TRANSIENT_END,
    t_end: float = DEFAULT_TOTAL_T_END,
    d_safe: float = DEFAULT_D_SAFE,
    r_c: float = DEFAULT_R_C,
    rerun: bool = False
) -> List[Dict[str, Any]]:
    """Execute Experiment 4: simulate expansion, build 2x3 figure and LaTeX table."""
    if output_dir is None:
        output_dir = REPO_ROOT / 'results'
    folder = Path(output_dir) / 'experiment_4'
    folder.mkdir(parents=True, exist_ok=True)

    cases_config = [
        {'label': 'Safe', 'm': m_safe, 'title': f'Row 1 (Certified Safe & Connected, m = {m_safe})'},
        {'label': 'Unsafe', 'm': m_unsafe, 'title': f'Row 2 (Unsafe Transient Disconnection, m = {m_unsafe})'}
    ]

    p0 = build_star_template(10)
    dq0, _, _, _ = template_geometry_metrics(p0)
    ell_h0, lambda2_backbone = backbone_geometry_metrics(p0)

    q0, _ = star_stations(0.0, kappa, p0, a0=a0, a_inf=a_inf, omega=DEFAULT_OMEGA)
    delta0 = generate_expansion_initial_perturbation(q0, seed=123)

    min_dx0, max_ell0, is_valid0 = verify_initial_conditions(q0 + delta0, d_safe=d_safe, r_c=r_c)
    eta0 = delta0 - np.mean(delta0, axis=0)
    rho0 = float(np.sqrt(2.0) * np.linalg.norm(eta0))
    s0 = float(min(a0 * dq0 - d_safe, r_c - a0 * ell_h0))

    if not is_valid0 or rho0 >= s0:
        raise ValueError(f"Initial configuration invalid: min_dx0={min_dx0:.3f}, max_ell0={max_ell0:.3f}, rho0={rho0:.3f}, s0={s0:.3f}")

    print(f"Running Experiment 4: Rapid star expansion (a0 = {a0:.2f} -> a_inf = {a_inf:.2f}, kappa = {kappa:.1f}, transient t in [0, {transient_end:.1f}] s)...")
    print(f"Initial condition: rho0 = {rho0:.4f} m < s(0) = {s0:.4f} m (margin: +{s0 - rho0:.4f} m)")
    print("=" * 140)
    print(f"{'Case':<10} | {'m':>4} | {'kappa':>5} | {'gamma':>7} | {'gamma_req':>9} | {'min d_x':>7} | {'max ell':>7} | {'breach window':>22} | {'Outcome'}")
    print("-" * 140)

    sim_results = []
    case_rows_data = []
    rows = []

    for case in cases_config:
        lab = case['label']
        m_val = float(case['m'])
        name = f"star_expansion_{lab}_m_{m_val:g}_kappa_{kappa:g}"

        res = simulate_star_maneuver(
            m=m_val,
            kappa=kappa,
            p0=p0,
            delta0=delta0,
            t_span=(0.0, t_end),
            d_safe=d_safe,
            r_c=r_c,
            a0=a0,
            a_inf=a_inf,
            omega=DEFAULT_OMEGA,
            **DEFAULT_SOLVER
        )
        res.metrics['m'] = m_val
        res.metrics['parameter_case'] = lab

        row = save_simulation_run(folder, name, res)
        rows.append(row)
        sim_results.append(res.to_dict())

        # Determine breach window during transient (connectivity breach ell_x > r_c or collision d_x < d_safe)
        mask_tr = res.t <= transient_end
        tz = res.t[mask_tr]
        conn_viol = res.ell_x[mask_tr] > r_c
        safe_viol = res.d_x[mask_tr] < d_safe

        bw_str = "none"
        if np.any(conn_viol):
            t_cv = tz[conn_viol]
            bw_str = f"conn [{t_cv[0]:.2f},{t_cv[-1]:.2f}]s"
        elif np.any(safe_viol):
            t_sv = tz[safe_viol]
            bw_str = f"coll [{t_sv[0]:.2f},{t_sv[-1]:.2f}]s"

        if res.metrics['certified']:
            outcome = "Certified; Safe, Connected"
        elif not res.metrics['observed_connected']:
            outcome = "Uncertified; Connectivity Violation"
        elif not res.metrics['observed_safe']:
            outcome = "Uncertified; Safety Violation"
        else:
            outcome = "Uncertified; Safe"

        case_rows_data.append((
            lab, m_val, kappa,
            res.metrics['gamma'], res.metrics['gamma_req'],
            res.metrics['settling_times']['1e-06'],
            res.metrics['min_separation'], res.metrics['max_backbone_edge'],
            res.metrics['peak_control'], res.metrics['J_u'], outcome
        ))

        g_str = f"{res.metrics['gamma']:7.4f}"
        g_req_str = f"{res.metrics['gamma_req']:9.4f}" if res.metrics['gamma_req'] is not None else f"{'--':>9}"
        print(f"{case['label']:<10} | {m_val:4.1f} | {kappa:5.1f} | {g_str} | {g_req_str} | {res.metrics['min_separation']:7.4f} | "
              f"{res.metrics['max_backbone_edge']:7.4f} | {bw_str:>22} | {outcome}")

    print("=" * 140)

    # Generate Figure 4 (2x3 grid)
    plot_transient_2x3_grid(
        sim_results[0], sim_results[1], folder,
        extra_paths=[REPO_ROOT / 'ECC_paper' / 'figures'],
        transient_end=transient_end,
        d_safe=d_safe,
        r_c=r_c,
        ylim_d=(1.0, 5.0),
        ylim_ell=(4.0, 7.5),
        ylim_margin=(-0.05, 2.80),
        filename='swarm_star_expansion_2x3_comparison'
    )
    print(f"Saved Figure 4 to {folder / 'swarm_star_expansion_2x3_comparison.pdf'}")

    # Generate Table in results/experiment_4/
    write_star_cases_table(case_rows_data, folder / 'star_expansion_cases_table.tex')
    print(f"Saved Table to {folder / 'star_expansion_cases_table.tex'}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 4: Expanding Star Formation Transient Connectivity")
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'results')
    parser.add_argument('--m-safe', type=float, default=DEFAULT_SAFE_M)
    parser.add_argument('--m-unsafe', type=float, default=DEFAULT_UNSAFE_M)
    parser.add_argument('--kappa', type=float, default=DEFAULT_KAPPA)
    parser.add_argument('--a0', type=float, default=DEFAULT_A0)
    parser.add_argument('--a-inf', type=float, default=DEFAULT_A_INF)
    parser.add_argument('--transient-end', type=float, default=DEFAULT_TRANSIENT_END)
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()

    run_experiment_4(
        output_dir=args.output,
        m_safe=args.m_safe,
        m_unsafe=args.m_unsafe,
        kappa=args.kappa,
        a0=args.a0,
        a_inf=args.a_inf,
        transient_end=args.transient_end,
        rerun=args.rerun
    )


if __name__ == '__main__':
    main()
