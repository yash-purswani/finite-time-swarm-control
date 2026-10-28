#!/usr/bin/env python3
r"""Experiment 3: Certified Safe vs. Mid-Transient Collision ($2 \times 3$ Grid).

Simulates rapid star compression (\kappa = 4.0) over t in [0, 2.5] s, comparing:
- Row 1: Certified Safe Case (m = 2.5, \gamma > \gamma_req)
- Row 2: Unsafe Mid-Transient Collision (m = 0.0, \gamma < \gamma_req)

Outputs:
- Figure 3 (2x3 grid): results/experiment_3/swarm_star_2x3_comparison.pdf/png
- Table 2: results/star_cases_table.tex

Can be run standalone:
    python3 experiments/exp3_transient_safety.py --m-safe 25.0 --m-unsafe 0.0 --kappa 5.0 --rerun
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finite_time.core.simulation import simulate_star_maneuver
from finite_time.plotting import plot_transient_2x3_grid
from finite_time.utils import save_simulation_run, write_star_cases_table

# =============================================================================
# DEFAULT PARAMETERS (Easily modified for custom runs)
# =============================================================================
DEFAULT_SAFE_M = 25.0
DEFAULT_UNSAFE_M = 0.0
DEFAULT_KAPPA = 5.0
DEFAULT_TRANSIENT_END = 2.5
DEFAULT_TOTAL_T_END = 12.0
DEFAULT_D_SAFE = 1.0
DEFAULT_R_C = 6.0
DEFAULT_OMEGA = 0.7853981633974483  # pi / 4
DEFAULT_A0 = 1.70
DEFAULT_A_INF = 0.72
DEFAULT_SOLVER = dict(method='DOP853', rtol=1e-6, atol=1e-8, max_step=0.02)


def run_experiment_3(
    output_dir: Optional[Path] = None,
    m_safe: float = DEFAULT_SAFE_M,
    m_unsafe: float = DEFAULT_UNSAFE_M,
    kappa: float = DEFAULT_KAPPA,
    transient_end: float = DEFAULT_TRANSIENT_END,
    t_end: float = DEFAULT_TOTAL_T_END,
    d_safe: float = DEFAULT_D_SAFE,
    r_c: float = DEFAULT_R_C,
    rerun: bool = False
) -> List[Dict[str, Any]]:
    """Execute Experiment 3: simulate the two cases, build 2x3 figure and Table 2."""
    if output_dir is None:
        output_dir = REPO_ROOT / 'results'
    folder = Path(output_dir) / 'experiment_3'
    folder.mkdir(parents=True, exist_ok=True)

    cases_config = [
        {'label': 'Safe', 'm': m_safe, 'title': f'Row 1 (Certified Safe Case, m = {m_safe})'},
        {'label': 'Unsafe', 'm': m_unsafe, 'title': f'Row 2 (Unsafe Mid-Transient Collision, m = {m_unsafe})'}
    ]

    print(f"Running Experiment 3: Rapid star compression (kappa = {kappa:.1f}, transient t in [0, {transient_end:.1f}] s)...")
    print("=" * 135)
    print(f"{'Case':<10} | {'m':>4} | {'kappa':>5} | {'gamma':>7} | {'gamma_req':>9} | {'min d_x':>7} | {'max ell':>7} | {'collision window':>18} | {'Outcome'}")
    print("-" * 135)

    sim_results = []
    case_rows_data = []
    rows = []

    for case in cases_config:
        lab = case['label']
        m_val = float(case['m'])
        name = f"star_case_{lab}_m_{m_val:g}_kappa_{kappa:g}"

        res = simulate_star_maneuver(
            m=m_val,
            kappa=kappa,
            t_span=(0.0, t_end),
            d_safe=d_safe,
            r_c=r_c,
            a0=DEFAULT_A0,
            a_inf=DEFAULT_A_INF,
            omega=DEFAULT_OMEGA,
            **DEFAULT_SOLVER
        )
        res.metrics['m'] = m_val
        res.metrics['parameter_case'] = lab

        row = save_simulation_run(folder, name, res)
        rows.append(row)
        sim_results.append(res.to_dict())

        # Format outcome string
        if res.metrics['certified']:
            outcome = "Certified; Safe, Connected"
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
        cw = f"[{res.metrics['safe_viol_start']:.2f},{res.metrics['safe_viol_end']:.2f}]" if res.metrics['safe_viol_start'] is not None else "none"
        print(f"{case['label']:<10} | {m_val:4.1f} | {kappa:5.1f} | {g_str} | {g_req_str} | {res.metrics['min_separation']:7.4f} | "
              f"{res.metrics['max_backbone_edge']:7.4f} | {cw:>18} | {outcome}")

    print("=" * 135)

    # Generate Figure 3 (2x3 grid)
    plot_transient_2x3_grid(
        sim_results[0], sim_results[1], folder,
        extra_paths=[REPO_ROOT / 'ECC_paper' / 'figures'],
        transient_end=transient_end,
        d_safe=d_safe,
        r_c=r_c,
        ylim_d=(0.5, 2.2),
        ylim_ell=(1.0, 6.5),
        ylim_margin=(-0.05, 1.60),
        filename='swarm_star_2x3_comparison'
    )
    print(f"Saved Figure 3 to {folder / 'swarm_star_2x3_comparison.pdf'}")

    # Generate Table 2 in results/ and results/experiment_3/
    write_star_cases_table(case_rows_data, Path(output_dir) / 'star_cases_table.tex')
    write_star_cases_table(case_rows_data, folder / 'star_cases_table.tex')
    print(f"Saved Table 2 to {Path(output_dir) / 'star_cases_table.tex'}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 3: Certified Safe vs Unsafe Transient")
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'results')
    parser.add_argument('--m-safe', type=float, default=DEFAULT_SAFE_M)
    parser.add_argument('--m-unsafe', type=float, default=DEFAULT_UNSAFE_M)
    parser.add_argument('--kappa', type=float, default=DEFAULT_KAPPA)
    parser.add_argument('--transient-end', type=float, default=DEFAULT_TRANSIENT_END)
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()

    run_experiment_3(
        output_dir=args.output,
        m_safe=args.m_safe,
        m_unsafe=args.m_unsafe,
        kappa=args.kappa,
        transient_end=args.transient_end,
        rerun=args.rerun
    )


if __name__ == '__main__':
    main()
