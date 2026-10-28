#!/usr/bin/env python3
"""Experiment 2: Finite-Time Convergence & Coupling Gain Evaluation.

Evaluates tracking errors, formation disagreement, and control effort
across coupling gains m in {0, 0.5, 2, 2.5, 5, 10} under rotating star formation (kappa = 2.0).

Outputs:
- Figure 2: results/experiment_2/error_disagreement_control.pdf/png
- Table 1: results/star_gain_table.tex

Can be run standalone:
    python3 experiments/exp2_gain_evaluation.py --m-values 0 0.5 2 2.5 5 10 --kappa 2.0 --rerun
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finite_time.core.simulation import simulate_star_maneuver
from finite_time.plotting import plot_gain_evaluation
from finite_time.utils import save_simulation_run, write_star_gain_table

# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================
DEFAULT_M_VALUES = [0.0, 0.5, 2.0, 5.0, 10.0]
DEFAULT_KAPPA = 2.0
DEFAULT_T_END = 12.0
DEFAULT_D_SAFE = 1.0
DEFAULT_R_C = 6.0
DEFAULT_OMEGA = np.pi / 4.0
DEFAULT_A0 = 1.70
DEFAULT_A_INF = 0.72
DEFAULT_SOLVER = dict(method='DOP853', rtol=1e-6, atol=1e-8, max_step=0.02)


def run_experiment_2(
    output_dir: Optional[Path] = None,
    m_values: Optional[List[float]] = None,
    kappa: float = DEFAULT_KAPPA,
    t_end: float = DEFAULT_T_END,
    d_safe: float = DEFAULT_D_SAFE,
    r_c: float = DEFAULT_R_C,
    omega: float = DEFAULT_OMEGA,
    rerun: bool = False
) -> List[dict]:
    """Execute Experiment 2 simulation, table, and figure generation."""
    if output_dir is None:
        output_dir = REPO_ROOT / 'results'
    folder = Path(output_dir) / 'experiment_2'
    folder.mkdir(parents=True, exist_ok=True)

    if m_values is None:
        m_values = DEFAULT_M_VALUES

    print(f"Running Experiment 2: Gain evaluation across m = {m_values} (kappa = {kappa:.1f})...")
    runs = []
    rows = []

    for m in m_values:
        name = f"m_{m:g}"
        json_path = folder / f"{name}.json"

        # Simulate
        res = simulate_star_maneuver(
            m=m,
            kappa=kappa,
            t_span=(0.0, t_end),
            d_safe=d_safe,
            r_c=r_c,
            a0=DEFAULT_A0,
            a_inf=DEFAULT_A_INF,
            omega=omega,
            **DEFAULT_SOLVER
        )
        res.metrics['m'] = m

        # Save artifacts
        row = save_simulation_run(folder, name, res)
        rows.append(row)
        runs.append((m, res.to_dict(), res.metrics))
        print(f"  m = {m:4.1f}: settling(1e-6) = {res.metrics['settling_times']['1e-06']:.4f} s, "
              f"peak ||u|| = {res.metrics['peak_control']:.4f}, Ju = {res.metrics['J_u']:.2f}")

    # Generate Figure 2
    plot_gain_evaluation(runs, folder)
    print(f"Saved Figure 2 to {folder / 'error_disagreement_control.pdf'}")

    # Generate Table 1 in results/ and results/experiment_2/
    write_star_gain_table(runs, Path(output_dir) / 'star_gain_table.tex')
    write_star_gain_table(runs, folder / 'star_gain_table.tex')
    print(f"Saved Table 1 to {Path(output_dir) / 'star_gain_table.tex'}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 2: Coupling Gain Evaluation")
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'results')
    parser.add_argument('--m-values', type=float, nargs='+', default=DEFAULT_M_VALUES)
    parser.add_argument('--kappa', type=float, default=DEFAULT_KAPPA)
    parser.add_argument('--t-end', type=float, default=DEFAULT_T_END)
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()

    run_experiment_2(
        output_dir=args.output,
        m_values=args.m_values,
        kappa=args.kappa,
        t_end=args.t_end,
        rerun=args.rerun
    )


if __name__ == '__main__':
    main()
