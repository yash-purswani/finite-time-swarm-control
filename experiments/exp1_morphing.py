#!/usr/bin/env python3
"""Experiment 1: Arbitrary Reconfiguring Formation Demonstration.

Simulates smooth C^2 transitions across hexagon, wedge, star, and arrow templates
along a moving reference trajectory.

Can be run standalone:
    python3 experiments/exp1_morphing.py [--output DIR] [--rerun]
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from scipy.integrate import solve_ivp

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finite_time.core.dynamics import closed_loop_ode_rhs
from finite_time.core.stations import build_morphing_schedule
from finite_time.formations import hexagon, wedge, star, arrow
from finite_time.plotting import plot_morphing_snapshots
from finite_time.utils import write_json, save_simulation_run

# =============================================================================
# DEFAULT PARAMETERS (Easily modified for custom runs)
# =============================================================================
DEFAULT_N = 10
DEFAULT_RADIUS = 2.0
DEFAULT_M = 2.0
DEFAULT_T_END = 10.0
DEFAULT_SWITCH_TIMES = [2.0, 4.5, 7.0]
DEFAULT_MORPH_DURATION = 1.5
DEFAULT_SNAPSHOT_TIMES = [1.8, 4.0, 6.5, 9.0]
DEFAULT_INITIAL_SEED = 42
DEFAULT_SOLVER = dict(method='DOP853', rtol=1e-6, atol=1e-8, max_step=0.02)


def reference_path(t: float) -> np.ndarray:
    return np.array([0.9 * t, 1.1 * np.sin(0.35 * t)], dtype=float)


def reference_velocity(t: float) -> np.ndarray:
    return np.array([0.9, 0.385 * np.cos(0.35 * t)], dtype=float)


def run_experiment_1(
    output_dir: Optional[Path] = None,
    n: int = DEFAULT_N,
    radius: float = DEFAULT_RADIUS,
    m: float = DEFAULT_M,
    t_end: float = DEFAULT_T_END,
    switch_times: Optional[list] = None,
    morph_duration: float = DEFAULT_MORPH_DURATION,
    snapshot_times: Optional[list] = None,
    seed: int = DEFAULT_INITIAL_SEED,
    solver_settings: Optional[Dict[str, Any]] = None,
    rerun: bool = False
) -> Dict[str, Any]:
    """Execute Experiment 1 simulation and figure generation."""
    if output_dir is None:
        output_dir = REPO_ROOT / 'results'
    folder = Path(output_dir) / 'experiment_1'
    folder.mkdir(parents=True, exist_ok=True)

    if switch_times is None:
        switch_times = DEFAULT_SWITCH_TIMES
    if snapshot_times is None:
        snapshot_times = DEFAULT_SNAPSHOT_TIMES
    if solver_settings is None:
        solver_settings = DEFAULT_SOLVER

    config = {
        'n': n,
        'radius': radius,
        'm': m,
        't_end': t_end,
        'switch_times': switch_times,
        'morph_duration': morph_duration,
        'snapshot_times': snapshot_times,
        'initial_error_seed': seed,
    }

    # Template definitions
    shapes = [
        hexagon(n, radius),
        wedge(n, 0.6 * radius),
        star(n, 1.1 * radius),
        arrow(n, radius),
    ]
    P_fn, Pdot_fn = build_morphing_schedule(shapes, switch_times, morph_duration)

    def station_fn(t_val: float):
        r = reference_path(t_val)
        rdot = reference_velocity(t_val)
        q = r[None, :] + P_fn(t_val)
        qdot = rdot[None, :] + Pdot_fn(t_val)
        return q, qdot

    # Initial state with random Gaussian error
    rng = np.random.default_rng(seed)
    q0, _ = station_fn(0.0)
    delta0 = 0.6 * rng.standard_normal((n, 2))
    x0 = q0 + delta0
    y0 = np.concatenate([x0.ravel(), [0.0]])

    def rhs(t_val: float, y_val: np.ndarray):
        return closed_loop_ode_rhs(
            t_val, y_val, station_fn, m=m, r_c=3.0, beta=0.5, eps_w=0.2, n=n, dim=2
        )

    sol = solve_ivp(
        rhs, (0.0, t_end), y0,
        t_eval=np.linspace(0.0, t_end, 5001),
        dense_output=True,
        **solver_settings
    )
    if not sol.success:
        raise RuntimeError(f"Exp 1 integration failed: {sol.message}")

    t_eval = sol.t
    x_eval = sol.y[:n * 2].T.reshape(-1, n, 2)
    q_eval = np.array([station_fn(t)[0] for t in t_eval])

    arrays = {'t': t_eval, 'X': x_eval, 'q': q_eval}
    plot_morphing_snapshots(arrays, config, folder)

    summary = {
        'name': 'morphing',
        'n': n,
        'm': m,
        't_end': t_end,
        'min_separation': float(np.min(np.linalg.norm(x_eval[:, :, None, :] - x_eval[:, None, :, :], axis=-1)[:, np.triu_indices(n, 1)[0], np.triu_indices(n, 1)[1]])),
        'configuration': config,
    }
    write_json(folder / 'morphing.json', summary)
    print(f"Experiment 1 complete. Saved figure to {folder / 'formation_snapshots.pdf'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 1: Morphing Formations")
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'results')
    parser.add_argument('--m', type=float, default=DEFAULT_M)
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()

    run_experiment_1(output_dir=args.output, m=args.m, rerun=args.rerun)


if __name__ == '__main__':
    main()
