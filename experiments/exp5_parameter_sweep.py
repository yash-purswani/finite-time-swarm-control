#!/usr/bin/env python3
r"""Experiment 5: Parameter Sweep ($m$ vs. Compression Rate $\kappa$).

Sweeps a 4x4 parameter grid over:
- Coupling gain m in {0.0, 1.5, 3.0, 5.0}
- Maneuver compression rate \kappa in {2.0, 4.0, 6.0, 8.0}

Evaluates the sufficient design condition:
    1 + m * \underline{w} * \lambda_2(L_H) > \gamma_req(\kappa)
and overlays the continuous analytical boundary m_req(\kappa).

Outputs:
- Figure 5: results/experiment_5/parameter_map.pdf/png
            results/experiment_5/parameter_control.pdf/png
- Data: results/experiment_5/grid.csv, summary.json, report_metrics.tex

Can be run standalone:
    python3 experiments/exp5_parameter_sweep.py [--workers 4] [--rerun]
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finite_time.core.stations import build_star_template, template_geometry_metrics, backbone_geometry_metrics, star_stations
from finite_time.core.spawn import generate_random_ring_perturbation
from finite_time.core.theory import (
    universal_settling_deadline,
    compute_required_rate,
    compute_required_gain,
)
from finite_time.core.simulation import simulate_star_maneuver
from finite_time.plotting import plot_parameter_map_and_control
from finite_time.utils import save_simulation_run, write_json, write_report_metrics

# =============================================================================
# DEFAULT PARAMETERS (Easily modified for custom runs)
# =============================================================================
GRID_M_VALUES = [0.0, 1.5, 3.0, 5.0]
GRID_KAPPA_VALUES = [2.0, 4.0, 6.0, 8.0]
BOUNDARY_POINTS = 401
DEFAULT_WORKERS = 4
DEFAULT_T_END = 12.0
DEFAULT_D_SAFE = 1.0
DEFAULT_R_C = 6.0
DEFAULT_OMEGA = 0.7853981633974483
DEFAULT_A0 = 1.70
DEFAULT_A_INF = 0.72
DEFAULT_SOLVER = dict(method='DOP853', rtol=1e-6, atol=1e-8, max_step=0.02)


def _grid_worker(job: Tuple[int, float, float, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Worker task executing a single (m, kappa) simulation."""
    index, m, kappa, folder, solver_cfg = job
    res = simulate_star_maneuver(
        m=m,
        kappa=kappa,
        t_span=(0.0, DEFAULT_T_END),
        d_safe=DEFAULT_D_SAFE,
        r_c=DEFAULT_R_C,
        a0=DEFAULT_A0,
        a_inf=DEFAULT_A_INF,
        omega=DEFAULT_OMEGA,
        **solver_cfg
    )
    res.metrics['m'] = m
    name = f"case_{index:04d}"
    runs_folder = folder / 'runs'
    row = save_simulation_run(runs_folder, name, res, metadata_extra=dict(grid_m=m, grid_kappa=kappa))
    row.update(
        index=index,
        kappa=kappa,
        m=m,
        grid_m=m,
        grid_kappa=kappa,
        m_required=res.metrics['m_required']
    )
    # Update json to record grid attributes for validation caching
    case_json = runs_folder / f"{name}.json"
    import json
    doc = json.loads(case_json.read_text())
    doc.update(grid_m=m, grid_kappa=kappa, grid_row=row)
    write_json(case_json, doc)
    return row


def run_experiment_5(
    output_dir: Optional[Path] = None,
    ms: Optional[List[float]] = None,
    kappas: Optional[List[float]] = None,
    workers: int = DEFAULT_WORKERS,
    boundary_points: int = BOUNDARY_POINTS,
    rerun: bool = False
) -> List[Dict[str, Any]]:
    """Execute Experiment 5 parameter sweep and map generation."""
    if output_dir is None:
        output_dir = REPO_ROOT / 'results'
    folder = Path(output_dir) / 'experiment_5'
    runs_folder = folder / 'runs'
    folder.mkdir(parents=True, exist_ok=True)
    runs_folder.mkdir(parents=True, exist_ok=True)

    if ms is None:
        ms = GRID_M_VALUES
    if kappas is None:
        kappas = GRID_KAPPA_VALUES

    m_arr = np.array(ms, dtype=float)
    k_arr = np.array(kappas, dtype=float)
    size = len(k_arr)

    jobs = [
        (i * size + j, float(m), float(k), folder, DEFAULT_SOLVER)
        for i, m in enumerate(m_arr)
        for j, k in enumerate(k_arr)
    ]

    print(f"Running Experiment 4: Parameter sweep ({len(jobs)} configurations on {workers} workers)...")
    rows = []
    pending = []

    for job in jobs:
        idx, m, k, _, _ = job
        case_json = runs_folder / f"case_{idx:04d}.json"
        if not rerun and case_json.exists():
            import json
            saved = json.loads(case_json.read_text())
            if saved.get('grid_m') == m and saved.get('grid_kappa') == k:
                rows.append(saved['grid_row'])
                continue
        pending.append(job)

    if pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = {pool.submit(_grid_worker, job): job for job in pending}
            for fut in as_completed(futures):
                row = fut.result()
                rows.append(row)
                if len(rows) % 4 == 0 or len(rows) == len(jobs):
                    print(f"  parameter map: {len(rows)}/{len(jobs)} validated", flush=True)
    else:
        print(f"  parameter map: reused all {len(rows)} cached runs", flush=True)

    rows.sort(key=lambda r: r['index'])

    # Grid arrays
    classes = np.array([r['classification'] for r in rows]).reshape(len(m_arr), len(k_arr))
    min_seps = np.array([r['min_separation'] for r in rows]).reshape(len(m_arr), len(k_arr))
    peak_ctrl = np.array([r['peak_control'] for r in rows]).reshape(len(m_arr), len(k_arr))
    m_req_grid = np.array([np.nan if r['m_required'] is None else r['m_required'] for r in rows], dtype=float).reshape(len(m_arr), len(k_arr))

    np.savez_compressed(
        folder / 'grid.npz',
        kappa=k_arr,
        m=m_arr,
        classification=classes,
        min_separation=min_seps,
        peak_control=peak_ctrl,
        m_required=m_req_grid
    )

    # Write grid.csv
    csv_fields = [
        'name', 'm', 'kappa', 'settling_1e-4', 'settling_1e-6', 'settling_1e-8',
        'T_bar', 'min_separation', 'max_backbone_edge', 'certified',
        'observed_safe', 'observed_connected', 'classification', 'gamma',
        'gamma_req', 'm_required', 'peak_control', 'J_u', 'minimum_clearance'
    ]
    with (folder / 'grid.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            entry = dict(r)
            if 'settling_times' in r:
                entry['settling_1e-4'] = r['settling_times'].get('0.0001')
                entry['settling_1e-6'] = r['settling_times'].get('1e-06')
                entry['settling_1e-8'] = r['settling_times'].get('1e-08')
            writer.writerow(entry)

    # Evaluate continuous theoretical design boundary m_req(kappa)
    p0 = build_star_template(n=10)
    dq0, _, _, _ = template_geometry_metrics(p0)
    ell_h0, lambda2_lh = backbone_geometry_metrics(p0)   # 15-edge backbone metrics
    q0_b, _ = star_stations(0.0, kappa=k_arr[0], p0=p0, a0=DEFAULT_A0, a_inf=DEFAULT_A_INF, omega=DEFAULT_OMEGA)
    delta0, _, _ = generate_random_ring_perturbation(q0_b, d_safe=DEFAULT_D_SAFE, r_c=DEFAULT_R_C)
    delta0_norm = float(np.linalg.norm(delta0))
    eta0 = delta0 - np.mean(delta0, axis=0, keepdims=True)
    rho0 = np.sqrt(2.0) * float(np.linalg.norm(eta0))
    t_bar_univ = universal_settling_deadline(delta0_norm)

    boundary_k = np.linspace(k_arr[0], k_arr[-1], boundary_points)
    boundary_m = []
    for k_val in boundary_k:
        g_req = compute_required_rate(
            float(k_val), rho0, dq0, ell_h0=ell_h0, d_safe=DEFAULT_D_SAFE,
            r_c=DEFAULT_R_C, a0=DEFAULT_A0, a_inf=DEFAULT_A_INF, t_bar=t_bar_univ
        )
        m_val = compute_required_gain(g_req, r_c=DEFAULT_R_C, eps_w=0.05, lambda2_lh=lambda2_lh)
        boundary_m.append(np.nan if m_val is None else m_val)

    boundary_m_arr = np.asarray(boundary_m, dtype=float)
    np.savez_compressed(folder / 'design_boundary.npz', kappa=boundary_k, m_required=boundary_m_arr)

    # Plot Figure 5
    plot_parameter_map_and_control(
        k_arr, m_arr, classes, rows, folder, boundary=(boundary_k, boundary_m_arr)
    )
    print(f"Saved Figure 5 to {folder / 'parameter_map.pdf'} and {folder / 'parameter_control.pdf'}")

    # Summary JSON and LaTeX metrics
    counts = {str(k): int(v) for k, v in zip(*np.unique(classes, return_counts=True))}
    summary = {
        'size': size,
        'counts': counts,
        'validation_failures': counts.get('VALIDATION_FAILURE', 0),
        'min_separation': float(np.min(min_seps)),
        'min_peak': float(np.min(peak_ctrl)),
        'max_peak': float(np.max(peak_ctrl)),
    }
    write_json(folder / 'summary.json', summary)

    # Write report_metrics.tex
    report_dict = {
        'count_A': counts.get('A', 0),
        'count_B': counts.get('B', 0),
        'count_C': counts.get('C', 0),
        'count_D': counts.get('D', 0),
        'validation_failures': counts.get('VALIDATION_FAILURE', 0),
        'total_cases': len(rows),
        'grid_size': size,
        'min_separation': float(np.min(min_seps)),
        'min_peak': float(np.min(peak_ctrl)),
        'max_peak': float(np.max(peak_ctrl)),
        'disconnected_count': 0,
        'separation_count': counts.get('C', 0),
    }
    write_report_metrics(report_dict, folder / 'report_metrics.tex')
    write_report_metrics(report_dict, Path(output_dir) / 'report_metrics.tex')

    return rows


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 5: Parameter Sweep (m vs kappa)")
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'results')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()

    run_experiment_5(output_dir=args.output, workers=args.workers, rerun=args.rerun)


if __name__ == '__main__':
    main()
