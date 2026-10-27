"""Regenerate publication figures from saved numerical trajectories."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import matplotlib
matplotlib.use('Agg')
import argparse
import json
from pathlib import Path
import numpy as np

from finite_time.plotting import (
    plot_morphing_snapshots,
    plot_gain_evaluation,
    plot_transient_2x3_grid,
    plot_parameter_map_and_control,
)
from finite_time.cache import active_grid_paths, experiment_paths


def load_run(folder: Path, name: str):
    with np.load(folder / f"{name}.npz") as data:
        arrays = {key: data[key] for key in data.files if key not in ('solver_nodes', 'solver_states')}
    meta = json.loads((folder / f"{name}.json").read_text())
    return arrays, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    args = p.parse_args()
    root = args.output

    # Experiment 1
    f1 = root / 'experiment_1'
    if (f1 / 'morphing.json').exists() and (f1 / 'morphing.npz').exists():
        ar, meta = load_run(f1, 'morphing')
        plot_morphing_snapshots(ar, meta.get('configuration', {}), f1)
        print("Regenerated Figure 1.")

    # Experiment 2
    f2 = root / 'experiment_2'
    if f2.exists():
        runs = []
        for path in sorted(f2.glob('m_*.json')):
            if path.stem.startswith('m_'):
                ar, meta = load_run(f2, path.stem)
                m = meta['metadata']['M'][0][0]
                runs.append((m, ar, meta['metrics']))
        if runs:
            plot_gain_evaluation(sorted(runs, key=lambda r: r[0]), f2)
            print("Regenerated Figure 2.")

    # Experiment 3
    f3 = root / 'experiment_3'
    if f3.exists():
        safe_json = list(f3.glob('star_case_Safe_*.json'))
        unsafe_json = list(f3.glob('star_case_Unsafe_*.json'))
        if safe_json and unsafe_json:
            ar_safe, _ = load_run(f3, safe_json[0].stem)
            ar_unsafe, _ = load_run(f3, unsafe_json[0].stem)
            plot_transient_2x3_grid(ar_safe, ar_unsafe, f3)
            print("Regenerated Figure 3.")

    # Experiment 4
    f4 = root / 'experiment_4'
    if (f4 / 'grid.npz').exists():
        with np.load(f4 / 'grid.npz') as grid:
            rows = [json.loads(p.read_text())['grid_row'] for p in sorted((f4 / 'runs').glob('case_*.json'))]
            rows.sort(key=lambda r: r['index'])
            boundary = None
            if (f4 / 'design_boundary.npz').exists():
                with np.load(f4 / 'design_boundary.npz') as b_data:
                    boundary = (b_data['kappa'], b_data['m_required'])
            plot_parameter_map_and_control(
                grid['kappa'], grid['m'], grid['classification'], rows, f4, boundary=boundary
            )
            print("Regenerated Figure 4.")


if __name__ == '__main__':
    main()
