#!/usr/bin/env python3
"""Unified entry point for finite-time swarm tracking experiments.

Dispatches to dedicated experiment modules in experiments/:
- Experiment 1: Shape morphing demonstration (experiments/exp1_morphing.py)
- Experiment 2: Convergence & coupling gain sweep (experiments/exp2_gain_evaluation.py)
- Experiment 3: Certified Safe vs. Mid-Transient Collision (experiments/exp3_transient_safety.py)
- Experiment 4: Parameter sweep m vs. kappa (experiments/exp4_parameter_sweep.py)

Usage:
    python3 run_experiments.py --experiment all --workers 4
    python3 run_experiments.py --experiment 2
    python3 run_experiments.py --experiment 3
    python3 run_experiments.py --experiment 4 --workers 4
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[key] = '1'

import argparse
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

from experiments.exp1_morphing import run_experiment_1
from experiments.exp2_gain_evaluation import run_experiment_2
from experiments.exp3_transient_safety import run_experiment_3
from experiments.exp4_parameter_sweep import run_experiment_4


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', choices=['all', '1', '2', '3', '4', 'validate'], default='all',
                        help='Experiment to run (default: all)')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results',
                        help='Output directory for generated results')
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1),
                        help='Parallel worker count for grid evaluations (default: 4)')
    parser.add_argument('--rerun', action='store_true',
                        help='Force fresh simulations and validation without reusing cached data')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    print(f"=== Finite-Time Swarm Tracking Suite (Experiment: {args.experiment}) ===")

    # Run experiments
    if args.experiment in ['all', '1']:
        print("\n--- Running Experiment 1 ---")
        run_experiment_1(output_dir=args.output, rerun=args.rerun)

    if args.experiment in ['all', '2']:
        print("\n--- Running Experiment 2 ---")
        run_experiment_2(output_dir=args.output, rerun=args.rerun)

    if args.experiment in ['all', '3']:
        print("\n--- Running Experiment 3 ---")
        run_experiment_3(output_dir=args.output, rerun=args.rerun)

    if args.experiment in ['all', '4']:
        print(f"\n--- Running Experiment 4 ({args.workers} workers) ---")
        run_experiment_4(output_dir=args.output, workers=args.workers, rerun=args.rerun)

    # Validate saved results if grid exists
    if args.experiment in ['all', '4', 'validate'] and (args.output / 'experiment_4' / 'grid.npz').exists():
        print("\n--- Validating Saved Adaptive-Node Results ---")
        from validate_saved_results import validate as validate_saved
        validate_saved(args.output, workers=args.workers, reuse=not args.rerun)

    elapsed = time.time() - start_time
    print(f"\nAll requested tasks completed in {elapsed:.1f} s.")
    print(f"Results stored in: {args.output.resolve()}")


if __name__ == '__main__':
    main()
