#!/usr/bin/env python3
"""Unified entry point for finite-time swarm tracking experiments.

Dispatches to dedicated experiment modules in experiments/:
- Experiment 1: Shape morphing demonstration (experiments/exp1_morphing.py)
- Experiment 2: Convergence & coupling gain sweep (experiments/exp2_gain_evaluation.py)
- Experiment 3: Compression and separation comparison (experiments/exp3_transient_safety.py)
- Experiment 4: Expansion and backbone preservation (experiments/exp4_expanding_safety.py)
- Experiment 6: Corridor traversal via shape morphing (experiments/exp6_corridor.py)

Usage:
    python3 run_experiments.py --experiment all
    python3 run_experiments.py --experiment 1
    python3 run_experiments.py --experiment 2
    python3 run_experiments.py --experiment 3
    python3 run_experiments.py --experiment 4
    python3 run_experiments.py --experiment 6
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
from experiments.exp4_expanding_safety import run_experiment_4
from experiments.exp6_corridor import run_experiment_6


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', choices=['all', '1', '2', '3', '4', '6'], default='all',
                        help='Experiment to run (default: all)')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results',
                        help='Output directory for generated results')
    parser.add_argument('--rerun', action='store_true',
                        help='Request a fresh run; required to bypass the Experiment 6 summary cache')
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
        print("\n--- Running Experiment 4 ---")
        run_experiment_4(output_dir=args.output, rerun=args.rerun)

    if args.experiment in ['all', '6']:
        print("\n--- Running Experiment 6 ---")
        run_experiment_6(output_dir=args.output, rerun=args.rerun)

    elapsed = time.time() - start_time
    print(f"\nAll requested tasks completed in {elapsed:.1f} s.")
    print(f"Results stored in: {args.output.resolve()}")


if __name__ == '__main__':
    main()
