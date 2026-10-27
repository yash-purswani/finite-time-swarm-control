"""Compatibility shim routing legacy finite_time.experiments imports to modular experiments/."""
from finite_time import run_shared, save_run, write_json
from experiments.exp1_morphing import run_experiment_1 as experiment1
from experiments.exp2_gain_evaluation import run_experiment_2 as experiment2
from experiments.exp3_transient_safety import run_experiment_3 as experiment3
from experiments.exp4_parameter_sweep import run_experiment_4 as experiment4

__all__ = [
    'run_shared',
    'save_run',
    'write_json',
    'experiment1',
    'experiment2',
    'experiment3',
    'experiment4',
]
