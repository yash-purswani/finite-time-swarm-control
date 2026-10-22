"""Regenerate publication figures from saved numerical trajectories."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import matplotlib
matplotlib.use('Agg')  # This command only writes files; never opens a GUI.
import argparse
import json
from pathlib import Path
import numpy as np
from finite_time.plotting import plot_morphing,plot_errors,plot_two_agent,plot_safety,plot_map,plot_star_maneuver
from finite_time.cache import active_grid_paths,experiment_paths


def load(folder,name):
    with np.load(folder/f'{name}.npz') as data:
        arrays={key:data[key] for key in data.files if key not in ('solver_nodes','solver_states')}
    meta=json.loads((folder/f'{name}.json').read_text())
    return arrays,meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    args=p.parse_args();root=args.output
    folder=root/'experiment_1'
    if (folder/'morphing.json').exists():
        ar,meta=load(folder,'morphing')
        if 'metadata' in meta:
            plot_morphing(ar,meta['metadata']['configuration']['experiment_1'],folder)
        else:
            print(f'Skipping {folder}/morphing: standalone-format metadata; '
                  'run run_experiments.py --experiment 1 to restore the ECC output.',flush=True)
    folder=root/'experiment_2'
    if folder.exists():
        runs=[]
        for path in experiment_paths(root,2):
            ar,meta=load(folder,path.stem)
            runs.append((meta['metadata']['M'][0][0],ar,meta['metrics']))
        if runs:
            plot_errors(sorted(runs,key=lambda r:r[0]),folder)
            plot_star_maneuver(max(runs,key=lambda r:r[0])[1],folder)
    folder=root/'experiment_3'
    if folder.exists():
        two=[];safety=[];fast=[]
        for path in experiment_paths(root,3):
            if not path.stem.startswith(('two_agent','m_','fast_m_')):continue
            ar,meta=load(folder,path.stem);m=meta['metadata']['M'][0][0]
            if path.stem.startswith('two_agent'):two.append((m,ar,meta['metrics']))
            elif path.stem.startswith('fast_m_'):
                fast.append((m,ar,meta['metrics'],meta['certificate']))
                fast_thresholds=(meta['metadata']['d_safe'],meta['metadata']['R_c'])
            elif path.stem.startswith('m_'):
                safety.append((m,ar,meta['metrics'],meta['certificate']))
                safety_thresholds=(meta['metadata']['d_safe'],meta['metadata']['R_c'])
        if two:plot_two_agent(sorted(two,key=lambda r:r[0]),folder)
        if safety:plot_safety(sorted(safety,key=lambda r:-r[0]),folder,*safety_thresholds)
        if fast:plot_safety(sorted(fast,key=lambda r:-r[0]),folder,*fast_thresholds,
                            name='fast_star_safety_connectivity',graph_name='fast_star_graph_history')
    folder=root/'experiment_4'
    if (folder/'grid.npz').exists():
        with np.load(folder/'grid.npz') as grid:
            rows=[json.loads(path.read_text())['grid_row'] for path in active_grid_paths(root)]
            rows.sort(key=lambda r:r['index'])
            curve=folder/'design_boundary.npz'
            boundary=None
            if curve.exists():
                with np.load(curve) as data:boundary=(data['kappa'],data['m_required'])
            plot_map(grid['kappa'],grid['m'],grid['classification'],rows,folder,boundary=boundary)
    from finite_time.experiments import summarize_outputs
    summarize_outputs(root)
    from finite_time.reporting import report
    report(root)


if __name__=='__main__':main()
