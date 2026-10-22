"""Single entry point for ECC experiments; see finite_time/README.md."""
import os
# Prevent BLAS oversubscription in deterministic process-based grid evaluation.
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')  # Batch figures do not need interactive backend startup.
from finite_time.experiments import experiment1,experiment2,experiment3,experiment4,write_table,write_json,summarize_outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['all','1','2','3','4','validate'],default='all')
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/ecc.json')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--grid-size',type=int)
    parser.add_argument('--workers',type=int,default=min(3,os.cpu_count() or 1),
                        help='parallel grid workers (default <=3 for full-horizon dense-output memory)')
    parser.add_argument('--rerun',action='store_true',help='force fresh simulations and validation')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if args.workers<1 or (args.grid_size is not None and args.grid_size<2):
        parser.error('--workers must be >=1 and --grid-size >=2')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];start=time.time()
    if args.experiment in ['all','validate']:
        from finite_time.verify import validate
        from finite_time.cache import signature,cache_valid,write_cache,digest
        folder=args.output/'validation'
        token=signature(config,{'verify':digest(Path(__file__).parent/'finite_time/verify.py'),
                                'workflow_tests':digest(Path(__file__).parent/'finite_time/test_workflow.py')})
        marker=folder/'validation_cache.json'
        if not args.rerun and cache_valid(marker,token):
            print('Validation: reused matching checksummed results',flush=True)
        else:
            validate(config,folder)
            files=[folder/'summary.json',folder/'diagnostics.npz']
            files+=list(folder.glob('star_adaptive_node_audit.json'))
            files+=list(folder.glob('*sensitivity_*.json'))+list(folder.glob('*sensitivity_*.npz'))
            write_cache(marker,token,files)
    for key,fn in [('1',experiment1),('2',experiment2),('3',experiment3),('4',experiment4)]:
        if args.experiment not in ['all',key]:continue
        print(f'Experiment {key}',flush=True)
        if key=='4':rows.extend(fn(config,args.output,args.grid_size,args.workers,reuse=not args.rerun))
        else:rows.extend(fn(config,args.output,reuse=not args.rerun))
    summarize_outputs(args.output)
    if args.experiment in ['all','4'] and (args.output/'experiment_4/grid.npz').exists():
        from validate_saved_results import validate as validate_saved
        validate_saved(args.output,min(4,args.workers),reuse=not args.rerun)
    from finite_time.reporting import report
    report(args.output)
    write_json(args.output/'execution.json',dict(elapsed_seconds=time.time()-start,experiment=args.experiment,
               configuration=config,grid_size=args.grid_size,workers=args.workers))
    print(f'Completed in {time.time()-start:.1f} s; results: {args.output.resolve()}',flush=True)


if __name__=='__main__':main()
