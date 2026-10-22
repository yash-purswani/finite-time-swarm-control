"""Audit raw adaptive nodes and refine observed parameter-map boundary cases."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import argparse
import json
import numpy as np
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from finite_time.common import shared_scenario
from finite_time.theory import R_bar,disagreement,classify
from finite_time.experiments import run_shared,save_run,write_json
from finite_time.sampling import station_history
from finite_time.cache import active_grid_paths,digest,signature,cache_valid,write_cache


def audit(path):
    path=Path(path);saved=json.loads(path.read_text())
    _,cfg,X0=shared_scenario(saved['metadata']['configuration'],saved['grid_m'],saved['grid_kappa'])
    with np.load(path.with_suffix('.npz')) as data:
        t=data['solver_nodes'];st=data['solver_states']
        nd=cfg.n*cfg.d
        X=(st if st.shape[-1]==nd else st[:,:nd]).reshape(-1,cfg.n,cfg.d)
        q=station_history(cfg,t)[0];delta=X-q
        dn=np.linalg.norm(delta.reshape(len(t),-1),axis=1)
        eta=disagreement(delta);en=np.linalg.norm(eta.reshape(len(t),-1),axis=1)
        integral=np.interp(t,data['t'],data['lambda2_integral'])
        eta0=disagreement(X0-cfg.stations(0))
        eb=np.linalg.norm(eta0)*np.exp(-t-cfg.lambda_min_M*integral)
        station_excess=float(np.max(dn-R_bar(t,X0-cfg.stations(0),cfg.beta)))
        eta_excess=float(np.max(en-eb))
        initial_difference=float(np.max(abs(X[0]-X0)))
    tol = max(2e-5, saved.get('solver', {}).get('rtol', 1e-6) * 20)
    return dict(case=path.stem,station_node_excess=station_excess,eta_node_excess=eta_excess,
                initial_state_difference=initial_difference,
                passed=bool(station_excess<tol and eta_excess<tol and initial_difference<1e-12))


def validate(root,workers=4,boundary=True,reuse=True):
    root=Path(root);paths=active_grid_paths(root)
    if not paths:raise ValueError('no saved parameter-map cases')
    if workers<1:raise ValueError('workers must be >=1')
    folder=root/'validation';folder.mkdir(parents=True,exist_ok=True)
    token=signature({},dict(boundary=boundary,auditor=digest(__file__),inputs={
        str(p.relative_to(root)): [digest(p),digest(p.with_suffix('.npz'))] for p in paths}))
    marker=folder/('audit_cache.json' if boundary else 'nodes_cache.json')
    if reuse and cache_valid(marker,token):
        print(f'Adaptive-node audit: reused matching checks for {len(paths)} cases.',flush=True)
        return
    with ProcessPoolExecutor(max_workers=min(workers,len(paths))) as pool:
        audits=list(pool.map(audit,paths))
    write_json(folder/'adaptive_node_audit.json',dict(cases=len(audits),checks=audits,
               maximum_station_node_excess=max(r['station_node_excess'] for r in audits),
               maximum_eta_node_excess=max(r['eta_node_excess'] for r in audits),
               all_passed=all(r['passed'] for r in audits),
               spectral_integral='linear interpolation of saved cumulative trapezoidal quadrature'))
    if not all(r['passed'] for r in audits):raise AssertionError('saved adaptive-node bound audit failed')
    comparisons=[]
    if boundary:
        saved_runs=[(path,json.loads(path.read_text())) for path in paths]
        for cls in ['B','C']:
            candidates=[(p,r) for p,r in saved_runs if r['metrics']['classification']==cls]
            if not candidates:continue
            path,saved=min(candidates,key=lambda pr:abs(pr[1]['metrics']['minimum_clearance']))
            cfg=saved['metadata']['configuration'];settings=saved['solver']
            tighter=dict(settings,rtol=settings['rtol']/10,atol=settings['atol']/10,max_step=settings['max_step']/2)
            refined_cfg=deepcopy(cfg)
            horizon=cfg['shared']['t_end']
            deadline=saved['certificate']['T_bar']
            cut=min(horizon,max(3.,deadline+.25))
            refined_cfg['shared'].update(t_end=cut,kappa=saved['grid_kappa'])
            res,ar,me,cert=run_shared(refined_cfg,saved['grid_m'],solver=tighter,spacing=.001)
            save_run(folder/'map_boundary',f'{cls}_{path.stem}',res,ar,me,refined_cfg,cert)
            refined_min,refined_max=me['min_separation'],me['max_backbone_edge']
            post_geometry=None
            if cut<horizon:
                # Universal station settling is graph independent, including for
                # uncertified/unsafe cases. After T_bar, exact errors vanish.
                # Keep this analytic extension separate from simulated states.
                station,scfg,_=shared_scenario(cfg,saved['grid_m'],saved['grid_kappa'])
                d0,ell0=station.geometry(scfg.H)
                s_min = float(min(station.scale(cut), station.scale(horizon)))
                s_max = float(max(station.scale(cut), station.scale(horizon)))
                lower = s_min * d0
                upper = s_max * ell0
                refined_min=min(refined_min,lower)
                refined_max=max(refined_max,upper)
                post_geometry=dict(interval=[cut,horizon],separation_minimum=lower,
                                   backbone_maximum=upper,
                                   justification='exact similarity geometry after the universal station deadline; full nominal nodes audited independently')
            refined_class=classify(cert,refined_min>cfg['shared']['d_safe'],refined_max<cfg['shared']['R_c'])
            difference=abs(refined_min-saved['metrics']['min_separation'])
            comparison=dict(case=path.stem,m=saved['grid_m'],kappa=saved['grid_kappa'],
                  nominal_classification=saved['metrics']['classification'],tighter_classification=refined_class,
                  nominal_clearance=saved['metrics']['minimum_clearance'],tighter_clearance=refined_min-cfg['shared']['d_safe'],
                  minimum_distance_difference=difference,tighter_solver=res.settings,
                  nominal_horizon=horizon,refined_horizon=cut,post_settling_geometry=post_geometry,
                  refined_transient_minimum=me['min_separation'],refined_full_interval_minimum=refined_min)
            comparisons.append(comparison)
            if refined_class!=cls or difference>1e-4:
                raise AssertionError(f'classification boundary sensitivity failed: {comparison}')
            print(f'Boundary {cls}, {path.stem}: min-distance difference {difference:.3g}',flush=True)
        write_json(folder/'map_boundary_summary.json',dict(comparisons=comparisons,all_classifications_stable=True))
    print(f'Adaptive-node audit passed for {len(audits)} cases.',flush=True)
    outputs=[folder/'adaptive_node_audit.json']
    if boundary:
        outputs.append(folder/'map_boundary_summary.json')
        outputs+=list((folder/'map_boundary').glob('*.json'))+list((folder/'map_boundary').glob('*.npz'))
    write_cache(marker,token,outputs)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--nodes-only',action='store_true')
    parser.add_argument('--rerun',action='store_true')
    args=parser.parse_args();validate(args.output,args.workers,not args.nodes_only,reuse=not args.rerun)
