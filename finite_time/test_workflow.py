"""Regression checks for accelerated trajectories, cache integrity and run selection."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .accelerate import morph_field
from .cache import active_grid_paths,cache_valid,write_cache
from .common import shared_scenario,reference,reference_velocity
from .experiments import summarize_outputs
from .formations import hexagon,wedge,star,arrow,morph_schedule
from .integrate import simulate
from .model import SwarmConfig,rhs,rhs_error
from .sampling import station_history


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).resolve().parents[1]/'configs/ecc.json').read_text())

    def test_compiled_morph_matches_both_controllers(self):
        P,Pdot=morph_schedule([hexagon(10,2),wedge(10,1.2),star(10,2.2),arrow(10,2)],
                             [2,4.5,7],1.5)
        cfg=SwarmConfig(10,2,np.array([[2,.3],[.3,1]]),reference,reference_velocity,P,Pdot,R_c=3)
        shapes,switches,duration=P.morph_data
        delta=np.linspace(-.6,.7,20).reshape(10,2)
        for t in [0,2,2.4,3.5,4.5,5.7,6,7,7.7,8.5,10]:
            for error in [False,True]:
                state=delta if error else delta+cfg.stations(t)
                compiled=morph_field(t,state.ravel(),shapes,switches,duration,cfg.M,
                    cfg.adjacency,cfg.beta,cfg.nu,cfg.eps,cfg.R_c,False,error)
                expected=(rhs_error if error else rhs)(t,state.ravel(),cfg)
                np.testing.assert_allclose(compiled,expected,rtol=1e-13,atol=1e-13)

    def test_batch_station_values_and_custom_reference(self):
        _,cfg,X0=shared_scenario(self.config,2)
        ts=np.array([0,.01,.3,1,3])
        q,qd=station_history(cfg,ts)
        np.testing.assert_array_equal(q,np.array([cfg.stations(t) for t in ts]))
        np.testing.assert_array_equal(qd,np.array([cfg.station_velocity(t) for t in ts]))
        shift=np.array([1.,-2.])
        custom=cfg.with_(r_fn=lambda t:cfg.r_fn(t)+shift)
        np.testing.assert_allclose(station_history(custom,ts)[0],q+shift,atol=1e-14)
        # A custom reference must not silently use the default compiled reference.
        a=simulate(cfg,X0,(0,.005),ts[:1].tolist()+[.005],**self.config['solver'])
        b=simulate(custom,X0+shift,(0,.005),a.t,**self.config['solver'])
        np.testing.assert_allclose(b.X,a.X+shift,atol=1e-8,rtol=0)

    def test_cache_detects_missing_and_modified_data(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);data=folder/'data.txt';marker=folder/'cache.json'
            data.write_text('original')
            write_cache(marker,'token',[data])
            self.assertTrue(cache_valid(marker,'token'))
            self.assertFalse(cache_valid(marker,'different'))
            data.write_text('changed');self.assertFalse(cache_valid(marker,'token'))
            data.unlink();self.assertFalse(cache_valid(marker,'token'))

    def test_custom_station_feedforward_is_not_overridden(self):
        from scipy.integrate import solve_ivp
        _,cfg,X0=shared_scenario(self.config,2)
        custom=cfg.with_(Pdot_fn=lambda t:cfg.Pdot_fn(t)+[.2,-.3])
        t=np.array([0,.003])
        np.testing.assert_allclose(station_history(custom,t)[1],
                                   np.array([custom.station_velocity(s) for s in t]),atol=1e-13)
        actual=simulate(custom,X0,(0,.003),t,**self.config['solver'])
        expected=solve_ivp(lambda s,z:rhs(s,z,custom),(0,.003),X0.ravel(),
                           t_eval=t,**self.config['solver']).y.T.reshape(-1,10,2)
        np.testing.assert_allclose(actual.X,expected,atol=1e-10,rtol=0)

    def test_active_grid_excludes_stale_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'experiment_4';runs=folder/'runs';runs.mkdir(parents=True)
            np.savez(folder/'grid.npz',m=[0,2],kappa=[1,3],classification=np.full((2,2),'A'))
            for i,m in enumerate([0,2]):
                for j,k in enumerate([1,3]):
                    (runs/f'case_{i*2+j:04d}.json').write_text(json.dumps(dict(grid_m=m,grid_kappa=k)))
            (runs/'case_0999.json').write_text('{}')
            self.assertEqual(len(active_grid_paths(root)),4)
            (runs/'case_0000.json').write_text(json.dumps(dict(grid_m=7,grid_kappa=1)))
            with self.assertRaises(ValueError):active_grid_paths(root)

    def test_standalone_outputs_do_not_enter_ecc_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'experiment_1';folder.mkdir()
            run=dict(metadata={'M':[[2,0],[0,2]]},certificate=None,
                     metrics={'settling_times':{'0.0001':1,'1e-06':1,'1e-08':1}})
            (folder/'morphing.json').write_text(json.dumps(run))
            (folder/'standalone_morphing.json').write_text(json.dumps({'metrics':{}}))
            self.assertEqual(len(summarize_outputs(root)),1)


if __name__=='__main__':unittest.main()
