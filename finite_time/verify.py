"""Automatic algebraic, analytical, numerical and solver-sensitivity checks."""
from pathlib import Path
import unittest
import json
import numpy as np
from .model import SwarmConfig,graph,laplacian,lambda2,validate_M,rhs,rhs_error,interaction_term,sig_beta,solver_jacobian
from .common import shared_scenario,CompressionStations,compression_field,reference_parameters
from .theory import projection,disagreement,R_bar,T_bar,rho_H,affine_exp_sup,compression_certificate,classify
from .integrate import simulate
from .experiments import run_shared,write_json,save_run,cached_shared

CONFIG_PATH=Path(__file__).resolve().parents[1]/'configs/ecc.json'


def audit_shared_nodes(root):
    """Check saved internal physical solver states, in bounded-memory batches."""
    from .cache import experiment_paths
    from .sampling import station_history
    records=[]
    for experiment in [2,3]:
        for path in experiment_paths(root,experiment):
            saved=json.loads(path.read_text())
            if path.stem.startswith('two_agent'):
                continue
            config=saved['metadata']['configuration'];m=saved['metadata']['M'][0][0]
            _,cfg,X0=shared_scenario(config,m)
            delta0=X0-cfg.stations(0);eta0=np.linalg.norm(disagreement(delta0))
            station_excess=eta_excess=0.
            with np.load(path.with_suffix('.npz')) as raw:
                nodes=raw['solver_nodes'];states=raw['solver_states'].reshape(-1,cfg.n,cfg.d)
                sample_t=raw['t'];integral=raw['lambda2_integral']
                for start in range(0,len(nodes),50000):
                    ts=nodes[start:start+50000]
                    delta=states[start:start+50000]-station_history(cfg,ts)[0]
                    norms=np.linalg.norm(delta.reshape(len(ts),-1),axis=1)
                    en=np.linalg.norm(disagreement(delta).reshape(len(ts),-1),axis=1)
                    bound=eta0*np.exp(-ts-cfg.lambda_min_M*np.interp(ts,sample_t,integral))
                    station_excess=max(station_excess,float(np.max(norms-R_bar(ts,delta0,cfg.beta))))
                    eta_excess=max(eta_excess,float(np.max(en-bound)))
                initial_difference=float(np.max(abs(states[0]-X0)))
            passed=station_excess<1e-8 and eta_excess<1e-8 and initial_difference<1e-12
            records.append(dict(case=str(path.relative_to(root)),nodes=len(nodes),
                                station_node_excess=station_excess,eta_node_excess=eta_excess,
                                initial_state_difference=initial_difference,passed=passed))
            if not passed:raise AssertionError(f'raw adaptive-node audit failed: {records[-1]}')
    write_json(Path(root)/'validation/star_adaptive_node_audit.json',dict(checks=records,
               all_passed=all(r['passed'] for r in records),cases=len(records),
               maximum_station_node_excess=max((r['station_node_excess'] for r in records),default=0.),
               maximum_eta_node_excess=max((r['eta_node_excess'] for r in records),default=0.),
               spectral_integral='interpolation of saved cumulative trapezoidal quadrature'))


class MathematicalTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads(CONFIG_PATH.read_text())
        self.station,self.cfg,self.X0=shared_scenario(self.config,self.config['experiment_3']['m_safe'])
        self.delta0=self.X0-self.cfg.stations(0)

    def test_laplacian_and_disconnected_spectrum(self):
        L,A=graph(self.X0,self.cfg)
        np.testing.assert_allclose(L,L.T,atol=1e-14)
        np.testing.assert_allclose(L@np.ones(self.cfg.n),0,atol=1e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(L)[0],-1e-12)
        self.assertEqual(lambda2(np.diag([0,0,1,2])),0)
        np.testing.assert_allclose(graph(self.X0,self.cfg.with_(R_c=.01))[0],0)
        # Explicit range semantics: reconnect immediately when below R_c.
        X=self.X0.copy();X[1]=X[0]+[self.cfg.R_c-.1,0]
        self.assertEqual(graph(X,self.cfg)[1][0,1],1)
        X[1]=X[0]+[self.cfg.R_c+.1,0]
        self.assertEqual(graph(X,self.cfg)[1][0,1],0)

    def test_psd_matrix_validation(self):
        validate_M(np.array([[2.,1.],[1.,2.]]),2)
        for M in [np.array([[1,2],[0,1]]),np.diag([1,-1]),np.eye(3)]:
            with self.assertRaises(ValueError):validate_M(M,2)

    def test_kronecker_and_physical_error_identity(self):
        cfg=self.cfg.with_(M=np.array([[2.,.3],[.3,1.]]))
        L,_=graph(self.X0,cfg)
        np.testing.assert_allclose(interaction_term(self.X0,self.delta0,cfg).ravel(),
                                   np.kron(L,cfg.M)@self.delta0.ravel(),atol=1e-13)
        np.testing.assert_allclose(rhs(0,self.X0.ravel(),cfg)-cfg.station_velocity(0).ravel(),
                                   rhs_error(0,self.delta0.ravel(),cfg),atol=1e-13)
        for coordinate,z in [('physical',self.X0),('error',self.delta0)]:
            accelerated=compression_field(0.,z.ravel(),self.station.p_initial,
                self.station.a_initial,self.station.a_final,self.station.kappa,self.station.omega,
                cfg.M,cfg.adjacency,cfg.beta,cfg.nu,cfg.eps,cfg.R_c,coordinate=='error',reference_parameters(cfg))
            original=rhs(0,z.ravel(),cfg) if coordinate=='physical' else rhs_error(0,z.ravel(),cfg)
            np.testing.assert_allclose(accelerated,original,atol=1e-12)

    def test_centering_and_analytic_velocity(self):
        for t in [0,.1,.7,2,10]:
            np.testing.assert_allclose(self.station.offsets(t).sum(axis=0),0,atol=1e-13)
            np.testing.assert_allclose(self.station.velocity(t).sum(axis=0),0,atol=1e-12)
            eps=1e-5
            numerical=(self.station.offsets(t+eps)-self.station.offsets(t-eps))/(2*eps)
            np.testing.assert_allclose(numerical,self.station.velocity(t),atol=1e-7,rtol=1e-8)
        # Changing only kappa must preserve initial stations and physical X0.
        _,cfg2,x2=shared_scenario(self.config,25,7)
        np.testing.assert_array_equal(cfg2.stations(0),self.cfg.stations(0))
        np.testing.assert_array_equal(x2,self.X0)

    def test_disagreement_and_distance_tubes(self):
        eta=disagreement(self.delta0)
        np.testing.assert_allclose(eta.sum(axis=0),0,atol=1e-13)
        np.testing.assert_allclose((projection(self.cfg.n)@self.delta0),eta,atol=1e-13)
        for i in range(self.cfg.n):
            for j in range(self.cfg.n):
                np.testing.assert_allclose(self.delta0[i]-self.delta0[j],eta[i]-eta[j],atol=1e-13)
                d=abs(np.linalg.norm(self.X0[i]-self.X0[j])-np.linalg.norm(self.cfg.stations(0)[i]-self.cfg.stations(0)[j]))
                self.assertLessEqual(d,np.sqrt(2)*np.linalg.norm(eta)+1e-12)

    def test_lyapunov_inequality(self):
        rng=np.random.default_rng(741) # finite deterministic algebraic tests, no Monte Carlo experiment
        for delta in [self.delta0,np.ones((10,2)),rng.normal(size=(10,2)),np.zeros((10,2))]:
            norm=np.linalg.norm(delta)
            dv=np.sum(delta*rhs_error(.2,delta.ravel(),self.cfg).reshape(10,2))
            self.assertLessEqual(dv,-norm**2-norm**1.5+1e-12)
        np.testing.assert_array_equal(sig_beta(np.array([-4,0,4]),.5),[-2,0,2])

    def test_global_supremum_and_design(self):
        cert=compression_certificate(self.station,self.cfg,self.delta0)
        self.assertTrue(cert['certified']);self.assertGreater(cert['continuous_tube_margin_lower_bound'],0)
        ts=np.geomspace(1e-8,cert['T_bar'],20000)
        d0,l0=self.station.geometry(self.cfg.H)
        s=np.minimum(d0*self.station.scale(ts)-self.cfg.d_safe,self.cfg.R_c-l0*self.station.scale(ts))
        sampled=np.max(np.log(cert['rho_0']/s)/ts)
        self.assertGreaterEqual(cert['gamma_req']+1e-10,sampled)
        self.assertLess(cert['gamma_req']-sampled,1e-5)
        # Test root-vs-endpoint and monotone-increasing margin branches.
        for B in [8.,-2.,0.]:
            req,tm=affine_exp_sup(10.,B,3.,2.,2.)
            ss=10+B*np.exp(-3*ts[ts<=2])
            vals=np.log(2/ss)/ts[ts<=2]
            self.assertGreaterEqual(req+1e-10,float(vals.max()))
        # Slow compression is also certified at zero gain; use the explicitly
        # separate faster star diagnostic to test the positive design boundary.
        k=self.config['experiment_3'].get('fast_diagnostic',{}).get('kappa',self.station.kappa)
        station=CompressionStations(self.station.p_initial,self.station.a_initial,self.station.a_final,k,self.station.omega)
        cert=compression_certificate(station,self.cfg,self.delta0)
        weak=compression_certificate(station,self.cfg.with_(M=np.zeros((2,2))),self.delta0)
        self.assertFalse(weak['certified'])
        self.assertAlmostEqual(cert['m_required'],(cert['gamma_req']-1)/(cert['w_lower']*cert['lambda2_H']))
        justbelow=self.cfg.with_(M=(cert['m_required']-1e-5)*np.eye(2))
        justabove=self.cfg.with_(M=(cert['m_required']+1e-5)*np.eye(2))
        self.assertFalse(compression_certificate(station,justbelow,self.delta0)['certified'])
        self.assertTrue(compression_certificate(station,justabove,self.delta0)['certified'])

    def test_zero_and_failed_initial_margin(self):
        common=np.tile([.1,.2],(10,1))
        cert=compression_certificate(self.station,self.cfg,common)
        self.assertTrue(cert['certified']);self.assertIsNone(cert['gamma_req'])
        cert=compression_certificate(self.station,self.cfg,np.zeros((10,2)))
        self.assertEqual(cert['T_bar'],0);self.assertTrue(cert['certified'])
        self.assertEqual(R_bar(1,np.zeros((10,2))),0)
        bad=self.delta0*2
        cert=compression_certificate(self.station,self.cfg,bad)
        self.assertFalse(cert['initial_margin']);self.assertIsNone(cert['gamma_req'])
        d0,_=self.station.geometry(self.cfg.H)
        infeasible=CompressionStations(self.station.p_initial,self.station.a_initial,.9*self.cfg.d_safe/d0,4.,.08)
        cert=compression_certificate(infeasible,self.cfg,self.delta0)
        self.assertFalse(cert['feasible']);self.assertIsNone(cert['gamma_req'])
        self.assertEqual(classify(cert,True,True),'D')
        static=CompressionStations(self.station.p_initial,self.station.a_initial,.9*self.cfg.d_safe/d0,0.,.08)
        self.assertTrue(compression_certificate(static,self.cfg,self.delta0)['certified'])

    def test_classification(self):
        feasible=dict(certified=False,feasible=True)
        self.assertEqual(classify(feasible,True,True),'B')
        self.assertEqual(classify(feasible,False,True),'C')
        self.assertEqual(classify(dict(certified=True,feasible=True),True,True),'A')
        self.assertEqual(classify(dict(certified=True,feasible=True),False,True),'VALIDATION_FAILURE')
        self.assertEqual(classify(dict(certified=False,feasible=False),True,True),'D')

    def test_slow_and_fast_star_share_initial_geometry(self):
        from copy import deepcopy
        if 'fast_diagnostic' not in self.config['experiment_3']:
            return
        slow,cfg,X0=shared_scenario(self.config,0.)
        fast,fcfg,fx=shared_scenario(self.config,0.,self.config['experiment_3']['fast_diagnostic']['kappa'])
        np.testing.assert_array_equal(X0,fx)
        np.testing.assert_array_equal(cfg.stations(0),fcfg.stations(0))
        self.assertTrue(compression_certificate(slow,cfg,X0-cfg.stations(0))['certified'])
        self.assertFalse(compression_certificate(fast,fcfg,fx-fcfg.stations(0))['certified'])
        self.assertLess(np.linalg.norm(self.station.p_initial.sum(axis=0)),1e-13)


def validate(config,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(MathematicalTests)
    from .test_workflow import WorkflowTests
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(WorkflowTests))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise AssertionError('mathematical unit tests failed')
    # Translation diagnostic: all agents share an error and coupling vanishes.
    gain=config['experiment_3']['m_safe']
    _,cfg,X0=shared_scenario(config,gain)
    common=np.tile([.4,.3],(cfg.n,1));X0=cfg.stations(0)+common
    np.testing.assert_allclose(interaction_term(X0,common,cfg),0,atol=1e-13)
    r1=simulate(cfg,X0,(0,.15),np.linspace(0,.15,151),**config['solver'])
    r0=simulate(cfg.with_(M=np.zeros((2,2))),X0,(0,.15),r1.t,**config['solver'])
    translation_difference=float(np.max(abs(r1.X-r0.X)))
    if translation_difference>2e-8:raise AssertionError('common translation depends on coupling')
    # Pure disagreement is active; physical and independent error solves agree.
    _,cfg,X0=shared_scenario(config,gain)
    delta=X0-cfg.stations(0)
    np.testing.assert_allclose(delta.sum(axis=0),0,atol=1e-13)
    interaction=np.linalg.norm(interaction_term(X0,delta,cfg))
    if interaction<=0:raise AssertionError('pure disagreement coupling is inactive')
    # Compare at tighter absolute accuracy: physical-state rtol scales with
    # positions up to 30, whereas error-state rtol scales with small residuals.
    # Matching solver input tolerances alone does not match those error budgets.
    comparison_settings=dict(config['solver'],rtol=1e-11,atol=1e-13,max_step=.005)
    physical=simulate(cfg,X0,(0,.2),np.linspace(0,.2,201),**comparison_settings)
    error=simulate(cfg,X0,(0,.2),physical.t,coordinates='error',**comparison_settings)
    equivalence=float(np.max(abs(physical.X-error.X)))
    if equivalence>2e-8:raise AssertionError('integrated physical/error implementations disagree')
    np.savez_compressed(folder/'diagnostics.npz',t=physical.t,physical_X=physical.X,error_X=error.X,
                        translation_t=r1.t,translation_m0_X=r0.X,translation_coupled_X=r1.X,
                        diagnostic_gain=gain)
    # High-gain implicit comparison before any error component settles. This
    # separate diagnostic adds common translation to avoid singular zero starts.
    Xdiag=X0+np.array([.2,.2])
    tdiag=np.linspace(0,.002,51)
    stiff=simulate(cfg,Xdiag,(0,.002),tdiag,method='Radau',rtol=1e-10,atol=1e-12,max_step=.0002)
    explicit=simulate(cfg,Xdiag,(0,.002),tdiag,**config['solver'])
    stiff_difference=float(np.max(abs(stiff.X-explicit.X)))
    if stiff_difference>1e-8:raise AssertionError('stiff solver diagnostic disagreement')
    # Full-horizon sensitivity for the approved slow maneuver and fast diagnostic.
    from copy import deepcopy
    from types import SimpleNamespace
    from .cache import cached_run
    import shutil
    sensitivities=[]
    scenarios=[('slow',config)]
    if 'fast_diagnostic' in config['experiment_3']:
        fast=deepcopy(config);dc=config['experiment_3']['fast_diagnostic']
        fast['shared'].update(kappa=dc['kappa'],t_end=dc['t_end'])
        scenarios.append(('fast',fast))
    for scenario,current in scenarios:
        for m in [current['experiment_3']['m_safe'],current['experiment_3']['m_counter']]:
            name=f'sensitivity_nominal_m_{m:g}' if scenario=='slow' else f'fast_sensitivity_nominal_m_{m:g}'
            if scenario=='slow':
                cached=cached_shared(folder,name,current,m)
            else:
                source_name=f'fast_m_{"safe" if m==current["experiment_3"]["m_safe"] else "counter"}_{m:g}'
                source=folder.parent/'experiment_3'
                cached=cached_run(source,source_name,current)
                if cached is not None:
                    for suffix in ['.json','.npz']:
                        shutil.copyfile(source/(source_name+suffix),folder/(name+suffix))
            if cached is None:
                nominal,ar,me,cert=run_shared(current,m)
                save_run(folder,name,nominal,ar,me,current,cert)
            else:
                ar,saved=cached;me=saved['metrics'];cert=saved['certificate']
                nominal=SimpleNamespace(X=ar['X'],settings=saved['solver'])
            # Refine the ACTUAL accepted settings, including automatic retries.
            tighter=dict(nominal.settings,rtol=nominal.settings['rtol']/10,
                         atol=nominal.settings['atol']/10,max_step=nominal.settings['max_step']/2)
            refinement_config=deepcopy(current)
            horizon=current['shared']['t_end'];cut=min(3.,horizon)
            post_window=None
            # Shorten only when the analytic post-settling geometry proves that
            # neither distance extremum can occur in the omitted [cut,horizon].
            # Every adaptive node of the FULL nominal run is still audited.
            station,_,_=shared_scenario(current,m)
            d0,ell0=station.geometry(cfg.H)
            lower_after=float(station.scale(horizon)*d0)
            upper_after=float(station.scale(cut)*ell0)
            if (cut<horizon and cert['certified'] and cut>cert['T_bar'] and
                    lower_after>me['min_separation']+1e-6 and
                    upper_after<me['max_backbone_edge']-1e-6):
                refinement_config['shared']['t_end']=cut
                post_window=dict(separation_lower_bound=lower_after,backbone_upper_bound=upper_after,
                                 justification='after T_bar the exact error is zero; monotone similarity geometry excludes later extrema')
            refined_name=f'sensitivity_tighter_m_{m:g}' if scenario=='slow' else f'fast_sensitivity_tighter_m_{m:g}'
            cached_refined=cached_run(folder,refined_name,refinement_config)
            if cached_refined is not None:
                ar2,saved_refined=cached_refined
                accepted=saved_refined['solver']
                if (accepted['method']!=tighter['method'] or
                        any(accepted[key]>tighter[key] for key in ['rtol','atol','max_step'])):
                    cached_refined=None
            if cached_refined is None:
                refined,ar2,me2,cert2=run_shared(refinement_config,m,solver=tighter,spacing=.001)
                # Persist before comparisons so an interrupted audit can resume.
                save_run(folder,refined_name,refined,ar2,me2,refinement_config,cert2)
            else:
                me2=saved_refined['metrics'];cert2=saved_refined['certificate']
                refined=SimpleNamespace(t=ar2['t'],X=ar2['X'],settings=accepted)
            minimum_difference=abs(me['min_separation']-me2['min_separation'])
            settle_differences={tol:abs(me['settling_times'][tol]-me2['settling_times'][tol]) for tol in me['settling_times']}
            compare=ar['t']<=refined.t[-1]+1e-12
            np.testing.assert_allclose(ar['t'][compare],refined.t,atol=1e-12,rtol=0)
            differences=nominal.X[compare]-refined.X
            coordinate_difference=float(np.max(abs(differences)))
            state_difference=float(np.linalg.norm(differences.reshape(len(refined.t),-1),axis=1).max())
            # solve_ivp tolerances control local errors, not global trajectory
            # differences. Compare global L2 disagreement at the SAME declared
            # 1e-8 absolute resolution used by the saved theorem-bound checks.
            # This does not alter any theorem inequality or its residual budget.
            state_budget=1e-8
            # A 1e-8 threshold can lie inside the nominal physical solver's error
            # budget. Detect and report this limitation rather than pretending that
            # its crossing time is resolved. Larger thresholds must be stable.
            resolved_1e8=settle_differences['1e-08']<=.002
            robust_differences=[settle_differences['0.0001'],settle_differences['1e-06']]
            if minimum_difference>1e-6 or state_difference>state_budget or max(robust_differences)>.002:
                raise AssertionError(f'solver sensitivity failure: {minimum_difference}, {settle_differences}, {state_difference}')
            if me['classification']!=me2['classification']:raise AssertionError('solver sensitivity changes classification')
            sensitivity=dict(m=m,scenario=scenario,min_separation_difference=minimum_difference,settling_time_differences=settle_differences,
                             maximum_state_difference=state_difference,physical_state_comparison_budget=state_budget,
                             maximum_coordinate_state_difference=coordinate_difference,
                             state_comparison_basis='global L2 difference at the predeclared 1e-8 numerical norm resolution; local solver tolerances are not global-error bounds',
                             nominal_settings=nominal.settings,tighter_settings=refined.settings,
                             nominal_horizon=horizon,refinement_horizon=float(refined.t[-1]),
                             post_refinement_geometric_bounds=post_window,
                             station_envelope_excess_nominal=me['station_envelope_excess'],
                             station_envelope_excess_tighter=me2['station_envelope_excess'],
                             nominal_settling_times=me['settling_times'],tighter_settling_times=me2['settling_times'],
                             threshold_1e8_resolved=resolved_1e8,
                             interpretation='1e-6 and 1e-4 validated; 1e-8 diagnostic only when solver-sensitive')
            sensitivities.append(sensitivity)
            print(f'  {scenario} sensitivity m={m:g}: minimum-distance change {minimum_difference:.3g}',flush=True)
    audit_shared_nodes(folder.parent)
    write_json(folder/'summary.json',dict(unit_tests_run=result.testsRun,unit_tests_passed=result.wasSuccessful(),
               common_translation_max_difference=translation_difference,pure_disagreement_interaction_norm=interaction,
               physical_error_max_difference=equivalence,radau_dop853_diagnostic_difference=stiff_difference,
               sensitivity=sensitivities,numerical_norm_allowance=1e-8,
               allowance_basis='absolute reported residual budget; audited against tighter tolerances, not a proof of exact arrival',
               sampled_settling='adaptive nodes and output samples, final crossing refined with dense interpolant'))
    return True


if __name__=='__main__':
    validate(json.loads(CONFIG_PATH.read_text()),CONFIG_PATH.parents[1]/'results/validation')
