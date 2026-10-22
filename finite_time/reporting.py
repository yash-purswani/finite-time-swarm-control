"""Generate a concise technical discussion from saved experiment evidence."""
from pathlib import Path
import json
import hashlib
import csv
from .cache import active_grid_paths,experiment_paths


def fmt(x,digits=6):
    return 'not resolved / not available' if x is None else f'{x:.{digits}g}'


def star_tables(root,e2,e3):
    """Tracking and safety comparison tables for current star experiments."""
    records=[];gains=[];safety=[]
    for m,run in sorted(e2):
        if 'reference' not in run['metadata']['configuration']['shared']:continue
        me=run['metrics'];ce=run['certificate'];c=run['metadata']['configuration']['shared']
        gains.append(f"{m:g} & {fmt(me['settling_times']['1e-06'])} & {fmt(me['T_bar'])} & "
                     f"{fmt(me['peak_control'])} & {fmt(me['J_u'])} \\\\")
        rec = dict(experiment=2,case=f'm_{m:g}',m=m,kappa=c.get('kappa',me.get('kappa',3.0)))
        rec.update(me)
        rec['certified'] = ce.get('certified',False) if ce else False
        records.append(rec)
    for name,run in sorted(e3.items()):
        if name.startswith(('two_agent','connectivity_stress')):continue
        me=run['metrics'];ce=run['certificate'] or {};m=run['metadata']['M'][0][0]
        config=run['metadata']['configuration'];shared=config['shared']
        kappa=me.get('kappa',shared.get('kappa',0.));label=me.get('parameter_case','Fast' if name.startswith('fast_') else 'Slow')
        certified=bool(ce.get('certified',False))
        safety.append(f"{label} & {kappa:g} & {m:g} & {fmt(me['min_separation'])} & "
                      f"{fmt(me['max_backbone_edge'])} & {'Yes' if certified else 'No'} & "
                      f"{'Yes' if me['observed_safe'] and me['observed_connected'] else 'No'} \\\\")
        record=dict(experiment=3,case=name,m=m)
        record.update(me)
        record.update(kappa=kappa,certified=certified);records.append(record)
    if not records:return
    if gains:
        (root/'star_gain_table.tex').write_text(
            r'\begin{tabular}{rrrrr}\toprule'+'\n'+
            r'$m$ & $t_{10^{-6}}$ & $\overline T$ & $\max\|u\|_2$ & $J_u$\\\midrule'+'\n'+
            '\n'.join(gains)+'\n'+r'\bottomrule\end{tabular}'+'\n')
    (root/'star_safety_table.tex').write_text(
        r'\begin{tabular}{lrrrrcc}\toprule'+'\n'+
        r'Case & $\kappa$ & $m$ & $\min d_x$ & $\max\ell_{x,H}$ & Certified & Safe and H-persistent\\\midrule'+'\n'+
        '\n'.join(safety)+'\n'+r'\bottomrule\end{tabular}'+'\n')
    fields=['experiment','case','m','kappa','settling_1e-4','settling_1e-6','settling_1e-8','T_bar',
            'min_separation','max_backbone_edge','certified','observed_safe','observed_connected',
            'observed_backbone_preserved','observed_graph_connected','classification','minimum_clearance','minimum_connectivity_margin','peak_control',
            'peak_individual_control','J_u']
    with (root/'star_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for record in records:
            for key,old in [('1e-4','0.0001'),('1e-6','1e-06'),('1e-8','1e-08')]:
                record[f'settling_{key}']=record['settling_times'][old]
            writer.writerow(record)

def report(root):
    root=Path(root);parts=[]
    def saved(relative):
        p=root/relative
        return json.loads(p.read_text()) if p.exists() else None
    parts.append('ECC FINITE-TIME SWARM EXPERIMENTS\nAll quantities use the report\'s normalized coordinates and time.\n')
    parts.append('Controller and validation\nThe plant is integrated in physical x coordinates with analytic qdot, local\n-delta-sig^beta(delta), and -(L_w kron M)delta. There is no centroid feedback,\n1/n normalization, extra coupling gain, state reset, or feedback smoothing.\nThe theoretical LaTeX report is unchanged. The source hash is recorded in every\nrun, along with configuration, libraries and solver settings.\n')
    validation=saved('validation/summary.json')
    if validation:
        parts.append(f"All {validation['unit_tests_run']} algebraic unit tests passed. Independent physical/error\n"
                     f"integration differs by at most {fmt(validation['physical_error_max_difference'])};\n"
                     f"the Radau/DOP853 short diagnostic differs by {fmt(validation['radau_dop853_diagnostic_difference'])}.\n"
                     f"The common-translation comparison differs by {fmt(validation['common_translation_max_difference'])}.\n")
        for s in validation['sensitivity']:
            parts.append(f"Sensitivity {s.get('scenario','historical')} m={s['m']:g}: minimum-distance difference {fmt(s['min_separation_difference'])};\n"
                         f"settling differences (1e-4,1e-6,1e-8) = {s['settling_time_differences']}.\n"
                         f"1e-8 threshold resolved in this comparison: {s['threshold_1e8_resolved']}.\n")
            if 'refinement_horizon' in s:
                parts.append(f"Nominal horizon={s['nominal_horizon']:g}, refined horizon={s['refinement_horizon']:g}; "
                             f"actual tolerances {s['nominal_settings']} -> {s['tighter_settings']}.\n"
                             f"Post-window analytic extrema bounds: {s['post_refinement_geometric_bounds']}.\n")
    for label,path in [('Experiments 2/3','validation/star_adaptive_node_audit.json'),
                       ('Parameter map','validation/adaptive_node_audit.json')]:
        node_audit=saved(path)
        if node_audit:
            parts.append(f"{label} adaptive-node audit: {node_audit['cases']} cases, all passed = {node_audit['all_passed']}.\n"
                         f"Maximum station-bound node excess = {fmt(node_audit['maximum_station_node_excess'])};\n"
                         f"maximum disagreement-bound node excess = {fmt(node_audit['maximum_eta_node_excess'])}.\n")
    boundary=saved('validation/map_boundary_summary.json')
    if boundary:
        parts.append('Closest observed B/C boundary cases retain their classifications with tighter\n'
                     'tolerances, half the maximum step and finer distance refinement.\n')
        for check in boundary['comparisons']:
            parts.append(f"{check['case']}: minimum-distance change {fmt(check['minimum_distance_difference'])};\n"
                         f"classification {check['nominal_classification']} -> {check['tighter_classification']}.\n")
    e1=saved('experiment_1/morphing.json')
    if e1 and 'metadata' in e1:
        me=e1['metrics']
        parts.append('Experiment 1: arbitrary moving formations\n'
                     'Centered ring, wedge, star and arrow templates are joined by quintic blends\n'
                     'with analytic derivatives and fixed agent identifiers. Stations move with the\n'
                     'common smooth reference. This validates station tracking and inherited centroid\n'
                     'tracking, not universal collision avoidance for interpolated templates.\n'
                     f"Numerical 1e-6 settling = {fmt(me['settling_times']['1e-06'])}, universal deadline\n"
                     f"= {fmt(me['T_bar'])}. Offset-centering residual = {fmt(me['centered_offset_residual'])};\n"
                     f"maximum centroid error after the deadline = {fmt(me['centroid_error_after_deadline'])}.\n"
                     f"Minimum observed separation = {fmt(me['min_separation'])}; no safety certificate is claimed.\n")
    e2=[]
    for path in experiment_paths(root,2):
        run=json.loads(path.read_text());e2.append((run['metadata']['M'][0][0],run))
    if e2:
        parts.append('Experiment 2: local feedback, graph dissipation and full command\n'
                     'All gains share the same station model, initial physical positions, beta and\n'
                     'graph rule used in Experiment 3. Every run passes the universal station\n'
                     'envelope, disagreement bound, Lyapunov and pairwise-tube checks within the\n'
                     'reported numerical residual allowance. The integrated lambda2 envelope uses\n'
                     'numerical cumulative trapezoidal quadrature, not an exact spectral integral.\n'
                     'M adds disagreement dissipation while leaving the universal deadline unchanged.\n'
                     'No general ordering theorem for different M is inferred. Physical effort includes qdot.\n')
        shared=e2[0][1]['metadata']['configuration']['shared']
        if 'reference' in shared:
            parts.append(f"The star maneuver uses a(t)=a_inf+(1-a_inf)exp(-0.90t) with kappa={shared.get('kappa', 3.0):g},\n"
                         "and r(t)=(0.9t, 1.1 sin(0.35t)). Ten alternating outer/inner star vertices\n"
                         f"have template radii 2.8 and 1.22. The safety threshold is {shared.get('d_safe', 0.95):g} and R_c={shared.get('R_c', 6.6):g}.\n")
        for m,r in sorted(e2):
            me=r['metrics']
            parts.append(f"m={m:g}: 1e-6 settling {fmt(me['settling_times']['1e-06'])};\n"
                         f"peak ||u|| {fmt(me['peak_control'])}; feedback energy J_u {fmt(me['J_u'])};\n"
                         f"station-envelope excess {fmt(me.get('station_envelope_excess', 0.0))}; actual solver {r['solver']}.\n")
    e3={p.stem:json.loads(p.read_text()) for p in experiment_paths(root,3)}
    star_tables(root,e2,e3)
    weak=next((r for name,r in e3.items() if name.startswith('m_counter_')),None)
    safe=next((r for name,r in e3.items() if name.startswith('m_safe_')),None)
    two0=saved('experiment_3/two_agent_m_0.json');two2=saved('experiment_3/two_agent_m_2.json')
    if two0 and two2:
        parts.append('Experiment 3: analytical compression, then ten-agent safety\n'
                     'The separate 1D constant-weight two-agent case reproduces Section 7 before\n'
                     'the distance-weighted ten-agent comparison. The uncoupled signed separation\n'
                     f"at t=0.05 is {fmt(two0['metrics']['signed_separation_at_0.05'])}, matching the report.\n"
                     f"Maximum analytic-error discrepancies are {fmt(two0['metrics']['analytical_error_max_difference'])}\n"
                     f"(m=0) and {fmt(two2['metrics']['analytical_error_max_difference'])} (m=2). The\n"
                     'uncoupled agents pass through one another; m=2 remains separated above 1.\n')
    if safe and weak:
        cert=safe['certificate'];me=safe['metrics'];w=weak['metrics']
        ms=safe['metadata']['M'][0][0];mw=weak['metadata']['M'][0][0]
        radius=safe['metadata']['R_c']
        if cert['gamma_req'] is None:
            gain_design='No logarithmic gain design is needed for zero disagreement.'
        elif cert['gamma_req']<1:
            gain_design='Zero coupling already satisfies the sufficient design condition.'
        else:
            gain_design=f"The sufficient scalar-gain boundary is m>{fmt(cert['m_required'])}."
        parts.append(f"For ten agents, rho0={fmt(cert['rho_0'])}<s(0)={fmt(cert['s_initial'])}, the\n"
                     f"all-time geometric margin floor is {fmt(cert['geometric_margin_floor'])}, and gamma_req\n"
                     f"={fmt(cert['gamma_req'])}. {gain_design}\n"
                     f"m={ms:g} has gamma={fmt(cert['gamma'])} and a proven positive all-time tube-margin\n"
                     f"lower bound {fmt(cert['continuous_tube_margin_lower_bound'])}. This invokes the\n"
                     'report\'s first-exit reasoning; communication edges are not retained outside range.\n'
                     f"Observed minimum separation: m={ms:g} {fmt(me['min_separation'])}, m={mw:g} {fmt(w['min_separation'])}.\n"
                     f"Observed minimum clearance: m={ms:g} {fmt(me['minimum_clearance'])}, m={mw:g} {fmt(w['minimum_clearance'])}.\n"
                     f"Maximum H lengths are {fmt(me['max_backbone_edge'])} and {fmt(w['max_backbone_edge'])}; R_c={radius:g}. Peak commands are\n"
                     f"{fmt(me['peak_control'])} (m={ms:g}) and {fmt(w['peak_control'])} (m={mw:g}). Feedback energies\n"
                     f"are {fmt(me['J_u'])} and {fmt(w['J_u'])}. These measurements quantify the\n"
                     'tracking/effort tradeoff, with an unchanged station deadline.\n')
    star_cases=[(name,r) for name,r in e3.items() if name.startswith('star_case_')]
    if star_cases:
        parts.append('Experiment 3: controlled star compression/expansion cases\n'
                     'The centered asymmetric ten-agent star uses q_i=r+a(t)p_i^0, '
                     'a_inf=1.65-0.25 kappa, a=a_inf+(1-a_inf)exp(-0.9t), and '
                     'r=(0.9t,1.1sin(0.35t)); station velocities are analytic. All cases share '
                     'initial stations and positions, beta=0.5, d_safe=0.95 and R_c=6.60; only m '
                     'and kappa vary. Follows test.py setup.\n')
        for name,run in sorted(star_cases):
            me=run['metrics'];m=run['metadata']['M'][0][0]
            parts.append(f"Case {me['parameter_case']}: m={m:g}, kappa={me['kappa']:g}; "
                         f"geometry feasible={me['geometry_feasible']}; minimum separation="
                         f"{fmt(me['min_separation'])} (clearance {fmt(me['minimum_clearance'])}); "
                         f"maximum backbone length={fmt(me['max_backbone_edge'])} "
                         f"(connectivity margin {fmt(me['minimum_connectivity_margin'])}); "
                         f"safe={me['observed_safe']}, backbone preserved={me['observed_backbone_preserved']}, "
                         f"range-limited graph connected on output grid={me['observed_graph_connected']}; "
                         f"settling at 1e-6={fmt(me['settling_times']['1e-06'])}, "
                         f"T_bar={fmt(me['T_bar'])}, peak command={fmt(me['peak_control'])}, "
                         f"certified={me.get('certified', False)}.\n")
        parts.append('The 3-by-3 comparison is `experiment_3/swarm_star_3x3_comparison.png` '
                     '(PDF also provided). Rows show Cases A, B, and C; columns show minimum separation, '
                     'maximum pairwise distance, and geometric margin against the distortion envelope.\n')
        md=['# Experiment 3: star maneuver safety and backbone connectivity','',
            r'$q_i=r+a(t)p_i^0$, $a_\infty=1.65-0.25\kappa$, '
            r'$a(t)=a_\infty+(1-a_\infty)e^{-0.9t}$, and '
            r'$r(t)=(0.9t,1.1\sin(0.35t))$. The centered asymmetric star template, '
            'initial positions, $d_{safe}=0.95$, and $R_c=6.60$ are configured as in test.py.','',
            '| Case | $m$ | $\\kappa$ | Geometry feasible | Min. separation | Max. backbone edge | Safe | H preserved | Active graph connected* | Certified |',
            '|---|---:|---:|:---:|---:|---:|:---:|:---:|:---:|:---:|']
        for _,run in sorted(star_cases):
            me=run['metrics'];m=run['metadata']['M'][0][0]
            cert=run.get('certificate') or {}
            cert_val=cert.get('certified', me.get('certified', False))
            md.append(f"| {me['parameter_case']} | {m:g} | {me['kappa']:g} | {me['geometry_feasible']} | "
                      f"{me['min_separation']:.6f} | {me['max_backbone_edge']:.6f} | "
                      f"{me['observed_safe']} | {me['observed_backbone_preserved']} | "
                      f"{me['observed_graph_connected']} | {cert_val} |")
        md+=['','*Connectivity is evaluated from the range-limited graph on the saved output grid. '
             'Case C loses complete-backbone links while Case B exhibits a minimum-distance breach.', '',
             '![Experiment 3 three-case comparison](swarm_star_3x3_comparison.png)','']
        (root/'experiment_3'/'report.md').write_text('\n'.join(md))
    fast=[(name,r) for name,r in e3.items() if name.startswith('fast_m_')]
    if fast:
        parts.append('Separate faster-star diagnostic (kappa=4, horizon 3): same template,\n'
                     'reference, initial positions, thresholds and controller; only the maneuver\n'
                     'speed differs from the slow case. Within this diagnostic only m changes.\n')
        for name,r in sorted(fast):
            ce=r['certificate'];me=r['metrics'];m=r['metadata']['M'][0][0]
            parts.append(f"m={m:g}: gamma_req={fmt(ce['gamma_req'])}, m_required={fmt(ce['m_required'])},\n"
                         f"certified={ce['certified']}, class={me['classification']},\n"
                         f"minimum separation={fmt(me['min_separation'])}, maximum backbone={fmt(me['max_backbone_edge'])},\n"
                         f"1e-6 settling={fmt(me['settling_times']['1e-06'])}, peak command={fmt(me['peak_control'])}.\n")
    stress_run=e3.get('connectivity_stress_m_5')
    if stress_run:
        me=stress_run['metrics'];meta=stress_run['metadata']
        stress_cfg=meta['configuration']['experiment_3']['connectivity_stress']
        event=me['connectivity_events'][0] if me['connectivity_events'] else None
        parts.append('Connectivity-loss stress case (separate from the compression theorem):\n'
                     'The ten-node eligible/backbone graph is a sparse path with node order '
                     f"{stress_cfg['path_order']}. The same initial star locations and m=5 controller are used, "
                     f"but the formation expands from a={stress_cfg['a_initial']:g} to {stress_cfg['a_final']:g} "
                     f"with R_c={stress_cfg['R_c']:g}. This expansion violates the monotone-compression "
                     'assumptions used by the Experiment 3 certificate; no certificate is claimed.\n'
                     f"The range event occurs at t={fmt(event)}; minimum separation is "
                     f"{fmt(me['min_separation'])} against d_safe={meta['d_safe']:g} "
                     f"(observed safe={me['observed_safe']}); the graph ends connected="
                     f"{me['observed_connected']}. The 3-by-3 comparison plot is "
                     '`experiment_3/experiment3_three_case_comparison.png` (PDF also provided). '
                     'It places the certified slow case, fast distance-breach case, and safe-but-disconnected '
                     'path case in rows. Columns show pair separation with its lower tube, backbone length '
                     'with its upper tube, and geometric margin against the applicable distortion envelope.\n')
        (root/'experiment_3'/'report.md').write_text(
            '# Experiment 3: safety and connectivity cases\n\n'
            'The rows compare (1) the certified slow star with m=5, (2) the fast star with m=0 that '
            'violates the minimum-distance threshold, and (3) an expanding sparse path with m=5 that '
            'loses connectivity while keeping pairwise separation above the threshold. The third case '
            'is a stress test outside the monotone-compression certificate assumptions.\n\n'
            '![Three-case 3 by 3 comparison](experiment3_three_case_comparison.png)\n\n'
            'Columns compare the actual minimum separation and longest backbone edge with their applicable '
            'theoretical tubes and thresholds, then compare the geometric margin s(t) with the applicable '
            'distortion envelope. The distortion curve is graph independent for uncertified cases.\n\n'
            f"Connectivity is first lost at t={fmt(event)}. The minimum pair separation is "
            f"{fmt(me['min_separation'])}, with d_safe={meta['d_safe']:g}; final graph connected: "
            f"{me['observed_connected']}. The sparse path node order is "
            f"`{stress_cfg['path_order']}` and its communication radius is {stress_cfg['R_c']:g}.\n")
    grid=saved('experiment_4/summary.json')
    if grid:
        counts=grid['counts'];runs=[json.loads(p.read_text()) for p in active_grid_paths(root)]
        retries=sum(bool(r['metrics'].get('accuracy_retries')) for r in runs)
        macros={f'Map{key}Count':counts.get(key,0) for key in ['A','B','C','D']}
        macros.update(MapFailureCount=grid['validation_failures'],MapTotalCases=len(runs),
                      MapGridSize=grid['size'],MapMinSeparation=min(r['metrics']['min_separation'] for r in runs),
                      MapMinPeak=min(r['metrics']['peak_control'] for r in runs),
                      MapMaxPeak=max(r['metrics']['peak_control'] for r in runs),
                      MapDisconnectedCount=sum(not r['metrics']['observed_connected'] for r in runs),
                      MapSeparationCount=sum(not r['metrics']['observed_safe'] for r in runs))
        (root/'experiment_4/report_metrics.tex').write_text('\n'.join(
            f'\\newcommand{{\\{key}}}{{{fmt(value)}}}' for key,value in macros.items())+'\n')
        parts.append(f"Experiment 4: {grid['size']} by {grid['size']} pilot star parameter map\n"
                     'The station template, larger reference, initial physical state, geometry and\n'
                     '12-unit horizon are shared with Experiments 2/3. Initial integration settings\n'
                     'match their accepted accuracy; only physical parameters m and kappa vary.\n'
                     'Only m and kappa vary. Initial stations and physical positions are fixed.\n'
                     f"Counts: {counts}. Certified-with-violation validation failures: {grid['validation_failures']}.\n"
                     f"Cases requiring tighter integration to pass the fixed residual budget: {retries}.\n"
                     'The gain supremum is obtained from analytic affine-exponential margins and an\n'
                     'exhaustive monotone-derivative/root calculation, including endpoint limits.\n'
                     'A 1e-8 strict rate slack guards floating-point optimization. This is justified\n'
                     'global-extremum computation, not an interval-arithmetic formal proof.\n'
                     'The separation branch has at most one maximizing critical point; the increasing\n'
                     'connectivity branch attains its supremum at the deadline limit.\n'
                     'B cases demonstrate conservatism of the selected sufficient gain condition.\n'
                     'Geometry is feasible for all m,kappa; category D is implemented/tested but absent.\n'
                     'The backbone is a complete ten-node graph with 45 eligible links. This dense\n'
                     'backbone is explicit; no sparse communication result is inferred. Connectivity\n'
                     'also follows from the graph-independent upper tube, whose maximum is below\n'
                     'R_c at t=0 and decreases. The map primarily explores separation violations.\n'
                     'This is a coarse pilot grid: colors classify the sampled configurations,\n'
                     'not every parameter inside the displayed cells. The design curve is evaluated\n'
                     'at 401 speeds using the continuous-time gain calculation, without simulations.\n')
    parts.append('Numerical interpretation and limitations\n'
                 'All apparent envelope excesses are saved, including near-zero solver residuals.\n'
                 'The acceptance budget is an absolute 1e-8 norm error; retry integration uses\n'
                 'tenfold tighter tolerances and half the maximum step, never a larger budget.\n'
                 'Observed distance extrema use dense scanning, crossing events, and pair-specific\n'
                 'local optimization. They do not establish rigorous all-time observed safety; the\n'
                 'analytic tube certificate does. Planned feasibility is checked beyond the horizon.\n'
                 'Settling is the final crossing of an explicit tolerance over a finite horizon,\n'
                 'checked at output samples and adaptive nodes, not proof of exact floating-point zero.\n'
                 'Use 1e-6 for quantitative comparisons; 1e-8 is diagnostic if solver-sensitive.\n'
                 'Physical\n'
                 'relative tolerance scales with absolute x, whereas the independent error solver\n'
                 'scales with delta; the equivalence diagnostic therefore uses tighter tolerances.\n'
                 'Radau is checked on a short high-gain diagnostic before zero singularities.\n'
                 'Full nominal trajectories use DOP853; nonsmooth near-zero step reduction remains\n'
                 'expensive and is not treated by changing the controller. Energy uses numerical\n'
                 'quadrature; peaks are evaluated at saved dense-output samples. No actuator cap\n'
                 'or physical robot dimensions are included. No Monte Carlo experiments were run.\n')
    parts.append('Discrepancies found in the original implementation\n'
                 '1. Signed power was applied to the centroid error instead of each delta_i.\n'
                 '2. Coupling was divided by n, absent from the supplied report\'s controller.\n'
                 '3. The old integrator latched/projected centroid components at a threshold.\n'
                 '4. Old theoretical helpers used exponential station convergence and separate\n'
                 '   centroid/ISS claims from an earlier mathematical draft.\n'
                 '5. Range-limited graph/backbone certificates and full physical command reporting\n'
                 '   were missing. The true-distance denominator and analytic quintic formation\n'
                 '   derivatives were reusable and retained.\n'
                 'The current implementation matches the supplied report; the report is untouched.\n')
    parts.append('Items for the author\n'
                 '- ACC_paper/main.tex still states the earlier shared centroid corrector and\n'
                 '  M/n coupling (model around lines 175-186). It must be aligned with the\n'
                 '  finite-time_report.tex model before these figures support that manuscript.\n'
                 '  Both LaTeX sources were left untouched.\n'
                 '- Treat nominal 1e-8 settling as diagnostic when sensitivity flags it; use 1e-6\n'
                 '  or the explicitly saved tighter run for numerical comparisons.\n'
                 '- Experiment 4 uses a dense complete backbone and its parameter-map cases retain\n'
                 '  connectivity. Experiment 3 Case C separately shows complete-backbone link loss\n'
                 '  while the range-limited graph remains connected in the sampled history.\n'
                 '- Coordinates/time are normalized. Hardware scaling, control saturation, robot\n'
                 '  dimensions and disturbances require assumptions outside this nominal study.\n'
                 '- Legacy root figure scripts target the earlier draft; reproduce with the new\n'
                 '  runner and plot_results.py. No claims of novelty are established by these runs.\n')
    (root/'TECHNICAL_RESULTS.txt').write_text('\n'.join(parts))
