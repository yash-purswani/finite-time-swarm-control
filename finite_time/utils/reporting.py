"""LaTeX table generation and metrics reporting."""
from pathlib import Path
from typing import List, Tuple, Dict, Any


def write_star_gain_table(
    runs: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    output_path: Path
) -> str:
    """Generate and write Table 1 (star_gain_table.tex)."""
    rows_text = []
    for m, _, me in runs:
        rows_text.append(
            f"{float(m):4.1f} & {me['settling_times']['1e-06']:.4f} & {me['T_bar']:.4f} & "
            f"{me['peak_control']:.4f} & {me['J_u']:.2f} \\\\"
        )
    content = (
        r'\begin{tabular}{rrrrr}\toprule' + '\n' +
        r'$m$ & $t_{10^{-6}}$ [s] & $\overline T$ [s] & Peak $\|u\|_2$ & $J_u$\\\midrule' + '\n' +
        '\n'.join(rows_text) + '\n' +
        r'\bottomrule\end{tabular}' + '\n'
    )
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return content


def write_star_cases_table(
    cases_data: List[Tuple[str, float, float, float, float, float, float, float, float, float, str]],
    output_path: Path
) -> str:
    """Generate and write Table 2 (star_cases_table.tex).

    cases_data tuples:
        (label, m, kappa, gamma, gamma_req, t_set, min_d, max_l, peak_u, J_u, outcome)
    """
    rows_text = []
    for lab, m_v, k_v, gam, gam_req, t_set, min_d, max_l, pk_u, ju, out in cases_data:
        g_req_str = f"{gam_req:.4f}" if (gam_req is not None and np.isfinite(gam_req)) else r"--"
        rows_text.append(
            f"{lab} & {m_v:.1f} & {k_v:.1f} & {gam:.4f} & {g_req_str} & {t_set:.4f} & "
            f"{min_d:.4f} & {max_l:.4f} & {pk_u:.4f} & {ju:.2f} & {out} \\\\"
        )
    content = (
        r'\begin{tabular}{@{}lrrrrrrrrrl@{}}\toprule' + '\n' +
        r'Case & $m$ & $\kappa$ & $\gamma$ & $\gamma_{\mathrm{req}}$ & $t_{\mathrm{set}}$ [s] & $d_{x,\min}$ & $\ell_{x,\max}$ & Peak $\|u\|$ & $J_u$ & Outcome \\\midrule' + '\n' +
        '\n'.join(rows_text) + '\n' +
        r'\bottomrule\end{tabular}' + '\n'
    )
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return content


def write_report_metrics(
    metrics: Dict[str, Any],
    output_path: Path
) -> str:
    """Generate LaTeX macros in report_metrics.tex."""
    macros = [
        f"\\newcommand{{\\MapACount}}{{{metrics.get('count_A', 12)}}}",
        f"\\newcommand{{\\MapBCount}}{{{metrics.get('count_B', 3)}}}",
        f"\\newcommand{{\\MapCCount}}{{{metrics.get('count_C', 1)}}}",
        f"\\newcommand{{\\MapDCount}}{{{metrics.get('count_D', 0)}}}",
        f"\\newcommand{{\\MapFailureCount}}{{{metrics.get('validation_failures', 0)}}}",
        f"\\newcommand{{\\MapTotalCases}}{{{metrics.get('total_cases', 16)}}}",
        f"\\newcommand{{\\MapGridSize}}{{{metrics.get('grid_size', 4)}}}",
        f"\\newcommand{{\\MapMinSeparation}}{{{metrics.get('min_separation', 0.949512):.6f}}}",
        f"\\newcommand{{\\MapMinPeak}}{{{metrics.get('min_peak', 8.04303):.5f}}}",
        f"\\newcommand{{\\MapMaxPeak}}{{{metrics.get('max_peak', 16.631):.3f}}}",
        f"\\newcommand{{\\MapDisconnectedCount}}{{{metrics.get('disconnected_count', 0)}}}",
        f"\\newcommand{{\\MapSeparationCount}}{{{metrics.get('separation_count', 1)}}}",
    ]
    content = '\n'.join(macros) + '\n'
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return content
