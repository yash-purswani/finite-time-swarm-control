"""Plotting utilities and figure generators."""
from .style import use_paper_style, save_figure, COL_W, COL2_W, COLORS
from .figures import (
    plot_morphing_snapshots,
    plot_gain_evaluation,
    plot_transient_2x3_grid,
    plot_parameter_map_and_control,
)

__all__ = [
    'use_paper_style',
    'save_figure',
    'COL_W',
    'COL2_W',
    'COLORS',
    'plot_morphing_snapshots',
    'plot_gain_evaluation',
    'plot_transient_2x3_grid',
    'plot_parameter_map_and_control',
]
