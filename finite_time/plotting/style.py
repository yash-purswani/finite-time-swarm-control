"""Matplotlib styling and publication aesthetics."""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# Column widths in inches (ECC / IEEE double-column format)
COL_W = 3.45     # Single column
COL2_W = 7.15    # Full width across two columns

# Color scheme for gains m
COLORS = {
    0.0: '#D55E00',   # Vermilion / Red-Orange
    0.5: '#CC79A7',   # Purple / Magenta
    1.0: '#E69F00',   # Orange
    2.0: '#0072B2',   # Blue
    2.5: '#56B4E9',   # Sky Blue
    3.0: '#F0E442',   # Yellow
    5.0: '#009E73',   # Green
    10.0: '#000000',  # Black
}


def use_paper_style() -> None:
    """Apply uniform paper style conforming to IEEE/ECC publication requirements."""
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.0,
    })


def save_figure(fig: plt.Figure, folder: Path, name: str) -> None:
    """Save figure in both vector (.pdf) and raster (.png) formats."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(folder / f"{name}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
