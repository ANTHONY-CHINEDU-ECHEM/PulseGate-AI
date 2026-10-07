"""One visual language for every chart in the project."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e6e5e1"

# Categorical colours follow the class, never its rank, and keep a fixed order.
CLASS_COLOURS = {
    "live": "#2a78d6",
    "print_matte": "#eb6834",
    "print_glossy": "#1baf7a",
    "replay_phone": "#eda100",
    "replay_monitor": "#e87ba4",
    "cutout_mask": "#008300",
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ATTACK = "#eb6834"          # all attacks pooled, against live in blue
SEQUENTIAL = ["#f3f7fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

CLASS_LABELS = {
    "live": "Live", "print_matte": "Print, matte", "print_glossy": "Print, glossy",
    "replay_phone": "Replay, phone", "replay_monitor": "Replay, monitor", "cutout_mask": "Cut out mask",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "text.color": INK, "axes.labelcolor": INK_SOFT, "axes.edgecolor": GRID, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.axisbelow": True,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9.5,
        "figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
    })


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def note(ax, text: str) -> None:
    """Small source or method note under an axis."""
    ax.annotate(text, xy=(0, 0), xycoords="axes fraction", xytext=(0, -42), textcoords="offset points",
                ha="left", va="top", fontsize=8.5, color=INK_MUTED)
