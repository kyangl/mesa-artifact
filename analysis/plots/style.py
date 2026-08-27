"""Shared colorblind-safe publication style and fixed series roles."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito palette with darker amber and rose for print contrast.
BLUE = "#0072B2"
AMBER = "#D98C00"          # was #E69F00
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#B05A8E"         # was #CC79A7
SKY = "#56B4E9"
YELLOW = "#F0E442"
GREY = "#7F7F7F"
INK = "#222222"
MUTED = "#666666"

# ---- fixed role assignments --------------------------------------------
ROLE = {
    "mesa":        {"color": BLUE,       "marker": "o", "ls": "-",
                    "label": "MESA-guided"},
    "random":      {"color": AMBER,      "marker": "s", "ls": "--",
                    "label": "Random allocation"},
    "oracle":      {"color": GREY,       "marker": "^", "ls": ":",
                    "label": "Oracle bound"},
    "monitor_all": {"color": PURPLE,     "marker": "D", "ls": "-.",
                    "label": "Monitor all"},
    "structural":  {"color": SKY,        "marker": "v", "ls": "-",
                    "label": "Structural (6)"},
    "extension":   {"color": VERMILLION, "marker": "P", "ls": "-",
                    "label": "Extension (10)"},
    "handcrafted": {"color": VERMILLION, "marker": "s", "ls": "--",
                    "label": "Handcrafted MAS stress"},
    "inter_agent": {"color": VERMILLION, "marker": "s", "ls": "--",
                    "label": "Inter-agent edges"},
    "tool_output": {"color": GREEN,      "marker": "o", "ls": "-",
                    "label": "Raw tool output"},
}

DOMAIN_TITLE = {"customer_service": "Customer service",
                "software_engineering": "Software engineering"}


# Paper omits narrative titles; slide variants retain them.
PAPER = "paper"
SLIDE = "slide"


def variant():
    import os
    return os.environ.get("MESA_FIG_VARIANT", PAPER)


def titled(fig, text, variant_=None):
    """Suptitle only in the slide variant."""
    if (variant_ or variant()) == SLIDE:
        fig.suptitle(text, fontsize=10)


def narrative(fig, text, y=-0.10, variant_=None):
    """Bottom narrative sentence only in the slide variant."""
    if (variant_ or variant()) == SLIDE:
        fig.text(0.5, y, text, ha="center", fontsize=7, color=MUTED)


def stem_for(stem, variant_=None):
    v = variant_ or variant()
    return stem if v == PAPER else "%s_slide" % stem


def apply():
    plt.rcParams.update({
        # Embed searchable TrueType fonts instead of Type 3.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Named explicitly rather than left to the default resolution order,
        # so the same file renders identically on a machine with different
        # fonts installed.
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "figure.constrained_layout.use": True,
    })


def save(fig, stem, figdir=None):
    """Write BOTH a 300-dpi PNG and a vector PDF, and report the paths."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    figdir = Path(figdir or repo / "figures")
    (figdir / "pdf").mkdir(parents=True, exist_ok=True)
    png = figdir / ("%s.png" % stem)
    pdf = figdir / "pdf" / ("%s.pdf" % stem)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("  wrote %s" % png.relative_to(repo))
    print("  wrote %s" % pdf.relative_to(repo))
    return png, pdf


def errorbar_ci(ax, x, mean, lo, hi, role, label=None, **kw):
    """A point with an asymmetric CI, styled by role."""
    r = ROLE[role]
    ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]],
                color=r["color"], marker=r["marker"], ls="none",
                capsize=3, capthick=1.2, elinewidth=1.4,
                label=label if label is not None else r["label"], **kw)


def zero_line(ax, orientation="h"):
    """The null. Every difference plot must show where zero is."""
    fn = ax.axhline if orientation == "h" else ax.axvline
    fn(0, color=INK, lw=1.0, ls="-", alpha=0.55, zorder=1)


def full_box(ax):
    """Close the axes box on all four sides, in black.

    The house style hides the top and right spines, which suits a line plot
    where the data runs to the edge. It suits a scatter far less: points sit
    in open space and the eye has no boundary to read position against. This
    is opt-in per figure rather than a default, so the two treatments stay a
    deliberate choice instead of drifting apart.
    """
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#000000")
        ax.spines[side].set_linewidth(0.8)
