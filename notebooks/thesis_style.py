"""Shared matplotlib style for the thesis figures."""
import os
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4
TEXTWIDTH = 150 * MM          # one figure across the page
HALFWIDTH = 72 * MM           # two subfigures side by side with a 6 mm gap
# Default: 0.75 \textwidth, the caption width set by the template.
FIGWIDTH = 0.75 * TEXTWIDTH
FIGHEIGHT = 70 * MM

# Okabe & Ito (2008), "Color Universal Design".
COLORS = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
    "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000",
    "grey": "#7F7F7F",
}


def apply():
    """Set the rcParams every thesis figure uses. Call once per notebook."""
    usetex = shutil.which("latex") is not None
    mpl.rcParams.update({
        "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,          # embed TrueType, keeps text selectable and editable
        "ps.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=[COLORS[c] for c in
                                             ("blue", "orange", "green", "vermillion",
                                              "purple", "sky", "black")]),
    })
    return usetex


def figure(width=None, height=None):
    """One diagram at its printed size. Default: FIGWIDTH x FIGHEIGHT."""
    return plt.subplots(figsize=(width or FIGWIDTH, height or FIGHEIGHT), layout="constrained")


def legend_below(fig, ncol=3, **kw):
    """The legend under the plotting area, so it can never cover data."""
    return fig.legend(loc="outside lower center", ncol=ncol, **kw)


def save(fig, name, outdir="figures"):
    """Write <name>.pdf at the drawn size."""
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    return os.path.join(outdir, f"{name}.pdf")


def tex(s):
    """Escape a plain string for LaTeX when usetex is on (run names, config values)."""
    if not mpl.rcParams["text.usetex"]:
        return s
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                 ("#", r"\#")):
        s = s.replace(a, b)
    return s
