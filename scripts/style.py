"""Shared figure style: page geometry, palette, fonts and layout helpers."""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Page geometry of the CVPR template (inches).
COL, TEXT = 3.25, 6.875

# One palette for all figures (Okabe-Ito, colour-blind safe).
C_IN = "#D55E00"      # the in-loop judge and what it reports
C_OUT = "#0072B2"     # the sealed judge and what it measures
C_GREY = "#8A8A8A"
C_LIGHT = "#E6E6E6"


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arimo", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.title_fontsize": 7, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.labelpad": 2.5, "axes.titlepad": 4, "lines.linewidth": 1.1, "lines.markersize": 3.2,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.handlelength": 1.6, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
        "hatch.linewidth": 0.5,
    })


def panel_title(ax, label, title, color="k", span=None, y=1.0):
    """Bold panel label and its title as one unit, centred above the panel.

    span: the axes the panel covers when it has more than one; their positions are read here, so call this after the
    layout is final (after fill_width).
    """
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
    from matplotlib.transforms import blended_transform_factory
    fig = ax.figure
    box = HPacker(children=[TextArea(label, textprops=dict(fontsize=8, fontweight="bold")),
                            TextArea(cap(title), textprops=dict(fontsize=plt.rcParams["axes.titlesize"], color=color))],
                  align="baseline", pad=0, sep=3)
    if span is None:
        x, tr = 0.5, ax.transAxes
    else:
        fig.canvas.draw()
        P = [a.get_position() for a in span]
        x, tr = (min(p.x0 for p in P) + max(p.x1 for p in P)) / 2, blended_transform_factory(fig.transFigure, ax.transAxes)
    ab = AnchoredOffsetbox(loc="lower center", child=box, pad=0, borderpad=0, frameon=False,
                           bbox_to_anchor=(x, y), bbox_transform=tr)
    ab.set_clip_on(False)
    ax.add_artist(ab)
    return ab


def fill_width(fig, pad=0.02, rounds=4):
    """Move the left and right margins until the drawn content spans the canvas width."""
    W = fig.get_figwidth()
    for _ in range(rounds):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        sp = fig.subplotpars
        fig.subplots_adjust(left=sp.left - (bb.x0 - pad) / W, right=sp.right + (W - pad - bb.x1) / W)


def cap(s):
    """First letter in upper case, as in every label of the figures."""
    k = re.match(r"\s*(\([a-z]\))?\s*", s).end()  # leading blanks and a panel label stay as they are
    return s[:k] + s[k:k + 1].upper() + s[k + 1:]


def capitalise(fig):
    """Capitalise every text of the figure: labels, titles, legends, annotations and category tick labels."""
    import matplotlib.text as mtext
    fig.canvas.draw()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labs = [t.get_text() for t in axis.get_ticklabels()]
            if any(cap(l) != l for l in labs):
                lim = axis.get_view_interval().copy()
                axis.set_ticks(axis.get_ticklocs())
                axis.set_ticklabels([cap(l) for l in labs])
                axis.set_view_interval(*lim, ignore=True)
    for t in fig.findobj(mtext.Text):
        s = t.get_text()
        if s and cap(s) != s:
            t.set_text(cap(s))


def save(fig, out_dir, name):
    capitalise(fig)
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, name + ".pdf")
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None}, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", pdf)
