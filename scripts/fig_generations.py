"""Every generation of every run, as reported and as measured.

For each simulation testbed two grids share their rows (one run per row, grouped by which judge decides) and
columns (generations): the gain the in-loop judge reported and the gain the sealed judge measured, both over the
policy the run held at that point. A ring marks a candidate the deciding judge accepted. Manipulation runs whose
in-loop judge had not failed continued to generation eight; cells of generations a run did not reach stay empty.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

import records as rec
import style as st

CMAP = LinearSegmentedColormap.from_list("sl_div", ["#B2182B", "#F4A582", "#FFFFFF", "#92C5DE", "#2166AC"])


def grid(ax, runs, key, lim, max_gen, groups, show_rows):
    A = np.full((len(runs), max_gen), np.nan)
    for i, gens in enumerate(runs):
        for g in gens:
            if g["generation"] <= max_gen:
                A[i, g["generation"] - 1] = g[key]
                if g["kept"]:
                    ax.scatter([g["generation"] - 1], [i], s=10, facecolor="none", edgecolor="k", lw=0.7, zorder=3)
    ax.set_facecolor("#F2F2F2")
    im = ax.imshow(A, cmap=CMAP, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", interpolation="nearest")
    ax.set_xticks(range(max_gen))
    ax.set_xticklabels(range(1, max_gen + 1))
    ax.set_xticks(np.arange(-0.5, max_gen), minor=True)
    ax.set_yticks(np.arange(-0.5, len(runs)), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    row = 0
    ticks, labels = [], []
    for name, n in groups:
        if row:
            ax.axhline(row - 0.5, color="k", lw=0.8)
        ticks.append(row + n / 2 - 0.5)
        labels.append(name)
        row += n
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels if show_rows else [], linespacing=1.05)
    return im


def ordered(runs, variants):
    """Runs grouped by variant in the given order; within a variant, in the order of the records."""
    return [gens for v in variants for gens in runs.values() if gens[0]["variant"] == v]


def main():
    p = rec.parser(__doc__)
    p.add_argument("--out", default="figures", help="output folder")
    a = p.parse_args()
    st.setup()
    rows = rec.generations(a.records)
    drive = ordered(rec.runs(rows, "driving"), ("sealed", "critic_true_labels", "critic_own_labels"))
    manip = ordered(rec.runs(rows, "manipulation"), ("sealed", "classifier"))
    fig = plt.figure(figsize=(st.TEXT, 1.75))
    gs = fig.add_gridspec(1, 9, width_ratios=[5, 5, 0.28, 4.6, 8, 8, 0.28, 0.01, 0.01], wspace=0.10)
    d_groups = [("sealed judge\ndecides", 3), ("critic decides,\ntrue labels", 3), ("critic decides,\nown labels", 3)]
    m_groups = [("sealed judge\ndecides", 9), ("classifier\ndecides", 9)]
    titles = []
    for tb, runs, lim, mg, groups, cols, unit, lab in (("drive", drive, 0.12, 5, d_groups, (0, 1, 2), "PDMS", ("(a)", "driving")),
                                                         ("manip", manip, 0.5, 8, m_groups, (4, 5, 6), "success rate", ("(b)", "manipulation"))):
        pair = []
        for j, (key, title) in enumerate((("inloop_gain", "in-loop judge"), ("sealed_gain", "sealed judge"))):
            ax = fig.add_subplot(gs[0, cols[j]])
            im = grid(ax, runs, key, lim, mg, groups, j == 0)
            ax.set_title(title, color=st.C_IN if j == 0 else st.C_OUT, fontsize=7, linespacing=1.05)
            ax.set_xlabel("generation")
            pair.append(ax)
            if j == 0 and tb == "manip":
                for k in range(1, 6):
                    if k != 3:
                        ax.axhline(3 * k - 0.5, color="k", lw=0.3, ls=(0, (2, 2)))
        cax = fig.add_subplot(gs[0, cols[2]])
        cb = fig.colorbar(im, cax=cax, ticks=[-lim, 0, lim])
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(length=1.5, width=0.4)
        cb.set_label("gain (%s)" % unit, labelpad=1)
        titles.append((pair, lab))
    st.fill_width(fig)
    for pair, lab in titles:
        st.panel_title(pair[0], lab[0], lab[1], span=pair, y=1.11)
    st.save(fig, a.out, "fig_generations")


if __name__ == "__main__":
    main()
