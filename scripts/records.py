"""Readers for the evaluation records of the paper.

The records hold one row per generation (`generations.csv`) and, for every generation, the per-item paired scores
of the current policy and the candidate under each judge. Download them next to this repository, or point
SEALEDLOOP_RECORDS (or --records) at their folder.
"""
import argparse, csv, gzip, os
from collections import defaultdict

import numpy as np

from sealedloop import GateConfig, gate_path

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.environ.get("SEALEDLOOP_RECORDS", os.path.join(os.path.dirname(HERE), "records"))

# The test of the paper: level 0.05, floor 0, no decision before 10 pairs.
TEST = dict(alpha=0.05, floor=0.0, min_n=10, score_range=(0.0, 1.0))
# Manipulation budgets in pairs. The driving tests read every recorded frame and every log.
BUDGET = {"sealed": 1000, "inloop": 300}

INT = ("control", "seed", "generation", "kept")


def parser(doc):
    p = argparse.ArgumentParser(description=doc)
    p.add_argument("--records", default=DEFAULT_ROOT, help="folder of the evaluation records")
    return p


def read_csv(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return list(csv.DictReader(f))


def generations(root):
    """Rows of generations.csv; numeric columns are converted, empty cells become None."""
    rows = read_csv(os.path.join(root, "generations.csv"))
    for r in rows:
        for k, v in r.items():
            if v == "":
                r[k] = None
            elif k in INT or k.endswith(("_read", "_items")):
                r[k] = int(v)
            elif k.endswith(("_gain", "_lo", "_hi", "_mean", "_frames")):
                r[k] = float(v)
    return rows


def runs(rows, testbed, control=0):
    """Generations of one testbed grouped by run, in the order of the file."""
    out = {}
    for r in rows:
        if r["testbed"] == testbed and r["control"] == control:
            out.setdefault(r["run"], []).append(r)
    return out


def pairs(root, row, judge):
    """Per-item paired scores of one generation under one judge ("sealed" or "inloop"), in test order."""
    return read_csv(os.path.join(root, row["pairs_file_" + judge]))


def scores(items):
    return (np.array([float(r["new"]) for r in items]), np.array([float(r["old"]) for r in items]))


def decide(root, row, judge):
    """Recompute the decision of one generation under one judge from its paired scores.

    Manipulation: one sequential test over the items in the order they were drawn. Driving: frames of one log are
    dependent, so the test is read twice, over frames and over per-log means, and where the two readings differ
    the per-log reading governs.
    """
    items = pairs(root, row, judge)
    new, old = scores(items)
    if row["testbed"] == "manipulation":
        return gate_path(new, old, GateConfig(**TEST, budget=BUDGET[judge]))
    frame = gate_path(new, old, GateConfig(**TEST, budget=len(items)))
    by = defaultdict(list)
    for i, r in enumerate(items):
        by[int(r["log_order"])].append(i)
    order = sorted(by)
    block = gate_path(np.array([new[by[l]].mean() for l in order]), np.array([old[by[l]].mean() for l in order]),
                      GateConfig(**TEST, budget=len(order)))
    return block if frame.decision != block.decision else frame
