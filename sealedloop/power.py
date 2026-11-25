"""Power of the sequential test under a stated budget, by simulation.

Correlation model: each drawn item i has a latent difficulty z_i ~ N(0, 1)
shared by both policies. Policy k succeeds on item i when

    sqrt(rho) * z_i + sqrt(1 - rho) * e_ki  <  Phi^{-1}(p_k),

with e_ki ~ N(0, 1) independent, so the marginal success rate is p_k and the
within-item tetrachoric correlation is rho. rho = 0 is unpaired noise; rho
near 1 means the two policies succeed and fail on the same items, where
pairing helps most.

With `binary=False` the score is a probit transform of the same latent
variable, which keeps the marginal mean at p_k only approximately.

At delta = 0 the same simulation checks the error rate: the fraction of runs
that accept must be at most alpha up to Monte Carlo error.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np

from .gate import GateConfig, gate_path

_N = NormalDist()


def simulate_gate(cfg: GateConfig, p0: float, delta: float, rho: float,
                  n_sims: int, seed: int, binary: bool = True) -> dict:
    if not (0.0 < p0 < 1.0) or not (0.0 <= p0 + delta <= 1.0):
        raise ValueError("need 0 < p0 < 1 and 0 <= p0 + delta <= 1")
    if not (0.0 <= rho < 1.0):
        raise ValueError("need 0 <= rho < 1")
    rng = np.random.default_rng(seed)
    p1 = p0 + delta
    t0, t1 = _N.inv_cdf(p0), _N.inv_cdf(p1) if 0 < p1 < 1 else (
        float("inf") if p1 >= 1 else float("-inf"))
    decisions = {"accept": 0, "reject": 0, "unresolved_at_budget": 0, "quote": 0,
                 "cs_empty": 0}
    stops = []
    n = cfg.budget
    for _ in range(n_sims):
        z = rng.standard_normal(n)
        e0 = rng.standard_normal(n)
        e1 = rng.standard_normal(n)
        l0 = np.sqrt(rho) * z + np.sqrt(1 - rho) * e0
        l1 = np.sqrt(rho) * z + np.sqrt(1 - rho) * e1
        if binary:
            s0 = (l0 < t0).astype(float)
            s1 = (l1 < t1).astype(float)
        else:
            s0 = np.clip(np.array([_N.cdf(v) for v in (t0 - l0)]), 0, 1)
            s1 = np.clip(np.array([_N.cdf(v) for v in (t1 - l1)]), 0, 1)
        r = gate_path(s1, s0, cfg)
        decisions[r.decision] += 1
        stops.append(r.stopped_at if r.stopped_at is not None else cfg.budget)
    stops = np.asarray(stops, dtype=float)
    out = {
        "config": cfg.to_dict(), "p0": p0, "delta": delta, "rho": rho,
        "n_sims": n_sims, "seed": seed, "binary": binary,
        "correlation_model": "shared latent difficulty, tetrachoric rho",
        "p_accept": decisions["accept"] / n_sims,
        "p_reject": decisions["reject"] / n_sims,
        "p_unresolved_at_budget": decisions["unresolved_at_budget"] / n_sims,
        "p_cs_empty": decisions["cs_empty"] / n_sims,
        "mean_pairs_to_stop": float(stops.mean()),
        "median_pairs_to_stop": float(np.median(stops)),
        "mc_se_of_p_accept": float(np.sqrt(
            max(decisions["accept"] / n_sims * (1 - decisions["accept"] / n_sims), 1e-9)
            / n_sims)),
    }
    if delta == 0.0:
        out["alpha_check"] = {
            "nominal": cfg.alpha,
            "observed_false_accept": out["p_accept"],
            "within_3_mc_se": out["p_accept"] <= cfg.alpha + 3 * out["mc_se_of_p_accept"],
        }
    return out


def power_table(cfg: GateConfig, p0s, deltas, rho: float, n_sims: int,
                seed: int) -> list:
    rows = []
    k = 0
    for p0 in p0s:
        for delta in deltas:
            if not (0 <= p0 + delta <= 1):
                continue
            rows.append(simulate_gate(cfg, float(p0), float(delta), rho,
                                      n_sims, seed + k))
            k += 1
    return rows
