"""Betting confidence sequence for the mean of bounded observations.

Hedged capital process of Waudby-Smith and Ramdas (2024). For observations
d_t in [a, b], write u_t = (d_t - a) / (b - a) in [0, 1] and, for a candidate
mean m, m' = (m - a) / (b - a). Under H0: E[d_t | past] = m,

    K_t^+(m) = prod_i (1 + nu_i (u_i - m'))
    K_t^-(m) = prod_i (1 - nu_i (u_i - m'))

are nonnegative martingales for any predictable nu_i that keeps both factors
positive. The two are mixed as K = (K^+ + K^-) / 2, and the confidence set at
time t is {m : max_{s <= t} K_s(m) < 1 / alpha}. The running maximum makes the
set a confidence sequence: a candidate once rejected stays rejected, so the
interval may be read after every observation without inflating the error.

The betting fractions follow the predictable plug-in rule of the paper above,
truncated at c / m' and c / (1 - m') (their Theorem 3, eq. 25) with c = 3/4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Truncation constant c of the betting fractions.
BET_CAP = 0.75

# "wsr":  candidate-dependent truncation, nu <= c / m' and nu <= c / (1 - m').
# "flat": additionally nu <= c. Wider at small sample sizes; kept for
#         sensitivity checks.
TRUNCATION = "wsr"

# Number of candidate means on the grid over which the wealth is inverted.
DEFAULT_GRID = 2001


@dataclass
class CSResult:
    """A confidence sequence read at one time.

    n is the number of observations consumed. rejected_all is True when the
    wealth process rejected every candidate mean on the grid.
    """

    lo: float
    hi: float
    n: int
    alpha: float
    method: str
    rejected_all: bool = False
    diagnostics: dict = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi


def _predictable_bets(u: np.ndarray, alpha: float, cap: float,
                      truncation: str = TRUNCATION) -> np.ndarray:
    """Betting fractions nu_i, each a function of u_1..u_{i-1} only."""
    n = u.size
    idx = np.arange(1, n + 1, dtype=float)
    csum = np.cumsum(u)
    mu_at = (0.5 + csum) / (1.0 + idx)
    mu_prev = np.concatenate(([0.5], mu_at[:-1]))
    sq = (u - mu_prev) ** 2
    var_at = (0.25 + np.cumsum(sq)) / (1.0 + idx)
    var_prev = np.concatenate(([0.25], var_at[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.sqrt(
            2.0 * math.log(2.0 / alpha) / (var_prev * idx * np.log(1.0 + idx))
        )
    raw = np.nan_to_num(raw, nan=cap, posinf=cap, neginf=0.0)
    raw = np.maximum(0.0, raw)
    return np.minimum(cap, raw) if truncation == "flat" else raw


def betting_cs(
    d,
    a,
    b,
    alpha: float = 0.05,
    grid: int = DEFAULT_GRID,
    cap: float = BET_CAP,
    truncation: str = TRUNCATION,
    read_at=None,
) -> CSResult:
    """Hedged capital confidence sequence for the mean of d_1..d_n in [a, b].

    a, b: lower and upper bounds, fixed before the observations are seen.
    Scalars are broadcast; per-observation bounds are accepted as arrays.
    read_at: optional counts at which to also report the interval. The wealth
    process is cumulative, so every prefix is computed in one pass; the
    intervals are returned in `diagnostics["path"]` as (n, lo, hi) triples.
    """
    if truncation not in ("flat", "wsr"):
        raise ValueError(f"unknown truncation rule {truncation!r}")
    d = np.asarray(d, dtype=float).ravel()
    n = d.size
    a = np.broadcast_to(np.asarray(a, dtype=float).ravel(), (n,)) if n else np.zeros(0)
    b = np.broadcast_to(np.asarray(b, dtype=float).ravel(), (n,)) if n else np.zeros(0)
    if n == 0:
        return CSResult(0.0, 1.0, 0, alpha, "wsr_hedged_capital")
    s = b - a
    if np.any(s <= 0):
        raise ValueError("every range must satisfy b > a")
    if np.any(d < a - 1e-9) or np.any(d > b + 1e-9):
        bad = int(np.argmax((d < a - 1e-9) | (d > b + 1e-9)))
        raise ValueError(
            f"observation {bad} = {d[bad]:.6g} lies outside its declared range "
            f"[{a[bad]:.6g}, {b[bad]:.6g}]"
        )
    u = np.clip((d - a) / s, 0.0, 1.0)

    nu = _predictable_bets(u, alpha, cap, truncation)

    # A mean outside the range of any observation is impossible under H0.
    m_lo, m_hi = float(a.max()), float(b.min())
    if m_lo >= m_hi:
        return CSResult(float(a.min()), float(b.max()), n, alpha,
                        "wsr_hedged_capital", rejected_all=True,
                        diagnostics={"reason": "ranges do not intersect"})
    m = np.linspace(m_lo, m_hi, grid)

    mt = (m[:, None] - a[None, :]) / s[None, :]          # (G, n) candidate on unit scale
    mt = np.clip(mt, 1e-12, 1.0 - 1e-12)
    resid = u[None, :] - mt                              # (G, n)
    nu_plus = np.minimum(nu[None, :], cap / mt)
    nu_minus = np.minimum(nu[None, :], cap / (1.0 - mt))

    # The truncations keep 1 + nu (u - m) strictly positive for every
    # attainable u. This is asserted instead of clipped: a clip would stop the
    # wealth process from being a martingale without leaving a trace.
    arg_plus = nu_plus * resid
    arg_minus = -nu_minus * resid
    worst = min(float(arg_plus.min()), float(arg_minus.min()))
    if worst <= -1.0 + 1e-12:
        raise AssertionError(
            "a betting factor reached the positivity barrier "
            f"(min 1 + nu(u - m) = {1.0 + worst:.3e})"
        )
    term_plus = np.log1p(arg_plus)
    term_minus = np.log1p(arg_minus)
    log_kp = np.cumsum(term_plus, axis=1)
    log_km = np.cumsum(term_minus, axis=1)
    mx = np.maximum(log_kp, log_km)
    log_k = mx + np.log(0.5 * (np.exp(log_kp - mx) + np.exp(log_km - mx)))
    running = np.maximum.accumulate(log_k, axis=1)
    running_max = running[:, -1]

    threshold = math.log(1.0 / alpha)
    path = []
    if read_at is not None:
        for t in read_at:
            t = int(t)
            if t < 1 or t > n:
                continue
            alive_t = running[:, t - 1] < threshold
            if alive_t.any():
                kept_t = m[alive_t]
                path.append((t, float(kept_t.min()), float(kept_t.max())))
            else:
                path.append((t, float("nan"), float("nan")))
    keep = running_max < threshold
    if not keep.any():
        return CSResult(m_lo, m_hi, n, alpha, "wsr_hedged_capital",
                        rejected_all=True,
                        diagnostics={"min_log_wealth": float(running_max.min()),
                                     "path": path})
    kept = m[keep]
    return CSResult(
        lo=float(kept.min()),
        hi=float(kept.max()),
        n=n,
        alpha=alpha,
        method="wsr_hedged_capital",
        diagnostics={"mean_bet": float(nu.mean()),
                     "truncation": truncation, "cap": cap,
                     "grid_lo": m_lo, "grid_hi": m_hi,
                     "path": path},
    )


def predicted_halfwidth(sigma2: float, range_width: float, n: int,
                        alpha: float) -> float:
    """Empirical-Bernstein proxy for the half-width after n observations.

        halfwidth ~ sqrt(2 sigma^2 L / n) + 7 R L / (3 (n - 1)),  L = log(2 / alpha)

    Used only to estimate how many more pairs an undecided test needs; it is
    never reported as an interval.
    """
    L = math.log(2.0 / alpha)
    n = max(int(n), 1)
    var_term = math.sqrt(2.0 * max(sigma2, 0.0) * L / n)
    rng_term = 7.0 * max(range_width, 0.0) * L / (3.0 * max(n - 1, 1))
    return var_term + rng_term


def pairs_to_separate(sigma2: float, range_width: float, n_now: int,
                      gap: float, alpha: float, max_pairs: int = 100_000):
    """Additional pairs until the predicted half-width falls below `gap`.

    Returns None when the gap is too small to estimate within `max_pairs`.
    A test that has not decided never returns zero.
    """
    if not np.isfinite(gap) or gap <= 1e-9:
        return None
    added = 0
    while added < max_pairs:
        if predicted_halfwidth(sigma2, range_width, n_now + added, alpha) < gap:
            return max(1, added)
        added += 1
    return None
