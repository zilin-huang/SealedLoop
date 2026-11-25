"""The sequential test that keeps or rolls back one generation.

The current policy and the candidate are evaluated on the same drawn items,
and each item i yields a paired difference

    d_i = s_new_i - s_old_i,   s in [r_lo, r_hi],   so d_i in [-(r_hi - r_lo), r_hi - r_lo].

A per-item score is a success indicator on the manipulation testbed and a
bounded driving score on the driving testbed. The test only requires that the
range is fixed before the first item and is the same for both policies. The
confidence sequence of `confseq.betting_cs` is valid at every pair count, so
the test may be read after each pair.

Decisions:

  accept                 lower end of the sequence above zero: the gain is
                         positive at level alpha. The candidate is kept.
  reject                 upper end below the floor: no gain. Rolled back.
  quote                  neither, budget not exhausted. The reading carries an
                         estimate of the additional pairs at which the current
                         trend would decide.
  unresolved_at_budget   neither, budget exhausted. Rolled back, and counted
                         separately from reject.
  cs_empty               the wealth process rejected every candidate mean. For
                         the true mean this has probability at most alpha.
                         Rolled back, and counted separately from reject.

A reading before `min_n` pairs is always "quote", so that no candidate is kept
on a handful of items.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .confseq import betting_cs, pairs_to_separate, BET_CAP, TRUNCATION


@dataclass(frozen=True)
class GateConfig:
    alpha: float = 0.05
    floor: float = 0.0
    budget: int = 100
    min_n: int = 10
    score_range: tuple = (0.0, 1.0)
    truncation: str = TRUNCATION
    cap: float = BET_CAP

    def __post_init__(self):
        lo, hi = self.score_range
        if not (hi > lo):
            raise ValueError("score_range must satisfy hi > lo")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1)")
        if self.floor > 0.0:
            raise ValueError(
                "floor must be <= 0: with a positive floor accept and reject "
                "could both hold")
        if self.min_n < 1 or self.budget < self.min_n:
            raise ValueError("need 1 <= min_n <= budget")

    @property
    def diff_range(self) -> tuple:
        w = float(self.score_range[1] - self.score_range[0])
        return (-w, w)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score_range"] = list(self.score_range)
        d["diff_range"] = list(self.diff_range)
        return d


@dataclass
class GateReading:
    decision: str
    lo: float
    hi: float
    n: int
    mean_diff: float
    alpha: float
    config: dict
    quote_additional_pairs: int | None = None
    quote_total_pairs: int | None = None
    quote_within_budget: bool | None = None
    stopped_at: int | None = None
    note: str = ""
    diagnostics: dict = field(default_factory=dict)

    @property
    def keep(self) -> bool:
        """True when the candidate replaces the current policy."""
        return self.decision == "accept"

    def to_dict(self) -> dict:
        return asdict(self)


def paired_differences(s_new, s_old, cfg: GateConfig) -> np.ndarray:
    s_new = np.asarray(s_new, dtype=float).ravel()
    s_old = np.asarray(s_old, dtype=float).ravel()
    if s_new.size != s_old.size:
        raise ValueError(
            f"{s_new.size} new against {s_old.size} old scores; the test is "
            "paired, so both policies must be scored on the same items")
    lo, hi = cfg.score_range
    for name, s in (("new", s_new), ("old", s_old)):
        if s.size and (s.min() < lo - 1e-9 or s.max() > hi + 1e-9):
            raise ValueError(
                f"{name} scores lie outside the declared range [{lo}, {hi}]: "
                f"[{s.min():.4g}, {s.max():.4g}]")
        if np.any(~np.isfinite(s)):
            raise ValueError(
                f"{name} scores contain a non-finite value; drop the pair "
                "instead of scoring a missing item as zero")
    return s_new - s_old


def _decide(lo: float, hi: float, n: int, cfg: GateConfig) -> str:
    if n < cfg.min_n:
        return "quote"
    if lo > 0.0:
        return "accept"
    if hi < cfg.floor:
        return "reject"
    if n >= cfg.budget:
        return "unresolved_at_budget"
    return "quote"


def read_gate(s_new, s_old, cfg: GateConfig) -> GateReading:
    """Read the test at the current pair count."""
    d = paired_differences(s_new, s_old, cfg)
    n = int(d.size)
    a, b = cfg.diff_range
    if n == 0:
        return GateReading("quote", a, b, 0, float("nan"), cfg.alpha,
                           cfg.to_dict(), note="no pairs yet")
    cs = betting_cs(d, a, b, alpha=cfg.alpha, cap=cfg.cap,
                    truncation=cfg.truncation)
    mean = float(d.mean())
    if cs.rejected_all:
        return GateReading("cs_empty", float("nan"), float("nan"), n, mean,
                           cfg.alpha, cfg.to_dict(),
                           note=("the wealth process rejected every candidate "
                                 "mean; rolled back and counted as its own "
                                 "outcome"),
                           diagnostics={"cs_method": cs.method,
                                        "cs_diagnostics": cs.diagnostics})
    decision = _decide(cs.lo, cs.hi, n, cfg)
    reading = GateReading(decision, float(cs.lo), float(cs.hi), n, mean,
                          cfg.alpha, cfg.to_dict(),
                          diagnostics={"cs_method": cs.method,
                                       "mean_bet": cs.diagnostics.get("mean_bet")})
    if decision == "quote":
        gap = abs(mean) if n >= 2 else 0.0
        sigma2 = float(np.var(d, ddof=1)) if n >= 2 else 0.0
        added = None
        if n >= 2 and gap > 1e-9 and sigma2 > 0.0:
            added = pairs_to_separate(sigma2, b - a, n, gap, cfg.alpha)
        if added is None:
            reading.note = ("the observed mean difference is too small or the "
                            "variance is degenerate at this count; the quote "
                            "is the budget remainder")
            reading.quote_additional_pairs = max(cfg.budget - n, 1)
        else:
            reading.quote_additional_pairs = int(added)
            reading.note = ("estimate from the half-width predictor at the "
                            "current mean and variance; not a guarantee")
        reading.quote_total_pairs = n + reading.quote_additional_pairs
        reading.quote_within_budget = reading.quote_total_pairs <= cfg.budget
        if n < cfg.min_n:
            reading.note = (f"below the minimum of {cfg.min_n} pairs; no "
                            f"decision is read before it. " + reading.note)
    return reading


def gate_path(s_new, s_old, cfg: GateConfig) -> GateReading:
    """Read the test after every pair and stop at the first decision.

    Returns the first decisive reading at or after `min_n` with `stopped_at`
    set, or the reading at the last pair if nothing decided. One pass over the
    wealth process; the running maximum makes the path monotone, so a decision
    once reached stays reached.
    """
    s_new = np.asarray(s_new, dtype=float).ravel()
    s_old = np.asarray(s_old, dtype=float).ravel()
    d = paired_differences(s_new, s_old, cfg)
    n = int(d.size)
    a, b = cfg.diff_range
    if n == 0:
        return read_gate(s_new, s_old, cfg)
    n_use = min(n, cfg.budget)
    cs = betting_cs(d[:n_use], a, b, alpha=cfg.alpha, cap=cfg.cap,
                    truncation=cfg.truncation,
                    read_at=list(range(cfg.min_n, n_use + 1)))
    for t, lo, hi in cs.diagnostics.get("path", []):
        if not (np.isfinite(lo) and np.isfinite(hi)):
            r = read_gate(s_new[:t], s_old[:t], cfg)
            if r.decision != "cs_empty":
                raise AssertionError(
                    f"path empty at pair {t} but read_gate at {t} says "
                    f"{r.decision}; the two readings must agree")
            r.stopped_at = int(t)
            return r
        if lo > 0.0 or hi < cfg.floor:
            r = read_gate(s_new[:t], s_old[:t], cfg)
            r.stopped_at = int(t)
            return r
    r = read_gate(s_new[:n_use], s_old[:n_use], cfg)
    r.stopped_at = None
    return r
