import numpy as np
import pytest

from sealedloop import GateConfig, betting_cs, gate_path, read_gate, simulate_gate
from sealedloop.confseq import CSResult
import sealedloop.gate as G


def test_gate_outcomes():
    cfg = GateConfig(alpha=0.05, budget=100, min_n=10)
    rng = np.random.default_rng(1)
    old = (rng.random(100) < 0.6).astype(float)
    new = np.maximum(old, (rng.random(100) < 0.5).astype(float))
    r = gate_path(new, old, cfg)
    assert r.decision == "accept" and r.keep and r.stopped_at is not None and r.lo > 0
    assert read_gate(new[:5], old[:5], cfg).decision == "quote"
    assert gate_path(old, old, cfg).decision == "unresolved_at_budget"
    worse = np.minimum(old, (rng.random(100) < 0.3).astype(float))
    r = gate_path(worse, old, cfg)
    assert r.decision == "reject" and not r.keep
    with pytest.raises(ValueError):
        read_gate(new[:9], old, cfg)
    with pytest.raises(ValueError):
        GateConfig(floor=0.1)


def test_gate_path_accepts_lists():
    cfg = GateConfig(alpha=0.05, budget=100, min_n=10)
    rng = np.random.default_rng(1)
    old = (rng.random(100) < 0.6).astype(float)
    new = np.maximum(old, (rng.random(100) < 0.5).astype(float))
    assert gate_path(list(new), list(old), cfg).to_dict() == gate_path(new, old, cfg).to_dict()


def test_scores_outside_range_are_refused():
    cfg = GateConfig()
    with pytest.raises(ValueError):
        read_gate([0.2, 1.3], [0.1, 0.4], cfg)
    with pytest.raises(ValueError):
        read_gate([0.2, float("nan")], [0.1, 0.4], cfg)


def test_alpha_control_and_power():
    cfg = GateConfig(alpha=0.05, budget=100, min_n=10)
    null = simulate_gate(cfg, 0.6, 0.0, 0.3, 300, 7)
    assert null["p_accept"] <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 300)
    strong = simulate_gate(cfg, 0.6, 0.3, 0.3, 200, 8)
    assert strong["p_accept"] >= 0.9


def test_coverage_of_the_sequence():
    """The interval read after every observation misses the mean in at most alpha of the runs."""
    rng = np.random.default_rng(3)
    misses, runs, mean = 0, 300, 0.1
    for _ in range(runs):
        d = rng.choice([-1.0, 0.0, 1.0], size=120, p=[0.2, 0.5, 0.3])
        cs = betting_cs(d, -1.0, 1.0, alpha=0.05, grid=401, read_at=range(1, 121))
        misses += any(not (lo <= mean <= hi) for _, lo, hi in cs.diagnostics["path"])
    assert misses / runs <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / runs)


def test_path_is_nested():
    rng = np.random.default_rng(4)
    d = rng.uniform(-1, 1, size=80)
    path = betting_cs(d, -1.0, 1.0, read_at=range(1, 81)).diagnostics["path"]
    for (_, lo0, hi0), (_, lo1, hi1) in zip(path, path[1:]):
        assert lo1 >= lo0 and hi1 <= hi0


def test_cs_empty_is_an_outcome(monkeypatch):
    """An empty confidence sequence is a recorded rollback, not a crash."""
    cfg = GateConfig(alpha=0.05, budget=100, min_n=10)
    real = G.betting_cs

    def fake(d, a, b, alpha=0.05, grid=2001, cap=0.75, truncation="wsr", read_at=None):
        r = real(d, a, b, alpha=alpha, grid=grid, cap=cap, truncation=truncation, read_at=read_at)
        n = len(d)
        if n >= 30:
            path = [(t, lo, hi) if t < 30 else (t, float("nan"), float("nan"))
                    for t, lo, hi in r.diagnostics.get("path", [])]
            return CSResult(r.lo, r.hi, n, alpha, r.method, rejected_all=True,
                            diagnostics={**r.diagnostics, "path": path})
        return r
    monkeypatch.setattr(G, "betting_cs", fake)
    old = np.zeros(100)
    new = np.zeros(100)
    r = G.gate_path(new, old, cfg)
    assert r.decision == "cs_empty" and r.stopped_at == 30 and not r.keep
