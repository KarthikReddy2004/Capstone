import numpy as np

from stocklab.optimization import (
    pso_optimize, gwo_optimize, hybrid_pso_gwo_optimize, adaptive_binary_hybrid, local_refine,
)

CFG = {"pop_size": 14, "max_iter": 25, "c1": 1.5, "c2": 2.0, "w": 0.72}


def _sphere(shift=1.3):
    return lambda v: float(np.sum((np.asarray(v) - shift) ** 2))


def test_pso_converges_with_telemetry():
    r = pso_optimize([(-5, 5)] * 5, _sphere(), CFG, seed=1)
    t = r["telemetry"]
    assert r["fitness"] < 0.5
    assert t["best"][-1] <= t["best"][0]
    assert len(t["best"]) == len(t["mean"]) == len(t["worst"])
    # exploration should fall as the swarm contracts
    assert t["exploration"][-1] <= t["exploration"][0]


def test_gwo_converges():
    r = gwo_optimize([(-5, 5)] * 5, _sphere(), CFG, seed=1)
    assert r["fitness"] < 0.5
    assert len(r["telemetry"]["a"]) >= 1


def test_hybrid_adaptive_lambda_schedule():
    r = hybrid_pso_gwo_optimize([(-5, 5)] * 5, _sphere(), CFG, seed=1, adaptive=True)
    lam = r["telemetry"]["lam"]
    assert lam[0] > 0.9 and lam[-1] < 0.1  # 1 - (t/T)^2 goes 1 -> 0
    assert r["fitness"] < 0.5


def test_binary_selection_locks_core():
    def obj(mask):
        sel = np.where(mask > 0.5)[0]
        return 1e9 if len(sel) < 2 else float(-sum(1 for i in sel if i % 2 == 0) + 0.05 * len(sel))
    r = adaptive_binary_hybrid(10, obj, {**CFG, "max_iter": 12}, seed=2, locked_idx=[0])
    assert r["binary"][0] == 1.0           # locked feature retained
    assert len(r["select_freq"]) == 10


def test_local_refine_improves():
    obj = _sphere()
    start = np.array([3.0] * 5)
    r = local_refine([(-5, 5)] * 5, start, obj, rounds=30, seed=7)
    assert r["fitness"] <= obj(start)
