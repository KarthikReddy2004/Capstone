import numpy as np

from stocklab.validation.metrics import (
    compute_metrics, rmse, mse, r2_score, arv, theil_u,
    directional_accuracy, tracking_signal,
)


def test_perfect_prediction():
    y = np.array([1.0, 2, 3, 4, 5])
    m = compute_metrics(y, y, prev=np.array([0.5, 1, 2, 3, 4]))
    assert m["rmse"] == 0.0
    assert m["mse"] == 0.0
    assert m["mae"] == 0.0
    assert abs(m["r2"] - 1.0) < 1e-9
    assert m["theil_u"] == 0.0
    assert m["arv"] == 0.0
    assert m["dir_acc"] == 100.0
    assert m["hit_ratio"] == 100.0


def test_mse_is_rmse_squared():
    y = np.array([10.0, 12, 14])
    p = np.array([11.0, 11, 15])
    assert abs(mse(y, p) - rmse(y, p) ** 2) < 1e-9


def test_arv_is_one_minus_r2():
    y = np.array([10.0, 12, 11, 15, 14])
    p = np.array([10.5, 11.5, 11.2, 14, 14.5])
    assert abs(arv(y, p) - (1.0 - r2_score(y, p))) < 1e-6


def test_theil_u_naive_baseline():
    # A pure no-change (random-walk) forecast equals the naive baseline -> U2 ≈ 1.
    prev = np.array([10.0, 11, 12, 11])
    y = np.array([11.0, 12, 11, 13])
    naive = prev.copy()                  # predict "no change" from the anchor
    assert abs(theil_u(y, naive, prev) - 1.0) < 1e-6
    # A near-perfect forecast beats the baseline -> U2 well below 1.
    good = y - 0.01
    assert theil_u(y, good, prev) < 1.0


def test_rmse_and_r2_known_values():
    y = np.array([10.0, 12, 14])
    p = np.array([11.0, 11, 15])
    assert abs(rmse(y, p) - 1.0) < 1e-9
    assert r2_score(y, y) == 1.0


def test_directional_accuracy_with_anchor():
    prev = np.array([10.0, 10, 10])
    y = np.array([11.0, 9, 12])       # up, down, up
    p = np.array([10.5, 9.5, 11])     # up, down, up -> 100%
    assert directional_accuracy(y, p, prev) == 100.0
    p2 = np.array([9.0, 9.5, 11])     # down, down, up -> 2/3
    assert abs(directional_accuracy(y, p2, prev) - (200.0 / 3)) < 1e-6


def test_tracking_signal_sign():
    y = np.array([10.0, 10, 10, 10])
    p = np.array([11.0, 11, 11, 11])  # always over-predict -> positive TS near +4
    assert tracking_signal(y, p) > 3.5


def test_empty_safe():
    m = compute_metrics(np.array([]), np.array([]))
    assert m["rmse"] == 0.0
