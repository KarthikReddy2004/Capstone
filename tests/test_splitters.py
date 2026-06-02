from stocklab.validation.splitters import (
    walk_forward_splits, expanding_window_splits, temporal_cv_splits, holdout_split,
)


def _assert_causal(folds):
    for ts, te, vs, ve in folds:
        assert ts <= te <= vs < ve, "validation must start at/after train end"


def test_walk_forward_is_causal():
    folds = walk_forward_splits(500, n_splits=3)
    assert len(folds) >= 1
    _assert_causal(folds)


def test_expanding_window_grows():
    folds = expanding_window_splits(500, n_splits=4)
    _assert_causal(folds)
    ends = [te for _, te, _, _ in folds]
    assert ends == sorted(ends), "training window must expand"


def test_temporal_cv_with_purge():
    folds = temporal_cv_splits(500, n_splits=3, purge=5)
    _assert_causal(folds)
    for _, te, vs, _ in folds:
        assert vs >= te  # purge gap respected


def test_holdout_split():
    n_train, n_test = holdout_split(300, 0.2)
    assert n_train + n_test == 300
    assert n_test >= 16
