"""Temporal split strategies.

All splitters are strictly causal: validation windows always sit *after* their
training window in time, so no future information leaks into model selection.
Each returns a list of ``(train_start, train_end, val_start, val_end)`` index
tuples, with ``train`` = ``[train_start, train_end)`` and ``val`` =
``[val_start, val_end)``.
"""

from __future__ import annotations

Fold = tuple[int, int, int, int]


def holdout_split(length: int, test_pct: float, min_test: int = 16) -> tuple[int, int]:
    """Return ``(n_train, n_test)`` for a trailing holdout test set."""
    n_test = max(min_test, int(round(length * test_pct)))
    n_test = min(n_test, max(min_test, length // 3))
    n_train = length - n_test
    return n_train, n_test


def walk_forward_splits(
    length: int,
    n_splits: int = 3,
    min_train: int = 60,
    min_val: int = 16,
) -> list[Fold]:
    """Rolling walk-forward: fixed-size validation blocks marching forward.

    The training window *expands* up to each validation block (anchored start),
    which is the standard expanding walk-forward used in time-series back-tests.
    """

    if length < min_train + min_val:
        return []
    val_size = max(min_val, int(round(length * 0.12)))
    max_folds = max(1, (length - min_train) // val_size)
    n_folds = max(1, min(int(n_splits), max_folds))
    start = max(min_train, length - n_folds * val_size)
    folds: list[Fold] = []
    for i in range(n_folds):
        train_end = start + i * val_size
        val_start = train_end
        val_end = min(length, val_start + val_size)
        if train_end >= min_train and (val_end - val_start) >= min_val:
            folds.append((0, train_end, val_start, val_end))
    return folds


def expanding_window_splits(
    length: int,
    n_splits: int = 4,
    min_train: int = 60,
    min_val: int = 16,
) -> list[Fold]:
    """Expanding-window CV: training grows from ``min_train`` to nearly full.

    Distinct from :func:`walk_forward_splits` in that folds are spread evenly
    across the series rather than clustered at the tail, giving a broader,
    lower-variance estimate of generalisation.
    """

    if length < min_train + min_val:
        return []
    val_size = max(min_val, int(round((length - min_train) / (n_splits + 1))))
    folds: list[Fold] = []
    for i in range(1, n_splits + 1):
        train_end = min_train + (i - 1) * val_size
        val_start = train_end
        val_end = min(length, val_start + val_size)
        if val_end - val_start >= min_val and train_end >= min_train:
            folds.append((0, train_end, val_start, val_end))
        if val_end >= length:
            break
    return folds


def temporal_cv_splits(
    length: int,
    n_splits: int = 5,
    min_train: int = 60,
    min_val: int = 12,
    purge: int = 0,
) -> list[Fold]:
    """Blocked temporal CV with an optional purge gap.

    A ``purge`` gap between train and validation removes look-ahead bias from
    overlapping rolling features (sequence windows). Used for fusion-weight
    calibration where leakage across the boundary would inflate scores.
    """

    if length < min_train + min_val:
        return []
    val_size = max(min_val, int(round(length * 0.14)))
    max_splits = max(1, (length - min_train) // val_size)
    n_folds = max(1, min(int(n_splits), max_splits))
    start = max(min_train, length - n_folds * val_size)
    folds: list[Fold] = []
    for i in range(n_folds):
        train_end = start + i * val_size
        val_start = min(length, train_end + purge)
        val_end = min(length, val_start + val_size)
        if train_end >= min_train and (val_end - val_start) >= min_val:
            folds.append((0, max(0, train_end - 0), val_start, val_end))
    return folds
