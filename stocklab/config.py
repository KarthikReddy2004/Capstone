"""Central configuration for Stocklab.

All tunable knobs live here so runs are reproducible and the system can be
re-targeted (hardware budgets, optimizer effort, Redis endpoint) from one place.
Values may be overridden through environment variables, which keeps secrets such
as the Redis URL out of source control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any


def _load_dotenv() -> None:
    """Load ``.env`` from the project root into the environment (no dependency).

    Existing environment variables always win, so an explicitly exported value
    overrides the file. Keeps secrets (the Redis URL) out of source while making
    ``python app.py`` "just work" without remembering to export anything.
    """
    root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except Exception:
        pass


_load_dotenv()


# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #
GLOBAL_SEED: int = int(os.environ.get("STOCKLAB_SEED", "1337"))


# --------------------------------------------------------------------------- #
# Infrastructure                                                              #
# --------------------------------------------------------------------------- #
REDIS_URL: str = os.environ.get(
    "STOCKLAB_REDIS_URL",
    # Falls back to a local server if the managed endpoint is not configured.
    os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"),
)
REDIS_NAMESPACE: str = os.environ.get("STOCKLAB_REDIS_NS", "stocklab")
# Time-to-live (seconds) for cached datasets and finished jobs.
CACHE_TTL: int = int(os.environ.get("STOCKLAB_CACHE_TTL", str(60 * 60 * 6)))
JOB_TTL: int = int(os.environ.get("STOCKLAB_JOB_TTL", str(60 * 60 * 24)))

# Worker pool size. Kept modest so the platform stays responsive on a 6-core
# i5-9400F while PyTorch and scikit-learn run their own internal threads.
MAX_WORKERS: int = int(os.environ.get("STOCKLAB_MAX_WORKERS", "2"))
# Cap intra-op threads per heavy library to avoid oversubscription on 6 cores.
TORCH_THREADS: int = int(os.environ.get("STOCKLAB_TORCH_THREADS", "4"))


# --------------------------------------------------------------------------- #
# Forecasting                                                                 #
# --------------------------------------------------------------------------- #
FORECAST_HORIZON: int = 11
SPOTLIGHT_HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 7, 11)
PREDICTION_INTERVALS: tuple[int, ...] = (80, 90, 95)


# --------------------------------------------------------------------------- #
# Feature engineering                                                         #
# --------------------------------------------------------------------------- #
# Core market-structure features that the Phase-1 selector is never allowed to
# drop. They anchor the model in fundamental price/volume dynamics.
CORE_FEATURES: tuple[str, ...] = (
    "Open", "High", "Low", "Volume",
    "LogReturn", "CloseLag1", "RetLag1",
    "SMA10", "EMA20", "RSI14", "MACD", "ATR14", "VWAPDist",
)
# Minimum usable rows after feature engineering before a run is allowed.
MIN_ROWS: int = 160


@dataclass
class OptimizerConfig:
    """Population-based optimizer effort.

    The defaults are calibrated for the target hardware: small populations and
    short horizons keep a full 16-model benchmark inside a few minutes while the
    Adaptive Dual-Phase framework is granted a deeper budget where it matters.
    """

    pop_size: int = 20
    max_iter: int = 40
    c1: float = 1.5          # PSO cognitive coefficient
    c2: float = 2.0          # PSO social coefficient
    w: float = 0.72          # PSO inertia weight

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunConfig:
    """Per-run configuration assembled from the UI / API request."""

    seq_len: int = 32
    test_pct: float = 0.2
    val_pct: float = 0.15
    n_splits: int = 3
    horizon: int = FORECAST_HORIZON
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    seed: int = GLOBAL_SEED
    # Effort multiplier ("fast" | "balanced" | "thorough") scales optimizer work.
    effort: str = "balanced"

    @classmethod
    def from_request(cls, cfg: dict[str, Any] | None) -> "RunConfig":
        cfg = cfg or {}
        opt = OptimizerConfig(
            pop_size=int(cfg.get("popSize", 20)),
            max_iter=int(cfg.get("maxIter", 40)),
            c1=float(cfg.get("c1", 1.5)),
            c2=float(cfg.get("c2", 2.0)),
            w=float(cfg.get("w", 0.72)),
        )
        test_pct = float(cfg.get("testPct", 20))
        # Accept either fraction (0.2) or percentage (20).
        if test_pct > 1:
            test_pct /= 100.0
        return cls(
            seq_len=int(cfg.get("seqLen", 32)),
            test_pct=test_pct,
            val_pct=float(cfg.get("valPct", 0.15)),
            n_splits=int(cfg.get("nSplits", 3)),
            horizon=int(cfg.get("horizon", FORECAST_HORIZON)),
            optimizer=opt,
            seed=int(cfg.get("seed", GLOBAL_SEED)),
            effort=str(cfg.get("effort", "balanced")),
        )

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["optimizer"] = self.optimizer.as_dict()
        return out
