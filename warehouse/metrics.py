"""Statistics helpers for the experiments: bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Mean and percentile-bootstrap ``1 - alpha`` interval.  ``(nan, nan, nan)``
    for an empty sample."""
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_diff_ci(a, b, **kw):
    """CI of the mean of ``a - b`` over paired samples (same seeds)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return bootstrap_ci([x - y for x, y in pairs], **kw)


def fmt_ci(mean: float, lo: float, hi: float, digits: int = 2) -> str:
    if mean != mean:
        return "n/a"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
