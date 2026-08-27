"""Feature normalization for MESA scoring.

``within`` uses tie-aware ranks per configuration; ``training_reference`` uses
a configuration-balanced training CDF; ``global_pool`` is retained only for
legacy reproduction. Binary values remain 0/1, constants map to 0.5, and
missing values remain missing.
"""

import numpy as np
from scipy import stats

WITHIN = "within"
TRAINING_REFERENCE = "training_reference"
GLOBAL_POOL = "global_pool"


def _midrank_unit(values):
    """Tie-aware midranks mapped to [0, 1]; constant input -> 0.5."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan, dtype=float)
    ok = ~np.isnan(v)
    n = int(ok.sum())
    if n == 0:
        return out
    if n == 1:
        out[ok] = 0.5
        return out
    vals = v[ok]
    if np.all(vals == vals[0]):
        out[ok] = 0.5           # constant feature cannot break a tie
        return out
    r = stats.rankdata(vals, method="average")   # midranks handle ties
    out[ok] = (r - 1.0) / (n - 1.0)
    return out


def normalize_within(X, groups, binary_cols=None):
    """Tie-aware midrank within each configuration. MESA-Local's transform."""
    X = np.asarray(X, dtype=float)
    binary_cols = set(binary_cols or [])
    out = np.array(X, dtype=float, copy=True)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for j in range(X.shape[1]):
            if j in binary_cols:
                continue
            out[idx, j] = _midrank_unit(X[idx, j])
    return out


def fit_reference_cdf(X, groups, binary_cols=None, n_quantiles=101):
    """Configuration-balanced reference CDF from TRAINING rows only.

    Each configuration contributes an equal-weight quantile summary of its own
    edges, and those summaries are pooled. Held-out rows must never be passed
    here.
    """
    X = np.asarray(X, dtype=float)
    binary_cols = set(binary_cols or [])
    qs = np.linspace(0.0, 1.0, n_quantiles)
    ref = {}
    for j in range(X.shape[1]):
        if j in binary_cols:
            ref[j] = None
            continue
        per_config = []
        for g in np.unique(groups):
            col = X[groups == g, j]
            col = col[~np.isnan(col)]
            if len(col) == 0:
                continue
            per_config.append(np.quantile(col, qs))
        ref[j] = np.sort(np.concatenate(per_config)) if per_config else None
    return {"ref": ref, "binary_cols": binary_cols}


def apply_reference_cdf(X, reference):
    """Transform through a frozen training reference CDF."""
    X = np.asarray(X, dtype=float)
    out = np.array(X, dtype=float, copy=True)
    ref, binary_cols = reference["ref"], reference["binary_cols"]
    for j in range(X.shape[1]):
        if j in binary_cols:
            continue
        table = ref.get(j)
        col = X[:, j]
        ok = ~np.isnan(col)
        if table is None or len(table) == 0 or ok.sum() == 0:
            out[ok, j] = 0.5
            continue
        if np.all(table == table[0]):
            out[ok, j] = 0.5
            continue
        out[ok, j] = np.searchsorted(table, col[ok], side="right") / float(len(table))
    return np.clip(out, 0.0, 1.0)


def normalize_global_pool(X, binary_cols=None):
    """Rank across the whole pool. LEGACY REPRODUCTION ONLY -- see module doc."""
    X = np.asarray(X, dtype=float)
    binary_cols = set(binary_cols or [])
    out = np.array(X, dtype=float, copy=True)
    for j in range(X.shape[1]):
        if j in binary_cols:
            continue
        out[:, j] = _midrank_unit(X[:, j])
    return out
