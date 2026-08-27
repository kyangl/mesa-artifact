"""Configuration-weighted binomial ridge for MESA-Learned.

Edge successes and failures are weighted so each configuration contributes
equally while better-estimated edges retain more within-configuration weight.
``weighting="raw"`` provides the unnormalized sensitivity check.
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


def expand_binomial(X, y_success, n_trials, groups, weighting="config"):
    """Expand per-edge counts into weighted binary rows.

    Returns (X2, y2, w2) with 2 * n_edges rows: the first block positive, the
    second negative.
    """
    X = np.asarray(X, dtype=float)
    y_success = np.asarray(y_success, dtype=float)
    n_trials = np.asarray(n_trials, dtype=float)
    groups = np.asarray(groups)

    pos = y_success
    neg = n_trials - y_success

    if weighting == "config":
        totals = {g: n_trials[groups == g].sum() for g in np.unique(groups)}
        denom = np.array([max(totals[g], 1.0) for g in groups])
        pos = pos / denom
        neg = neg / denom
    elif weighting != "raw":
        raise ValueError("weighting must be 'config' or 'raw'")

    X2 = np.vstack([X, X])
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w2 = np.concatenate([pos, neg]).astype(float)

    mean = w2.mean()
    if mean > 0:
        w2 = w2 / mean
    return X2, y2, w2


class MesaLearned:
    def __init__(self, C=1.0, weighting="config", feature_names=None):
        self.C = C
        self.weighting = weighting
        self.feature_names: Optional[List[str]] = feature_names
        self.coef_ = None
        self.intercept_ = None
        self._est = None

    def fit(self, X, y_success, n_trials, groups):
        X2, y2, w2 = expand_binomial(X, y_success, n_trials, groups,
                                     self.weighting)
        est = LogisticRegression(penalty="l2", C=self.C, solver="lbfgs",
                                 max_iter=5000)
        est.fit(X2, y2, sample_weight=w2)
        self._est = est
        self.coef_ = est.coef_[0]
        self.intercept_ = float(est.intercept_[0])
        return self

    def predict_proba(self, X):
        if self._est is None:
            raise RuntimeError("fit() or load() first")
        return self._est.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "C": self.C,
            "weighting": self.weighting,
            "feature_names": self.feature_names,
            "coef": list(map(float, self.coef_)),
            "intercept": self.intercept_,
        }, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        m = cls(C=d["C"], weighting=d["weighting"],
                feature_names=d.get("feature_names"))
        m.coef_ = np.array(d["coef"], dtype=float)
        m.intercept_ = float(d["intercept"])
        est = LogisticRegression(penalty="l2", C=m.C, solver="lbfgs")
        est.coef_ = m.coef_.reshape(1, -1)
        est.intercept_ = np.array([m.intercept_])
        est.classes_ = np.array([0, 1])
        m._est = est
        return m
