"""MESA Local, Reference, and Learned edge-risk scores.

All variants share the effective-graph feature matrix. Local uses within-
configuration midranks and equal weights; Reference uses a training-only CDF;
Learned fits ridge aggregation on training configurations. Features use no
attack outcomes.
"""

import numpy as np

from src.saliency.normalization import (
    GLOBAL_POOL, TRAINING_REFERENCE, WITHIN, _midrank_unit,
    apply_reference_cdf, fit_reference_cdf, normalize_global_pool,
    normalize_within,
)

# +1 vulnerability-increasing (use z), -1 protective/redundancy (use 1 - z).
FEATURE_SIGNS = {
    "betweenness_centrality": +1,
    "information_bottleneck": +1,
    "is_bridge": +1,
    "endpoint_centrality_max": -1,
    "source_degree_centrality": -1,
    "target_degree_centrality": -1,
    "ablation_delta": +1,
    "perturbation_delta": +1,
    "semantic_non_recoverability": +1,
    "receiver_response_sensitivity": +1,
    "consequence_proximity": +1,
}


def direction_align(Z, feature_names):
    """Map normalized features so that larger always means more vulnerable.

        a_f(e) = z_f(e)        if s_f = +1   (risk-increasing)
        a_f(e) = 1 - z_f(e)    if s_f = -1   (protective)

    Reflection, not negation: every aligned feature stays on [0, 1], so the
    equal average MESA-Local(e) = (1/|F|) * sum_f a_f(e) is on that scale too.
    """
    A = np.array(Z, dtype=float, copy=True)
    for j, name in enumerate(feature_names):
        if FEATURE_SIGNS.get(name, +1) < 0:
            A[:, j] = 1.0 - A[:, j]
    return A


def _equal_average(A, complete_case=True):
    """Average aligned features; incomplete rows are unscored by default."""
    A = np.asarray(A, dtype=float)
    if not complete_case:
        with np.errstate(invalid="ignore"):
            return np.nanmean(A, axis=1)
    out = np.full(A.shape[0], np.nan, dtype=float)
    ok = ~np.isnan(A).any(axis=1)
    if ok.any():
        out[ok] = A[ok].mean(axis=1)
    return out


def completeness_report(X, feature_names):
    """Which edges are scoreable, and which declared feature blocks them."""
    X = np.asarray(X, dtype=float)
    miss = np.isnan(X)
    return {
        "n_rows": int(X.shape[0]),
        "n_complete": int((~miss.any(axis=1)).sum()),
        "n_incomplete": int(miss.any(axis=1).sum()),
        "missing_by_feature": {name: int(miss[:, j].sum())
                               for j, name in enumerate(feature_names)
                               if miss[:, j].any()},
    }


class MesaLocal:
    """Attack-outcome-free within-configuration score.

    Uses no vulnerable-edge labels, no attacked configurations, and fits no
    parameters. Not free of all supervision: the dynamic deltas are measured
    on clean calibration tasks with the domain task evaluator.
    """

    name = "mesa_local"

    def __init__(self, feature_names, binary_cols=None):
        self.feature_names = list(feature_names)
        self.binary_cols = binary_cols or []

    def score(self, X, groups):
        Z = normalize_within(X, groups, self.binary_cols)
        return _equal_average(direction_align(Z, self.feature_names))


class MesaReference:
    """Equal average over a frozen, configuration-balanced training CDF."""

    name = "mesa_reference"

    def __init__(self, feature_names, binary_cols=None):
        self.feature_names = list(feature_names)
        self.binary_cols = binary_cols or []
        self.reference_ = None

    def fit(self, X, groups):
        """Estimate the reference CDF. Pass TRAINING rows only."""
        self.reference_ = fit_reference_cdf(X, groups, self.binary_cols)
        return self

    def score(self, X, groups=None):
        if self.reference_ is None:
            raise RuntimeError("fit() on training configurations first")
        Z = apply_reference_cdf(X, self.reference_)
        return _equal_average(direction_align(Z, self.feature_names))


def center_within_configuration(values, groups):
    """Subtract each configuration's mean.

    Fitting against within-configuration centered ASR stops the model from
    learning merely that one topology is globally more attackable, which would
    not transfer to an unseen topology.
    """
    v = np.asarray(values, dtype=float)
    out = np.array(v, dtype=float, copy=True)
    for g in np.unique(groups):
        idx = groups == g
        out[idx] = v[idx] - np.nanmean(v[idx])
    return out


# ---------------------------------------------------- construct-balanced Local

# Construct groups, recorded because "common social-network features" motivates
# considering them but does not establish independence. Note especially that
# endpoint_centrality_max is DERIVED as max(source_degree, target_degree), so
# the connectivity triple contains a near-duplicate by construction.
CONSTRUCT_GROUPS = {
    "graph_bottleneck": ["betweenness_centrality", "information_bottleneck",
                         "is_bridge"],
    "graph_connectivity": ["endpoint_centrality_max",
                           "source_degree_centrality",
                           "target_degree_centrality"],
    "mas_end_to_end": ["ablation_delta", "perturbation_delta"],
}

BALANCED_MAS_TERMS = ["mas_end_to_end", "receiver_response_sensitivity",
                      "consequence_proximity"]


def _mean_of(aligned, feature_names, wanted):
    idx = [feature_names.index(f) for f in wanted if f in feature_names]
    if not idx:
        return None
    with np.errstate(invalid="ignore"):
        return np.nanmean(aligned[:, idx], axis=1)


def mesa_local_balanced(A, feature_names):
    """Compute the predeclared construct-balanced sensitivity score."""
    parts = {}
    for name, feats in CONSTRUCT_GROUPS.items():
        v = _mean_of(A, feature_names, feats)
        if v is not None:
            parts[name] = v

    graph_terms = [parts[k] for k in ("graph_bottleneck", "graph_connectivity")
                   if k in parts]
    mas_terms = []
    for term in BALANCED_MAS_TERMS:
        if term in parts:
            mas_terms.append(parts[term])
        elif term in feature_names:
            mas_terms.append(A[:, feature_names.index(term)])

    with np.errstate(invalid="ignore"):
        graph_score = (np.nanmean(np.vstack(graph_terms), axis=0)
                       if graph_terms else None)
        mas_score = (np.nanmean(np.vstack(mas_terms), axis=0)
                     if mas_terms else None)
        both = [v for v in (graph_score, mas_score) if v is not None]
        balanced = np.nanmean(np.vstack(both), axis=0) if both else None

    parts["graph_score"] = graph_score
    parts["mas_score"] = mas_score
    parts["mesa_local_balanced"] = balanced
    return balanced, parts


class MesaLocalBalanced:
    """MESA-Local with construct-balanced aggregation."""

    name = "mesa_local_balanced"

    def __init__(self, feature_names, binary_cols=None):
        self.feature_names = list(feature_names)
        self.binary_cols = binary_cols or []

    def score(self, X, groups):
        Z = normalize_within(X, groups, self.binary_cols)
        A = direction_align(Z, self.feature_names)
        balanced, _ = mesa_local_balanced(A, self.feature_names)
        return balanced

    def components(self, X, groups):
        """Raw aggregate scores AND tie-aware within-config ranks."""
        Z = normalize_within(X, groups, self.binary_cols)
        A = direction_align(Z, self.feature_names)
        balanced, parts = mesa_local_balanced(A, self.feature_names)
        ranks = {}
        for name, v in parts.items():
            if v is None:
                continue
            r = np.full(len(v), np.nan)
            for g in np.unique(groups):
                i = groups == g
                r[i] = _midrank_unit(v[i])
            ranks[name] = r
        return {"raw": parts, "ranks": ranks, "score": balanced}
