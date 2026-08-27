"""Directional task cross-fitting for the MESA feature matrix.

Task-derived features from fold A score outcomes on fold B, and vice versa.
Each direction is normalized separately, missing fold estimates are dropped,
directions receive equal weight, and both stay in the same outer split.
Graph-only features are shared across directions.
"""

import collections
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

from analysis.build_feature_matrix import FeatureMatrix

# feature_fold, outcome_fold
DIRECTIONS = (("A", "B"), ("B", "A"))

# Task-derived features carry a per-fold estimate. Everything else is
# graph-only and identical in both directions.
FOLD_SPLIT_FEATURES = {
    "ablation_delta": "ablation_delta_fold_%s",
    "perturbation_delta": "perturbation_delta_fold_%s",
    "receiver_response_sensitivity": "rs_fold_%s",
}


def direction_label(feature_fold, outcome_fold):
    return "%s->%s" % (feature_fold, outcome_fold)


def config_key(record):
    return "%s|%s|%s" % (record["scenario"], record["model"],
                         record["topology"])


def group_label(record, feature_fold, outcome_fold):
    """Configuration-direction: the normalization and scoring unit."""
    return "%s|%s" % (config_key(record),
                      direction_label(feature_fold, outcome_fold))


def edge_key(record):
    return (record["scenario"], record["model"], record["topology"],
            record["edge_src"], record["edge_dst"])


# ------------------------------------------------------------------- outcomes

def fold_outcomes_from_ledger(path=None, only_ranking=True):
    """{(scenario, model, topology, src, dst, fold): (successes, trials)}.

    Read from the canonical ledger so the cross-fitted run counts exactly the
    trials every other reported number counts. The ledger already carries the
    frozen `task_fold` label per trial; it is never re-derived here, and never
    derived from outcomes.
    """
    path = Path(path or REPO / "data" / "trial_ledger.json")
    payload = json.loads(path.read_text())
    acc = collections.defaultdict(lambda: [0, 0])
    for r in payload.get("rows", []):
        fold = r.get("task_fold")
        if fold not in ("A", "B"):
            continue
        if only_ranking and not r.get("enters_ranking"):
            continue
        edge = r.get("edge") or ""
        if "->" not in edge:
            continue
        src, dst = edge.split("->", 1)
        key = (r["scenario"], r["model"], r["topology"], src, dst, fold)
        acc[key][1] += 1
        if r.get("attacked_incorrect"):
            acc[key][0] += 1
    return {k: (v[0], v[1]) for k, v in acc.items()}


# --------------------------------------------------------- directional matrix

def _feature_value(record, name, fold, cross_fitted):
    """The value of `name` for this direction, or None if unavailable.

    Returning None drops the edge from the direction. That is deliberate: the
    pooled value is right there in the record and using it would silently
    reintroduce the leak this module exists to prevent.
    """
    if not cross_fitted or name not in FOLD_SPLIT_FEATURES:
        return record.get(name)
    return record.get(FOLD_SPLIT_FEATURES[name] % fold)


def _outcome(outcomes, record, outcome_fold):
    """Outcome counts for this direction; both folds summed when pooled."""
    if outcome_fold != "pooled":
        return outcomes.get(edge_key(record) + (outcome_fold,))
    k = n = 0
    seen = False
    for fold in ("A", "B"):
        got = outcomes.get(edge_key(record) + (fold,))
        if got is not None:
            seen = True
            k, n = k + got[0], n + got[1]
    return (k, n) if seen else None


def build_directional_matrices(records, outcomes, feature_names,
                               cross_fitted=True, min_edges=2):
    """One FeatureMatrix per direction. Returns [((f_fold, o_fold), fm), ...].

    `records` are per (configuration, edge) feature dicts; `outcomes` maps
    (scenario, model, topology, src, dst, fold) -> (successes, trials).

    cross_fitted=False reproduces the legacy path exactly -- pooled features
    scored against pooled outcomes, one group per configuration -- so the two
    arms differ in the way the pipeline actually differed, and the comparison
    measures the leak rather than a diluted shadow of it.
    """
    out = []
    directions = DIRECTIONS if cross_fitted else (("pooled", "pooled"),)
    for feature_fold, outcome_fold in directions:
        X, ys, ns, groups, rows = [], [], [], [], []
        for rec in records:
            vals = [_feature_value(rec, n, feature_fold, cross_fitted)
                    for n in feature_names]
            if any(v is None or (isinstance(v, float) and np.isnan(v))
                   for v in vals):
                continue
            got = _outcome(outcomes, rec, outcome_fold)
            if got is None or got[1] == 0:
                continue
            row = dict(rec)
            row["feature_fold"] = feature_fold
            row["outcome_fold"] = outcome_fold
            row["direction"] = direction_label(feature_fold, outcome_fold)
            row["config"] = config_key(rec)
            rows.append(row)
            X.append([float(v) for v in vals])
            ys.append(int(got[0]))
            ns.append(int(got[1]))
            groups.append(group_label(rec, feature_fold, outcome_fold))

        keep = [i for i, g in enumerate(groups)
                if groups.count(g) >= min_edges]
        if len(keep) != len(groups):
            X = [X[i] for i in keep]
            ys = [ys[i] for i in keep]
            ns = [ns[i] for i in keep]
            rows = [rows[i] for i in keep]
            groups = [groups[i] for i in keep]

        fm = FeatureMatrix(
            X=(np.array(X, dtype=float) if X
               else np.zeros((0, len(feature_names)))),
            y_success=np.array(ys, dtype=int),
            n_trials=np.array(ns, dtype=int),
            groups=np.array(groups),
            feature_names=list(feature_names),
            rows=rows)
        out.append(((feature_fold, outcome_fold), fm))
    return out


# -------------------------------------------------------------- weighting

def directional_weights(per_group):
    """Half weight per direction so each configuration totals 1.

    Without this a configuration whose two directions both survive would
    outvote one that lost a direction to a missing feature, purely because of
    data availability rather than anything about the graph.
    """
    by_config = collections.Counter(g.rsplit("|", 1)[0] for g in per_group)
    return {g: 1.0 / by_config[g.rsplit("|", 1)[0]] for g in per_group}


def macro_average(per_group, weights, with_denominator=False):
    """Weighted mean over configuration-directions, skipping NaN.

    NaN means "this direction has no ranking to be right about" -- almost
    always zero attack successes. Such a direction is excluded from the
    average rather than scored 0.5, and the denominator is reported so the
    exclusion is visible instead of implicit.
    """
    used = [(per_group[g], weights[g]) for g in per_group
            if not (per_group[g] is None or np.isnan(per_group[g]))]
    total_w = sum(w for _, w in used)
    value = (sum(v * w for v, w in used) / total_w) if total_w else float("nan")
    if with_denominator:
        return float(value), len(used), len(per_group)
    return float(value)
