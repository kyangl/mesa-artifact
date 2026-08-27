"""Nested grouped cross-validation for MESA-Learned.

Whole scenario-model-topology configurations stay within folds. Metrics are
computed per held-out configuration, macro-averaged, and resampled at the
configuration level. Coverage-AUC is the primary metric.
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.build_feature_matrix import (
    BINARY_FEATURES, DYNAMIC_FEATURES, MAS_FEATURES, NS_V1,
    STRUCTURAL_FEATURES, load_matrix, rank_normalize,
)
from src.saliency.learned import MesaLearned

C_GRID = np.logspace(-4, 4, 17)

BLOCKS = {
    "structural": list(STRUCTURAL_FEATURES),
    "structural_dynamic": list(STRUCTURAL_FEATURES) + list(DYNAMIC_FEATURES),
    "structural_dynamic_mas": (list(STRUCTURAL_FEATURES)
                               + list(DYNAMIC_FEATURES) + list(MAS_FEATURES)),
}

# Legacy label-free score: signed rank sum, the published unweighted composite.
LEGACY_SIGNS = {
    "betweenness_centrality": +1,
    "information_bottleneck": +1,
    "is_bridge": +1,
    "endpoint_centrality_max": -1,
    "source_degree_centrality": -1,
    "target_degree_centrality": -1,
}


def _auc_from_order(y, order):
    cum = np.cumsum(y[order]) / y.sum()
    cum = np.concatenate([[0.0], cum])
    x = np.linspace(0.0, 1.0, len(cum))
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trap(cum, x))


def expected_tie_credit(y_success, scores):
    """Return score-ordered labels with exact mean credit within tie groups."""
    y = np.asarray(y_success, dtype=float)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    out = ys.astype(float).copy()
    start = 0
    for i in range(1, len(ss) + 1):
        # NaN never equals itself, so a NaN score forms its own group; that is
        # the conservative reading of "unscored".
        if i == len(ss) or not (ss[i] == ss[start]):
            out[start:i] = ys[start:i].mean()
            start = i
    return out


def coverage_auc(y_success, scores):
    """Area under the coverage curve, with exact expected tie credit.

    A constant predictor returns exactly 0.5, deterministically.
    """
    y = np.asarray(y_success, dtype=float)
    if y.sum() <= 0:
        return float("nan")
    credited = expected_tie_credit(y, scores)
    n = len(credited)
    total = float(y.sum())
    # Weighted-sum form avoids cumulative rounding error at the 0.5 baseline.
    weighted = math.fsum(float(c) * (n - j) for j, c in enumerate(credited))
    return (weighted / total - 0.5) / n


def coverage_at_k(y_success, scores, frac):
    """Coverage at a budget, exact when the cutoff falls inside a tied group.

    A cutoff landing mid-group takes a fraction of that group's positives in
    expectation, which the group mean gives directly.
    """
    y = np.asarray(y_success, dtype=float)
    if y.sum() <= 0:
        return float("nan")
    k = int(math.ceil(frac * len(y)))
    k = max(1, min(k, len(y)))
    return float(expected_tie_credit(y, scores)[:k].sum() / y.sum())


def ndcg_at_k(y_success, scores, k=None):
    y = np.asarray(y_success, dtype=float)
    s = np.asarray(scores, dtype=float)
    if y.sum() <= 0:
        return float("nan")
    k = k or len(s)
    rng = np.random.default_rng(0)
    order = (np.argsort(-s, kind="stable") if len(np.unique(s)) == len(s)
             else np.lexsort((rng.permutation(len(s)), -s)))
    gains = y[order][:k]
    disc = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    ideal = np.sort(y)[::-1][:k]
    idcg = float((ideal * disc[:len(ideal)]).sum())
    return float((gains * disc).sum() / idcg) if idcg > 0 else float("nan")


def brier(y_success, n_trials, scores):
    y = np.asarray(y_success, dtype=float)
    n = np.asarray(n_trials, dtype=float)
    p = np.asarray(scores, dtype=float)
    obs = np.divide(y, np.maximum(n, 1.0))
    return float(np.mean((p - obs) ** 2))


def model_family(model: str) -> str:
    return model.split(":")[0].split("3.")[0].rstrip("0123456789.-") or model


def _row_key(row, axis):
    if axis == "topology":
        return row["topology"]
    if axis == "domain":
        return row["scenario"]
    if axis == "model_family":
        return model_family(row["model"])
    raise ValueError("unknown axis %r" % axis)


def outer_folds(groups, rows, axis) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
    levels = sorted({_row_key(r, axis) for r in rows})
    keys = np.array([_row_key(r, axis) for r in rows])
    for level in levels:
        test = np.where(keys == level)[0]
        train = np.where(keys != level)[0]
        if len(train) == 0 or len(test) == 0:
            continue
        yield train, test, level


def _grouped_kfold(groups, n_splits):
    uniq = list(dict.fromkeys(groups))
    n_splits = min(n_splits, len(uniq))
    if n_splits < 2:
        return []
    folds = [[] for _ in range(n_splits)]
    for i, g in enumerate(uniq):
        folds[i % n_splits].append(g)
    out = []
    for f in folds:
        te = np.where(np.isin(groups, f))[0]
        tr = np.where(~np.isin(groups, f))[0]
        if len(te) and len(tr):
            out.append((tr, te))
    return out


def _macro_coverage(y, n, groups, scores):
    vals = []
    for g in np.unique(groups):
        idx = groups == g
        v = coverage_auc(y[idx], scores[idx])
        if not np.isnan(v):
            vals.append(v)
    return vals


def tune_C(X, y, n, groups, grid=C_GRID, binary_cols=None):
    """Inner tuning of C by grouped coverage-AUC under the 1-SE rule."""
    folds = _grouped_kfold(groups, 5)
    if not folds:
        return float(grid[len(grid) // 2])
    means, ses = [], []
    for C in grid:
        vals = []
        for tr, te in folds:
            Xtr = rank_normalize(X[tr], groups[tr], binary_cols)
            Xte = rank_normalize(X[te], groups[te], binary_cols)
            try:
                m = MesaLearned(C=C).fit(Xtr, y[tr], n[tr], groups[tr])
                vals.extend(_macro_coverage(y[te], n[te], groups[te],
                                            m.predict_proba(Xte)))
            except Exception:
                continue
        if vals:
            means.append(float(np.mean(vals)))
            ses.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
                       if len(vals) > 1 else 0.0)
        else:
            means.append(float("-inf"))
            ses.append(0.0)
    best = int(np.argmax(means))
    threshold = means[best] - ses[best]
    for i, m in enumerate(means):
        if m >= threshold:
            return float(grid[i])
    return float(grid[best])


def legacy_signed_rank_sum(fm, normalize="global"):
    """Published label-free composite, on the effective graph.

    ``normalize="global"`` ranks each feature across all edges, which is what
    the published cross-topology analysis does and what reproduces its
    per-model rho. Normalising within a configuration instead destroys the
    cross-topology contrast the composite is built on -- pooled rho collapses
    from +0.32 to +0.02 -- so it is available only for deliberate
    within-graph comparisons.
    """
    cols = [fm.feature_names.index(f) for f in LEGACY_SIGNS]
    signs = np.array([LEGACY_SIGNS[f] for f in LEGACY_SIGNS], dtype=float)
    binary = [i for i, c in enumerate(cols)
              if fm.feature_names[c] in BINARY_FEATURES]
    if normalize == "global":
        groups = np.zeros(len(fm.rows), dtype=int)
    elif normalize == "within":
        groups = fm.groups
    else:
        raise ValueError("normalize must be 'global' or 'within'")
    R = np.nan_to_num(rank_normalize(fm.X[:, cols], groups, binary), nan=0.5)
    return (R * signs).sum(axis=1)


def pooled_spearman(fm, scores):
    """Pooled rho against observed ASR -- the paper's headline metric."""
    asr = fm.y_success / np.maximum(fm.n_trials, 1)
    r, p = stats.spearmanr(scores, asr)
    return float(r), float(p)


def score_fixed(fm, scores, axis):
    """Evaluate a fixed (unfitted) score, for label-free baselines."""
    per_config = collections.defaultdict(dict)
    for g in np.unique(fm.groups):
        idx = fm.groups == g
        per_config[g] = {
            "coverage_auc": coverage_auc(fm.y_success[idx], scores[idx]),
            "coverage_at_10": coverage_at_k(fm.y_success[idx], scores[idx], 0.10),
            "coverage_at_20": coverage_at_k(fm.y_success[idx], scores[idx], 0.20),
            "ndcg": ndcg_at_k(fm.y_success[idx], scores[idx]),
        }
    return _summarize(per_config, chosen_C=None)


def _summarize(per_config, chosen_C):
    def macro(key):
        vals = [v[key] for v in per_config.values() if not np.isnan(v[key])]
        return float(np.mean(vals)) if vals else float("nan")
    n_zero = sum(1 for v in per_config.values()
                 if np.isnan(v.get("coverage_auc", float("nan"))))
    cov = [v["coverage_auc"] for v in per_config.values()
           if not np.isnan(v["coverage_auc"])]
    lo, hi = grouped_bootstrap(cov) if cov else (float("nan"), float("nan"))
    return {
        "macro_coverage_auc": macro("coverage_auc"),
        "coverage_at_10": macro("coverage_at_10"),
        "coverage_at_20": macro("coverage_at_20"),
        "ndcg": macro("ndcg"),
        "coverage_auc_ci": [lo, hi],
        "n_config": len(per_config),
        "n_zero_success": n_zero,
        "chosen_C": chosen_C,
    }


def grouped_bootstrap(values, n=2000, seed=0):
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_cv(fm, feature_cols, axis):
    idx_cols = [fm.feature_names.index(f) for f in feature_cols]
    binary = [i for i, c in enumerate(idx_cols)
              if fm.feature_names[c] in BINARY_FEATURES]
    X = fm.X[:, idx_cols]
    if np.any(np.all(np.isnan(X), axis=0)):
        missing = [feature_cols[i] for i in range(X.shape[1])
                   if np.all(np.isnan(X[:, i]))]
        raise ValueError(
            "refusing to fit: features are entirely missing and must not be "
            "imputed: %s" % missing)

    per_config, chosen = {}, {}
    for tr, te, level in outer_folds(fm.groups, fm.rows, axis):
        C = tune_C(X[tr], fm.y_success[tr], fm.n_trials[tr], fm.groups[tr],
                   binary_cols=binary)
        chosen[level] = C
        Xtr = rank_normalize(X[tr], fm.groups[tr], binary)
        Xte = rank_normalize(X[te], fm.groups[te], binary)
        m = MesaLearned(C=C).fit(Xtr, fm.y_success[tr], fm.n_trials[tr],
                                 fm.groups[tr])
        p = m.predict_proba(Xte)
        gte = fm.groups[te]
        for g in np.unique(gte):
            sel = gte == g
            per_config[g] = {
                "coverage_auc": coverage_auc(fm.y_success[te][sel], p[sel]),
                "coverage_at_10": coverage_at_k(fm.y_success[te][sel], p[sel], 0.10),
                "coverage_at_20": coverage_at_k(fm.y_success[te][sel], p[sel], 0.20),
                "ndcg": ndcg_at_k(fm.y_success[te][sel], p[sel]),
                "brier": brier(fm.y_success[te][sel], fm.n_trials[te][sel], p[sel]),
            }
    return _summarize(per_config, chosen)


def best_single_feature(fm, axis, candidates=None):
    candidates = candidates or [f for f in STRUCTURAL_FEATURES
                                if f in fm.complete_features()]
    best, best_val = None, -np.inf
    for f in candidates:
        try:
            r = run_cv(fm, [f], axis)
        except Exception:
            continue
        if not np.isnan(r["macro_coverage_auc"]) and \
                r["macro_coverage_auc"] > best_val:
            best, best_val = f, r["macro_coverage_auc"]
    return best


def run_blocks(fm, axis):
    out = {}
    complete = set(fm.complete_features())
    for name, cols in BLOCKS.items():
        if not set(cols).issubset(complete):
            out[name] = {"skipped": "missing features: %s"
                         % sorted(set(cols) - complete)}
            continue
        out[name] = run_cv(fm, cols, axis)
    bsf = best_single_feature(fm, axis)
    if bsf:
        out["best_single"] = run_cv(fm, [bsf], axis)
        out["best_single"]["feature"] = bsf
    out["legacy"] = score_fixed(fm, legacy_signed_rank_sum(fm), axis)
    return out


def leave_one_feature_out(fm, block_name, axis):
    cols = BLOCKS[block_name]
    complete = set(fm.complete_features())
    if not set(cols).issubset(complete):
        return {}
    full = run_cv(fm, cols, axis)["macro_coverage_auc"]
    out = {}
    for f in cols:
        rest = [c for c in cols if c != f]
        out[f] = full - run_cv(fm, rest, axis)["macro_coverage_auc"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="all",
                    choices=["topology", "domain", "model_family", "all"])
    ap.add_argument("--out", default=str(REPO / "data"
                                         / "nested_cv_diagnostic.json"))
    args = ap.parse_args()

    fm = load_matrix(scenarios=["customer_service", "software_engineering"])
    has_v1 = any(r["routing_namespace"] == NS_V1 for r in fm.rows)
    status = "PAPER" if has_v1 else "DIAGNOSTIC"

    print("MESA-Learned nested grouped CV -- %s" % status)
    print("  rows=%d configs=%d topologies=%s"
          % (len(fm.rows), len(set(fm.groups)),
             sorted(set(r["topology"] for r in fm.rows))))
    print("  missing features (never imputed): %s" % fm.missing_features())
    if not has_v1:
        print("  NOTE: no routing_v1 rows yet. These numbers are diagnostic "
              "only and must not be reported as paper statistics.")

    axes = (["topology", "domain", "model_family"]
            if args.axis == "all" else [args.axis])
    report = {"status": status, "axes": {},
              "missing_features": fm.missing_features(),
              "topologies": sorted(set(r["topology"] for r in fm.rows)),
              "n_rows": len(fm.rows), "n_configs": len(set(fm.groups))}
    for axis in axes:
        blocks = run_blocks(fm, axis)
        report["axes"][axis] = {
            "blocks": blocks,
            "lofo_structural_dynamic": leave_one_feature_out(
                fm, "structural_dynamic", axis),
        }
        print("\n  axis = %s" % axis)
        for name, res in blocks.items():
            if "skipped" in res:
                print("    %-24s SKIPPED (%s)" % (name, res["skipped"]))
                continue
            print("    %-24s covAUC=%.3f  cov@20=%.3f  CI=[%.3f, %.3f]  n=%d"
                  % (name, res["macro_coverage_auc"], res["coverage_at_20"],
                     res["coverage_auc_ci"][0], res["coverage_auc_ci"][1],
                     res["n_config"]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
