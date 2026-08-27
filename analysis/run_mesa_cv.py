"""Evaluate MESA variants on held-out configurations.

Metrics are computed within each configuration and macro-averaged. Reference
normalization and learned weights use training configurations only; MESA-Local
normalizes each configuration independently.

Run: ``python analysis/run_mesa_cv.py --axis all``.
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.build_feature_matrix import (
    BINARY_FEATURES, DYNAMIC_FEATURES, MAS_FEATURES, NS_V1,
    STRUCTURAL_FEATURES, load_canonical_matrix, load_matrix,
)
from analysis.run_nested_cv import (
    coverage_at_k, coverage_auc, grouped_bootstrap, model_family, ndcg_at_k,
    outer_folds,
)
from src.saliency.mesa_scores import (
    MesaLocal, MesaReference, center_within_configuration, direction_align,
)
from src.saliency.learned import MesaLearned
from src.saliency.normalization import (
    apply_reference_cdf, fit_reference_cdf, normalize_within,
)

from sklearn.linear_model import Ridge

ALPHA_GRID = np.logspace(-4, 4, 17)
C_GRID = np.logspace(-4, 4, 17)

# Predeclared blocks; dynamic-only isolates the probe contribution.
BLOCKS = {
    "structural": list(STRUCTURAL_FEATURES),
    "dynamic": list(DYNAMIC_FEATURES),
    "structural_dynamic": list(STRUCTURAL_FEATURES) + list(DYNAMIC_FEATURES),
    "structural_dynamic_mas": (list(STRUCTURAL_FEATURES)
                               + list(DYNAMIC_FEATURES) + list(MAS_FEATURES)),
}


def _binary_idx(feature_names):
    return [i for i, n in enumerate(feature_names) if n in BINARY_FEATURES]


def within_config_metrics(y, n, scores, groups):
    """Tie-aware metrics inside each configuration."""
    per = {}
    for g in np.unique(groups):
        i = groups == g
        s = np.asarray(scores)[i]
        s = np.where(np.isnan(s), np.nanmedian(s) if np.any(~np.isnan(s)) else 0.0, s)
        yi = y[i]
        rho = float("nan")
        if len(s) > 2 and len(np.unique(s)) > 1 and len(np.unique(yi)) > 1:
            rho = float(stats.spearmanr(s, yi).statistic)
        per[g] = {
            "coverage_auc": coverage_auc(yi, s),
            "coverage_at_10": coverage_at_k(yi, s, 0.10),
            "coverage_at_20": coverage_at_k(yi, s, 0.20),
            "ndcg": ndcg_at_k(yi, s),
            "spearman": rho,
        }
    return per


def summarize(per_config, extra=None):
    def macro(key):
        v = [m[key] for m in per_config.values() if not np.isnan(m[key])]
        return float(np.mean(v)) if v else float("nan")
    cov = [m["coverage_auc"] for m in per_config.values()
           if not np.isnan(m["coverage_auc"])]
    lo, hi = grouped_bootstrap(cov) if cov else (float("nan"), float("nan"))
    out = {
        "macro_coverage_auc": macro("coverage_auc"),
        "coverage_at_10": macro("coverage_at_10"),
        "coverage_at_20": macro("coverage_at_20"),
        "ndcg": macro("ndcg"),
        "mean_within_spearman": macro("spearman"),
        "coverage_auc_ci": [lo, hi],
        "n_config": len(per_config),
        "n_zero_success": sum(1 for m in per_config.values()
                              if np.isnan(m["coverage_auc"])),
    }
    out.update(extra or {})
    return out


def eval_mesa_local(fm, cols):
    """No training split needed: the score is label-free and self-normalizing."""
    idx = [fm.feature_names.index(c) for c in cols]
    names = [fm.feature_names[i] for i in idx]
    scorer = MesaLocal(names, _binary_idx(names))
    s = scorer.score(fm.X[:, idx], fm.groups)
    return summarize(within_config_metrics(fm.y_success, fm.n_trials, s,
                                           fm.groups))


def eval_mesa_reference(fm, cols, axis):
    idx = [fm.feature_names.index(c) for c in cols]
    names = [fm.feature_names[i] for i in idx]
    per = {}
    for tr, te, _ in outer_folds(fm.groups, fm.rows, axis):
        scorer = MesaReference(names, _binary_idx(names))
        scorer.fit(fm.X[np.ix_(tr, idx)], fm.groups[tr])   # training only
        s = scorer.score(fm.X[np.ix_(te, idx)])
        per.update(within_config_metrics(fm.y_success[te], fm.n_trials[te],
                                         s, fm.groups[te]))
    return summarize(per)


def _tune_alpha(X, y, groups, binary, grid=ALPHA_GRID):
    """Inner grouped CV on the ridge penalty, by macro coverage-AUC."""
    uniq = list(dict.fromkeys(groups))
    if len(uniq) < 3:
        return float(grid[len(grid) // 2])
    folds = []
    k = min(5, len(uniq))
    for f in range(k):
        held = uniq[f::k]
        te = np.where(np.isin(groups, held))[0]
        tr = np.where(~np.isin(groups, held))[0]
        if len(te) and len(tr):
            folds.append((tr, te))
    best, best_val = grid[0], -np.inf
    for a in grid:
        vals = []
        for tr, te in folds:
            ref = fit_reference_cdf(X[tr], groups[tr], binary)
            Ztr = direction_align(apply_reference_cdf(X[tr], ref), [""] * X.shape[1])
            Zte = direction_align(apply_reference_cdf(X[te], ref), [""] * X.shape[1])
            ytr = center_within_configuration(
                y[tr].astype(float), groups[tr])
            m = Ridge(alpha=a).fit(np.nan_to_num(Ztr, nan=0.5), ytr)
            p = m.predict(np.nan_to_num(Zte, nan=0.5))
            for g in np.unique(groups[te]):
                i = groups[te] == g
                v = coverage_auc(y[te][i], p[i])
                if not np.isnan(v):
                    vals.append(v)
        if vals and np.mean(vals) > best_val:
            best, best_val = a, float(np.mean(vals))
    return float(best)


def _tune_C(X, y, n, groups, binary, grid=C_GRID):
    """Inner grouped CV on the L2 penalty, by macro coverage-AUC."""
    uniq = list(dict.fromkeys(groups))
    if len(uniq) < 3:
        return float(grid[len(grid) // 2])
    k = min(5, len(uniq))
    folds = []
    for f in range(k):
        held = uniq[f::k]
        te = np.where(np.isin(groups, held))[0]
        tr = np.where(~np.isin(groups, held))[0]
        if len(te) and len(tr):
            folds.append((tr, te))
    best, best_val = float(grid[0]), -np.inf
    for C in grid:
        vals = []
        for tr, te in folds:
            Ztr = np.nan_to_num(normalize_within(X[tr], groups[tr], binary), nan=0.5)
            Zte = np.nan_to_num(normalize_within(X[te], groups[te], binary), nan=0.5)
            try:
                m = MesaLearned(C=C).fit(Ztr, y[tr], n[tr], groups[tr])
                p = m.predict_proba(Zte)
            except Exception:
                continue
            for g in np.unique(groups[te]):
                i = groups[te] == g
                v = coverage_auc(y[te][i], p[i])
                if not np.isnan(v):
                    vals.append(v)
        if vals and np.mean(vals) > best_val:
            best, best_val = float(C), float(np.mean(vals))
    return best


def eval_mesa_learned(fm, cols, axis):
    """Configuration-weighted, L2-regularized binomial logistic aggregation.

    Consumes exactly the same within-configuration local ranks as MESA-Local,
    so the only difference between the two methods is equal versus learned
    aggregation. Success/trial counts are used directly, and each
    configuration carries equal total weight, so a large graph with many
    trials cannot dominate. Normalization is within-configuration and
    therefore cannot leak across the held-out split.
    """
    idx = [fm.feature_names.index(c) for c in cols]
    names = [fm.feature_names[i] for i in idx]
    binary = _binary_idx(names)
    X = fm.X[:, idx]

    per, chosen = {}, {}
    for tr, te, level in outer_folds(fm.groups, fm.rows, axis):
        C = _tune_C(X[tr], fm.y_success[tr], fm.n_trials[tr], fm.groups[tr],
                    binary)
        chosen[level] = C
        Ztr = np.nan_to_num(normalize_within(X[tr], fm.groups[tr], binary), nan=0.5)
        Zte = np.nan_to_num(normalize_within(X[te], fm.groups[te], binary), nan=0.5)
        m = MesaLearned(C=C).fit(Ztr, fm.y_success[tr], fm.n_trials[tr],
                                 fm.groups[tr])
        p = m.predict_proba(Zte)
        per.update(within_config_metrics(fm.y_success[te], fm.n_trials[te],
                                         p, fm.groups[te]))
    return summarize(per, {"chosen_C": chosen})


def pick_best_single(fm, axis, candidates=None):
    cands = candidates or [f for f in STRUCTURAL_FEATURES + DYNAMIC_FEATURES
                           if f in fm.complete_features()]
    best, val = None, -np.inf
    for f in cands:
        r = eval_mesa_local(fm, [f])
        if not np.isnan(r["macro_coverage_auc"]) and \
                r["macro_coverage_auc"] > val:
            best, val = f, r["macro_coverage_auc"]
    return best


def run_all(fm, axis):
    complete = set(fm.complete_features())
    out = {}
    for name, cols in BLOCKS.items():
        if not set(cols).issubset(complete):
            out[name] = {"skipped": "missing: %s" % sorted(set(cols) - complete)}
            continue
        entry = {"features": cols,
                 "mesa_local": eval_mesa_local(fm, cols)}
        try:
            entry["mesa_learned"] = eval_mesa_learned(fm, cols, axis)
        except Exception as exc:
            entry["mesa_learned"] = {"error": str(exc)}
        out[name] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="all",
                    choices=["topology", "domain", "model_family", "all"])
    ap.add_argument("--include-v1", action="store_true",
                    help="Include corrected routing_v1 rows.")
    ap.add_argument("--out", default=str(REPO / "data" / "mesa_cv.json"))
    args = ap.parse_args()

    fm = load_canonical_matrix()
    has_v1 = any(r.get("routing_namespace") == NS_V1 for r in fm.rows)
    status = "PAPER" if has_v1 else "DIAGNOSTIC"

    print("MESA within-configuration evaluation -- %s" % status)
    print("  rows=%d configs=%d topologies=%s"
          % (len(fm.rows), len(set(fm.groups)),
             sorted(set(r["topology"] for r in fm.rows))))
    print("  missing (never imputed): %s" % fm.missing_features())

    axes = (["topology", "domain", "model_family"]
            if args.axis == "all" else [args.axis])
    report = {"status": status, "axes": {}}
    for axis in axes:
        res = run_all(fm, axis)
        report["axes"][axis] = res
        print("\n  axis = %s        %10s %10s"
              % (axis, "Local", "Learned"))
        for name, e in res.items():
            if "skipped" in e:
                print("    %-24s SKIPPED (%s)" % (name, e["skipped"]))
                continue
            def g(k):
                v = e.get(k, {})
                return v.get("macro_coverage_auc", float("nan"))
            print("    %-24s %10.3f %10.3f"
                  % (name, g("mesa_local"), g("mesa_learned")))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
