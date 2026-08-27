"""Compare fixed and learned aggregation on the enriched feature matrix.

Both methods use the same within-configuration midranks; only aggregation
differs. Learned weights and tuning are refit inside each outer fold. F1 is
excluded because its max statistic is confounded with receiver in-degree.
"""

import argparse
import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.build_feature_matrix import (
    BINARY_FEATURES, DYNAMIC_FEATURES, FeatureMatrix, STRUCTURAL_FEATURES,
    load_canonical_matrix,
)
from analysis.crossfit import (
    build_directional_matrices, fold_outcomes_from_ledger,
)
from analysis.run_mesa_cv import _binary_idx, within_config_metrics
from analysis.run_nested_cv import coverage_auc, coverage_at_k, outer_folds
from src.saliency.learned import MesaLearned
from src.saliency.mesa_scores import MesaLocal
from src.saliency.normalization import normalize_within

C_GRID = np.logspace(-4, 4, 17)
F2 = "receiver_response_sensitivity"
F3 = "consequence_proximity"

BLOCKS = {
    "structural": list(STRUCTURAL_FEATURES),
    "dynamic": list(DYNAMIC_FEATURES),
    "structural_dynamic": list(STRUCTURAL_FEATURES) + list(DYNAMIC_FEATURES),
    "structural_dynamic_f2": list(STRUCTURAL_FEATURES) + list(DYNAMIC_FEATURES) + [F2],
    "structural_dynamic_f3": list(STRUCTURAL_FEATURES) + list(DYNAMIC_FEATURES) + [F3],
    "structural_dynamic_f2_f3": (list(STRUCTURAL_FEATURES)
                                 + list(DYNAMIC_FEATURES) + [F2, F3]),
}
# Frozen by the predeclared simplest-within-one-SE rule.
CANONICAL_BLOCK = "structural_dynamic"

REFERENCE_BLOCK = "structural_dynamic"


# ------------------------------------------------------------------ enrichment

def load_enriched(mas_path=None, delta_path=None, scenarios=None):
    """Canonical matrix + measured deltas + F2/F3, keyed by (config, edge).

    Additive: the canonical loader is untouched and every existing column
    keeps its meaning. Only edges with a complete feature vector are returned,
    because averaging different feature subsets across edges would compare
    edges on different quantities.
    """
    mas_path = Path(mas_path or REPO / "data" / "mas_features.json")
    delta_path = Path(delta_path or REPO / "data" / "dynamic_deltas.json")

    mas = {}
    for r in json.loads(mas_path.read_text())["records"]:
        mas[(r["scenario"], r["topology"], r["model"],
             r["edge_src"], r["edge_dst"])] = r

    deltas = collections.defaultdict(dict)
    payload = json.loads(delta_path.read_text())["configurations"]
    for key, block in payload.items():
        mode, scenario, model, topology = key.split("|")
        for edge_str, v in block["edges"].items():
            src, dst = edge_str.split("->")
            deltas[(scenario, topology, model, src, dst)][mode] = v["delta"]

    fm = load_canonical_matrix()
    names = list(fm.feature_names)
    keep_names = [n for n in names if n in STRUCTURAL_FEATURES] + \
                 list(DYNAMIC_FEATURES) + [F2, F3]
    si = {n: names.index(n) for n in STRUCTURAL_FEATURES}

    rows, X, ys, ns, groups = [], [], [], [], []
    dropped = collections.Counter()
    for i, r in enumerate(fm.rows):
        if scenarios and r["scenario"] not in scenarios:
            continue
        key = (r["scenario"], r["topology"], r["model"],
               r["edge_src"], r["edge_dst"])
        m, d = mas.get(key), deltas.get(key)
        if m is None:
            dropped["no_mas_features"] += 1
            continue
        if not d or "ablation" not in d or "perturbation" not in d:
            dropped["no_dynamic_deltas"] += 1
            continue
        if m.get(F2) is None or m.get(F3) is None:
            dropped["incomplete_mas"] += 1
            continue
        vec = [fm.X[i, si[n]] for n in STRUCTURAL_FEATURES]
        vec += [d["ablation"], d["perturbation"], m[F2], m[F3]]
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vec):
            dropped["nan_in_vector"] += 1
            continue
        rows.append(dict(r))
        X.append([float(v) for v in vec])
        ys.append(int(fm.y_success[i]))
        ns.append(int(fm.n_trials[i]))
        groups.append(fm.groups[i])

    return FeatureMatrix(
        X=np.array(X, dtype=float) if X else np.zeros((0, len(keep_names))),
        y_success=np.array(ys, dtype=int), n_trials=np.array(ns, dtype=int),
        groups=np.array(groups), feature_names=keep_names, rows=rows), dict(dropped)


# --------------------------------------------------------- cross-fitted load

def load_enriched_directional(mas_path=None, delta_path=None, scenarios=None,
                              ledger_path=None, cross_fitted=True):
    """Load one enriched matrix containing separate cross-fit directions.

    Task-derived columns and outcomes use opposite folds. Graph-only columns
    are shared, and group labels keep directions separate during scoring.
    """
    mas_path = Path(mas_path or REPO / "data" / "mas_features.json")
    delta_path = Path(delta_path or REPO / "data" / "dynamic_deltas.json")

    mas = {}
    for r in json.loads(mas_path.read_text())["records"]:
        mas[(r["scenario"], r["topology"], r["model"],
             r["edge_src"], r["edge_dst"])] = r

    deltas = collections.defaultdict(dict)
    payload = json.loads(delta_path.read_text())["configurations"]
    for key, block in payload.items():
        mode, scenario, model, topology = key.split("|")
        for edge_str, v in block["edges"].items():
            src, dst = edge_str.split("->")
            deltas[(scenario, topology, model, src, dst)][mode] = v

    fm = load_canonical_matrix()
    names = list(fm.feature_names)
    si = {n: names.index(n) for n in STRUCTURAL_FEATURES}
    feature_names = list(STRUCTURAL_FEATURES) + list(DYNAMIC_FEATURES) + [F2, F3]

    records, dropped = [], collections.Counter()
    for i, r in enumerate(fm.rows):
        if scenarios and r["scenario"] not in scenarios:
            continue
        key = (r["scenario"], r["topology"], r["model"],
               r["edge_src"], r["edge_dst"])
        m, d = mas.get(key), deltas.get(key)
        if m is None:
            dropped["no_mas_features"] += 1
            continue
        if not d or "ablation" not in d or "perturbation" not in d:
            dropped["no_dynamic_deltas"] += 1
            continue
        rec = dict(r)
        for n in STRUCTURAL_FEATURES:
            rec[n] = float(fm.X[i, si[n]])
        rec["consequence_proximity"] = m.get(F3)
        # pooled values, used only by the non-cross-fitted control arm
        rec["ablation_delta"] = d["ablation"].get("delta")
        rec["perturbation_delta"] = d["perturbation"].get("delta")
        rec[F2] = m.get(F2)
        for fold in ("A", "B"):
            rec["ablation_delta_fold_%s" % fold] = \
                d["ablation"].get("delta_fold_%s" % fold)
            rec["perturbation_delta_fold_%s" % fold] = \
                d["perturbation"].get("delta_fold_%s" % fold)
            rec["rs_fold_%s" % fold] = m.get("rs_fold_%s" % fold)
        records.append(rec)

    outcomes = fold_outcomes_from_ledger(ledger_path)
    mats = build_directional_matrices(records, outcomes, feature_names,
                                      cross_fitted=cross_fitted)

    X = np.vstack([fm_d.X for _, fm_d in mats if len(fm_d.rows)]) \
        if any(len(f.rows) for _, f in mats) else np.zeros((0, len(feature_names)))
    combined = FeatureMatrix(
        X=X,
        y_success=np.concatenate([f.y_success for _, f in mats]),
        n_trials=np.concatenate([f.n_trials for _, f in mats]),
        groups=np.concatenate([f.groups for _, f in mats]),
        feature_names=feature_names,
        rows=[r for _, f in mats for r in f.rows])
    # A group must never contain more than one direction, or the separation
    # this whole module exists to enforce would be undone silently.
    per_group_dirs = collections.defaultdict(set)
    for g, r in zip(combined.groups, combined.rows):
        per_group_dirs[g].add(r["direction"])
    mixed = {g: d for g, d in per_group_dirs.items() if len(d) > 1}
    assert not mixed, mixed
    return combined, dict(dropped), mats


def collapse_directions(per_group):
    """{configuration-direction: value} -> {configuration: value}.

    Each direction carries half weight so a configuration totals 1. When one
    direction is NaN -- almost always zero attack successes, which leaves no
    ranking to be right about -- the surviving direction stands alone rather
    than being averaged against a fabricated 0.5.
    """
    acc = collections.defaultdict(list)
    for g, v in per_group.items():
        acc[g.rsplit("|", 1)[0]].append(v)
    out = {}
    for config, vals in acc.items():
        good = [v for v in vals if v is not None and not np.isnan(v)]
        out[config] = float(np.mean(good)) if good else float("nan")
    return out


# ------------------------------------------------------------ redundancy audit

def redundancy_audit(fm, flag=0.80, duplicate=0.90, min_share=0.5):
    """Within-configuration pairwise Spearman, summarized across configs.

    Correlations are computed inside each configuration and summarized by
    median |rho|, never pooled: pooling would mix within-graph structure with
    between-topology differences and inflate apparent redundancy.

    Outcome-blind -- it never looks at attack labels. A feature is proposed
    for removal only when a pair measures the same construct AND exceeds the
    duplicate threshold in most evaluable configurations.
    """
    names = fm.feature_names
    out = []
    for a, b in itertools.combinations(range(len(names)), 2):
        rhos = []
        for g in np.unique(fm.groups):
            idx = fm.groups == g
            xa, xb = fm.X[idx, a], fm.X[idx, b]
            if len(xa) < 4:
                continue
            if len(np.unique(xa)) < 2 or len(np.unique(xb)) < 2:
                continue          # constant inside this graph: undefined
            r = stats.spearmanr(xa, xb).statistic
            if not np.isnan(r):
                rhos.append(abs(float(r)))
        if not rhos:
            out.append({"a": names[a], "b": names[b], "n_defined": 0,
                        "median_abs_rho": float("nan"), "flagged": False,
                        "duplicate_candidate": False})
            continue
        med = float(np.median(rhos))
        share_dup = float(np.mean([r >= duplicate for r in rhos]))
        out.append({
            "a": names[a], "b": names[b], "n_defined": len(rhos),
            "median_abs_rho": med,
            "share_ge_duplicate": share_dup,
            "flagged": med >= flag,
            # Same-construct judgement is NOT automated; this only marks the
            # statistical precondition.
            "duplicate_candidate": bool(med >= duplicate
                                        and share_dup >= min_share),
        })
    return sorted(out, key=lambda d: -(d["median_abs_rho"]
                                       if not np.isnan(d["median_abs_rho"]) else -1))


# ---------------------------------------------------------------- evaluation

def _inner_folds(groups, k=5):
    """Inner CV folds that keep a configuration whole.

    In cross-fitted mode `groups` are configuration-DIRECTIONS. Striding over
    that list put "cfg|A->B" and "cfg|B->A" in different inner folds, so C was
    tuned on one direction of a configuration and validated on the other --
    and the two directions are computed from the same 20 or 50 tasks. That is
    leakage inside the hyper-parameter search, and the outer LOTO split cannot
    see it. Folds are therefore built over configurations, and every direction
    of a configuration moves with it.
    """
    configs = list(dict.fromkeys(g.rsplit("|", 1)[0] for g in groups))
    if len(configs) < 3:
        return []
    k = min(k, len(configs))
    config_of = np.array([g.rsplit("|", 1)[0] for g in groups])
    folds = []
    for f in range(k):
        held = configs[f::k]
        te = np.where(np.isin(config_of, held))[0]
        tr = np.where(~np.isin(config_of, held))[0]
        if len(te) and len(tr):
            folds.append((tr, te))
    return folds


def _tune_C(X, y, n, groups, binary, grid=C_GRID):
    folds = _inner_folds(groups)
    if not folds:
        return float(grid[len(grid) // 2])
    best, best_val = float(grid[0]), -np.inf
    for C in grid:
        vals = []
        for tr, te in folds:
            Ztr = normalize_within(X[tr], groups[tr], binary)
            Zte = normalize_within(X[te], groups[te], binary)
            try:
                m = MesaLearned(C=C).fit(np.nan_to_num(Ztr, nan=0.5), y[tr],
                                         n[tr], groups[tr])
                p = m.predict_proba(np.nan_to_num(Zte, nan=0.5))
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


def per_config_scores(fm, cols, axis, method):
    """Per-configuration metrics. Returns {config: {metric: value}}."""
    idx = [fm.feature_names.index(c) for c in cols]
    names = [fm.feature_names[i] for i in idx]
    binary = _binary_idx(names)
    X = fm.X[:, idx]

    if method == "local":
        s = MesaLocal(names, binary).score(X, fm.groups)
        return within_config_metrics(fm.y_success, fm.n_trials, s, fm.groups), {}

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
    return per, chosen


def collapse_per_config(per):
    """{configuration-direction: {metric: v}} -> {configuration: {metric: v}}.

    Applied before any bootstrap. The two directions of a configuration are
    computed on the same 20 or 50 tasks and are therefore dependent; leaving
    them separate would let one configuration enter the resampling twice as
    two "independent" units and understate every interval.
    """
    metrics = set()
    for m in per.values():
        metrics |= set(m)
    out = {}
    for metric in sorted(metrics):
        collapsed = collapse_directions(
            {g: m[metric] for g, m in per.items() if metric in m})
        for config, v in collapsed.items():
            out.setdefault(config, {})[metric] = v
    return out


def scores_for(fm, cols, axis, method, cross_fitted=False):
    per, chosen = per_config_scores(fm, cols, axis, method)
    if cross_fitted:
        per = collapse_per_config(per)
    return per, chosen


def paired_bootstrap(a_by_cfg, b_by_cfg, n=5000, seed=0):
    """Paired CI on the per-configuration difference a - b.

    Configurations are the resampling unit; pairing preserves the fact that
    both methods saw the same graphs.
    """
    common = sorted(set(a_by_cfg) & set(b_by_cfg))
    d = np.array([a_by_cfg[c] - b_by_cfg[c] for c in common
                  if not (np.isnan(a_by_cfg[c]) or np.isnan(b_by_cfg[c]))])
    if len(d) == 0:
        return float("nan"), (float("nan"), float("nan")), 0
    rng = np.random.default_rng(seed)
    means = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)]
    return (float(d.mean()),
            (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))),
            len(d))


def summarize(per):
    cov = {c: m["coverage_auc"] for c, m in per.items()}
    vals = [v for v in cov.values() if not np.isnan(v)]
    c20 = [m["coverage_at_20"] for m in per.values()
           if not np.isnan(m["coverage_at_20"])]
    sp = [m["spearman"] for m in per.values() if not np.isnan(m["spearman"])]
    return {
        "macro_coverage_auc": float(np.mean(vals)) if vals else float("nan"),
        "coverage_at_20": float(np.mean(c20)) if c20 else float("nan"),
        "median_within_spearman": float(np.median(sp)) if sp else float("nan"),
        "spearman_iqr": ([float(np.percentile(sp, 25)),
                          float(np.percentile(sp, 75))] if len(sp) > 1
                         else [float("nan")] * 2),
        "n_config": len(per),
        "n_zero_success": sum(1 for v in cov.values() if np.isnan(v)),
        "_per_config_auc": cov,
    }


def per_model_report(fm, axis="topology", cross_fitted=False):
    """Per-model results. Combining models asks one ordering to fit two
    different vulnerability profiles, which measurably costs signal."""
    from src.saliency.mesa_scores import MesaLocal as _ML
    out = {}
    models = sorted({r["model"] for r in fm.rows})
    for model in models:
        sel = np.array([r["model"] == model for r in fm.rows])
        sub = FeatureMatrix(
            X=fm.X[sel], y_success=fm.y_success[sel], n_trials=fm.n_trials[sel],
            groups=fm.groups[sel], feature_names=fm.feature_names,
            rows=[r for r, k in zip(fm.rows, sel) if k])
        blocks = {}
        for name, cols in BLOCKS.items():
            loc, _ = scores_for(sub, cols, axis, "local", cross_fitted)
            lea, _ = scores_for(sub, cols, axis, "learned", cross_fitted)
            sl, se_ = summarize(loc), summarize(lea)
            d, ci, npair = paired_bootstrap(sl["_per_config_auc"],
                                            se_["_per_config_auc"])
            # Spearman on the same score, for continuity with the published
            # numbers. Different scale from coverage-AUC: rho has a null of 0,
            # coverage-AUC a null of 0.5. Never present them as one quantity.
            idx = [sub.feature_names.index(c) for c in cols]
            names = [sub.feature_names[i] for i in idx]
            sc = _ML(names, _binary_idx(names)).score(sub.X[:, idx], sub.groups)
            asr = sub.y_success / np.maximum(sub.n_trials, 1)
            pooled = stats.spearmanr(sc, asr)
            within = []
            for g in np.unique(sub.groups):
                i = sub.groups == g
                if i.sum() > 3 and len(set(asr[i])) > 1 and len(set(sc[i])) > 1:
                    within.append(stats.spearmanr(sc[i], asr[i]).statistic)
            blocks[name] = {
                "local": {k: v for k, v in sl.items() if not k.startswith("_")},
                "learned": {k: v for k, v in se_.items() if not k.startswith("_")},
                "local_minus_learned": {"delta": d, "ci95": ci, "n_paired": npair},
                "local_spearman_pooled": float(pooled.statistic),
                "local_spearman_pooled_p": float(pooled.pvalue),
                "local_spearman_within_mean": (float(np.mean(within))
                                               if within else float("nan")),
                "n_config": sl["n_config"],
            }
        out[model] = blocks
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data" / "mesa_fit.json"))
    ap.add_argument("--scenarios", nargs="+", default=None)
    ap.add_argument("--cross-fitted", action="store_true",
                    help="Take task-derived features from one task fold and "
                         "outcomes from the other. Required for any reported "
                         "number: without it the dynamic deltas and F2 are "
                         "estimated on the very tasks they are scored against.")
    args = ap.parse_args()

    if args.cross_fitted:
        fm, dropped, _mats = load_enriched_directional(scenarios=args.scenarios)
    else:
        fm, dropped = load_enriched(scenarios=args.scenarios)
    scen = sorted({r["scenario"] for r in fm.rows})
    status = "PRELIMINARY (CS only)" if scen == ["customer_service"] else "FULL"
    print("MESA-Local vs MESA-Learned -- %s" % status)
    print("  edges=%d configs=%d scenarios=%s" % (len(fm.rows),
                                                  len(set(fm.groups)), scen))
    print("  features=%s" % fm.feature_names)
    print("  dropped during join: %s" % (dropped or "none"))

    audit = redundancy_audit(fm)
    flagged = [d for d in audit if d["flagged"]]
    dups = [d for d in audit if d.get("duplicate_candidate")]
    print("\nREDUNDANCY AUDIT (within-configuration, outcome-blind)")
    print("  %-34s %-34s %8s %10s" % ("feature A", "feature B", "n_cfg",
                                      "med |rho|"))
    for d in audit[:8]:
        print("  %-34s %-34s %8d %10.3f%s"
              % (d["a"], d["b"], d["n_defined"], d["median_abs_rho"],
                 "  FLAG" if d["flagged"] else ""))
    print("  pairs with median |rho| >= 0.80: %d" % len(flagged))
    print("  duplicate candidates (>=0.90 in most configs): %d" % len(dups))
    if not dups:
        print("  -> no feature removed; the predeclared rule requires a "
              "same-construct pair above 0.90")

    axes = ["topology", "model_family"]
    report = {"status": status, "cross_fitted": bool(args.cross_fitted),
              "n_edges": len(fm.rows),
              "n_configs": len(set(fm.groups)), "scenarios": scen,
              "features": fm.feature_names, "join_dropped": dropped,
              "redundancy_audit": audit, "axes": {}}

    for axis in axes:
        label = "PRIMARY" if axis == "topology" else "descriptive stress test"
        print("\n=== axis = %s (%s) ===" % (axis, label))
        print("  %-28s %-24s %-24s" % ("block", "Local covAUC", "Learned covAUC"))
        res = {}
        for name, cols in BLOCKS.items():
            loc, _ = scores_for(fm, cols, axis, "local", args.cross_fitted)
            lea, chosen = scores_for(fm, cols, axis, "learned",
                                     args.cross_fitted)
            sl, se_ = summarize(loc), summarize(lea)
            diff, ci, npair = paired_bootstrap(sl["_per_config_auc"],
                                               se_["_per_config_auc"])
            res[name] = {"local": {k: v for k, v in sl.items()
                                   if not k.startswith("_")},
                         "learned": {k: v for k, v in se_.items()
                                     if not k.startswith("_")},
                         "chosen_C": chosen,
                         "local_minus_learned": {"delta": diff, "ci95": ci,
                                                 "n_paired": npair},
                         "_local_auc": sl["_per_config_auc"],
                         "_learned_auc": se_["_per_config_auc"]}
            print("  %-28s %.3f                    %.3f   (L-L %+.3f [%+.3f,%+.3f])"
                  % (name, sl["macro_coverage_auc"], se_["macro_coverage_auc"],
                     diff, ci[0], ci[1]))

        # Incremental value of F2, F3 and F2+F3 over the reference block.
        print("\n  paired dAUC vs %s" % REFERENCE_BLOCK)
        incr = {}
        for name in ("structural_dynamic_f2", "structural_dynamic_f3",
                     "structural_dynamic_f2_f3"):
            for method in ("local", "learned"):
                key = "_%s_auc" % method
                d, ci, npair = paired_bootstrap(res[name][key],
                                                res[REFERENCE_BLOCK][key])
                incr["%s|%s" % (name, method)] = {"delta": d, "ci95": ci,
                                                  "n_paired": npair}
                print("    %-26s %-8s %+0.3f  [%+0.3f, %+0.3f]  (n=%d)"
                      % (name, method, d, ci[0], ci[1], npair))
        for v in res.values():
            v.pop("_local_auc", None)
            v.pop("_learned_auc", None)
        report["axes"][axis] = {"blocks": res, "incremental_vs_reference": incr}

    print("\n=== PER-MODEL (PRIMARY reporting unit, LOTO) ===")
    pm = per_model_report(fm, "topology", args.cross_fitted)
    report["per_model"] = pm
    print("  %-12s %-26s %7s %7s %8s %8s"
          % ("model", "block", "LocAUC", "LrnAUC", "rho_win", "rho_pool"))
    for model, blocks in pm.items():
        for name, b in blocks.items():
            print("  %-12s %-26s %7.3f %7.3f %8.3f %8.3f"
                  % (model, name, b["local"]["macro_coverage_auc"],
                     b["learned"]["macro_coverage_auc"],
                     b["local_spearman_within_mean"],
                     b["local_spearman_pooled"]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
