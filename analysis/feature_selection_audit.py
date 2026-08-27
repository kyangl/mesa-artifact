"""Audit activity and redundancy in the frozen eight-feature schema.

The script verifies that constant features do not change within-configuration
rankings or monitored sets. Outcome-blind checks are separated from post-hoc
leave-one-out and block comparisons; the latter do not alter the freeze.
"""

import collections
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.run_mesa_fit import (load_enriched_directional,   # noqa: E402
                                   collapse_directions, BLOCKS)
from analysis.run_mesa_cv import _binary_idx                    # noqa: E402
from analysis.run_nested_cv import (expected_tie_credit,        # noqa: E402
                                    coverage_auc)
from src.saliency.mesa_scores import MesaLocal, FEATURE_SIGNS   # noqa: E402
from src.saliency.normalization import normalize_within, _midrank_unit  # noqa: E402

OUT = REPO / "data" / "feature_selection_audit.json"
OUT_MD = REPO / "data" / "TABLE_feature_selection_audit.md"

SCENARIOS = ("customer_service", "software_engineering")
CANONICAL = "structural_dynamic"
N_DECLARED = 8
BUDGET = 0.20                      # the outcome-aware diagnostics use this one
EQUIV_BUDGETS = (0.10, 0.20, 0.40)  # rank-equivalence is checked at all three
B = 10000
SEED = 20260817

# Round before tie checks to suppress summation-order noise near 1e-16.
ROUND_DECIMALS = 12

# Pinned denominators used by audit assertions.
EXPECT_CONFIGURATIONS_PER_DOMAIN = 15
EXPECT_CONFIGURATIONS_TOTAL = 30
EXPECT_DIRECTIONS_PER_DOMAIN = 30
EXPECT_DIRECTIONS_TOTAL = 60
EXPECT_ENFORCEMENT_CONFIGURATIONS = 10

ENF_DIR = REPO / "results" / "enforcement" / "customer_service" / "6041555_eight"

# F1 was measured but excluded because nr_max is confounded with in-degree.
ARCHIVAL_MAS = REPO / "data" / "mas_features.json"
F1_PRIMARY = "nr_max"
F1_CONFOUND = "n_context"

F1 = "semantic_non_recoverability"
F2 = "receiver_response_sensitivity"
F3 = "consequence_proximity"
GRAPH = list(BLOCKS["structural"])
PROBES = list(BLOCKS["dynamic"])

# The seven blocks compared, defined MECHANISTICALLY -- by what a practitioner
# would have to run to obtain them, not by searching subsets for a winner.
# There is no eighth block, and none of these is chosen by its score.
BLOCK_SPECS = collections.OrderedDict([
    ("graph_only", GRAPH),
    ("graph_ablation", GRAPH + ["ablation_delta"]),
    ("graph_perturbation", GRAPH + ["perturbation_delta"]),
    ("canonical_graph_both_probes", GRAPH + PROBES),
    ("canonical_f2", GRAPH + PROBES + [F2]),
    ("canonical_f3", GRAPH + PROBES + [F3]),
    ("canonical_f2_f3", GRAPH + PROBES + [F2, F3]),
])

MECHANISM = {
    "betweenness_centrality":
        "graph: fraction of shortest paths through the edge",
    "information_bottleneck":
        "graph: does removing the edge disconnect a source-sink information path",
    "is_bridge":
        "graph: edge is a bridge of the undirected support (binary)",
    "endpoint_centrality_max":
        "graph: max(source degree centrality, target degree centrality) -- DERIVED "
        "from the two features below",
    "source_degree_centrality": "graph: degree centrality of the sending agent",
    "target_degree_centrality": "graph: degree centrality of the receiving agent",
    "ablation_delta":
        "probe: remove the edge, re-run the clean workflow, paired drop in "
        "decision accuracy",
    "perturbation_delta":
        "probe: inject noise on the edge, re-run the clean workflow, paired drop "
        "in decision accuracy",
    F1: "probe: 1 - E_t[max_c sim(message, receiver's prior context)]",
    F2: "probe: E_t[1 - sim(receiver output, receiver output under a neutral "
        "placeholder)]",
    F3: "graph: 1 / (1 + directed distance from receiver to nearest declared sink)",
}

COST_CLASS = {
    "betweenness_centrality": "graph-only",
    "information_bottleneck": "graph-only",
    "is_bridge": "graph-only",
    "endpoint_centrality_max": "graph-only",
    "source_degree_centrality": "graph-only",
    "target_degree_centrality": "graph-only",
    "ablation_delta": "workflow re-execution",
    "perturbation_delta": "workflow re-execution",
    F1: "clean transcripts + embedding",
    F2: "receiver probe generation + embedding",
    F3: "graph-only",
}


# --------------------------------------------------------------- primitives

def _cov(y, s, k):
    if y.sum() <= 0:
        return float("nan")
    return float(expected_tie_credit(y, s)[:k].sum() / y.sum())


def _k_for(b, E):
    return max(1, min(E, int(math.ceil(b * E))))


def _boot(d, b=B, seed=SEED):
    d = np.asarray([x for x in d if x is not None and not
                    (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(d) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    rng = np.random.default_rng(seed)
    dr = rng.choice(d, size=(b, len(d)), replace=True).mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(dr, 2.5)),
            "hi": float(np.percentile(dr, 97.5)), "n": int(len(d))}


def sign_flip(diffs):
    """Exact paired sign-flip over all 2^n patterns. Same test as elsewhere."""
    d = np.asarray([x for x in diffs if not (isinstance(x, float)
                                             and math.isnan(x))], dtype=float)
    n = len(d)
    if n == 0 or not np.any(np.abs(d) > 0):
        return {"n": int(n), "p_two_sided": 1.0,
                "min_attainable_two_sided": (2.0 / 2 ** n) if n else None,
                "exact": True}
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    obs = float(d.mean())
    means = signs.dot(d) / n
    return {"n": int(n), "observed_mean": obs, "exact": True,
            "p_two_sided": float(np.mean(np.abs(means) >= abs(obs) - 1e-12)),
            "min_attainable_two_sided": 2.0 / len(signs)}


def holm(pvals):
    """Holm-Bonferroni step-down. Returns adjusted p in the input order."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, adj, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def score_of(fm, names):
    """MESA-Local over a named subset of the enriched matrix."""
    idx = [fm.feature_names.index(c) for c in names]
    nm = [fm.feature_names[i] for i in idx]
    return MesaLocal(nm, _binary_idx(nm)).score(fm.X[:, idx], fm.groups)


def active_names(fm, sel, names):
    """Features that actually vary among THIS configuration-direction's edges.

    A feature constant here contributes an identical amount to every edge and
    cannot affect the ordering. Constancy is judged on the raw values, before
    normalization, because that is what makes the normalized column constant.
    """
    out = []
    for n in names:
        v = fm.X[sel, fm.feature_names.index(n)]
        v = v[~np.isnan(v)]
        if len(v) and len(np.unique(np.round(v, 12))) > 1:
            out.append(n)
    return out


def per_group_score(fm, sel, names):
    """Score a group ALONE on a named subset -- never against other groups."""
    if not names:
        return np.full(len(sel), 0.5)
    idx = [fm.feature_names.index(c) for c in names]
    nm = [fm.feature_names[i] for i in idx]
    g = np.zeros(len(sel), dtype=int)
    return MesaLocal(nm, _binary_idx(nm)).score(fm.X[np.ix_(sel, idx)], g)


# ----------------------------------------------- F1: measured, then rejected

def f1_archival_record():
    """Recompute F1 coverage and its in-degree confounding from archival data."""
    if not ARCHIVAL_MAS.exists():
        return {"available": False, "status": "archival matrix not present"}
    d = json.loads(ARCHIVAL_MAS.read_text())
    recs = d["records"]

    def _rho(rows):
        nr = np.array([r.get(F1) for r in rows], dtype=float)
        nc = np.array([r.get(F1_CONFOUND) for r in rows], dtype=float)
        ok = ~(np.isnan(nr) | np.isnan(nc))
        if ok.sum() < 3:
            return None
        r = stats.spearmanr(nr[ok], nc[ok])
        return {"n": int(ok.sum()), "spearman_rho": float(r.statistic),
                "p_value": float(r.pvalue)}

    scens = sorted({r["scenario"] for r in recs})
    measured = sum(1 for r in recs if r.get(F1) is not None)
    tasks = int(sum(r.get("f1_n_tasks") or 0 for r in recs))
    n_cfg = len(d.get("complete_configurations") or [])
    return {
        "available": True,
        "source": str(ARCHIVAL_MAS.relative_to(REPO)),
        "n_records": len(recs),
        "n_records_with_f1": measured,
        "coverage": measured / float(len(recs)) if recs else 0.0,
        "n_configurations": n_cfg,
        "models": sorted({r["model"] for r in recs}),
        "records_by_scenario": {sc: sum(1 for r in recs
                                        if r["scenario"] == sc)
                                for sc in scens},
        "f1_task_evaluations": tasks,
        "primary_statistic": d.get("f1_primary", F1_PRIMARY),
        "bias_note_as_recorded": d.get("f1_bias_note"),
        "confounding": {
            "against": F1_CONFOUND,
            "all_archival_records": _rho(recs),
            "by_scenario": {sc: _rho([r for r in recs
                                      if r["scenario"] == sc])
                            for sc in scens},
            "reading": (
                "nr_max falls as the number of alternative contexts rises. "
                "That count is essentially receiver in-degree, which the "
                "structural block already carries at zero marginal cost, so "
                "the feature largely re-expresses a graph quantity rather "
                "than adding semantic information."),
        },
        "rebuilt_for_current_matrix": False,
        "current_matrix_coverage": 0.0,
        "status": ("earlier two-model diagnostic; rejected for strong "
                   "context-count confounding; not rebuilt for the expanded "
                   "30-configuration matrix"),
        "excluded_from_block_comparison_because": (
            "the block comparison holds the sample fixed and varies only the "
            "feature set. F1 exists on %d archival two-model configurations "
            "and the comparison runs on %d three-model ones, so including it "
            "would change the feature set and the sample in the same step and "
            "no difference could be attributed to either."
            % (n_cfg, EXPECT_CONFIGURATIONS_TOTAL)),
        "not_a_claim": (
            "This is NOT 'never measured'. F1 was computed at full coverage "
            "on %d records over %d task evaluations, and rejected on the "
            "evidence recorded here." % (measured, tasks)),
    }


# --------------------------------------------------------- 1. the inventory

def inventory(mats):
    """Mechanism, cost, coverage and ACTIVITY for every candidate feature.

    Activity is the point of the table. A declared feature that never varies
    inside a configuration is inert under within-configuration normalization,
    no matter how well motivated it is.
    """
    names = list(mats[SCENARIOS[0]].feature_names)
    archival = f1_archival_record()

    out = collections.OrderedDict()
    for n in GRAPH + PROBES + [F1, F2, F3]:
        rec = {
            "mechanism": MECHANISM[n],
            "cost_class": COST_CLASS[n],
            "sign": FEATURE_SIGNS[n],
            "in_canonical_block": n in BLOCKS[CANONICAL],
        }
        if n not in names:
            # Absent from the CURRENT matrix. That is a statement about this
            # matrix, not about whether the feature was ever measured, and the
            # two must not be conflated -- see f1_archival_record().
            cf = (archival.get("confounding") or {})
            allr = cf.get("all_archival_records") or {}
            cs = (cf.get("by_scenario") or {}).get("customer_service") or {}
            rec.update({
                # `available` means "present in the CURRENT matrix", which is
                # the only sense the activity columns can be computed in.
                "available": False,
                "measured_in_archival_study": bool(archival.get("available")),
                "coverage": 0.0,
                "archival_coverage": archival.get("coverage"),
                "n_configurations_varies": None,
                "fraction_configurations_varies": None,
                "median_distinct_values": None,
                "archival": archival,
                "status": archival.get("status", "not in the current matrix"),
                "reason": (
                    "MEASURED, THEN REJECTED -- not absent for want of "
                    "effort. The archival two-model matrix (%s) carries %d F1 "
                    "records at %.0f%% coverage over %d task evaluations, "
                    "spanning %d configurations and %s. It was rejected "
                    "because its primary statistic (%s) is strongly "
                    "confounded with %s, the count of alternative contexts "
                    "available to the receiver: customer service n=%d, "
                    "Spearman rho=%.3f, p=%.1e; all archival records n=%d, "
                    "rho=%.3f. That count is essentially receiver in-degree, "
                    "which the structural block already supplies at no "
                    "marginal cost. F1 was therefore NOT rebuilt when the "
                    "matrix was expanded to three models and %d "
                    "configurations, and it is held out of the equal-coverage "
                    "block comparison so that the feature set and the sample "
                    "are never varied in the same step."
                    % (archival.get("source", "data/mas_features.json"),
                       archival.get("n_records_with_f1", 0),
                       100.0 * (archival.get("coverage") or 0.0),
                       archival.get("f1_task_evaluations", 0),
                       archival.get("n_configurations", 0),
                       " and ".join(archival.get("models") or []),
                       archival.get("primary_statistic", F1_PRIMARY),
                       F1_CONFOUND,
                       cs.get("n", 0), cs.get("spearman_rho", float("nan")),
                       cs.get("p_value", float("nan")),
                       allr.get("n", 0),
                       allr.get("spearman_rho", float("nan")),
                       EXPECT_CONFIGURATIONS_TOTAL)),
            })
            out[n] = rec
            continue

        by_domain, distinct_all = {}, []
        varies_cfg_total, varies_dir_total = 0, 0
        for sc in SCENARIOS:
            fm = mats[sc]
            j = fm.feature_names.index(n)
            per_cfg, distinct, vdir = collections.defaultdict(list), [], 0
            for g in np.unique(fm.groups):
                v = fm.X[fm.groups == g, j]
                v = v[~np.isnan(v)]
                nd = int(len(np.unique(np.round(v, 12))))
                distinct.append(nd)
                vdir += int(nd > 1)
                per_cfg[g.rsplit("|", 1)[0]].append(nd > 1)
            n_any = sum(1 for v in per_cfg.values() if any(v))
            n_all = sum(1 for v in per_cfg.values() if all(v))
            by_domain[sc] = {
                "n_configurations": len(per_cfg),
                "n_configurations_varies": n_any,
                "n_configurations_varies_in_both_directions": n_all,
                "n_directions": len(distinct),
                "n_directions_varies": vdir,
                "median_distinct_values": float(np.median(distinct)),
                "max_distinct_values": int(max(distinct)),
            }
            varies_cfg_total += n_any
            varies_dir_total += vdir
            distinct_all += distinct

        # Coverage inside the analysed matrix: every retained row has a value,
        # so this is 1.0 by construction of the complete-case loader and is
        # recorded to make that explicit rather than to discriminate.
        cov = float(np.mean([np.mean(~np.isnan(
            mats[sc].X[:, mats[sc].feature_names.index(n)]))
            for sc in SCENARIOS]))
        inert = varies_cfg_total == 0
        rec.update({
            "available": True,
            "coverage": cov,
            "n_configurations_varies": varies_cfg_total,
            "fraction_configurations_varies": varies_cfg_total
            / float(EXPECT_CONFIGURATIONS_TOTAL),
            "n_directions_varies": varies_dir_total,
            "median_distinct_values": float(np.median(distinct_all)),
            "by_domain": by_domain,
            "status": "included (inert)" if inert
                      else ("included" if n in BLOCKS[CANONICAL]
                            else "declared, not in canonical block"),
            "reason": _reason(n, inert, varies_cfg_total),
        })
        out[n] = rec
    return out


def _reason(n, inert, varies):
    if inert:
        return ("In the canonical block, but constant across the edges of every "
                "one of the %d configurations, in both cross-fit directions. "
                "Within-configuration normalization maps a constant column to a "
                "single value, so this feature adds the same amount to every "
                "edge and cannot change any ranking. It is declared, not active."
                % EXPECT_CONFIGURATIONS_TOTAL)
    if n in BLOCKS[CANONICAL]:
        return ("In the canonical block; varies in %d of the %d configurations."
                % (varies, EXPECT_CONFIGURATIONS_TOTAL))
    if n == F2:
        return ("Excluded by the freeze: the ten-feature block never beat the "
                "eight by more than one SE and was worse on software "
                "engineering. Most expensive candidate measured.")
    if n == F3:
        return ("Excluded by the freeze, with the same rule as F2. Graph-only, "
                "so it is the cheapest excluded candidate.")
    return "not in the canonical block"


# ------------------------------------------------- 2. redundancy, per domain

def redundancy(mats, flag=0.80, duplicate=0.90):
    """Summarize within-configuration absolute Spearman correlation by domain.

    Correlations with a constant feature are undefined and excluded, not zero.
    """
    out = {}
    for sc in SCENARIOS:
        fm = mats[sc]
        names = list(fm.feature_names)
        pairs = []
        for a, b in itertools.combinations(range(len(names)), 2):
            rhos, undefined = [], 0
            for g in np.unique(fm.groups):
                idx = fm.groups == g
                xa, xb = fm.X[idx, a], fm.X[idx, b]
                ok = ~(np.isnan(xa) | np.isnan(xb))
                xa, xb = xa[ok], xb[ok]
                if len(xa) < 4 or len(np.unique(np.round(xa, 12))) < 2 \
                        or len(np.unique(np.round(xb, 12))) < 2:
                    undefined += 1
                    continue
                r = stats.spearmanr(xa, xb).statistic
                if r is None or (isinstance(r, float) and math.isnan(r)):
                    undefined += 1
                    continue
                rhos.append(abs(float(r)))
            n_eval = len(rhos)
            pairs.append({
                "a": names[a], "b": names[b],
                "n_directions_total": int(len(np.unique(fm.groups))),
                "n_directions_evaluable": n_eval,
                "n_directions_undefined": undefined,
                "median_abs_spearman": float(np.median(rhos)) if n_eval else None,
                "n_above_0.80": int(sum(r > flag for r in rhos)) if n_eval else 0,
                "n_above_0.90": int(sum(r > duplicate for r in rhos)) if n_eval else 0,
                "fraction_above_0.80": (float(np.mean([r > flag for r in rhos]))
                                        if n_eval else None),
                "fraction_above_0.90": (float(np.mean([r > duplicate for r in rhos]))
                                        if n_eval else None),
                "undefined_note": (
                    "no evaluable direction: at least one feature is constant "
                    "in every configuration" if not n_eval else None),
            })
        # Defined pairs first, strongest correlation first; pairs that could
        # never be evaluated sort last rather than to the top of the table.
        pairs.sort(key=lambda p: (p["median_abs_spearman"] is None,
                                  -(p["median_abs_spearman"] or 0.0)))
        defined = [p for p in pairs if p["median_abs_spearman"] is not None]
        out[sc] = {
            "n_configurations": EXPECT_CONFIGURATIONS_PER_DOMAIN,
            "n_directions": int(len(np.unique(fm.groups))),
            "flag_threshold": flag, "duplicate_threshold": duplicate,
            "n_pairs": len(pairs),
            "n_pairs_evaluable": len(defined),
            "n_pairs_never_evaluable": len(pairs) - len(defined),
            "n_pairs_median_above_flag": sum(
                1 for p in defined if p["median_abs_spearman"] > flag),
            "n_pairs_median_above_duplicate": sum(
                1 for p in defined if p["median_abs_spearman"] > duplicate),
            "note": ("Fractions divide by n_directions_evaluable, NOT by "
                     "n_directions_total. Undefined correlations are never "
                     "replaced by zero. A pair involving a feature that is "
                     "constant in every configuration is never evaluable at "
                     "all, which is a statement about the feature, not "
                     "evidence that the two are independent."),
            "pairs": pairs,
        }
    return out


# ----------------------------------------------- 3. rank equivalence, 30 cfgs

def rank_equivalence(mats):
    """Declared-eight vs active-only: identical ranks and identical top-k.

    Checked on every configuration-direction of every domain, at all three
    budgets. Both scores are computed group-by-group so neither can borrow
    information from another configuration.
    """
    checks, failures = [], []
    max_residual, n_float_only = 0.0, 0
    for sc in SCENARIOS:
        fm = mats[sc]
        for g in sorted(np.unique(fm.groups)):
            sel = np.where(fm.groups == g)[0]
            E = len(sel)
            declared = BLOCKS[CANONICAL]
            act = active_names(fm, sel, declared)
            s8 = np.nan_to_num(per_group_score(fm, sel, declared), nan=0.5)
            sa = np.nan_to_num(per_group_score(fm, sel, act), nan=0.5)

            # The affine relation itself, checked directly. If this residual
            # is ever larger than rounding noise the equivalence is genuinely
            # broken and no tolerance should hide it.
            offset = s8 - (len(act) / float(len(declared))) * sa
            residual = float(np.max(np.abs(offset - np.median(offset))))
            max_residual = max(max_residual, residual)

            q8, qa = np.round(s8, ROUND_DECIMALS), np.round(sa, ROUND_DECIMALS)
            r8, ra = _midrank_unit(q8), _midrank_unit(qa)
            same_ranks = bool(np.array_equal(r8, ra))
            exact_ranks = bool(np.array_equal(_midrank_unit(s8),
                                              _midrank_unit(sa)))
            if same_ranks and not exact_ranks:
                n_float_only += 1

            topk = {}
            for b in EQUIV_BUDGETS:
                k = _k_for(b, E)
                o8 = np.argsort(-q8, kind="stable")[:k]
                oa = np.argsort(-qa, kind="stable")[:k]
                same_set = set(o8.tolist()) == set(oa.tolist())
                topk["%d%%" % round(b * 100)] = {
                    "k": int(k), "identical_set": bool(same_set),
                    "identical_order": bool(np.array_equal(o8, oa))}
                if not same_set:
                    failures.append({"scenario": sc, "group": str(g),
                                     "budget": b, "kind": "top_k_set"})
            if not same_ranks:
                failures.append({"scenario": sc, "group": str(g),
                                 "kind": "ranks", "residual": residual})
            checks.append({
                "scenario": sc, "group": str(g), "n_edges": int(E),
                "n_declared": len(declared), "n_active": len(act),
                "active": act,
                "inactive": [n for n in declared if n not in act],
                "identical_ranks": same_ranks,
                "identical_ranks_unrounded": exact_ranks,
                "affine_residual": residual, "top_k": topk,
            })
    n_active = [c["n_active"] for c in checks]
    return {
        "budgets": list(EQUIV_BUDGETS),
        "n_checks": len(checks),
        "n_configuration_directions": len(checks),
        "comparison_tolerance_decimals": ROUND_DECIMALS,
        "all_identical": not failures,
        "failures": failures,
        "max_affine_residual": max_residual,
        "n_identical_only_after_rounding": n_float_only,
        "float_note": (
            "%d of %d configuration-directions agree only after rounding to "
            "%d decimals. In every one of them the largest departure from the "
            "exact affine relation is %.2e -- IEEE summation order over six "
            "versus eight terms, not a difference in ordering. Tie detection "
            "compares for equality, so a last-bit difference splits a genuine "
            "tie and moves a midrank. The unrounded result is reported here "
            "rather than suppressed because it is also a caveat about the "
            "score itself: MESA-Local's tie structure is reproducible only "
            "when the features are summed in a fixed order, which the frozen "
            "block does guarantee."
            % (n_float_only, len(checks), ROUND_DECIMALS, max_residual)),
        "n_active_min": int(min(n_active)), "n_active_max": int(max(n_active)),
        "n_active_median": float(np.median(n_active)),
        "distribution_n_active": {str(k): int(v) for k, v in sorted(
            collections.Counter(n_active).items())},
        "algebraic_note": (
            "eight(e) = (active_sum(e) + c)/8 with c constant within the "
            "configuration, and active(e) = active_sum(e)/n_active, so "
            "eight = (n_active/8)*active + c/8 is a strictly increasing affine "
            "map of the active-only score. Equal ranks are therefore an "
            "identity; this check exists to catch an implementation that "
            "departs from it."),
        "checks": checks,
    }


# ------------------------------------ 4. the executed enforcement allocations

def enforcement_monitored_sets():
    """Would the active-only score have monitored the same edges? Ten configs.

    The executed sets are read from the result files themselves, which are the
    record of what actually ran -- not recomputed from a plan that might have
    drifted from it.
    """
    fm, _d, _m = load_enriched_directional(scenarios=["customer_service"])

    orders = {}
    for g in np.unique(fm.groups):
        sel = np.where(fm.groups == g)[0]
        rows = [fm.rows[i] for i in sel]
        model, topo = rows[0]["model"], rows[0]["topology"]
        direction = rows[0].get("direction") or (
            "A->B" if "A->B" in str(g) else "B->A")
        edges = [(r["edge_src"], r["edge_dst"]) for r in rows]
        act = active_names(fm, sel, BLOCKS[CANONICAL])
        s8 = np.nan_to_num(per_group_score(fm, sel, BLOCKS[CANONICAL]), nan=0.5)
        sa = np.nan_to_num(per_group_score(fm, sel, act), nan=0.5)

        def order_by(s):
            # Identical ordering rule to the executed run: a stable sort on
            # the negated score, so ties keep matrix order in both.
            return [e for _v, e in sorted(zip(-s, edges), key=lambda p: p[0])]

        orders[(model, topo, direction)] = {
            "edges": edges, "s8": s8,
            "declared": order_by(s8), "active": order_by(sa),
            "declared_rounded": order_by(np.round(s8, ROUND_DECIMALS)),
            "active_rounded": order_by(np.round(sa, ROUND_DECIMALS)),
        }

    executed, changes, control, cells, n_tie = {}, [], [], 0, 0
    agree = collections.Counter()
    for p in sorted(ENF_DIR.glob("cs_enforcement_*.json")):
        d = json.loads(p.read_text())
        key = (d["model"], d["topology"])
        executed.setdefault(key, [])
        for c in d["cells"]:
            if c.get("policy") != "mesa_local_crossfitted_k":
                continue
            mon = sorted(tuple(e) for e in c["monitored"])
            o = orders.get((c["model"], c["topology"], c["direction"]))
            cells += 1
            if o is None:
                control.append({"model": c["model"], "topology": c["topology"],
                                "direction": c["direction"],
                                "budget": c["budget"],
                                "reason": "no recomputed ordering"})
                continue
            k = _k_for(c["budget"], len(o["edges"]))

            # Is the k-th/k+1-th boundary a MATHEMATICAL tie? If so, no score
            # decided this cell; the arrangement of the matrix did.
            srt = np.sort(o["s8"])[::-1]
            tie = bool(k < len(srt)
                       and abs(srt[k - 1] - srt[k]) < 10.0 ** -ROUND_DECIMALS)
            n_tie += tie

            got = {t: sorted(o[t][:k]) for t in
                   ("declared", "active", "declared_rounded", "active_rounded")}
            for t, v in got.items():
                agree[t] += (v == mon)
            if got["declared"] != mon:
                control.append({
                    "model": c["model"], "topology": c["topology"],
                    "direction": c["direction"], "budget": c["budget"],
                    "executed": [list(e) for e in mon],
                    "declared_eight": [list(e) for e in got["declared"]]})
            if got["active"] != mon:
                a = set(mon) - set(got["active"])
                b = set(got["active"]) - set(mon)
                changes.append({
                    "model": c["model"], "topology": c["topology"],
                    "direction": c["direction"], "budget": c["budget"],
                    "k": int(k), "boundary_is_a_tie": tie,
                    "executed": [list(e) for e in mon],
                    "active_only": [list(e) for e in got["active"]],
                    "dropped": [list(e) for e in sorted(a)],
                    "added": [list(e) for e in sorted(b)],
                    "explanation": (
                        "The two edges score identically under the declared "
                        "eight, so the ordering never distinguished them; the "
                        "executed run separated them on IEEE summation order "
                        "alone." if tie else
                        "The scores genuinely differ -- this is a real change "
                        "of allocation, not a tie-break.")})
            executed[key].append({"direction": c["direction"],
                                  "budget": c["budget"], "k": int(k),
                                  "identical": got["active"] == mon})
    return {
        "source": str(ENF_DIR.relative_to(REPO)),
        "policy": "mesa_local_crossfitted_k",
        "n_configurations": len(executed),
        "n_allocation_cells": cells,
        "configurations": ["%s/%s" % k for k in sorted(executed)],
        "n_cells_matching": {t: int(v) for t, v in agree.items()},
        "n_cells_with_boundary_tie": int(n_tie),
        "all_identical": not changes,
        "n_changed": len(changes),
        "changes": changes,
        "n_changed_that_are_boundary_ties": sum(1 for c in changes
                                                if c["boundary_is_a_tie"]),
        "declared_eight_reproduces_executed": not control,
        "declared_eight_mismatches": control,
        "note": ("Executed monitored sets are read from the enforcement result "
                 "files, which record what actually ran. Comparison uses the "
                 "UNROUNDED score, because that is what the executed run used; "
                 "the rounded counts are reported beside it to show how much "
                 "of the allocation rides on last-bit arithmetic. The "
                 "declared-eight row is the control: it must reproduce the "
                 "executed sets exactly, or nothing can be concluded about any "
                 "other schema."),
    }


# ------------------------------------------- 5/6. outcome-aware diagnostics

def _per_config(fm, s_all, budget=BUDGET, round_scores=True):
    """Collapse tie-aware coverage-AUC and budget coverage by configuration.

    Optional rounding prevents numerical noise from splitting exact ties.
    """
    auc, cov = {}, {}
    for g in np.unique(fm.groups):
        sel = np.where(fm.groups == g)[0]
        y = fm.y_success[sel].astype(float)
        s = np.nan_to_num(s_all[sel], nan=0.5)
        if round_scores:
            s = np.round(s, ROUND_DECIMALS)
        auc[g] = coverage_auc(y, s)
        cov[g] = _cov(y, s, _k_for(budget, len(y)))
    return collapse_directions(auc), collapse_directions(cov)


def float_tie_sensitivity(mats):
    """How much of the published metric rides on last-bit arithmetic?

    Reported because the audit's own diagnostics are computed on rounded
    scores while `data/canonical_ranking.json` is not, and a reader
    comparing the two deserves the reconciliation rather than a discrepancy.
    """
    out = {}
    for sc in SCENARIOS:
        fm = mats[sc]
        s = score_of(fm, BLOCKS[CANONICAL])
        ra, rc = _per_config(fm, s, round_scores=True)
        ua, uc = _per_config(fm, s, round_scores=False)
        cfgs = sorted(ra)
        m = lambda d: float(np.nanmean([d[c] for c in cfgs]))  # noqa: E731
        out[sc] = {
            "coverage_auc_unrounded": m(ua), "coverage_auc_rounded": m(ra),
            "coverage_auc_difference": m(ra) - m(ua),
            "coverage_at_20_unrounded": m(uc), "coverage_at_20_rounded": m(rc),
            "coverage_at_20_difference": m(rc) - m(uc),
        }
    out["note"] = (
        "coverage@20% is IDENTICAL either way: a budget cutoff is a coarse "
        "enough functional that last-bit tie splitting does not reach it. "
        "coverage-AUC is not -- it integrates over every cutoff, so every "
        "split tie contributes. The canonical table reports the unrounded "
        "AUC; the difference is in the fourth decimal and changes no "
        "conclusion, but it is arithmetic rather than signal and the "
        "audit's own numbers are the rounded ones.")
    return out


def leave_one_out(mats):
    """Remove one declared feature at a time. Post-hoc, explanatory only.

    This does not select anything. It reports what each declared feature is
    contributing to the frozen block's own numbers, which is a different
    question from whether it should have been declared.
    """
    out = {}
    for sc in SCENARIOS:
        fm = mats[sc]
        full = score_of(fm, BLOCKS[CANONICAL])
        f_auc, f_cov = _per_config(fm, full)
        cfgs = sorted(f_auc)
        rows, p_auc, p_cov = [], [], []
        for drop in BLOCKS[CANONICAL]:
            kept = [n for n in BLOCKS[CANONICAL] if n != drop]
            r_auc, r_cov = _per_config(fm, score_of(fm, kept))
            d_auc = [r_auc[c] - f_auc[c] for c in cfgs]
            d_cov = [r_cov[c] - f_cov[c] for c in cfgs]
            sa, sv = sign_flip(d_auc), sign_flip(d_cov)
            p_auc.append(sa["p_two_sided"])
            p_cov.append(sv["p_two_sided"])
            rows.append({
                "removed": drop, "n_kept": len(kept),
                "delta_auc": _boot(d_auc), "signflip_auc": sa,
                "delta_coverage_at_20": _boot(d_cov), "signflip_coverage_at_20": sv,
                "inert": bool(all(abs(x) < 1e-12 for x in d_auc + d_cov)),
            })
        for r, a, c in zip(rows, holm(p_auc), holm(p_cov)):
            r["holm_p_auc"] = float(a)
            r["holm_p_coverage_at_20"] = float(c)
        out[sc] = {
            "analysis_type": "post-hoc sensitivity analysis",
            "is_feature_selection": False,
            "n_configurations": len(cfgs),
            "budget": BUDGET,
            "full_block_auc": float(np.nanmean([f_auc[c] for c in cfgs])),
            "full_block_coverage_at_20": float(np.nanmean([f_cov[c] for c in cfgs])),
            "holm_family": ("the %d features removed, within this domain and "
                            "metric" % len(BLOCKS[CANONICAL])),
            "note": ("POST-HOC SENSITIVITY ANALYSIS, NOT FEATURE SELECTION. "
                     "Sign is 'reduced minus full': a NEGATIVE delta means "
                     "removing the feature HURT. These per-feature results "
                     "describe how the FROZEN block behaves when perturbed; "
                     "they were computed after the freeze, on the same data "
                     "the freeze used, and they are not a criterion. No "
                     "feature is added, removed or reweighted on the strength "
                     "of any number in this table, and a reader must not read "
                     "the ranking of these deltas as a recommended subset."),
            "rows": rows,
        }
    return out


def block_comparison(mats):
    """Seven mechanistically defined blocks. No exhaustive subset search."""
    out = {}
    for sc in SCENARIOS:
        fm = mats[sc]
        base_auc, base_cov = _per_config(fm, score_of(fm, BLOCK_SPECS[
            "canonical_graph_both_probes"]))
        cfgs = sorted(base_auc)
        rows = []
        for name, cols in BLOCK_SPECS.items():
            a, c = _per_config(fm, score_of(fm, cols))
            d_auc = [a[k] - base_auc[k] for k in cfgs]
            d_cov = [c[k] - base_cov[k] for k in cfgs]
            n_active = []
            for g in np.unique(fm.groups):
                sel = np.where(fm.groups == g)[0]
                n_active.append(len(active_names(fm, sel, cols)))
            rows.append({
                "block": name, "n_features": len(cols), "features": list(cols),
                "median_active_features": float(np.median(n_active)),
                "auc": float(np.nanmean([a[k] for k in cfgs])),
                "coverage_at_20": float(np.nanmean([c[k] for k in cfgs])),
                "delta_auc_vs_canonical": _boot(d_auc),
                "delta_coverage_at_20_vs_canonical": _boot(d_cov),
                "signflip_auc_vs_canonical": sign_flip(d_auc),
            })
        out[sc] = {"analysis_type": "post-hoc sensitivity analysis",
                   "is_feature_selection": False,
                   "n_configurations": len(cfgs), "budget": BUDGET,
                   "reference": "canonical_graph_both_probes",
                   "equal_coverage": True,
                   "note": (
                       "POST-HOC SENSITIVITY ANALYSIS, NOT ALTERNATIVE "
                       "FEATURE SELECTION. Deltas are block minus canonical, "
                       "paired on the configuration. Every block here is "
                       "scored on the SAME %d configurations with the same "
                       "coverage, which is what makes the contrast about the "
                       "feature set alone -- and is why F1 is not in this "
                       "table (see inventory[%r].archival). `graph_only`, "
                       "`graph_ablation` and `graph_perturbation` in "
                       "particular are single-probe sensitivity arms: "
                       "`graph_perturbation` scoring at or above the "
                       "canonical block in a domain is a statement about how "
                       "little the second probe adds THERE, not a candidate "
                       "replacement for the frozen eight."
                       % (len(cfgs), F1)),
                   "rows": rows}
    return out


# ------------------------------------------------ top-k tie behaviour, exactly

def tie_behaviour(enf):
    """How a top-k set is resolved when the budget boundary lands in a tie.

    DOCUMENTATION, NOT REINTERPRETATION. The completed enforcement runs stand
    exactly as executed; nothing here re-scores them, re-labels an outcome, or
    revises a prevention number. What it does is make the selection rule
    explicit, because a reader deserves to know that a top-k SET is not fully
    determined by a score when several edges hold the same score.
    """
    return {
        "scope": ("Documents the selection rule as implemented. Completed "
                  "enforcement runs are NOT reinterpreted: the executed "
                  "monitored sets remain what ran, and every reported "
                  "prevention number stands."),
        "rule_as_implemented": [
            "1. MESA-Local scores every edge of a configuration-direction as "
            "a float64 equal average over the direction-aligned, "
            "within-configuration midranks of the declared features.",
            "2. Edges are ordered by `sorted(zip(-score, edges), key=first)`. "
            "Python's sort is STABLE and the key is the negated score alone, "
            "so edges comparing equal keep their relative order from the "
            "feature matrix.",
            "3. The monitored set is the first k, with "
            "k = max(1, min(|E|, ceil(b|E|))).",
        ],
        "consequences": [
            "Deterministic and reproducible: for a fixed matrix and a fixed "
            "feature order the same set comes out every time.",
            "NOT a property of the score: when the k-th and (k+1)-th edges "
            "hold the same score, which one is monitored is decided by matrix "
            "row order, which encodes nothing about vulnerability.",
            "Sensitive to summation order: two edges that are mathematically "
            "tied can differ by ~1e-16 depending on how many features were "
            "averaged, in which case the float difference decides before row "
            "order is ever consulted. This is why the same ordering rounded "
            "to 1e-%d does not always reproduce itself." % ROUND_DECIMALS,
        ],
        "observed_in_the_executed_sweep": {
            "n_allocation_cells": enf["n_allocation_cells"],
            "n_cells_with_boundary_tie": enf["n_cells_with_boundary_tie"],
            "declared_eight_unrounded_reproduces": enf["n_cells_matching"].get(
                "declared", 0),
            "declared_eight_rounded_reproduces": enf["n_cells_matching"].get(
                "declared_rounded", 0),
            "reading": (
                "%d of %d cells have a mathematical tie at the budget "
                "boundary. The declared eight-feature score as executed "
                "(unrounded) reproduces all %d monitored sets exactly, so the "
                "executed sweep is fully accounted for. Rounding the same "
                "score reproduces %d, which measures how much of the "
                "allocation rests on last-bit arithmetic rather than on the "
                "feature values."
                % (enf["n_cells_with_boundary_tie"], enf["n_allocation_cells"],
                   enf["n_cells_matching"].get("declared", 0),
                   enf["n_cells_matching"].get("declared_rounded", 0))),
        },
        "recommended_for_future_runs": (
            "Declare an explicit, score-independent tie-break -- a documented "
            "edge key rather than matrix position -- so a top-k set is a "
            "function of the graph and the score alone. This is a change to "
            "future allocation, not a correction to past results."),
    }


# ----------------------------------------------------------------- 7. cost

def cost_table():
    """What each block costs, counted from the logs that produced it."""
    dd = json.loads((REPO / "data" / "dynamic_deltas.json").read_text())
    runs = collections.defaultdict(collections.Counter)
    for key, blk in dd["configurations"].items():
        mode, scen, _model, _topo = key.split("|")
        for _edge, cov in blk["coverage"].items():
            runs[mode][scen] += cov["n_paired"]
            runs[mode]["edges"] += 1

    mas = json.loads((REPO / "data" / "mas_features.json").read_text())
    f2 = collections.Counter()
    for r in mas["records"]:
        for k in ("f2_n_occurrences", "f2_n_tasks", "f2_skipped_occurrences",
                  "f1_n_tasks"):
            f2[k] += (r.get(k) or 0)

    occ = int(f2["f2_n_occurrences"])
    arch = f1_archival_record()
    return {
        "note": ("Counted from the artifacts that recorded the work, not "
                 "estimated. A 'paired run' is one intervened workflow "
                 "execution matched to its clean counterpart."),
        "graph_features": {
            "features": GRAPH + [F3],
            "cost_class": "graph-only",
            "workflow_executions": 0, "llm_calls": 0, "embedding_calls": 0,
            "detail": ("Computed from the topology YAML with NetworkX. No "
                       "model is loaded and no workflow runs; cost is "
                       "milliseconds per configuration and independent of the "
                       "number of tasks."),
        },
        "ablation_delta": {
            "cost_class": "workflow re-execution",
            "paired_workflow_executions": int(runs["ablation"]["customer_service"]
                                              + runs["ablation"]["software_engineering"]),
            "by_domain": {s: int(runs["ablation"][s]) for s in SCENARIOS},
            "edge_configuration_cells": int(runs["ablation"]["edges"]),
            "detail": ("One clean workflow re-run per (edge, task) with the "
                       "edge removed, paired against the unmodified run."),
        },
        "perturbation_delta": {
            "cost_class": "workflow re-execution",
            "paired_workflow_executions": int(runs["perturbation"]["customer_service"]
                                              + runs["perturbation"]["software_engineering"]),
            "by_domain": {s: int(runs["perturbation"][s]) for s in SCENARIOS},
            "edge_configuration_cells": int(runs["perturbation"]["edges"]),
            "detail": ("One clean workflow re-run per (edge, task) with noise "
                       "injected on the edge, paired against the unmodified run."),
        },
        "receiver_response_sensitivity_f2": {
            "cost_class": "receiver probe generation + embedding",
            "receiver_probe_generations": occ,
            "embedding_calls": 2 * occ,
            "n_tasks_covered": int(f2["f2_n_tasks"]),
            "skipped_occurrences": int(f2["f2_skipped_occurrences"]),
            "detail": ("One extra receiver generation per message occurrence "
                       "with the message replaced by a neutral placeholder, "
                       "then two sentence-transformer embeddings per occurrence "
                       "(clean output and probe output). The most expensive "
                       "candidate in the inventory, and excluded by the freeze."),
        },
        "consequence_proximity_f3": {
            "cost_class": "graph-only",
            "workflow_executions": 0, "llm_calls": 0, "embedding_calls": 0,
            "detail": ("Shortest-path distance from the receiver to the nearest "
                       "predeclared sink. Same cost class as the six graph "
                       "features; excluded by the freeze on performance, not "
                       "on cost."),
        },
        "semantic_non_recoverability_f1": {
            "cost_class": "clean transcripts + embedding",
            "receiver_probe_generations": 0,
            "n_tasks_covered_current_matrix": int(f2["f1_n_tasks"]),
            "n_tasks_covered_archival": arch.get("f1_task_evaluations", 0),
            "archival_records": arch.get("n_records_with_f1", 0),
            "archival_configurations": arch.get("n_configurations", 0),
            "detail": (
                "COMPUTED, then rejected. The archival two-model study "
                "evaluated F1 over %d tasks across %d configurations at full "
                "coverage; no receiver probe is needed, because F1 is read "
                "off clean transcripts, but every message and every "
                "alternative context must be embedded. It was rejected for "
                "confounding with %s rather than for cost, and was not "
                "rebuilt for the current matrix -- which is why f1_n_tasks "
                "sums to %d there. The zero is the cost of NOT rebuilding it, "
                "not evidence that it was never run."
                % (arch.get("f1_task_evaluations", 0),
                   arch.get("n_configurations", 0), F1_CONFOUND,
                   int(f2["f1_n_tasks"]))),
        },
    }


# ------------------------------------------------------------------ assemble

def check(payload):
    """Refuse to write an audit whose denominators moved."""
    d = payload["denominators"]
    if d["n_configurations_total"] != EXPECT_CONFIGURATIONS_TOTAL:
        raise SystemExit("configuration count: expected %d, got %d"
                         % (EXPECT_CONFIGURATIONS_TOTAL,
                            d["n_configurations_total"]))
    for sc in SCENARIOS:
        n = payload["redundancy"][sc]["n_configurations"]
        if n != EXPECT_CONFIGURATIONS_PER_DOMAIN:
            raise SystemExit("%s: expected %d configurations, got %d"
                             % (sc, EXPECT_CONFIGURATIONS_PER_DOMAIN, n))
        if payload["leave_one_out"][sc]["n_configurations"] != \
                EXPECT_CONFIGURATIONS_PER_DOMAIN:
            raise SystemExit("%s: leave-one-out configuration count moved" % sc)
    if payload["canonical_block"] != CANONICAL:
        raise SystemExit("canonical block is not %s" % CANONICAL)
    if len(BLOCKS[CANONICAL]) != N_DECLARED:
        raise SystemExit("canonical block is not %d features" % N_DECLARED)

    re_ = payload["rank_equivalence"]
    if re_["n_checks"] != EXPECT_DIRECTIONS_TOTAL:
        raise SystemExit("rank equivalence: expected %d configuration-"
                         "directions, checked %d"
                         % (EXPECT_DIRECTIONS_TOTAL, re_["n_checks"]))
    if not re_["all_identical"]:
        raise SystemExit(
            "RANK EQUIVALENCE FAILED -- the active-only score does not "
            "reproduce the declared-eight ordering: %s" % re_["failures"][:5])

    enf = payload["enforcement_monitored_sets"]
    if enf["n_configurations"] != EXPECT_ENFORCEMENT_CONFIGURATIONS:
        raise SystemExit("enforcement: expected %d configurations, found %d"
                         % (EXPECT_ENFORCEMENT_CONFIGURATIONS,
                            enf["n_configurations"]))
    if not enf["declared_eight_reproduces_executed"]:
        raise SystemExit(
            "STOP -- the DECLARED eight-feature score does not reproduce the "
            "executed monitored sets, so nothing can be concluded about the "
            "smaller schema from them. This is a provenance problem in the "
            "enforcement artifacts, not a feature-selection result: %s"
            % json.dumps(enf["declared_eight_mismatches"][:5], indent=1))
    # Tie-boundary changes are reported; non-tie changes violate equivalence.
    real = [c for c in enf["changes"] if not c["boundary_is_a_tie"]]
    if real:
        raise SystemExit(
            "STOP -- MONITORED SETS CHANGED WHERE THE SCORES GENUINELY "
            "DIFFER. This contradicts the rank equivalence and means the "
            "smaller schema does not describe the executed defence. No "
            "artifact and no code is modified: %s"
            % json.dumps(real[:5], indent=1))

    # F1 must never be described as unmeasured. It was measured and rejected,
    # and the difference matters: one is an omission, the other is a result.
    a = payload.get("f1_archival") or {}
    if a.get("available"):
        if not a.get("n_records_with_f1"):
            raise SystemExit("F1 archival record carries no measured records")
        allr = (a.get("confounding") or {}).get("all_archival_records") or {}
        if allr.get("spearman_rho") is None:
            raise SystemExit("F1 rejection must carry its confounding "
                             "statistic, not an assertion")
        if a.get("rebuilt_for_current_matrix"):
            raise SystemExit("F1 is recorded as rebuilt but is absent from "
                             "the current matrix")
        blob = payload["inventory"][F1]["reason"].lower()
        if "never measured" in blob or "does not exist" in blob:
            raise SystemExit("the F1 inventory entry still says the feature "
                             "was never measured; it was measured on %d "
                             "records and rejected"
                             % a["n_records_with_f1"])
        for sc in SCENARIOS:
            if F1 in [r["block"] for r in
                      payload["block_comparison"][sc]["rows"]]:
                raise SystemExit(
                    "F1 entered the equal-coverage block comparison; it has "
                    "0.0 coverage on the current matrix and must stay out.")

    # The post-hoc arms must say so in the artifact, not only in prose.
    for sc in SCENARIOS:
        for section in ("leave_one_out", "block_comparison"):
            s = payload[section][sc]
            if s.get("analysis_type") != "post-hoc sensitivity analysis" \
                    or s.get("is_feature_selection") is not False:
                raise SystemExit(
                    "%s/%s must be labelled a post-hoc sensitivity analysis "
                    "and disclaim feature selection" % (section, sc))

    # An inert feature must have exactly zero leave-one-out effect. If it does
    # not, the "constant features are inert" claim is false somewhere.
    inert = [n for n, v in payload["inventory"].items()
             if v.get("available") and v.get("n_configurations_varies") == 0]
    for sc in SCENARIOS:
        for r in payload["leave_one_out"][sc]["rows"]:
            if r["removed"] in inert and not r["inert"]:
                raise SystemExit(
                    "%s: %s is constant everywhere but removing it changed the "
                    "score -- the inertness argument is broken"
                    % (sc, r["removed"]))


def main():
    mats = {}
    for sc in SCENARIOS:
        fm, dropped, _m = load_enriched_directional(scenarios=[sc])
        mats[sc] = fm

    inv = inventory(mats)
    active_always = [n for n, v in inv.items()
                     if v.get("available")
                     and v.get("n_configurations_varies") == EXPECT_CONFIGURATIONS_TOTAL
                     and v["in_canonical_block"]]
    inert = [n for n, v in inv.items()
             if v.get("available") and v.get("n_configurations_varies") == 0]

    payload = {
        "note": ("Feature-selection audit. Existing data only; no experiment "
                 "was launched and the feature freeze is unchanged. The "
                 "outcome-blind sections (inventory, redundancy, rank "
                 "equivalence, monitored sets) never see an attack label; the "
                 "leave-one-out and block sections are post-hoc explanatory "
                 "diagnostics and are not a selection rule."),
        "canonical_block": CANONICAL,
        "n_features_declared": N_DECLARED,
        "declared_features": list(BLOCKS[CANONICAL]),
        "denominators": {
            "n_configurations_total": sum(
                len({g.rsplit("|", 1)[0] for g in mats[s].groups})
                for s in SCENARIOS),
            "n_configurations_per_domain": {
                s: len({g.rsplit("|", 1)[0] for g in mats[s].groups})
                for s in SCENARIOS},
            "n_configuration_directions_total": sum(
                len(np.unique(mats[s].groups)) for s in SCENARIOS),
            "n_edge_rows_per_domain": {s: int(mats[s].X.shape[0])
                                       for s in SCENARIOS},
            "n_enforcement_configurations": EXPECT_ENFORCEMENT_CONFIGURATIONS,
            "rank_equivalence_budgets": list(EQUIV_BUDGETS),
            "diagnostic_budget": BUDGET,
        },
        "inventory": inv,
        "f1_archival": f1_archival_record(),
        "float_tie_sensitivity": float_tie_sensitivity(mats),
        "redundancy": redundancy(mats),
        "rank_equivalence": rank_equivalence(mats),
        "enforcement_monitored_sets": enforcement_monitored_sets(),
        "leave_one_out": leave_one_out(mats),
        "block_comparison": block_comparison(mats),
        "cost": cost_table(),
    }
    payload["tie_behaviour"] = tie_behaviour(
        payload["enforcement_monitored_sets"])
    enf = payload["enforcement_monitored_sets"]
    re_ = payload["rank_equivalence"]
    payload["verdict"] = {
        "inert_features": inert,
        "always_active_features": active_always,
        "effective_schema_size_median": re_["n_active_median"],
        "effective_schema_size_range": [re_["n_active_min"], re_["n_active_max"]],
        "monitored_sets_unchanged": enf["all_identical"],
        "recommended_description": (
            "an eight-feature declared schema with outcome-blind local "
            "activation"),
        "why": (
            "A single smaller schema does not exist. Which features are active "
            "is a property of the configuration, not of the study: it ranges "
            "from %d to %d features across the %d configuration-directions, so "
            "fixing one subset globally would either drop a feature that "
            "carries the ordering in some topology or keep one that is inert "
            "in most. %s %s constant in every configuration and can be "
            "reported as declared-but-inert without moving a single number, "
            "which is exactly what rank equivalence establishes."
            % (re_["n_active_min"], re_["n_active_max"], re_["n_checks"],
               " and ".join("`%s`" % f for f in inert) or "No feature is",
               "are" if len(inert) != 1 else "is")),
        "blocking_finding": (None if enf["all_identical"] else {
            "what": ("%d of the %d executed allocation cells select a "
                     "different monitored set under the active-only score."
                     % (enf["n_changed"], enf["n_allocation_cells"])),
            "why_it_is_not_a_ranking_difference": (
                "All %d are cells where the k-th and (k+1)-th edges score "
                "IDENTICALLY under the declared eight. The ranking is the same "
                "-- rank equivalence holds at every one of the %d "
                "configuration-directions -- but a top-k SET is not determined "
                "by a ranking when the budget boundary falls inside a tie. "
                "The executed run resolved those ties on IEEE summation order, "
                "which is reproducible but arbitrary."
                % (enf["n_changed_that_are_boundary_ties"], re_["n_checks"])),
            "scope": ("%d of %d cells have a tie at the budget boundary, so "
                      "this is not a rare edge case at these budgets. "
                      "Re-deriving the allocation from the declared eight "
                      "reproduces all %d executed sets exactly; rounding the "
                      "same score to %d decimals already changes %d of them."
                      % (enf["n_cells_with_boundary_tie"],
                         enf["n_allocation_cells"],
                         enf["n_cells_matching"].get("declared", 0),
                         ROUND_DECIMALS,
                         enf["n_allocation_cells"]
                         - enf["n_cells_matching"].get("declared_rounded", 0))),
            "consequence": (
                "Do not adopt a smaller effective schema in the paper on the "
                "strength of rank equivalence alone: it would change which "
                "edges the reported defence monitored in %d cells, even though "
                "no edge changed rank. The blocker is a missing tie-breaking "
                "rule, not the feature set."
                % enf["n_changed"]),
            "recommended_next_step": (
                "Declare an explicit, score-independent tie-break for top-k "
                "selection and state it in the paper. Nothing here requires "
                "re-running an experiment; the enforcement artifacts are "
                "untouched and remain valid under the declared eight."),
        }),
        "caveat": (
            "This is a REPORTING recommendation. It does not change the "
            "freeze, the score, or any published number -- rank equivalence is "
            "exactly the statement that the ORDERING cannot change. Top-k set "
            "selection under ties is a separate question, and it is the one "
            "that blocks the smaller schema."),
    }

    check(payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    write_markdown(payload)

    print("FEATURE SELECTION AUDIT")
    print("  configurations      : %d (%s)"
          % (payload["denominators"]["n_configurations_total"],
             payload["denominators"]["n_configurations_per_domain"]))
    print("  inert features      : %s" % (", ".join(inert) or "none"))
    print("  active per config   : %d-%d (median %.1f) of %d declared"
          % (payload["rank_equivalence"]["n_active_min"],
             payload["rank_equivalence"]["n_active_max"],
             payload["rank_equivalence"]["n_active_median"], N_DECLARED))
    print("  rank equivalence    : %s"
          % ("IDENTICAL at %s" % ", ".join("%d%%" % round(b * 100)
                                           for b in EQUIV_BUDGETS)))
    e = payload["enforcement_monitored_sets"]
    print("  monitored sets      : %d/%d cells reproduce the executed set "
          "(declared-eight control %d/%d)"
          % (e["n_cells_matching"].get("active", 0), e["n_allocation_cells"],
             e["n_cells_matching"].get("declared", 0), e["n_allocation_cells"]))
    if not e["all_identical"]:
        print("  *** STOP: %d monitored set(s) CHANGED, all %d at a budget-"
              "boundary tie (%d/%d cells tie there). Reporting only; no "
              "artifact or code modified."
              % (e["n_changed"], e["n_changed_that_are_boundary_ties"],
                 e["n_cells_with_boundary_tie"], e["n_allocation_cells"]))
        for c in e["changes"]:
            print("      %s/%s %s @%d%%: -%s +%s"
                  % (c["model"], c["topology"], c["direction"],
                     round(c["budget"] * 100),
                     ",".join("%s->%s" % tuple(x) for x in c["dropped"]),
                     ",".join("%s->%s" % tuple(x) for x in c["added"])))
    print("wrote %s" % OUT.relative_to(REPO))
    print("wrote %s" % OUT_MD.relative_to(REPO))
    return 0


def write_markdown(p):
    L = ["# Feature-selection audit", "",
         "Generated by `python analysis/feature_selection_audit.py`. "
         "Existing data only; the freeze is unchanged. "
         "**%d configurations** (%s), %d configuration-directions."
         % (p["denominators"]["n_configurations_total"],
            ", ".join("%s %d" % (k.split("_")[0], v) for k, v in
                      p["denominators"]["n_configurations_per_domain"].items()),
            p["denominators"]["n_configuration_directions_total"]), "",
         "## 1. Candidate inventory", "",
         "`varies` counts configurations in which the feature is non-constant "
         "across that configuration's edges, out of %d. A constant feature is "
         "inert under within-configuration normalization."
         % p["denominators"]["n_configurations_total"], "",
         "| feature | mechanism | cost class | coverage | varies | median "
         "distinct | status |", "|---|---|---|---|---|---|---|"]
    for n, v in p["inventory"].items():
        vr = ("%d/%d (%.0f%%)" % (v["n_configurations_varies"],
                                  p["denominators"]["n_configurations_total"],
                                  100 * v["fraction_configurations_varies"])
              if v.get("available") else "—")
        L.append("| `%s` | %s | %s | %.2f | %s | %s | %s |"
                 % (n, v["mechanism"].split(":", 1)[0], v["cost_class"],
                    v["coverage"], vr,
                    "%.1f" % v["median_distinct_values"]
                    if v.get("median_distinct_values") is not None else "—",
                    v["status"]))

    a = p.get("f1_archival") or {}
    if a.get("available"):
        cf = a["confounding"]
        L += ["", "### F1 was measured, then rejected — not skipped", "",
              "`%s` is absent from the current matrix (coverage 0.00 above), "
              "which is a fact about **this** matrix and not about whether the "
              "feature was ever computed. The archival two-model study "
              "(`%s`) measured it at %.0f%% coverage: **%d records** over "
              "**%d task evaluations**, %d configurations, %s."
              % (F1, a["source"], 100 * a["coverage"], a["n_records_with_f1"],
                 a["f1_task_evaluations"], a["n_configurations"],
                 " and ".join(a["models"])), "",
              "It was rejected because its primary statistic `%s` is strongly "
              "confounded with `%s`, the number of alternative contexts "
              "available to the receiver:"
              % (a["primary_statistic"], cf["against"]), "",
              "| stratum | n | Spearman ρ vs `%s` | p |" % cf["against"],
              "|---|---|---|---|"]
        rows = [("all archival records", cf["all_archival_records"])]
        rows += [(sc, r) for sc, r in sorted(cf["by_scenario"].items()) if r]
        for lab, r in rows:
            L.append("| %s | %d | %+.3f | %.1e |"
                     % (lab, r["n"], r["spearman_rho"], r["p_value"]))
        L += ["", "That count is essentially receiver in-degree, which the "
              "structural block already supplies at no marginal cost, so F1 "
              "largely re-expresses a graph quantity rather than adding "
              "semantic information. **F1 was therefore not rebuilt for the "
              "expanded %d-configuration matrix**, and it is deliberately held "
              "out of the equal-coverage block comparison in section 6: it "
              "exists on %d archival two-model configurations against %d "
              "three-model ones there, so including it would vary the feature "
              "set and the sample in the same step."
              % (p["denominators"]["n_configurations_total"],
                 a["n_configurations"],
                 p["denominators"]["n_configurations_total"]), ""]

    fs = p["float_tie_sensitivity"]
    L += ["", "### Reconciliation with `data/canonical_ranking.json`", "",
          "Every diagnostic below scores on values rounded to 1e-%d, so that a "
          "feature which provably cannot change an ordering is credited with "
          "exactly zero. The canonical table does not round. The difference:"
          % ROUND_DECIMALS, "",
          "| domain | cov-AUC (canonical) | cov-AUC (rounded) | Δ | cov@20% Δ |",
          "|---|---|---|---|---|"]
    for sc in SCENARIOS:
        v = fs[sc]
        L.append("| %s | %.6f | %.6f | %+.6f | %+.6f |"
                 % (sc, v["coverage_auc_unrounded"], v["coverage_auc_rounded"],
                    v["coverage_auc_difference"], v["coverage_at_20_difference"]))
    L += ["", "coverage@20% is bit-identical either way. coverage-AUC differs "
          "in the fourth decimal because it integrates over every cutoff, so "
          "every split tie reaches it. No conclusion changes.", ""]

    L += ["", "## 2. Outcome-blind redundancy (per domain, never pooled)", ""]
    for sc in SCENARIOS:
        r = p["redundancy"][sc]
        L += ["### %s — top pairs by median |ρ| (%d configurations)"
              % (sc, r["n_configurations"]), "",
              "%d pairs: %d evaluable, %d never evaluable (a constant feature "
              "on at least one side in every configuration). Of the evaluable, "
              "%d have median |ρ| > %.2f and %d > %.2f. Fractions divide by "
              "**evaluable** directions; undefined correlations are never "
              "counted as zero."
              % (r["n_pairs"], r["n_pairs_evaluable"],
                 r["n_pairs_never_evaluable"], r["n_pairs_median_above_flag"],
                 r["flag_threshold"], r["n_pairs_median_above_duplicate"],
                 r["duplicate_threshold"]), "",
              "| pair | median &#124;ρ&#124; | evaluable | undefined | >0.80 | >0.90 |",
              "|---|---|---|---|---|---|"]
        for pr in r["pairs"][:8]:
            L.append("| `%s` × `%s` | %s | %d/%d | %d | %s | %s |"
                     % (pr["a"], pr["b"],
                        "%.3f" % pr["median_abs_spearman"]
                        if pr["median_abs_spearman"] is not None else "undefined",
                        pr["n_directions_evaluable"], pr["n_directions_total"],
                        pr["n_directions_undefined"],
                        "%d (%.0f%%)" % (pr["n_above_0.80"],
                                         100 * pr["fraction_above_0.80"])
                        if pr["fraction_above_0.80"] is not None else "—",
                        "%d (%.0f%%)" % (pr["n_above_0.90"],
                                         100 * pr["fraction_above_0.90"])
                        if pr["fraction_above_0.90"] is not None else "—"))
        L.append("")

    re_ = p["rank_equivalence"]
    L += ["## 3. Rank equivalence: declared eight vs active-only", "",
          "Checked on all **%d configuration-directions** at %s. "
          "Identical tie-aware ranks and identical top-k sets: **%s**."
          % (re_["n_checks"],
             ", ".join("%d%%" % round(b * 100) for b in re_["budgets"]),
             "yes, everywhere" if re_["all_identical"] else "NO"), "",
          "Active features per configuration-direction: %d–%d, median %.1f of "
          "%d declared. Distribution: %s."
          % (re_["n_active_min"], re_["n_active_max"], re_["n_active_median"],
             p["n_features_declared"],
             ", ".join("%s active in %d" % (k, v) for k, v in
                       re_["distribution_n_active"].items())), ""]

    enf = p["enforcement_monitored_sets"]
    L += ["## 4. Executed enforcement monitored sets", "",
          "%d configurations, %d allocation cells. Comparison uses the "
          "unrounded score, because that is what the executed run used."
          % (enf["n_configurations"], enf["n_allocation_cells"]), "",
          "| ordering | cells reproducing the executed set |", "|---|---|"]
    for t, lab in (("declared", "declared eight (control)"),
                   ("active", "**active-only**"),
                   ("declared_rounded", "declared eight, rounded to 1e-%d"
                    % ROUND_DECIMALS),
                   ("active_rounded", "active-only, rounded to 1e-%d"
                    % ROUND_DECIMALS)):
        L.append("| %s | %d / %d |" % (lab, enf["n_cells_matching"].get(t, 0),
                                       enf["n_allocation_cells"]))
    L += ["", "%d of %d cells have a **tie at the budget boundary**: the k-th "
          "and (k+1)-th edges score identically, so no ordering distinguishes "
          "them and the selection falls to whatever the sort does with a tie."
          % (enf["n_cells_with_boundary_tie"], enf["n_allocation_cells"]), ""]
    if not enf["all_identical"]:
        L += ["> **STOP — %d monitored set(s) change under the active-only "
              "score**, all %d of them at such a tie. The ranking is "
              "unchanged (section 3); a top-k *set* simply is not determined "
              "by a ranking when the budget boundary falls inside a tie. "
              "No artifact or code was modified."
              % (enf["n_changed"], enf["n_changed_that_are_boundary_ties"]), ""]
        for c in enf["changes"]:
            L.append("- `%s`/`%s` %s @%d%%: dropped %s, added %s"
                     % (c["model"], c["topology"], c["direction"],
                        round(c["budget"] * 100),
                        ", ".join("`%s→%s`" % tuple(e) for e in c["dropped"]),
                        ", ".join("`%s→%s`" % tuple(e) for e in c["added"])))
        L.append("")

    tb = p["tie_behaviour"]
    o = tb["observed_in_the_executed_sweep"]
    L += ["## 4b. Exact top-k tie behaviour", "",
          "*%s*" % tb["scope"], "", "**Rule as implemented.**", ""]
    L += ["%s" % s for s in tb["rule_as_implemented"]]
    L += ["", "**Consequences.**", ""]
    L += ["- %s" % s for s in tb["consequences"]]
    L += ["", "In the executed sweep: %d of %d cells tie at the budget "
          "boundary. The declared eight as executed (unrounded) reproduces "
          "**%d/%d** monitored sets exactly; the same score rounded to 1e-%d "
          "reproduces %d/%d."
          % (o["n_cells_with_boundary_tie"], o["n_allocation_cells"],
             o["declared_eight_unrounded_reproduces"], o["n_allocation_cells"],
             ROUND_DECIMALS, o["declared_eight_rounded_reproduces"],
             o["n_allocation_cells"]), "",
          "> %s" % tb["recommended_for_future_runs"], ""]

    L += ["## 5. Leave-one-feature-out — POST-HOC SENSITIVITY ANALYSIS", "",
          "**Not feature selection.** These per-feature results describe how "
          "the *frozen* block behaves when perturbed. They were computed after "
          "the freeze, on the data the freeze used, and no feature is added, "
          "removed or reweighted on the strength of any number here. The "
          "ordering of these deltas is not a recommended subset.", "",
          "Δ is *reduced minus full*; negative means removing it hurt. "
          "Holm-corrected within each (domain, metric) family.", ""]
    for sc in SCENARIOS:
        d = p["leave_one_out"][sc]
        L += ["### %s (n=%d, budget %d%%)" % (sc, d["n_configurations"],
                                              round(d["budget"] * 100)), "",
              "| removed | Δ cov-AUC | 95% CI | Holm p | Δ cov@20% | 95% CI | "
              "Holm p |", "|---|---|---|---|---|---|---|"]
        for r in d["rows"]:
            a, c = r["delta_auc"], r["delta_coverage_at_20"]
            L.append("| `%s` | %+.4f | [%+.4f, %+.4f] | %.3f | %+.4f | "
                     "[%+.4f, %+.4f] | %.3f |"
                     % (r["removed"], a["mean"], a["lo"], a["hi"],
                        r["holm_p_auc"], c["mean"], c["lo"], c["hi"],
                        r["holm_p_coverage_at_20"]))
        L.append("")

    L += ["## 6. Mechanistic block comparison — POST-HOC SENSITIVITY ANALYSIS",
          "",
          "**Not alternative feature selection.** Seven predeclared blocks, no "
          "exhaustive subset search, and no block is promoted by its score. "
          "Every block is scored on the same %d configurations at equal "
          "coverage, which is what makes the contrast about the feature set "
          "alone — and is why F1 is absent (see section 1). `graph_only`, "
          "`graph_ablation` and `graph_perturbation` are single-probe "
          "sensitivity arms: `graph_perturbation` scoring at or above the "
          "canonical block in a domain says how little the second probe adds "
          "*there*, not that it is a candidate replacement for the frozen "
          "eight." % p["denominators"]["n_configurations_per_domain"][
              "customer_service"], ""]
    for sc in SCENARIOS:
        d = p["block_comparison"][sc]
        L += ["### %s (n=%d)" % (sc, d["n_configurations"]), "",
              "| block | features | median active | cov-AUC | cov@20% | "
              "Δ cov-AUC vs canonical |", "|---|---|---|---|---|---|"]
        for r in d["rows"]:
            mark = " **(canonical)**" if r["block"] == d["reference"] else ""
            L.append("| `%s`%s | %d | %.1f | %.3f | %.3f | %+.4f [%+.4f, %+.4f] |"
                     % (r["block"], mark, r["n_features"],
                        r["median_active_features"], r["auc"],
                        r["coverage_at_20"],
                        r["delta_auc_vs_canonical"]["mean"],
                        r["delta_auc_vs_canonical"]["lo"],
                        r["delta_auc_vs_canonical"]["hi"]))
        L.append("")

    c = p["cost"]
    L += ["## 7. Cost, counted from the logs", "",
          "| block | cost class | workflow executions | LLM/probe calls | "
          "embeddings |", "|---|---|---|---|---|",
          "| six graph features + F3 | graph-only | 0 | 0 | 0 |",
          "| ablation | workflow re-execution | %d | — | 0 |"
          % c["ablation_delta"]["paired_workflow_executions"],
          "| perturbation | workflow re-execution | %d | — | 0 |"
          % c["perturbation_delta"]["paired_workflow_executions"],
          "| F2 | receiver probe + embedding | 0 | %d | %d |"
          % (c["receiver_response_sensitivity_f2"]["receiver_probe_generations"],
             c["receiver_response_sensitivity_f2"]["embedding_calls"]),
          "| F1 (archival, rejected) | clean transcripts + embedding | 0 | 0 | "
          "%d task evaluations |"
          % c["semantic_non_recoverability_f1"]["n_tasks_covered_archival"],
          "", "F1's row is archival: it was computed and rejected, and not "
          "rebuilt for the current matrix. The zero in the current matrix is "
          "the cost of not rebuilding it, not evidence it was never run.", "",
          "## Verdict", "",
          "**%s**" % p["verdict"]["recommended_description"], "",
          p["verdict"]["why"], "", "*%s*" % p["verdict"]["caveat"], ""]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
