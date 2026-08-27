"""Build the full coverage curve for the frozen eight-feature score.

The 10% and 20% points must match the canonical table and are the only budgets
with inference. Other points are descriptive. Random coverage is the exact
``k/|E|`` expectation; the oracle is an outcome-aware upper bound. Domains
remain separate.
"""

import collections
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.run_mesa_fit import (load_enriched_directional,  # noqa: E402
                                   collapse_directions, BLOCKS)
from analysis.run_mesa_cv import _binary_idx                   # noqa: E402
from analysis.run_nested_cv import expected_tie_credit         # noqa: E402
from src.saliency.mesa_scores import MesaLocal                 # noqa: E402
from src.saliency.normalization import normalize_within        # noqa: E402

OUT = REPO / "data" / "canonical_ranking_curve.json"
CANONICAL_SRC = REPO / "data" / "canonical_ranking.json"

SCENARIOS = ("customer_service", "software_engineering")
MODELS = ("gemma4:e4b", "qwen3.5:9b", "llama3.1:8b")
BUDGETS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00)
PREDECLARED = (0.10, 0.20)              # the only budgets carrying inference
CANONICAL = "structural_dynamic"        # frozen; see freeze_feature_decision.py
N_FEATURES = 8
EXPECT_CONFIGURATIONS_PER_DOMAIN = 15
# ``struct`` is the six-feature graph-only baseline.
METHODS = ("mesa", "struct", "dyn", "random", "oracle")
# ``dyn`` is curve-only; the other arms must match the canonical table.
METHODS_VS_CANONICAL = ("mesa", "struct", "random", "oracle")

# Canonical rows may use corrected routing or audited unchanged legacy routing.
ALLOWED_NAMESPACES = {"routing_v0_valid", "routing_v1"}
FORBIDDEN_NAMESPACES = {"routing_v0_invalid"}

# Reject legacy rows from protocols changed by the routing repair.
ROUTING_UNCHANGED = {"sequential", "centralized", "decentralized"}
ROUTING_REPAIRED = {"hierarchical", "hybrid", "mesh"}
REUSED_NAMESPACE = "routing_v0_valid"

# Must match analysis/canonical_ranking_table.py exactly, or the bootstrap CIs
# at the predeclared budgets will not reproduce.
B = 10000
SEED = 20260817


def _cov(y, s, k):
    if y.sum() <= 0:
        return float("nan")
    return float(expected_tie_credit(y, s)[:k].sum() / y.sum())


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
    """Exact paired sign-flip over all 2^n patterns."""
    d = np.asarray(list(diffs), dtype=float)
    n = len(d)
    if n == 0 or not np.any(np.abs(d) > 0):
        return {"n": int(n), "p_two_sided": 1.0,
                "min_attainable_two_sided": (2.0 / 2 ** n) if n else None}
    if n > 20:
        rng = np.random.default_rng(SEED)
        signs = rng.choice((1.0, -1.0), size=(200000, n))
        exact = False
    else:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        exact = True
    obs = float(d.mean())
    means = signs.dot(d) / n
    eps = 1e-12
    return {"n": int(n), "observed_mean": obs, "exact": exact,
            "p_two_sided": float(np.mean(np.abs(means) >= abs(obs) - eps)),
            "n_patterns": int(len(signs)),
            "min_attainable_two_sided": 2.0 / len(signs)}


def tag(b):
    """Budget -> the integer suffix the canonical table uses (0.10 -> '10')."""
    return "%d" % round(b * 100)


# ------------------------------------------------------------- guard rails

def assert_clean_inputs(fm, scenario):
    """No superseded routing, no pooled domain, no mixed cross-fit direction."""
    namespaces = {r.get("routing_namespace") for r in fm.rows}
    bad = namespaces & FORBIDDEN_NAMESPACES
    if bad:
        raise SystemExit("SUPERSEDED ROUTING reached the ranking matrix: %s. "
                         "Those rows describe a graph that no longer exists."
                         % sorted(bad))
    unknown = namespaces - ALLOWED_NAMESPACES
    if unknown:
        raise SystemExit("UNKNOWN routing namespace(s) %s; allowed %s."
                         % (sorted(unknown), sorted(ALLOWED_NAMESPACES)))

    scen = {r.get("scenario") for r in fm.rows}
    if scen != {scenario}:
        raise SystemExit("DOMAIN POOLING: matrix for %s carries scenarios %s. "
                         "Customer service and software engineering are "
                         "analysed and reported separately, never pooled."
                         % (scenario, sorted(scen)))

    dirs = collections.defaultdict(set)
    for g, r in zip(fm.groups, fm.rows):
        dirs[g].add(r.get("direction"))
    mixed = {g: sorted(v) for g, v in dirs.items() if len(v) > 1}
    if mixed:
        raise SystemExit("CROSS-FIT LEAK: A->B and B->A share a group: %s"
                         % mixed)


def audit_routing_provenance(fm, scenario):
    """Where does every reused (non-rerun) row come from?

    Returns the provenance record that goes into the artifact. The point is to
    make the reuse legible instead of letting "the canonical matrix" stand in
    for "everything was measured under the current routing". It was not: a
    substantial minority of rows are original measurements, admissible only
    because the repair did not touch their protocol.
    """
    rows_by_ns = collections.Counter()
    topos_by_ns = collections.defaultdict(set)
    cfgs_by_ns = collections.defaultdict(set)
    for r in fm.rows:
        ns = r.get("routing_namespace")
        rows_by_ns[ns] += 1
        topos_by_ns[ns].add(r["topology"])
        cfgs_by_ns[ns].add((r["model"], r["topology"]))

    # THE ASSERTION. A reused row from a repaired protocol describes a graph
    # that no longer exists, and would be indistinguishable in the matrix from
    # a valid one.
    offending = sorted(topos_by_ns.get(REUSED_NAMESPACE, set())
                       & ROUTING_REPAIRED)
    if offending:
        raise SystemExit(
            "REUSED ROWS FROM A REPAIRED PROTOCOL in %s: %s carry %s. Those "
            "measurements describe pre-repair routing and may not be reused; "
            "only topologies the repair provably did not touch (%s) may."
            % (scenario, offending, REUSED_NAMESPACE,
               sorted(ROUTING_UNCHANGED)))
    unexpected = sorted(topos_by_ns.get(REUSED_NAMESPACE, set())
                        - ROUTING_UNCHANGED)
    if unexpected:
        raise SystemExit(
            "REUSED ROWS FROM AN UNCLASSIFIED PROTOCOL in %s: %s. A topology "
            "that is neither audited-unchanged nor known-repaired cannot be "
            "asserted safe." % (scenario, unexpected))

    total = sum(rows_by_ns.values())
    reused = rows_by_ns.get(REUSED_NAMESPACE, 0)
    return {
        "n_edge_rows": int(total),
        "rows_by_namespace": {k: int(v) for k, v in sorted(rows_by_ns.items())},
        "n_rows_reused_not_rerun": int(reused),
        "fraction_rows_reused_not_rerun": reused / float(total) if total else 0.0,
        "audited_routing_unchanged_configurations": sorted(
            "%s/%s" % c for c in cfgs_by_ns.get(REUSED_NAMESPACE, set())),
        "n_audited_routing_unchanged_configurations": len(
            cfgs_by_ns.get(REUSED_NAMESPACE, set())),
        "topologies_reused": sorted(topos_by_ns.get(REUSED_NAMESPACE, set())),
        "topologies_rerun_under_routing_v1": sorted(
            topos_by_ns.get("routing_v1", set())),
        "claim": (
            "AUDITED ROUTING-UNCHANGED CONFIGURATIONS. %d of %d edge rows "
            "(%.0f%%) are original `%s` measurements that were NOT re-run "
            "under routing-v1. They are admissible because every one of them "
            "comes from a protocol the routing repair provably did not touch "
            "(%s); the repaired protocols (%s) carry routing_v1 rows only, "
            "which is asserted here rather than assumed. This artifact does "
            "NOT claim that every underlying row was newly re-executed under "
            "routing-v1."
            % (reused, total, 100.0 * reused / total if total else 0.0,
               REUSED_NAMESPACE, ", ".join(sorted(ROUTING_UNCHANGED)),
               ", ".join(sorted(ROUTING_REPAIRED)))),
    }


def assert_within_configuration(X, groups, binary):
    """Scoring a group alone must give the identical normalized features.

    A global normalization would rank every edge in the study against every
    other, which mixes within-graph edge prediction with between-topology
    differences. The cheap way to be sure it did not happen is to renormalize
    each configuration-direction in isolation and require the numbers to be
    bit-identical to the ones produced with the whole matrix present.
    """
    Z_all = normalize_within(X, groups, binary)
    for g in np.unique(groups):
        i = np.where(groups == g)[0]
        Z_one = normalize_within(X[i], groups[i], binary)
        if not np.array_equal(np.nan_to_num(Z_all[i], nan=-7.0),
                              np.nan_to_num(Z_one, nan=-7.0)):
            raise SystemExit(
                "GLOBAL NORMALIZATION DETECTED: configuration %s is scored "
                "differently when the rest of the study is present." % g)


# ---------------------------------------------------------------- analysis

def analyse(scenario):
    fm, _dropped, _mats = load_enriched_directional(scenarios=[scenario])
    assert_clean_inputs(fm, scenario)
    provenance = audit_routing_provenance(fm, scenario)

    cols = BLOCKS[CANONICAL]
    if len(cols) != N_FEATURES:
        raise SystemExit("FEATURE BLOCK: %s has %d features, expected %d."
                         % (CANONICAL, len(cols), N_FEATURES))
    idx = [fm.feature_names.index(c) for c in cols]
    names = [fm.feature_names[i] for i in idx]
    binary = _binary_idx(names)
    assert_within_configuration(fm.X[:, idx], fm.groups, binary)
    s_all = MesaLocal(names, binary).score(fm.X[:, idx], fm.groups)

    # Six-feature graph-only baseline, scored on the same path as MESA.
    cols_s = BLOCKS["structural"]
    idx_s = [fm.feature_names.index(c) for c in cols_s]
    names_s = [fm.feature_names[i] for i in idx_s]
    struct_all = MesaLocal(names_s, _binary_idx(names_s)).score(
        fm.X[:, idx_s], fm.groups)

    # DYNAMIC-ONLY arm: the two workflow probes, no graph features. Together
    # with static-only it decomposes the frozen block into the two things it
    # is made of, so a reader can see which half carries the ordering and
    # whether combining them is worth anything.
    cols_d = BLOCKS["dynamic"]
    idx_d = [fm.feature_names.index(c) for c in cols_d]
    names_d = [fm.feature_names[i] for i in idx_d]
    dyn_all = MesaLocal(names_d, _binary_idx(names_d)).score(
        fm.X[:, idx_d], fm.groups)

    per = collections.defaultdict(dict)
    meta = {}
    for g in np.unique(fm.groups):
        sel = np.where(fm.groups == g)[0]
        y = fm.y_success[sel].astype(float)
        E = len(y)
        meta[g] = {"model": fm.rows[sel[0]]["model"],
                   "topology": fm.rows[sel[0]]["topology"], "n_edges": E}
        s = np.nan_to_num(s_all[sel], nan=0.5)
        st = np.nan_to_num(struct_all[sel], nan=0.5)
        dy = np.nan_to_num(dyn_all[sel], nan=0.5)
        for b in BUDGETS:
            t = tag(b)
            k = max(1, min(E, int(math.ceil(b * E))))
            per["k" + t][g] = float(k)
            per["mesa" + t][g] = _cov(y, s, k)
            per["struct" + t][g] = _cov(y, st, k)
            per["dyn" + t][g] = _cov(y, dy, k)
            per["random" + t][g] = k / E            # exact expectation, k/|E|
            per["oracle" + t][g] = _cov(y, y, k)

    # Directions collapse to ONE value per configuration before anything is
    # averaged or resampled: A->B and B->A share tasks and are not independent.
    col = {kk: collapse_directions(v) for kk, v in per.items()}
    cfgs = sorted(col["mesa" + tag(BUDGETS[0])])
    cfg_model, cfg_topology = {}, {}
    for g, m in meta.items():
        for c in cfgs:
            if str(c) in str(g) or str(g) in str(c):
                cfg_model[c] = m["model"]
                cfg_topology[c] = m["topology"]

    def block(sub):
        out = {"n_configurations": len(sub), "curve": []}
        for b in BUDGETS:
            t = tag(b)
            point = {"budget": b,
                     "mean_k": float(np.nanmean([col["k" + t][c] for c in sub])),
                     "predeclared": b in PREDECLARED}
            for meth in METHODS:
                point[meth] = float(np.nanmean([col[meth + t][c] for c in sub]))
            # Store configuration-level spread with the curve.
            point["band"] = {}
            for meth in METHODS:
                v = np.asarray([col[meth + t][c] for c in sub], dtype=float)
                v = v[~np.isnan(v)]
                n = len(v)
                point["band"][meth] = {
                    "n": int(n),
                    "sd": float(v.std(ddof=1)) if n > 1 else 0.0,
                    "sem": float(v.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
                    "ci": _boot(list(v)),
                }
            if b in PREDECLARED:
                d = [col["mesa" + t][c] - col["random" + t][c] for c in sub]
                point["mesa_minus_random"] = _boot(d)
                point["signflip"] = sign_flip(d)
                point["ratio"] = (point["mesa"] / point["random"]
                                  if point["random"] else float("nan"))
            out["curve"].append(point)
        return out

    res = {"n_configurations": len(cfgs),
           "configurations": sorted("%s/%s" % (cfg_model.get(c),
                                               cfg_topology.get(c))
                                    for c in cfgs),
           "routing_provenance": provenance,
           "by_model": {}, "macro": None}
    for m in MODELS:
        sub = [c for c in cfgs if cfg_model.get(c) == m]
        if sub:
            res["by_model"][m] = block(sub)
    res["macro"] = block(cfgs)
    return res


# -------------------------------------------------------------- assertions

def _point(block, b):
    for p in block["curve"]:
        if abs(p["budget"] - b) < 1e-12:
            return p
    raise SystemExit("budget %s missing from curve" % b)


def check(payload):
    """Every invariant this artifact promises, checked before it is written."""
    if payload["canonical_block"] != CANONICAL:
        raise SystemExit("FEATURE BLOCK is %s, expected %s"
                         % (payload["canonical_block"], CANONICAL))
    if payload["n_features"] != N_FEATURES:
        raise SystemExit("n_features is %s, expected %d"
                         % (payload["n_features"], N_FEATURES))

    canon = json.loads(CANONICAL_SRC.read_text())
    if canon["canonical_block"] != CANONICAL:
        raise SystemExit("canonical_ranking.json is on block %s"
                         % canon["canonical_block"])

    for dom, res in payload["scenarios"].items():
        if res["n_configurations"] != EXPECT_CONFIGURATIONS_PER_DOMAIN:
            raise SystemExit(
                "%s has %d configurations, expected %d. A domain quietly "
                "covering less than the study is the failure mode this "
                "assertion exists for."
                % (dom, res["n_configurations"],
                   EXPECT_CONFIGURATIONS_PER_DOMAIN))

        # -- reuse is disclosed, and only from unrepaired protocols ---------
        prov = res.get("routing_provenance")
        if not prov:
            raise SystemExit("%s carries no routing provenance record; reuse "
                             "of non-rerun rows must be disclosed, not "
                             "implicit." % dom)
        bad = sorted(set(prov["topologies_reused"]) - ROUTING_UNCHANGED)
        if bad:
            raise SystemExit(
                "%s reuses non-rerun rows from %s, which the routing repair "
                "changed." % (dom, bad))
        # Every repaired protocol actually present must be routing_v1 only.
        present = {r.split("/")[-1] for r in res["configurations"]}
        for t in sorted(ROUTING_REPAIRED & present):
            if t not in prov["topologies_rerun_under_routing_v1"]:
                raise SystemExit(
                    "%s: repaired protocol %s is present but carries no "
                    "routing_v1 rows." % (dom, t))

        cref = canon["scenarios"][dom]
        blocks = [("macro", res["macro"], cref["macro"])]
        blocks += [(m, res["by_model"][m], cref["by_model"][m])
                   for m in res["by_model"]]

        for label, blk, cblk in blocks:
            if blk["n_configurations"] != cblk["n_configurations"]:
                raise SystemExit("%s/%s configuration count %d != canonical %d"
                                 % (dom, label, blk["n_configurations"],
                                    cblk["n_configurations"]))
            # -- 10% and 20% must reproduce the main table EXACTLY ----------
            for b in PREDECLARED:
                t, p = tag(b), _point(blk, b)
                for meth in METHODS_VS_CANONICAL:
                    got, want = p[meth], cblk["%s%s" % (meth, t)]
                    if not (got == want or (math.isnan(got)
                                            and math.isnan(want))):
                        raise SystemExit(
                            "TRAJECTORY DISAGREES WITH THE MAIN TABLE at "
                            "%s/%s %s@%d%%: %r vs canonical %r. The two "
                            "artifacts would describe different analyses."
                            % (dom, label, meth, round(b * 100), got, want))
                cd = cblk["diff%s" % t]
                gd = p["mesa_minus_random"]
                for key in ("mean", "lo", "hi", "n"):
                    if gd[key] != cd[key]:
                        raise SystemExit(
                            "PAIRED DIFFERENCE at %s/%s @%d%% disagrees with "
                            "the main table on %s: %r vs %r"
                            % (dom, label, round(b * 100), key, gd[key],
                               cd[key]))
            # -- everything is covered at a 100% budget --------------------
            p100 = _point(blk, 1.00)
            for meth in METHODS:
                if abs(p100[meth] - 1.0) > 1e-12:
                    raise SystemExit(
                        "%s/%s %s at a 100%% budget is %r, not 1.0. Monitoring "
                        "every edge covers every attack by construction; a "
                        "value below 1 means k or the tie credit is wrong."
                        % (dom, label, meth, p100[meth]))
            # -- inference stays where it was predeclared ------------------
            for p in blk["curve"]:
                declared = p["budget"] in PREDECLARED
                if declared != ("mesa_minus_random" in p):
                    raise SystemExit(
                        "%s/%s @%s: inference attached to a budget that was "
                        "not predeclared (or missing where it was)."
                        % (dom, label, p["budget"]))
            # -- the oracle is a bound, so nothing may exceed it -----------
            for p in blk["curve"]:
                if p["mesa"] - p["oracle"] > 1e-9:
                    raise SystemExit(
                        "%s/%s @%s: MESA %.6f exceeds the oracle bound %.6f."
                        % (dom, label, p["budget"], p["mesa"], p["oracle"]))

    doms = set(payload["scenarios"])
    if doms != set(SCENARIOS):
        raise SystemExit("domains %s, expected %s" % (sorted(doms),
                                                      sorted(SCENARIOS)))


def main():
    payload = {
        "generated_by": "python analysis/canonical_ranking_curve.py",
        "canonical_block": CANONICAL,
        "n_features": N_FEATURES,
        "budgets": list(BUDGETS),
        "predeclared_budgets": list(PREDECLARED),
        "note": ("Full ranking-coverage trajectory on the frozen eight-feature "
                 "MESA-Local score, same cross-fitted within-configuration "
                 "analysis as data/canonical_ranking.json, whose 10% and 20% "
                 "values this file reproduces exactly. Random is the EXACT "
                 "expectation k/|E|, recomputed at every budget because k = "
                 "ceil(b|E|) changes with b. The oracle is a BOUND on any "
                 "ordering, never a method. Inference is confined to the "
                 "predeclared 10% and 20% budgets -- paired configuration-level "
                 "bootstrap CIs and an exact sign-flip test; the remaining "
                 "budgets describe the trajectory and carry no p-value. "
                 "Domains are never pooled and edges are never normalized "
                 "across configurations. ROUTING PROVENANCE: not every "
                 "underlying row was re-run under routing-v1. See "
                 "scenarios[*].routing_provenance -- the reused rows are "
                 "original measurements from AUDITED ROUTING-UNCHANGED "
                 "CONFIGURATIONS, admissible only because the repair did not "
                 "touch those protocols, which is asserted here."),
        "scenarios": {s: analyse(s) for s in SCENARIOS},
    }
    check(payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    for dom, res in payload["scenarios"].items():
        m = res["macro"]
        print("=== %s  (macro over %d configurations)"
              % (dom, m["n_configurations"]))
        print("  %-8s %8s %8s %8s %8s" % ("budget", "MESA", "random", "oracle",
                                          "mean k"))
        for p in m["curve"]:
            star = " *" if p["predeclared"] else ""
            print("  %-8s %8.3f %8.3f %8.3f %8.2f%s"
                  % ("%d%%" % round(p["budget"] * 100), p["mesa"], p["random"],
                     p["oracle"], p["mean_k"], star))
        print("  * predeclared; MESA - random, paired on the configuration:")
        for p in m["curve"]:
            if not p["predeclared"]:
                continue
            d, sf = p["mesa_minus_random"], p["signflip"]
            print("      @%d%%  %+.3f [%+.3f, %+.3f]  sign-flip p=%.5f "
                  "(min attainable %.5f)"
                  % (round(p["budget"] * 100), d["mean"], d["lo"], d["hi"],
                     sf["p_two_sided"], sf["min_attainable_two_sided"]))
        pv = res["routing_provenance"]
        print("  routing: %d/%d rows reused (not re-run) from %d audited "
              "routing-unchanged configurations %s"
              % (pv["n_rows_reused_not_rerun"], pv["n_edge_rows"],
                 pv["n_audited_routing_unchanged_configurations"],
                 pv["topologies_reused"]))
        print()
    print("wrote %s" % OUT.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
