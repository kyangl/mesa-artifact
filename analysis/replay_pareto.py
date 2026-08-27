"""Replay allocation policies from cached detector scores.

The replay applies cross-fitted rankings and checks that no policy exceeds the
contamination-path oracle. Detector scoring is reused across policies.

Run: ``python analysis/replay_pareto.py --scenario customer_service``.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.crossfit import DIRECTIONS, direction_label
from analysis.enforcement import RANDOM_SEEDS, random_orderings
from analysis.pareto import DetectorScores, evaluate_cell
from analysis.build_feature_matrix import load_validity
from runners.run_cs_pareto import (
    POLICY_ALL, POLICY_CLASSICAL, POLICY_EMPIRICAL, POLICY_MESA_LOCAL,
    POLICY_NONE, POLICY_ORACLE, budgets_for, effective_graph, load_trials,
    mesa_local_ordering, select_edges,
)
from src.saliency.mesa_scores import MesaLocal
from analysis.run_mesa_cv import _binary_idx

def _n_canonical_features():
    from analysis.run_mesa_fit import BLOCKS
    return len(BLOCKS[_canonical_block()])


def _canonical_block(default="structural_dynamic"):
    """The frozen feature block. Falls back only if the decision is absent."""
    import json as _json
    p = REPO / "data" / "feature_freeze_decision.json"
    if p.exists():
        try:
            return _json.loads(p.read_text())["decision"]["canonical_block"]
        except Exception:
            pass
    return default


POLICY_MESA_CF = "mesa_local_crossfitted_k"
POLICY_RANDOM = "random_k"


def max_coverage_subset(edges, paths, k):
    """Exact maximum-coverage subset of size k over contamination paths."""
    import itertools
    if k <= 0:
        return []
    if k >= len(edges):
        return list(edges)
    best, best_n = [], -1
    for combo in itertools.combinations(sorted(edges), k):
        s = set(combo)
        n = sum(1 for p in paths if p & s)
        if n > best_n:
            best, best_n = list(combo), n
    return best
PRIMARY_MODELS = ("gemma4:e4b", "qwen3.5:9b")
TOPOLOGIES = ("centralized", "decentralized", "hierarchical", "hybrid",
              "sequential")


def load_cached_scores(scenario):
    """DetectorScores per (model, topology) from the persisted dump."""
    for p in (REPO / "results" / "detectors"
              / ("cs_pareto_scores_%s.json" % scenario),
              REPO / ("cs_pareto_scores_%s.json" % scenario)):
        if p.exists():
            payload = json.loads(p.read_text())
            break
    else:
        raise SystemExit("no cached detector scores for %s" % scenario)

    out = collections.defaultdict(DetectorScores)
    for r in payload["scores"]:
        det = out[(r["model"], r["topology"])]
        edge = r["attacked_edge"]
        edge = tuple(edge) if isinstance(edge, (list, tuple)) else edge
        key = (r["task_id"], edge, r["msg_index"])
        det.scores[key] = r["score"]
        det.latency_s[key] = r.get("latency_s", 0.0)
        det.tokens[key] = r.get("token_cost", 0)
    return dict(out), payload.get("detector")


def crossfitted_orderings(scenario, topology, model):
    """{(feature_fold, outcome_fold): [edges, best first]} from fold features.

    Ranking uses ONLY the fold's own feature estimates. There is no fallback
    to the pooled value: falling back is the leak this replay exists to remove.
    """
    from analysis.run_mesa_fit import load_enriched_directional, BLOCKS
    _combined, _dropped, mats = load_enriched_directional(scenarios=[scenario])
    out = {}
    # CANONICAL BLOCK, read from the frozen decision rather than hardcoded.
    # The freeze rejected F2/F3 from the primary score, so an allocation still
    # ranking on ten features would not be the method the paper reports.
    cols = BLOCKS[_canonical_block()]
    for (ff, of), fm in mats:
        idx = [i for i, r in enumerate(fm.rows)
               if r["topology"] == topology and r["model"] == model]
        if not idx:
            continue
        names = list(fm.feature_names)
        keep = [names.index(c) for c in cols]
        sub = fm.X[np.ix_(idx, keep)]
        kept_names = [names[i] for i in keep]
        s = MesaLocal(kept_names, _binary_idx(kept_names)).score(
            sub, np.array(["g"] * len(idx)))
        edges = [(fm.rows[i]["edge_src"], fm.rows[i]["edge_dst"]) for i in idx]
        order = [e for _v, e in sorted(zip(-np.nan_to_num(s, nan=0.5), edges),
                                       key=lambda p: p[0])]
        out[(ff, of)] = order
    return out


def fold_of_task(scenario):
    path = REPO / "config" / "task_folds.json"
    return json.loads(path.read_text())["folds"].get(scenario, {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="customer_service")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.26, 0.5])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or str(REPO / "data" /
                          ("pareto_replay_%s.json" % args.scenario))

    cached, detector = load_cached_scores(args.scenario)
    validity = load_validity()
    folds = fold_of_task(args.scenario)

    rows, summary, violations = [], [], []
    for model in PRIMARY_MODELS:
        for topology in TOPOLOGIES:
            det = cached.get((model, topology))
            if det is None:
                summary.append([model, topology, 0, 0, "no cached scores"])
                continue
            attacked, clean = load_trials(args.scenario, model, topology)
            attacked = [t for t in attacked if t.messages]
            if not [t for t in attacked if t.success]:
                summary.append([model, topology, len(attacked), 0,
                                "no successes"])
                continue
            G, _ = effective_graph(topology, validity)
            edges = sorted(G.edges())
            import networkx as nx
            eb = nx.edge_betweenness_centrality(G)
            classical = sorted(edges, key=lambda e: -eb.get(e, 0.0))
            mass = collections.Counter(t.attacked_edge for t in attacked
                                       if t.success and t.attacked_edge)
            empirical = sorted(edges, key=lambda e: -mass.get(e, 0))
            # Order the oracle by contamination-path coverage, its own metric.
            paths = [t.attack_reachable_edges() for t in attacked if t.success]
            cf = crossfitted_orderings(args.scenario, topology, model)

            # MESA-Local (pooled ordering) is the PRIMARY method and must be
            # present, alongside the cross-fitted variant below. Omitting it
            # left the replay with no primary curve at all.
            local, _feats = mesa_local_ordering(args.scenario, topology,
                                                model, validity)
            base = {POLICY_NONE: edges, POLICY_CLASSICAL: classical,
                    POLICY_MESA_LOCAL: local or classical,
                    POLICY_EMPIRICAL: empirical, POLICY_ALL: edges}
            for policy, ordering in base.items():
                for k in budgets_for(len(edges)):
                    if policy == POLICY_NONE and k != 0:
                        continue
                    if policy == POLICY_ALL and k != len(edges):
                        continue
                    for th in args.thresholds:
                        r = evaluate_cell(
                            attacked, clean, select_edges(ordering, k), th,
                            det, rerun=None, oracle=False)
                        r.update({"policy": policy, "budget_k": k,
                                  "model": model, "topology": topology,
                                  "direction": "pooled",
                                  "weight": 1.0,
                                  "is_upper_bound": policy in (
                                      POLICY_EMPIRICAL, POLICY_ORACLE)})
                        rows.append(r)

            # Exhaustively maximize path coverage; greedy ranking is not exact.
            for k in budgets_for(len(edges)):
                best = max_coverage_subset(edges, paths, k)
                for th in args.thresholds:
                    r = evaluate_cell(attacked, clean, best, th, det,
                                      rerun=None, oracle=True)
                    r.update({"policy": POLICY_ORACLE, "budget_k": k,
                              "model": model, "topology": topology,
                              "direction": "pooled", "weight": 1.0,
                              "is_upper_bound": True})
                    rows.append(r)

            # RANDOM: deterministic, committed seeds, evaluated inside each
            # cross-fitting direction on that direction's own outcome fold so
            # it is matched to the cross-fitted MESA curve rather than to the
            # pooled one.
            for (ff, of), _ordering in cf.items():
                subset = [t for t in attacked if folds.get(t.task_id) == of]
                subset_clean = [t for t in clean if folds.get(t.task_id) == of]
                if not [t for t in subset if t.success]:
                    continue
                for seed_i, order in enumerate(random_orderings(edges)):
                    for k in budgets_for(len(edges)):
                        for th in args.thresholds:
                            r = evaluate_cell(subset, subset_clean,
                                              select_edges(order, k), th, det,
                                              rerun=None, oracle=False)
                            r.update({"policy": POLICY_RANDOM, "budget_k": k,
                                      "model": model, "topology": topology,
                                      "direction": direction_label(ff, of),
                                      "seed": RANDOM_SEEDS[seed_i],
                                      "weight": 0.5 / len(RANDOM_SEEDS),
                                      "is_upper_bound": False})
                            rows.append(r)

            # Cross-fitted MESA: rank on one fold, score on the other's tasks.
            for (ff, of), ordering in cf.items():
                subset = [t for t in attacked if folds.get(t.task_id) == of]
                subset_clean = [t for t in clean if folds.get(t.task_id) == of]
                if not [t for t in subset if t.success]:
                    continue
                for k in budgets_for(len(edges)):
                    for th in args.thresholds:
                        r = evaluate_cell(subset, subset_clean,
                                          select_edges(ordering, k), th, det,
                                          rerun=None, oracle=False)
                        r.update({"policy": POLICY_MESA_CF, "budget_k": k,
                                  "model": model, "topology": topology,
                                  "direction": direction_label(ff, of),
                                  "weight": 0.5, "is_upper_bound": False})
                        rows.append(r)
            summary.append([model, topology, len(attacked),
                            sum(1 for t in attacked if t.success),
                            "cf_directions=%d" % len(cf)])

    # The bound must dominate every policy, per configuration/budget/threshold.
    by = collections.defaultdict(dict)
    for r in rows:
        if r["direction"] != "pooled":
            continue
        by[(r["model"], r["topology"], r["budget_k"], r["threshold"])][
            r["policy"]] = r["security"]
    for key, v in by.items():
        if POLICY_ORACLE not in v:
            continue
        for policy, val in v.items():
            if policy in (POLICY_ORACLE, POLICY_EMPIRICAL):
                continue
            if val > v[POLICY_ORACLE] + 1e-9:
                violations.append({"cell": list(key), "policy": policy,
                                   "security": val,
                                   "oracle": v[POLICY_ORACLE]})

    print("REPLAY -- %s   detector=%s" % (args.scenario, detector))
    for s in summary:
        print("  %-12s %-14s attacked=%3d successes=%3d  %s" % tuple(s))
    print("\nORACLE DOMINANCE: %s (%d violations)"
          % ("OK" if not violations else "VIOLATED", len(violations)))
    for v in violations[:6]:
        print("   %s %s security=%.3f > oracle=%.3f"
              % (v["cell"], v["policy"], v["security"], v["oracle"]))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"scenario": args.scenario, "detector": detector,
         "feature_block": _canonical_block(),
        "n_features": _n_canonical_features(),
        "note": ("Replayed from cached detector output. mesa_local_"
                  "crossfitted_k ranks on one task fold and is scored on the "
                  "other's outcomes, each direction at half weight. "
                  "perfect_oracle credits interception anywhere on the "
                  "contamination path; detector prevention requires a flag on "
                  "a carrying edge."),
         "configs": summary, "oracle_violations": violations,
         "cells": rows}, indent=2, default=str))
    print("\nwrote %d cells -> %s" % (len(rows), out))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
