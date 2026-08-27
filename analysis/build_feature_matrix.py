"""Build the routing-aware edge feature matrix.

Structural features use the effective graph of consumed deliveries. Superseded
routing rows remain namespaced as ``routing_v0_invalid`` and are never fitted;
corrected rows use ``routing_v1``.
"""

import argparse
import collections
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.topology.scope import PRIMARY_TOPOLOGIES, assert_primary_scope

from src.topology.builder import load_topology, build_graph
from src.saliency.structural import compute_all_structural_features

STRUCTURAL_FEATURES = [
    "betweenness_centrality", "information_bottleneck", "is_bridge",
    "endpoint_centrality_max", "source_degree_centrality",
    "target_degree_centrality",
]
DYNAMIC_FEATURES = ["ablation_delta", "perturbation_delta"]
BINARY_FEATURES = ["is_bridge"]

# Workflow-aware candidates. Missing measurements are never zero-filled.
MAS_FEATURES = ["semantic_non_recoverability", "receiver_response_sensitivity",
                "consequence_proximity"]

# Protocols changed by the routing repair; mesh remains for auditability.
ROUTING_CHANGED = {"hierarchical", "hybrid", "mesh"}
# Excluded from the feature-complete ranking matrix, not the breadth study.
DROPPED_TOPOLOGIES = {"random_er", "random_ba", "random_ws",
                      "random_er_2", "random_er_3", "random_er_4",
                      "random_ba_2", "random_ba_3", "random_ba_4"}
# Pre-repair rows remain valid for unchanged protocols.
ROUTING_UNCHANGED = {"sequential", "centralized", "decentralized"}

NS_V0_VALID = "routing_v0_valid"
NS_V0_INVALID = "routing_v0_invalid"
NS_V1 = "routing_v1"

REACHING = "reaching"
DEFAULT_VALIDITY_CSV = REPO / "data" / "edge_routing_validity.csv"
DEFAULT_MASTER_CSV = REPO / "data" / "per_edge_master_table.csv"


@dataclass
class FeatureMatrix:
    X: np.ndarray
    y_success: np.ndarray
    n_trials: np.ndarray
    groups: np.ndarray
    feature_names: List[str]
    rows: List[Dict]

    def missing_features(self) -> List[str]:
        """Feature names that are entirely missing (all NaN)."""
        return [name for i, name in enumerate(self.feature_names)
                if np.all(np.isnan(self.X[:, i]))]

    def complete_features(self) -> List[str]:
        return [n for n in self.feature_names if n not in self.missing_features()]


def load_validity(path=DEFAULT_VALIDITY_CSV) -> Dict:
    """{(topology, src, dst): routing_validity_class}."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[(r["topology"], r["edge_src"], r["edge_dst"])] = \
                r["routing_validity"]
    return out


def effective_graph(topology_name, validity):
    """The subgraph of edges whose delivery a recipient actually consumes."""
    topo = load_topology(str(REPO / "config" / "topologies"
                             / ("%s.yaml" % topology_name)))
    G = build_graph(topo)
    drop = [(u, v) for u, v in G.edges()
            if validity.get((topology_name, u, v), REACHING) != REACHING]
    G.remove_edges_from(drop)
    return G, drop


def effective_structural_features(topology_name, validity):
    """Structural features recomputed on the effective graph."""
    G, _ = effective_graph(topology_name, validity)
    return compute_all_structural_features(G)


def namespace_for(topology, source_namespace):
    if source_namespace == NS_V1:
        return NS_V1
    if topology in ROUTING_CHANGED or topology in DROPPED_TOPOLOGIES:
        return NS_V0_INVALID
    return NS_V0_VALID


def rank_normalize(X, groups, binary_cols=None):
    """Within-group percentile ranks in [0, 1]; binary columns untouched.

    NaN entries stay NaN and are excluded from the ranking of their column.
    """
    X = np.asarray(X, dtype=float)
    binary_cols = set(binary_cols or [])
    out = np.array(X, dtype=float, copy=True)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for j in range(X.shape[1]):
            if j in binary_cols:
                continue
            col = X[idx, j]
            ok = ~np.isnan(col)
            if ok.sum() == 0:
                continue
            if ok.sum() == 1:
                out[idx[ok], j] = 0.5
                continue
            r = stats.rankdata(col[ok], method="average")
            out[idx[ok], j] = (r - 1.0) / (len(r) - 1.0)
    return out


def load_matrix(csv_path=DEFAULT_MASTER_CSV, validity_path=DEFAULT_VALIDITY_CSV,
                scenarios=None, topologies=None, models=None,
                namespaces=(NS_V0_VALID, NS_V1),
                require_reaching=True, source_namespace=NS_V0_VALID,
                feature_names=None):
    """Load edges into a fitting-ready matrix.

    Rows are excluded when their edge is not part of the effective graph, or
    when their namespace is not requested. Structural features come from the
    effective graph, not the declared topology.
    """
    validity = load_validity(validity_path)
    feature_names = list(feature_names or
                         (STRUCTURAL_FEATURES + DYNAMIC_FEATURES + MAS_FEATURES))
    struct_cache = {}

    rows, X, ys, ns, groups = [], [], [], [], []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            topo = r["topology"]
            if scenarios and r["scenario"] not in scenarios:
                continue
            if topologies and topo not in topologies:
                continue
            if models and r["model"] not in models:
                continue
            if topo in DROPPED_TOPOLOGIES:
                continue
            ns_row = namespace_for(topo, source_namespace)
            if ns_row not in namespaces:
                continue
            edge = (topo, r["edge_src"], r["edge_dst"])
            cls = validity.get(edge)
            if cls is None:
                continue  # unaudited: never fitted
            if require_reaching and cls != REACHING:
                continue
            try:
                n_elig = int(float(r["n_eligible"] or 0))
            except ValueError:
                continue
            if n_elig <= 0:
                continue

            if topo not in struct_cache:
                struct_cache[topo] = effective_structural_features(topo, validity)
            sfeat = struct_cache[topo].get((r["edge_src"], r["edge_dst"]), {})

            vec = []
            for name in feature_names:
                if name in STRUCTURAL_FEATURES:
                    val = sfeat.get(name, np.nan)
                elif name in r and r[name] not in ("", None):
                    val = float(r[name])
                else:
                    val = np.nan  # missing, NOT zero
                vec.append(float(val) if val is not None else np.nan)

            rec = dict(r)
            rec["routing_validity"] = cls
            rec["routing_namespace"] = ns_row
            rows.append(rec)
            X.append(vec)
            ys.append(int(float(r["n_flipped"] or 0)))
            ns.append(n_elig)
            groups.append("%s|%s|%s" % (r["scenario"], topo, r["model"]))

    return FeatureMatrix(
        X=np.array(X, dtype=float) if X else np.zeros((0, len(feature_names))),
        y_success=np.array(ys, dtype=int),
        n_trials=np.array(ns, dtype=int),
        groups=np.array(groups),
        feature_names=feature_names,
        rows=rows,
    )


PILOT_MIN_TASKS = 20


def load_canonical_matrix(master_csv=DEFAULT_MASTER_CSV,
                          v1_csv=None, validity_path=DEFAULT_VALIDITY_CSV,
                          scenarios=("customer_service",
                                     "software_engineering")):
    """The primary-statistics matrix: routing_v0_valid + routing_v1.

    Explicitly excluded:
      * routing_v0_invalid -- hierarchical/hybrid/mesh measured under the
        superseded routing; those rows describe a system that no longer exists.
      * pilot-scale rows (n_tasks < 20) -- currently the mesh pilot. A
        five-task estimate is not a paper statistic.
      * dropped topologies (ER/BA).
    """
    v1_csv = v1_csv or (REPO / "data" / "per_edge_routing_v1.csv")
    fm = load_matrix(master_csv, validity_path, scenarios=list(scenarios),
                     namespaces=(NS_V0_VALID,))
    rows, X, ys, ns, groups = (list(fm.rows), list(fm.X), list(fm.y_success),
                               list(fm.n_trials), list(fm.groups))

    validity = load_validity(validity_path)
    struct_cache = {}
    # Do not duplicate edges present in both routing namespaces.
    have = {(r["scenario"], r["topology"], r["model"],
             r["edge_src"], r["edge_dst"]) for r in rows}
    if Path(v1_csv).exists():
        with open(v1_csv) as fh:
            for r in csv.DictReader(fh):
                if r["scenario"] not in scenarios:
                    continue
                if (r["scenario"], r["topology"], r["model"],
                        r["edge_src"], r["edge_dst"]) in have:
                    continue
                if int(float(r.get("n_tasks") or 0)) < PILOT_MIN_TASKS:
                    continue                      # pilot, not a paper statistic
                topo = r["topology"]
                if validity.get((topo, r["edge_src"], r["edge_dst"])) != REACHING:
                    continue
                n_elig = int(float(r["n_eligible"] or 0))
                if n_elig <= 0:
                    continue
                if topo not in struct_cache:
                    struct_cache[topo] = effective_structural_features(topo,
                                                                       validity)
                sf = struct_cache[topo].get((r["edge_src"], r["edge_dst"]), {})
                vec = []
                for name in fm.feature_names:
                    if name in STRUCTURAL_FEATURES:
                        vec.append(float(sf.get(name, np.nan)))
                    elif r.get(name) not in ("", None):
                        vec.append(float(r[name]))
                    else:
                        vec.append(np.nan)
                rec = dict(r)
                rec["routing_validity"] = REACHING
                rec["routing_namespace"] = NS_V1
                rows.append(rec)
                X.append(vec)
                ys.append(int(float(r["n_flipped"] or 0)))
                ns.append(n_elig)
                groups.append("%s|%s|%s" % (r["scenario"], topo, r["model"]))

    return FeatureMatrix(
        X=np.array(X, dtype=float) if X else np.zeros((0, len(fm.feature_names))),
        y_success=np.array(ys, dtype=int), n_trials=np.array(ns, dtype=int),
        groups=np.array(groups), feature_names=fm.feature_names, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_MASTER_CSV))
    ap.add_argument("--validity", default=str(DEFAULT_VALIDITY_CSV))
    args = ap.parse_args()

    validity = load_validity(args.validity)
    print("Effective graph sizes")
    for topo in sorted(ROUTING_UNCHANGED | ROUTING_CHANGED):
        G, dropped = effective_graph(topo, validity)
        print("  %-14s effective edges=%2d  dropped=%d %s"
              % (topo, G.number_of_edges(), len(dropped),
                 dropped if dropped else ""))

    fm = load_matrix(args.csv, args.validity,
                     scenarios=["customer_service", "software_engineering"])
    print("\nDiagnostic set (%s only -- routing-unchanged protocols)"
          % NS_V0_VALID)
    print("  rows=%d  configs=%d  eligible=%d  successes=%d"
          % (len(fm.rows), len(set(fm.groups)),
             int(fm.n_trials.sum()), int(fm.y_success.sum())))
    print("  topologies: %s" % sorted(set(r["topology"] for r in fm.rows)))
    print("  complete features: %s" % fm.complete_features())
    print("  MISSING features: %s" % fm.missing_features())


if __name__ == "__main__":
    main()
