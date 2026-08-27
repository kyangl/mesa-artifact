"""
Build the per-edge master table and correlation summaries from raw result JSONs.

Writes to data/: per_edge_master_table.csv, cross_experiment_summary.csv,
pooled_rho_per_scenario.csv, saliency_variants_random_topo.csv.
"""

import json
import sys
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topology.scope import PRIMARY_TOPOLOGIES, assert_primary_scope
from src.topology.builder import load_topology, build_graph
from src.saliency.structural import compute_all_structural_features

RESULTS = ROOT / "results"
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_TOPOS = ["centralized", "sequential", "hierarchical", "decentralized",
             "hybrid", "random_er", "random_ba"]
ORIGINAL_TOPOS = {"centralized", "sequential", "hierarchical",
                  "decentralized", "hybrid"}
SCENARIOS = {
    "customer_service": ("pilot_clean", "pilot_attack",
                         "dynamic_ablation_customer_service",
                         "dynamic_perturbation_customer_service"),
    "software_engineering": ("se_clean", "se_attack",
                             "dynamic_ablation_software_engineering",
                             "dynamic_perturbation_software_engineering"),
    "homogeneous_debate": ("debate_clean", "debate_attack",
                           "dynamic_ablation_homogeneous_debate",
                           "dynamic_perturbation_homogeneous_debate"),
}
STRUCT_KEYS = ["betweenness_centrality", "information_bottleneck", "is_bridge",
               "endpoint_centrality_max", "source_degree_centrality",
               "target_degree_centrality"]


def load_valid(prefix):
    """Load result records from results/{prefix}_*.json, dropping errors."""
    out = []
    for p in sorted(RESULTS.glob(f"{prefix}_*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        if not isinstance(d, list):
            continue
        for r in d:
            if not isinstance(r, dict) or "error" in r:
                continue
            sc = r.get("scores")
            if not isinstance(sc, dict) or sc.get("decision_accuracy", -1) < 0:
                continue
            out.append(r)
    return out


def get_features(topo):
    p = ROOT / f"config/topologies/{topo}.yaml"
    if not p.exists():
        return None
    return compute_all_structural_features(build_graph(load_topology(str(p))))


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan")
    return float(stats.spearmanr(x[m], y[m]).correlation)


def struct_score(feats_per_edge, edges):
    """Signed rank-sum of the six static features, normalized by n_edges.

    Ranks are summed in raw integer space and the total divided by n once, so
    exact rank ties are preserved (per-feature division would break them in
    floating point and perturb Spearman rho).
    """
    n = len(edges)
    bottleneck = ["betweenness_centrality", "information_bottleneck", "is_bridge"]
    degree = ["endpoint_centrality_max", "source_degree_centrality",
              "target_degree_centrality"]
    s = np.zeros(n)
    for k in bottleneck:
        v = np.array([feats_per_edge[e].get(k, 0) for e in edges])
        if np.std(v) > 0:
            s += stats.rankdata(v)
    for k in degree:
        v = np.array([feats_per_edge[e].get(k, 0) for e in edges])
        if np.std(v) > 0:
            s -= stats.rankdata(v)
    return s / n


def rank_fill(arr):
    """Normalized rank in (0, 1]; NaNs filled with the mid-rank to stay neutral."""
    n = len(arr)
    ranks = np.full_like(arr, np.nan, dtype=float)
    mask = np.isfinite(arr)
    ranks[mask] = stats.rankdata(arr[mask])
    ranks[~mask] = mask.sum() / 2 + 0.5
    return ranks / n


print("Building per-edge master table...")
rows = []
for scenario, (cp, ap, abp, ptp) in SCENARIOS.items():
    print(f"  loading {scenario}...")
    clean = load_valid(cp)
    attack = load_valid(ap)
    abl_data = load_valid(abp)
    pert_data = load_valid(ptp)

    cl_acc = {(r["model"], r["topology"], r["task_id"]): r["scores"]["decision_accuracy"]
              for r in clean}

    # Corrected ASR: count a flip only where the clean baseline was correct.
    asr_s = defaultdict(lambda: {"e": 0, "f": 0})
    for r in attack:
        e = r.get("attack_edge")
        if not e:
            continue
        e = tuple(e)
        ck = (r["model"], r["topology"], r["task_id"])
        if cl_acc.get(ck) == 1:
            key = (r["model"], r["topology"], e)
            asr_s[key]["e"] += 1
            if r["scores"]["decision_accuracy"] == 0:
                asr_s[key]["f"] += 1

    clean_mean = defaultdict(list)
    for r in clean:
        clean_mean[(r["model"], r["topology"])].append(r["scores"]["decision_accuracy"])
    clean_mean = {k: float(np.mean(v)) for k, v in clean_mean.items()}

    def per_edge_dyn(records):
        """Dynamic delta per edge: clean topology accuracy minus intervened accuracy."""
        by_edge = defaultdict(list)
        for r in records:
            ie = r.get("intervention_edge")
            if not ie:
                continue
            by_edge[(r["model"], r["topology"], tuple(ie))].append(
                r["scores"]["decision_accuracy"])
        out = {}
        for k, vals in by_edge.items():
            ca = clean_mean.get((k[0], k[1]))
            if ca is not None:
                out[k] = ca - float(np.mean(vals))
        return out

    abl_d = per_edge_dyn(abl_data)
    pert_d = per_edge_dyn(pert_data)

    topo_feats = {}
    for topo in ALL_TOPOS:
        f = get_features(topo)
        if f:
            topo_feats[topo] = f

    for (model, topo, edge), counts in asr_s.items():
        if counts["e"] < 1:
            continue
        feats = topo_feats.get(topo, {}).get(edge, {})
        row = {
            "scenario": scenario,
            "model": model,
            "topology": topo,
            "edge_src": edge[0],
            "edge_dst": edge[1],
            "n_eligible": counts["e"],
            "n_flipped": counts["f"],
            "corrected_asr": counts["f"] / counts["e"],
            "clean_acc_topo": clean_mean.get((model, topo)),
            "ablation_delta": abl_d.get((model, topo, edge)),
            "perturbation_delta": pert_d.get((model, topo, edge)),
        }
        for k in STRUCT_KEYS:
            row[k] = feats.get(k)
        rows.append(row)

print(f"  {len(rows)} per-edge rows total")

with open(OUT_DIR / "per_edge_master_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"  wrote {OUT_DIR / 'per_edge_master_table.csv'}")


# Per (scenario, model, topology) summary
print("\nBuilding cross-experiment summary...")
groups = defaultdict(list)
for r in rows:
    groups[(r["scenario"], r["model"], r["topology"])].append(r)

summary_rows = []
for (sc, model_name, t), grp in groups.items():
    asrs = [r["corrected_asr"] for r in grp if r["corrected_asr"] is not None]
    if len(asrs) < 5:
        continue
    edges = [(r["edge_src"], r["edge_dst"]) for r in grp]
    feats_dict = {(r["edge_src"], r["edge_dst"]): {k: r.get(k) for k in STRUCT_KEYS}
                  for r in grp}
    asr_arr = np.array([r["corrected_asr"] for r in grp])
    abl_arr = np.array([r["ablation_delta"] if r["ablation_delta"] is not None else np.nan
                        for r in grp])
    pert_arr = np.array([r["perturbation_delta"] if r["perturbation_delta"] is not None else np.nan
                         for r in grp])

    combined = struct_score(feats_dict, edges).copy()
    if np.isfinite(abl_arr).any():
        combined = combined + rank_fill(abl_arr)
    if np.isfinite(pert_arr).any():
        combined = combined + rank_fill(pert_arr)

    summary_rows.append({
        "scenario": sc, "model": model_name, "topology": t,
        "n_edges": len(grp),
        "clean_acc": grp[0]["clean_acc_topo"],
        "mean_asr": float(np.mean(asrs)),
        "max_asr": float(np.max(asrs)),
        "rho_struct": spearman(struct_score(feats_dict, edges), asr_arr),
        "rho_abl_vs_asr": spearman(abl_arr, asr_arr) if np.isfinite(abl_arr).any() else float("nan"),
        "rho_pert_vs_asr": spearman(pert_arr, asr_arr) if np.isfinite(pert_arr).any() else float("nan"),
        "rho_combined": spearman(combined, asr_arr),
    })

with open(OUT_DIR / "cross_experiment_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    w.writeheader()
    w.writerows(summary_rows)
print(f"  wrote {OUT_DIR / 'cross_experiment_summary.csv'} ({len(summary_rows)} rows)")


# Headline: rho pooled across topologies, per (scenario, model)
print("\nBuilding pooled rho per (scenario, model)...")
pooled_groups = defaultdict(list)
for r in rows:
    pooled_groups[(r["scenario"], r["model"])].append(r)

pooled_rows = []
for (sc, m), grp in pooled_groups.items():
    for tag, keep in [("all_8_topos", lambda r: True),
                      ("original_6_topos", lambda r: r["topology"] in ORIGINAL_TOPOS)]:
        sub = [r for r in grp if keep(r)]
        if len(sub) < 5:
            continue
        edges = [(r["topology"], r["edge_src"], r["edge_dst"]) for r in sub]
        feats = {(r["topology"], r["edge_src"], r["edge_dst"]):
                 {k: r.get(k) for k in STRUCT_KEYS} for r in sub}
        asr_arr = np.array([r["corrected_asr"] for r in sub])
        abl_arr = np.array([r["ablation_delta"] if r["ablation_delta"] is not None else np.nan
                            for r in sub])
        pert_arr = np.array([r["perturbation_delta"] if r["perturbation_delta"] is not None else np.nan
                             for r in sub])

        struct_s = struct_score(feats, edges)
        combined = struct_s.copy()
        dyn_only = np.zeros(len(sub))
        n_dyn = 0
        for arr in (abl_arr, pert_arr):
            if np.isfinite(arr).sum() >= 5:
                ranks = rank_fill(arr)
                combined = combined + ranks
                dyn_only = dyn_only + ranks
                n_dyn += 1

        pooled_rows.append({
            "topology_set": tag, "scenario": sc, "model": m,
            "n_edges": len(sub),
            "rho_struct_pooled": spearman(struct_s, asr_arr),
            "rho_dynamic_only": spearman(dyn_only, asr_arr) if n_dyn else float("nan"),
            "rho_combined_pooled": spearman(combined, asr_arr),
            "rho_ablation_only": spearman(abl_arr, asr_arr) if np.isfinite(abl_arr).any() else float("nan"),
            "rho_perturbation_only": spearman(pert_arr, asr_arr) if np.isfinite(pert_arr).any() else float("nan"),
        })

with open(OUT_DIR / "pooled_rho_per_scenario.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=pooled_rows[0].keys())
    w.writeheader()
    w.writerows(pooled_rows)
print(f"  wrote {OUT_DIR / 'pooled_rho_per_scenario.csv'} ({len(pooled_rows)} rows)\n")

for tag in ("original_6_topos", "all_8_topos"):
    print(f"  -- {tag}")
    print(f"  {'scenario':<22} {'model':<12} {'n_edges':>7}  {'rho_struct':>10}  {'rho_combined':>12}")
    for r in [x for x in pooled_rows if x["topology_set"] == tag]:
        print(f"  {r['scenario']:<22} {r['model']:<12} {r['n_edges']:>7}  "
              f"{r['rho_struct_pooled']:>+10.3f}  {r['rho_combined_pooled']:>+12.3f}")
    print()


# Random-topology diagnostic
variant_rows = []
for r in summary_rows:
    if r["topology"] not in ("random_er", "random_ba"):
        continue
    variant_rows.append({
        "model": r["model"], "topology": r["topology"], "n_edges": r["n_edges"],
        "clean_acc": r["clean_acc"], "mean_asr": r["mean_asr"],
        "rho_struct": r["rho_struct"], "rho_abl_only": r["rho_abl_vs_asr"],
        "rho_pert_only": r["rho_pert_vs_asr"], "rho_combined_default_signs": r["rho_combined"],
    })

with open(OUT_DIR / "saliency_variants_random_topo.csv", "w", newline="") as f:
    if variant_rows:
        w = csv.DictWriter(f, fieldnames=variant_rows[0].keys())
        w.writeheader()
        w.writerows(variant_rows)
print(f"  wrote {OUT_DIR / 'saliency_variants_random_topo.csv'} ({len(variant_rows)} rows)")
print(f"\nDone. Tables in {OUT_DIR}")
