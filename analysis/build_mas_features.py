"""Aggregate workflow features by task, then by configuration edge.

Tasks receive equal weight regardless of turn count. Recoverability reports
both max- and mean-similarity forms plus context count to expose in-degree
bias. Downstream scores must use complete feature blocks.

Run: ``python analysis/build_mas_features.py --out data/mas_features.json``.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.build_feature_matrix import effective_graph, load_validity
from src.saliency.mas_features import (
    _install_cpu_limit_handler, consequence_proximity, embed,
    save_embedding_cache,
)
from src.topology.builder import load_topology

PROBE_GLOB = "results/**/feature_probe_*.jsonl"


def _cos(a, b):
    return float(np.dot(a, b))


def f1_from_records(records, model_name=None):
    """NR per (edge, task) from f1_context records, then equal-mean over tasks.

    Returns {(edge): {"nr_max", "nr_mean", "n_context", "n_tasks"}}.
    """
    by_edge_task = collections.defaultdict(list)
    for r in records:
        content = (r.get("content") or "").strip()
        if not content:
            continue
        alts = [c for c in (r.get("prior_context") or []) if c and c.strip()]
        key = ((r["edge_src"], r["edge_dst"]), r["task_id"])
        if not alts:
            by_edge_task[key].append((1.0, 1.0, 0))
            continue
        vecs = embed([content] + alts) if model_name is None else embed(
            [content] + alts, model_name)
        sims = vecs[1:] @ vecs[0]
        by_edge_task[key].append((
            float(min(1.0, max(0.0, 1.0 - float(np.max(sims))))),
            float(min(1.0, max(0.0, 1.0 - float(np.mean(sims))))),
            len(alts)))

    # occurrence -> mean within (edge, task)
    per_task = collections.defaultdict(list)
    for (edge, task), vals in by_edge_task.items():
        per_task[edge].append((float(np.mean([v[0] for v in vals])),
                               float(np.mean([v[1] for v in vals])),
                               float(np.mean([v[2] for v in vals]))))
    # (edge, task) -> equal mean across tasks
    return {edge: {"nr_max": float(np.mean([t[0] for t in ts])),
                   "nr_mean": float(np.mean([t[1] for t in ts])),
                   "n_context": float(np.mean([t[2] for t in ts])),
                   "n_tasks": len(ts)}
            for edge, ts in per_task.items()}


def f2_from_records(records, fold_of=None):
    """RS per (edge, task) from probed f2 records, then equal-mean over tasks.

    Unprobed occurrences contribute nothing and are counted separately; an
    edge with no probed occurrence is ABSENT, never zero.

    ``fold_of`` maps a task id to its frozen fold label. When supplied, a
    per-fold mean is emitted alongside the pooled one so F2 can enter a
    cross-fitted block: a direction estimating features on fold A must not see
    any fold B task. Without this the pooled RS carries every task, and using
    it against fold B outcomes leaks the outcome tasks into their own
    features.
    """
    by_edge_task = collections.defaultdict(list)
    skipped = collections.Counter()
    for r in records:
        edge = (r["edge_src"], r["edge_dst"])
        if not r.get("probed"):
            skipped[edge] += 1
            continue
        clean, probe = r.get("clean_output"), r.get("probe_output")
        if not (clean or "").strip() or not (probe or "").strip():
            skipped[edge] += 1
            continue
        v = embed([clean, probe])
        by_edge_task[(edge, r["task_id"])].append(
            max(0.0, 1.0 - _cos(v[0], v[1])))

    per_task = collections.defaultdict(list)
    per_task_fold = collections.defaultdict(lambda: collections.defaultdict(list))
    n_occ = collections.defaultdict(int)
    for (edge, task), vals in by_edge_task.items():
        m = float(np.mean(vals))                        # occurrences -> task
        per_task[edge].append(m)
        if fold_of is not None:
            f = fold_of.get(task)
            if f is not None:
                per_task_fold[edge][f].append(m)
        n_occ[edge] += len(vals)

    def fold_stat(edge, label):
        vals = per_task_fold.get(edge, {}).get(label, [])
        return (float(np.mean(vals)) if vals else None), len(vals)
    # Retain spread and occurrence counts. Collapsing occurrences discards the
    # within-cell variance, so it is summarised here rather than lost: it is
    # what any later reliability weighting would need.
    out = {}
    for edge, ts in per_task.items():
        a, na = fold_stat(edge, "A")
        b, nb = fold_stat(edge, "B")
        out[edge] = {"rs": float(np.mean(ts)),
                     "rs_sd_across_tasks": (float(np.std(ts, ddof=1))
                                            if len(ts) > 1 else 0.0),
                     "n_tasks": len(ts),
                     "n_occurrences": n_occ[edge],
                     "rs_fold_A": a, "n_fold_A": na,
                     "rs_fold_B": b, "n_fold_B": nb}
    return out, dict(skipped)


def f3_for(topology, scenario_name, validity):
    G, _ = effective_graph(topology, validity)
    cfg = load_topology(str(REPO / "config" / "topologies"
                            / ("%s.yaml" % topology)))
    return consequence_proximity(G, cfg, scenario_name)


def _cluster_of(path):
    """Cluster id from the run_tag directory, e.g. ..._6011568_5."""
    for part in reversed(Path(path).parts):
        bits = part.split("_")
        if len(bits) >= 2 and bits[-2].isdigit():
            return int(bits[-2])
    return -1


def select_newest_runs(probe_files):
    """Keep only the latest cluster per (model, topology).

    Superseded pilot runs sit in the same tree as their reruns. Mixing an old
    run's records with its correction would silently average two different
    measurement definitions together.
    """
    best = {}
    for path in probe_files:
        parts = Path(path).parts
        # scenario MUST be in the key: results are laid out as
        # <scenario>/<model>/<topology>/<run_tag>/, and keying on model and
        # topology alone let a newer SE run evict the CS run for the same
        # cell, silently dropping most of the CS features.
        key = (parts[-5], parts[-4], parts[-3])   # scenario, model, topology
        c = _cluster_of(path)
        if key not in best or c > best[key][0]:
            best[key] = (c, path)
    return [p for _, p in sorted(best.values())]


def _fold_map():
    """Frozen task->fold split, per scenario. Never derived from outcomes."""
    p = REPO / "config" / "task_folds.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("folds", {})


def build(probe_files, validity, features=("f1", "f2", "f3")):
    """One record per (scenario, topology, model, edge).

    `features` selects which blocks to compute. F1
    (semantic_non_recoverability) is DROPPED from every scoring block but is
    still the most expensive thing here -- it embeds every alternative context
    of every message, which dominated a 90-minute rebuild. `--features f2 f3`
    skips it entirely and leaves its columns null, which is exactly how a
    dropped feature should read. The embedding cache is shared, so an F1 value
    already computed is still reused if it is ever asked for again.
    """
    want_f1 = "f1" in features
    want_f2 = "f2" in features
    want_f3 = "f3" in features
    grouped = collections.defaultdict(list)
    for path in probe_files:
        for line in open(path):
            r = json.loads(line)
            grouped[(r["scenario"], r["topology"], r["model"])].append(r)

    out = []
    for (scenario, topology, model), recs in sorted(grouped.items()):
        f1 = (f1_from_records([r for r in recs if r.get("kind") == "f1_context"])
              if want_f1 else {})
        if want_f2:
            f2, skipped = f2_from_records(
                [r for r in recs if r.get("kind") == "f2_probe"],
                fold_of=_fold_map().get(scenario))
        else:
            f2, skipped = {}, {}
        f3 = f3_for(topology, scenario, validity) if want_f3 else {}
        G, _ = effective_graph(topology, validity)
        for edge in sorted(G.edges()):
            a, b = f1.get(edge), f2.get(edge)
            out.append({
                "scenario": scenario, "topology": topology, "model": model,
                "edge_src": edge[0], "edge_dst": edge[1],
                "semantic_non_recoverability": a["nr_max"] if a else None,
                "nr_mean": a["nr_mean"] if a else None,
                "n_context": a["n_context"] if a else None,
                "f1_n_tasks": a["n_tasks"] if a else 0,
                "receiver_response_sensitivity": b["rs"] if b else None,
                "rs_sd_across_tasks": b["rs_sd_across_tasks"] if b else None,
                "f2_n_tasks": b["n_tasks"] if b else 0,
                "rs_fold_A": b["rs_fold_A"] if b else None,
                "rs_fold_B": b["rs_fold_B"] if b else None,
                "f2_n_fold_A": b["n_fold_A"] if b else 0,
                "f2_n_fold_B": b["n_fold_B"] if b else 0,
                "f2_n_occurrences": b["n_occurrences"] if b else 0,
                "f2_skipped_occurrences": skipped.get(edge, 0),
                "consequence_proximity": f3.get(edge),
            })
    return out


def coverage_report(records, require_f1=False):
    """Per-configuration completeness. Incomplete configurations are named."""
    by_cfg = collections.defaultdict(list)
    for r in records:
        by_cfg[(r["scenario"], r["topology"], r["model"])].append(r)
    report, complete = {}, []
    # F1 is dropped from every scoring block, so by default a configuration is
    # complete without it. Requiring it would mark every future incremental
    # build incomplete for a feature no block consumes.
    required = ["receiver_response_sensitivity", "consequence_proximity"]
    if require_f1:
        required.insert(0, "semantic_non_recoverability")
    for cfg, rows in sorted(by_cfg.items()):
        n = len(rows)
        cov = {f: sum(1 for r in rows if r[f] is not None) / n
               for f in ("semantic_non_recoverability",
                         "receiver_response_sensitivity",
                         "consequence_proximity")}
        full = all(cov[f] == 1.0 for f in required)
        report["|".join(cfg)] = {"n_edges": n, "coverage": cov,
                                 "complete": full}
        if full:
            complete.append(cfg)
    return report, complete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data" / "mas_features.json"))
    ap.add_argument("--features", nargs="+", default=["f2", "f3"],
                    choices=["f1", "f2", "f3"],
                    help="Blocks to compute. Default f2 f3: F1 is dropped from "
                         "every scoring block and embedding it dominates the "
                         "build. Pass --features f1 f2 f3 to reproduce the "
                         "legacy full build.")
    args = ap.parse_args()

    all_files = sorted(REPO.glob(PROBE_GLOB))
    if not all_files:
        print("no probe files found under %s" % PROBE_GLOB)
        return
    _install_cpu_limit_handler()
    files = select_newest_runs(all_files)
    print("reading %d probe file(s) (%d superseded run(s) skipped)"
          % (len(files), len(all_files) - len(files)))
    for f in files:
        print("   %s" % Path(f).parent.name)
    validity = load_validity()
    try:
        records = build(files, validity, features=tuple(args.features))
    finally:
        # Persist whatever was embedded, even if the process is about to be
        # killed by the CPU limit -- the next run resumes from here.
        save_embedding_cache()
    report, complete = coverage_report(records,
                                       require_f1="f1" in args.features)

    print("\n%-46s %6s %7s %7s %7s %s"
          % ("configuration", "edges", "F1", "F2", "F3", "complete"))
    for cfg, info in report.items():
        c = info["coverage"]
        print("%-46s %6d %7.2f %7.2f %7.2f %s"
              % (cfg, info["n_edges"], c["semantic_non_recoverability"],
                 c["receiver_response_sensitivity"],
                 c["consequence_proximity"],
                 "yes" if info["complete"] else "NO"))

    payload = {
        "aggregation": ("occurrence -> mean within (edge, task) -> equal mean "
                        "across tasks; tasks with more agent turns get no "
                        "extra weight"),
        "f1_primary": "nr_max",
        "f1_bias_note": ("nr_max falls as the number of alternative contexts "
                         "rises; n_context and nr_mean are emitted so the "
                         "bias can be modelled or checked against in-degree"),
        "coverage": report,
        "complete_configurations": ["|".join(c) for c in complete],
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print("\nwrote %d edge records (%d/%d complete configurations) -> %s"
          % (len(records), len(complete), len(report), args.out))


if __name__ == "__main__":
    main()
