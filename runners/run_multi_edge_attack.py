"""Evaluate simultaneous attacks on top-, random-, or bottom-ranked edges.

Budgets default to 2, 3, and 5. Output records include the selected edges,
strategy, budget, and seed.
"""

import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.topology.builder import load_topology
from src.agents.mas_runner import MASRunner
from src.evaluation.metrics import evaluate_task
from src.attacks.misinformation import dynamic_misinformation_attack
from src.saliency.structural import compute_all_structural_features

RESULTS_DIR = Path("results")
DEFAULT_SCENARIO = "config/scenarios/customer_service.yaml"
BOT_KEYS = ["betweenness_centrality", "information_bottleneck", "is_bridge"]
DEG_KEYS = ["endpoint_centrality_max", "source_degree_centrality", "target_degree_centrality"]


def load_scenario(path):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_struct_saliency(graph):
    """Unsupervised structural saliency = bottleneck-rank-sum minus degree-rank-sum.
    Same sign convention as paper (no learned parameters)."""
    feats = compute_all_structural_features(graph)
    edges = list(feats.keys())
    bot = np.column_stack([np.array([feats[e][k] for e in edges]) for k in BOT_KEYS])
    deg = np.column_stack([np.array([feats[e][k] for e in edges]) for k in DEG_KEYS])
    s = np.zeros(len(edges))
    for j in range(bot.shape[1]):
        s += stats.rankdata(bot[:, j])
    for j in range(deg.shape[1]):
        s -= stats.rankdata(deg[:, j])
    return list(zip(edges, s))


def compute_combined_saliency(graph, model: str, topology_name: str,
                                scenario_slug: str = "customer_service"):
    """Combined saliency = structural + ablation_delta + perturbation_delta
    rank sums, loaded from existing per-(model, topology) ablation/perturbation
    JSON files.  Falls back to structural-only if dynamic data isn't available
    for this (model, topology, scenario) combination.

    This breaks structural ties (e.g., in centralized where all hub-spoke
    edges share features) by adding the direction-aware dynamic signals.
    """
    import json
    feats = compute_all_structural_features(graph)
    edges = list(feats.keys())
    # Start from structural rank-sum
    bot = np.column_stack([np.array([feats[e][k] for e in edges]) for k in BOT_KEYS])
    deg = np.column_stack([np.array([feats[e][k] for e in edges]) for k in DEG_KEYS])
    s = np.zeros(len(edges))
    for j in range(bot.shape[1]): s += stats.rankdata(bot[:, j])
    for j in range(deg.shape[1]): s -= stats.rankdata(deg[:, j])

    # Add ablation + perturbation rank components.  Files have keys like
    # dynamic_ablation_customer_service_qwen3.5-9b_centralized.json.
    model_slug = model.replace(":", "-")
    abl_path = RESULTS_DIR / f"dynamic_ablation_{scenario_slug}_{model_slug}_{topology_name}.json"
    pert_path = RESULTS_DIR / f"dynamic_perturbation_{scenario_slug}_{model_slug}_{topology_name}.json"

    def _edge_deltas(path):
        if not path.exists(): return None
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list): return None
        from collections import defaultdict
        # For each (intervention_edge), compute (clean_topology_acc - acc_with_intervention)
        per_edge = defaultdict(list)
        for r in data:
            if not isinstance(r, dict) or "error" in r: continue
            sc = r.get("scores")
            if not isinstance(sc, dict): continue
            acc = sc.get("decision_accuracy")
            if acc is None or acc < 0: continue
            ie = r.get("intervention_edge")
            if not ie: continue
            per_edge[tuple(ie)].append(acc)
        if not per_edge: return None
        # Approximate per-edge delta as (1 - mean_acc).  Not exact (we don't
        # have the per-task clean baseline here) but the *ranking* is the same.
        return {edge: 1 - sum(vs)/len(vs) for edge, vs in per_edge.items()}

    abl = _edge_deltas(abl_path)
    pert = _edge_deltas(pert_path)
    if abl:
        abl_vec = np.array([abl.get(e, 0.0) for e in edges])
        s += stats.rankdata(abl_vec)
    if pert:
        pert_vec = np.array([pert.get(e, 0.0) for e in edges])
        s += stats.rankdata(pert_vec)
    if not abl and not pert:
        print(f"  [combined] no dynamic data for {model_slug}/{topology_name}; "
              f"falling back to struct-only")
    else:
        n_avail = sum(1 for x in [abl, pert] if x)
        print(f"  [combined] using struct + {n_avail} dynamic feature(s) for "
              f"{model_slug}/{topology_name}")
    return list(zip(edges, s))


def select_edges(edge_saliency, k, strategy, rng=None):
    """Return list of (src, tgt) tuples chosen by the strategy."""
    if not edge_saliency:
        return []
    if strategy == "top":
        sorted_edges = sorted(edge_saliency, key=lambda x: -x[1])
        return [e for e, _ in sorted_edges[:k]]
    elif strategy == "bottom":
        sorted_edges = sorted(edge_saliency, key=lambda x: x[1])
        return [e for e, _ in sorted_edges[:k]]
    elif strategy == "random":
        idx = rng.choice(len(edge_saliency), size=min(k, len(edge_saliency)), replace=False)
        return [edge_saliency[i][0] for i in idx]
    raise ValueError(f"unknown strategy {strategy}")


def get_role_for_agent(topology_cfg, agent_id):
    for a in topology_cfg["agents"]:
        if a["id"] == agent_id:
            return a["role"]
    return None


def run_one_trial(topology_cfg, scenario, model, task, attack_edges):
    """Set up a MASRunner and run one task with multi-edge attack."""
    runner = MASRunner(topology_cfg, scenario, model=model)
    pairs = []
    for (u, v) in attack_edges:
        sender_role = get_role_for_agent(topology_cfg, u)
        # Capture vars in lambda by default-arg trick
        fn = (lambda msg, role=sender_role, m=model, tid=task["id"]:
              dynamic_misinformation_attack(msg, sender_role=role, model=m,
                                              task_id=tid))
        pairs.append((u, v, fn))
    runner.set_multi_attack(pairs)
    try:
        result = runner.run(task)
    except Exception as e:
        return {"error": str(e), "task_id": task.get("id", "?")}
    resolution = result.get("final_resolution", "")
    scenario_name = scenario.get("name", "")
    scores = evaluate_task(resolution=resolution, task=task,
                            task_description=task["description"],
                            scenario_name=scenario_name, model=model)
    return {
        "task_id": task["id"],
        "final_resolution": resolution,
        "scores": scores,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO)
    ap.add_argument("--budgets", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--random_seeds", type=int, default=10,
                    help="Number of random-strategy seeds per budget")
    ap.add_argument("--saliency", choices=["struct", "combined"], default="combined",
                    help="Saliency to use for top/bottom edge selection")
    ap.add_argument("--output_dir", default=str(RESULTS_DIR))
    args = ap.parse_args()

    print(f"=== Multi-edge attack ===")
    print(f"  model    : {args.model}")
    print(f"  topology : {args.topology}")
    print(f"  scenario : {args.scenario}")
    print(f"  budgets  : {args.budgets}")
    print(f"  rand seeds: {args.random_seeds}")

    topology_cfg = load_topology(f"config/topologies/{args.topology}.yaml")
    scenario = load_scenario(args.scenario)
    runner_dummy = MASRunner(topology_cfg, scenario, model=args.model)
    if args.saliency == "combined":
        scenario_slug = Path(args.scenario).stem
        edge_saliency = compute_combined_saliency(
            runner_dummy.graph, args.model, args.topology, scenario_slug)
    else:
        edge_saliency = compute_struct_saliency(runner_dummy.graph)
    n_edges = len(edge_saliency)
    print(f"  edges    : {n_edges}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_slug = Path(args.scenario).stem
    model_slug = args.model.replace(":", "-")
    out_path = out_dir / f"multiedge_{scenario_slug}_{model_slug}_{args.topology}.json"

    # Resume support: if file exists, skip already-done tuples
    completed = set()
    existing = []
    if out_path.exists():
        try:
            with open(out_path) as f: existing = json.load(f)
            for r in existing:
                completed.add((r.get("strategy"), r.get("k"), r.get("seed", 0),
                                r.get("task_id")))
            print(f"  resume   : {len(existing)} trials already done")
        except Exception:
            pass

    tasks = scenario["tasks"]
    results = list(existing)

    rng = np.random.RandomState(42)
    for k in args.budgets:
        if k > n_edges:
            print(f"  skip k={k} (only {n_edges} edges)")
            continue
        # Deterministic strategies
        for strategy in ["top", "bottom"]:
            edges = select_edges(edge_saliency, k, strategy)
            for task in tasks:
                key = (strategy, k, 0, task["id"])
                if key in completed: continue
                t0 = time.time()
                trial = run_one_trial(topology_cfg, scenario, args.model, task, edges)
                trial.update({
                    "model": args.model, "topology": args.topology,
                    "scenario": scenario_slug,
                    "k": k, "strategy": strategy, "seed": 0,
                    "attack_edges": [list(e) for e in edges],
                    "elapsed_s": round(time.time() - t0, 1),
                })
                results.append(trial)
                with open(out_path, "w") as f: json.dump(results, f, indent=1)
                print(f"  {strategy} k={k} seed=0 task={task['id']} "
                      f"acc={trial.get('scores',{}).get('decision_accuracy','?')} "
                      f"({trial.get('elapsed_s', '?')}s)")
        # Random strategy: multiple seeds
        for seed in range(args.random_seeds):
            local_rng = np.random.RandomState(1000 + seed)
            edges = select_edges(edge_saliency, k, "random", rng=local_rng)
            for task in tasks:
                key = ("random", k, seed, task["id"])
                if key in completed: continue
                t0 = time.time()
                trial = run_one_trial(topology_cfg, scenario, args.model, task, edges)
                trial.update({
                    "model": args.model, "topology": args.topology,
                    "scenario": scenario_slug,
                    "k": k, "strategy": "random", "seed": seed,
                    "attack_edges": [list(e) for e in edges],
                    "elapsed_s": round(time.time() - t0, 1),
                })
                results.append(trial)
                with open(out_path, "w") as f: json.dump(results, f, indent=1)
                print(f"  random k={k} seed={seed} task={task['id']} "
                      f"acc={trial.get('scores',{}).get('decision_accuracy','?')} "
                      f"({trial.get('elapsed_s', '?')}s)")

    print(f"\nDone. Total trials: {len(results)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
