"""Compare surface-aware defenses under a restricted attacker.

The attacker selects one direct or low-trust tool edge. Defenses compare MESA,
random, oracle, unrestricted, and no-monitoring allocations.
"""

import argparse
import json
import sys
import time
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

REALISTIC_EXPOSURE = {"direct", "tool_low"}

# Defense strategies to evaluate
DEFENSE_STRATEGIES = [
    "undefended",
    "mesa_combined",   # struct + dynamic (0.5/0.5) — proposed
    "mesa_surface",    # struct-only surface ranking — ablation
    "oracle_surface",  # oracle upper bound
    "random_surface",  # random surface baseline
    "mesa_any",        # struct-only global (budget waste test)
]


# ── Load surface edges from topology YAML ────────────────────────────────────

def load_surface_edges(topo_name: str) -> list:
    """Return list of (src, tgt) tuples with exposure in REALISTIC_EXPOSURE."""
    fpath = Path(f"config/topologies/{topo_name}.yaml")
    cfg = yaml.safe_load(open(fpath))
    result = []
    for e in cfg.get("edges", []):
        src, tgt = e["source"], e["target"]
        if e.get("exposure", "internal") in REALISTIC_EXPOSURE:
            result.append((src, tgt))
        if e.get("bidirectional"):
            if e.get("exposure_reverse", "internal") in REALISTIC_EXPOSURE:
                result.append((tgt, src))
    return result


# ── Load oracle ASR from existing pilot_attack results ───────────────────────

def load_oracle_asr(topo_name: str) -> dict:
    """Return {(src, tgt): corrected_asr} for the given topology, pooled over models."""
    clean = {}
    for f in sorted(RESULTS_DIR.glob("pilot_clean_*.json")):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for r in data:
            if "error" in r:
                continue
            a = r.get("scores", {}).get("decision_accuracy", -1)
            if a >= 0:
                clean[(r["model"], r["topology"], r["task_id"])] = a

    pool = defaultdict(lambda: {"flip": 0, "elig": 0})
    for f in sorted(RESULTS_DIR.glob("pilot_attack_*.json")):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for r in data:
            if "error" in r or r.get("topology") != topo_name:
                continue
            a = r.get("scores", {}).get("decision_accuracy", -1)
            if a < 0:
                continue
            cl = clean.get((r["model"], r["topology"], r["task_id"]), -1)
            if cl != 1:
                continue
            edge = r.get("attack_edge")
            if not edge:
                continue
            key = (edge[0], edge[1])
            pool[key]["elig"] += 1
            if a == 0:
                pool[key]["flip"] += 1

    return {k: v["flip"] / v["elig"] for k, v in pool.items() if v["elig"] >= 5}


# ── Dynamic saliency scores (ablation / perturbation) ────────────────────────

def load_dynamic_scores(topo_name: str, model: str) -> dict:
    """Return {(src, tgt): ablation_delta} from pre-computed dynamic saliency runs.

    ablation_delta = clean_acc - ablation_intervened_acc  (positive = edge matters)
    Falls back to 0.0 for edges with no data (safe: they rank lower).
    """
    model_slug = model.replace(":", "-").replace("/", "-")
    scores = {}
    for phase in ["ablation", "perturbation"]:
        pat = RESULTS_DIR / f"dynamic_{phase}_customer_service_{model_slug}_{topo_name}.json"
        if not pat.exists():
            continue
        # Compute per-edge mean accuracy when that edge is intervened on
        edge_accs = defaultdict(list)
        for r in json.load(open(pat)):
            if "error" in r:
                continue
            e = tuple(r.get("intervention_edge", []))
            a = r.get("scores", {}).get("decision_accuracy", -1)
            if e and a >= 0:
                edge_accs[e].append(a)
        for e, accs in edge_accs.items():
            scores.setdefault(e, {})[phase] = float(np.mean(accs))

    # ablation_delta = how much accuracy drops when edge is removed
    # Use clean accuracy from pilot_clean as reference
    clean_acc = {}
    for f in sorted(RESULTS_DIR.glob("pilot_clean_*.json")):
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for r in data:
            if r.get("topology") == topo_name and not r.get("error"):
                a = r.get("scores", {}).get("decision_accuracy", -1)
                if a >= 0:
                    clean_acc[r.get("task_id", "")] = a

    mean_clean = float(np.mean(list(clean_acc.values()))) if clean_acc else 0.8

    result = {}
    for e, d in scores.items():
        abl = d.get("ablation", mean_clean)
        result[e] = mean_clean - abl  # positive = edge is important
    return result


# ── Structural MESA ranking ───────────────────────────────────────────────────

BOT_KEYS = ["betweenness_centrality", "information_bottleneck", "is_bridge"]
DEG_KEYS = ["endpoint_centrality_max", "source_degree_centrality",
            "target_degree_centrality"]


def compute_mesa_ranking(graph, dynamic_scores: dict = None) -> list:
    """Return list of ((src, tgt), score) sorted descending by MESA.

    If dynamic_scores is provided, uses combined struct+dynamic (0.5/0.5).
    Otherwise falls back to structural-only rank-sum.
    """
    feats = compute_all_structural_features(graph)
    edges = list(feats.keys())
    bot = np.column_stack([np.array([feats[e][k] for e in edges]) for k in BOT_KEYS])
    deg = np.column_stack([np.array([feats[e][k] for e in edges]) for k in DEG_KEYS])
    s_struct = np.zeros(len(edges))
    for j in range(bot.shape[1]):
        s_struct += stats.rankdata(bot[:, j])
    for j in range(deg.shape[1]):
        s_struct -= stats.rankdata(deg[:, j])

    if dynamic_scores is None:
        return sorted(zip(edges, s_struct), key=lambda x: -x[1])

    # Combined: normalize struct rank-sum to [0,1], then average with normalized ablation_delta
    abl_vals = np.array([dynamic_scores.get(e, 0.0) for e in edges])
    def norm01(v):
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 1e-9 else np.zeros_like(v)

    struct_norm = norm01(s_struct)
    abl_norm    = norm01(abl_vals)
    combined    = 0.5 * struct_norm + 0.5 * abl_norm
    return sorted(zip(edges, combined), key=lambda x: -x[1])


# ── Defense edge selection per strategy ──────────────────────────────────────

def select_defense(strategy: str, k_d: int, surface_edges: list,
                   ranked_all: list, ranked_combined: list,
                   oracle_asr: dict, trial_seed: int) -> list:
    """
    Returns list of (src, tgt) edges for the defender to monitor.

    strategy:
      undefended     — []
      mesa_combined  — top-k_d surface edges by struct+dynamic combined ranking
      mesa_surface   — top-k_d surface edges by structural-only MESA rank
      oracle_surface — top-k_d surface edges by known corrected ASR (oracle)
      random_surface — k_d random surface edges (seed = trial_seed)
      mesa_any       — top-k_d edges globally by structural MESA (ignores surface)
    """
    if strategy == "undefended" or k_d == 0:
        return []

    if strategy == "mesa_combined":
        ranked_surf = [(e, sc) for e, sc in ranked_combined if tuple(e) in set(surface_edges)]
        return [list(e) for e, _ in ranked_surf[:k_d]]

    if strategy == "mesa_surface":
        ranked_surf = [(e, sc) for e, sc in ranked_all if tuple(e) in set(surface_edges)]
        return [list(e) for e, _ in ranked_surf[:k_d]]

    if strategy == "oracle_surface":
        sorted_surf = sorted(surface_edges, key=lambda e: -oracle_asr.get(e, 0.0))
        return [list(e) for e in sorted_surf[:k_d]]

    if strategy == "random_surface":
        rng = np.random.RandomState(trial_seed + 9999)
        n = min(k_d, len(surface_edges))
        chosen = rng.choice(len(surface_edges), size=n, replace=False)
        return [list(surface_edges[i]) for i in chosen]

    if strategy == "mesa_any":
        return [list(e) for e, _ in ranked_all[:k_d]]

    raise ValueError(f"Unknown defense strategy: {strategy}")


# ── Trial runner ──────────────────────────────────────────────────────────────

def run_trial(topology_cfg, scenario, model, task,
              attack_edge: tuple, defense_edges: list,
              save_edge_log: bool = False) -> dict:
    """Run one task with a single surface-edge attack and a defense set."""
    runner = MASRunner(topology_cfg, scenario, model=model)

    src, tgt = attack_edge
    role = next(
        (a["role"] for a in topology_cfg["agents"] if a["id"] == src), src
    )
    attack_fn = (lambda msg, r=role, m=model, tid=task["id"]:
                 dynamic_misinformation_attack(msg, sender_role=r, model=m, task_id=tid))
    runner.set_attack(src, tgt, attack_fn)

    if defense_edges:
        runner.set_defense([tuple(e) for e in defense_edges])

    t0 = time.time()
    try:
        result = runner.run(task)
    except Exception as e:
        return {"error": str(e), "task_id": task.get("id", "?")}

    resolution = result.get("final_resolution", "")
    scores = evaluate_task(
        resolution=resolution, task=task,
        task_description=task["description"],
        scenario_name=scenario.get("name", ""),
        model=model,
    )

    out = {
        "final_resolution": resolution,
        "scores": scores,
        "elapsed_s": round(time.time() - t0, 1),
    }

    if save_edge_log:
        # Full edge log with message content — for B2 propagation analysis
        out["edge_log"] = runner.get_edge_log()

    return out


# ── Propagation probe (B2 Phase 1, optional) ─────────────────────────────────

def count_keyword_propagation(edge_log: list, attack_edge: tuple,
                               task_id: str) -> dict:
    """
    Given a full edge_log (list of EdgeMessage dicts), check whether the attack
    keywords injected on attack_edge appear in messages on downstream edges.

    Returns:
      {
        "attack_edge": (src, tgt),
        "n_downstream_msgs": int,
        "n_keyword_hits": int,
        "propagation_rate": float,  # keyword_hits / downstream_msgs
        "attacked_msg_content": str | None,
      }
    """
    # Import task-level attack goals to get keywords
    try:
        from src.attacks.misinformation import ATTACK_GOALS
        goal = ATTACK_GOALS.get(task_id, {})
        keywords = goal.get("attack_keywords", [])
        if not keywords:
            # Fall back to attack direction
            attack_dir = goal.get("attack_direction", "")
            keywords = [w for w in attack_dir.lower().split() if len(w) > 4]
    except Exception:
        keywords = []

    attacked_content = None
    downstream_msgs = []

    for msg in edge_log:
        e = (msg["source"], msg["target"])
        if e == attack_edge and msg.get("was_attacked"):
            attacked_content = msg.get("content", "")
        elif attacked_content is not None:
            # All messages AFTER the attack edge are "downstream"
            downstream_msgs.append(msg.get("content", ""))

    if not downstream_msgs or not keywords:
        return {
            "attack_edge": list(attack_edge),
            "n_downstream_msgs": len(downstream_msgs),
            "n_keyword_hits": 0,
            "propagation_rate": 0.0,
            "attacked_msg_content": attacked_content,
        }

    hits = sum(
        1 for msg in downstream_msgs
        if any(kw.lower() in msg.lower() for kw in keywords)
    )
    return {
        "attack_edge": list(attack_edge),
        "n_downstream_msgs": len(downstream_msgs),
        "n_keyword_hits": hits,
        "propagation_rate": hits / len(downstream_msgs),
        "attacked_msg_content": attacked_content,
    }


# ── Incremental result saver ──────────────────────────────────────────────────

def load_existing(out_path: Path):
    if not out_path.exists():
        return [], set()
    try:
        existing = json.load(open(out_path))
    except Exception:
        return [], set()
    completed = {
        (r["defense_strategy"], r["attack_edge_str"],
         r.get("trial_seed", 0), r["task_id"])
        for r in existing if "error" not in r
    }
    return existing, completed


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_surface_defense(args):
    model_slug   = args.model.replace(":", "-")
    scenario_cfg = yaml.safe_load(open(args.scenario))
    scenario_slug = Path(args.scenario).stem
    tasks = scenario_cfg["tasks"]

    print(f"=== Surface-Restricted Defense (B1) ===")
    print(f"  model    : {args.model}")
    print(f"  topology : {args.topology}")
    print(f"  n_trials : {args.n_trials}  k_d : {args.k_d}")

    # Load topology and compute infrastructure
    topology_cfg   = load_topology(f"config/topologies/{args.topology}.yaml")
    runner_tmp     = MASRunner(topology_cfg, scenario_cfg, model=args.model)
    dynamic_scores = load_dynamic_scores(args.topology, args.model)
    ranked_all     = compute_mesa_ranking(runner_tmp.graph)
    ranked_combined = compute_mesa_ranking(runner_tmp.graph, dynamic_scores)
    surface        = load_surface_edges(args.topology)
    oracle_asr     = load_oracle_asr(args.topology)

    if not surface:
        print(f"  [error] no surface edges found for {args.topology} — check YAML exposure labels")
        return

    n_dyn = sum(1 for e in surface if e in dynamic_scores)
    print(f"  surface edges ({len(surface)}): {surface}")
    print(f"  dynamic scores available: {n_dyn}/{len(surface)} surface edges")
    print(f"  top-1 MESA struct (any)  : {ranked_all[0][0]}")
    print(f"  top-1 MESA combined (any): {ranked_combined[0][0]}")

    ranked_surf = [(e, sc) for e, sc in ranked_all if tuple(e) in set(surface)]
    if ranked_surf:
        print(f"  top-1 MESA struct (surf) : {ranked_surf[0][0]}")
    ranked_surf_c = [(e, sc) for e, sc in ranked_combined if tuple(e) in set(surface)]
    if ranked_surf_c:
        print(f"  top-1 MESA combined(surf): {ranked_surf_c[0][0]}")

    oracle_best = max(surface, key=lambda e: oracle_asr.get(e, 0.0))
    print(f"  oracle best surf : {oracle_best}  ASR={oracle_asr.get(oracle_best, '?'):.3f}")

    out_path = (Path(args.output_dir) /
                f"surface_defense_{scenario_slug}_{model_slug}_{args.topology}.json")
    results, completed = load_existing(out_path)
    print(f"  resume   : {len(results)} existing results")

    total_new = 0
    rng = np.random.RandomState(args.seed)

    for trial_seed in range(args.n_trials):
        # Attacker randomly picks one surface edge for this trial
        attack_idx  = rng.randint(0, len(surface))
        attack_edge = surface[attack_idx]
        atk_str     = f"{attack_edge[0]}->{attack_edge[1]}"

        for strategy in DEFENSE_STRATEGIES:
            defense_edges = select_defense(
                strategy, args.k_d, surface, ranked_all, ranked_combined,
                oracle_asr, trial_seed
            )
            defense_set = frozenset(tuple(e) for e in defense_edges)

            for task in tasks:
                key = (strategy, atk_str, trial_seed, task["id"])
                if key in completed:
                    continue

                trial = run_trial(
                    topology_cfg, scenario_cfg, args.model, task,
                    attack_edge, defense_edges,
                    save_edge_log=args.save_edge_log,
                )

                # Propagation stats if edge_log available
                prop_stats = None
                if args.save_edge_log and "edge_log" in trial:
                    prop_stats = count_keyword_propagation(
                        trial["edge_log"], attack_edge, task["id"]
                    )

                # Was the attack intercepted? (attacked edge was in defense set)
                intercepted = tuple(attack_edge) in defense_set

                record = {
                    "model":            args.model,
                    "topology":         args.topology,
                    "scenario":         scenario_slug,
                    "defense_strategy": strategy,
                    "attack_edge":      list(attack_edge),
                    "attack_edge_str":  atk_str,
                    "defense_edges":    [list(e) for e in defense_edges],
                    "intercepted":      intercepted,
                    "trial_seed":       trial_seed,
                    "task_id":          task["id"],
                    "scores":           trial.get("scores"),
                    "elapsed_s":        trial.get("elapsed_s"),
                    "propagation":      prop_stats,
                }
                if "error" in trial:
                    record["error"] = trial["error"]

                results.append(record)
                completed.add(key)
                total_new += 1

                acc = trial.get("scores", {}).get("decision_accuracy", "?")
                icpt = "DEFENDED" if intercepted else "attacked"
                print(f"  [seed={trial_seed:2d} {atk_str:30s}] "
                      f"{strategy:<16} {icpt:<9} {task['id']} acc={acc} "
                      f"({trial.get('elapsed_s','?')}s)")

                with open(out_path, "w") as f:
                    json.dump(results, f, indent=1)

    print(f"\nDone. New records: {total_new}, total: {len(results)}")
    print(f"Output: {out_path}")

    # Print summary
    print_summary(results, args.topology)


def print_summary(results: list, topology: str):
    """Print mean accuracy per defense strategy."""
    by_strat = defaultdict(list)
    for r in results:
        if "error" in r or r.get("topology") != topology:
            continue
        acc = r.get("scores", {}).get("decision_accuracy")
        if acc is not None:
            by_strat[r["defense_strategy"]].append(acc)

    print(f"\n=== Summary: {topology} ===")
    print(f"  {'Strategy':<18} {'Mean acc':>9} {'n':>5}")
    for strat in DEFENSE_STRATEGIES:
        vals = by_strat[strat]
        if vals:
            print(f"  {strat:<18} {np.mean(vals):>9.3f} {len(vals):>5}")


def main():
    ap = argparse.ArgumentParser(description="Surface-restricted defense experiment (B1)")
    ap.add_argument("--model",       required=True)
    ap.add_argument("--topology",    required=True)
    ap.add_argument("--scenario",    default=DEFAULT_SCENARIO)
    ap.add_argument("--n_trials",    type=int, default=20,
                    help="Number of random attacker trials (different surface edges)")
    ap.add_argument("--k_d",         type=int, default=1,
                    help="Defender monitoring budget (edges to watch)")
    ap.add_argument("--seed",        type=int, default=42,
                    help="Master RNG seed for attacker edge sampling")
    ap.add_argument("--output_dir",  default=str(RESULTS_DIR))
    ap.add_argument("--save_edge_log", action="store_true",
                    help="Store full edge_log with message content (for B2 propagation analysis)")
    args = ap.parse_args()

    run_surface_defense(args)


if __name__ == "__main__":
    main()
