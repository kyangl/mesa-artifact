"""Compare MESA-guided monitoring with adaptive multi-edge attacks.

``exp1`` uses fixed defender and attacker budgets; ``exp2`` sweeps budget
asymmetry. Corrected ASR includes only tasks with correct clean baselines.
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

BOT_KEYS = ["betweenness_centrality", "information_bottleneck", "is_bridge"]
DEG_KEYS = ["endpoint_centrality_max", "source_degree_centrality", "target_degree_centrality"]

# Experiment 1: fixed budget
EXP1_K_D = 2
EXP1_K_A = 3

# Experiment 2: (k_d, k_a) pairs to sweep
EXP2_SWEEP = [
    (1, 1), (1, 2), (1, 3),
    (2, 2), (2, 3), (2, 5),
    (3, 3), (3, 5),
]

DEFENSE_STRATEGIES = ["none", "top_k", "bottom_k", "random_k",
                      "random_top_r3", "random_top_r4",
                      "mixed_top1_r3", "mixed_top1_r4"]
ATTACK_STRATEGIES  = ["naive", "gray_box", "adaptive", "adaptive_policy"]

# Exp2 uses worst-case adaptive attacker only, three defense strategies
EXP2_DEFENSES = ["none", "top_k", "random_k"]
EXP2_ATTACK   = "adaptive"


# ── Saliency ranking ──────────────────────────────────────────────────────────

def compute_struct_saliency(graph):
    """Structural saliency rank-sum (same as run_multi_edge_attack.py)."""
    feats = compute_all_structural_features(graph)
    edges = list(feats.keys())
    bot = np.column_stack([np.array([feats[e][k] for e in edges]) for k in BOT_KEYS])
    deg = np.column_stack([np.array([feats[e][k] for e in edges]) for k in DEG_KEYS])
    s = np.zeros(len(edges))
    for j in range(bot.shape[1]):
        s += stats.rankdata(bot[:, j])
    for j in range(deg.shape[1]):
        s -= stats.rankdata(deg[:, j])
    # Return sorted descending (highest saliency first)
    ranked = sorted(zip(edges, s), key=lambda x: -x[1])
    return ranked  # list of ((src, dst), score)


# ── Edge selection ────────────────────────────────────────────────────────────

def _select_top(ranked, k):
    return [e for e, _ in ranked[:k]]

def _select_bottom(ranked, k):
    return [e for e, _ in ranked[-k:]]

def _select_random(ranked, k, seed):
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(ranked), size=min(k, len(ranked)), replace=False)
    return [ranked[i][0] for i in idxs]


def _select_random_top_r(ranked, k_d, r, seed):
    """Sample k_d edges uniformly without replacement from the top-r MESA edges."""
    r = min(r, len(ranked))
    k_d = min(k_d, r)
    rng = np.random.RandomState(seed)
    pool = [e for e, _ in ranked[:r]]
    idxs = rng.choice(r, size=k_d, replace=False)
    return [pool[i] for i in idxs]


def _select_mixed_top1(ranked, k_d, r, seed):
    """Mixed policy: top-1 always defended + (k_d-1) random from ranks 2..r."""
    if k_d <= 0:
        return []
    out = [ranked[0][0]]   # top-1 fixed
    remaining = k_d - 1
    if remaining <= 0 or r <= 1:
        return out
    r_eff = min(r, len(ranked))
    pool = [e for e, _ in ranked[1:r_eff]]
    if not pool:
        return out
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(pool), size=min(remaining, len(pool)), replace=False)
    out.extend([pool[i] for i in idxs])
    return out


def select_defense_edges(ranked, k_d, strategy, seed=0):
    """
    strategy: 'none' | 'top_k' | 'bottom_k' | 'random_k' |
              'random_top_r3' | 'random_top_r4' |
              'mixed_top1_r3' | 'mixed_top1_r4'
    Returns list of (src, dst) tuples to monitor/defend.
    """
    if strategy == "none" or k_d == 0:
        return []
    if strategy == "top_k":
        return _select_top(ranked, k_d)
    if strategy == "bottom_k":
        return _select_bottom(ranked, k_d)
    if strategy == "random_k":
        return _select_random(ranked, k_d, seed=1000 + seed)
    if strategy == "random_top_r3":
        return _select_random_top_r(ranked, k_d, r=3, seed=3000 + seed)
    if strategy == "random_top_r4":
        return _select_random_top_r(ranked, k_d, r=4, seed=4000 + seed)
    if strategy == "mixed_top1_r3":
        return _select_mixed_top1(ranked, k_d, r=3, seed=5000 + seed)
    if strategy == "mixed_top1_r4":
        return _select_mixed_top1(ranked, k_d, r=4, seed=6000 + seed)
    raise ValueError(f"Unknown defense strategy: {strategy}")


def _adaptive_policy_aware(ranked, k_a, defense_policy, k_d):
    """
    Policy-aware white-box adaptive attacker.

    The attacker knows the defense POLICY (and its k_d, r) but not the
    specific draw. For each edge, computes expected-unprotected MESA score:

        E[unprotected_MESA(e)] = (1 - P(e defended | policy)) * MESA(e)

    then picks top-k_a edges by expected unprotected MESA.
    For deterministic policies (top_k, bottom_k) this reduces to picking
    the highest-MESA edge not in the deterministic defended set, which
    matches the original adaptive behaviour.
    """
    n = len(ranked)
    p_defended = np.zeros(n)
    if defense_policy in ("none",):
        pass
    elif defense_policy == "top_k":
        for i in range(min(k_d, n)):
            p_defended[i] = 1.0
    elif defense_policy == "bottom_k":
        for i in range(max(0, n - k_d), n):
            p_defended[i] = 1.0
    elif defense_policy == "random_k":
        p_defended[:] = k_d / n
    elif defense_policy in ("random_top_r3", "random_top_r4"):
        r = 3 if defense_policy.endswith("r3") else 4
        r = min(r, n)
        p_defended[:r] = k_d / r
    elif defense_policy in ("mixed_top1_r3", "mixed_top1_r4"):
        r = 3 if defense_policy.endswith("r3") else 4
        r = min(r, n)
        p_defended[0] = 1.0
        # remaining k_d-1 spread uniformly over ranks 2..r-1
        pool_size = max(0, r - 1)
        if pool_size > 0:
            p_defended[1:r] = (k_d - 1) / pool_size

    # Expected unprotected MESA score (MESA in `ranked` is already sorted desc).
    mesa_scores = np.array([s for _, s in ranked])
    exp_unprot = (1.0 - p_defended) * mesa_scores
    order = np.argsort(-exp_unprot)
    k_eff = min(k_a, n)
    return [ranked[i][0] for i in order[:k_eff]]


def select_attack_edges(ranked, k_a, strategy, defense_set, seed=0,
                        defense_policy="top_k", k_d=1):
    """
    strategy: 'naive' | 'gray_box' | 'adaptive' | 'adaptive_policy'
    defense_set: frozenset of defended (src, dst) tuples (used by classic adaptive).
    defense_policy / k_d: required for policy-aware adaptive against randomized policies.
    """
    if strategy == "naive":
        return _select_random(ranked, k_a, seed=2000 + seed)
    if strategy == "gray_box":
        return _select_top(ranked, k_a)
    if strategy == "adaptive":
        # Classic: knows the exact realized defended set
        undefended = [(e, s) for (e, s) in ranked if tuple(e) not in defense_set]
        k_eff = min(k_a, len(undefended))
        return _select_top(undefended, k_eff)
    if strategy == "adaptive_policy":
        # Policy-aware: knows the policy but not the realised draw
        return _adaptive_policy_aware(ranked, k_a, defense_policy, k_d)
    raise ValueError(f"Unknown attack strategy: {strategy}")


# ── Trial runner ──────────────────────────────────────────────────────────────

def get_role_for_agent(topology_cfg, agent_id):
    for a in topology_cfg["agents"]:
        if a["id"] == agent_id:
            return a["role"]
    return None


def run_trial(topology_cfg, scenario, model, task, attack_edges, defense_edges):
    """Run one task with multi-edge attack + defense. Returns result dict."""
    runner = MASRunner(topology_cfg, scenario, model=model)

    # Set attacks
    pairs = []
    for (u, v) in attack_edges:
        role = get_role_for_agent(topology_cfg, u)
        fn = (lambda msg, r=role, m=model, tid=task["id"]:
              dynamic_misinformation_attack(msg, sender_role=r, model=m, task_id=tid))
        pairs.append((u, v, fn))
    if pairs:
        runner.set_multi_attack(pairs)

    # Set defense (neutralizes attacks on monitored edges)
    if defense_edges:
        runner.set_defense(defense_edges)

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
        model=model)

    # Count how many attacks actually landed (not defended)
    defended_set = {tuple(e) for e in defense_edges}
    attacked_set = {tuple(e) for e in attack_edges}
    n_landed = len(attacked_set - defended_set)

    return {
        "final_resolution": resolution,
        "scores": scores,
        "n_effective_attacks": n_landed,
        "elapsed_s": round(time.time() - t0, 1),
    }


# ── Clean baseline loader ─────────────────────────────────────────────────────

def load_clean_baseline(model, topology, scenario_slug):
    """Load per-task decision_accuracy from clean pilot run."""
    slug = model.replace(":", "-")
    # Try both pilot_clean and lg_pilot_clean prefixes
    for prefix in ["pilot_clean", f"{scenario_slug.split('_')[0]}_clean"]:
        p = RESULTS_DIR / f"{prefix}_{slug}_{topology}.json"
        if not p.exists():
            p = RESULTS_DIR / f"pilot_clean_{slug}_{topology}.json"
        if p.exists():
            break
    if not p.exists():
        print(f"  [warn] no clean baseline found for {model}/{topology}")
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
    except Exception:
        return {}
    baseline = {}
    for r in data:
        if isinstance(r, dict) and "error" not in r:
            tid = r.get("task_id")
            acc = r.get("scores", {}).get("decision_accuracy", -1)
            if tid and acc >= 0:
                baseline[tid] = acc
    return baseline


# ── Completed-key tracking ────────────────────────────────────────────────────

def make_key(defense_strat, attack_strat, k_d, k_a, d_seed, a_seed, task_id):
    return (defense_strat, attack_strat, k_d, k_a, d_seed, a_seed, task_id)


def load_existing(out_path):
    """Load existing results and build completed-key set for resume."""
    if not out_path.exists():
        return [], set()
    try:
        with open(out_path) as f:
            existing = json.load(f)
    except Exception:
        return [], set()
    completed = set()
    for r in existing:
        if "error" not in r:
            completed.add(make_key(
                r.get("defense_strategy"), r.get("attack_strategy"),
                r.get("k_d"), r.get("k_a"),
                r.get("d_seed", 0), r.get("a_seed", 0),
                r.get("task_id")))
    return existing, completed


# ── Experiment 1 ──────────────────────────────────────────────────────────────

def run_exp1(topology_cfg, scenario, model, ranked, tasks, n_seeds,
             k_d, k_a, results, completed, out_path, model_str, topo_name, scenario_slug):
    n_edges = len(ranked)
    total_new = 0

    # Randomized policies need multiple draws to estimate expected behaviour
    RANDOMIZED_POLICIES = {"random_k", "random_top_r3", "random_top_r4",
                           "mixed_top1_r3", "mixed_top1_r4"}
    for d_strat in DEFENSE_STRATEGIES:
        n_d_seeds = n_seeds if d_strat in RANDOMIZED_POLICIES else 1
        for d_seed in range(n_d_seeds):
            defense_edges = select_defense_edges(ranked, k_d, d_strat, seed=d_seed)
            defense_set = frozenset(tuple(e) for e in defense_edges)

            for a_strat in ATTACK_STRATEGIES:
                # none + adaptive / adaptive_policy reduce to gray_box (no defense to dodge)
                if d_strat == "none" and a_strat in ("adaptive", "adaptive_policy"):
                    continue
                n_a_seeds = n_seeds if a_strat == "naive" else 1
                for a_seed in range(n_a_seeds):
                    attack_edges = select_attack_edges(
                        ranked, k_a, a_strat, defense_set, seed=a_seed,
                        defense_policy=d_strat, k_d=k_d)
                    # Cap attack edges if topology is small
                    if not attack_edges:
                        print(f"  skip: no attack edges for {d_strat}+{a_strat} "
                              f"k_d={k_d} k_a={k_a}")
                        continue

                    for task in tasks:
                        key = make_key(d_strat, a_strat, k_d, k_a, d_seed, a_seed, task["id"])
                        if key in completed:
                            continue

                        t0 = time.time()
                        trial = run_trial(topology_cfg, scenario, model, task,
                                          attack_edges, defense_edges)
                        trial.update({
                            "model": model_str,
                            "topology": topo_name,
                            "scenario": scenario_slug,
                            "mode": "exp1",
                            "defense_strategy": d_strat,
                            "attack_strategy": a_strat,
                            "k_d": k_d,
                            "k_a": k_a,
                            "d_seed": d_seed,
                            "a_seed": a_seed,
                            "defense_edges": [list(e) for e in defense_edges],
                            "attack_edges": [list(e) for e in attack_edges],
                            "task_id": task["id"],
                        })
                        results.append(trial)
                        completed.add(key)
                        total_new += 1

                        acc = trial.get("scores", {}).get("decision_accuracy", "?")
                        elapsed = trial.get("elapsed_s", "?")
                        print(f"  exp1 [{d_strat}+{a_strat}] "
                              f"k_d={k_d} k_a={k_a} d={d_seed} a={a_seed} "
                              f"{task['id']} acc={acc} ({elapsed}s)")

                        with open(out_path, "w") as f:
                            json.dump(results, f, indent=1)

    return total_new


# ── Experiment 2 ──────────────────────────────────────────────────────────────

def run_exp2(topology_cfg, scenario, model, ranked, tasks, n_seeds,
             results, completed, out_path, model_str, topo_name, scenario_slug):
    n_edges = len(ranked)
    total_new = 0

    for k_d, k_a in EXP2_SWEEP:
        if k_d >= n_edges:
            print(f"  skip (k_d={k_d} >= n_edges={n_edges})")
            continue

        for d_strat in EXP2_DEFENSES:
            n_d_seeds = n_seeds if d_strat == "random_k" else 1
            for d_seed in range(n_d_seeds):
                defense_edges = select_defense_edges(ranked, k_d, d_strat, seed=d_seed)
                defense_set = frozenset(tuple(e) for e in defense_edges)

                # Adaptive attacker picks best from undefended edges
                attack_edges = select_attack_edges(
                    ranked, k_a, EXP2_ATTACK, defense_set, seed=0)
                k_a_eff = len(attack_edges)
                if k_a_eff == 0:
                    print(f"  skip (no undefended edges: k_d={k_d} k_a={k_a} {d_strat})")
                    continue

                for task in tasks:
                    key = make_key(d_strat, EXP2_ATTACK, k_d, k_a, d_seed, 0, task["id"])
                    if key in completed:
                        continue

                    trial = run_trial(topology_cfg, scenario, model, task,
                                      attack_edges, defense_edges)
                    trial.update({
                        "model": model_str,
                        "topology": topo_name,
                        "scenario": scenario_slug,
                        "mode": "exp2",
                        "defense_strategy": d_strat,
                        "attack_strategy": EXP2_ATTACK,
                        "k_d": k_d,
                        "k_a": k_a,
                        "k_a_eff": k_a_eff,
                        "d_seed": d_seed,
                        "a_seed": 0,
                        "defense_edges": [list(e) for e in defense_edges],
                        "attack_edges": [list(e) for e in attack_edges],
                        "task_id": task["id"],
                    })
                    results.append(trial)
                    completed.add(key)
                    total_new += 1

                    acc = trial.get("scores", {}).get("decision_accuracy", "?")
                    elapsed = trial.get("elapsed_s", "?")
                    print(f"  exp2 [{d_strat}+adaptive] "
                          f"k_d={k_d} k_a={k_a}(eff={k_a_eff}) d={d_seed} "
                          f"{task['id']} acc={acc} ({elapsed}s)")

                    with open(out_path, "w") as f:
                        json.dump(results, f, indent=1)

    return total_new


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["exp1", "exp2"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO)
    ap.add_argument("--k_d", type=int, default=EXP1_K_D,
                    help="Defender budget (exp1 only)")
    ap.add_argument("--k_a", type=int, default=EXP1_K_A,
                    help="Attacker budget (exp1 only)")
    ap.add_argument("--n_seeds", type=int, default=5,
                    help="Random seeds for non-deterministic strategies")
    ap.add_argument("--output_dir", default=str(RESULTS_DIR))
    ap.add_argument("--fast", action="store_true",
                    help="Drop naïve attack + bottom_k defense (already covered "
                         "by Defense Exp 1); focuses on randomization story.")
    args = ap.parse_args()

    if args.fast:
        # Trim strategy lists to the randomization-story-relevant subset
        global DEFENSE_STRATEGIES, ATTACK_STRATEGIES
        DEFENSE_STRATEGIES = [d for d in DEFENSE_STRATEGIES if d != "bottom_k"]
        ATTACK_STRATEGIES  = [a for a in ATTACK_STRATEGIES  if a != "naive"]

    model_slug = args.model.replace(":", "-")
    scenario_slug = Path(args.scenario).stem

    print(f"=== Defense Experiment ({args.mode}) ===")
    print(f"  model    : {args.model}")
    print(f"  topology : {args.topology}")
    print(f"  scenario : {args.scenario}")
    print(f"  n_seeds  : {args.n_seeds}")
    if args.mode == "exp1":
        print(f"  k_d={args.k_d}  k_a={args.k_a}")

    topology_cfg = load_topology(f"config/topologies/{args.topology}.yaml")
    with open(args.scenario) as f:
        scenario = yaml.safe_load(f)
    tasks = scenario["tasks"]

    # Compute saliency ranking
    dummy = MASRunner(topology_cfg, scenario, model=args.model)
    ranked = compute_struct_saliency(dummy.graph)
    n_edges = len(ranked)
    print(f"  edges    : {n_edges}")
    print(f"  top-3 MESA: {[list(e) for e, _ in ranked[:3]]}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"defense_{args.mode}_{scenario_slug}_{model_slug}_{args.topology}.json"

    results, completed = load_existing(out_path)
    print(f"  resume   : {len(results)} existing, {len(completed)} completed keys")

    if args.mode == "exp1":
        n_new = run_exp1(
            topology_cfg, scenario, args.model, ranked, tasks,
            args.n_seeds, args.k_d, args.k_a,
            results, completed, out_path,
            args.model, args.topology, scenario_slug)
    else:
        n_new = run_exp2(
            topology_cfg, scenario, args.model, ranked, tasks,
            args.n_seeds,
            results, completed, out_path,
            args.model, args.topology, scenario_slug)

    print(f"\nDone. New trials: {n_new}, total: {len(results)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
