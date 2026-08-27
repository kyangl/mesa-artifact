"""Measure edge ablation and perturbation accuracy deltas.

Ablation removes the signal; perturbation replaces it with uninformative text.
Both are compared with clean baselines.
"""

import argparse
import json
import time
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.topology.builder import load_topology, build_graph
from src.agents.mas_runner import MASRunner
from src.evaluation.metrics import evaluate_task
from src.attacks.perturbation import dynamic_perturbation_attack, perturbation_attack
from src.attacks.perturbation_cache import PerturbationCache


TOPOLOGIES = ["centralized", "sequential", "hierarchical",
              "decentralized", "mesh", "hybrid"]
MODELS = ["llama3.1:8b", "qwen3.5:9b", "qwen3.5:27b", "gemma4:e4b", "gemma4:26b"]
RESULTS_DIR = Path("results")
DEFAULT_SCENARIO = "config/scenarios/customer_service.yaml"

# Fixed information-free ablation message.
ABLATION_MESSAGE = (
    "[SYSTEM: Communication channel unavailable. "
    "No response received from this agent on this channel.]"
)


def load_scenario(path: str = DEFAULT_SCENARIO) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_edges(topo_name: str) -> list:
    config = load_topology(f"config/topologies/{topo_name}.yaml")
    G = build_graph(config)
    return list(G.edges())


def model_slug(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def save_results(results: list, filename: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def load_results(filename: str) -> list:
    path = RESULTS_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def load_all_results(prefix: str) -> list:
    results = []
    for path in sorted(RESULTS_DIR.glob(f"{prefix}_*.json")):
        with open(path) as f:
            results.extend(json.load(f))
    return results


try:
    from src.agents.langgraph_runner import LangGraphMASRunner
    _HAS_LANGGRAPH = True
except Exception:
    _HAS_LANGGRAPH = False


def _make_runner(runner_name, config, scenario, model):
    if runner_name == "langgraph":
        if not _HAS_LANGGRAPH:
            raise RuntimeError("LangGraph not installed; pip install langgraph")
        return LangGraphMASRunner(config, scenario, model=model)
    return MASRunner(config, scenario, model=model)


def run_single_intervention(topo_name: str, task: dict, scenario: dict,
                            model: str, intervention_edge: tuple,
                            mode: str, runner_name: str = "mas", perturbation_cache=None) -> dict:
    """Run one trial with ablation or perturbation on a specific edge."""
    config = load_topology(f"config/topologies/{topo_name}.yaml")
    runner = _make_runner(runner_name, config, scenario, model=model)

    # Look up sender role for perturbation
    sender_id = intervention_edge[0]
    sender_role = None
    for agent_cfg in config["agents"]:
        if agent_cfg["id"] == sender_id:
            sender_role = agent_cfg["role"]
            break

    if mode == "ablation":
        # Replace every message on this edge with a silent "no signal" message
        attack_fn = lambda msg: ABLATION_MESSAGE
    elif mode == "perturbation":
        # One cached perturbation per (task, sender role), reused across every
        # edge and rerun. Generating fresh noise per call would compare each
        # edge against a different random draw instead of against each other.
        if perturbation_cache is None:
            raise ValueError("perturbation mode requires a PerturbationCache")
        _text = perturbation_cache.get(task["id"], sender_role)
        attack_fn = lambda msg, _t=_text: _t
    else:
        raise ValueError(f"Unknown mode: {mode}")

    runner.set_attack(intervention_edge[0], intervention_edge[1], attack_fn)

    start = time.time()
    try:
        result = runner.run(task)
        elapsed = time.time() - start
        resolution = result.get("final_resolution", "")
    except Exception as e:
        elapsed = time.time() - start
        return {
            "topology": topo_name,
            "task_id": task["id"],
            "model": model,
            "phase": mode,
            "intervention_edge": list(intervention_edge),
            "error": str(e),
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

    scenario_name = scenario.get("name", "customer_service")
    scores = evaluate_task(resolution, task, task["description"],
                           scenario_name=scenario_name, model=model)

    return {
        "topology": topo_name,
        "task_id": task["id"],
        "model": model,
        "scenario": scenario_name,
        "phase": mode,
        "intervention_edge": list(intervention_edge),
        "final_resolution": resolution,
        "scores": scores,
        "n_edges_logged": len(runner.get_edge_log()),
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }


def run_intervention_phase(mode: str, models=MODELS, topologies=TOPOLOGIES,
                           scenario_path: str = DEFAULT_SCENARIO,
                           runner_name: str = "mas", max_tasks=None):
    """Run ablation or perturbation on all edges × tasks × models."""
    scenario = load_scenario(scenario_path)
    tasks = scenario["tasks"]
    if max_tasks:
        tasks = tasks[:max_tasks]

    # Filename prefix encodes the scenario name
    scenario_name = scenario.get("name", "cs")

    total = 0
    for topo in topologies:
        total += len(models) * len(get_edges(topo)) * len(tasks)

    count = 0
    for model in models:
        slug = model_slug(model)

        print(f"\n{'='*60}")
        print(f"MODE: {mode.upper()} | MODEL: {model}")
        print(f"{'='*60}")

        for topo in topologies:
            # One file per (mode, model, topology) — safe for parallel cluster jobs
            # Namespace LangGraph outputs to avoid colliding with MASRunner files.
            ns_prefix = "lg_dynamic" if runner_name == "langgraph" else "dynamic"
            filename = f"{ns_prefix}_{mode}_{scenario_name}_{slug}_{topo}.json"
            topo_results = load_results(filename)
            # One cache per (scenario, model, topology): every edge and every
            # rerun of the same (task, sender role) reuses identical noise
            # text, so perturbation deltas compare edges rather than draws.
            pcache = PerturbationCache(
                path=RESULTS_DIR / ("perturbation_cache_%s_%s_%s.json"
                                    % (scenario_name, model_slug(model), topo)),
                generator=(lambda msg, role, _m=model:
                           dynamic_perturbation_attack(msg, sender_role=role,
                                                       model=_m)),
                scenario=scenario_name, model=model, topology=topo,
            ) if mode == "perturbation" else None

            done = set()
            for r in topo_results:
                if "ERROR" in str(r.get("final_resolution", "")) or "error" in r:
                    continue
                edge_key = tuple(r["intervention_edge"]) if r.get("intervention_edge") else None
                done.add((r["task_id"], edge_key))

            edges = get_edges(topo)
            print(f"\n--- {topo}: {len(edges)} directed edges "
                  f"(file: {filename}, existing: {len(done)}) ---")

            for edge in edges:
                for task in tasks:
                    count += 1
                    edge_str = f"{edge[0]}->{edge[1]}"

                    if (task["id"], tuple(edge)) in done:
                        print(f"[{count}/{total}] {mode} | {model} | {topo} | "
                              f"{edge_str} | {task['id']} ... SKIP (exists)")
                        continue

                    print(f"[{count}/{total}] {mode} | {model} | {topo} | "
                          f"{edge_str} | {task['id']}",
                          end=" ... ", flush=True)

                    r = run_single_intervention(topo, task, scenario, model,
                                               tuple(edge), mode,
                                               runner_name=runner_name,
                                               perturbation_cache=pcache)
                    topo_results.append(r)

                    if "error" in r:
                        print(f"ERROR: {r['error'][:50]}")
                    else:
                        s = r["scores"]
                        print(f"dec={s.get('decision_accuracy', '?')} "
                              f"({r['elapsed_seconds']:.1f}s)")

                    save_results(topo_results, filename)

    print(f"\n{mode.capitalize()} phase done.")
    return load_all_results(f"dynamic_{mode}_{scenario_name}")


def compute_deltas(mode: str, scenario_name: str = "customer_service") -> dict:
    """Compute per-edge accuracy deltas vs clean baseline.

    Returns:
        {model: {(topology, edge_tuple): {"clean_acc", "intervened_acc", "delta"}}}
    """
    clean_all = load_all_results("pilot_clean")
    intervened_all = load_all_results(f"dynamic_{mode}_{scenario_name}")

    deltas = {}
    for model in MODELS:
        deltas[model] = {}

        clean_m = [r for r in clean_all
                   if r.get("model") == model and "error" not in r]

        # Per-topology baseline (average across tasks)
        topo_clean = defaultdict(list)
        for r in clean_m:
            sc = r.get("scores", {})
            dec = sc.get("decision_accuracy", -1)
            if dec >= 0:
                topo_clean[r["topology"]].append(dec)

        intervened_m = [r for r in intervened_all
                        if r.get("model") == model and "error" not in r]

        # Per-edge intervened accuracy
        edge_intervened = defaultdict(list)
        for r in intervened_m:
            edge_key = tuple(r["intervention_edge"]) if r.get("intervention_edge") else None
            sc = r.get("scores", {})
            dec = sc.get("decision_accuracy", -1)
            if edge_key and dec >= 0:
                edge_intervened[(r["topology"], edge_key)].append(dec)

        for (topo, edge), acc_list in edge_intervened.items():
            clean_acc = (sum(topo_clean[topo]) / len(topo_clean[topo])
                         if topo_clean[topo] else None)
            intervened_acc = sum(acc_list) / len(acc_list)
            delta = (clean_acc - intervened_acc) if clean_acc is not None else None
            deltas[model][(topo, edge)] = {
                "clean_acc": clean_acc,
                "intervened_acc": intervened_acc,
                "delta": delta,
                "n_trials": len(acc_list),
            }

    return deltas


def print_report(scenario_name: str = "customer_service"):
    """Print delta tables for all computed modes."""
    for mode in ["ablation", "perturbation"]:
        results = load_all_results(f"dynamic_{mode}_{scenario_name}")
        if not results:
            print(f"\nNo {mode} results found.")
            continue

        print(f"\n{'='*70}")
        print(f"DYNAMIC SALIENCY — {mode.upper()} RESULTS ({scenario_name})")
        print(f"{'='*70}")

        deltas = compute_deltas(mode, scenario_name)

        for model, edge_deltas in deltas.items():
            print(f"\n--- {model} ---")
            print(f"{'Topology':<16} {'Edge':<35} {'Clean':>7} {'Interv':>7} {'Delta':>7}")
            print("-" * 72)

            # Sort by delta descending (most impactful edges first)
            for (topo, edge), info in sorted(
                    edge_deltas.items(),
                    key=lambda x: (x[1]["delta"] or -99), reverse=True):
                clean = info["clean_acc"]
                interv = info["intervened_acc"]
                delta = info["delta"]
                edge_str = f"{edge[0]}->{edge[1]}"
                print(f"{topo:<16} {edge_str:<35} "
                      f"{(clean or 0):.3f}  {interv:.3f}  "
                      f"{(delta or 0):+.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tasks", type=int, default=None,
                    help="Use only the first N scenario tasks (technical preflight).")
    parser.add_argument("--mode", choices=["ablation", "perturbation", "both", "report"],
                        default="report")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--topologies", nargs="+", default=TOPOLOGIES)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO,
                        help="Path to scenario YAML")
    parser.add_argument("--runner", choices=["mas", "langgraph"], default="mas",
                        help="Orchestration runner (default: built-in MASRunner)")
    args = parser.parse_args()

    scenario_name = yaml.safe_load(open(args.scenario)).get("name", "cs")

    if args.mode == "ablation":
        run_intervention_phase("ablation", models=args.models, max_tasks=args.max_tasks,
                               topologies=args.topologies,
                               scenario_path=args.scenario,
                               runner_name=args.runner)
        print_report(scenario_name)
    elif args.mode == "perturbation":
        run_intervention_phase("perturbation", models=args.models, max_tasks=args.max_tasks,
                               topologies=args.topologies,
                               scenario_path=args.scenario,
                               runner_name=args.runner)
        print_report(scenario_name)
    elif args.mode == "both":
        run_intervention_phase("ablation", models=args.models, max_tasks=args.max_tasks,
                               topologies=args.topologies,
                               scenario_path=args.scenario,
                               runner_name=args.runner)
        run_intervention_phase("perturbation", models=args.models, max_tasks=args.max_tasks,
                               topologies=args.topologies,
                               scenario_path=args.scenario,
                               runner_name=args.runner)
        print_report(scenario_name)
    elif args.mode == "report":
        print_report(scenario_name)
