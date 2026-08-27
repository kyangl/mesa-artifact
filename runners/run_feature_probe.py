"""Generate receiver-response-sensitivity probes.

Each consumed message occurrence is replaced in place by fixed ablation text,
and the receiver is called once. Similarity is computed offline; missing probes
are never recorded as zero.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.topology.builder import load_topology
from src.agents.mas_runner import MASRunner
from src.agents.base_agent import Agent
from src.logging.transcripts import ReceiverCall, RunContext, write_jsonl
from runners.run_dynamic_saliency import ABLATION_MESSAGE
from runners.run_preflight import _link_receiver_call, ollama_version
from src.saliency.consumption import (
    build_consumption_map, comparison_context,
)


def probe_trial(runner, task, scenario_name, topology, model, run_context,
                placeholder=ABLATION_MESSAGE):
    """One clean trial, then probe EVERY consumed occurrence of every edge.

    Aggregating protocols fold several messages into one receiver call and
    multi-round protocols deliver an edge more than once, so occurrences --
    not edges -- are the unit here. F2 averages per (edge, task) afterwards.
    """
    runner.reset()
    t0 = time.time()
    runner.run(task)
    clean_elapsed = time.time() - t0

    edge_log = runner.get_edge_log()
    occurrences = build_consumption_map(edge_log, runner.agents)

    records = []
    for occ in occurrences:
        edge = (occ["src"], occ["dst"])
        base = {"scenario": scenario_name, "topology": topology,
                "model": model, "task_id": task["id"],
                "edge_src": edge[0], "edge_dst": edge[1],
                "step_idx": occ["step_idx"],
                "call_index": occ["call_index"]}

        # F1: the message paired with everything else the receiver had.
        records.append(dict(base, kind="f1_context",
                            content=occ["content"],
                            prior_context=comparison_context(
                                occurrences, occ, runner.agents)))

        if not occ["content"]:
            continue
        if occ["call_index"] is None:
            records.append(dict(base, kind="f2_probe", probed=False,
                                reason="message never consumed by a receiver call",
                                clean_output=None, probe_output=None))
            continue

        agent = runner.agents[occ["dst"]]
        clean_call = agent.call_log[occ["call_index"]]
        probe_messages = copy.deepcopy(clean_call.messages)
        last = probe_messages[-1]
        if occ["content"] not in last.get("content", ""):
            records.append(dict(base, kind="f2_probe", probed=False,
                                reason="content not located in consuming call",
                                clean_output=None, probe_output=None))
            continue
        last["content"] = last["content"].replace(occ["content"], placeholder)
        t1 = time.time()
        probe_output = agent._call_ollama(probe_messages)
        records.append(dict(base, kind="f2_probe", probed=True, reason=None,
                            placeholder=placeholder,
                            clean_output=clean_call.response,
                            probe_output=probe_output,
                            probe_latency_s=time.time() - t1,
                            clean_trial_s=clean_elapsed,
                            run_context=run_context.to_dict()))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--topology", default="sequential")
    ap.add_argument("--scenario",
                    default="config/scenarios/customer_service.yaml")
    ap.add_argument("--max-tasks", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    topology_path = str(REPO / "config" / "topologies"
                        / ("%s.yaml" % args.topology))
    topo = load_topology(topology_path)
    with open(args.scenario) as fh:
        scenario = yaml.safe_load(fh)
    scenario_name = scenario.get("name", "unknown")
    tasks = scenario["tasks"][:args.max_tasks]

    ctx = RunContext.capture(model=args.model,
                             ollama_version=ollama_version(),
                             quantization="unknown",
                             config_paths=(args.scenario, topology_path))
    runner = MASRunner(topo, scenario, model=args.model)

    out = args.out or ("results/feature_probe_%s_%s_%s.jsonl"
                       % (scenario_name,
                          args.model.replace(":", "-"), args.topology))
    all_records = []
    for task in tasks:
        recs = probe_trial(runner, task, scenario_name, args.topology,
                           args.model, ctx)
        all_records.extend(recs)
        nf2 = [r for r in recs if r.get("kind") == "f2_probe"]
        done = sum(1 for r in nf2 if r.get("probed"))
        print("  %s: probed %d/%d edges, %d F1 records"
              % (task["id"], done, len(nf2), len(recs) - len(nf2)))

    write_jsonl(out, all_records)
    f2 = [r for r in all_records if r.get("kind") == "f2_probe"]
    f1 = [r for r in all_records if r.get("kind") == "f1_context"]
    probed = sum(1 for r in f2 if r["probed"])
    print("wrote %d records -> %s" % (len(all_records), out))
    print("  F1 context records: %d" % len(f1))
    print("  F2 probes: %d probed, %d skipped" % (probed, len(f2) - probed))
    if probed < len(f2):
        print("NOTE: skipped edges have NO RS value. They must stay missing, "
              "never zero-filled.")


if __name__ == "__main__":
    main()
