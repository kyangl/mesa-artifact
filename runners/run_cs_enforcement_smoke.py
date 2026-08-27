"""Smoke-test enforced reruns, intervention, regeneration, and caching.

``--dry-run`` uses deterministic stand-ins and tests plumbing only; the default
uses the real model and detector on a small customer-service subset.
"""

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.enforcement import NEUTRAL_PLACEHOLDER, content_sha16, make_rerun
from analysis.enforcement_executor import MASRerunExecutor, RerunCache
from analysis.pareto import DetectorScores, evaluate_cell
from runners.run_cs_pareto import load_trials, task_evidence

SMOKE_TOPOLOGIES = ("sequential", "hierarchical")
SMOKE_MODELS = ("gemma4:e4b", "qwen3.5:9b")


def _fake_llm(self, messages):
    digest = hashlib.sha256(
        json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]
    return "reply(%s):%s" % (self.agent_id, digest)


class PayloadOracleDetector:
    """Dry-run stand-in: fires on exactly the text the attacker substituted.

    Content-only, like any real detector -- it holds the hashes of the stored
    payloads and nothing else. Its purpose is to make the gate fire
    deterministically so the PLUMBING can be exercised without a GPU. It is
    trivially perfect and must never be used for a number.
    """

    detector_id = "dry_run_payload_oracle"

    def __init__(self, payload_hashes, threshold=0.5):
        self.hashes = set(payload_hashes)
        self.threshold = threshold

    def score(self, message, edge, local_context=None, evidence=None):
        from src.defenses.base import DetectorVerdict
        conf = 1.0 if content_sha16(message) in self.hashes else 0.0
        return DetectorVerdict(flag=conf >= self.threshold, confidence=conf,
                               latency_s=0.0, token_cost=len((message or "").split()))


def _detector(dry_run, threshold, trials=()):
    if dry_run:
        hashes = {content_sha16(m.get("attacked_content"))
                  for t in trials for m in t.messages if m.get("was_attacked")}
        return PayloadOracleDetector(hashes, threshold=threshold)
    from src.defenses.real_detectors import NLIEvidenceDetector
    return NLIEvidenceDetector(threshold=threshold)


def _monitored_for(trials, k):
    """The k injection edges carrying the most sampled attacks.

    A smoke-test allocation, chosen so the gate has something to fire on. It is
    NOT a policy and nothing here may be reported as one: it uses the outcomes
    of the very trials it is evaluated on.
    """
    mass = collections.Counter()
    for t in trials:
        if t.attacked_edge:
            mass[tuple(t.attacked_edge)] += 1
    return [e for e, _n in mass.most_common(k)]


def _schedule(msgs):
    return [(m.get("source"), m.get("target")) for m in msgs]


def check_records(records, baselines):
    """The invariants a real rerun must satisfy. Returns (checks, notes)."""
    by_task = {(t.task_id, tuple(t.attacked_edge or ())): t for t in baselines}
    executed = [r for r in records if not r.get("cache_hit")]
    resolved = [r for r in records if r["defended_success"] is not None]
    intervened = [r for r in records if r["n_interventions"] > 0]

    schedule_ok, regenerated, with_downstream, no_downstream = True, 0, 0, 0
    notes = []
    for r in resolved:
        prov = r.get("provenance") or {}
        base = by_task.get((prov.get("task_id"),
                            tuple(prov.get("attacked_edge") or ())))
        defended = r.get("defended_messages") or []
        if base is None or not defended or r["n_interventions"] == 0:
            continue
        first = min(i["msg_index"] for i in r["interventions"]
                    if i["intervention_applied"])

        # Up to and including the intervention the two runs must agree on WHO
        # talked to WHOM: the defense replaces content, it does not rewire the
        # graph or reorder the schedule.
        if _schedule(defended[:first + 1]) != _schedule(base.messages[:first + 1]):
            schedule_ok = False
            notes.append("schedule diverged before the intervention for %s"
                         % (prov.get("task_id"),))
            continue

        # AFTER the intervention the schedule may legitimately change -- a CEO
        # that saw a withheld report may ask a follow-up it did not ask before.
        # That is recomputation, not a defect, so a changed schedule counts as
        # evidence rather than a failure.
        tail_def = [(m.get("source"), m.get("target"), m.get("content"))
                    for m in defended[first + 1:]]
        tail_base = [(m.get("source"), m.get("target"), m.get("content"))
                     for m in base.messages[first + 1:]]
        if not tail_def and not tail_base:
            no_downstream += 1          # attacked edge was the last delivery
            continue
        with_downstream += 1
        if tail_def != tail_base:
            regenerated += 1
        else:
            notes.append("downstream identical to the stored transcript for "
                         "%s -- the rerun replayed instead of regenerating"
                         % (prov.get("task_id"),))

    checks = {
        "executed": len(executed) > 0,
        "intervened": len(intervened) > 0,
        "resolved": len(resolved) == len(records),
        "schedule_preserved": schedule_ok,
        # Vacuously true only when every intervention was on the last delivery
        # of its run, which is reported in the counts.
        "downstream_regenerated": regenerated == with_downstream,
    }
    return checks, notes, {"n_executed": len(executed),
                           "n_resolved": len(resolved),
                           "n_intervened": len(intervened),
                           "n_with_downstream": with_downstream,
                           "n_no_downstream": no_downstream,
                           "n_regenerated": regenerated}


def run_config(scenario, model, topology, args, evidence):
    attacked, clean = load_trials(scenario, model, topology)
    attacked = [t for t in attacked if t.messages]
    pool = [t for t in attacked if t.success]
    if not pool:
        return {"model": model, "topology": topology,
                "status": "no baseline-successful attacked trials with "
                          "transcripts"}

    detector = _detector(args.dry_run, args.threshold, pool)
    monitored = _monitored_for(pool, args.k)
    watched = set(monitored)

    # Score the offline gate trial by trial and keep the first `max_trials`
    # that actually flag. A mechanism smoke test needs reruns to happen; a
    # sample where nothing fires tests nothing. This is selection on the
    # detector, which is exactly why the output is not a rate.
    det = DetectorScores()
    successes, n_scanned = [], 0
    for t in pool:
        if len(successes) >= args.max_trials:
            break
        n_scanned += 1
        ev = evidence.get(t.task_id, "")
        fired = False
        for i, m in enumerate(t.messages):
            if (m.get("source"), m.get("target")) not in watched:
                continue                    # offline gate scores what is watched
            v = detector.score(m.get("content") or "",
                               (m.get("source"), m.get("target")), evidence=ev)
            key = (t.task_id, t.attacked_edge, i)
            det.scores[key] = v.confidence
            det.latency_s[key] = v.latency_s
            det.tokens[key] = v.token_cost
            fired = fired or v.confidence >= args.threshold
        if fired:
            successes.append(t)
    if not successes:
        return {"model": model, "topology": topology,
                "status": ("nothing flagged on %s at threshold %.2f in %d "
                           "baseline successes" % (monitored, args.threshold,
                                                   n_scanned)),
                "monitored": [list(e) for e in monitored]}

    # In memory by default: a smoke test that read a previous run's cache
    # would report "ok" without executing anything, which is the one thing it
    # exists to catch. --cache-path opts into persistence.
    cache = RerunCache(Path(args.cache_path) if args.cache_path else None)
    # Dry runs replace the judge because only plumbing is under test.
    ex = MASRerunExecutor(scenario=scenario, model=model, topology=topology,
                          detector=detector, evidence=evidence, cache=cache,
                          evaluate_fn=((lambda res, task, sname, m:
                                        {"decision_accuracy": 1,
                                         "dry_run": True})
                                       if args.dry_run else None))
    rerun = make_rerun(ex, det=det)
    cell = evaluate_cell(successes, [], monitored, args.threshold, det,
                         rerun=rerun)
    n_first = ex.n_executions
    # Same cell twice: the second pass must be served entirely from the cache.
    evaluate_cell(successes, [], monitored, args.threshold, det, rerun=rerun)
    cache_reused = (ex.n_executions == n_first)
    if args.cache_path:
        cache.save()

    checks, notes, counts = check_records(ex.records, successes)
    checks["cache_reused"] = cache_reused
    counts["n_executions"] = ex.n_executions
    return {"model": model, "topology": topology, "status": "ran",
            "detector": getattr(detector, "detector_id", "unknown"),
            "monitored": [list(e) for e in monitored],
            "threshold": args.threshold,
            "n_baseline_successes": len(successes),
            "n_scanned_for_flags": n_scanned,
            "outcomes_meaningful": not args.dry_run,
            "cell": {k: cell[k] for k in
                     ("security_prevented", "security_baseline_successes",
                      "security", "n_rerun", "n_reused", "n_unresolved_rerun")},
            "summary": ex.summary(), "checks": checks, "notes": notes,
            "counts": counts,
            "records": [{k: v for k, v in r.items()
                         if k != "defended_messages"} for r in ex.records]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="customer_service")
    ap.add_argument("--models", nargs="+", default=list(SMOKE_MODELS))
    ap.add_argument("--topologies", nargs="+", default=list(SMOKE_TOPOLOGIES))
    ap.add_argument("--threshold", type=float, default=0.26)
    ap.add_argument("--k", type=int, default=1,
                    help="monitored edges (smoke allocation, not a policy)")
    ap.add_argument("--max-trials", type=int, default=3,
                    help="baseline-successful attacks per configuration")
    ap.add_argument("--cache-path", default=None,
                    help="persist the rerun cache (default: in memory, so the "
                         "smoke test always executes)")
    ap.add_argument("--dry-run", action="store_true",
                    help="stub model and stub detector: plumbing only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or (REPO / "results" / "enforcement"
                            / ("cs_enforcement_smoke_%s.json" % args.scenario)))
    evidence = task_evidence(args.scenario)

    if args.dry_run:
        print("DRY RUN: stub model, stub detector. Plumbing only -- this says "
              "nothing about model behaviour.")

    results = []
    ctx = None
    if args.dry_run:
        from unittest.mock import patch
        from src.agents.base_agent import Agent
        ctx = patch.object(Agent, "_call_ollama", _fake_llm)
        ctx.__enter__()
    try:
        for model in args.models:
            for topology in args.topologies:
                r = run_config(args.scenario, model, topology, args, evidence)
                results.append(r)
                if r["status"] != "ran":
                    print("  %-12s %-14s SKIP: %s"
                          % (model, topology, r["status"]))
                    continue
                print("  %-12s %-14s executed=%d resolved=%d intervened=%d "
                      "downstream=%d regenerated=%d prevented=%s"
                      % (model, topology, r["counts"]["n_executed"],
                         r["counts"]["n_resolved"], r["counts"]["n_intervened"],
                         r["counts"]["n_with_downstream"], r["counts"]["n_regenerated"],
                         r["cell"]["security_prevented"]))
                for name, ok in r["checks"].items():
                    print("      %-24s %s" % (name, "ok" if ok else "FAIL"))
                for n in r["notes"]:
                    print("      note: %s" % n)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

    ran = [r for r in results if r["status"] == "ran"]
    passed = bool(ran) and all(all(r["checks"].values()) for r in ran)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"scenario": args.scenario, "dry_run": args.dry_run,
         "placeholder": NEUTRAL_PLACEHOLDER,
         "note": ("Mechanism smoke test for the enforced rerun. The monitored "
                  "set is chosen from the sampled outcomes to make the gate "
                  "fire; it is not an allocation policy and nothing here is a "
                  "security result."),
         "smoke_pass": passed, "configs": results}, indent=2, default=str))
    print("\nwrote %s" % out)
    print("SMOKE %s" % ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
