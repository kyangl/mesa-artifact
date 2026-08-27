"""Run the matched no-withholding control for enforcement trials.

Controls use the same task, model, payload, and executor with an empty monitored
set. They measure rerun variance and are reused across policies, budgets, seeds,
and cross-fit directions.
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.build_feature_matrix import load_validity            # noqa: E402
from analysis.enforcement_executor import MASRerunExecutor, RerunCache  # noqa: E402
from analysis.replay_pareto import load_cached_scores              # noqa: E402
from runners.run_cs_enforcement import (SCENARIO, THRESHOLD,       # noqa: E402
                                        build_detector, make_rerun)
from runners.run_cs_pareto import load_trials, task_evidence       # noqa: E402

DEFAULT_CACHE_DIR = REPO / "results" / "enforcement" / "customer_service" / "6039372"
MODELS = ("gemma4:e4b", "qwen3.5:9b")
TOPOLOGIES = ("sequential", "centralized", "decentralized", "hierarchical",
              "hybrid")


def control_keys(cache_dir=DEFAULT_CACHE_DIR, model=None, topology=None):
    """Distinct controls implied by an executed intervention sweep.

    Derived from the committed rerun caches rather than recomputed from the
    policy grid, so the control set is exactly the trials that were actually
    re-run -- not a superset the grid might have visited.
    """
    out = {}
    for p in sorted(glob.glob(str(Path(cache_dir) / "rerun_cache_*.json"))):
        rows = (json.loads(Path(p).read_text()).get("rows") or {})
        for r in (rows.values() if isinstance(rows, dict) else rows):
            pr = r.get("provenance") or {}
            ae = pr.get("attacked_edge")
            if ae is None:
                continue
            if model and pr.get("model") != model:
                continue
            if topology and pr.get("topology") != topology:
                continue
            key = (pr.get("scenario"), pr.get("model"), pr.get("topology"),
                   pr.get("task_id"), tuple(ae),
                   pr.get("attack_payload_sha16"), pr.get("threshold"))
            out.setdefault(key, {
                "scenario": key[0], "model": key[1], "topology": key[2],
                "task_id": key[3], "attacked_edge": list(key[4]),
                "attack_payload_sha16": key[5], "threshold": key[6],
                # What the ORIGINAL (undefended) run did. The control asks
                # whether an unmodified rerun reproduces it.
                "original_success": r.get("original_success"),
                "clean_correct": r.get("clean_correct"),
            })
    return out


def run_config(model, topology, scenario=SCENARIO, cache_dir=DEFAULT_CACHE_DIR,
               out_path=None, verbose=False):
    wanted = control_keys(cache_dir, model, topology)
    if not wanted:
        return {"model": model, "topology": topology,
                "status": "no controls implied by the intervention sweep",
                "cells": []}

    cached, det_id = load_cached_scores(scenario)
    det = cached.get((model, topology))
    if det is None:
        return {"model": model, "topology": topology,
                "status": "no cached detector scores", "cells": []}

    attacked, _clean = load_trials(scenario, model, topology)
    attacked = [t for t in attacked if t.messages]
    by_trial = {(t.task_id, tuple(t.attacked_edge) if t.attacked_edge else None): t
                for t in attacked}

    evidence = task_evidence(scenario)
    detector = build_detector(False, attacked)
    cache = RerunCache(Path(out_path).with_name(
        "control_cache_%s_%s.json" % (model.replace(":", "-"), topology))
        if out_path else None)
    ex = MASRerunExecutor(
        scenario=scenario, model=model, topology=topology, detector=detector,
        evidence=evidence, cache=cache, store_messages=False)

    # Call the executor directly so an empty monitor set still triggers a rerun.
    def rerun(trial, monitored, threshold):
        return ex(trial, None, monitored=monitored, threshold=threshold,
                  flags=[], det=det)

    rows, t0 = [], time.time()
    for key, meta in sorted(wanted.items(), key=lambda kv: str(kv[0])):
        trial = by_trial.get((meta["task_id"], tuple(meta["attacked_edge"])))
        if trial is None:
            rows.append(dict(meta, status="trial not found in loaded outcomes",
                             control_success=None))
            continue
        i0 = len(ex.records)
        # EMPTY monitored set: nothing can be flagged, nothing withheld.
        new_t = rerun(trial, [], THRESHOLD)
        rec = ex.records[i0:]
        rec = rec[-1] if rec else {}
        n_int = sum(r.get("n_interventions", 0) for r in ex.records[i0:])
        control_bad = None if (new_t is None or new_t.success is None) \
            else bool(new_t.success)
        rows.append(dict(
            meta,
            control_success=control_bad,
            reproduced=(None if control_bad is None
                        else control_bad == bool(meta["original_success"])),
            n_interventions=n_int,
            unresolved=control_bad is None,
            latency_s=rec.get("latency_s"),
            failure_reason=rec.get("failure_reason"),
            cache_hit=bool(rec.get("cache_hit")),
        ))
        if verbose:
            print("  %-10s %-22s orig=%s control=%s"
                  % (meta["task_id"], str(meta["attacked_edge"]),
                     meta["original_success"], control_bad))

    resolved = [r for r in rows if r.get("control_success") is not None]
    reproduced = [r for r in resolved if r["reproduced"]]
    bad_int = [r for r in rows if r.get("n_interventions")]
    # A control that did not execute is not a control. Without this the arm
    # returns reproducibility 1.000 -- true by construction, measured from
    # nothing -- and looks like a clean result.
    executed = [r for r in rows if r.get("latency_s") or r.get("cache_hit")]
    no_execution = len(rows) - len(executed)
    return {
        "scenario": scenario, "model": model, "topology": topology,
        "arm": "no_intervention_control",
        "threshold": THRESHOLD,
        "detector": det_id,
        "note": ("Matched control: same model, task, cached attack payload, "
                 "initial state and inference configuration as the "
                 "intervention arm, executed through the same "
                 "MASRerunExecutor with an EMPTY monitored set so nothing can "
                 "be withheld. No execution seed exists in BaseAgent, so the "
                 "arms are matched on configuration rather than seed; the "
                 "residual variance is what this arm measures."),
        "n_controls": len(rows),
        "n_resolved": len(resolved),
        "n_unresolved": len(rows) - len(resolved),
        "n_reproduced": len(reproduced),
        "control_reproducibility": (len(reproduced) / len(resolved)
                                    if resolved else None),
        # A control that withheld anything is a bug, not a datum.
        "n_with_interventions": len(bad_int),
        # As is a control that never ran.
        "n_executions": ex.n_executions,
        "n_no_execution": no_execution,
        "elapsed_s": time.time() - t0,
        "complete": True,
        "cells": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=SCENARIO)
    ap.add_argument("--model", default=None)
    ap.add_argument("--topology", default=None)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--seconds-per-rerun", type=float, default=105.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.plan_only:
        keys = control_keys(args.cache_dir, args.model, args.topology)
        per = {}
        for k in keys:
            per.setdefault((k[1], k[2]), 0)
            per[(k[1], k[2])] += 1
        for (m, t), n in sorted(per.items()):
            print("  %-12s %-14s %4d controls  %5.2f GPU-h"
                  % (m, t, n, n * args.seconds_per_rerun / 3600.0))
        print("TOTAL %d controls  %.2f GPU-h at %.0f s each"
              % (len(keys), len(keys) * args.seconds_per_rerun / 3600.0,
                 args.seconds_per_rerun))
        return 0

    if not args.model or not args.topology:
        raise SystemExit("--model and --topology are required (or --plan-only)")

    out = Path(args.out or (REPO / ("cs_control_%s_%s.json"
                                    % (args.model.replace(":", "-"),
                                       args.topology))))
    payload = run_config(args.model, args.topology, args.scenario,
                         args.cache_dir, str(out), args.verbose)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print("%s/%s: %d controls, %d resolved, reproducibility %s"
          % (args.model, args.topology, payload.get("n_controls", 0),
             payload.get("n_resolved", 0),
             ("%.3f" % payload["control_reproducibility"])
             if payload.get("control_reproducibility") is not None else "n/a"))
    if payload.get("n_with_interventions"):
        print("ERROR: %d controls withheld a message; the control arm is not "
              "a control" % payload["n_with_interventions"])
        return 1
    if payload.get("n_no_execution"):
        print("ERROR: %d of %d controls never executed the workflow. A "
              "reproducibility computed from these is 1.000 by construction "
              "and measures nothing."
              % (payload["n_no_execution"], payload.get("n_controls", 0)))
        return 1
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
