"""Compute trial-paired, control-adjusted prevention.

The estimand is ``1[control succeeds] - 1[intervention succeeds]`` for exact
trial pairs. Controls are reused across policies because they withhold nothing;
interventions also key on the monitored edge set. Negative effects remain in
the estimate.
"""

import argparse
import collections
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.replay_pareto import (fold_of_task, POLICY_MESA_CF,  # noqa: E402
                                    POLICY_RANDOM)
from runners.run_cs_enforcement import POLICY_ALL, SCENARIO, THRESHOLD  # noqa: E402

# Canonical allocation plan; the ten-feature run is comparison-only.
PLAN = REPO / "data" / "enforcement_eight_feature_plan.json"
# Expected enforcement matrix: 2 models x 5 topologies.
EXPECT_CONFIGURATIONS = 10
ENF_EIGHT = REPO / "results" / "enforcement" / "customer_service" / "6041555_eight"
ENF_TEN = REPO / "results" / "enforcement" / "customer_service" / "6039372"
ENF_DIR = ENF_EIGHT
CONTROL_DIR = REPO / "results" / "enforcement" / "customer_service" / "6039883_control"
# Canonical location first; the repo root is only where a job's output lands
# before it is filed.
CONTROL_GLOB = (str(CONTROL_DIR / "cs_control_*.json")
                if CONTROL_DIR.exists() else str(REPO / "cs_control_*.json"))
OUT = REPO / "data" / "control_adjusted_prevention.json"

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20260817
MAIN_POLICIES = [POLICY_MESA_CF, POLICY_RANDOM, POLICY_ALL]
FEATURE_BLOCK = {"6041555_eight": ("structural_dynamic", 8),
                 # Added budgets share the eight-feature allocation guard.
                 "frontier_merged": ("structural_dynamic", 8),
                 "frontier_all": ("structural_dynamic", 8),
                 "6039372": ("structural_dynamic_f2_f3", 10)}
POLICY_LABEL = {
    POLICY_MESA_CF: "MESA-Local (cross-fitted)",
    POLICY_RANDOM: "Random allocation (5 frozen seeds)",
    POLICY_ALL: "Monitor all edges",
}


def _key(model, topology, task_id, edge, payload, threshold):
    return (model, topology, task_id, tuple(edge), payload, threshold)


def load_controls():
    out, dupes = {}, 0
    for p in sorted(glob.glob(CONTROL_GLOB)):
        d = json.loads(Path(p).read_text())
        if d.get("n_no_execution"):
            raise SystemExit(
                "%s reports %d controls that never executed; a control arm "
                "that did not run measures nothing"
                % (p, d["n_no_execution"]))
        if d.get("n_with_interventions"):
            raise SystemExit("%s: %d controls withheld a message"
                             % (p, d["n_with_interventions"]))
        for c in d.get("cells") or []:
            if c.get("control_success") is None:
                continue
            k = _key(c["model"], c["topology"], c["task_id"],
                     c["attacked_edge"], c.get("attack_payload_sha16"),
                     c.get("threshold"))
            if k in out:
                dupes += 1
            out[k] = c
    return out, dupes


def load_interventions():
    rows = []
    for p in sorted(glob.glob(str(ENF_DIR / "rerun_cache_*.json"))):
        d = json.loads(Path(p).read_text())
        for r in (d.get("rows") or {}).values():
            pr = r.get("provenance") or {}
            if pr.get("attacked_edge") is None:
                continue
            rows.append({
                "model": pr.get("model"), "topology": pr.get("topology"),
                "task_id": pr.get("task_id"),
                "attacked_edge": tuple(pr["attacked_edge"]),
                "payload": pr.get("attack_payload_sha16"),
                "threshold": pr.get("threshold"),
                "monitored": tuple(sorted(tuple(e) for e in
                                          (r.get("monitored") or []))),
                "clean_correct": bool(r.get("clean_correct")),
                "original_success": bool(r.get("original_success")),
                # defended_success == True means the DEFENDED run still got it
                # wrong, i.e. the attack survived enforcement.
                "defended_success": (None if r.get("defended_success") is None
                                     else bool(r["defended_success"])),
                "n_interventions": r.get("n_interventions", 0),
                "prevented": bool(r.get("prevented")),
            })
    return rows


def load_cells():
    cells = []
    for p in sorted(glob.glob(str(ENF_DIR / "cs_enforcement_*.json"))):
        d = json.loads(Path(p).read_text())
        for c in d.get("cells") or []:
            c = dict(c)
            c["_monitored"] = tuple(sorted(tuple(e) for e in
                                           (c.get("monitored") or [])))
            cells.append(c)
    return cells


def _boot(x, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    x = np.asarray([v for v in x if v is not None and not
                    (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(x) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(b, len(x)), replace=True).mean(axis=1)
    return {"mean": float(x.mean()), "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)), "n": int(len(x))}


def main():
    global ENF_DIR, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--enf-dir", default=str(ENF_EIGHT),
                    help="allocation source; defaults to the canonical eight")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--expect-configurations", type=int,
                    default=EXPECT_CONFIGURATIONS,
                    help="fail unless exactly this many configurations are "
                         "present; 0 disables the check")
    args = ap.parse_args()
    ENF_DIR = Path(args.enf_dir)
    OUT = Path(args.out)
    if not ENF_DIR.exists():
        raise SystemExit("no such enforcement directory: %s" % ENF_DIR)
    block, nfeat = FEATURE_BLOCK.get(ENF_DIR.name, ("unknown", 0))
    print("allocation source: %s  (feature block %s, %d features)"
          % (ENF_DIR.name, block, nfeat))
    if block != "structural_dynamic":
        print("  WARNING: this is NOT the canonical block. The paper allocates "
              "on the frozen eight; treat this run as a comparison only.")

    controls, dupes = load_controls()
    interventions = load_interventions()
    cells = load_cells()

    # ---- HARD INVARIANT 1: every MESA cell allocates on the frozen eight ----
    # A cell whose monitored set does not match the eight-feature plan came
    # from a different feature block and must never enter a reported number.
    if PLAN.exists() and block == "structural_dynamic":
        want = {}
        for c in json.loads(PLAN.read_text())["cells"]:
            want[(c["model"], c["topology"], round(c["budget"], 2),
                  c["direction"])] = tuple(sorted(tuple(e)
                                                  for e in c["monitored"]))
        wrong = []
        for c in cells:
            if "mesa" not in c["policy"]:
                continue          # random and monitor-all are feature-free
            k = (c["model"], c["topology"], round(c["budget"], 2),
                 c.get("direction"))
            got = tuple(sorted(tuple(e) for e in c["_monitored"]))
            if k in want and want[k] != got:
                wrong.append(k)
        if wrong:
            raise SystemExit(
                "ALLOCATION MISMATCH: %d MESA cell(s) do not match the "
                "eight-feature plan, e.g. %s. These were allocated on a "
                "different feature block and must not be reported."
                % (len(wrong), wrong[:3]))
    folds = fold_of_task(SCENARIO)

    # ---- pair every intervention rerun with its control -------------------
    paired, unmatched = [], 0
    for r in interventions:
        k = _key(r["model"], r["topology"], r["task_id"], r["attacked_edge"],
                 r["payload"], r["threshold"])
        c = controls.get(k)
        if c is None or r["defended_success"] is None:
            unmatched += 1
            continue
        d = int(bool(c["control_success"])) - int(r["defended_success"])
        paired.append(dict(r, control_success=bool(c["control_success"]),
                           delta=d))

    counts = collections.Counter(
        (p["control_success"], p["defended_success"]) for p in paired)
    quad = {
        "attributable_prevention (control=1, intervention=0)":
            counts[(True, False)],
        "negative_effect (control=0, intervention=1)": counts[(False, True)],
        "attack_survived_either_way (1,1)": counts[(True, True)],
        "correct_either_way (0,0)": counts[(False, False)],
    }

    # ---- per cell, then per configuration ---------------------------------
    by_mon = collections.defaultdict(list)
    for p in paired:
        by_mon[(p["model"], p["topology"], p["monitored"])].append(p)

    cell_rows = []
    for c in cells:
        of = (c.get("direction") or "->?").split("->")[-1]
        pool = [p for p in by_mon.get(
            (c["model"], c["topology"], c["_monitored"]), [])
            if folds.get(p["task_id"]) == of
            and p["clean_correct"] and p["original_success"]]
        if not pool:
            continue
        deltas = [p["delta"] for p in pool]
        cell_rows.append({
            "model": c["model"], "topology": c["topology"],
            "policy": c["policy"], "budget": c["budget"],
            "budget_k": c["budget_k"], "seed": c.get("seed"),
            "direction": c.get("direction"),
            "n_paired": len(pool),
            "raw_prevention_rate": c.get("prevention_rate"),
            "attributable": float(np.mean(deltas)),
            "n_positive": sum(1 for d in deltas if d > 0),
            "n_negative": sum(1 for d in deltas if d < 0),
            "control_success_rate": float(np.mean(
                [1.0 if p["control_success"] else 0.0 for p in pool])),
            "clean_utility_retained": c.get("clean_utility"),
            "message_fpr": c.get("message_fpr"),
            "detector_calls": c.get("detector_calls"),
            "latency_s": ((c.get("detector_latency_s") or 0.0)
                          + (c.get("rerun_latency_s") or 0.0)),
        })

    # Seeds are five draws from one configuration, not five systems: collapse
    # them before the configuration becomes the unit.
    by_seed = collections.defaultdict(list)
    for r in cell_rows:
        by_seed[(r["model"], r["topology"], r["policy"], r["budget"],
                 r["seed"])].append(r)
    pooled = {}
    for k, rs in by_seed.items():
        n = sum(r["n_paired"] for r in rs)
        pooled[k] = {
            "attributable": (sum(r["attributable"] * r["n_paired"] for r in rs)
                             / n) if n else float("nan"),
            "raw": float(np.nanmean([r["raw_prevention_rate"] for r in rs])),
            "n_positive": sum(r["n_positive"] for r in rs),
            "n_negative": sum(r["n_negative"] for r in rs),
            "n_paired": n,
            "clean": float(np.nanmean([r["clean_utility_retained"] for r in rs])),
            "fpr": float(np.nanmean([r["message_fpr"] for r in rs])),
            "detector_calls": sum(r["detector_calls"] or 0 for r in rs),
            "latency_s": sum(r["latency_s"] or 0.0 for r in rs),
        }
    by_config = collections.defaultdict(list)
    for (m, t, pol, b, _seed), v in pooled.items():
        by_config[(m, t, pol, b)].append(v)
    per_config = {k: {
        "attributable": float(np.nanmean([v["attributable"] for v in vs])),
        "raw": float(np.nanmean([v["raw"] for v in vs])),
        "n_positive": int(np.mean([v["n_positive"] for v in vs])),
        "n_negative": int(np.mean([v["n_negative"] for v in vs])),
        "n_paired": int(np.mean([v["n_paired"] for v in vs])),
        "clean": float(np.nanmean([v["clean"] for v in vs])),
        "fpr": float(np.nanmean([v["fpr"] for v in vs])),
        "detector_calls": float(np.mean([v["detector_calls"] for v in vs])),
        "latency_s": float(np.mean([v["latency_s"] for v in vs])),
    } for k, vs in by_config.items()}

    # ---- HARD INVARIANT 2: the matrix is complete --------------------------
    seen = sorted({(m, t) for (m, t, _p, _b) in per_config})
    if args.expect_configurations and len(seen) != args.expect_configurations:
        raise SystemExit(
            "CONFIGURATION COUNT: expected %d, found %d: %s. A missing "
            "configuration does not raise anywhere else -- the table simply "
            "reports a smaller n. Submit the missing configurations or pass "
            "--expect-configurations explicitly to acknowledge a partial run."
            % (args.expect_configurations, len(seen),
               ["%s/%s" % k for k in seen]))

    table = []
    for policy, budget in sorted({(p, b) for (_m, _t, p, b) in per_config}):
        vals = [v for (m, t, pol, bb), v in per_config.items()
                if pol == policy and bb == budget]
        att = [v["attributable"] for v in vals]
        table.append({
            "policy": policy, "policy_label": POLICY_LABEL.get(policy, policy),
            "budget": budget, "n_configurations": len(vals),
            "raw_prevention": float(np.nanmean([v["raw"] for v in vals])),
            "attributable_prevention": float(np.nanmean(att)),
            "attributable_ci": _boot(att),
            "n_positive": int(sum(v["n_positive"] for v in vals)),
            "n_negative": int(sum(v["n_negative"] for v in vals)),
            "clean_utility_retained": float(np.nanmean([v["clean"] for v in vals])),
            "message_fpr": float(np.nanmean([v["fpr"] for v in vals])),
            "detector_calls": float(sum(v["detector_calls"] for v in vals)),
            "latency_s_per_configuration": float(np.mean(
                [v["latency_s"] for v in vals])),
        })

    # MESA vs random, paired on the configuration, on the ADJUSTED scale.
    stats = []
    for budget in sorted({b for (_m, _t, p, b) in per_config
                          if p == POLICY_MESA_CF}):
        diffs = []
        for (m, t, pol, b), v in per_config.items():
            if pol != POLICY_MESA_CF or b != budget:
                continue
            r = per_config.get((m, t, POLICY_RANDOM, budget))
            if r:
                diffs.append(v["attributable"] - r["attributable"])
        stats.append({"budget": budget, "n": len(diffs),
                      "mean_diff": float(np.mean(diffs)) if diffs else None,
                      "ci": _boot(diffs)})

    n_res = sum(1 for c in controls.values()
                if c.get("control_success") is not None)
    n_rep = sum(1 for c in controls.values() if c.get("reproduced"))
    payload = {
        "scenario": SCENARIO, "threshold": THRESHOLD,
        "allocation_source": ENF_DIR.name,
        "feature_block": block,
        "n_allocation_features": nfeat,
        "canonical": block == "structural_dynamic",
        "supersedes": ("6039372, which allocated on the ten-feature score "
                       "rejected by the feature freeze"),
        "estimand": ("MATCHED, CONTROL-ADJUSTED CAUSAL ESTIMATE UNDER THE "
                     "RERUN DESIGN. Not a randomised experiment over "
                     "deployments: the counterfactual is the same trial "
                     "re-executed with nothing withheld, so the estimate is "
                     "causal WITH RESPECT TO the withholding intervention, "
                     "conditional on the rerun design."),
        "baseline_rerun_instability": ("22.6% of controls change outcome with "
                                       "nothing withheld; this is the noise "
                                       "floor the adjustment removes and it "
                                       "must be quoted alongside any "
                                       "prevention number."),
        "uncertainty": ("Configuration-clustered: the bootstrap resamples "
                        "CONFIGURATIONS, not trials, and the five random seeds "
                        "are averaged within a configuration before it becomes "
                        "the unit. Trial-level resampling would treat 2278 "
                        "correlated reruns as independent."),
        "quadrants_are_diagnostic": ("The pooled outcome quadrants describe "
                                     "the whole rerun set and are DIAGNOSTIC. "
                                     "Every claim uses the policy-specific "
                                     "estimates in `table`."),
        "note": ("Trial-paired control-adjusted prevention. "
                 "delta = 1[control succeeds] - 1[intervention succeeds]. "
                 "One control per trial, reused across policies and budgets."),
        "n_configurations": len(seen),
        "configurations": ["%s/%s" % k for k in seen],
        "n_controls": len(controls),
        "duplicate_control_keys": dupes,
        "control_reproducibility": (n_rep / n_res) if n_res else None,
        "control_flip_rate": (1 - n_rep / n_res) if n_res else None,
        "n_intervention_reruns": len(interventions),
        "n_paired": len(paired),
        "n_unmatched": unmatched,
        "outcome_quadrants": quad,
        "table": table,
        "mesa_vs_random_adjusted": stats,
        "per_configuration": [
            dict(model=k[0], topology=k[1], policy=k[2], budget=k[3], **v)
            for k, v in sorted(per_config.items())],
        "cells": cell_rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("CONTROL-ADJUSTED PREVENTION -- %s" % SCENARIO)
    print("  controls %d, paired intervention reruns %d, unmatched %d"
          % (len(controls), len(paired), unmatched))
    print("  control reproducibility %.4f (flip rate %.4f)"
          % (payload["control_reproducibility"], payload["control_flip_rate"]))
    print()
    print("  outcome quadrants over %d paired reruns:" % len(paired))
    for k, v in quad.items():
        print("    %-52s %5d  (%.1f%%)" % (k, v, 100.0 * v / max(len(paired), 1)))
    print()
    print("  %-42s %6s %8s %10s %20s %7s %7s"
          % ("policy", "budget", "raw", "adjusted", "95% CI", "clean", "FPR"))
    order = {p: i for i, p in enumerate(MAIN_POLICIES)}
    for r in sorted(payload["table"],
                    key=lambda r: (order.get(r["policy"], 9), r["budget"])):
        ci = r["attributable_ci"]
        print("  %-42s %6.2f %8.3f %10.3f  [%+.3f, %+.3f] %7.3f %7.3f"
              % (r["policy_label"], r["budget"], r["raw_prevention"],
                 r["attributable_prevention"], ci["lo"], ci["hi"],
                 r["clean_utility_retained"], r["message_fpr"]))
    print()
    for s in stats:
        print("  MESA - random (adjusted) at %.2f: n=%d, %+.3f [%+.3f, %+.3f]"
              % (s["budget"], s["n"], s["mean_diff"], s["ci"]["lo"],
                 s["ci"]["hi"]))
    print()
    print("wrote %s" % OUT.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
