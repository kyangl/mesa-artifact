"""Apply the predeclared feature-freeze rule.

Paired configuration-level contrasts compare structural, eight-feature, and
ten-feature blocks by domain. The simplest block within one standard error of
the best is selected after cross-fit directions are collapsed.
"""

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.run_mesa_fit import (load_enriched_directional,  # noqa: E402
                                   collapse_directions, BLOCKS)
from analysis.run_mesa_cv import _binary_idx                   # noqa: E402
from analysis.run_nested_cv import (expected_tie_credit,       # noqa: E402
                                    coverage_auc)
from src.saliency.mesa_scores import MesaLocal                 # noqa: E402

OUT = REPO / "data" / "feature_freeze_decision.json"
OUT_MD = REPO / "data" / "TABLE_feature_freeze.md"
SCENARIOS = ("customer_service", "software_engineering")
BUDGET = 0.20
B = 10000
SEED = 20260817
BLOCK_OF = {"structural": "structural",
            "eight": "structural_dynamic",
            "ten": "structural_dynamic_f2_f3"}


def _cov(y, s, k):
    if y.sum() <= 0:
        return float("nan")
    return float(expected_tie_credit(y, s)[:k].sum() / y.sum())


def _boot(d, b=B, seed=SEED):
    d = np.asarray([x for x in d if x is not None and not
                    (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(d) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "se": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    draws = rng.choice(d, size=(b, len(d)), replace=True).mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            "se": float(draws.std(ddof=1)), "n": int(len(d))}


def per_config_metrics(scenario):
    fm, _d, _m = load_enriched_directional(scenarios=[scenario])
    scores = {}
    for name, key in BLOCK_OF.items():
        cols = BLOCKS[key]
        idx = [fm.feature_names.index(c) for c in cols]
        nm = [fm.feature_names[i] for i in idx]
        scores[name] = MesaLocal(nm, _binary_idx(nm)).score(fm.X[:, idx],
                                                            fm.groups)
    auc, cov, rnd = ({n: {} for n in BLOCK_OF} for _ in range(3))
    for g in np.unique(fm.groups):
        sel = np.where(fm.groups == g)[0]
        y = fm.y_success[sel].astype(float)
        E = len(y)
        k = max(1, min(E, int(math.ceil(BUDGET * E))))
        for n in BLOCK_OF:
            s = np.nan_to_num(scores[n][sel], nan=0.5)
            auc[n][g] = coverage_auc(y, s)
            cov[n][g] = _cov(y, s, k)
        rnd["eight"][g] = k / E
    out = {}
    for n in BLOCK_OF:
        out[("auc", n)] = collapse_directions(auc[n])
        out[("cov20", n)] = collapse_directions(cov[n])
    out[("cov20", "random")] = collapse_directions(rnd["eight"])
    # coverage-AUC has a null of 0.5, so "random" on that scale is the null.
    cfgs = sorted(out[("auc", "eight")])
    out[("auc", "random")] = {c: 0.5 for c in cfgs}
    return out, cfgs


def main():
    payload = {
        # Name the blocks explicitly. The contrast keys say "eight" and
        # "ten"; a reader auditing the freeze needs the block identifiers those
        # labels refer to, or the decision cannot be checked against the code.
        "blocks_compared": {
            "structural": {"name": "structural", "n_features": 6},
            "eight": {"name": "structural_dynamic", "n_features": 8,
                      "outcome": "PROMOTED -- canonical"},
            "ten": {"name": "structural_dynamic_f2_f3", "n_features": 10,
                    "outcome": ("REJECTED -- evaluated extension that failed "
                                "the prespecified promotion rule")},
        },
        "decision_rule": ("Prefer the SIMPLEST block within one standard error "
                          "of the best. A larger block is promoted only if it "
                          "beats the simpler one by more than one SE of the "
                          "PAIRED difference. Declared before the numbers."),
        "pairing": ("configuration-level; cross-fit directions collapsed to "
                    "one value per configuration first"),
        "budget": BUDGET,
        "scenarios": {},
    }

    for scenario in SCENARIOS:
        m, cfgs = per_config_metrics(scenario)
        s = {"n_configurations": len(cfgs), "contrasts": {}}
        for metric in ("auc", "cov20"):
            for label, (a, b) in (("eight_minus_structural",
                                   ("eight", "structural")),
                                  ("ten_minus_eight", ("ten", "eight")),
                                  ("eight_minus_random",
                                   ("eight", "random"))):
                diffs = [m[(metric, a)][c] - m[(metric, b)][c] for c in cfgs
                         if c in m[(metric, a)] and c in m[(metric, b)]]
                s["contrasts"]["%s|%s" % (metric, label)] = _boot(diffs)
        s["levels"] = {"%s|%s" % (metric, n):
                       float(np.nanmean([m[(metric, n)][c] for c in cfgs]))
                       for metric in ("auc", "cov20")
                       for n in list(BLOCK_OF) + ["random"]
                       if (metric, n) in m}
        payload["scenarios"][scenario] = s

    # ---- apply the rule ---------------------------------------------------
    verdict = {}
    for scenario, s in payload["scenarios"].items():
        c = s["contrasts"]
        e_minus_s = c["auc|eight_minus_structural"]
        t_minus_e = c["auc|ten_minus_eight"]
        e_minus_r = c["auc|eight_minus_random"]
        # Promote ten over eight only if it beats it by more than one SE.
        promote_ten = t_minus_e["mean"] > t_minus_e["se"]
        # Promote eight over structural on the same test.
        promote_eight = e_minus_s["mean"] > e_minus_s["se"]
        verdict[scenario] = {
            "eight_beats_structural_by_more_than_1SE": bool(promote_eight),
            "ten_beats_eight_by_more_than_1SE": bool(promote_ten),
            "eight_beats_random": {
                "mean": e_minus_r["mean"], "lo": e_minus_r["lo"],
                "hi": e_minus_r["hi"],
                "excludes_zero": bool(e_minus_r["lo"] > 0)},
        }
    payload["per_scenario_verdict"] = verdict

    promote_ten_anywhere = any(v["ten_beats_eight_by_more_than_1SE"]
                               for v in verdict.values())
    payload["decision"] = {
        "canonical_block": ("structural_dynamic_f2_f3" if promote_ten_anywhere
                            else "structural_dynamic"),
        "n_features": 10 if promote_ten_anywhere else 8,
        "f2_f3_in_primary_score": bool(promote_ten_anywhere),
        "rationale": (
            "Ten beats eight by more than one SE in at least one scenario."
            if promote_ten_anywhere else
            "Ten does not beat eight by more than one standard error in "
            "EITHER scenario, so the simpler block is canonical and F2/F3 are "
            "rejected from the primary score. They remain reported as "
            "descriptive contributions."),
        "exploration_status": "FROZEN -- no further feature or aggregation "
                              "variants are to be evaluated",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    L = ["# Feature-set freeze", "",
         "Generated by `python analysis/freeze_feature_decision.py`.", "",
         "**Rule, declared before the numbers:** prefer the simplest block "
         "within one standard error of the best. A larger block is promoted "
         "only if it beats the simpler one by more than one SE of the paired "
         "difference.", "",
         "Paired at the configuration level; cross-fit directions collapsed "
         "first.", ""]
    for scenario, s in payload["scenarios"].items():
        L += ["## %s (%d configurations)" % (scenario, s["n_configurations"]),
              "", "| contrast | metric | Δ | 95% CI | SE | > 1 SE? |",
              "|---|---|---|---|---|---|"]
        for label in ("eight_minus_structural", "ten_minus_eight",
                      "eight_minus_random"):
            for metric in ("auc", "cov20"):
                v = s["contrasts"]["%s|%s" % (metric, label)]
                L.append("| %s | %s | %+.3f | [%+.3f, %+.3f] | %.3f | %s |"
                         % (label.replace("_", " "),
                            "coverage-AUC" if metric == "auc" else "cov@20%",
                            v["mean"], v["lo"], v["hi"], v["se"],
                            "yes" if v["mean"] > v["se"] else "**no**"))
        L.append("")
    d = payload["decision"]
    L += ["## Decision", "",
          "**Canonical MESA-Local = `%s` (%d features).**"
          % (d["canonical_block"], d["n_features"]), "",
          d["rationale"], "",
          "F2 and F3 are **%s** the primary score."
          % ("in" if d["f2_f3_in_primary_score"] else "rejected from"), "",
          "**Feature and aggregation exploration is now FROZEN.**", ""]
    OUT_MD.write_text("\n".join(L))

    for scenario, s in payload["scenarios"].items():
        print("=== %s (%d configurations)" % (scenario, s["n_configurations"]))
        for label in ("eight_minus_structural", "ten_minus_eight",
                      "eight_minus_random"):
            for metric in ("auc", "cov20"):
                v = s["contrasts"]["%s|%s" % (metric, label)]
                print("  %-26s %-7s %+.3f [%+.3f, %+.3f]  SE %.3f  %s"
                      % (label, metric, v["mean"], v["lo"], v["hi"], v["se"],
                         ">1SE" if v["mean"] > v["se"] else "within 1SE"))
        print()
    print("DECISION: canonical = %s (%d features); F2/F3 %s primary"
          % (d["canonical_block"], d["n_features"],
             "IN" if d["f2_f3_in_primary_score"] else "REJECTED FROM"))
    print(d["rationale"])
    print()
    print("wrote %s" % OUT.relative_to(REPO))
    print("wrote %s" % OUT_MD.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
