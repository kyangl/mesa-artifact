"""Build the canonical eight-feature ranking table.

Each budget has its own exact random and oracle values. Inference uses paired
configuration-level differences, bootstrap intervals, and exact sign-flip
tests; ratios are descriptive only.
"""

import collections
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.run_mesa_fit import (load_enriched_directional,  # noqa: E402
                                   collapse_directions, BLOCKS)
from analysis.run_mesa_cv import _binary_idx                   # noqa: E402
from analysis.run_nested_cv import (expected_tie_credit,       # noqa: E402
                                    coverage_auc)
from src.saliency.mesa_scores import MesaLocal                 # noqa: E402

OUT = REPO / "data" / "canonical_ranking.json"
OUT_MD = REPO / "data" / "TABLE_canonical_ranking.md"
SCENARIOS = ("customer_service", "software_engineering")
MODELS = ("gemma4:e4b", "qwen3.5:9b", "llama3.1:8b")
BUDGETS = (0.10, 0.20)
CANONICAL = "structural_dynamic"      # frozen; see freeze_feature_decision.py
REGISTRY = REPO / "data" / "experiment_registry.json"
# Expected matrix: 3 models x 5 topologies x 2 scenarios.
EXPECT_CONFIGURATIONS_TOTAL = 30
B = 10000
SEED = 20260817


def _cov(y, s, k):
    if y.sum() <= 0:
        return float("nan")
    return float(expected_tie_credit(y, s)[:k].sum() / y.sum())


def _boot(d, b=B, seed=SEED):
    d = np.asarray([x for x in d if x is not None and not
                    (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(d) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    rng = np.random.default_rng(seed)
    dr = rng.choice(d, size=(b, len(d)), replace=True).mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(dr, 2.5)),
            "hi": float(np.percentile(dr, 97.5)), "n": int(len(d))}


def sign_flip(diffs):
    """Exact paired sign-flip over all 2^n patterns."""
    d = np.asarray(list(diffs), dtype=float)
    n = len(d)
    if n == 0 or not np.any(np.abs(d) > 0):
        return {"n": int(n), "p_two_sided": 1.0,
                "min_attainable_two_sided": (2.0 / 2 ** n) if n else None}
    if n > 20:                       # exact enumeration is infeasible above this
        rng = np.random.default_rng(SEED)
        signs = rng.choice((1.0, -1.0), size=(200000, n))
        exact = False
    else:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        exact = True
    obs = float(d.mean())
    means = signs.dot(d) / n
    eps = 1e-12
    return {"n": int(n), "observed_mean": obs, "exact": exact,
            "p_two_sided": float(np.mean(np.abs(means) >= abs(obs) - eps)),
            "n_patterns": int(len(signs)),
            "min_attainable_two_sided": 2.0 / len(signs)}


def registry_configurations(scenario):
    """(model, topology) the registry marks feature-complete for this scenario."""
    if not REGISTRY.exists():
        return None
    rows = json.loads(REGISTRY.read_text())["rows"]
    return {(r["model"], r["topology"]) for r in rows
            if r.get("feature_complete") and r.get("scenario") == scenario}


def analyse(scenario):
    fm, _d, _m = load_enriched_directional(scenarios=[scenario])
    cols = BLOCKS[CANONICAL]
    idx = [fm.feature_names.index(c) for c in cols]
    names = [fm.feature_names[i] for i in idx]
    s_all = MesaLocal(names, _binary_idx(names)).score(fm.X[:, idx], fm.groups)
    cols_s = BLOCKS["structural"]
    idx_s = [fm.feature_names.index(c) for c in cols_s]
    names_s = [fm.feature_names[i] for i in idx_s]
    struct_all = MesaLocal(names_s, _binary_idx(names_s)).score(
        fm.X[:, idx_s], fm.groups)

    per = collections.defaultdict(dict)
    meta = {}
    for g in np.unique(fm.groups):
        sel = np.where(fm.groups == g)[0]
        y = fm.y_success[sel].astype(float)
        E = len(y)
        meta[g] = {"model": fm.rows[sel[0]]["model"],
                   "topology": fm.rows[sel[0]]["topology"], "n_edges": E}
        s = np.nan_to_num(s_all[sel], nan=0.5)
        st = np.nan_to_num(struct_all[sel], nan=0.5)
        per["auc"][g] = coverage_auc(y, s)
        per["auc_struct"][g] = coverage_auc(y, st)
        per["rho"][g] = (float(stats.spearmanr(s, y).statistic)
                         if len(set(np.round(y, 12))) > 1
                         and len(set(np.round(s, 12))) > 1 else float("nan"))
        for b in BUDGETS:
            t = int(b * 100)
            k = max(1, min(E, int(math.ceil(b * E))))
            per["mesa%d" % t][g] = _cov(y, s, k)
            per["struct%d" % t][g] = _cov(y, st, k)
            per["random%d" % t][g] = k / E          # per-budget, k/|E|
            per["oracle%d" % t][g] = _cov(y, y, k)

    col = {kk: collapse_directions(v) for kk, v in per.items()}
    cfgs = sorted(col["auc"])
    cfg_model, meta_topology = {}, {}
    for g, m in meta.items():
        for c in cfgs:
            if str(c) in str(g) or str(g) in str(c):
                cfg_model[c] = m["model"]
                meta_topology[c] = m["topology"]

    def block(sub):
        out = {kk: float(np.nanmean([col[kk][c] for c in sub])) for kk in col}
        for b in BUDGETS:
            t = int(b * 100)
            d = [col["mesa%d" % t][c] - col["random%d" % t][c] for c in sub]
            out["diff%d" % t] = _boot(d)
            out["signflip%d" % t] = sign_flip(d)
            # Descriptive only.
            out["ratio%d" % t] = (out["mesa%d" % t] / out["random%d" % t]
                                  if out["random%d" % t] else float("nan"))
        d8 = [col["auc"][c] - col["auc_struct"][c] for c in sub]
        out["eight_minus_structural_auc"] = _boot(d8)
        out["n_configurations"] = len(sub)
        return out

    # ---- HARD INVARIANT: every feature-complete configuration is present ---
    # The registry is the authority on what SHOULD be here; the matrix is what
    # IS here. Silence between them is the failure mode being guarded.
    present = {(cfg_model.get(c), meta_topology.get(c)) for c in cfgs}
    expected = registry_configurations(scenario)
    if expected is not None:
        missing = expected - present
        extra = present - expected
        if missing or extra:
            raise SystemExit(
                "%s RANKING MATRIX DOES NOT MATCH THE REGISTRY.\n"
                "  missing (registry says feature-complete, absent here): %s\n"
                "  extra   (present here, not feature-complete): %s\n"
                "A missing configuration does not raise anywhere else; the "
                "table just reports a smaller n."
                % (scenario, sorted(missing) or "none", sorted(extra) or "none"))

    res = {"by_model": {}, "macro": None, "n_configurations": len(cfgs),
           "configurations": sorted("%s/%s" % (cfg_model.get(c),
                                               meta_topology.get(c))
                                    for c in cfgs)}
    for m in MODELS:
        sub = [c for c in cfgs if cfg_model.get(c) == m]
        if sub:
            res["by_model"][m] = block(sub)
    res["macro"] = block(cfgs)
    return res


def main():
    payload = {
        "canonical_block": CANONICAL, "n_features": 8,
        "note": ("Main-paper ranking table on the frozen eight-feature score. "
                 "EVERY BUDGET CARRIES ITS OWN random and oracle column: exact "
                 "random is k/|E| and k changes with the budget, so one random "
                 "column is always wrong for one of them. Ratios are "
                 "DESCRIPTIVE; inference is the paired configuration-level "
                 "difference with a bootstrap CI and an exact sign-flip test."),
        "scenarios": {s: analyse(s) for s in SCENARIOS},
    }
    total = sum(payload["scenarios"][s]["n_configurations"] for s in SCENARIOS)
    if total != EXPECT_CONFIGURATIONS_TOTAL:
        raise SystemExit(
            "CONFIGURATION COUNT: expected %d across %s, found %d (%s). "
            "Refusing to write a table that silently covers less than the "
            "study."
            % (EXPECT_CONFIGURATIONS_TOTAL, list(SCENARIOS), total,
               {s: payload["scenarios"][s]["n_configurations"]
                for s in SCENARIOS}))
    payload["n_configurations_total"] = total
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))

    L = ["# Canonical ranking — eight-feature MESA-Local", "",
         "Generated by `python analysis/canonical_ranking_table.py`. "
         "Within-configuration; cross-fit directions collapsed before "
         "averaging. **Each budget has its own random and oracle column.**", ""]
    for scenario, r in payload["scenarios"].items():
        L += ["## %s" % scenario, "",
              "| model | cfgs | MESA@10 | random@10 | oracle@10 | MESA@20 | "
              "random@20 | oracle@20 | cov-AUC | ρ |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for m, v in list(r["by_model"].items()) + [("**macro**", r["macro"])]:
            L.append("| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | "
                     "%.3f | %.3f |"
                     % (m, v["n_configurations"], v["mesa10"], v["random10"],
                        v["oracle10"], v["mesa20"], v["random20"],
                        v["oracle20"], v["auc"], v["rho"]))
        L += ["", "### MESA − random, paired on the configuration", "",
              "| model | budget | Δ | 95% CI | sign-flip p | min attainable p "
              "| ratio (descriptive) |", "|---|---|---|---|---|---|---|"]
        for m, v in list(r["by_model"].items()) + [("**macro**", r["macro"])]:
            for b in BUDGETS:
                t = int(b * 100)
                d, sf = v["diff%d" % t], v["signflip%d" % t]
                L.append("| %s | %d%% | %+.3f | [%+.3f, %+.3f] | %.4f | %.4f | "
                         "%.2fx |"
                         % (m, t, d["mean"], d["lo"], d["hi"],
                            sf["p_two_sided"], sf["min_attainable_two_sided"],
                            v["ratio%d" % t]))
        L.append("")
    OUT_MD.write_text("\n".join(L))

    for scenario, r in payload["scenarios"].items():
        print("=== %s" % scenario)
        print("  %-13s %5s %8s %8s %8s %8s %8s %8s"
              % ("model", "cfgs", "MESA@10", "rand@10", "orac@10",
                 "MESA@20", "rand@20", "orac@20"))
        for m, v in list(r["by_model"].items()) + [("MACRO", r["macro"])]:
            print("  %-13s %5d %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
                  % (m, v["n_configurations"], v["mesa10"], v["random10"],
                     v["oracle10"], v["mesa20"], v["random20"], v["oracle20"]))
        print("  MESA - random, paired:")
        for m, v in list(r["by_model"].items()) + [("MACRO", r["macro"])]:
            for b in BUDGETS:
                t = int(b * 100)
                d, sf = v["diff%d" % t], v["signflip%d" % t]
                print("    %-13s @%d%%  %+.3f [%+.3f, %+.3f]  p=%.4f "
                      "(min %.4f)"
                      % (m, t, d["mean"], d["lo"], d["hi"],
                         sf["p_two_sided"], sf["min_attainable_two_sided"]))
        print()
    print("wrote %s" % OUT.relative_to(REPO))
    print("wrote %s" % OUT_MD.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
