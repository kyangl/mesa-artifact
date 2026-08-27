"""Final cross-fitted CS feature-contribution table, incl. leave-one-out.

Blocks: structural, dynamic, structural+dynamic, full 10, and full-10 minus
each task-derived feature in turn. Metrics per configuration-direction,
collapsed to one value per configuration (half weight each) BEFORE any
bootstrap, because the two directions share tasks.
"""
import json, sys
import numpy as np
from scipy import stats
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from analysis.run_mesa_fit import (CANONICAL_BLOCK, load_enriched_directional, collapse_directions,
                                   paired_bootstrap, BLOCKS)
from analysis.run_nested_cv import coverage_auc, coverage_at_k
from analysis.run_mesa_cv import _binary_idx
from src.saliency.mesa_scores import MesaLocal

# Use the F2/F3 superset for leave-one-out; include the canonical block separately.
FULL = BLOCKS["structural_dynamic_f2_f3"]
LOO = {"full10_minus_ablation": "ablation_delta",
       "full10_minus_perturbation": "perturbation_delta",
       "full10_minus_F2": "receiver_response_sensitivity",
       "full10_minus_F3": "consequence_proximity"}

def score_block(fm, cols):
    idx=[fm.feature_names.index(c) for c in cols]
    names=[fm.feature_names[i] for i in idx]
    s=MesaLocal(names,_binary_idx(names)).score(fm.X[:,idx], fm.groups)
    auc,c20,rho={},{},{}
    for g in np.unique(fm.groups):
        i=fm.groups==g; y=fm.y_success[i]; sc=s[i]
        auc[g]=coverage_auc(y,sc); c20[g]=coverage_at_k(y,sc,0.20)
        asr=y/np.maximum(fm.n_trials[i],1)
        rho[g]=(float(stats.spearmanr(sc,asr).statistic)
                if len(sc)>2 and len(set(sc))>1 and len(set(asr))>1 else np.nan)
    return (collapse_directions(auc), collapse_directions(c20),
            collapse_directions(rho))

def main(scenario="customer_service"):
    fm,_,_=load_enriched_directional(scenarios=[scenario])
    blocks={"structural":BLOCKS["structural"],"dynamic":BLOCKS["dynamic"],
            "structural_dynamic":BLOCKS["structural_dynamic"],"full10":FULL}
    for name,drop in LOO.items(): blocks[name]=[c for c in FULL if c!=drop]
    res={n:score_block(fm,c) for n,c in blocks.items()}
    print("CROSS-FITTED CS FEATURE CONTRIBUTION (MESA-Local, n=%d configs)\n"
          % len(res["full10"][0]))
    print("  %-28s %8s %9s %9s" % ("block","covAUC","cov@20%","within-rho"))
    for n in blocks:
        a,c,r=res[n]
        f=lambda d: np.nanmean(list(d.values()))
        print("  %-28s %8.3f %9.3f %9.3f" % (n,f(a),f(c),np.nanmedian(list(r.values()))))
    print("\n  paired configuration-bootstrap 95%% CIs")
    pairs=[("full10","structural_dynamic"),("full10","structural"),
           ("full10","dynamic")]+[("full10",n) for n in LOO]
    out={}
    for a,b in pairs:
        d,ci,n=paired_bootstrap(res[a][0],res[b][0])
        out["%s - %s"%(a,b)]={"delta":d,"ci95":list(ci),"n_paired":n}
        star="  <-- headline" if b=="structural_dynamic" else ""
        print("    %-34s %+0.3f [%+0.3f, %+0.3f] n=%d%s"%("%s - %s"%(a,b),d,ci[0],ci[1],n,star))
    Path(REPO/"data"/("contribution_%s.json"%scenario)).write_text(json.dumps(
        {"scenario":scenario,
         "blocks":{n:{"coverage_auc":res[n][0],"coverage_at_20":res[n][1],
                      "within_spearman":res[n][2]} for n in blocks},
         "paired_ci":out},indent=2,default=str))
    print("\nwrote data/contribution_%s.json"%scenario)

if __name__=="__main__": main(*(sys.argv[1:] or []))
