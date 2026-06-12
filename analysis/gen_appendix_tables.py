"""
Two appendix LaTeX tables, both derived from results/tables/per_edge_master_table.csv:

  concentration_table.tex   vulnerability concentration (top-k% ASR-success share)
                            by topology and scenario.
  clean_accuracy_table.tex  clean baseline accuracy by topology, model, scenario.

Output: results/tables/
"""

import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
MASTER = TABLES / "per_edge_master_table.csv"

TOPO_ORDER = ["sequential", "decentralized", "centralized",
              "hierarchical", "hybrid", "mesh"]
TOPO_FULL = {
    "sequential": "Sequential", "decentralized": "Decentralized",
    "centralized": "Centralized", "hierarchical": "Hierarchical",
    "hybrid": "Hybrid", "mesh": "Mesh",
}
MODEL_ORDER = ["llama3.1:8b", "qwen3.5:9b", "qwen3.5:27b", "gemma4:e4b", "gemma4:26b"]
MODEL_LABEL = {
    "llama3.1:8b": "Llama-8B", "qwen3.5:9b": "Qwen-9B", "qwen3.5:27b": "Qwen-27B",
    "gemma4:e4b": "Gemma-E4B", "gemma4:26b": "Gemma-26B",
}
SC_ORDER = ["customer_service", "software_engineering", "homogeneous_debate"]
SC_LABEL = {
    "customer_service": "Customer Service",
    "software_engineering": "Software Engineering",
    "homogeneous_debate": "Debate",
}
RELIABLE = {"qwen3.5:9b", "qwen3.5:27b", "gemma4:e4b"}

# Load master table
rows = []
with open(MASTER) as f:
    for r in csv.DictReader(f):
        try:
            r["corrected_asr"] = float(r["corrected_asr"])
        except (ValueError, KeyError):
            continue
        try:
            r["clean_acc_topo"] = float(r["clean_acc_topo"])
        except (ValueError, KeyError):
            r["clean_acc_topo"] = None
        rows.append(r)


def top_k_share(asr_values, k_pct):
    """Fraction of total corrected-ASR mass held by the top k% of edges."""
    asr = np.sort(np.asarray(asr_values, dtype=float))[::-1]
    n = len(asr)
    if n == 0 or asr.sum() == 0:
        return 0.0
    k = max(1, int(round(k_pct * n)))
    return float(asr[:k].sum() / asr.sum())


# Table 1: vulnerability concentration
by_cell = defaultdict(list)
for r in rows:
    by_cell[(r["scenario"], r["model"], r["topology"])].append(r["corrected_asr"])

share = defaultdict(dict)   # (scenario, topology) -> {model: (top10, top20)}
for (sc, m, t), asrs in by_cell.items():
    if len(asrs) < 4:
        continue
    share[(sc, t)][m] = (top_k_share(asrs, 0.10), top_k_share(asrs, 0.20))


def fmt_conc(vals):
    if not vals:
        return "---"
    m = np.mean(vals)
    s = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
    pct = lambda v: f"{v * 100:.0f}\\%"
    return f"{pct(m)} {{\\tiny{{$\\pm${pct(s)}}}}}" if s >= 0.005 else pct(m)


def conc_vals(idx, topo, scen):
    return [v[idx] for m, v in share.get((scen, topo), {}).items() if m in RELIABLE]


SCENS_TBL = [("customer_service", "CS"), ("software_engineering", "SE"),
             ("homogeneous_debate", "Debate")]

lines = [
    r"\begin{table}[t]", r"\centering", r"\small",
    (r"\caption{Vulnerability concentration by topology and scenario. "
     r"Top-$k$\% share is the fraction of total corrected-ASR attack mass "
     r"held by the top $k$\% of edges. "
     r"Values are mean $\pm$ SEM over reliable LLMs (Qwen-9B, Qwen-27B, Gemma-E4B). "
     r"Debate coverage is sparser (1--2 models per configuration).}"),
    r"\label{tab:concentration}",
    r"\begin{tabular}{l" + "cc" * len(SCENS_TBL) + r"}",
    r"\toprule",
]

header1 = [r"\multirow{2}{*}{Topology}"]
for _, sc_short in SCENS_TBL:
    header1.append(r"\multicolumn{2}{c}{" + sc_short + r"}")
lines.append(" & ".join(header1) + r" \\")

col = 2
cmidrules = []
for _ in SCENS_TBL:
    cmidrules.append(rf"\cmidrule(lr){{{col}-{col + 1}}}")
    col += 2
lines.append("".join(cmidrules))

header2 = [""]
for _ in SCENS_TBL:
    header2 += [r"Top-10\%", r"Top-20\%"]
lines.append(" & ".join(header2) + r" \\")
lines.append(r"\midrule")

for topo in TOPO_ORDER:
    row = [TOPO_FULL[topo]]
    for sc_key, _ in SCENS_TBL:
        row.append(fmt_conc(conc_vals(0, topo, sc_key)))
        row.append(fmt_conc(conc_vals(1, topo, sc_key)))
    lines.append(" & ".join(row) + r" \\")

lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TABLES / "concentration_table.tex").write_text("\n".join(lines) + "\n")
print(f"Saved: {TABLES / 'concentration_table.tex'}")


# Table 2: clean baseline accuracy (clean_acc_topo is constant per config; dedup)
clean = {}
for r in rows:
    if r["topology"] not in TOPO_ORDER or r["clean_acc_topo"] is None:
        continue
    clean[(r["scenario"], r["model"], r["topology"])] = r["clean_acc_topo"]


def fmt_acc(val):
    return "---" if val is None else f"{val * 100:.0f}\\%"


n_models = len(MODEL_ORDER)
lines2 = [
    r"\begin{table}[t]", r"\centering", r"\small",
    (r"\caption{Clean baseline accuracy (decision accuracy, \%) by topology, model, "
     r"and scenario. Each cell is the mean over all tasks in that configuration. "
     r"Dash (---) indicates no data collected for that configuration.}"),
    r"\label{tab:clean_accuracy}",
    r"\begin{tabular}{l" + "c" * n_models + r"}",
    r"\toprule",
    r"Topology & " + " & ".join(MODEL_LABEL[m] for m in MODEL_ORDER) + r" \\",
    r"\midrule",
]

for sc_idx, sc_key in enumerate(SC_ORDER):
    lines2.append(r"\multicolumn{" + str(n_models + 1) + r"}{l}{\textit{"
                  + SC_LABEL[sc_key] + r"}} \\")
    lines2.append(r"\cmidrule(lr){1-" + str(n_models + 1) + r"}")
    for topo in TOPO_ORDER:
        row = [TOPO_FULL[topo]]
        for model in MODEL_ORDER:
            row.append(fmt_acc(clean.get((sc_key, model, topo))))
        lines2.append(" & ".join(row) + r" \\")
    if sc_idx < len(SC_ORDER) - 1:
        lines2.append(r"\midrule")

lines2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TABLES / "clean_accuracy_table.tex").write_text("\n".join(lines2) + "\n")
print(f"Saved: {TABLES / 'clean_accuracy_table.tex'}")
