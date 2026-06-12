"""
LaTeX table: Spearman rho for ES_struct vs ES_combined against corrected ASR.

Input:  results/tables/pooled_rho_per_scenario.csv (original_6_topos rows).
Output: results/tables/mesa_rho_table.tex
"""

import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"

ALL_MODELS = ["llama3.1:8b", "qwen3.5:9b", "qwen3.5:27b", "gemma4:e4b", "gemma4:26b"]
MODEL_LABEL = {
    "llama3.1:8b": "Llama-8B",
    "qwen3.5:9b": "Qwen-9B",
    "qwen3.5:27b": "Qwen-27B",
    "gemma4:e4b": "Gemma-E4B",
    "gemma4:26b": "Gemma-26B",
}
RELIABLE = {"qwen3.5:9b", "qwen3.5:27b", "gemma4:e4b"}
SCENS = [("customer_service", "Customer Service"),
         ("software_engineering", "Software Engineering")]

data = {}        # (scenario, model) -> (rho_struct, rho_combined)
n_edges_map = {}
with open(TABLES / "pooled_rho_per_scenario.csv") as f:
    for r in csv.DictReader(f):
        if r["topology_set"] != "original_6_topos":
            continue
        try:
            data[(r["scenario"], r["model"])] = (float(r["rho_struct_pooled"]),
                                                 float(r["rho_combined_pooled"]))
            n_edges_map[(r["scenario"], r["model"])] = int(r["n_edges"])
        except (ValueError, KeyError):
            continue

print("Data coverage (original 6 topos):")
for sc, sc_label in SCENS:
    present = [m for m in ALL_MODELS if (sc, m) in data]
    edges = [n_edges_map.get((sc, m), "?") for m in present]
    print(f"  {sc_label:25s}: {len(present)} models, edges per model: {edges}")

helps = hurts = neutral = 0
for sc, _ in SCENS:
    for m in ALL_MODELS:
        if (sc, m) not in data:
            continue
        rs, rc = data[(sc, m)]
        if rc - rs > 0.005:
            helps += 1
        elif rc - rs < -0.005:
            hurts += 1
        else:
            neutral += 1
print(f"\nCS+SE: dynamic features help {helps}, hurt {hurts}, neutral {neutral} "
      f"out of {helps + hurts + neutral} (model, scenario) pairs")


def fmt_rho(rho, bold=False):
    if np.isnan(rho):
        return "---"
    s = f"{rho:+.3f}"
    return f"\\textbf{{{s}}}" if bold else s


def rel_cell(vals, bold=False):
    if not vals:
        return "---"
    m = np.mean(vals)
    s = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    body = f"{m:+.3f}" + (f"{{\\tiny{{$\\pm${s:.3f}}}}}" if s >= 0.0005 else "")
    return f"\\textbf{{{body}}}" if bold else body


lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\small",
    (r"\caption{Spearman $\rho$ between MESA scoring variants and corrected ASR "
     r"(original six topologies, per-edge pooled across tasks). "
     r"\textbf{ES$_\text{struct}$}: signed rank-sum of the six static features "
     r"($+$Betweenness, $+$Info\,Bottleneck, $+$Bridge, "
     r"$-$Endpoint centrality, $-$Source degree, $-$Target degree). "
     r"\textbf{ES$_\text{combined}$}: ES$_\text{struct}$ plus the rank-sums of "
     r"Ablation\,$\Delta$ and Perturbation\,$\Delta$. "
     r"Bold = higher $\rho$ per row. "
     r"Debate excluded (Llama sign-reversal; Qwen-27B absent).}"),
    r"\label{tab:mesa_rho}",
    r"\begin{tabular}{lcc}",
    r"\toprule",
    r"Model & ES$_\text{struct}$ & ES$_\text{combined}$ \\",
    r"\midrule",
]

for sc, sc_label in SCENS:
    lines.append(r"\multicolumn{3}{l}{\textit{" + sc_label + r"}} \\")
    lines.append(r"\cmidrule(lr){1-3}")

    rel_struct, rel_combined = [], []
    for model in ALL_MODELS:
        if (sc, model) not in data:
            lines.append(MODEL_LABEL[model] + r" & --- & --- \\")
            continue
        rs, rc = data[(sc, model)]
        lines.append(" & ".join([MODEL_LABEL[model],
                                 fmt_rho(rs, bold=rs > rc),
                                 fmt_rho(rc, bold=rc >= rs)]) + r" \\")
        if model in RELIABLE:
            rel_struct.append(rs)
            rel_combined.append(rc)

    m_s = np.mean(rel_struct) if rel_struct else np.nan
    m_c = np.mean(rel_combined) if rel_combined else np.nan
    lines.append(" & ".join([
        r"\textit{Mean (reliable)}",
        rel_cell(rel_struct, bold=np.isfinite(m_s) and (not np.isfinite(m_c) or m_s > m_c)),
        rel_cell(rel_combined, bold=np.isfinite(m_c) and (not np.isfinite(m_s) or m_c >= m_s)),
    ]) + r" \\")

    if sc != SCENS[-1][0]:
        lines.append(r"\midrule")

lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

out = TABLES / "mesa_rho_table.tex"
out.write_text("\n".join(lines) + "\n")
print(f"\nSaved: {out}")
