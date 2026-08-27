"""LangGraph parity rank scatter: MESA rank against LangGraph ASR rank.

Tests whether the edge ranking transfers across orchestration frameworks by
re-implementing the customer-service workflow in LangGraph with the same roles,
tasks and topologies. Reads data/langgraph_per_edge_asr.csv (one row per shared
edge-cell), so it runs without the raw pilot transcripts.

Output: figures/fig_langgraph_parity_rank_paper.{png,pdf}
"""

import csv
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import rankdata

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# The raw per-run JSONs are gitignored, so they exist only in a checkout that
# has actually run the pilot. These overrides let the figure be rebuilt from
# another checkout's results without copying half a gigabyte of transcripts,
# and without writing the output back into that checkout.
R       = Path(os.environ.get("MESA_PILOT_RESULTS", "results"))
FIG_DIR = Path(os.environ.get("MESA_PILOT_FIGDIR", "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── data loading ──────────────────────────────────────────────────────────────

def load_valid(prefix):
    out = []
    for p in sorted(R.glob(f"{prefix}_*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, list):
            continue
        for r in d:
            if (isinstance(r, dict) and "error" not in r
                    and isinstance(r.get("scores"), dict)
                    and r["scores"].get("decision_accuracy", -1) >= 0):
                out.append(r)
    return out


def compute_per_edge_asr(clean_pfx, attack_pfx, min_eligible=3):
    cl = {(r["model"], r["topology"], r["task_id"]): r["scores"]["decision_accuracy"]
          for r in load_valid(clean_pfx)}
    asr = defaultdict(lambda: {"e": 0, "f": 0})
    for r in load_valid(attack_pfx):
        e = tuple(r.get("attack_edge") or ())
        if not e:
            continue
        ck = (r["model"], r["topology"], r["task_id"])
        if cl.get(ck) == 1:
            asr[(r["model"], r["topology"], e)]["e"] += 1
            if r["scores"]["decision_accuracy"] == 0:
                asr[(r["model"], r["topology"], e)]["f"] += 1
    return {k: v["f"] / v["e"] for k, v in asr.items() if v["e"] >= min_eligible}


# Refresh from raw results when present; otherwise use the shipped aggregate.
AGG = Path(os.environ.get(
    "MESA_PILOT_AGG", "data/langgraph_per_edge_asr.csv"))


def write_aggregate(mas, lg, path):
    rows = sorted(set(mas) & set(lg))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "topology", "edge_src", "edge_dst",
                    "mesa_asr", "langgraph_asr"])
        for model, topo, edge in rows:
            src, dst = (list(edge) + ["", ""])[:2]
            w.writerow([model, topo, src, dst,
                        "%.10g" % mas[(model, topo, edge)],
                        "%.10g" % lg[(model, topo, edge)]])
    return len(rows)


def read_aggregate(path):
    mas, lg = {}, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            k = (r["model"], r["topology"], (r["edge_src"], r["edge_dst"]))
            mas[k] = float(r["mesa_asr"])
            lg[k] = float(r["langgraph_asr"])
    return mas, lg


mas_asr = compute_per_edge_asr("pilot_clean", "pilot_attack")
lg_asr  = compute_per_edge_asr("lg_pilot_clean", "lg_pilot_attack")
if set(mas_asr) & set(lg_asr):
    n = write_aggregate(mas_asr, lg_asr, AGG)
    print(f"Aggregate refreshed from raw results: {n} rows -> {AGG}")
elif AGG.exists():
    mas_asr, lg_asr = read_aggregate(AGG)
    print(f"Raw pilot results absent; built from aggregate {AGG}")
else:
    raise SystemExit(
        f"no pilot results under {R} and no aggregate at {AGG}; set "
        "MESA_PILOT_RESULTS to a checkout that has them")
shared  = set(mas_asr) & set(lg_asr)
print(f"Shared edges: {len(shared)}")

# ── palette / markers ─────────────────────────────────────────────────────────

PAPER_COLORS = {
    "centralized":   "#3A7EBF",
    "sequential":    "#4E9A56",
    "hierarchical":  "#C96A2A",
    "decentralized": "#7B5EA7",
    "hybrid":        "#B55050",
}
SHOW_TOPOS = ["centralized", "sequential", "hierarchical", "decentralized", "hybrid"]

MODEL_ORDER = [
    ("gemma4:e4b", "o", "Gemma-E4B"),
    ("qwen3.5:9b", "s", "Qwen-9B"),
]

MIN_PER_TOPO = 4


def plotted_ranks(model_slug):
    """Return within-topology ranks used by both the scatter and statistic."""
    per_topo, xs, ys = [], [], []
    for topo in SHOW_TOPOS:
        pts = [(mas_asr[k], lg_asr[k]) for k in shared
               if k[0] == model_slug and k[1] == topo]
        if len(pts) < MIN_PER_TOPO:
            continue
        a = np.array([q[0] for q in pts], dtype=float)
        b = np.array([q[1] for q in pts], dtype=float)
        n = len(a)
        rx = rankdata(a, method="average") / n
        ry = rankdata(b, method="average") / n
        per_topo.append((topo, rx, ry))
        xs += list(rx); ys += list(ry)
    return per_topo, np.array(xs), np.array(ys)


# ── per-model rho, on the points the figure actually shows ───────────────────
model_stats = {}
for model_slug, _, mlabel in MODEL_ORDER:
    _pt, mas_v, lg_v = plotted_ranks(model_slug)
    if len(mas_v) < 5:
        continue
    rho, pv = stats.spearmanr(mas_v, lg_v)
    n = len(mas_v)
    model_stats[model_slug] = dict(rho=rho, p=float(pv), n=n)
    print(f"  {mlabel}: rho={rho:+.3f}  p={pv:.3g}  (n={n})")

def _p_text(pv):
    """Format small asymptotic p-values as bounds."""
    if pv < 0.001:
        return r"$p<0.001$"
    if pv < 0.01:
        return r"$p<0.01$"
    return r"$p=%.3f$" % pv


TOPO_DISPLAY = {"centralized":"Hub","sequential":"Chain","hierarchical":"Hierarchy",
                "decentralized":"Ring","hybrid":"Hybrid"}

n_drawn = {}

# ── two-panel figure, one per model ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4),
                         gridspec_kw={"wspace": 0.28})

for ax, (model_slug, marker, mlabel) in zip(axes, MODEL_ORDER):
    per_topo, _x, _y = plotted_ranks(model_slug)
    n_drawn[model_slug] = 0
    for topo, rx, ry in per_topo:
        n_drawn[model_slug] += len(rx)
        ax.scatter(rx, ry,
                   facecolors=PAPER_COLORS[topo], edgecolors="white",
                   marker="o", s=48, alpha=0.85, linewidths=0.6, zorder=3,
                   label=TOPO_DISPLAY.get(topo, topo.capitalize()))

    ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=0.9, alpha=0.6, zorder=1)
    ax.set_xlim(0.0, 1.05); ax.set_ylim(0.0, 1.05)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.14, color="#999999", lw=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333"); sp.set_linewidth(0.9)
    ax.tick_params(labelsize=8, colors="#1a1a1a")
    ax.set_xlabel("MESA rank", fontsize=10, color="#1a1a1a")
    ax.set_ylabel("LangGraph ASR rank", fontsize=10, color="#1a1a1a")
    ax.set_title(mlabel, fontsize=11, color="#1a1a1a", pad=4)

    # ρ and p, lower-right corner, text left-aligned
    st = model_stats[model_slug]
    ax.text(0.62, 0.05,
            r"$\rho=%+.2f$" % st["rho"] + "\n" + _p_text(st["p"]),
            transform=ax.transAxes, ha="left", va="bottom",
            multialignment="left",
            fontsize=9, color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.92))

# single shared topology legend, no y=x, no model names
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06),
           ncol=5, fontsize=9, framealpha=0.93, edgecolor="#cccccc",
           handletextpad=0.4, columnspacing=1.0)

def assert_stats_match_scatter():
    """Every edge behind the quoted rho must be a dot on the panel."""
    for slug, st in model_stats.items():
        if st["n"] != n_drawn.get(slug):
            raise SystemExit(
                "%s: rho is computed on %d points but %d were drawn -- the "
                "annotation would describe data the reader cannot see"
                % (slug, st["n"], n_drawn.get(slug)))
    print("  rho and scatter agree on %s points"
          % ", ".join("%s=%d" % (s_, model_stats[s_]["n"])
                      for s_ in model_stats))


assert_stats_match_scatter()

fig.tight_layout(rect=[0, 0.06, 1, 1])
# Written into the tracked figure set alongside every other paper figure.
STEM = "fig_langgraph_parity_rank_paper"
CANON = Path(os.environ.get("MESA_FIGDIR", "figures"))
(CANON / "pdf").mkdir(parents=True, exist_ok=True)

targets = [FIG_DIR / f"{STEM}.pdf", FIG_DIR / f"{STEM}.png",
           CANON / "pdf" / f"{STEM}.pdf", CANON / f"{STEM}.png"]
for out in targets:
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved: {out}")
plt.close(fig)
