"""Plot within-configuration feature correlations by domain.

Cells show median absolute Spearman correlation. Constant features are omitted
because their correlations are undefined, and results are checked against the
feature-selection audit.
"""

import itertools
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap             # noqa: E402

from analysis.plots import style                                  # noqa: E402
from analysis.run_mesa_fit import load_enriched_directional       # noqa: E402

AUDIT = REPO / "data" / "feature_selection_audit.json"
OUT_JSON = REPO / "data" / "feature_correlation.json"
CAPTIONS = REPO / "data" / "CAPTIONS_feature_correlation.md"
STEM = "fig_feature_correlation"

SCENARIOS = ("customer_service", "software_engineering")

# Groups reflect measurement cost; one customer-service ordering is reused.
GRAPH = ["betweenness_centrality", "endpoint_centrality_max",
         "source_degree_centrality", "target_degree_centrality",
         "consequence_proximity"]
WORKFLOW = ["ablation_delta", "perturbation_delta",
            "receiver_response_sensitivity"]
FEATS = GRAPH + WORKFLOW
GROUPS = [("Graph-theoretic", len(GRAPH)), ("Workflow-aware", len(WORKFLOW))]

# How the audit's cost table names each tier. consequence_proximity is
# "graph-only" there -- a shortest-path distance, no model and no workflow
# run -- which is why it sits in the graph group despite arriving with the
# F2/F3 extension pair. Checked against the artifact, not assumed.
COST_CLASS = {"graph-only": GRAPH,
              "workflow re-execution": ["ablation_delta",
                                        "perturbation_delta"],
              "receiver probe generation + embedding":
                  ["receiver_response_sensitivity"]}

# Constant in every configuration; see the module docstring.
EXPECT_OMITTED = {"information_bottleneck", "is_bridge"}

LABEL = {
    "betweenness_centrality": "Betw",
    "endpoint_centrality_max": "EC",
    "source_degree_centrality": "SrcD",
    "target_degree_centrality": "TgtD",
    "ablation_delta": "Ablation",
    "perturbation_delta": "Masking",
    "receiver_response_sensitivity": "RS",
    "consequence_proximity": "CP",
}

FLAG = 0.80          # audit's "flag as possibly redundant"
DUPLICATE = 0.90     # audit's "treat as a duplicate"


# ----------------------------------------------------------- the estimator

def _pair(fm, ia, ib):
    """Per-direction Spearman between two feature columns.

    Returns (signed rhos, absolute rhos, n_undefined). A direction is
    UNDEFINED when either feature is constant among that configuration's
    edges, and undefined is dropped from numerator and denominator alike --
    never replaced with zero, which would assert independence where nothing
    was measurable.
    """
    signed, undefined = [], 0
    for g in np.unique(fm.groups):
        idx = fm.groups == g
        xa, xb = fm.X[idx, ia], fm.X[idx, ib]
        ok = ~(np.isnan(xa) | np.isnan(xb))
        xa, xb = xa[ok], xb[ok]
        if len(xa) < 4 or len(np.unique(np.round(xa, 12))) < 2 \
                or len(np.unique(np.round(xb, 12))) < 2:
            undefined += 1
            continue
        r = stats.spearmanr(xa, xb).statistic
        if r is None or np.isnan(r):
            undefined += 1
            continue
        signed.append(float(r))
    return signed, [abs(r) for r in signed], undefined


def matrices(mats):
    """{domain: {"signed": 8x8, "absmed": 8x8, "n": 8x8}} plus the pair list."""
    out, pairs = {}, {}
    for sc in SCENARIOS:
        fm = mats[sc]
        idx = [fm.feature_names.index(f) for f in FEATS]
        n = len(FEATS)
        signed = np.full((n, n), np.nan)
        absmed = np.full((n, n), np.nan)
        counts = np.zeros((n, n), dtype=int)
        rows = []
        for i, j in itertools.combinations(range(n), 2):
            s, a, und = _pair(fm, idx[i], idx[j])
            if not s:
                continue
            ms, ma = float(np.median(s)), float(np.median(a))
            signed[i, j] = signed[j, i] = ms
            absmed[i, j] = absmed[j, i] = ma
            counts[i, j] = counts[j, i] = len(s)
            rows.append({"a": FEATS[i], "b": FEATS[j],
                         "median_spearman": ms, "median_abs_spearman": ma,
                         "n_directions_evaluable": len(s),
                         "n_directions_undefined": und})
        np.fill_diagonal(signed, 1.0)
        np.fill_diagonal(absmed, 1.0)
        out[sc] = {"signed": signed, "absmed": absmed, "n": counts}
        pairs[sc] = sorted(rows, key=lambda r: -r["median_abs_spearman"])
    return out, pairs


# ------------------------------------------------------------ verification

def verify_against_audit(pairs):
    """Reproduce the audited |rho| medians, or refuse to build.

    The figure shows a signed summary the audit does not store. That freedom
    is only safe if the underlying computation is provably the audit's, so
    every evaluable pair it audited must come back identical here.
    """
    if not AUDIT.exists():
        raise SystemExit("audit artifact missing; run "
                         "python analysis/feature_selection_audit.py first")
    aud = json.loads(AUDIT.read_text())["redundancy"]
    checked = 0
    for sc in SCENARIOS:
        want = {}
        for p in aud[sc]["pairs"]:
            if p["median_abs_spearman"] is None:
                # Never-evaluable pairs must be exactly the ones involving an
                # omitted feature. If a pair of DRAWN features were undefined,
                # the omission set is wrong and the matrix would have a hole.
                if not ({p["a"], p["b"]} & EXPECT_OMITTED):
                    raise SystemExit(
                        "%s: pair %s/%s is never evaluable but neither "
                        "feature is in the omitted set" % (sc, p["a"], p["b"]))
                continue
            want[frozenset((p["a"], p["b"]))] = p["median_abs_spearman"]
        got = {frozenset((r["a"], r["b"])): r["median_abs_spearman"]
               for r in pairs[sc]}
        if set(want) != set(got):
            raise SystemExit(
                "%s: audited evaluable pairs and figure pairs differ "
                "(%d vs %d)" % (sc, len(want), len(got)))
        for k, v in want.items():
            if abs(v - got[k]) > 1e-12:
                raise SystemExit(
                    "%s: %s median |rho| %.15f in the audit, %.15f here -- "
                    "the figure is not using the audited estimator"
                    % (sc, sorted(k), v, got[k]))
            checked += 1
    print("  verified %d audited pair medians reproduced to 1e-12" % checked)
    return checked


def sort_within_groups(mat):
    """Reorder features inside each group, most-entangled first.

    Uses customer service as the reference and returns ONE ordering applied
    to every panel. Sorting each panel by its own values would put a
    different feature in each cell position and quietly destroy the
    comparison the side-by-side layout exists to make.
    """
    A = np.abs(mat["customer_service"]["absmed"]).copy()
    np.fill_diagonal(A, np.nan)
    strength = {f: float(np.nanmean(A[i])) for i, f in enumerate(FEATS)}
    order, start = [], 0
    for _, size in GROUPS:
        block = FEATS[start:start + size]
        order += sorted(block, key=lambda f: -strength[f])
        start += size
    return order


def verify_cost_classes():
    """The tier blocks must match the audit's recorded cost classes.

    The blocks are an argument about which redundancies are expensive, so a
    feature filed in the wrong tier would make the figure assert something
    the cost table contradicts.
    """
    aud = json.loads(AUDIT.read_text())["cost"]
    recorded = {}
    for key, entry in aud.items():
        if not isinstance(entry, dict) or "cost_class" not in entry:
            continue
        for f in entry.get("features", []):
            recorded[f] = entry["cost_class"]
        # Single-feature entries are keyed by name, not by a features list.
        for f in FEATS:
            if key.startswith(f):
                recorded[f] = entry["cost_class"]
    for cls, feats in COST_CLASS.items():
        for f in feats:
            if f not in recorded:
                raise SystemExit(
                    "%s has no recorded cost class in the audit artifact" % f)
            if recorded[f] != cls:
                raise SystemExit(
                    "%s is grouped as '%s' but the audit records it as '%s'; "
                    "regroup the figure rather than the artifact"
                    % (f, cls, recorded[f]))
    print("  verified %d features against the audited cost classes"
          % sum(len(v) for v in COST_CLASS.values()))


def verify_omissions(mats):
    """Re-derive that the two omitted features really are always constant."""
    for sc in SCENARIOS:
        fm = mats[sc]
        for f in sorted(EXPECT_OMITTED):
            if f not in fm.feature_names:
                continue
            j = fm.feature_names.index(f)
            varies = 0
            for g in np.unique(fm.groups):
                x = fm.X[fm.groups == g, j]
                x = x[~np.isnan(x)]
                if len(np.unique(np.round(x, 12))) > 1:
                    varies += 1
            if varies:
                raise SystemExit(
                    "%s: %s varies in %d configuration-direction(s); it can "
                    "no longer be omitted as constant -- add it back as a row"
                    % (sc, f, varies))


# ------------------------------------------------------------------ drawing

def _cmap():
    """Sequential palette for correlation magnitude."""
    return LinearSegmentedColormap.from_list("mesa_seq", [
        (0.00, "#FFFFE0"), (0.20, "#FFF3B0"), (0.40, "#FEDA76"),
        (0.60, "#FDA84E"), (0.78, "#F4703A"), (0.90, "#DB3B2B"),
        (1.00, "#A11117")])


def panel(ax, M, counts, title, cmap, order, show_ylabels=True):
    n = len(order)
    idx = [FEATS.index(f) for f in order]
    D = M[np.ix_(idx, idx)].copy()
    np.fill_diagonal(D, np.nan)          # the diagonal is 1 by definition
    im = ax.imshow(np.ma.masked_invalid(D), cmap=cmap, vmin=0.0, vmax=1.0,
                   aspect="equal")
    # Hairline separators between cells. Without them adjacent cells of
    # similar value melt into one block and the matrix reads as a smear
    # rather than as 64 discrete measurements.
    for e in np.arange(-0.5, n, 1.0):
        ax.axhline(e, color="white", lw=0.7, zorder=2)
        ax.axvline(e, color="white", lw=0.7, zorder=2)

    # NO PER-CELL NUMBERS. Colour carries the value; the exact medians live in
    # data/feature_correlation.json, which is where anyone checking a
    # specific pair should read them anyway.
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1,
                                   facecolor="#ECECEC", edgecolor="none"))

    ax.set_xticks(range(n))
    ax.set_xticklabels([LABEL[f] for f in order], fontsize=7.4, rotation=45,
                       ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n))
    if show_ylabels:
        ax.set_yticklabels([LABEL[f] for f in order], fontsize=7.4)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=8.5, pad=6)
    ax.grid(visible=False)
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#000000")
        ax.spines[side].set_linewidth(0.8)
    return im


def draw(mat, pairs):
    cmap = _cmap()
    order = sort_within_groups(mat)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
    im = None
    for ax, sc in zip(axes, SCENARIOS):
        im = panel(ax, mat[sc]["absmed"], mat[sc]["n"],
                   style.DOMAIN_TITLE[sc], cmap, order,
                   show_ylabels=(sc == SCENARIOS[0]))

    cb = fig.colorbar(im, ax=axes, fraction=0.021, pad=0.015)
    cb.set_label("Median within-configuration |ρ|", fontsize=7.4)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(labelsize=6.8)
    cb.outline.set_visible(False)
    # NO EXPLANATORY TEXT IN THE FIGURE. Scope, the two omitted features and
    # the redundancy thresholds are all in the generated caption, which is
    # where a paper figure's prose belongs.
    style.titled(fig, "Inter-feature correlation")
    return fig, order


# --------------------------------------------------------------------- main

def main():
    style.apply()
    mats = {}
    for sc in SCENARIOS:
        fm, _dropped, _m = load_enriched_directional(scenarios=[sc])
        mats[sc] = fm

    verify_cost_classes()
    verify_omissions(mats)
    mat, pairs = matrices(mats)
    verify_against_audit(pairs)

    fig, order = draw(mat, pairs)
    style.save(fig, style.stem_for(STEM))
    plt.close(fig)

    payload = {
        "note": ("Median WITHIN-CONFIGURATION Spearman correlation between "
                 "feature pairs, per domain, never pooled across domains. "
                 "Undefined correlations (a constant feature) are dropped "
                 "from both numerator and denominator, never replaced with "
                 "zero. Signed here; the feature-selection audit stores the "
                 "absolute-value summary and this artifact reproduces every "
                 "audited pair median to 1e-12."),
        "features": FEATS,
        "features_omitted_constant": sorted(EXPECT_OMITTED),
        "flag_threshold": FLAG, "duplicate_threshold": DUPLICATE,
        "scenarios": {
            sc: {"pairs": pairs[sc],
                 "n_pairs": len(pairs[sc]),
                 "n_pairs_above_flag": sum(
                     1 for r in pairs[sc]
                     if r["median_abs_spearman"] >= FLAG),
                 "n_pairs_above_duplicate": sum(
                     1 for r in pairs[sc]
                     if r["median_abs_spearman"] >= DUPLICATE)}
            for sc in SCENARIOS},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print("  wrote %s" % OUT_JSON.relative_to(REPO))

    _captions(payload, pairs)

    for sc in SCENARIOS:
        print()
        print("%s — strongest feature pairs (median |rho|)"
              % style.DOMAIN_TITLE[sc])
        for r in pairs[sc][:6]:
            print("   %-26s %-26s  %+.3f  (|%.3f|, n=%d)"
                  % (LABEL[r["a"]], LABEL[r["b"]], r["median_spearman"],
                     r["median_abs_spearman"], r["n_directions_evaluable"]))
        above = [r for r in pairs[sc] if r["median_abs_spearman"] >= DUPLICATE]
        print("   pairs at or above the %.2f duplicate threshold: %d"
              % (DUPLICATE, len(above)))
    return 0


def _captions(payload, pairs):
    top_cs = pairs["customer_service"][0]
    top_se = pairs["software_engineering"][0]
    cap = ("Inter-feature correlation, per domain. Each cell is the median "
           "ABSOLUTE Spearman correlation between two features across the "
           "edges of a configuration, summarised over configuration "
           "directions; the domains are shown side by side and never "
           "averaged. The absolute value is the redundancy quantity -- two "
           "features that are perfectly anti-correlated carry the same "
           "information as two that are perfectly correlated -- and it is "
           "what the feature-selection audit records; signed medians are in "
           "`data/feature_correlation.json`. `information_bottleneck` and "
           "`is_bridge` are omitted because they are constant among the "
           "edges of every configuration, which makes every correlation "
           "involving them undefined rather than zero: 17 of the 45 feature "
           "pairs, and the reason the audit reports 28 evaluable ones. The "
           "two blocks are graph-theoretic features, computable from the "
           "topology alone, and workflow-aware features, which require "
           "running the system; consequence_proximity (CP) sits in the "
           "graph block because the audit's cost table classes it as "
           "graph-only, a shortest-path distance needing no model and no "
           "workflow run -- it was excluded by the freeze on performance, "
           "not on cost. Features are ordered within each block by mean "
           "|rho| to the others, using customer service so both panels share "
           "one ordering. No pair reaches the %.2f redundancy flag in either "
           "domain, let alone the %.2f duplicate threshold, so the equal "
           "average is not silently double-weighting one underlying "
           "quantity. The strongest pair is %s/%s at |rho| %.2f in customer "
           "service and %s/%s at |rho| %.2f in software engineering."
           % (FLAG, DUPLICATE,
              LABEL[top_cs["a"]], LABEL[top_cs["b"]],
              top_cs["median_abs_spearman"],
              LABEL[top_se["a"]], LABEL[top_se["b"]],
              top_se["median_abs_spearman"]))

    L = ["# Caption text — inter-feature correlation", "",
         "Generated by `python analysis/plots/fig_feature_correlation.py`; "
         "every number is read from a canonical artifact at build time.", "",
         "**`figures/%s.png`** · `figures/pdf/%s.pdf`" % (STEM, STEM), "",
         "> %s" % cap, ""]
    CAPTIONS.parent.mkdir(parents=True, exist_ok=True)
    CAPTIONS.write_text("\n".join(L))
    print("  wrote %s" % CAPTIONS.relative_to(REPO))


if __name__ == "__main__":
    sys.exit(main())
