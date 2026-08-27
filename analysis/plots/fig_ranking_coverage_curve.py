"""Plot attack coverage against monitoring budget.

Produces customer-service and two-domain versions from the frozen aggregate.
Series are MESA, exact random allocation, and the outcome-aware oracle bound.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt      # noqa: E402
from analysis.plots import style     # noqa: E402

SRC = REPO / "data" / "canonical_ranking_curve.json"
CAPTIONS = REPO / "data" / "CAPTIONS_ranking_coverage_curve.md"
STEM_CS = "fig_ranking_coverage_curve"
STEM_TWO = "fig_ranking_coverage_curve_two_domain"
DOMAINS = ("customer_service", "software_engineering")

# Bound at the back, method on top.
DRAW_ORDER = ("oracle", "random", "mesa")
# Method, baseline, bound -- the order the argument runs in.
LEGEND_ORDER = ("mesa", "random", "oracle")


def curve(block, key):
    return [p[key] for p in block["curve"]]


def draw(ax, block):
    """One panel: the complete 10-100% trajectory, and nothing else."""
    x = [round(p["budget"] * 100) for p in block["curve"]]
    handles = {}
    for role in DRAW_ORDER:
        r = style.ROLE[role]
        line, = ax.plot(x, curve(block, role), color=r["color"],
                        marker=r["marker"], ls=r["ls"], label=r["label"],
                        zorder=3, clip_on=False)
        handles[role] = line

    ax.set_xticks(x)
    ax.set_xticklabels(["%d" % b for b in x])
    ax.set_xlim(4, 104)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Monitoring budget (% of a configuration's edges)")
    # The light grid comes from analysis/plots/style.py and is deliberately
    # kept: it is what lets a reader read a value off the trajectory at a
    # budget that carries no annotation.
    return [handles[r] for r in LEGEND_ORDER], \
           [style.ROLE[r]["label"] for r in LEGEND_ORDER]


def figure_cs(d):
    m = d["scenarios"]["customer_service"]["macro"]
    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    h, l = draw(ax, m)
    ax.set_ylabel("Attack coverage at budget")
    ax.legend(h, l, loc="lower right", ncol=1)
    style.titled(fig, "Ranking coverage across the full budget sweep")
    return style.save(fig, style.stem_for(STEM_CS)), m


def figure_two_domain(d):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    blocks, h, l = {}, None, None
    for ax, dom in zip(axes, DOMAINS):
        m = d["scenarios"][dom]["macro"]
        blocks[dom] = m
        h, l = draw(ax, m)
        # Concise heading only: the panels must be tellable apart at a glance,
        # and everything else about them belongs to the caption.
        ax.set_title(style.DOMAIN_TITLE[dom])
    axes[0].set_ylabel("Attack coverage at budget")
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    style.titled(fig, "Ranking coverage: separated by domain, never averaged "
                      "across them")
    return style.save(fig, style.stem_for(STEM_TWO)), blocks


# ------------------------------------------------------------------ captions

def _pp(block, budget):
    """The paired difference at a budget, in percentage points."""
    for p in block["curve"]:
        if abs(p["budget"] - budget) < 1e-12:
            d = p["mesa_minus_random"]
            return (100 * d["mean"], 100 * d["lo"], 100 * d["hi"],
                    p["signflip"]["p_two_sided"])
    raise SystemExit("budget %s missing" % budget)


def _inference(block):
    a = _pp(block, 0.10)
    b = _pp(block, 0.20)
    return ("At 10%% and 20%% budgets, MESA improves coverage over random by "
            "%.1f and %.1f percentage points, respectively (paired over "
            "configurations; 95%% CI [%.1f, %.1f] and [%.1f, %.1f]; exact "
            "sign-flip p = %.5f and %.5f); other budgets describe the "
            "trajectory."
            % (a[0], b[0], a[1], a[2], b[1], b[2], a[3], b[3]))


def _inference_compact(block):
    """Same numbers, one clause per budget -- for the two-domain caption.

    The full sentence twice over reads as boilerplate and buries the contrast
    between the domains, which is the only reason the panels are side by side.
    """
    out = []
    for b in (0.10, 0.20):
        m, lo, hi, p = _pp(block, b)
        out.append("%d%% %+.1f pp [%+.1f, %+.1f], p = %.5f"
                   % (round(b * 100), m, lo, hi, p))
    return "; ".join(out)


def _provenance(d):
    p = d["scenarios"]["customer_service"]["routing_provenance"]
    q = d["scenarios"]["software_engineering"]["routing_provenance"]
    return ("Routing provenance: %d of %d customer-service edge rows and %d of "
            "%d software-engineering rows are original measurements that were "
            "NOT re-run under routing-v1. Every one comes from an audited "
            "routing-unchanged configuration -- %s -- protocols the routing "
            "repair provably did not touch; the repaired protocols (%s) carry "
            "routing-v1 rows only. This is asserted at build time, and no "
            "claim is made that every underlying row was newly re-executed."
            % (p["n_rows_reused_not_rerun"], p["n_edge_rows"],
               q["n_rows_reused_not_rerun"], q["n_edge_rows"],
               ", ".join(p["topologies_reused"]), "hierarchical, hybrid"))


def captions(d, cs, two):
    n = cs["n_configurations"]
    main = ("Attack coverage as a function of monitoring budget in customer "
            "service, across %d configurations spanning three models and five "
            "topologies. MESA-Local uses the frozen, cross-fitted feature "
            "score; random allocation is recomputed for each integer edge "
            "budget, and the oracle is an empirical upper bound. %s"
            % (n, _inference(cs)))

    appendix = ("Attack coverage as a function of monitoring budget, per "
                "domain: customer service and software engineering, %d "
                "configurations each, shown side by side and never averaged "
                "into a single number. Scores, budgets and baselines are as in "
                "the main figure. MESA-Local minus random allocation, paired "
                "over configurations, at the two predeclared budgets — "
                "customer service %s; software engineering %s. Other budgets "
                "describe the trajectory. Software engineering tracks random "
                "across the whole sweep while the oracle stays well above "
                "both, so the null there is a ranking failure rather than "
                "absent headroom."
                % (two["customer_service"]["n_configurations"],
                   _inference_compact(two["customer_service"]),
                   _inference_compact(two["software_engineering"])))

    method = ("Attack coverage is the share of a configuration's attack mass "
              "falling on its top-k monitored edges, with k = ceil(b|E|) "
              "computed inside each configuration and tie credit taken in "
              "expectation. Scores come from the frozen eight-feature "
              "MESA-Local block (%s); the A->B and B->A cross-fit directions "
              "are collapsed to one value per configuration before any "
              "averaging or resampling, and configurations are macro-averaged "
              "with equal weight. Random allocation is the exact expectation "
              "k/|E|. The oracle is a bound on what any ordering could reach, "
              "never a method. Inference is confined to the predeclared 10%% "
              "and 20%% budgets." % d["canonical_block"])

    L = ["# Caption text — ranking-coverage trajectory", "",
         "Generated by `python analysis/plots/fig_ranking_coverage_curve.py`. "
         "Every number here is read from "
         "`data/canonical_ranking_curve.json` at build time, so a caption "
         "cannot drift from the figure it describes.", "",
         "## Main paper", "",
         "**`figures/%s.png`** · `figures/pdf/%s.pdf`" % (STEM_CS, STEM_CS), "",
         "> %s" % main, "",
         "## Appendix", "",
         "**`figures/%s.png`** · `figures/pdf/%s.pdf`" % (STEM_TWO, STEM_TWO),
         "", "> %s" % appendix, "",
         "## Method note (for the figure's method paragraph or a footnote)", "",
         "> %s" % method, "",
         "## Provenance note (required disclosure)", "",
         "> %s" % _provenance(d), ""]
    CAPTIONS.parent.mkdir(parents=True, exist_ok=True)
    CAPTIONS.write_text("\n".join(L))
    print("  wrote %s" % CAPTIONS.relative_to(REPO))


def main():
    style.apply()
    d = json.loads(SRC.read_text())
    assert d["canonical_block"] == "structural_dynamic", d["canonical_block"]
    assert d["n_features"] == 8, d["n_features"]
    for dom in DOMAINS:
        res = d["scenarios"][dom]
        assert res["macro"]["n_configurations"] == 15, dom
        # Promotion does not relax the provenance disclosure.
        assert res["routing_provenance"]["topologies_reused"], dom
        assert not (set(res["routing_provenance"]["topologies_reused"])
                    - {"sequential", "centralized", "decentralized"}), dom
    for dom in DOMAINS:
        for p in d["scenarios"][dom]["macro"]["curve"]:
            if abs(p["budget"] - 1.0) < 1e-12:
                for k in ("mesa", "random", "oracle"):
                    assert abs(p[k] - 1.0) <= 1e-12, (dom, k, p[k])

    print("main-paper figure (customer service):")
    _p_cs, cs = figure_cs(d)
    print("appendix figure (both domains):")
    _p_two, two = figure_two_domain(d)
    captions(d, cs, two)

    print()
    print("RANKING COVERAGE TRAJECTORY (eight-feature, macro over 15 configs)")
    for dom in DOMAINS:
        m = d["scenarios"][dom]["macro"]
        print("  %s" % dom)
        for p in m["curve"]:
            extra = ""
            if p["predeclared"]:
                q = p["mesa_minus_random"]
                extra = "   d=%+.3f [%+.3f, %+.3f]" % (q["mean"], q["lo"],
                                                       q["hi"])
            print("    %4d%%  MESA %.3f  random %.3f  oracle %.3f%s"
                  % (round(p["budget"] * 100), p["mesa"], p["random"],
                     p["oracle"], extra))
    return 0


if __name__ == "__main__":
    sys.exit(main())
