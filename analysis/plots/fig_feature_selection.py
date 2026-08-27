"""Plot paired coverage-AUC contrasts for the feature freeze.

The figure compares eight versus six features and ten versus eight by domain,
using the predeclared simplest-within-one-SE rule.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt  # noqa: E402
from analysis.plots import style  # noqa: E402

SRC = REPO / "data" / "feature_freeze_decision.json"
STEM = "fig_feature_selection"


def main():
    style.apply()
    d = json.loads(SRC.read_text())
    scen = d["scenarios"]
    domains = ["customer_service", "software_engineering"]
    contrasts = [("auc|eight_minus_structural", "Eight − six\n(dynamic block)",
                  "mesa"),
                 ("auc|ten_minus_eight", "Ten − eight\n(F2/F3 extension)",
                  "extension")]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, dom in zip(axes, domains):
        c = scen[dom]["contrasts"]
        for i, (key, label, role) in enumerate(contrasts):
            v = c[key]
            style.errorbar_ci(ax, i, v["mean"], v["lo"], v["hi"], role,
                              label=None)
            # one SE, the unit the promotion rule is written in
            ax.errorbar(i, v["mean"], yerr=v["se"], color=style.ROLE[role]["color"],
                        ls="none", elinewidth=3.4, alpha=0.32, capsize=0,
                        zorder=2)
        style.zero_line(ax)
        ax.set_xticks(range(len(contrasts)))
        ax.set_xticklabels([c[1] for c in contrasts])
        ax.set_xlim(-0.55, len(contrasts) - 0.45)
        ax.set_title("%s  (n=%d configurations)"
                     % (style.DOMAIN_TITLE[dom], scen[dom]["n_configurations"]))
    axes[0].set_ylabel("Δ coverage-AUC\n(paired, configuration-level)")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=style.ROLE["mesa"]["color"],
               marker=style.ROLE["mesa"]["marker"], ls="none",
               label="Eight − six: promoted"),
        Line2D([], [], color=style.ROLE["extension"]["color"],
               marker=style.ROLE["extension"]["marker"], ls="none",
               label="Ten − eight: rejected by the rule"),
        Line2D([], [], color=style.INK, lw=1.0, alpha=0.55, label="no difference"),
        Line2D([], [], color=style.MUTED, lw=3.4, alpha=0.32,
               label="±1 SE (the promotion unit)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.13))
    style.titled(fig, "Feature selection: the extension does not earn "
                      "promotion in either domain")
    style.save(fig, style.stem_for(STEM))
    print("FEATURE SELECTION")
    for dom in domains:
        c = scen[dom]["contrasts"]
        for key, label, _ in contrasts:
            v = c[key]
            print("  %-22s %-28s %+.3f [%+.3f, %+.3f]  SE %.3f"
                  % (dom, key.split("|")[1], v["mean"], v["lo"], v["hi"], v["se"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
