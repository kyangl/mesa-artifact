"""Plot the control-adjusted security-utility frontier.

Raw and matched-control-adjusted prevention are shown for MESA, random, and
monitor-all policies using the frozen eight-feature allocation.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt  # noqa: E402
from analysis.plots import style  # noqa: E402

SRC = REPO / "data" / "control_adjusted_prevention.json"
STEM = "fig_security_utility"
MESA = "mesa_local_crossfitted_k"
RAND = "random_k"
ALL = "monitor_all"


def main():
    style.apply()
    d = json.loads(SRC.read_text())
    assert d["feature_block"] == "structural_dynamic"
    assert not d.get("INCOMPLETE")
    tab = {(r["policy"], round(r["budget"], 2)): r for r in d["table"]}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.3))

    # ---- left: frontier, utility against adjusted prevention -------------
    series = [("mesa", [(MESA, 0.20, "20%"), (MESA, 0.40, "40%")]),
              ("random", [(RAND, 0.20, "20%"), (RAND, 0.40, "40%")]),
              ("monitor_all", [(ALL, 1.00, "100%")])]
    for role, pts in series:
        r = style.ROLE[role]
        xs = [tab[(p, b)]["clean_utility_retained"] for p, b, _ in pts]
        ys = [tab[(p, b)]["attributable_prevention"] for p, b, _ in pts]
        ax.plot(xs, ys, color=r["color"], marker=r["marker"], ls=r["ls"],
                label=r["label"])
        for (p, b, lab), x, y in zip(pts, xs, ys):
            c = tab[(p, b)]["attributable_ci"]
            ax.errorbar(x, y, yerr=[[y - c["lo"]], [c["hi"] - y]],
                        color=r["color"], elinewidth=1.2, capsize=2.5, ls="none")
            ax.annotate(lab, xy=(x, y), xytext=(4, 5),
                        textcoords="offset points", fontsize=7,
                        color=style.MUTED)
    ax.set_xlabel("Clean utility retained")
    ax.set_ylabel("Control-adjusted prevention")
    ax.set_title("Frontier (n=%d configurations)" % d["n_configurations"])
    # Utility increases from left to right.
    ax.legend(loc="upper left")

    # ---- right: raw against adjusted, per policy -------------------------
    labels, raws, adjs, cols = [], [], [], []
    for role, p, b, lab in (("mesa", MESA, 0.20, "MESA 20%"),
                            ("mesa", MESA, 0.40, "MESA 40%"),
                            ("random", RAND, 0.20, "Random 20%"),
                            ("random", RAND, 0.40, "Random 40%"),
                            ("monitor_all", ALL, 1.00, "Monitor all")):
        r = tab[(p, b)]
        labels.append(lab)
        raws.append(r["raw_prevention"])
        adjs.append(r["attributable_prevention"])
        cols.append(style.ROLE[role]["color"])
    y = range(len(labels))
    for i, (rw, aj, c) in enumerate(zip(raws, adjs, cols)):
        bx.plot([rw, aj], [i, i], color=c, lw=1.4, alpha=0.55, zorder=1)
        bx.scatter(rw, i, facecolors="white", edgecolors=c, s=34, zorder=2,
                   linewidths=1.4)
        bx.scatter(aj, i, color=c, s=34, zorder=3)
    bx.set_yticks(list(y))
    bx.set_yticklabels(labels)
    bx.invert_yaxis()
    bx.set_xlabel("Prevention")
    bx.set_title("Raw (hollow) → control-adjusted (filled)")
    bx.set_xlim(min(raws + adjs) - 0.045, max(raws + adjs) + 0.035)

    style.titled(fig, "MESA at a 20% budget reaches blanket monitoring's "
                      "prevention at a fifth of the coverage")
    style.narrative(fig,
                    "The matched control flips %.1f%% of outcomes with NOTHING "
                    "withheld; the gap between hollow and filled is that noise "
                    "being removed." % (100 * d["control_flip_rate"]),
                    y=-0.06)
    style.save(fig, style.stem_for(STEM))

    print("SECURITY-UTILITY (control-adjusted, n=%d)" % d["n_configurations"])
    for lab, rw, aj in zip(labels, raws, adjs):
        print("  %-14s raw %.3f -> adjusted %.3f" % (lab, rw, aj))
    for s in d["mesa_vs_random_adjusted"]:
        print("  MESA - random @%.0f%%: %+.3f [%+.3f, %+.3f] (n=%d)"
              % (100 * s["budget"], s["mean_diff"], s["ci"]["lo"],
                 s["ci"]["hi"], s["n"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
