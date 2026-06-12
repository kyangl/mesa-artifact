"""
Compute structural edge saliency for all topologies (no LLM calls).
Prints a per-topology saliency report and the top-3 most vulnerable edges.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topology.builder import load_all_topologies, print_topology_summary
from src.saliency.structural import (
    compute_es_struct, compute_spectral_features, print_saliency_report
)


def main():
    topologies = load_all_topologies(str(ROOT / "config/topologies"))
    print_topology_summary(topologies)

    all_saliency = {}
    for name, data in topologies.items():
        G = data["graph"]
        features = compute_es_struct(G)
        all_saliency[name] = features
        print_saliency_report(features, name)

        spectral = compute_spectral_features(G)
        print(f"\n  Spectral: algebraic_connectivity={spectral['algebraic_connectivity']:.3f}, "
              f"spectral_gap={spectral['spectral_gap']:.3f}")

    print("\n" + "=" * 60)
    print("Top 3 most vulnerable edges per topology")
    print("=" * 60)
    for name, features in all_saliency.items():
        top = sorted(features.items(), key=lambda x: x[1].get("es_struct", 0),
                     reverse=True)[:3]
        print(f"\n{name}:")
        for (u, v), feat in top:
            print(f"  {feat.get('label', f'{u}->{v}')}: "
                  f"ES={feat.get('es_struct', 0):.3f} "
                  f"(betw={feat.get('betweenness_centrality', 0):.3f}, "
                  f"bridge={'Y' if feat.get('is_bridge', 0) > 0 else 'N'}, "
                  f"bottleneck={feat.get('information_bottleneck', 0):.3f})")


if __name__ == "__main__":
    main()
