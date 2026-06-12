"""
Structural edge saliency metrics (ES_struct).
These are computed from graph structure alone — no LLM calls needed.
"""

import networkx as nx
import numpy as np
from scipy.stats import rankdata

# ES_struct = signed rank-sum of these six features (positive then negative).
ES_STRUCT_POS = ["betweenness_centrality", "information_bottleneck", "is_bridge"]
ES_STRUCT_NEG = ["endpoint_centrality_max", "source_degree_centrality",
                 "target_degree_centrality"]


def compute_es_struct(G: nx.DiGraph) -> dict:
    """Compute MESA structural saliency (ES_struct) for all edges.

    ES_struct is the signed rank-sum of six structural features, each rank
    normalized to (0, 1] within this graph (rank / n_edges):

        ES_struct(e) = +r(betweenness) +r(info_bottleneck) +r(is_bridge)
                       -r(endpoint_centrality) -r(source_degree) -r(target_degree)

    Higher ES_struct = predicted more vulnerable. Individual features (and their
    min-max normalized variants) are preserved for downstream analysis.
    """
    features = compute_all_structural_features(G)

    edge_list = list(features.keys())
    if not edge_list:
        return {}

    # Per-feature min-max normalization within this graph (kept for analysis).
    for fname in ES_STRUCT_POS + ES_STRUCT_NEG + ["endpoint_centrality_product"]:
        vals = [features[e].get(fname, 0) for e in edge_list]
        min_val, max_val = min(vals), max(vals)
        rng = max_val - min_val
        for e in edge_list:
            raw = features[e].get(fname, 0)
            features[e][f"{fname}_norm"] = (raw - min_val) / rng if rng > 0 else 0

    # ES_struct: signed rank-sum, normalized by n_edges (divide once at the end
    # so that exact rank ties are preserved).
    n = len(edge_list)
    scores = {e: 0.0 for e in edge_list}
    for fname, sign in [(f, 1) for f in ES_STRUCT_POS] + [(f, -1) for f in ES_STRUCT_NEG]:
        vals = np.array([features[e].get(fname, 0.0) for e in edge_list])
        if np.std(vals) == 0:
            continue
        ranks = rankdata(vals)
        for e, r in zip(edge_list, ranks):
            scores[e] += sign * r

    for e in edge_list:
        features[e]["es_struct"] = scores[e] / n

    return features


def compute_all_structural_features(G: nx.DiGraph) -> dict:
    """Compute all structural features for each edge."""
    features = {}

    # Edge betweenness centrality
    ebc = nx.edge_betweenness_centrality(G)

    # Node centralities
    degree_cent = nx.degree_centrality(G)
    try:
        closeness_cent = nx.closeness_centrality(G)
    except Exception:
        closeness_cent = {n: 0 for n in G.nodes()}

    # Check weak connectivity
    is_connected = nx.is_weakly_connected(G)

    for u, v, data in G.edges(data=True):
        feat = {}

        # Edge betweenness
        feat["betweenness_centrality"] = ebc.get((u, v), 0)

        # Bridge detection
        if is_connected:
            G_copy = G.copy()
            G_copy.remove_edge(u, v)
            feat["is_bridge"] = 1.0 if not nx.is_weakly_connected(G_copy) else 0.0
        else:
            feat["is_bridge"] = 0.0

        # Endpoint centralities
        feat["source_degree_centrality"] = degree_cent[u]
        feat["target_degree_centrality"] = degree_cent[v]
        feat["endpoint_centrality_max"] = max(degree_cent[u], degree_cent[v])
        feat["endpoint_centrality_product"] = degree_cent[u] * degree_cent[v]

        # Source/target degree
        feat["source_out_degree"] = G.out_degree(u)
        feat["source_in_degree"] = G.in_degree(u)
        feat["target_out_degree"] = G.out_degree(v)
        feat["target_in_degree"] = G.in_degree(v)

        # Information bottleneck: how hard is u -> v to reach if this edge is
        # removed. Longer detours -> higher score; no path -> 1.0 (only route).
        G_copy = G.copy()
        G_copy.remove_edge(u, v)
        try:
            alt_path_len = nx.shortest_path_length(G_copy, u, v)
            feat["information_bottleneck"] = 1.0 - 1.0 / alt_path_len
        except nx.NetworkXNoPath:
            feat["information_bottleneck"] = 1.0

        # Edge metadata
        feat["label"] = data.get("label", f"{u}-{v}")
        feat["edge_type"] = data.get("edge_type", "default")

        features[(u, v)] = feat

    return features


def compute_spectral_features(G: nx.DiGraph) -> dict:
    """Compute graph-level spectral features.

    Returns dict with algebraic connectivity, spectral gap, etc.
    """
    G_undirected = G.to_undirected()
    L = nx.laplacian_matrix(G_undirected).toarray().astype(float)
    eigenvalues = np.sort(np.linalg.eigvalsh(L))

    result = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "eigenvalues": eigenvalues.tolist(),
    }

    if len(eigenvalues) >= 2:
        result["algebraic_connectivity"] = float(eigenvalues[1])  # Fiedler value
        result["spectral_gap"] = float(eigenvalues[1] - eigenvalues[0])
    else:
        result["algebraic_connectivity"] = 0
        result["spectral_gap"] = 0

    if len(eigenvalues) >= 3:
        result["spectral_radius"] = float(eigenvalues[-1])

    return result


def print_saliency_report(features: dict, topology_name: str = ""):
    """Print a formatted report of edge saliency scores."""
    print(f"\n{'=' * 60}")
    print(f"Edge Saliency Report: {topology_name}")
    print(f"{'=' * 60}")

    # Sort by saliency score
    sorted_edges = sorted(
        features.items(),
        key=lambda x: x[1].get("es_struct", 0),
        reverse=True,
    )

    print(f"\n{'Edge':<35} {'ES_struct':<10} {'Betw.':<8} {'Bridge':<8} {'Bottleneck':<10}")
    print("-" * 75)
    for (u, v), feat in sorted_edges:
        label = feat.get("label", f"{u}->{v}")
        print(f"{label:<35} {feat.get('es_struct', 0):<10.3f} "
              f"{feat.get('betweenness_centrality', 0):<8.3f} "
              f"{feat.get('is_bridge', 0):<8.0f} "
              f"{feat.get('information_bottleneck', 0):<10.3f}")
