"""
Build NetworkX graphs from YAML topology configs.
Computes structural features for each edge.
"""

import yaml
import networkx as nx
from pathlib import Path
from typing import Optional


def load_topology(config_path: str) -> dict:
    """Load a topology YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_graph(config: dict) -> nx.DiGraph:
    """Build a NetworkX DiGraph from a topology config.

    Returns a directed graph. Bidirectional edges become two directed edges.
    Each node has attributes from the agent config.
    Each edge has attributes from the edge config.
    """
    G = nx.DiGraph()

    for agent in config["agents"]:
        G.add_node(agent["id"], role=agent["role"], description=agent["description"])

    for edge in config["edges"]:
        attrs = {
            "label": edge.get("label", f"{edge['source']}-{edge['target']}"),
            "edge_type": edge.get("edge_type", "default"),
        }
        G.add_edge(edge["source"], edge["target"], **attrs)
        if edge.get("bidirectional", False):
            G.add_edge(edge["target"], edge["source"], **attrs)

    return G


def compute_edge_features(G: nx.DiGraph) -> dict:
    """Compute structural features for each edge in the graph.

    Returns dict mapping (source, target) -> feature_dict.
    """
    G_undirected = G.to_undirected()
    edge_betweenness_ud = nx.edge_betweenness_centrality(G_undirected)
    edge_betweenness_dir = nx.edge_betweenness_centrality(G)

    features = {}
    for u, v, data in G.edges(data=True):
        feat = {}

        feat["betweenness_centrality"] = edge_betweenness_dir.get((u, v), 0)
        feat["betweenness_centrality_undirected"] = edge_betweenness_ud.get(
            (u, v), edge_betweenness_ud.get((v, u), 0)
        )

        feat["source_out_degree"] = G.out_degree(u)
        feat["source_in_degree"] = G.in_degree(u)
        feat["target_out_degree"] = G.out_degree(v)
        feat["target_in_degree"] = G.in_degree(v)

        # Bridge: does removing this edge disconnect the graph?
        G_copy = G.copy()
        G_copy.remove_edge(u, v)
        feat["is_bridge"] = not nx.is_weakly_connected(G_copy) if nx.is_weakly_connected(G) else False

        degree_centrality = nx.degree_centrality(G)
        feat["source_degree_centrality"] = degree_centrality[u]
        feat["target_degree_centrality"] = degree_centrality[v]

        closeness = nx.closeness_centrality(G)
        feat["source_closeness"] = closeness[u]
        feat["target_closeness"] = closeness[v]

        feat["label"] = data.get("label", f"{u}-{v}")
        feat["edge_type"] = data.get("edge_type", "default")

        features[(u, v)] = feat

    return features


def load_all_topologies(config_dir: str = "config/topologies") -> dict:
    """Load all topology configs from a directory.

    Returns dict mapping topology_name -> (config, graph, edge_features).
    """
    topologies = {}
    config_path = Path(config_dir)
    for yaml_file in sorted(config_path.glob("*.yaml")):
        config = load_topology(str(yaml_file))
        G = build_graph(config)
        features = compute_edge_features(G)
        topologies[config["name"]] = {
            "config": config,
            "graph": G,
            "edge_features": features,
        }
    return topologies


def print_topology_summary(topologies: dict):
    """Print a summary of all loaded topologies."""
    print(f"{'Topology':<15} {'Agents':<8} {'Edges':<8} {'Directed Edges':<15}")
    print("-" * 50)
    for name, data in topologies.items():
        G = data["graph"]
        n_undirected = len(data["config"]["edges"])
        print(f"{name:<15} {G.number_of_nodes():<8} {n_undirected:<8} {G.number_of_edges():<15}")
