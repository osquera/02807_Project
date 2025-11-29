"""Analyze movie network based on shared actors.

This script constructs a graph where:
- Nodes = movies
- Edges = weighted by number of shared actors between movies

Performs community detection using:
1. Girvan-Newman method
2. Louvain method
3. Spectral clustering

Generates visualizations and comparison statistics.
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
from helpers.logger import logger
from networkx.algorithms import community
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, SpectralClustering

CLEAN_LOCATION = Path("data/clean")
ANALYSIS_LOCATION = Path("data/analysis")
ANALYSIS_LOCATION.mkdir(parents=True, exist_ok=True)

# Configuration
USE_MERGED_DATASET = False  # Set to True to use merged dataset instead
MIN_ACTORS_PER_MOVIE = 2  # Minimum actors for a movie to be included
MIN_SHARED_ACTORS = 1  # Minimum shared actors to create an edge
MIN_COMMUNITY_SIZE = 100  # Minimum community size to show in visualizations


def load_actor_film_data(use_merged: bool = False) -> pl.DataFrame:
    """Load actor-film relationships.

    Args:
        use_merged: If True, load from merged dataset. Otherwise from actorfilms_clean.csv

    Returns:
        DataFrame with Film and Actor columns

    """
    logger.info("📊 Loading actor-film data...")

    if use_merged:
        path = Path("data/merged/movies_merged.csv")
        if not path.exists():
            msg = f"Merged dataset not found at {path}"
            raise FileNotFoundError(msg)

        df = pl.read_csv(path)
        # Extract actor-film pairs from merged dataset
        # Assuming af_actors_list contains pipe-separated actor names
        if "af_film_original" not in df.columns or "af_actors_list" not in df.columns:
            msg = "Merged dataset missing required actor columns"
            raise ValueError(msg)

        # Explode the pipe-separated actor lists
        actor_films = []
        for row in df.iter_rows(named=True):
            film = row.get("af_film_original") or row.get("movie_title")
            actors_str = row.get("af_actors_list", "")
            if film and actors_str:
                actors = actors_str.split("|")
                for actor in actors:
                    if actor:
                        actor_films.extend([{"Film": film, "Actor": actor.strip()}])

        return pl.DataFrame(actor_films)

    path = CLEAN_LOCATION / "actorfilms_clean.csv"
    if not path.exists():
        msg = f"Actor films dataset not found at {path}"
        raise FileNotFoundError(msg)

    df = pl.read_csv(
        path,
        schema_overrides={
            "Actor": pl.Utf8,
            "Film": pl.Utf8,
        },
    )

    return df.select(["Film", "Actor"])


def build_movie_coactor_graph(actor_film_df: pl.DataFrame) -> nx.Graph:
    """Build a graph where movies are connected by shared actors.

    Args:
        actor_film_df: DataFrame with Film and Actor columns

    Returns:
        NetworkX graph with movies as nodes and shared actors as edge weights

    """
    logger.info("🔨 Building movie co-actor graph...")

    # Convert to dict: actor -> list of films
    actor_to_films = defaultdict(list)
    for row in actor_film_df.iter_rows(named=True):
        actor = row["Actor"]
        film = row["Film"]
        if actor and film:
            actor_to_films[actor].append(film)

    logger.info(f"   Found {len(actor_to_films)} unique actors")

    # Build graph by finding all pairs of movies that share actors
    G = nx.Graph()  # noqa: N806
    edge_weights = defaultdict(int)

    # For each actor, connect all pairs of their films
    for films in actor_to_films.values():
        if len(films) < 2:  # noqa: PLR2004
            continue

        # Create edges between all pairs of films for this actor
        for i in range(len(films)):
            for j in range(i + 1, len(films)):
                film1, film2 = films[i], films[j]
                # Create edge key (sorted to avoid duplicates)
                edge_key = tuple(sorted([film1, film2]))
                edge_weights[edge_key] += 1

    logger.info(f"   Found {len(edge_weights)} movie pairs with shared actors")

    # Add edges to graph
    for (film1, film2), weight in edge_weights.items():
        if weight >= MIN_SHARED_ACTORS:
            G.add_edge(film1, film2, weight=weight)

    logger.info(f"✅ Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


def apply_girvan_newman(G: nx.Graph, num_communities: int = 10, sample_size: int = 50, max_nodes: int = 1000) -> dict:  # noqa: N803
    """Apply Girvan-Newman community detection with sampling for large graphs.

    For very large graphs, works on a subgraph sample to make computation tractable.
    For large graphs, uses a random sample of vertices as BFS roots to speed up computation.

    Args:
        G: NetworkX graph
        num_communities: Target number of communities
        sample_size: Number of random vertices to use as BFS roots (for large graphs)
        max_nodes: Maximum nodes to include in analysis (samples if exceeded)

    Returns:
        Dictionary mapping node to community id

    """
    logger.info(f"🔍 Applying Girvan-Newman method (target: {num_communities} communities)...")

    # Get the largest connected component for analysis
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc).copy()  # noqa: N806
        logger.info(f"   Using largest connected component: {len(G_connected)} nodes")
    else:
        G_connected = G  # noqa: N806

    # For extremely large graphs, sample a subgraph to make it tractable
    if G_connected.number_of_nodes() > max_nodes:
        logger.info(
            f"   Graph too large ({G_connected.number_of_nodes()} nodes), sampling {max_nodes} nodes for analysis"
        )
        random.seed(42)
        sampled_graph_nodes = random.sample(list(G_connected.nodes()), min(max_nodes, G_connected.number_of_nodes()))
        G_to_analyze = G_connected.subgraph(sampled_graph_nodes).copy()  # noqa: N806
        logger.info(
            f"   Sampled subgraph: {G_to_analyze.number_of_nodes()} nodes, {G_to_analyze.number_of_edges()} edges"
        )
    else:
        G_to_analyze = G_connected  # noqa: N806

    # For large graphs, sample vertices for BFS roots to speed up computation
    # The betweenness computation is O(nm), so sampling helps significantly
    if G_to_analyze.number_of_nodes() > 1000:  # noqa: PLR2004
        logger.info(f"   Using {sample_size} sampled vertices for BFS roots")
        random.seed(42)
        sampled_nodes = random.sample(list(G_to_analyze.nodes()), min(sample_size, G_to_analyze.number_of_nodes()))

        # Use edge betweenness with sampled nodes as sources
        def most_central_edge_sampled(G: nx.Graph) -> tuple:  # noqa: N803
            """Compute edge betweenness using only sampled nodes as sources."""
            # Compute betweenness only from sampled nodes to speed up computation
            betweenness = nx.edge_betweenness_centrality_subset(
                G, sources=sampled_nodes, targets=G.nodes(), normalized=False, weight="weight"
            )
            return max(betweenness.items(), key=lambda x: x[1])[0]

        communities_generator = community.girvan_newman(G_to_analyze, most_valuable_edge=most_central_edge_sampled)
    else:
        # Run standard Girvan-Newman for smaller graphs
        communities_generator = community.girvan_newman(G_to_analyze)

    # Get communities at desired level
    communities = None
    for i, comms in enumerate(communities_generator):
        communities = comms
        logger.info(f"   Iteration {i + 1}: {len(comms)} communities")
        if len(comms) >= num_communities:
            break
        if i >= 50:  # Reduced safety limit since we're sampling  # noqa: PLR2004
            logger.warning(f"   Stopped at iteration {i}, found {len(comms)} communities")
            break

    # Handle case where no communities were found
    if communities is None:
        logger.warning("   No communities found, treating each node as separate community")
        communities = [{node} for node in G_to_analyze.nodes()]

    # Convert to dict (for analyzed subgraph)
    node_to_community = {}
    for comm_id, comm_nodes in enumerate(communities):
        for node in comm_nodes:
            node_to_community[node] = comm_id

    # Assign unanalyzed nodes to nearest community based on edge weights
    next_comm_id = len(communities)
    unassigned_nodes = [node for node in G.nodes() if node not in node_to_community]

    for node in unassigned_nodes:
        # Find which community this node is most connected to
        community_weights = defaultdict(int)
        for neighbor in G.neighbors(node):
            if neighbor in node_to_community:
                neighbor_comm = node_to_community[neighbor]
                community_weights[neighbor_comm] += G[node][neighbor].get("weight", 1)

        # Assign to most connected community, or create new one if isolated
        if community_weights:
            best_community = max(community_weights.items(), key=lambda x: x[1])[0]
            node_to_community[node] = best_community
        else:
            # Isolated node with no connections to analyzed nodes
            node_to_community[node] = next_comm_id
            next_comm_id += 1

    logger.info(
        f"✅ Girvan-Newman: Found {len(communities)} communities (assigned {len(unassigned_nodes)} unanalyzed nodes)"
    )

    return node_to_community


def apply_louvain(G: nx.Graph) -> dict:  # noqa: N803
    """Apply Louvain community detection.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary mapping node to community id

    """
    logger.info("🔍 Applying Louvain method...")

    # Louvain algorithm
    communities = community.louvain_communities(G, weight="weight", seed=42)

    # Convert to dict
    node_to_community = {}
    for comm_id, comm_nodes in enumerate(communities):
        for node in comm_nodes:
            node_to_community[node] = comm_id

    logger.info(f"✅ Louvain: Found {len(communities)} communities")

    return node_to_community


def apply_spectral_clustering(G: nx.Graph, num_clusters: int = 10) -> dict:  # noqa: N803
    """Apply spectral clustering.

    Args:
        G: NetworkX graph
        num_clusters: Number of clusters to create

    Returns:
        Dictionary mapping node to community id

    """
    logger.info(f"🔍 Applying Spectral Clustering ({num_clusters} clusters)...")

    # Get the largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc).copy()  # noqa: N806
        logger.info(f"   Using largest connected component: {len(G_connected)} nodes")
    else:
        G_connected = G  # noqa: N806

    # Get adjacency matrix
    nodes = list(G_connected.nodes())
    adj_matrix = nx.to_scipy_sparse_array(G_connected, nodelist=nodes, weight="weight")

    # Convert to int32 indices to avoid scikit-learn compatibility issue
    adj_matrix = adj_matrix.astype(np.float32)
    adj_matrix.indices = adj_matrix.indices.astype(np.int32)
    adj_matrix.indptr = adj_matrix.indptr.astype(np.int32)

    # Apply spectral clustering
    n_clusters = min(num_clusters, len(nodes))
    clustering = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42, n_init=10)

    labels = clustering.fit_predict(adj_matrix)

    # Convert to dict
    node_to_community = {node: int(label) for node, label in zip(nodes, labels, strict=False)}

    # Assign isolated nodes to separate communities
    next_comm_id = n_clusters
    for node in G.nodes():
        if node not in node_to_community:
            node_to_community[node] = next_comm_id
            next_comm_id += 1

    logger.info(f"✅ Spectral Clustering: {n_clusters} clusters (+ {next_comm_id - n_clusters} isolated)")

    return node_to_community


def apply_fast_spectral_clustering(G: nx.Graph, num_clusters: int = 10) -> dict:  # noqa: N803
    """Apply fast spectral clustering from "Fast and Simple Spectral Clustering in Theory and Practice".

    This method uses power iteration on the normalized signless Laplacian matrix,
    which is much faster than traditional spectral clustering for large graphs.

    Args:
        G: NetworkX graph
        num_clusters: Number of clusters to create

    Returns:
        Dictionary mapping node to community id

    """
    logger.info(f"🔍 Applying Fast Spectral Clustering ({num_clusters} clusters)...")

    # Get the largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc).copy()  # noqa: N806
        logger.info(f"   Using largest connected component: {len(G_connected)} nodes")
    else:
        G_connected = G  # noqa: N806

    n = G_connected.number_of_nodes()
    k = num_clusters

    # Calculate parameters from the paper
    l = max(2, math.ceil(math.log2(k)))  # Number of random vectors  # noqa: E741
    t = 10 * math.ceil(math.log2(n / k))  # Number of power iterations

    logger.info(f"   Using {l} random vectors and {t} power iterations")

    # Get adjacency matrix and compute degree matrix
    nodes = list(G_connected.nodes())
    A = nx.to_scipy_sparse_array(G_connected, nodelist=nodes, weight="weight", format="csr")  # noqa: N806

    # Compute degree matrix D
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = csr_matrix(np.diag(1.0 / np.sqrt(degrees + 1e-10)))  # noqa: N806

    # Compute normalized signless Laplacian: M = D^(-1/2) (D + A) D^(-1/2)
    # This is equivalent to: I + D^(-1/2) A D^(-1/2)
    M = D_inv_sqrt @ (A + csr_matrix(np.diag(degrees))) @ D_inv_sqrt  # noqa: N806

    # Initialize random vectors
    np.random.seed(42)  # noqa: NPY002
    Y = np.random.normal(size=(n, l))  # noqa: N806, NPY002

    # Power iteration: repeatedly multiply by M
    logger.info("   Running power iteration...")
    for i in range(t):
        Y = M @ Y  # noqa: N806
        if (i + 1) % 10 == 0:
            logger.info(f"   Iteration {i + 1}/{t}")

    # SVD to orthogonalize
    Y, _, _ = np.linalg.svd(Y, full_matrices=False)  # noqa: N806

    # K-means clustering on the embedding
    logger.info("   Running K-means on embedding...")
    n_clusters = min(k, len(nodes))
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(Y)

    # Convert to dict
    node_to_community = {node: int(label) for node, label in zip(nodes, labels, strict=False)}

    # Assign isolated nodes to separate communities
    next_comm_id = n_clusters
    for node in G.nodes():
        if node not in node_to_community:
            node_to_community[node] = next_comm_id
            next_comm_id += 1

    logger.info(f"✅ Fast Spectral Clustering: {n_clusters} clusters (+ {next_comm_id - n_clusters} isolated)")

    return node_to_community


def calculate_modularity(G: nx.Graph, communities: dict) -> float:  # noqa: N803
    """Calculate modularity score for a community assignment.

    Args:
        G: NetworkX graph
        communities: Dictionary mapping node to community id

    Returns:
        Modularity score

    """
    # Convert dict to list of sets
    comm_sets = defaultdict(set)
    for node, comm_id in communities.items():
        comm_sets[comm_id].add(node)

    comm_list = list(comm_sets.values())

    return community.modularity(G, comm_list, weight="weight")


def generate_configuration_model(G: nx.Graph, num_models: int = 10) -> list[nx.Graph]:  # noqa: N803
    """Generate random Erdős-Rényi null models.

    Creates null model networks using the Erdős-Rényi random graph model, which
    preserves the number of nodes and edge density but randomizes all connections.
    This is much faster than configuration models and provides a good baseline
    for statistical comparison.

    Args:
        G: Original NetworkX graph
        num_models: Number of random graphs to generate

    Returns:
        List of randomized NetworkX graphs

    """
    logger.info(f"🎲 Generating {num_models} Erdős-Rényi null model graphs...")

    random_graphs = []
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Calculate edge probability for Erdős-Rényi model
    # p = m / (n * (n-1) / 2) where m = edges, n = nodes
    max_edges = num_nodes * (num_nodes - 1) / 2
    edge_probability = num_edges / max_edges

    # Store edge weights for later random assignment
    all_weights = [data.get("weight", 1) for _, _, data in G.edges(data=True)]

    # Get node labels from original graph
    nodes_list = list(G.nodes())

    logger.info(
        f"   Using edge probability p={edge_probability:.6f} ({num_nodes:,} nodes, ~{num_edges:,} edges expected)"
    )

    for i in range(num_models):
        try:
            # Generate Erdős-Rényi random graph
            random_graph = nx.erdos_renyi_graph(num_nodes, edge_probability, seed=42 + i)

            # Relabel nodes to match original graph
            mapping = dict(zip(range(num_nodes), nodes_list, strict=True))
            random_graph = nx.relabel_nodes(random_graph, mapping)

            # Randomly assign edge weights from the original distribution
            random.seed(42 + i)
            if random_graph.number_of_edges() > 0:
                shuffled_weights = random.choices(all_weights, k=random_graph.number_of_edges())  # noqa: S311
                for (u, v), weight in zip(random_graph.edges(), shuffled_weights, strict=True):
                    random_graph[u][v]["weight"] = weight

            random_graphs.append(random_graph)
            logger.info(
                f"   Model {i + 1}/{num_models}: {random_graph.number_of_nodes()} nodes, "
                f"{random_graph.number_of_edges()} edges"
            )

        except (nx.NetworkXError, nx.NetworkXAlgorithmError) as e:
            logger.warning(f"   Failed to generate model {i + 1}: {e}")
            continue

    logger.info(f"✅ Generated {len(random_graphs)} Erdős-Rényi null models")

    return random_graphs


def calculate_null_model_modularity(random_graphs: list[nx.Graph], communities: dict, method_name: str) -> dict:
    """Calculate modularity statistics for null model comparison.

    Args:
        random_graphs: List of random configuration model graphs
        communities: Community assignments from real network
        method_name: Name of the clustering method

    Returns:
        Dictionary with null model statistics

    """
    logger.info(f"📊 Calculating null model modularity for {method_name}...")

    modularities = []
    for i, g_random in enumerate(random_graphs):
        try:
            # Filter communities to only include nodes present in this random graph
            random_nodes = set(g_random.nodes())
            filtered_communities = {node: comm_id for node, comm_id in communities.items() if node in random_nodes}

            if not filtered_communities:
                logger.warning(f"   Model {i + 1}: No overlapping nodes between graph and communities")
                continue

            mod = calculate_modularity(g_random, filtered_communities)
            modularities.append(mod)
        except (KeyError, ValueError, nx.NetworkXError) as e:
            logger.warning(f"   Failed to calculate modularity for model {i + 1}: {e}")
            continue

    if not modularities:
        logger.warning("   No valid modularity scores calculated")
        return {
            "method": method_name,
            "num_models": 0,
            "mean_modularity": 0.0,
            "std_modularity": 0.0,
            "min_modularity": 0.0,
            "max_modularity": 0.0,
        }

    stats = {
        "method": method_name,
        "num_models": len(modularities),
        "mean_modularity": float(np.mean(modularities)),
        "std_modularity": float(np.std(modularities)),
        "min_modularity": float(np.min(modularities)),
        "max_modularity": float(np.max(modularities)),
        "all_modularities": [float(m) for m in modularities],
    }

    logger.info(f"   Null model modularity: {stats['mean_modularity']:.4f} ± {stats['std_modularity']:.4f}")

    return stats


def visualize_network(
    G: nx.Graph,  # noqa: N803
    max_nodes: int | None = None,
) -> None:
    """Visualize the network graph without community coloring.

    Args:
        G: NetworkX graph
        max_nodes: Maximum nodes to visualize (for performance), None for full network

    """
    logger.info("📊 Visualizing network graph...")

    # Use full graph unless max_nodes is specified
    if max_nodes is not None and G.number_of_nodes() > max_nodes:
        logger.info(f"   Sampling {max_nodes} nodes for visualization...")
        # Get largest connected component and sample from it
        if nx.is_connected(G):
            nodes_to_plot = list(G.nodes())[:max_nodes]
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            nodes_to_plot = list(largest_cc)[:max_nodes]

        G_plot = G.subgraph(nodes_to_plot).copy()  # noqa: N806
    else:
        logger.info(f"   Visualizing full network with {G.number_of_nodes()} nodes")
        G_plot = G  # noqa: N806

    # Create figure with compact layout
    _fig, ax = plt.subplots(figsize=(14, 10))

    # Use spring layout for positioning
    pos = nx.spring_layout(G_plot, k=0.5, iterations=50, seed=42)

    # Draw graph
    nx.draw_networkx_nodes(G_plot, pos, node_color="skyblue", node_size=50, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G_plot, pos, alpha=0.2, width=0.5, ax=ax)

    ax.set_title("Movie Co-Actor Network", fontsize=18, pad=15)
    ax.axis("off")

    # Compact layout
    plt.tight_layout()

    # Save
    output_path = ANALYSIS_LOCATION / "network_graph.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"   Saved network visualization to {output_path}")


def visualize_communities(
    G: nx.Graph,  # noqa: N803
    communities: dict,
    method_name: str,
    max_nodes: int | None = None,
    min_community_size: int = 100,
) -> None:
    """Visualize the graph with community coloring.

    Args:
        G: NetworkX graph
        communities: Dictionary mapping node to community id
        method_name: Name of the clustering method
        max_nodes: Maximum nodes to visualize (for performance), None for full network
        min_community_size: Minimum community size to label in visualization

    """
    logger.info(f"📊 Visualizing {method_name} communities...")

    # Use full graph unless max_nodes is specified
    if max_nodes is not None and G.number_of_nodes() > max_nodes:
        logger.info(f"   Sampling {max_nodes} nodes for visualization...")
        # Get largest connected component and sample from it
        if nx.is_connected(G):
            nodes_to_plot = list(G.nodes())[:max_nodes]
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            nodes_to_plot = list(largest_cc)[:max_nodes]

        G_plot = G.subgraph(nodes_to_plot).copy()  # noqa: N806
    else:
        logger.info(f"   Visualizing full network with {G.number_of_nodes()} nodes")
        G_plot = G  # noqa: N806

    # Create figure with compact layout for LaTeX reports
    _fig, ax = plt.subplots(figsize=(14, 10))

    # Use spring layout for positioning
    pos = nx.spring_layout(G_plot, k=0.5, iterations=50, seed=42)

    # Get colors for communities
    node_colors = [communities.get(node, -1) for node in G_plot.nodes()]

    # Draw graph on main axis
    nx.draw_networkx_nodes(G_plot, pos, node_color=node_colors, node_size=50, cmap="tab20", alpha=0.8, ax=ax)

    nx.draw_networkx_edges(G_plot, pos, alpha=0.2, width=0.5, ax=ax)

    # Find and label the most connected node in each community
    comm_nodes = defaultdict(list)
    for node in G_plot.nodes():
        comm_id = communities.get(node, -1)
        if comm_id != -1:
            comm_nodes[comm_id].append(node)

    # For each community, find the node with highest weighted degree
    community_labels = []
    cmap = plt.colormaps["tab20"]

    for comm_id, nodes in comm_nodes.items():
        if len(nodes) < min_community_size:  # Skip small communities
            continue

        # Find node with highest weighted degree in this community
        max_weighted_degree = -1
        top_node = None
        for node in nodes:
            if node in G:
                weighted_degree = sum(G[node][neighbor].get("weight", 1) for neighbor in G.neighbors(node))
                if weighted_degree > max_weighted_degree:
                    max_weighted_degree = weighted_degree
                    top_node = node

        if top_node:
            # Truncate long movie names
            label = top_node[:40] + "..." if len(top_node) > 40 else top_node  # noqa: PLR2004
            color = cmap(comm_id % 20)
            community_labels.append((comm_id, label, color))

    # Sort by community ID for consistent ordering
    community_labels.sort(key=lambda x: x[0])

    # Add compact labels to the right side closer to the graph
    y_start = 0.98
    y_step = 0.90 / max(len(community_labels), 1)

    for idx, (comm_id, label, color) in enumerate(community_labels):
        y_pos = y_start - idx * y_step
        ax.text(
            1.01,
            y_pos,
            f"C{comm_id}: {label}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color=color,
            verticalalignment="top",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 2.5,
                "alpha": 0.95,
            },
        )

    ax.set_title(f"Movie Network Communities - {method_name}", fontsize=18, pad=15)
    ax.axis("off")

    # Compact layout for LaTeX reports
    plt.subplots_adjust(right=0.82, left=0.02, top=0.96, bottom=0.02)

    # Save
    output_path = ANALYSIS_LOCATION / f"communities_{method_name.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"   Saved visualization to {output_path}")


def analyze_communities(G: nx.Graph, communities: dict, method_name: str) -> dict:  # noqa: N803
    """Analyze community structure and generate statistics.

    Args:
        G: NetworkX graph
        communities: Dictionary mapping node to community id
        method_name: Name of the clustering method

    Returns:
        Dictionary with analysis results

    """
    logger.info(f"📈 Analyzing {method_name} communities...")

    # Calculate modularity
    modularity = calculate_modularity(G, communities)

    # Community sizes
    comm_sizes = defaultdict(int)
    for comm_id in communities.values():
        comm_sizes[comm_id] += 1

    sizes = sorted(comm_sizes.values(), reverse=True)

    # Number of communities
    num_communities = len(comm_sizes)

    stats = {
        "method": method_name,
        "num_communities": num_communities,
        "modularity": float(modularity),
        "largest_community": sizes[0] if sizes else 0,
        "smallest_community": sizes[-1] if sizes else 0,
        "average_community_size": float(np.mean(sizes)) if sizes else 0,
        "median_community_size": float(np.median(sizes)) if sizes else 0,
        "community_size_distribution": sizes[:20],  # Top 20 communities
    }

    logger.info(f"   Communities: {num_communities}")
    logger.info(f"   Modularity: {modularity:.4f}")
    logger.info(f"   Largest community: {stats['largest_community']} nodes")
    logger.info(f"   Average community size: {stats['average_community_size']:.1f}")

    return stats


def get_top_movies_per_community(G: nx.Graph, communities: dict, top_n: int = 5, min_community_size: int = 5) -> dict:  # noqa: N803
    """Get the most connected movies in each community.

    Args:
        G: NetworkX graph
        communities: Dictionary mapping node to community id
        top_n: Number of top movies to return per community
        min_community_size: Minimum community size to include

    Returns:
        Dictionary mapping community id to list of (movie, degree, weighted_degree) tuples

    """
    logger.info(f"🎬 Finding top {top_n} most connected movies per community...")

    # Group nodes by community
    comm_nodes = defaultdict(list)
    for node, comm_id in communities.items():
        comm_nodes[comm_id].append(node)

    # For each community, find top connected movies
    top_movies = {}
    for comm_id, nodes in comm_nodes.items():
        if len(nodes) < min_community_size:
            continue

        # Calculate degree and weighted degree for each node in this community
        node_stats = []
        for node in nodes:
            if node in G:
                degree = len(list(G.neighbors(node)))
                weighted_degree = sum(G[node][neighbor].get("weight", 1) for neighbor in G.neighbors(node))
                node_stats.append((node, degree, weighted_degree))

        # Sort by weighted degree (number of shared actors)
        node_stats.sort(key=lambda x: x[2], reverse=True)
        top_movies[comm_id] = node_stats[:top_n]

    logger.info(f"   Found top movies for {len(top_movies)} communities")

    return top_movies


def visualize_top_movies_per_community(top_movies: dict, method_name: str, max_communities: int = 10) -> None:
    """Visualize the top movies per community as a table/chart.

    Args:
        top_movies: Dictionary from get_top_movies_per_community
        method_name: Name of the clustering method
        max_communities: Maximum number of communities to visualize

    """
    logger.info(f"📊 Visualizing top movies for {method_name}...")

    # Sort communities by size (largest first)
    sorted_communities = sorted(top_movies.items(), key=lambda x: len(x[1]), reverse=True)[:max_communities]

    # Create figure with subplots
    n_communities = len(sorted_communities)
    _fig, axes = plt.subplots(n_communities, 1, figsize=(16, 4 * n_communities))

    if n_communities == 1:
        axes = [axes]

    for idx, (comm_id, movies) in enumerate(sorted_communities):
        ax = axes[idx]

        # Prepare data
        movie_names = [
            m[0][:60] + "..." if len(m[0]) > 60 else m[0]  # noqa: PLR2004
            for m in movies
        ]  # Truncate long names
        weighted_degrees = [m[2] for m in movies]

        # Create horizontal bar chart
        y_pos = np.arange(len(movie_names))
        colors = plt.colormaps["tab20"](comm_id % 20)
        bars = ax.barh(y_pos, weighted_degrees, color=colors, height=0.7)

        # Add movie titles inside the bars with white text
        for bar, movie_name in zip(bars, movie_names, strict=True):
            width = bar.get_width()
            # Place text inside bar, aligned to the left with some padding
            ax.text(
                width * 0.02,  # 2% from left edge of bar
                bar.get_y() + bar.get_height() / 2,
                movie_name,
                ha="left",
                va="center",
                color="white",
                fontsize=13,
                fontweight="bold",
            )

        # Remove y-axis labels since titles are now in bars
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xlabel("Weighted Degree (Shared Actors)", fontsize=11)
        ax.set_title(f"Community {comm_id} - Top {len(movies)} Most Connected Movies", fontsize=12, pad=10)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(f"Top Connected Movies per Community - {method_name}", fontsize=16, y=0.995)
    plt.tight_layout()

    # Save
    output_path = ANALYSIS_LOCATION / f"top_movies_{method_name.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"   Saved visualization to {output_path}")


def save_results(
    G: nx.Graph,  # noqa: N803
    gn_communities: dict,
    louvain_communities: dict,
    spectral_communities: dict,
    fast_spectral_communities: dict,
    stats: list,
    top_movies_data: dict | None = None,
    null_model_stats: list | None = None,
) -> None:
    """Save graph and analysis results.

    Args:
        G: NetworkX graph
        gn_communities: Girvan-Newman community assignments
        louvain_communities: Louvain community assignments
        spectral_communities: Spectral clustering assignments
        fast_spectral_communities: Fast spectral clustering assignments
        stats: List of statistics dictionaries
        top_movies_data: Dictionary mapping method name to top movies per community
        null_model_stats: List of null model statistics dictionaries

    """
    logger.info("💾 Saving results...")

    # Save graph
    graph_path = ANALYSIS_LOCATION / "movie_coactor_graph.gexf"
    nx.write_gexf(G, graph_path)
    logger.info(f"   Saved graph to {graph_path}")

    # Save community assignments
    communities_data = {
        "girvan_newman": gn_communities,
        "louvain": louvain_communities,
        "spectral": spectral_communities,
        "fast_spectral": fast_spectral_communities,
    }

    communities_path = ANALYSIS_LOCATION / "community_assignments.json"
    with open(communities_path, "w") as f:  # noqa: PTH123
        json.dump(communities_data, f, indent=2)
    logger.info(f"   Saved community assignments to {communities_path}")

    # Save statistics
    stats_path = ANALYSIS_LOCATION / "community_statistics.json"
    with open(stats_path, "w") as f:  # noqa: PTH123
        json.dump(stats, f, indent=2)
    logger.info(f"   Saved statistics to {stats_path}")

    # Save null model statistics if provided
    if null_model_stats is not None:
        null_stats_path = ANALYSIS_LOCATION / "null_model_statistics.json"
        with open(null_stats_path, "w") as f:  # noqa: PTH123
            json.dump(null_model_stats, f, indent=2)
        logger.info(f"   Saved null model statistics to {null_stats_path}")

    # Save top movies data if provided
    if top_movies_data is not None:
        top_movies_path = ANALYSIS_LOCATION / "top_movies_per_community.json"
        # Convert tuples to dicts for JSON serialization
        serializable_data = {}
        for method, communities in top_movies_data.items():
            serializable_data[method] = {}
            for comm_id, movies in communities.items():
                serializable_data[method][str(comm_id)] = [
                    {"movie": m[0], "degree": m[1], "weighted_degree": m[2]} for m in movies
                ]

        with open(top_movies_path, "w") as f:  # noqa: PTH123
            json.dump(serializable_data, f, indent=2)
        logger.info(f"   Saved top movies to {top_movies_path}")

    # Create comparison table
    logger.info("\n" + "=" * 80)
    logger.info("COMMUNITY DETECTION COMPARISON")
    logger.info("=" * 80)
    logger.info(f"{'Method':<20} {'Communities':<15} {'Modularity':<15} {'Largest':<15}")
    logger.info("-" * 80)
    for stat in stats:
        logger.info(
            f"{stat['method']:<20} "
            f"{stat['num_communities']:<15} "
            f"{stat['modularity']:<15.4f} "
            f"{stat['largest_community']:<15}"
        )
    logger.info("=" * 80 + "\n")

    # Create null model comparison table if available
    if null_model_stats:
        logger.info("=" * 80)
        logger.info("NULL MODEL COMPARISON (Erdős-Rényi Baseline)")
        logger.info("=" * 80)
        logger.info(f"{'Method':<20} {'Real Modularity':<18} {'Null Mean ± Std':<25} {'Significance':<15}")
        logger.info("-" * 80)

        # Create lookup for real modularity values
        real_mod = {stat["method"]: stat["modularity"] for stat in stats}

        for null_stat in null_model_stats:
            method = null_stat["method"]
            real_value = real_mod.get(method, 0.0)
            null_mean = null_stat["mean_modularity"]
            null_std = null_stat["std_modularity"]

            # Calculate z-score (how many standard deviations above null model)
            if null_std > 0:
                z_score = (real_value - null_mean) / null_std
                significance = f"{z_score:.2f} sigma"
            else:
                significance = "N/A"

            logger.info(f"{method:<20} {real_value:<18.4f} {null_mean:.4f} ± {null_std:.4f}    {significance:<15}")
        logger.info("=" * 80 + "\n")


def main() -> None:
    """Analyze pipeline."""
    logger.info("🚀 Starting actor network analysis...")

    # Load data
    actor_film_df = load_actor_film_data(use_merged=USE_MERGED_DATASET)
    logger.info(f"   Loaded {len(actor_film_df)} actor-film relationships")

    # Build graph
    G = build_movie_coactor_graph(actor_film_df)  # noqa: N806

    if G.number_of_nodes() == 0:
        logger.error("❌ Graph is empty! Cannot proceed with analysis.")
        return

    # Apply community detection methods
    num_target_communities = 10

    gn_communities = apply_girvan_newman(G, num_communities=num_target_communities)
    louvain_communities = apply_louvain(G)
    # spectral_communities = apply_spectral_clustering(G, num_clusters=num_target_communities)
    fast_spectral_communities = apply_fast_spectral_clustering(G, num_clusters=num_target_communities)

    # Analyze each method
    all_stats = []
    all_stats.append(analyze_communities(G, gn_communities, "Girvan-Newman"))
    all_stats.append(analyze_communities(G, louvain_communities, "Louvain"))
    # all_stats.append(analyze_communities(G, spectral_communities, "Spectral Clustering"))
    all_stats.append(analyze_communities(G, fast_spectral_communities, "Fast Spectral Clustering"))

    # Get top movies per community for each method
    logger.info("\n" + "=" * 80)
    logger.info("TOP CONNECTED MOVIES PER COMMUNITY")
    logger.info("=" * 80)

    top_movies_data = {}

    gn_top_movies = get_top_movies_per_community(G, gn_communities, top_n=5, min_community_size=MIN_COMMUNITY_SIZE)
    top_movies_data["girvan_newman"] = gn_top_movies

    louvain_top_movies = get_top_movies_per_community(
        G, louvain_communities, top_n=5, min_community_size=MIN_COMMUNITY_SIZE
    )
    top_movies_data["louvain"] = louvain_top_movies

    fast_spectral_top_movies = get_top_movies_per_community(
        G, fast_spectral_communities, top_n=5, min_community_size=MIN_COMMUNITY_SIZE
    )
    top_movies_data["fast_spectral"] = fast_spectral_top_movies

    # Generate null models for comparison
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING NULL MODELS (Erdős-Rényi Model)")
    logger.info("=" * 80)

    random_graphs = generate_configuration_model(G, num_models=10)

    # Calculate null model modularity for each method
    null_model_stats = []
    null_model_stats.append(calculate_null_model_modularity(random_graphs, gn_communities, "Girvan-Newman"))
    null_model_stats.append(calculate_null_model_modularity(random_graphs, louvain_communities, "Louvain"))
    null_model_stats.append(
        calculate_null_model_modularity(random_graphs, fast_spectral_communities, "Fast Spectral Clustering")
    )

    # Visualize the network (sample 10% for performance)
    sample_size = max(1000, G.number_of_nodes() // 10)
    visualize_network(G, max_nodes=sample_size)

    # Visualize communities (sample 10% for performance)
    visualize_communities(
        G, gn_communities, "Girvan-Newman", max_nodes=sample_size, min_community_size=MIN_COMMUNITY_SIZE
    )
    visualize_communities(
        G, louvain_communities, "Louvain", max_nodes=sample_size, min_community_size=MIN_COMMUNITY_SIZE
    )
    # visualize_communities(
    #     G, spectral_communities, "Spectral Clustering", max_nodes=sample_size, min_community_size=MIN_COMMUNITY_SIZE
    # )
    visualize_communities(
        G,
        fast_spectral_communities,
        "Fast Spectral Clustering",
        max_nodes=sample_size,
        min_community_size=MIN_COMMUNITY_SIZE,
    )

    # Visualize top movies per community
    visualize_top_movies_per_community(gn_top_movies, "Girvan-Newman", max_communities=10)
    visualize_top_movies_per_community(louvain_top_movies, "Louvain", max_communities=10)
    visualize_top_movies_per_community(fast_spectral_top_movies, "Fast Spectral Clustering", max_communities=10)

    # Save results
    # save_results(G, gn_communities, louvain_communities, spectral_communities, fast_spectral_communities, all_stats)
    save_results(
        G,
        gn_communities,
        louvain_communities,
        {},
        fast_spectral_communities,
        all_stats,
        top_movies_data,
        null_model_stats,
    )

    logger.info("🎉 Actor network analysis completed!")


if __name__ == "__main__":
    main()
