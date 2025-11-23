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
    for _actor, films in actor_to_films.values():
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

    # Assign unanalyzed nodes to separate communities
    next_comm_id = len(communities)
    for node in G.nodes():
        if node not in node_to_community:
            node_to_community[node] = next_comm_id
            next_comm_id += 1
            next_comm_id += 1

    logger.info(
        f"✅ Girvan-Newman: Found {len(communities)} communities (+ {next_comm_id - len(communities)} isolated)"
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


def visualize_communities(
    G: nx.Graph,  # noqa: N803
    communities: dict,
    method_name: str,
    max_nodes: int = 10000,
    min_community_size: int = 100,
) -> None:
    """Visualize the graph with community coloring.

    Args:
        G: NetworkX graph
        communities: Dictionary mapping node to community id
        method_name: Name of the clustering method
        max_nodes: Maximum nodes to visualize (for performance)
        min_community_size: Minimum community size to label in visualization

    """
    logger.info(f"📊 Visualizing {method_name} communities...")

    # Sample nodes if graph is too large
    if G.number_of_nodes() > max_nodes:
        logger.info(f"   Sampling {max_nodes} nodes for visualization...")
        # Get largest connected component and sample from it
        if nx.is_connected(G):
            nodes_to_plot = list(G.nodes())[:max_nodes]
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            nodes_to_plot = list(largest_cc)[:max_nodes]

        G_plot = G.subgraph(nodes_to_plot).copy()  # noqa: N806
    else:
        G_plot = G  # noqa: N806

    # Create figure with space for legend on the right
    _fig, ax = plt.subplots(figsize=(20, 12))

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
    cmap = plt.cm.get_cmap("tab20")

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

    # Add labels to the right side of the plot
    y_start = 0.95
    y_step = 0.85 / max(len(community_labels), 1)

    for idx, (comm_id, label, color) in enumerate(community_labels):
        y_pos = y_start - idx * y_step
        ax.text(
            1.02,
            y_pos,
            f"Community {comm_id}: {label}",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=color,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": color, "linewidth": 2, "alpha": 0.9},
        )

    ax.set_title(f"Movie Network Communities - {method_name}", fontsize=20, pad=20)
    ax.axis("off")

    # Adjust layout to prevent label cutoff
    plt.subplots_adjust(right=0.75)

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
            m[0][:50] + "..." if len(m[0]) > 50 else m[0] for m in movies  # noqa: PLR2004
        ]  # Truncate long names
        weighted_degrees = [m[2] for m in movies]

        # Create horizontal bar chart
        y_pos = np.arange(len(movie_names))
        colors = plt.cm.get_cmap("tab20")(comm_id % 20)
        ax.barh(y_pos, weighted_degrees, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(movie_names, fontsize=10)
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

    # Visualize communities
    visualize_communities(G, gn_communities, "Girvan-Newman", min_community_size=MIN_COMMUNITY_SIZE)
    visualize_communities(G, louvain_communities, "Louvain", min_community_size=MIN_COMMUNITY_SIZE)
    # visualize_communities(G, spectral_communities, "Spectral Clustering", min_community_size=MIN_COMMUNITY_SIZE)
    visualize_communities(
        G, fast_spectral_communities, "Fast Spectral Clustering", min_community_size=MIN_COMMUNITY_SIZE
    )

    # Visualize top movies per community
    visualize_top_movies_per_community(gn_top_movies, "Girvan-Newman", max_communities=10)
    visualize_top_movies_per_community(louvain_top_movies, "Louvain", max_communities=10)
    visualize_top_movies_per_community(fast_spectral_top_movies, "Fast Spectral Clustering", max_communities=10)

    # Save results
    # save_results(G, gn_communities, louvain_communities, spectral_communities, fast_spectral_communities, all_stats)
    save_results(G, gn_communities, louvain_communities, {}, fast_spectral_communities, all_stats, top_movies_data)

    logger.info("🎉 Actor network analysis completed!")


if __name__ == "__main__":
    main()
