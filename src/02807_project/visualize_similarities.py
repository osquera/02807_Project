"""Visualize movie similarity matrix for paper/presentation."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collaborative_filtering import compute_item_similarity, load_ratings_data
from helpers.logger import logger


def find_movie_cluster(similarities: dict[tuple[str, str], float], n_movies: int = 5) -> tuple[list[str], np.ndarray]:
    """Find a cluster of highly interconnected similar movies.

    Args:
        similarities: Dictionary of movie similarity pairs
        n_movies: Number of movies to include in cluster

    Returns:
        Tuple of (movie_names, similarity_matrix)

    """
    logger.info(f"Finding cluster of {n_movies} highly similar movies...")

    # Build adjacency dictionary: movie -> list of (similar_movie, similarity)
    movie_connections: dict[str, list[tuple[str, float]]] = {}
    for (m1, m2), sim in similarities.items():
        if m1 not in movie_connections:
            movie_connections[m1] = []
        movie_connections[m1].append((m2, sim))

    # Find movie with highest average similarity to its neighbors
    best_seed = None
    best_avg_sim = 0.0

    for movie, neighbors in movie_connections.items():
        if len(neighbors) >= n_movies - 1:
            # Sort by similarity and take top neighbors
            top_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)[: n_movies - 1]
            avg_sim = sum(sim for _, sim in top_neighbors) / len(top_neighbors)

            if avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_seed = (movie, top_neighbors)

    if best_seed is None:
        logger.warning("Could not find a good cluster, using top similar pairs instead")
        # Fallback: just get top N-1 most similar pairs
        top_pairs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[: n_movies - 1]
        movies = set()
        for (m1, m2), _ in top_pairs:
            movies.add(m1)
            movies.add(m2)
            if len(movies) >= n_movies:
                break
        cluster_movies = list(movies)[:n_movies]
    else:
        seed_movie, top_neighbors = best_seed
        cluster_movies = [seed_movie] + [m for m, _ in top_neighbors]

    # Build similarity matrix for these movies
    n = len(cluster_movies)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 1.0  # Self-similarity
            else:
                m1, m2 = cluster_movies[i], cluster_movies[j]
                sim_matrix[i, j] = similarities.get((m1, m2), 0.0)

    logger.info(f"Found cluster with average similarity: {best_avg_sim:.3f}")
    logger.info(f"Movies in cluster: {', '.join(cluster_movies)}")

    return cluster_movies, sim_matrix


def find_top_similar_pairs(
    similarities: dict[tuple[str, str], float], n_pairs: int = 10
) -> list[tuple[str, str, float]]:
    """Find the top N most similar movie pairs.

    Args:
        similarities: Dictionary of movie similarity pairs
        n_pairs: Number of pairs to return

    Returns:
        List of (movie1, movie2, similarity) tuples

    """
    # Since similarities dict has both (m1,m2) and (m2,m1), we need to deduplicate
    unique_pairs = {}
    for (m1, m2), sim in similarities.items():
        pair = tuple(sorted([m1, m2]))
        if pair not in unique_pairs:
            unique_pairs[pair] = sim

    # Sort by similarity
    top_pairs = sorted(unique_pairs.items(), key=lambda x: x[1], reverse=True)[:n_pairs]

    return [(m1, m2, sim) for (m1, m2), sim in top_pairs]


def condense_movie_title(title: str) -> str:
    """Condense long movie titles by adding newlines for better plot readability."""
    return title.replace(":", ":\n").replace(" - ", " -\n").replace("(", "\n(").replace(",", ",\n")


def plot_similarity_heatmap(movie_names: list[str], sim_matrix: np.ndarray, output_path: Path | None = None) -> None:
    """Create a heatmap visualization of movie similarities.

    Args:
        movie_names: List of movie names
        sim_matrix: Similarity matrix (n x n)
        output_path: Path to save figure (None = display only)

    """
    # Condense long movie titles for better readability
    condensed_names = [condense_movie_title(name) for name in movie_names]

    # Create figure
    _, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=condensed_names,
        yticklabels=condensed_names,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Cosine Similarity"},
        annot_kws={"size": 8},
        ax=ax,
    )

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Add title
    plt.title("Movie Similarity Matrix\n(Item-based Collaborative Filtering)", fontsize=14, pad=20)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved heatmap to {output_path}")
    else:
        plt.show()

    plt.close()


def main() -> None:
    """Generate similarity visualization."""
    parser = argparse.ArgumentParser(description="Visualize movie similarity matrix")
    parser.add_argument("--n-movies", type=int, default=5, help="Number of movies in cluster (default: 5)")
    parser.add_argument("--n-pairs", type=int, default=10, help="Number of top pairs to show (default: 10)")
    parser.add_argument("--output", type=str, default=Path("data/analysis/"), help="Output path for heatmap (default: data/analysis/)")  # noqa: E501
    parser.add_argument("--min-common-users", type=int, default=100, help="Minimum common users for similarity")
    parser.add_argument("--min-similarity", type=float, default=0.0, help="Minimum similarity threshold (default: 0.0)")
    parser.add_argument("--min-rating", type=float, default=0.0, help="Minimum rating threshold")
    parser.add_argument("--seed-movie", type=str, default=None, help="Seed movie to build cluster around")

    args = parser.parse_args()

    # Load data
    logger.info("Loading ratings data...")
    ratings_df, _ = load_ratings_data(test_split=False, min_rating=args.min_rating)

    # Compute similarities
    logger.info("Computing item similarities...")
    similarities = compute_item_similarity(
        ratings_df,
        min_common_users=args.min_common_users,
        min_similarity=args.min_similarity,
    )

    # Show top similar pairs
    logger.info(f"\n{'=' * 80}")
    logger.info(f"TOP {args.n_pairs} MOST SIMILAR MOVIE PAIRS")
    logger.info(f"{'=' * 80}")

    top_pairs = find_top_similar_pairs(similarities, n_pairs=args.n_pairs)
    for i, (m1, m2, sim) in enumerate(top_pairs, 1):
        logger.info(f"{i:2d}. {m1:<40s} <-> {m2:<40s} (similarity: {sim:.4f})")

    # Find and visualize cluster
    logger.info(f"\n{'=' * 80}")
    logger.info("MOVIE CLUSTER VISUALIZATION")
    logger.info(f"{'=' * 80}\n")

    # If seed movie specified, find its neighbors
    if args.seed_movie:
        logger.info(f"Building cluster around: {args.seed_movie}")

        # Find the seed movie in similarities
        seed_neighbors = []
        for (m1, m2), sim in similarities.items():
            if m1.lower() == args.seed_movie.lower():
                seed_neighbors.append((m2, sim))
            elif m2.lower() == args.seed_movie.lower():
                seed_neighbors.append((m1, sim))

        if not seed_neighbors:
            logger.error(f"Movie '{args.seed_movie}' not found in similarities!")
            logger.info("Available movies (sample):")
            all_movies = set()
            for m1, m2 in list(similarities.keys())[:100]:
                all_movies.add(m1)
                all_movies.add(m2)
            for movie in sorted(all_movies)[:20]:
                logger.info(f"  - {movie}")
            return

        # Get top N-1 most similar movies
        top_neighbors = sorted(seed_neighbors, key=lambda x: x[1], reverse=True)[: args.n_movies - 1]
        cluster_movies = [args.seed_movie] + [m for m, _ in top_neighbors]

        # Build similarity matrix
        n = len(cluster_movies)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    m1, m2 = cluster_movies[i], cluster_movies[j]
                    sim_matrix[i, j] = similarities.get((m1, m2), 0.0)

        logger.info(f"Found {len(top_neighbors)} similar movies")
    else:
        cluster_movies, sim_matrix = find_movie_cluster(similarities, n_movies=args.n_movies)

    # Print similarity matrix
    logger.info("\nSimilarity Matrix:")
    logger.info(f"{'':40s} " + " ".join(f"{i + 1:>6d}" for i in range(len(cluster_movies))))
    for i, movie in enumerate(cluster_movies):
        values = " ".join(f"{sim_matrix[i, j]:6.3f}" for j in range(len(cluster_movies)))
        logger.info(f"{i + 1}. {movie:<37s} {values}")

    # Create heatmap
    output_path = Path(args.output) if args.output else None
    plot_similarity_heatmap(cluster_movies, sim_matrix, output_path)

    logger.info("\nVisualization complete!")


if __name__ == "__main__":
    main()
