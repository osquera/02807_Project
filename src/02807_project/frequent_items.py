import argparse
import random
from pathlib import Path

import polars as pl
from helpers.logger import logger

# Data paths
DATA_LOCATION = Path("data/clean")
DATA_PATH = DATA_LOCATION / "large_movie_dataset_clean.csv"

# Constants
ITEMSET_LENGTH_TWO = 2


def create_user_movie_df(
    min_rating: float | None = None,
    seed: int = 42,
    min_movies_for_split: int = 20,
    min_train_movies: int = 15,
    train_ratio: float = 0.8,
) -> tuple[dict[int, list[str]], dict[int, list[str]], set[int]]:
    """Create a dictionary mapping User_Id to list of movies they watched/rated.

    Splits users' movies into train/test sets for evaluation.

    Args:
        min_rating: Minimum rating threshold to filter movies (None = include all)
        seed: Random seed for train/test split
        min_movies_for_split: Minimum movies a user must have to be included in split
        min_train_movies: Minimum movies required in training set after split
        train_ratio: Ratio of movies to use for training (e.g., 0.8 for 80/20 split)

    Returns:
        Tuple of (train_user_movies, test_user_movies, qualifying_users)
        - train_user_movies: Dict with User_Id -> list of training movies
        - test_user_movies: Dict with User_Id -> list of test movies
        - qualifying_users: Set of user IDs that have test data

    """
    df = pl.scan_csv(DATA_PATH, infer_schema=True)

    # Filter by minimum rating if specified
    if min_rating is not None:
        total_ratings = df.select(pl.len()).collect().item()
        df = df.filter(pl.col("Rating") >= min_rating)
        filtered_ratings = df.select(pl.len()).collect().item()
        removed_ratings = total_ratings - filtered_ratings
        percentage = removed_ratings / total_ratings * 100
        logger.info(
            f"Filtering movies with rating >= {min_rating}: "
            f"removed {percentage:.1f}% of ratings"
        )

    # Create a dataframe with User_Id and list of Movie_Names they rated
    user_movies_df = df.group_by("User_Id").agg(pl.col("Movie_Name_Normalized")
                                                .alias("movies_watched")).sort("User_Id")
    user_movies_df = user_movies_df.collect()

    logger.info(f"Creating train/test split for users with {min_movies_for_split}+ movies")

    # Count movies per user
    user_movies_df = user_movies_df.with_columns(
        pl.col("movies_watched").list.len().alias("movie_count")
    )

    # Mark qualifying users
    qualifying_users_df = user_movies_df.filter(pl.col("movie_count") >= min_movies_for_split)
    qualifying_user_ids = set(qualifying_users_df["User_Id"].to_list())

    logger.info(f"Found {len(qualifying_user_ids)} users with {min_movies_for_split}+ movies")

    train_user_movies: dict[int, list[str]] = {}
    test_user_movies: dict[int, list[str]] = {}

    for row in user_movies_df.iter_rows(named=True):
        user_id = row["User_Id"]
        movies = list(row["movies_watched"])
        n_movies = len(movies)

        if user_id not in qualifying_user_ids:
            # Non-qualifying users: all movies go to training
            train_user_movies[user_id] = movies
            continue

        # Shuffle with seed for reproducibility
        random.seed(seed + user_id)
        random.shuffle(movies)

        n_train = int(n_movies * train_ratio)

        # Ensure minimum train size
        if n_train < min_train_movies:
            n_train = min(min_train_movies, n_movies - 1)

        # Skip if can't meet minimum requirements (put all in training)
        if n_train < min_train_movies or n_movies - n_train < 1:
            train_user_movies[user_id] = movies
            continue

        train_user_movies[user_id] = movies[:n_train]
        test_user_movies[user_id] = movies[n_train:]

    total_train_movies = sum(len(m) for m in train_user_movies.values())
    total_test_movies = sum(len(m) for m in test_user_movies.values())
    logger.info(f"Train/Test split complete: {total_train_movies} train movies, {total_test_movies} test movies")
    logger.info(f"Test set: {len(test_user_movies)} users with held-out movies")

    return train_user_movies, test_user_movies, qualifying_user_ids


def find_frequent_itemsets(user_movies: dict, min_support: float = 0.25) -> pl.DataFrame:
    """Find frequent itemsets using Polars for memory efficiency.

    This implementation uses the Apriori algorithm to find frequent 1-itemsets
    and 2-itemsets (movie pairs) that appear together frequently.

    Args:
        user_movies: Dictionary mapping User_Id to list of movies
        min_support: Minimum support threshold (0.0 to 1.0)

    Returns:
        DataFrame with columns: itemset, support, length

    """
    n_transactions = len(user_movies)
    min_count = int(min_support * n_transactions)

    # Step 1: Count individual item frequencies across all transactions
    item_counts: dict[str, int] = {}
    for movies in user_movies.values():
        for movie in movies:
            item_counts[movie] = item_counts.get(movie, 0) + 1

    # Step 2: Filter items that meet minimum support threshold (frequent 1-itemsets)
    frequent_items = {item: count for item, count in item_counts.items() if count >= min_count}
    logger.info(f"Found {len(frequent_items)} frequent 1-itemsets")

    # Create list of frequent itemsets with support
    itemsets_data = []
    for item, count in frequent_items.items():
        itemsets_data.append({
            "itemset": frozenset([item]),
            "support": count / n_transactions,
            "length": 1
        })

    # Step 3: Find frequent 2-itemsets (movie pairs)
    pair_counts: dict[frozenset[str], int] = {}
    frequent_items_set = set(frequent_items.keys())

    for movies in user_movies.values():
        # Only consider movies that are themselves frequent (optimization)
        freq_movies = [m for m in movies if m in frequent_items_set]
        # Generate all pairs from this user's frequent movies
        for i in range(len(freq_movies)):
            for j in range(i + 1, len(freq_movies)):
                pair = frozenset([freq_movies[i], freq_movies[j]])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Step 4: Add pairs that meet minimum support threshold
    for itemset, count in pair_counts.items():
        if count >= min_count:
            itemsets_data.append({
                "itemset": itemset,
                "support": count / n_transactions,
                "length": ITEMSET_LENGTH_TWO
            })

    number_of_two_itemsets = sum(1 for d in itemsets_data if d["length"] == ITEMSET_LENGTH_TWO)
    logger.info(f"""Found {number_of_two_itemsets} frequent 2-itemsets with {min_support=} ({min_count=})""")

    # Create DataFrame
    return pl.DataFrame(itemsets_data)


def generate_association_rules(
    itemsets_df: pl.DataFrame,
    min_confidence: float = 0.75
) -> pl.DataFrame:
    """Generate association rules from frequent itemsets.

    Creates rules of the form "if user watched A, then they also watched B"
    and calculates confidence and lift metrics.

    Args:
        itemsets_df: DataFrame of frequent itemsets from find_frequent_itemsets
        min_confidence: Minimum confidence threshold (0.0 to 1.0)

    Returns:
        DataFrame with columns: antecedents, consequents, support, confidence, lift

    """
    # Create a dictionary for fast itemset support lookup
    itemset_support_map = {}
    for row in itemsets_df.iter_rows(named=True):
        itemset_support_map[row["itemset"]] = row["support"]

    # Only use 2-itemsets for rules
    two_itemsets = itemsets_df.filter(pl.col("length") == ITEMSET_LENGTH_TWO)

    rules_data = []

    for row in two_itemsets.iter_rows(named=True):
        itemset = row["itemset"]
        itemset_support = row["support"]

        # Generate all possible rules from this itemset
        items = list(itemset)
        if len(items) != ITEMSET_LENGTH_TWO:
            continue

        # For 2-itemsets, we have 2 possible rules: A->B and B->A
        for i in range(ITEMSET_LENGTH_TWO):
            antecedent = frozenset([items[i]])
            consequent = frozenset([items[1-i]])

            # Calculate antecedent support using dictionary lookup
            antecedent_support = itemset_support_map.get(antecedent, 0)

            if antecedent_support == 0:
                continue

            # Calculate confidence: support(A U B) / support(A)
            confidence = itemset_support / antecedent_support

            if confidence >= min_confidence:
                # Calculate consequent support using dictionary lookup
                consequent_support = itemset_support_map.get(consequent, 0)

                # Calculate lift: confidence / support(B)
                lift = confidence / consequent_support if consequent_support > 0 else 0

                rules_data.append({
                    "antecedents": antecedent,
                    "consequents": consequent,
                    "support": itemset_support,
                    "confidence": confidence,
                    "lift": lift
                })

    logger.info(f"Generated {len(rules_data)} association rules with {min_confidence=}")

    if not rules_data:
        return pl.DataFrame({
            "antecedents": [],
            "consequents": [],
            "support": [],
            "confidence": [],
            "lift": []
        })

    return pl.DataFrame(rules_data)


def get_recommendations_for_movies(
    movies: list[str],
    rules: pl.DataFrame
) -> pl.DataFrame:
    """Get movie recommendations based on a list of movies using association rules.

    Args:
        movies: List of movie names that the user has watched
        rules: DataFrame of association rules

    Returns:
        DataFrame with recommended movies and their aggregated scores (sorted by avg_score)

    """
    if len(rules) == 0:
        logger.warning("No rules available for generating recommendations")
        return pl.DataFrame({"movie": [], "score": [], "count": []})

    # Find all rules where the antecedent is one of the input movies
    recommendations = []

    for movie in movies:
        movie_set = frozenset([movie])
        # Filter rules where this movie is the antecedent
        for row in rules.iter_rows(named=True):
            if row["antecedents"] == movie_set:
                # Extract the consequent movie
                consequent_movie = next(iter(row["consequents"]))
                # Use confidence * lift as the recommendation score
                score = row["confidence"] * row["lift"]
                recommendations.append({
                    "movie": consequent_movie,
                    "score": score,
                    "confidence": row["confidence"],
                    "lift": row["lift"]
                })

    if not recommendations:
        return pl.DataFrame({"movie": [], "score": [], "count": []})

    # Create DataFrame and aggregate scores for movies recommended multiple times
    rec_df = pl.DataFrame(recommendations)
    return rec_df.group_by("movie").agg([
        pl.col("score").mean().alias("avg_score"),
        pl.col("score").max().alias("max_score"),
        pl.len().alias("count")
    ]).sort("avg_score", descending=True)


def evaluate_recommendations(
    test_movies: list[str],
    recommendations: pl.DataFrame,
    verbose: bool = True
) -> dict[str, float | int]:
    """Evaluate recommendation quality using test data.

    Args:
        test_movies: List of movies the user actually watched (ground truth)
        recommendations: DataFrame with recommended movies
        verbose: Whether to log warnings

    Returns:
        Dictionary with evaluation metrics

    """
    if len(recommendations) == 0:
        if verbose:
            logger.warning("No recommendations to evaluate")
        return {
            "hits": 0,
            "total_recommendations": 0,
            "total_test_movies": len(test_movies)
        }

    # Get set of recommended movies
    recommended_movies = set(recommendations["movie"].to_list())
    test_movies_set = set(test_movies)

    # Calculate hits (movies that were both recommended and in test set)
    hits = len(recommended_movies & test_movies_set)

    return {
        "hits": hits,
        "total_recommendations": len(recommended_movies),
        "total_test_movies": len(test_movies_set)
    }


def evaluate_on_test_set(
    train_user_movies: dict[int, list[str]],
    test_user_movies: dict[int, list[str]],
    rules: pl.DataFrame,
) -> dict[str, float]:
    """Evaluate association rules on test set across all test users.

    Args:
        train_user_movies: Dictionary mapping User_Id to list of training movies
        test_user_movies: Dictionary mapping User_Id to list of test movies
        rules: Association rules DataFrame
        seed: Random seed for user sampling

    Returns:
        Dictionary with aggregated evaluation metrics

    """
    logger.info("Evaluating on test set...")

    # Get test users and optionally sample
    test_users = list(test_user_movies.keys())

    logger.info(f"Evaluating {len(test_users)} test users")

    users_with_recommendations = 0
    total_hits = 0
    total_recommendations = 0
    total_test_movies = 0

    for i, user_id in enumerate(test_users):
        if (i + 1) % 10000 == 0:
            logger.info(f"  Processed {i + 1}/{len(test_users)} users")

        # Get user's training and test movies
        train_movies = train_user_movies.get(user_id, [])
        test_movies = test_user_movies.get(user_id, [])

        if not train_movies or not test_movies:
            continue

        # Generate recommendations based on training movies
        recommendations = get_recommendations_for_movies(train_movies, rules)

        # Filter out movies already in training set
        train_set = set(train_movies)
        new_recommendations = recommendations.filter(
            ~pl.col("movie").is_in(train_set)
        )

        if len(new_recommendations) == 0:
            continue

        # Evaluate
        metrics = evaluate_recommendations(test_movies, new_recommendations, verbose=False)

        if metrics["total_recommendations"] > 0:
            users_with_recommendations += 1
            total_hits += metrics["hits"]
            total_recommendations += metrics["total_recommendations"]
            total_test_movies += metrics["total_test_movies"]

    # Calculate coverage
    user_coverage = users_with_recommendations / len(test_users) if test_users else 0.0

    # Aggregate metrics
    return {
        "user_coverage": user_coverage,
        "users_evaluated": len(test_users),
        "users_with_recommendations": users_with_recommendations,
        "total_hits": total_hits,
        "total_recommendations": total_recommendations,
        "total_test_movies": total_test_movies,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Frequent itemsets and association rules for movie recommendations"
    )

    # Data parameters
    parser.add_argument(
        "--min-rating",
        type=float,
        default=3.0,
        help="Minimum rating to include (default: 3.0)"
    )

    # Algorithm parameters
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.1,
        help="Minimum support threshold for frequent itemsets (default: 0.1)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="Minimum confidence threshold for association rules (default: 0.60)"
    )

    # Display parameters
    parser.add_argument(
        "--top-rules-display",
        type=int,
        default=10,
        help="Number of top rules to display (default: 10)"
    )

    # Train/test split parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of user's movies to use for training (default: 0.8)"
    )
    parser.add_argument(
        "--min-movies-split",
        type=int,
        default=100,
        help="Minimum movies for user to be in test split (default: 100)"
    )
    parser.add_argument(
        "--min-train-movies",
        type=int,
        default=50,
        help="Minimum movies required in training set (default: 50)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Step 1: Load and prepare data with train/test split
    logger.info("Creating user-movie dictionary with train/test split")
    train_user_movies, test_user_movies, qualifying_users = create_user_movie_df(
        min_rating=args.min_rating,
        seed=args.seed,
        min_movies_for_split=args.min_movies_split,
        min_train_movies=args.min_train_movies,
        train_ratio=args.train_ratio,
    )

    # Step 2: Find frequent itemsets using Apriori algorithm on TRAINING data
    logger.info("Finding frequent itemsets on training data")
    itemsets = find_frequent_itemsets(train_user_movies, min_support=args.min_support)

    # Step 3: Generate association rules from frequent itemsets
    logger.info("Generating association rules")
    rules = generate_association_rules(itemsets, min_confidence=args.min_confidence)

    # Step 4: Sort rules by confidence, then by lift
    rules_sorted = rules.sort(["confidence", "lift"], descending=True) if len(rules) > 0 else rules

    if len(rules) > 0:
        # Display top rules
        rules_display = rules_sorted.select(
            ["support", "confidence", "lift", "antecedents", "consequents"]
        ).head(args.top_rules_display)
        logger.info(f"\nAssociation rules (top {args.top_rules_display} by confidence):\n{rules_display}")
    else:
        logger.info("No association rules found with current thresholds")

    # Step 5: Evaluate on test set
    if test_user_movies and len(rules) > 0:
        results = evaluate_on_test_set(
            train_user_movies,
            test_user_movies,
            rules_sorted,
        )

        logger.info("EVALUATION RESULTS")
        logger.info(f"Users evaluated: {results['users_evaluated']}")
        logger.info(f"Users with recommendations: {results['users_with_recommendations']}")
        logger.info(f"User coverage: {results['user_coverage']:.2%}")
        logger.info(f"Total hits: {results['total_hits']}")
        logger.info(f"Total recommendations: {results['total_recommendations']}")
        logger.info(f"Total test movies: {results['total_test_movies']}")
        logger.info(f"Precision: {results['total_hits'] / results['total_recommendations']:.2%}")
        logger.info(f"Recall: {results['total_hits'] / results['total_test_movies']:.2%}")
    else:
        logger.warning("No test data or rules available for evaluation")
