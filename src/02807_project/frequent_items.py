import argparse
import random
from pathlib import Path

import polars as pl
from helpers.logger import logger

# Data paths
DATA_LOCATION = Path("data/merged")
DATA_PATH = DATA_LOCATION / "movies_merged.csv"

# Constants
ITEMSET_LENGTH_TWO = 2

def create_user_movie_df(
    sample_size: int | None = None,
    min_rating: float | None = None,
    exclude_user: int | None = None
) -> tuple[dict, list[str] | None]:
    """Create a dictionary mapping User_Id to list of movies they watched/rated.

    Args:
        sample_size: Number of users to include (None = all users)
        min_rating: Minimum rating threshold to filter movies (None = include all)
        exclude_user: User ID to exclude from the dataset (for testing)

    Returns:
        Tuple of (user_movies_dict, excluded_user_movies)
        - user_movies_dict: Dictionary with User_Id as keys and list of movie names as values
        - excluded_user_movies: List of movies watched by excluded user (None if no user excluded)

    """
    df = pl.scan_csv(DATA_PATH, infer_schema=True)
    df = df.filter(
            pl.col("rotten_tomatoes_link").is_not_null()
            & pl.col("lm_movie_name_original").is_not_null()
            & pl.col("af_film_original").is_not_null()
        )

    # Select and process columns to explode user IDs and ratings
    df = df.select([
        pl.col("lm_movie_name_original").alias("Movie_Name"),
        pl.col("lm_user_ids_list").str.split("|").alias("User_Id_List"),
        pl.col("lm_ratings_list").str.split("|").alias("Rating_List")
    ])

    # Explode both lists simultaneously
    df = df.explode(["User_Id_List", "Rating_List"])

    # Cast to appropriate types
    df = df.select([
        pl.col("Movie_Name"),
        pl.col("User_Id_List").cast(pl.Int64).alias("User_Id"),
        pl.col("Rating_List").cast(pl.Float64).alias("Rating")
    ])

    # Count rows before deduplication
    n_rows_raw = df.select(pl.len()).collect().item()

    # Handle duplicates by averaging ratings for same user-movie pair
    df = df.group_by(["User_Id", "Movie_Name"]).agg(pl.col("Rating").mean())

    # Count rows after deduplication
    n_rows_unique = df.select(pl.len()).collect().item()
    n_duplicates = n_rows_raw - n_rows_unique
    logger.info(f"Found {n_duplicates} duplicate user-movie entries (averaged)")

    # Filter by minimum rating if specified
    if min_rating is not None:
        total_ratings = n_rows_unique
        df = df.filter(pl.col("Rating") >= min_rating)
        filtered_ratings = df.select(pl.len()).collect().item()
        removed_ratings = total_ratings - filtered_ratings
        percentage = removed_ratings / total_ratings * 100
        logger.info(
            f"Filtering movies with rating >= {min_rating}: "
            f"removed {percentage:.1f}% of ratings"
        )

    # Create a dataframe with User_Id and list of Movie_Names they rated
    user_movies_df = df.group_by("User_Id").agg(pl.col("Movie_Name").alias("movies_watched")).sort("User_Id")

    # Take a subset if sample_size is specified
    if sample_size is not None:
        user_movies_df = user_movies_df.head(sample_size)

    user_movies_df = user_movies_df.collect()

    # Save excluded user's movies before removing them
    excluded_user_movies = None
    if exclude_user is not None:
        excluded_user_row = user_movies_df.filter(pl.col("User_Id") == exclude_user)
        if len(excluded_user_row) > 0:
            excluded_user_movies = excluded_user_row["movies_watched"][0]
            logger.info(f"Saved {len(excluded_user_movies)} movies for excluded user {exclude_user}")
        user_movies_df = user_movies_df.filter(pl.col("User_Id") != exclude_user)
        logger.info(f"Excluded user {exclude_user} from training data")

    logger.info(f"Created user-movie dataframe with {len(user_movies_df)} unique users")

    # Convert to dictionary
    user_dict = dict(zip(user_movies_df["User_Id"], user_movies_df["movies_watched"], strict=True))
    return user_dict, excluded_user_movies


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

    logger.info(f"Finding frequent itemsets with min_support={min_support} (min_count={min_count})")

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

    logger.info(f"Found {sum(1 for d in itemsets_data if d['length'] == ITEMSET_LENGTH_TWO)} frequent 2-itemsets")

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

    logger.info(f"Generated {len(rules_data)} association rules")

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
        logger.info("No recommendations found for the given movies")
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
    recommendations: pl.DataFrame
) -> dict[str, float | int]:
    """Evaluate recommendation quality using test data.

    Args:
        test_movies: List of movies the user actually watched (ground truth)
        recommendations: DataFrame with recommended movies

    Returns:
        Dictionary with evaluation metrics

    """
    if len(recommendations) == 0:
        logger.warning("No recommendations to evaluate")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "hits": 0,
            "total_recommendations": 0,
            "total_test_movies": len(test_movies)
        }

    # Get set of recommended movies
    recommended_movies = set(recommendations["movie"].to_list())
    test_movies_set = set(test_movies)

    # Calculate hits (movies that were both recommended and in test set)
    hits = len(recommended_movies & test_movies_set)

    # Calculate precision and recall
    precision = hits / len(recommended_movies) if len(recommended_movies) > 0 else 0.0
    recall = hits / len(test_movies_set) if len(test_movies_set) > 0 else 0.0

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hits": hits,
        "total_recommendations": len(recommended_movies),
        "total_test_movies": len(test_movies_set)
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Frequent itemsets and association rules for movie recommendations"
    )

    # Data parameters
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of users to analyze (default: None = all users)"
    )
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

    # Test case parameters
    parser.add_argument(
        "--test-user-id",
        type=int,
        default=1,
        help="User ID to hold out for testing (default: 1)"
    )
    parser.add_argument(
        "--top-recommendations",
        type=int,
        default=5,
        help="Number of recommendations to generate for test user (default: 5)"
    )
    parser.add_argument(
        "--no-test-user",
        action="store_true",
        help="Disable test user holdout"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)"
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of test user's movies to use for training (default: 0.8)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Step 1: Load and prepare data
    logger.info("Creating user-movie dictionary from cleaned dataset")
    user_movies, test_user_movies = create_user_movie_df(
        sample_size=args.sample_size,
        min_rating=args.min_rating,
        exclude_user=None if args.no_test_user else args.test_user_id
    )

    # Step 2: Find frequent itemsets using Apriori algorithm
    logger.info("Finding frequent itemsets")
    itemsets = find_frequent_itemsets(user_movies, min_support=args.min_support)

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

    # Step 5: Test case - generate recommendations for held-out user
    if test_user_movies is not None and len(rules) > 0:
        logger.info(f"Test Case: Generating recommendations for User {args.test_user_id}")
        logger.info(f"Test user watched {len(test_user_movies)} movies")

        # Split user's movies into train and test for evaluation
        n_movies = len(test_user_movies)
        n_train = int(n_movies * args.train_fraction)

        # Ensure at least one movie for training if we have movies
        if n_train < 1 and n_movies > 0:
            n_train = 1

        if n_train < n_movies:
            # Shuffle movies deterministically with seed
            # Convert to list to ensure it's mutable for shuffling
            movies_shuffled = list(test_user_movies)
            random.seed(args.seed)
            random.shuffle(movies_shuffled)

            train_movies = movies_shuffled[:n_train]
            eval_movies = movies_shuffled[n_train:]
            logger.info(f"Split user's movies: {len(train_movies)} for input, {len(eval_movies)} for evaluation")
        else:
            train_movies = test_user_movies
            eval_movies = []
            logger.info(f"Using all {len(train_movies)} movies for input (too few to split)")

        # Generate ALL recommendations based on TRAIN movies (for proper evaluation)
        all_recommendations = get_recommendations_for_movies(
            train_movies,
            rules_sorted
        )

        if len(all_recommendations) > 0:
            # Show only top N to user
            logger.info(f"Top {args.top_recommendations} recommendations for User {args.test_user_id}:")
            logger.info(f"\n{all_recommendations.head(args.top_recommendations)}")

            # Show which movies weren't in the train set (actual new recommendations)
            train_watched = set(train_movies)
            new_recs = all_recommendations.filter(
                ~pl.col("movie").is_in(train_watched)
            )
            logger.info(f"New recommendations (not in training set): {len(new_recs)}")
            if len(new_recs) > 0:
                logger.info(f"\n{new_recs.head(args.top_recommendations)}")

            # Evaluate against held-out movies if we have them
            if len(eval_movies) > 0:
                logger.info("\nEvaluating recommendations against held-out movies...")
                logger.info(f"Using all new {len(new_recs)} recommendations for evaluation")
                metrics = evaluate_recommendations(eval_movies, new_recs)
                logger.info("Evaluation Metrics:")
                logger.info(f"  Test movies: {metrics['total_test_movies']}")
                logger.info(f"  Recommendations: {metrics['total_recommendations']}")
                logger.info(f"  Hits: {metrics['hits']}")
                logger.info(f"  Precision: {metrics['precision']:.3f}")
                logger.info(f"  Recall: {metrics['recall']:.3f}")
                logger.info(f"  F1 Score: {metrics['f1']:.3f}")
        else:
            logger.info("No recommendations could be generated for this user")
