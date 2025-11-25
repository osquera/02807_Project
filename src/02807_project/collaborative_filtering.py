import argparse
from pathlib import Path

import numpy as np
import polars as pl
from helpers.logger import logger
from scipy import sparse
from sklearn.preprocessing import normalize

# Data paths
DATA_LOCATION = Path("data/merged")
DATA_PATH = DATA_LOCATION / "movies_merged.csv"


def load_ratings_data(
    sample_size: int | None = None,
    min_rating: float | None = 0.0,
    exclude_user: int | None = None,
    seed: int = 42,
    test_split: bool = False,
    min_ratings_for_split: int = 20,
    min_train_ratings: int = 15,
    train_ratio: float = 0.8,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Load and prepare ratings data for collaborative filtering.

    Args:
        sample_size: Number of users to include (None = all users)
        min_rating: Minimum rating threshold to filter (None = include all)
        exclude_user: User ID to exclude from training data (for testing)
        seed: Random seed for sampling
        test_split: If True, split users' ratings into train/test sets
        min_ratings_for_split: Minimum ratings a user must have to be included in split
        min_train_ratings: Minimum ratings required in training set after split
        train_ratio: Ratio of ratings to use for training (e.g., 0.8 for 80/20 split)

    Returns:
        Tuple of (train_df, test_df)
        - train_df: Training ratings DataFrame
        - test_df: Test ratings DataFrame (None if test_split=False)

    """
    df = pl.scan_csv(DATA_PATH, infer_schema=True)
    df = df.filter(
        pl.col("rotten_tomatoes_link").is_not_null()
        & pl.col("lm_movie_name_original").is_not_null()
        & pl.col("af_film_original").is_not_null()
    )

    # Select and process columns to explode user IDs and ratings
    df = df.select(
        [
            pl.col("lm_movie_name_original").alias("Movie_Name"),
            pl.col("lm_user_ids_list").str.split("|").alias("User_Id_List"),
            pl.col("lm_ratings_list").str.split("|").alias("Rating_List"),
        ]
    )

    # Explode both lists simultaneously
    df = df.explode(["User_Id_List", "Rating_List"])

    # Cast to appropriate types
    df = df.select(
        [
            pl.col("Movie_Name"),
            pl.col("User_Id_List").cast(pl.Int64).alias("User_Id"),
            pl.col("Rating_List").cast(pl.Float64).alias("Rating"),
        ]
    )

    # Handle duplicates by averaging ratings for same user-movie pair
    df = df.group_by(["User_Id", "Movie_Name"]).agg(pl.col("Rating").mean())

    # Filter by minimum rating if specified
    if min_rating is not None:
        total_ratings = df.select(pl.len()).collect().item()
        df = df.filter(pl.col("Rating") >= min_rating)
        filtered_ratings = df.select(pl.len()).collect().item()
        removed_ratings = total_ratings - filtered_ratings
        percentage = removed_ratings / total_ratings * 100
        logger.info(f"Filtering movies with rating >= {min_rating}: removed {percentage:.1f}% of ratings")

    # Collect the data
    ratings_df = df.collect()

    # Save excluded user's ratings before removing them
    test_user_ratings = None
    if exclude_user is not None:
        test_user_ratings = ratings_df.filter(pl.col("User_Id") == exclude_user)
        if len(test_user_ratings) > 0:
            logger.info(f"Saved {len(test_user_ratings)} ratings for excluded user {exclude_user}")
        ratings_df = ratings_df.filter(pl.col("User_Id") != exclude_user)
        logger.info(f"Excluded user {exclude_user} from training data")

    # Apply sample size if specified
    if sample_size is not None:
        # Randomly sample users with seed
        # Sort unique users to ensure deterministic sampling with seed
        unique_users = ratings_df["User_Id"].unique().sort()
        if len(unique_users) > sample_size:
            sampled_users = unique_users.sample(n=sample_size, seed=seed)
            # Convert to list to avoid Polars deprecation warning
            ratings_df = ratings_df.filter(pl.col("User_Id").is_in(sampled_users.to_list()))
            logger.info(f"Sampled {sample_size} users with seed {seed}")

    logger.info(
        f"Loaded {len(ratings_df)} ratings from {ratings_df['User_Id'].n_unique()} users "
        f"for {ratings_df['Movie_Name'].n_unique()} movies"
    )

    # If test_split is enabled, split users with sufficient ratings
    if test_split:
        logger.info(f"Creating train/test split for users with {min_ratings_for_split}+ ratings")

        # Count ratings per user
        user_counts = ratings_df.group_by("User_Id").agg(pl.len().alias("count"))

        # Mark qualifying users
        qualifying_users_df = user_counts.filter(pl.col("count") >= min_ratings_for_split)
        qualifying_user_ids = set(qualifying_users_df["User_Id"].to_list())

        logger.info(f"Found {len(qualifying_user_ids)} users with {min_ratings_for_split}+ ratings")

        # Add a column to mark qualifying users
        ratings_df = ratings_df.with_columns(pl.col("User_Id").is_in(list(qualifying_user_ids)).alias("is_qualifying"))

        # Vectorized split function for each user group
        def split_user_ratings(group_df: pl.DataFrame) -> pl.DataFrame:
            user_id = group_df["User_Id"][0]
            is_qualifying = group_df["is_qualifying"][0]

            if not is_qualifying:
                # Non-qualifying users: all ratings go to training
                return group_df.with_columns(pl.lit("train").alias("split"))

            # Shuffle with seed for reproducibility
            group_df = group_df.sample(fraction=1.0, shuffle=True, seed=seed + user_id)

            n_ratings = len(group_df)
            n_train = int(n_ratings * train_ratio)

            # Ensure minimum train size
            if n_train < min_train_ratings:
                n_train = min(min_train_ratings, n_ratings - 1)

            # Skip if can't meet minimum requirements (put all in training)
            if n_train < min_train_ratings or n_ratings - n_train < 1:
                return group_df.with_columns(pl.lit("train").alias("split"))

            # Add row index and mark train/test
            group_df = group_df.with_row_index("row_idx")
            return group_df.with_columns(
                pl.when(pl.col("row_idx") < n_train).then(pl.lit("train")).otherwise(pl.lit("test")).alias("split")
            ).drop("row_idx")

        # Apply split to all user groups at once
        logger.info("Applying train/test split to all users...")
        ratings_with_split = ratings_df.group_by("User_Id", maintain_order=True).map_groups(split_user_ratings)

        # Split into train and test
        train_df = ratings_with_split.filter(pl.col("split") == "train").drop(["split", "is_qualifying"])
        test_df = ratings_with_split.filter(pl.col("split") == "test").drop(["split", "is_qualifying"])

        logger.info(f"Train/Test split complete: {len(train_df)} train ratings, {len(test_df)} test ratings")
        logger.info(f"Test set: {test_df['User_Id'].n_unique()} users, {test_df['Movie_Name'].n_unique()} movies")

        return train_df, test_df

    return ratings_df, test_user_ratings


def compute_item_similarity(
    ratings_df: pl.DataFrame,
    min_common_users: int = 5,
    min_similarity: float = 0.0,
) -> dict[tuple[str, str], float]:
    """Compute pairwise similarity between all items using sparse matrix operations.

    Args:
        ratings_df: DataFrame with User_Id, Movie_Name, Rating columns
        min_common_users: Minimum number of common users required to compute similarity
        min_similarity: Minimum similarity score to store (filters out weak similarities)

    Returns:
        Dictionary mapping (movie1, movie2) tuples to similarity scores

    """
    # Get list of all movies
    movies_list = ratings_df["Movie_Name"].unique().to_list()
    logger.info(f"Computing similarity for {len(movies_list)} movies")

    logger.info("Constructing sparse user-item matrix...")

    # Map User_Id and Movie_Name to indices
    # We need consistent mapping for movies
    movie_to_idx = {movie: i for i, movie in enumerate(movies_list)}

    # For users, we just need a mapping to 0..N-1
    unique_users = ratings_df["User_Id"].unique()
    user_to_idx = {user: i for i, user in enumerate(unique_users.to_list())}

    # Extract data for sparse matrix
    # Data is already deduplicated in load_ratings_data
    users = ratings_df["User_Id"].to_list()
    movies = ratings_df["Movie_Name"].to_list()
    ratings = ratings_df["Rating"].to_list()

    row_indices = [user_to_idx[u] for u in users]
    col_indices = [movie_to_idx[m] for m in movies]

    n_users = len(unique_users)
    n_movies = len(movies_list)

    # Create sparse matrix (rows=users, cols=movies)
    # Use float32 to save memory
    user_item_matrix = sparse.csr_matrix(
        (ratings, (row_indices, col_indices)), shape=(n_users, n_movies), dtype=np.float32
    )

    logger.info(f"Sparse matrix shape: {user_item_matrix.shape}, stored elements: {user_item_matrix.nnz}")

    logger.info("Computing item-item similarity using sparse matrix operations...")

    # 1. Normalize columns (items) to unit length for Cosine Similarity
    # CSC format is faster for column operations
    item_user_matrix = user_item_matrix.tocsc()

    # Compute column norms
    # We can use sklearn's normalize or do it manually
    # normalize(X, axis=0) normalizes columns
    item_user_matrix_norm = normalize(item_user_matrix, norm="l2", axis=0)  # type: ignore  # noqa: PGH003

    # 2. Compute Cosine Similarity: S = R_norm.T @ R_norm
    # Result is (n_movies, n_movies)
    # This computes dot product of normalized columns -> Cosine Similarity
    similarity_matrix = item_user_matrix_norm.T @ item_user_matrix_norm

    # 3. Compute Common Users Count: C = R_bin.T @ R_bin
    # Binarize matrix
    item_user_matrix_bin = item_user_matrix.copy()
    item_user_matrix_bin.data[:] = 1.0

    common_users_matrix = item_user_matrix_bin.T @ item_user_matrix_bin

    logger.info("Filtering similarities...")

    # Convert to dictionary, applying filters
    similarities: dict[tuple[str, str], float] = {}

    # Iterate the similarity matrix and check common users count
    sim_dense = similarity_matrix.toarray()
    common_dense = common_users_matrix.toarray()

    count = 0
    for i in range(n_movies):
        for j in range(i + 1, n_movies):
            if common_dense[i, j] < min_common_users:
                continue

            sim = sim_dense[i, j]
            if sim > min_similarity:
                movie1 = movies_list[i]
                movie2 = movies_list[j]
                similarities[(movie1, movie2)] = float(sim)
                similarities[(movie2, movie1)] = float(sim)
                count += 1

    logger.info(f"Computed {count:,} item-item similarity pairs")
    return similarities


def predict_rating(
    user_ratings: dict[str, float], target_movie: str, similarities: dict[tuple[str, str], float], top_k: int = 20
) -> float | None:
    """Predict rating for a target movie based on similar items the user has rated.

    Args:
        user_ratings: Dictionary mapping movie names to user's ratings
        target_movie: Movie to predict rating for
        similarities: Item-item similarity dictionary
        top_k: Number of most similar items to use for prediction

    Returns:
        Predicted rating or None if prediction cannot be made

    """
    # Find rated movies similar to target movie
    similar_items = []

    for rated_movie, user_rating in user_ratings.items():
        if rated_movie == target_movie:
            continue

        # Get similarity between target and rated movie
        sim = similarities.get((target_movie, rated_movie))
        if sim is not None and sim > 0:
            similar_items.append((rated_movie, user_rating, sim))

    if not similar_items:
        return None

    # Sort by similarity and take top k
    similar_items.sort(key=lambda x: x[2], reverse=True)
    similar_items = similar_items[:top_k]

    # Weighted average prediction
    numerator = sum(rating * similarity for _, rating, similarity in similar_items)
    denominator = sum(similarity for _, _, similarity in similar_items)

    if denominator == 0:
        return None

    return numerator / denominator


def get_recommendations(
    user_ratings: dict[str, float],
    similarities: dict[tuple[str, str], float],
    top_k_similar: int = 20,
    verbose: bool = True,
) -> pl.DataFrame:
    """Generate top-N recommendations for a user.

    Args:
        user_ratings: Dictionary of movies the user has rated
        similarities: Item-item similarity dictionary
        top_k_similar: Number of similar items to use for each prediction
        verbose: Whether to log progress messages

    Returns:
        DataFrame with recommended movies and predicted ratings

    """
    # Optimization: Only consider movies that are similar to what the user has rated
    # Instead of iterating all unrated movies, we find neighbors of rated movies

    candidate_movies = set()
    rated_movies_set = set(user_ratings.keys())

    # Iterate similarities keys to find neighbors of rated movies
    for m1, m2 in similarities:
        if m1 in rated_movies_set and m2 not in rated_movies_set:
            candidate_movies.add(m2)
        elif m2 in rated_movies_set and m1 not in rated_movies_set:
            candidate_movies.add(m1)

    if verbose:
        logger.info(f"Found {len(candidate_movies)} candidate movies based on similarity")

    if not candidate_movies:
        if verbose:
            logger.warning("No similar movies found for recommendations")
        return pl.DataFrame({"movie": [], "predicted_rating": []})

    if verbose:
        logger.info(f"Generating predictions for {len(candidate_movies)} candidate movies...")

    # Predict ratings for candidate movies
    predictions = []
    for movie in candidate_movies:
        predicted_rating = predict_rating(user_ratings, movie, similarities, top_k=top_k_similar)
        if predicted_rating is not None:
            predictions.append({"movie": movie, "predicted_rating": predicted_rating})

    if not predictions:
        if verbose:
            logger.warning("No predictions could be made")
        return pl.DataFrame({"movie": [], "predicted_rating": []})

    # Create DataFrame and sort by predicted rating
    recommendations_df = pl.DataFrame(predictions).sort("predicted_rating", descending=True)

    if verbose:
        logger.info(f"Generated {len(recommendations_df)} recommendations")
    return recommendations_df


def evaluate_recommendations(
    test_ratings: pl.DataFrame,
    recommendations: pl.DataFrame,
    precision_k: int = 10,
    relevant_threshold: float = 4.0,
    verbose: bool = True,
) -> dict[str, float | None]:
    """Evaluate recommendation quality using test data.

    Args:
        test_ratings: Actual ratings from test user
        recommendations: Predicted ratings for recommended movies
        precision_k: Number of top recommendations to use for precision calculation
        relevant_threshold: Rating threshold for considering a movie "relevant"
        verbose: Whether to log warnings

    Returns:
        Dictionary with evaluation metrics

    """
    # Create a dictionary of actual ratings
    actual_ratings = dict(zip(test_ratings["Movie_Name"], test_ratings["Rating"], strict=True))

    # Find movies that appear in both predictions and actuals
    common_movies = []
    for row in recommendations.iter_rows(named=True):
        movie = row["movie"]
        if movie in actual_ratings:
            common_movies.append(
                {"movie": movie, "predicted": row["predicted_rating"], "actual": actual_ratings[movie]}
            )

    if not common_movies:
        if verbose:
            logger.warning("No overlap between recommendations and test ratings")
        return {"rmse": None, "mae": None, "precision_at_k": None, "common_movies": 0}

    # Calculate RMSE and MAE
    predictions = np.array([m["predicted"] for m in common_movies])
    actuals = np.array([m["actual"] for m in common_movies])

    rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
    mae = float(np.mean(np.abs(predictions - actuals)))

    # Calculate Precision@K
    # Sort by predicted rating and take top K
    common_movies_sorted = sorted(common_movies, key=lambda x: x["predicted"], reverse=True)
    top_k = common_movies_sorted[:precision_k]

    if top_k:
        relevant_in_top_k = sum(1 for m in top_k if m["actual"] >= relevant_threshold)
        precision = relevant_in_top_k / len(top_k)
    else:
        precision = None

    return {"rmse": rmse, "mae": mae, "precision_at_k": precision, "common_movies": len(common_movies)}


def evaluate_on_test_set(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    similarities: dict[tuple[str, str], float],
    top_k_similar: int = 20,
    precision_k: int = 10,
    relevant_threshold: float = 4.0,
    max_test_users: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate collaborative filtering on test set across all test users.

    Args:
        train_df: Training ratings DataFrame
        test_df: Test ratings DataFrame
        similarities: Item-item similarity dictionary
        top_k_similar: Number of similar items to use for prediction
        precision_k: Number of top recommendations for precision calculation
        relevant_threshold: Rating threshold for relevance
        max_test_users: Maximum number of test users to evaluate (None = all)
        seed: Random seed for user sampling

    Returns:
        Dictionary with aggregated evaluation metrics

    """
    logger.info("Evaluating on test set...")

    # Get test users and optionally sample
    test_users = test_df["User_Id"].unique().sort()
    total_test_users = len(test_users)

    if max_test_users is not None and total_test_users > max_test_users:
        logger.info(f"Sampling {max_test_users} users from {total_test_users} total test users")
        test_users = test_users.sample(n=max_test_users, seed=seed)

    test_users_list = test_users.to_list()
    logger.info(f"Evaluating {len(test_users_list)} test users")

    # Pre-group train and test DataFrames by User_Id for O(1) lookup
    logger.info("Pre-grouping train and test data by user...")

    # Filter to only users we're evaluating
    train_filtered = train_df.filter(pl.col("User_Id").is_in(test_users_list))
    test_filtered = test_df.filter(pl.col("User_Id").is_in(test_users_list))

    # Create dictionaries: user_id -> DataFrame of ratings
    # Note: group_by returns (key,) tuples, so we need to extract the first element
    train_by_user = {key[0]: group for key, group in train_filtered.group_by("User_Id")}
    test_by_user = {key[0]: group for key, group in test_filtered.group_by("User_Id")}

    logger.info(f"Pre-grouping complete. Train groups: {len(train_by_user)}, Test groups: {len(test_by_user)}")

    all_rmse = []
    all_mae = []
    all_precision = []
    users_with_predictions = 0
    total_predictions = 0

    for i, user_id in enumerate(test_users_list):
        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i + 1}/{len(test_users_list)} users")

        # Get user's training and test ratings from pre-grouped dictionaries
        user_train = train_by_user.get(user_id)
        user_test = test_by_user.get(user_id)

        if user_train is None or user_test is None or len(user_train) == 0 or len(user_test) == 0:
            continue

        # Create user ratings dictionary
        user_ratings_dict = dict(zip(user_train["Movie_Name"], user_train["Rating"], strict=True))

        # Generate predictions (suppress log messages for speed)
        recommendations = get_recommendations(
            user_ratings_dict, similarities, top_k_similar=top_k_similar, verbose=False
        )

        if len(recommendations) == 0:
            continue

        # Evaluate
        metrics = evaluate_recommendations(
            user_test, recommendations, precision_k=precision_k, relevant_threshold=relevant_threshold, verbose=False
        )

        if metrics["common_movies"] and metrics["common_movies"] > 0:
            users_with_predictions += 1
            total_predictions += metrics["common_movies"]

            if metrics["rmse"] is not None:
                all_rmse.append(metrics["rmse"])
            if metrics["mae"] is not None:
                all_mae.append(metrics["mae"])
            if metrics["precision_at_k"] is not None:
                all_precision.append(metrics["precision_at_k"])

    # Calculate coverage
    user_coverage = users_with_predictions / len(test_users_list) if test_users_list else 0.0

    # Aggregate metrics
    return {
        "mean_rmse": float(np.mean(all_rmse)) if all_rmse else 0.0,
        "mean_mae": float(np.mean(all_mae)) if all_mae else 0.0,
        "mean_precision_at_k": float(np.mean(all_precision)) if all_precision else 0.0,
        "user_coverage": user_coverage,
        "users_evaluated": len(test_users_list),
        "users_with_predictions": users_with_predictions,
        "total_predictions": total_predictions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collaborative filtering on merged dataset")

    parser.add_argument("--sample-size", type=int, default=None, help="Number of users to sample (None = all)")
    parser.add_argument("--min-rating", type=float, default=0.0, help="Minimum rating to include")
    parser.add_argument("--exclude-user", type=int, default=1, help="User ID to hold out for testing (default: 1)")
    parser.add_argument("--no-test-user", action="store_true", help="Do not hold out a test user")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    # Train/test split parameters
    parser.add_argument("--test-split", action="store_true", help="Enable train/test split for evaluation")
    parser.add_argument("--min-ratings-split", type=int, default=20, help="Min ratings for user to be in test split")
    parser.add_argument("--min-train-ratings", type=int, default=15, help="Min ratings required in training set")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/test split ratio (default: 0.8)")
    parser.add_argument("--max-test-users", type=int, default=50000, help="Max test users to evaluate (default: 50000)")
    parser.add_argument("--precision-k", type=int, default=10, help="K for Precision@K calculation")
    parser.add_argument("--relevant-threshold", type=float, default=4.0, help="Rating threshold for relevance")

    parser.add_argument("--min-common-users", type=int, default=5, help="Minimum common users for similarity")
    parser.add_argument("--min-similarity", type=float, default=0.0, help="Minimum similarity threshold to keep")
    parser.add_argument(
        "--top-similar-items",
        type=int,
        default=20,
        help="How many similar items to use for prediction",
    )
    parser.add_argument(
        "--top-recommendations",
        type=int,
        default=10,
        help="How many recommendations to show",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Step 1: Load ratings data
    logger.info("Loading ratings data from merged dataset")

    if args.test_split:
        # Use train/test split mode
        train_df, test_df = load_ratings_data(
            sample_size=args.sample_size,
            min_rating=args.min_rating,
            seed=args.seed,
            test_split=True,
            min_ratings_for_split=args.min_ratings_split,
            min_train_ratings=args.min_train_ratings,
            train_ratio=args.train_ratio,
        )

        # Step 2: Compute item-item similarity matrix on TRAINING data
        logger.info("Computing item similarity matrix on training data")
        similarities = compute_item_similarity(
            train_df,
            min_common_users=args.min_common_users,
            min_similarity=args.min_similarity,
        )

        # Step 3: Evaluate on test set
        if test_df is not None and len(test_df) > 0:
            results = evaluate_on_test_set(
                train_df,
                test_df,
                similarities,
                top_k_similar=args.top_similar_items,
                precision_k=args.precision_k,
                relevant_threshold=args.relevant_threshold,
                max_test_users=args.max_test_users,
                seed=args.seed,
            )

            logger.info("\n" + "=" * 80)
            logger.info("EVALUATION RESULTS")
            logger.info("=" * 80)
            logger.info(f"Users evaluated: {results['users_evaluated']}")
            logger.info(f"Users with predictions: {results['users_with_predictions']}")
            logger.info(f"User coverage: {results['user_coverage']:.2%}")
            logger.info(f"Total predictions made: {results['total_predictions']}")
            logger.info(f"Mean RMSE: {results['mean_rmse']:.3f}")
            logger.info(f"Mean MAE: {results['mean_mae']:.3f}")
            logger.info(f"Mean Precision@{args.precision_k}: {results['mean_precision_at_k']:.3f}")
            logger.info("=" * 80)
        else:
            logger.warning("No test data available for evaluation")
    else:
        # Use original single-user mode
        ratings_df, test_user_ratings = load_ratings_data(
            sample_size=args.sample_size,
            min_rating=args.min_rating,
            exclude_user=None if args.no_test_user else args.exclude_user,
            seed=args.seed,
        )

        # Step 2: Compute item-item similarity matrix
        logger.info("Computing item similarity matrix")
        similarities = compute_item_similarity(
            ratings_df,
            min_common_users=args.min_common_users,
            min_similarity=args.min_similarity,
        )

        # Get all available movies
        all_movies = set(ratings_df["Movie_Name"].unique().to_list())

        # Step 3: Test case - generate recommendations for held-out user
        if test_user_ratings is not None and len(test_user_ratings) > 0:
            logger.info(f"Test Case: Generating recommendations for User {args.exclude_user}")

            # Split user ratings into input (for recommendation) and test (for evaluation)
            # Sort first to ensure deterministic split with seed
            test_user_ratings = test_user_ratings.sort("Movie_Name").sample(fraction=1.0, shuffle=True, seed=args.seed)
            n_ratings = len(test_user_ratings)

            # Use 80% for input, 20% for evaluation
            n_train = int(n_ratings * 0.8)

            # Ensure at least one rating for input if we have ratings
            if n_train < 1 and n_ratings > 0:
                n_train = 1

            if n_train < n_ratings:
                train_ratings = test_user_ratings.head(n_train)
                eval_ratings = test_user_ratings.tail(n_ratings - n_train)
                logger.info(
                    f"Splitting user ratings: {len(train_ratings)} for input, {len(eval_ratings)} for evaluation"
                )
            else:
                train_ratings = test_user_ratings
                eval_ratings = None
                logger.info(f"Using all {len(train_ratings)} ratings for input (too few to split)")

            # Create user rating dictionary from TRAIN ratings
            user_ratings_dict = dict(zip(train_ratings["Movie_Name"], train_ratings["Rating"], strict=True))

            logger.info(f"User has rated {len(user_ratings_dict)} movies (in input set)")
            logger.info(f"Average rating: {train_ratings['Rating'].mean():.2f}")

            # Show sample of user's highest rated movies
            top_rated = train_ratings.sort("Rating", descending=True).head(5)
            logger.info("User's top 5 highest rated movies (from input set):")
            logger.info(f"\n{top_rated}")

            # Generate recommendations
            logger.info("Generating recommendations...")
            recommendations = get_recommendations(
                user_ratings_dict,
                similarities,
                top_k_similar=args.top_similar_items,
            )

            if len(recommendations) > 0:
                logger.info(f"Top {args.top_recommendations} recommendations for User {args.exclude_user}:")
                logger.info(f"\n{recommendations.head(args.top_recommendations)}")

                # Evaluate if we have test data
                if eval_ratings is not None and len(eval_ratings) > 0:
                    metrics = evaluate_recommendations(
                        eval_ratings,
                        recommendations,
                        precision_k=args.precision_k,
                        relevant_threshold=args.relevant_threshold,
                    )
                    common_movies_count = metrics.get("common_movies", 0)
                    if common_movies_count and common_movies_count > 0:
                        logger.info("Evaluation Metrics (on held-out ratings):")
                        logger.info(f"  Common movies: {metrics['common_movies']}")
                        logger.info(f"  RMSE: {metrics['rmse']:.3f}")
                        logger.info(f"  MAE: {metrics['mae']:.3f}")
                        if metrics["precision_at_k"] is not None:
                            logger.info(f"  Precision@{args.precision_k}: {metrics['precision_at_k']:.3f}")
                    else:
                        logger.info("Evaluation: No overlap between recommendations and held-out ratings")
            else:
                logger.warning("No recommendations could be generated for this user")
        else:
            logger.info("No test user specified. Use --exclude-user to set a test user or omit --no-test-user flag.")
