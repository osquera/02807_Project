import argparse
from pathlib import Path

import numpy as np
import polars as pl
from helpers.logger import logger
from scipy import sparse
from sklearn.preprocessing import normalize

# Data paths
DATA_LOCATION = Path("data/clean")
DATA_PATH = DATA_LOCATION / "large_movie_dataset_clean.csv"

def load_ratings_data(
    min_rating: float | None = 3.0,
    exclude_user: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Load and prepare ratings data for collaborative filtering.

    Args:
        min_rating: Minimum rating threshold to filter (None = include all)
        exclude_user: User ID to exclude from training data (for testing)

    Returns:
        Tuple of (ratings_df, test_user_ratings_df)
        - ratings_df: DataFrame with User_Id, Movie_Name, Rating columns
        - test_user_ratings_df: Ratings for excluded user (None if no user excluded)

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

    # Collect the data
    # Ensure we have a long-format DataFrame (one row per rating)
    # Group by User_Id and Movie_Name to handle duplicates by averaging
    df = df.select([
        pl.col("User_Id"),
        pl.col("Movie_Name_Normalized").alias("Movie_Name"),
        pl.col("Rating")
    ]).group_by(["User_Id", "Movie_Name"]).agg(pl.col("Rating").mean())

    ratings_df = df.collect()

    # Save excluded user's ratings before removing them
    test_user_ratings = None
    if exclude_user is not None:
        test_user_ratings = ratings_df.filter(pl.col("User_Id") == exclude_user)
        if len(test_user_ratings) > 0:
            logger.info(f"Saved {len(test_user_ratings)} ratings for excluded user {exclude_user}")
        ratings_df = ratings_df.filter(pl.col("User_Id") != exclude_user)
        logger.info(f"Excluded user {exclude_user} from training data")

    logger.info(
        f"Loaded {len(ratings_df)} ratings from {ratings_df['User_Id'].n_unique()} users "
        f"for {ratings_df['Movie_Name'].n_unique()} movies"
    )

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
        (ratings, (row_indices, col_indices)),
        shape=(n_users, n_movies),
        dtype=np.float32
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
    # OPTIMIZATION: Use sparse matrix operations instead of converting to dense
    # This avoids MemoryError on large datasets

    # 1. Filter common_users_matrix to keep only entries with >= min_common_users
    common_users_matrix = common_users_matrix.tocsr()
    common_users_matrix.data[common_users_matrix.data < min_common_users] = 0
    common_users_matrix.eliminate_zeros()

    # 2. Create a binary mask from the filtered common_users_matrix
    mask = common_users_matrix.copy()
    mask.data[:] = 1.0

    # 3. Apply mask to similarity_matrix (element-wise multiplication)
    # This keeps only similarities where we have enough common users
    similarity_matrix = similarity_matrix.multiply(mask)

    # 4. Filter by min_similarity
    if min_similarity > 0:
        similarity_matrix = similarity_matrix.tocsr()
        similarity_matrix.data[similarity_matrix.data <= min_similarity] = 0
        similarity_matrix.eliminate_zeros()

    # 5. Get upper triangle to avoid duplicates and self-loops (diagonal)
    similarity_triu = sparse.triu(similarity_matrix, k=1)

    # 6. Convert to COO for efficient iteration
    similarity_coo = similarity_triu.tocoo()

    count = 0
    for i, j, sim in zip(similarity_coo.row, similarity_coo.col, similarity_coo.data, strict=True):
        movie1 = movies_list[i]
        movie2 = movies_list[j]
        similarities[(movie1, movie2)] = float(sim)
        similarities[(movie2, movie1)] = float(sim)
        count += 1

    logger.info(f"Computed {count:,} item-item similarity pairs")
    return similarities


def predict_rating(
    user_ratings: dict[str, float],
    target_movie: str,
    similarities: dict[tuple[str, str], float],
    top_k: int = 20,
    min_support: int = 1
) -> float | None:
    """Predict rating for a target movie based on similar items the user has rated.

    Args:
        user_ratings: Dictionary mapping movie names to user's ratings
        target_movie: Movie to predict rating for
        similarities: Item-item similarity dictionary
        top_k: Number of most similar items to use for prediction
        min_support: Minimum number of similar items required to make a prediction

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

    # Require minimum number of similar items for confidence
    if len(similar_items) < min_support:
        return None

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
    min_support: int = 1,
    min_predicted_rating: float | None = None,
) -> pl.DataFrame:
    """Generate top-N recommendations for a user.

    Args:
        user_ratings: Dictionary of movies the user has rated
        similarities: Item-item similarity dictionary
        top_k_similar: Number of similar items to use for each prediction
        min_support: Minimum number of similar items required for a prediction
        min_predicted_rating: Minimum predicted rating to include (None = no filter)

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

    logger.info(f"Found {len(candidate_movies)} candidate movies based on similarity")

    if not candidate_movies:
        logger.warning("No similar movies found for recommendations")
        return pl.DataFrame({"movie": [], "predicted_rating": []})

    logger.info(f"Generating predictions for {len(candidate_movies)} candidate movies...")

    # Predict ratings for candidate movies
    predictions = []
    filtered_by_support = 0
    filtered_by_rating = 0
    for movie in candidate_movies:
        predicted_rating = predict_rating(
            user_ratings,
            movie,
            similarities,
            top_k=top_k_similar,
            min_support=min_support
        )
        if predicted_rating is None:
            filtered_by_support += 1
            continue
        if min_predicted_rating is not None and predicted_rating < min_predicted_rating:
            filtered_by_rating += 1
            continue
        predictions.append({
            "movie": movie,
            "predicted_rating": predicted_rating
        })

    if filtered_by_support > 0:
        logger.info(
            f"Filtered {filtered_by_support} movies due to insufficient support "
            f"(< {min_support} similar items)"
        )
    if filtered_by_rating > 0:
        logger.info(f"Filtered {filtered_by_rating} movies due to low predicted rating (< {min_predicted_rating})")

    if not predictions:
        logger.warning("No predictions could be made")
        return pl.DataFrame({"movie": [], "predicted_rating": []})

    # Create DataFrame and sort by predicted rating
    recommendations_df = pl.DataFrame(predictions).sort(
        "predicted_rating",
        descending=True
    )

    logger.info(f"Generated {len(recommendations_df)} recommendations")
    return recommendations_df


def evaluate_recommendations(
    test_ratings: pl.DataFrame,
    recommendations: pl.DataFrame
) -> dict[str, float | None]:
    """Evaluate recommendation quality using test data.

    Args:
        test_ratings: Actual ratings from test user
        recommendations: Predicted ratings for recommended movies

    Returns:
        Dictionary with evaluation metrics

    """
    # Create a dictionary of actual ratings
    actual_ratings = dict(
        zip(test_ratings["Movie_Name"], test_ratings["Rating"], strict=True)
    )

    # Find movies that appear in both predictions and actuals
    common_movies = []
    for row in recommendations.iter_rows(named=True):
        movie = row["movie"]
        if movie in actual_ratings:
            common_movies.append({
                "movie": movie,
                "predicted": row["predicted_rating"],
                "actual": actual_ratings[movie]
            })

    if not common_movies:
        logger.warning("No overlap between recommendations and test ratings")
        return {
            "rmse": None,
            "mae": None,
            "common_movies": 0
        }

    # Calculate RMSE and MAE
    predictions = np.array([m["predicted"] for m in common_movies])
    actuals = np.array([m["actual"] for m in common_movies])

    rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
    mae = float(np.mean(np.abs(predictions - actuals)))

    return {
        "rmse": rmse,
        "mae": mae,
        "common_movies": len(common_movies)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collaborative filtering on merged dataset")

    parser.add_argument("--min-rating", type=float, default=3.0, help="Minimum rating to include")
    parser.add_argument("--exclude-user", type=int, default=1, help="User ID to hold out for testing (default: 1)")
    parser.add_argument("--no-test-user", action="store_true", help="Do not hold out a test user")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    parser.add_argument("--min-common-users", type=int, default=5, help="Minimum common users for similarity")
    parser.add_argument("--min-similarity", type=float, default=0.0, help="Minimum similarity threshold to keep")
    parser.add_argument(
        "--top-similar-items",
        type=int,
        default=20,
        help="How many similar items to use for prediction",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="Minimum number of similar items required to make a prediction (default: 1)",
    )
    parser.add_argument(
        "--min-predicted-rating",
        type=float,
        default=None,
        help="Minimum predicted rating to include in recommendations (default: None)",
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
    logger.info("Loading ratings data from cleaned dataset")
    ratings_df, test_user_ratings = load_ratings_data(
        min_rating=args.min_rating,
        exclude_user=None if args.no_test_user else args.exclude_user,
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
            logger.info(f"Splitting user ratings: {len(train_ratings)} for input, {len(eval_ratings)} for evaluation")
        else:
            train_ratings = test_user_ratings
            eval_ratings = None
            logger.info(f"Using all {len(train_ratings)} ratings for input (too few to split)")

        # Create user rating dictionary from TRAIN ratings
        user_ratings_dict = dict(
            zip(train_ratings["Movie_Name"], train_ratings["Rating"], strict=True)
        )

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
            min_support=args.min_support,
            min_predicted_rating=args.min_predicted_rating,
        )

        if len(recommendations) > 0:
            logger.info(f"Top {args.top_recommendations} recommendations for User {args.exclude_user}:")
            logger.info(f"\n{recommendations.head(args.top_recommendations)}")

            # Evaluate if we have test data
            if eval_ratings is not None and len(eval_ratings) > 0:
                metrics = evaluate_recommendations(eval_ratings, recommendations)
                common_movies_count = metrics.get("common_movies", 0)
                if common_movies_count and common_movies_count > 0:
                    logger.info("Evaluation Metrics (on held-out ratings):")
                    logger.info(f"  Common movies: {metrics['common_movies']}")
                    logger.info(f"  RMSE: {metrics['rmse']:.3f}")
                    logger.info(f"  MAE: {metrics['mae']:.3f}")
                else:
                    logger.info("Evaluation: No overlap between recommendations and held-out ratings")
        else:
            logger.warning("No recommendations could be generated for this user")
    else:
        logger.info("No test user specified. Use --exclude-user to set a test user or omit --no-test-user flag.")
