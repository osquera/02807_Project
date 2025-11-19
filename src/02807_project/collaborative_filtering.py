from pathlib import Path

import numpy as np
import polars as pl
from helpers.logger import logger
from tqdm import tqdm

# Data paths
DATA_LOCATION = Path("data/clean")
DATA_PATH = DATA_LOCATION / "large_movie_dataset_clean.csv"

# Analysis parameters - adjust these to tune the algorithm
SAMPLE_SIZE = 50000  # Number of users to analyze (None = all users)
MIN_RATING = 3.0  # Minimum rating to include (None = include all ratings)
MIN_COMMON_USERS = 5  # Minimum number of common users between items for similarity calculation
TOP_SIMILAR_ITEMS = 20  # Number of similar items to consider for predictions
TOP_RECOMMENDATIONS = 10  # Number of recommendations to generate
MIN_MOVIE_RATINGS = 50  # Only compute similarities for movies with at least this many ratings
MAX_MOVIES = 3000  # Limit to top N most-rated movies for similarity computation

# Test case parameters
TEST_USER_ID = 1  # User ID to hold out for testing (None = no test case)


def load_ratings_data(
    sample_size: int | None = None,
    min_rating: float | None = None,
    exclude_user: int | None = None
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Load and prepare ratings data for collaborative filtering.

    Args:
        sample_size: Number of users to include (None = all users)
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

    # Select only relevant columns
    df = df.select(["User_Id", "Movie_Name", "Rating"])

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
        # Sort to ensure deterministic user selection across runs
        unique_users = ratings_df["User_Id"].unique().sort()[:sample_size]
        # Convert to set to avoid is_in deprecation warning
        ratings_df = ratings_df.filter(pl.col("User_Id").is_in(set(unique_users.to_list()))).sort("User_Id")

    logger.info(
        f"Loaded {len(ratings_df)} ratings from {ratings_df['User_Id'].n_unique()} users "
        f"for {ratings_df['Movie_Name'].n_unique()} movies"
    )

    return ratings_df, test_user_ratings


def compute_item_similarity(
    ratings_df: pl.DataFrame,
    min_common_users: int = 5,
    min_similarity: float = 0.0,
    min_movie_ratings: int = 50,
    max_movies: int | None = 5000,
) -> dict[tuple[str, str], float]:
    """Compute pairwise similarity between all items using cosine similarity.

    Uses vectorized operations with numpy for efficiency.

    Args:
        ratings_df: DataFrame with User_Id, Movie_Name, Rating columns
        min_common_users: Minimum number of common users required to compute similarity
        min_similarity: Minimum similarity score to store (filters out weak similarities)
        min_movie_ratings: Only include movies with at least this many ratings
        max_movies: Limit to top N most-rated movies (None for no limit)

    Returns:
        Dictionary mapping (movie1, movie2) tuples to similarity scores

    """
    logger.info("Filtering to frequent movies to reduce computation...")

    # Count ratings per movie and filter to frequent movies
    movie_counts = (
        ratings_df.group_by("Movie_Name")
        .agg(pl.len().alias("rating_count"))
        .filter(pl.col("rating_count") >= min_movie_ratings)
        .sort("rating_count", descending=True)
    )

    # Optionally limit to top N movies
    if max_movies is not None:
        movie_counts = movie_counts.head(max_movies)

    frequent_movies_set = set(movie_counts["Movie_Name"].to_list())

    logger.info(
        f"Filtered to {len(frequent_movies_set)} movies (min {min_movie_ratings} ratings each)"
    )

    # Filter ratings to only include frequent movies
    # Using a set avoids the is_in deprecation warning
    filtered_ratings = ratings_df.filter(pl.col("Movie_Name").is_in(frequent_movies_set))

    logger.info("Computing item-item similarity matrix...")

    # Create user-item matrix using pivot with aggregation
    # Group by User_Id and Movie_Name, taking mean in case of duplicates
    ratings_agg = filtered_ratings.group_by(["User_Id", "Movie_Name"]).agg(
        pl.col("Rating").mean()
    )

    # Pivot to create user-item matrix
    user_item_matrix = ratings_agg.pivot(
        on="Movie_Name",
        index="User_Id",
        values="Rating"
    )

    # Sort movie names alphabetically for deterministic results across runs
    movies = sorted([col for col in user_item_matrix.columns if col != "User_Id"])
    logger.info(f"Computing similarities for {len(movies)} movies")

    # Convert to numpy for vectorized computation
    # Replace nulls with 0 and convert to numpy array
    matrix_values = user_item_matrix.select(movies).fill_null(0).to_numpy()

    # Create boolean mask for non-zero values (more memory efficient than int)
    non_zero_mask = matrix_values != 0

    logger.info("Computing pairwise similarities using vectorized operations...")

    similarities: dict[tuple[str, str], float] = {}
    n_movies = len(movies)

    # Compute similarities in batches to avoid memory issues
    batch_size = 500
    for i in tqdm(range(0, n_movies, batch_size), desc="Computing similarities"):
        end_i = min(i + batch_size, n_movies)

        # Get batch of movies
        batch_vectors = matrix_values[:, i:end_i]
        batch_norms = np.linalg.norm(batch_vectors, axis=0)
        batch_nonzero_mask = non_zero_mask[:, i:end_i]

        for j in range(end_i, n_movies):
            movie_j_vec = matrix_values[:, j]
            movie_j_norm = np.linalg.norm(movie_j_vec)

            if movie_j_norm == 0:
                continue

            # Compute dot products for entire batch at once
            dot_products = batch_vectors.T @ movie_j_vec  # Shape: (batch_size,)

            # Compute common users count for batch using boolean operations
            movie_j_nonzero_mask = non_zero_mask[:, j]
            # Count common non-zero entries: sum of (batch_mask AND movie_j_mask)
            common_users_counts = (batch_nonzero_mask & movie_j_nonzero_mask[:, np.newaxis]).sum(axis=0)

            # Compute similarities for valid pairs
            for k in range(len(dot_products)):
                movie_i_idx = i + k
                if common_users_counts[k] < min_common_users:
                    continue

                if batch_norms[k] == 0:
                    continue

                similarity = dot_products[k] / (batch_norms[k] * movie_j_norm)

                if similarity > min_similarity:
                    movie1 = movies[movie_i_idx]
                    movie2 = movies[j]
                    similarities[(movie1, movie2)] = similarity
                    similarities[(movie2, movie1)] = similarity

    logger.info(f"Computed {len(similarities) // 2} item-item similarity pairs")
    return similarities


def predict_rating(
    user_ratings: dict[str, float],
    target_movie: str,
    similarities: dict[tuple[str, str], float],
    top_k: int = 20
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
    all_movies: set[str],
    similarities: dict[tuple[str, str], float],
    top_k_similar: int = 20,
) -> pl.DataFrame:
    """Generate top-N recommendations for a user.

    Args:
        user_ratings: Dictionary of movies the user has rated
        all_movies: Set of all available movies
        similarities: Item-item similarity dictionary
        top_k_similar: Number of similar items to use for each prediction

    Returns:
        DataFrame with recommended movies and predicted ratings

    """
    # Find movies user hasn't rated
    rated_movies = set(user_ratings.keys())
    unrated_movies = all_movies - rated_movies

    logger.info(f"Generating predictions for {len(unrated_movies)} unrated movies...")

    # Predict ratings for unrated movies
    predictions = []
    for movie in unrated_movies:
        predicted_rating = predict_rating(
            user_ratings,
            movie,
            similarities,
            top_k=top_k_similar
        )
        if predicted_rating is not None:
            predictions.append({
                "movie": movie,
                "predicted_rating": predicted_rating
            })

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


if __name__ == "__main__":
    # Step 1: Load and prepare data
    logger.info("Loading ratings data from cleaned dataset")
    ratings_df, test_user_ratings = load_ratings_data(
        sample_size=SAMPLE_SIZE,
        min_rating=MIN_RATING,
        exclude_user=TEST_USER_ID
    )

    # Step 2: Compute item-item similarity matrix
    logger.info("Computing item similarity matrix")
    similarities = compute_item_similarity(
        ratings_df,
        min_common_users=MIN_COMMON_USERS,
        min_movie_ratings=MIN_MOVIE_RATINGS,
        max_movies=MAX_MOVIES,
    )

    # Get all available movies
    all_movies = set(ratings_df["Movie_Name"].unique().to_list())

    # Step 3: Test case - generate recommendations for held-out user
    if test_user_ratings is not None and len(test_user_ratings) > 0:
        logger.info(f"Test Case: Generating recommendations for User {TEST_USER_ID}")

        # Create user rating dictionary from test user's ratings
        user_ratings_dict = dict(
            zip(test_user_ratings["Movie_Name"], test_user_ratings["Rating"], strict=True)
        )

        logger.info(f"Test user has rated {len(user_ratings_dict)} movies")
        logger.info(f"Average rating: {test_user_ratings['Rating'].mean():.2f}")

        # Show sample of user's highest rated movies
        top_rated = test_user_ratings.sort("Rating", descending=True).head(5)
        logger.info("User's top 5 highest rated movies:")
        logger.info(f"\n{top_rated}")

        # Generate recommendations
        logger.info("Generating recommendations...")
        recommendations = get_recommendations(
            user_ratings_dict,
            all_movies,
            similarities,
            top_k_similar=TOP_SIMILAR_ITEMS,
        )

        if len(recommendations) > 0:
            logger.info(f"Top {TOP_RECOMMENDATIONS} recommendations for User {TEST_USER_ID}:")
            logger.info(f"\n{recommendations.head(TOP_RECOMMENDATIONS)}")

            # Evaluate if we have test data
            metrics = evaluate_recommendations(test_user_ratings, recommendations)
            common_movies_count = metrics.get("common_movies", 0)
            if common_movies_count and common_movies_count > 0:
                logger.info("Evaluation Metrics:")
                logger.info(f"  Common movies: {metrics['common_movies']}")
                logger.info(f"  RMSE: {metrics['rmse']:.3f}")
                logger.info(f"  MAE: {metrics['mae']:.3f}")
        else:
            logger.warning("No recommendations could be generated for this user")
    else:
        logger.info("No test user specified. Set TEST_USER_ID to enable testing.")
