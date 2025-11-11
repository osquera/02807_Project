from pathlib import Path

import polars as pl
from helpers.logger import logger

RAW_LOCATION = Path("data/raw")


def explore_dataset(name: str, file_path: Path) -> None:
    logger.info(f"=== Exploring {name} ===")

    # Scan the CSV lazily
    # Define schema overrides based on dataset
    if "large_movie" in str(file_path):
        schema_overrides = {"User_Id": pl.Int64, "Movie_Name": pl.Utf8, "Rating": pl.Float32, "Genre": pl.Utf8}
    elif "rotten_tomatoes" in str(file_path):
        schema_overrides = {
            "rotten_tomatoes_link": pl.Utf8,
            "critic_name": pl.Utf8,
            "top_critic": pl.Boolean,
            "publisher_name": pl.Utf8,
            "review_type": pl.Categorical,
            "review_score": pl.Categorical,
            "review_date": pl.Datetime,
            "review_content": pl.Utf8,
        }
    elif "actorfilms" in str(file_path):
        schema_overrides = {
            "Actor": pl.Utf8,
            "ActorID": pl.Utf8,
            "Film": pl.Utf8,
            "Year": pl.Int64,
            "Votes": pl.Int64,
            "Rating": pl.Float64,
            "FilmID": pl.Utf8,
        }
    else:
        schema_overrides = None

    df_lazy = pl.scan_csv(file_path, schema_overrides=schema_overrides, ignore_errors=True)

    # Collect a small sample for exploration
    df_sample = df_lazy.limit(1000).collect()

    # Basic statistics
    logger.info("Basic statistics:")
    logger.info(str(df_sample.describe()))

    # Check for nulls
    null_counts = df_sample.null_count()
    logger.info(f"Null counts:{null_counts}")

    # Show first few rows
    logger.info("First 5 rows:")
    logger.info(str(df_sample.head(5)))


if __name__ == "__main__":
    datasets = [
        ("Large Movie Dataset", RAW_LOCATION / "large_movie_dataset.csv"),
        ("Rotten Tomatoes Reviews", RAW_LOCATION / "rotten_tomatoes_critic_reviews.csv"),
        ("Actor Films", RAW_LOCATION / "actorfilms.csv"),
    ]

    for name, path in datasets:
        if path.exists():
            explore_dataset(name, path)
        else:
            logger.warning(f"Dataset {name} not found at {path}")
