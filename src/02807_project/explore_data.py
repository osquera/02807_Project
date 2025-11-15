import argparse
from pathlib import Path

import polars as pl
from helpers.logger import logger

RAW_LOCATION = Path("data/raw")
CLEAN_LOCATION = Path("data/clean")


def explore_dataset(name: str, file_path: Path) -> None:
    logger.info(f"=== Exploring {name} ===")

    # Scan the CSV lazily
    # Define schema overrides based on dataset
    if "large_movie" in str(file_path):
        schema_overrides = {"User_Id": pl.Int64, "Movie_Name": pl.Utf8, "Rating": pl.Float32, "Genre": pl.Utf8}
    elif "rotten_tomatoes" in str(file_path):
        if "movie_details" in str(file_path):
            # Scraped movie details
            if "clean" in str(file_path):
                schema_overrides = {
                    "rotten_tomatoes_link": pl.Utf8,
                    "title": pl.Utf8,
                    "description": pl.Utf8,
                    "release_year": pl.Utf8,
                }
            else:
                schema_overrides = {
                    "rotten_tomatoes_link": pl.Utf8,
                    "title": pl.Utf8,
                    "description": pl.Utf8,
                    "release_year": pl.Utf8,
                    "scrape_status": pl.Utf8,
                }
        elif "clean" in str(file_path):
            # Clean version has numeric review scores
            schema_overrides = {
                "rotten_tomatoes_link": pl.Utf8,
                "critic_name": pl.Utf8,
                "top_critic": pl.Boolean,
                "publisher_name": pl.Utf8,
                "review_type": pl.Categorical,
                "review_score": pl.Utf8,
                "review_score_numeric": pl.Float64,
                "review_date": pl.Datetime,
                "review_content": pl.Utf8,
            }
        else:
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

    # Get full dataset size
    total_rows = df_lazy.select(pl.len()).collect().item()
    logger.info(f"📊 Dataset size: {total_rows:,} total rows")
    logger.info(f"🔍 Sample size: {df_sample.shape[0]} rows x {df_sample.shape[1]} columns")
    logger.info("Basic statistics:")
    logger.info(str(df_sample.describe()))

    # Check for nulls
    null_counts = df_sample.null_count()
    logger.info(f"Null counts:{null_counts}")

    # Show first few rows
    logger.info("First 5 rows:")
    logger.info(str(df_sample.head(5)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore datasets")
    parser.add_argument(
        "--version",
        choices=["raw", "clean", "both"],
        default="both",
        help="Which version of the data to explore (default: both)",
    )
    args = parser.parse_args()

    if args.version == "raw":
        datasets = [
            ("Large Movie Dataset (Raw)", RAW_LOCATION / "large_movie_dataset.csv"),
            ("Rotten Tomatoes Reviews (Raw)", RAW_LOCATION / "rotten_tomatoes_critic_reviews.csv"),
            ("Rotten Tomatoes Movie Details (Raw)", RAW_LOCATION / "rotten_tomatoes_movie_details.csv"),
            ("Actor Films (Raw)", RAW_LOCATION / "actorfilms.csv"),
        ]
    elif args.version == "clean":
        datasets = [
            ("Large Movie Dataset (Clean)", CLEAN_LOCATION / "large_movie_dataset_clean.csv"),
            ("Rotten Tomatoes Reviews (Clean)", CLEAN_LOCATION / "rotten_tomatoes_critic_reviews_clean.csv"),
            ("Rotten Tomatoes Movie Details (Clean)", CLEAN_LOCATION / "rotten_tomatoes_movie_details_clean.csv"),
            ("Actor Films (Clean)", CLEAN_LOCATION / "actorfilms_clean.csv"),
        ]
    else:  # both
        datasets = [
            ("Large Movie Dataset (Raw)", RAW_LOCATION / "large_movie_dataset.csv"),
            ("Large Movie Dataset (Clean)", CLEAN_LOCATION / "large_movie_dataset_clean.csv"),
            ("Rotten Tomatoes Reviews (Raw)", RAW_LOCATION / "rotten_tomatoes_critic_reviews.csv"),
            ("Rotten Tomatoes Reviews (Clean)", CLEAN_LOCATION / "rotten_tomatoes_critic_reviews_clean.csv"),
            ("Rotten Tomatoes Movie Details (Raw)", RAW_LOCATION / "rotten_tomatoes_movie_details.csv"),
            ("Rotten Tomatoes Movie Details (Clean)", CLEAN_LOCATION / "rotten_tomatoes_movie_details_clean.csv"),
            ("Actor Films (Raw)", RAW_LOCATION / "actorfilms.csv"),
            ("Actor Films (Clean)", CLEAN_LOCATION / "actorfilms_clean.csv"),
        ]

    for name, path in datasets:
        if path.exists():
            explore_dataset(name, path)
        else:
            logger.warning(f"Dataset {name} not found at {path}")
