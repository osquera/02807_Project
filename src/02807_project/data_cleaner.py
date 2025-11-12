from pathlib import Path

import polars as pl
from helpers.logger import logger

RAW_LOCATION = Path("data/raw")
CLEAN_LOCATION = Path("data/clean")
CLEAN_LOCATION.mkdir(parents=True, exist_ok=True)

# Constants for data validation
MAX_RATING = 10.0
MIN_RATING = 0.0
MIN_YEAR = 1900
MAX_YEAR = 2030

# Mapping for Rotten Tomatoes review scores
REVIEW_SCORE_MAPPING = {
    "A+": 5.0,
    "A": 4.5,
    "A-": 4.0,
    "B+": 3.5,
    "B": 3.0,
    "B-": 2.5,
    "C+": 2.0,
    "C": 1.5,
    "C-": 1.0,
    "D+": 0.7,
    "D": 0.5,
    "D-": 0.3,
    "F": 0.0,
}


def clean_large_movie_dataset() -> None:
    """Clean the large movie dataset."""
    logger.info("🧹 Cleaning Large Movie Dataset...")

    raw_path = RAW_LOCATION / "large_movie_dataset.csv"
    clean_path = CLEAN_LOCATION / "large_movie_dataset_clean.csv"

    if not raw_path.exists():
        logger.warning("Raw large movie dataset not found, skipping")
        return

    df = pl.scan_csv(
        raw_path, schema_overrides={"User_Id": pl.Int64, "Movie_Name": pl.Utf8, "Rating": pl.Float64, "Genre": pl.Utf8}
    )

    # Clean: drop nulls, filter valid ratings (0-10)
    df_clean = (
        df.filter(pl.col("Rating").is_not_null())
        .filter(pl.col("User_Id").is_not_null())
        .filter(pl.col("Movie_Name").is_not_null())
        .filter(pl.col("Genre").is_not_null())
        .filter((pl.col("Rating") >= MIN_RATING) & (pl.col("Rating") <= MAX_RATING))
    )

    df_clean.sink_csv(clean_path)
    logger.info(f"✅ Saved cleaned dataset to {clean_path}")


def clean_rotten_tomatoes_dataset() -> None:
    """Clean the Rotten Tomatoes dataset."""
    logger.info("🧹 Cleaning Rotten Tomatoes Dataset...")

    raw_path = RAW_LOCATION / "rotten_tomatoes_critic_reviews.csv"
    clean_path = CLEAN_LOCATION / "rotten_tomatoes_critic_reviews_clean.csv"

    if not raw_path.exists():
        logger.warning("Raw Rotten Tomatoes dataset not found, skipping")
        return

    df = pl.scan_csv(
        raw_path,
        schema_overrides={
            "rotten_tomatoes_link": pl.Utf8,
            "critic_name": pl.Utf8,
            "top_critic": pl.Boolean,
            "publisher_name": pl.Utf8,
            "review_type": pl.Categorical,
            "review_score": pl.Utf8,  # Keep as string for mapping
            "review_date": pl.Datetime,
            "review_content": pl.Utf8,
        },
        ignore_errors=True,
    )

    # Clean: drop nulls in essential columns, map review scores
    df_clean = (
        df.filter(pl.col("rotten_tomatoes_link").is_not_null())
        .filter(pl.col("critic_name").is_not_null())
        .filter(pl.col("review_type").is_not_null())
        .filter(pl.col("review_content").is_not_null())
        .with_columns(
            pl.col("review_score")
            .map_elements(lambda x: REVIEW_SCORE_MAPPING.get(str(x).strip(), None), return_dtype=pl.Float64)
            .alias("review_score_numeric")
        )
        .filter(pl.col("review_score_numeric").is_not_null())
    )

    df_clean.sink_csv(clean_path)
    logger.info(f"✅ Saved cleaned dataset to {clean_path}")


def clean_actorfilms_dataset() -> None:
    """Clean the actor films dataset."""
    logger.info("🧹 Cleaning Actor Films Dataset...")

    raw_path = RAW_LOCATION / "actorfilms.csv"
    clean_path = CLEAN_LOCATION / "actorfilms_clean.csv"

    if not raw_path.exists():
        logger.warning("Raw actor films dataset not found, skipping")
        return

    df = pl.scan_csv(
        raw_path,
        schema_overrides={
            "Actor": pl.Utf8,
            "ActorID": pl.Utf8,
            "Film": pl.Utf8,
            "Year": pl.Int64,
            "Votes": pl.Int64,
            "Rating": pl.Float64,
            "FilmID": pl.Utf8,
        },
    )

    # Clean: drop nulls, filter valid ratings and years
    df_clean = (
        df.filter(pl.col("Actor").is_not_null())
        .filter(pl.col("Film").is_not_null())
        .filter(pl.col("Year").is_not_null())
        .filter(pl.col("Rating").is_not_null())
        .filter(pl.col("Votes").is_not_null())
        .filter((pl.col("Rating") >= MIN_RATING) & (pl.col("Rating") <= MAX_RATING))
        .filter((pl.col("Year") >= MIN_YEAR) & (pl.col("Year") <= MAX_YEAR))  # Reasonable year range
        .filter(pl.col("Votes") >= 0)
    )

    df_clean.sink_csv(clean_path)
    logger.info(f"✅ Saved cleaned dataset to {clean_path}")


if __name__ == "__main__":
    logger.info("🚀 Starting data cleaning process...")

    clean_large_movie_dataset()
    clean_rotten_tomatoes_dataset()
    clean_actorfilms_dataset()

    logger.info("🎉 Data cleaning completed!")
