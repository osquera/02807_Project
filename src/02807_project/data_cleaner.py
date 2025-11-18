from pathlib import Path
import re

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
YEAR_LENGTH = 4  # Length of valid year strings

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


def normalize_movie_title(title: str) -> str:
    """Normalize movie title from Large Movie Dataset format.

    Handles titles like:
    - "Matrix, The (1999)" -> "The Matrix"
    - "Bug's Life, A (1998)" -> "A Bug's Life"
    - "Seventh Seal, The (Sjunde inseglet, Det) (1957)" -> "The Seventh Seal"

    Args:
        title: Original movie title

    Returns:
        Normalized title with articles moved to front and foreign titles removed

    """
    if not title or not isinstance(title, str):
        return title

    # Remove year patterns at the end: (1999) or [1999]
    title_no_year = title

    title_no_year = re.sub(r"\s*[\(\[]?\d{4}[\)\]]?\s*$", "", title).strip()

    # Remove foreign language titles in parentheses
    # Pattern: remove anything like "(Foreign Title)" or "(Foreign, Title)"
    # Keep only the first part before any parenthetical foreign titles
    title_no_foreign = re.sub(r"\s*\([^)]*[^\w\s)][^)]*\)", "", title_no_year).strip()

    # Now handle article repositioning
    # Common articles in multiple languages
    articles = [
        "the",
        "a",
        "an",  # English
        "le",
        "la",
        "les",
        "l'",  # French
        "der",
        "die",
        "das",  # German
        "il",
        "lo",
        "i",
        "gli",  # Italian
        "el",
        "los",
        "las",  # Spanish
        "de",
        "het",  # Dutch
        "o",
        "os",
        "as",  # Portuguese
        "det",
        "den",
    ]  # Scandinavian

    # Check if title ends with ", Article"
    for article in articles:
        # Case-insensitive pattern: ", Article" at the end
        pattern = rf",\s*{re.escape(article)}\s*$"
        if re.search(pattern, title_no_foreign, re.IGNORECASE):
            # Extract the article and move it to the front
            parts = re.split(pattern, title_no_foreign, flags=re.IGNORECASE)
            if len(parts) >= 1:
                main_part = parts[0].strip()
                # Capitalize the article properly
                article_proper = article.capitalize() if article.lower() != "l'" else article
                title_no_foreign = f"{article_proper} {main_part}"
                break

    return title_no_foreign.strip()


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

    # Clean: drop nulls, filter valid ratings (0-10), normalize movie titles, extract year
    df_clean = (
        df.filter(pl.col("Rating").is_not_null())
        .filter(pl.col("User_Id").is_not_null())
        .filter(pl.col("Movie_Name").is_not_null())
        .filter(pl.col("Genre").is_not_null())
        .filter((pl.col("Rating") >= MIN_RATING) & (pl.col("Rating") <= MAX_RATING))
        .with_columns(
            [
                # Extract year first (before removing it)
                pl.col("Movie_Name")
                .str.extract(r"[\(\[]?(\d{4})[\)\]]?\s*$", group_index=1)
                .cast(pl.Int64, strict=False)
                .alias("Release_Year"),
                # Normalize title using native Polars string operations (much faster than map_elements)
                pl.col("Movie_Name")
                # Step 1: Remove year patterns at the end
                .str.replace(r"\s*[\(\[]?\d{4}[\)\]]?\s*$", "")
                # Step 2: Remove foreign language titles in parentheses
                .str.replace_all(r"\s*\([^)]*[^\w\s)][^)]*\)", "")
                # Step 3: Move articles to front - handle common English articles (case-insensitive)
                .str.replace(r"(?i)^(.*),\s*(The|A|An)$", "$2 $1", literal=False)
                # Step 4: Convert to lowercase for matching consistency
                .str.to_lowercase()
                # Step 5: Clean up whitespace
                .str.strip_chars()
                .str.replace_all(r"\s+", " ", literal=False)
                .alias("Movie_Name_Normalized"),
            ]
        )
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


def clean_rotten_tomatoes_movie_details() -> None:
    """Clean the scraped Rotten Tomatoes movie details dataset."""
    logger.info("🧹 Cleaning Rotten Tomatoes Movie Details...")

    raw_path = RAW_LOCATION / "rotten_tomatoes_movie_details.csv"
    clean_path = CLEAN_LOCATION / "rotten_tomatoes_movie_details_clean.csv"

    if not raw_path.exists():
        logger.warning("Raw Rotten Tomatoes movie details not found, skipping")
        return

    df = pl.scan_csv(
        raw_path,
        schema_overrides={
            "rotten_tomatoes_link": pl.Utf8,
            "title": pl.Utf8,
            "description": pl.Utf8,
            "release_year": pl.Utf8,  # Keep as string for validation
            "scrape_status": pl.Utf8,
        },
        ignore_errors=True,
    )

    # Clean: keep only successful scrapes, validate data
    df_clean = (
        df.filter(pl.col("scrape_status") == "success")
        .filter(pl.col("rotten_tomatoes_link").is_not_null())
        .filter(pl.col("title").is_not_null())
        .with_columns(
            # Clean release year: ensure it's a valid 4-digit year
            pl.when(
                (pl.col("release_year").is_not_null())
                & (pl.col("release_year").str.len_chars() == YEAR_LENGTH)
                & (pl.col("release_year").str.contains(r"^\d{4}$"))
                & (pl.col("release_year").cast(pl.Int64, strict=False).is_between(MIN_YEAR, MAX_YEAR))
            )
            .then(pl.col("release_year"))
            .otherwise(None)
            .alias("release_year")
        )
        .with_columns(
            # Clean description: remove excessive whitespace
            pl.col("description").str.strip_chars().str.replace_all(r"\s+", " ", literal=False).alias("description")
        )
        .drop("scrape_status")  # No longer needed after filtering
    )

    df_clean.sink_csv(clean_path)
    logger.info(f"✅ Saved cleaned movie details to {clean_path}")


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
    clean_rotten_tomatoes_movie_details()
    clean_actorfilms_dataset()

    logger.info("🎉 Data cleaning completed!")
