"""Merge all movie datasets into a single comprehensive dataset.

This script merges multiple movie datasets using normalized movie titles as the merge key.
It handles inconsistencies like case variations and year annotations in titles.
"""

import re
from pathlib import Path

import polars as pl
from helpers.logger import logger

CLEAN_LOCATION = Path("data/clean")
MERGED_LOCATION = Path("data/merged")
MERGED_LOCATION.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = MERGED_LOCATION / "movies_merged.csv"


def normalize_title(title: str) -> str:
    """Normalize movie title for matching.

    Removes year annotations, converts to lowercase, strips whitespace.

    Args:
        title: Original movie title

    Returns:
        Normalized title for matching

    """
    if not title or title is None:
        return ""

    # Remove year patterns like (2020) or [2020]
    normalized = re.sub(r"\s*[\(\[]?\d{4}[\)\]]?\s*$", "", str(title))

    # Convert to lowercase and strip whitespace
    normalized = normalized.lower().strip()

    # Remove extra whitespace
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized


def extract_year_from_title(title: str) -> int | None:
    """Extract year from title if present.

    Args:
        title: Movie title that may contain a year

    Returns:
        Extracted year or None

    """
    if not title or title is None:
        return None

    # Look for 4-digit year in parentheses or brackets at the end
    match = re.search(r"[\(\[]?(\d{4})[\)\]]?\s*$", str(title))
    if match:
        year = int(match.group(1))
        # Validate year is reasonable
        if 1900 <= year <= 2030:
            return year

    return None


def merge_rotten_tomatoes_data() -> pl.DataFrame:
    """Merge Rotten Tomatoes reviews with movie details.

    Returns:
        Merged Rotten Tomatoes dataset with aggregated reviews

    """
    logger.info("🔗 Merging Rotten Tomatoes datasets...")

    reviews_path = CLEAN_LOCATION / "rotten_tomatoes_critic_reviews_clean.csv"
    details_path = CLEAN_LOCATION / "rotten_tomatoes_movie_details_clean.csv"

    if not reviews_path.exists() or not details_path.exists():
        logger.warning("Rotten Tomatoes datasets not found, skipping")
        return pl.DataFrame()

    # Load movie details
    details = pl.read_csv(
        details_path,
        schema_overrides={
            "rotten_tomatoes_link": pl.Utf8,
            "title": pl.Utf8,
            "description": pl.Utf8,
            "release_year": pl.Utf8,
        },
    )

    # Load reviews
    reviews = pl.read_csv(
        reviews_path,
        schema_overrides={
            "rotten_tomatoes_link": pl.Utf8,
            "critic_name": pl.Utf8,
            "top_critic": pl.Boolean,
            "publisher_name": pl.Utf8,
            "review_type": pl.Utf8,
            "review_score": pl.Utf8,
            "review_score_numeric": pl.Float64,
            "review_date": pl.Utf8,
            "review_content": pl.Utf8,
        },
    )

    # Aggregate reviews per movie
    reviews_agg = reviews.group_by("rotten_tomatoes_link").agg(
        [
            pl.col("critic_name").alias("rt_critics_list"),
            pl.col("publisher_name").alias("rt_publishers_list"),
            pl.col("review_content").alias("rt_reviews_list"),
            pl.col("review_score_numeric").mean().alias("rt_avg_review_score"),
            pl.col("review_score_numeric").alias("rt_review_scores_list"),
            pl.col("top_critic").sum().alias("rt_top_critic_count"),
            pl.len().alias("rt_review_count"),
        ]
    )

    # Merge reviews with movie details
    merged = details.join(reviews_agg, on="rotten_tomatoes_link", how="left")

    # Add normalized title column
    merged = merged.with_columns(
        pl.col("title").map_elements(normalize_title, return_dtype=pl.Utf8).alias("title_normalized")
    )

    # Convert release_year to integer if possible
    merged = merged.with_columns(pl.col("release_year").cast(pl.Int64, strict=False).alias("rt_release_year"))

    logger.info(f"✅ Merged Rotten Tomatoes data: {len(merged)} movies with {reviews.height} total reviews")

    return merged


def prepare_large_movie_dataset() -> pl.DataFrame:
    """Prepare large movie dataset with normalized titles.

    Returns:
        Prepared dataset with normalized titles

    """
    logger.info("📊 Preparing Large Movie Dataset...")

    path = CLEAN_LOCATION / "large_movie_dataset_clean.csv"

    if not path.exists():
        logger.warning("Large movie dataset not found, skipping")
        return pl.DataFrame()

    df = pl.read_csv(
        path, schema_overrides={"User_Id": pl.Int64, "Movie_Name": pl.Utf8, "Rating": pl.Float64, "Genre": pl.Utf8}
    )

    # Extract year from movie name
    df = df.with_columns(
        pl.col("Movie_Name").map_elements(extract_year_from_title, return_dtype=pl.Int64).alias("lm_year_from_title")
    )

    # Normalize title
    df = df.with_columns(
        pl.col("Movie_Name").map_elements(normalize_title, return_dtype=pl.Utf8).alias("title_normalized")
    )

    # Aggregate by movie (multiple users rated same movie)
    df_agg = df.group_by("title_normalized").agg(
        [
            pl.col("Movie_Name").first().alias("lm_movie_name_original"),
            pl.col("Rating").mean().alias("lm_avg_rating"),
            pl.col("Rating").alias("lm_ratings_list"),
            pl.col("User_Id").alias("lm_user_ids_list"),
            pl.col("Genre").first().alias("lm_genre"),
            pl.col("lm_year_from_title").first().alias("lm_year_from_title"),
            pl.len().alias("lm_rating_count"),
        ]
    )

    logger.info(f"✅ Prepared Large Movie Dataset: {len(df_agg)} unique movies from {len(df)} ratings")

    return df_agg


def prepare_actorfilms_dataset() -> pl.DataFrame:
    """Prepare actor films dataset with normalized titles.

    Returns:
        Prepared dataset with aggregated actors per film

    """
    logger.info("🎬 Preparing Actor Films Dataset...")

    path = CLEAN_LOCATION / "actorfilms_clean.csv"

    if not path.exists():
        logger.warning("Actor films dataset not found, skipping")
        return pl.DataFrame()

    df = pl.read_csv(
        path,
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

    # Normalize film title
    df = df.with_columns(pl.col("Film").map_elements(normalize_title, return_dtype=pl.Utf8).alias("title_normalized"))

    # Aggregate by film (multiple actors per film)
    df_agg = df.group_by("title_normalized").agg(
        [
            pl.col("Film").first().alias("af_film_original"),
            pl.col("Actor").alias("af_actors_list"),
            pl.col("ActorID").alias("af_actor_ids_list"),
            pl.col("Year").first().alias("af_year"),
            pl.col("Rating").first().alias("af_rating"),
            pl.col("Votes").first().alias("af_votes"),
            pl.col("FilmID").first().alias("af_film_id"),
            pl.len().alias("af_actor_count"),
        ]
    )

    logger.info(f"✅ Prepared Actor Films Dataset: {len(df_agg)} unique films from {len(df)} actor-film records")

    return df_agg


def merge_all_datasets() -> None:
    """Merge all movie datasets into a single comprehensive dataset."""
    logger.info("🚀 Starting dataset merge process...")

    # Prepare individual datasets
    rt_data = merge_rotten_tomatoes_data()
    large_movie_data = prepare_large_movie_dataset()
    actorfilms_data = prepare_actorfilms_dataset()

    # Check if we have any data to merge
    datasets_available = []
    if not rt_data.is_empty():
        datasets_available.append("Rotten Tomatoes")
    if not large_movie_data.is_empty():
        datasets_available.append("Large Movie Dataset")
    if not actorfilms_data.is_empty():
        datasets_available.append("Actor Films")

    if not datasets_available:
        logger.error("❌ No datasets available to merge!")
        return

    logger.info(f"📋 Datasets available for merge: {', '.join(datasets_available)}")

    # Start with the dataset that has the most complete information (RT data with titles)
    if not rt_data.is_empty():
        merged = rt_data
        logger.info(f"Starting with Rotten Tomatoes data: {len(merged)} movies")
    elif not large_movie_data.is_empty():
        merged = large_movie_data
        logger.info(f"Starting with Large Movie Dataset: {len(merged)} movies")
    else:
        merged = actorfilms_data
        logger.info(f"Starting with Actor Films Dataset: {len(merged)} movies")

    # Merge with Large Movie Dataset
    if not large_movie_data.is_empty() and not merged.is_empty():
        before_count = len(merged)
        merged = merged.join(large_movie_data, on="title_normalized", how="full", coalesce=True)
        logger.info(f"✅ Merged with Large Movie Dataset: {before_count} → {len(merged)} movies")

    # Merge with Actor Films Dataset
    if not actorfilms_data.is_empty() and not merged.is_empty():
        before_count = len(merged)
        merged = merged.join(actorfilms_data, on="title_normalized", how="full", coalesce=True)
        logger.info(f"✅ Merged with Actor Films Dataset: {before_count} → {len(merged)} movies")

    # Add a consolidated title column (prefer RT title, then large movie, then actor film)
    title_columns = []
    if "title" in merged.columns:
        title_columns.append(pl.col("title"))
    if "lm_movie_name_original" in merged.columns:
        title_columns.append(pl.col("lm_movie_name_original"))
    if "af_film_original" in merged.columns:
        title_columns.append(pl.col("af_film_original"))

    if title_columns:
        merged = merged.with_columns(pl.coalesce(*title_columns).alias("movie_title"))

    # Add a consolidated year column (prefer RT year, then actor film year, then large movie year)
    year_columns = []
    if "rt_release_year" in merged.columns:
        year_columns.append(pl.col("rt_release_year"))
    if "af_year" in merged.columns:
        year_columns.append(pl.col("af_year"))
    if "lm_year_from_title" in merged.columns:
        year_columns.append(pl.col("lm_year_from_title"))

    if year_columns:
        merged = merged.with_columns(pl.coalesce(*year_columns).alias("release_year"))

    # Reorder columns to put consolidated fields first
    ordered_columns = ["title_normalized", "movie_title", "release_year"]
    remaining_columns = [col for col in merged.columns if col not in ordered_columns]
    merged = merged.select(ordered_columns + remaining_columns)

    # Convert list columns to string representations for CSV compatibility
    logger.info("📝 Converting list columns to strings for CSV export...")
    list_columns = []
    for col in merged.columns:
        if merged[col].dtype == pl.List:
            list_columns.append(col)
            # Convert list to pipe-separated string (or empty string if null)
            merged = merged.with_columns(
                pl.col(col)
                .map_elements(lambda x: "|".join(map(str, x)) if x is not None else "", return_dtype=pl.Utf8)
                .alias(col)
            )

    if list_columns:
        logger.info(f"   Converted {len(list_columns)} list columns: {', '.join(list_columns)}")

    # Save merged dataset
    merged.write_csv(OUTPUT_FILE)

    # Print summary statistics
    logger.info(f"🎉 Merge completed successfully!")
    logger.info(f"📊 Final dataset statistics:")
    logger.info(f"   Total movies: {len(merged):,}")
    logger.info(f"   Total columns: {len(merged.columns)}")

    # Count coverage from each dataset
    has_rt = False
    has_lm = False
    has_af = False

    if "rotten_tomatoes_link" in merged.columns:
        rt_coverage = merged.filter(pl.col("rotten_tomatoes_link").is_not_null()).height
        logger.info(f"   Movies with RT data: {rt_coverage:,} ({rt_coverage/len(merged)*100:.1f}%)")
        has_rt = True

    if "lm_movie_name_original" in merged.columns:
        lm_coverage = merged.filter(pl.col("lm_movie_name_original").is_not_null()).height
        logger.info(f"   Movies with Large Movie data: {lm_coverage:,} ({lm_coverage/len(merged)*100:.1f}%)")
        has_lm = True

    if "af_film_original" in merged.columns:
        af_coverage = merged.filter(pl.col("af_film_original").is_not_null()).height
        logger.info(f"   Movies with Actor Films data: {af_coverage:,} ({af_coverage/len(merged)*100:.1f}%)")
        has_af = True

    # Calculate intersection statistics
    logger.info(f"\n📈 Data Completeness Statistics:")

    # Movies in all three datasets
    if has_rt and has_lm and has_af:
        all_three = merged.filter(
            pl.col("rotten_tomatoes_link").is_not_null()
            & pl.col("lm_movie_name_original").is_not_null()
            & pl.col("af_film_original").is_not_null()
        ).height
        logger.info(f"   Movies in ALL 3 datasets: {all_three:,} ({all_three/len(merged)*100:.1f}%)")

    # Movies in exactly two datasets
    if has_rt and has_lm:
        rt_and_lm = merged.filter(
            pl.col("rotten_tomatoes_link").is_not_null()
            & pl.col("lm_movie_name_original").is_not_null()
            & (pl.col("af_film_original").is_null() if has_af else True)
        ).height
        logger.info(f"   Movies in RT + Large Movie only: {rt_and_lm:,} ({rt_and_lm/len(merged)*100:.1f}%)")

    if has_rt and has_af:
        rt_and_af = merged.filter(
            pl.col("rotten_tomatoes_link").is_not_null()
            & pl.col("af_film_original").is_not_null()
            & (pl.col("lm_movie_name_original").is_null() if has_lm else True)
        ).height
        logger.info(f"   Movies in RT + Actor Films only: {rt_and_af:,} ({rt_and_af/len(merged)*100:.1f}%)")

    if has_lm and has_af:
        lm_and_af = merged.filter(
            pl.col("lm_movie_name_original").is_not_null()
            & pl.col("af_film_original").is_not_null()
            & (pl.col("rotten_tomatoes_link").is_null() if has_rt else True)
        ).height
        logger.info(f"   Movies in Large Movie + Actor Films only: {lm_and_af:,} ({lm_and_af/len(merged)*100:.1f}%)")

    # Movies in only one dataset
    if has_rt:
        rt_only = merged.filter(
            pl.col("rotten_tomatoes_link").is_not_null()
            & (pl.col("lm_movie_name_original").is_null() if has_lm else True)
            & (pl.col("af_film_original").is_null() if has_af else True)
        ).height
        logger.info(f"   Movies in RT only: {rt_only:,} ({rt_only/len(merged)*100:.1f}%)")

    if has_lm:
        lm_only = merged.filter(
            pl.col("lm_movie_name_original").is_not_null()
            & (pl.col("rotten_tomatoes_link").is_null() if has_rt else True)
            & (pl.col("af_film_original").is_null() if has_af else True)
        ).height
        logger.info(f"   Movies in Large Movie only: {lm_only:,} ({lm_only/len(merged)*100:.1f}%)")

    if has_af:
        af_only = merged.filter(
            pl.col("af_film_original").is_not_null()
            & (pl.col("rotten_tomatoes_link").is_null() if has_rt else True)
            & (pl.col("lm_movie_name_original").is_null() if has_lm else True)
        ).height
        logger.info(f"   Movies in Actor Films only: {af_only:,} ({af_only/len(merged)*100:.1f}%)")

    # Per-column null statistics
    logger.info(f"\n📋 Column Completeness (non-null counts):")
    null_counts = merged.null_count()
    for col in merged.columns:
        non_null = len(merged) - null_counts[col][0]
        pct = non_null / len(merged) * 100
        logger.info(f"   {col}: {non_null:,} ({pct:.1f}%)")

    logger.info(f"\n💾 Saved merged dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    merge_all_datasets()
