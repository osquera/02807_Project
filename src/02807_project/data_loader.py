import zipfile
from pathlib import Path

import kagglehub
import polars as pl

RAW_LOCATION = Path("data/raw")
RAW_LOCATION.mkdir(parents=True, exist_ok=True)

# Load the latest version
large_movie_path = Path(
    kagglehub.dataset_download(
        handle="chaitanyahivlekar/large-movie-dataset",
        path="movies_dataset.csv",
    )
)

with zipfile.ZipFile(large_movie_path, "r") as zip_ref:
    zip_ref.extract("movies_dataset.csv", RAW_LOCATION)

(RAW_LOCATION / "movies_dataset.csv").rename(RAW_LOCATION / "large_movie_dataset.raw.csv")

large_movie_dataset = pl.scan_csv(
    RAW_LOCATION / "large_movie_dataset.raw.csv",
    encoding="utf8-lossy",
    truncate_ragged_lines=True,
    ignore_errors=True,
    infer_schema_length=100000,
    quote_char='"',  # Enable quote handling for fields with commas
    schema_overrides={"User_Id": pl.Int64, "Movie_Name": pl.Utf8, "Rating": pl.Float64, "Genre": pl.Utf8},
)
large_movie_dataset.sink_csv(RAW_LOCATION / "large_movie_dataset.csv")
(RAW_LOCATION / "large_movie_dataset.raw.csv").unlink()

rotten_tomatoes_path = Path(
    kagglehub.dataset_download(
        handle="stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset",
        path="rotten_tomatoes_critic_reviews.csv",
    )
)


with zipfile.ZipFile(rotten_tomatoes_path, "r") as zip_ref:
    zip_ref.extract("rotten_tomatoes_critic_reviews.csv", RAW_LOCATION)

(RAW_LOCATION / "rotten_tomatoes_critic_reviews.csv").rename(RAW_LOCATION / "rotten_tomatoes_critic_reviews.raw.csv")

rotten_tomatoes_dataset = pl.scan_csv(
    RAW_LOCATION / "rotten_tomatoes_critic_reviews.raw.csv",
    encoding="utf8-lossy",
    truncate_ragged_lines=True,
    ignore_errors=True,
    infer_schema_length=100000,
    quote_char='"',  # Enable quote handling for fields with commas
    try_parse_dates=True,
    schema_overrides={
        "rotten_tomatoes_link": pl.Utf8,
        "critic_name": pl.Utf8,
        "top_critic": pl.Boolean,
        "publisher_name": pl.Utf8,
        "review_type": pl.Categorical,
        "review_score": pl.Categorical,
        "review_date": pl.Datetime,
        "review_content": pl.Utf8,
    },
)
rotten_tomatoes_dataset.sink_csv(RAW_LOCATION / "rotten_tomatoes_critic_reviews.csv")
(RAW_LOCATION / "rotten_tomatoes_critic_reviews.raw.csv").unlink()

actors_path = Path(
    kagglehub.dataset_download(
        handle="darinhawley/imdb-films-by-actor-for-10k-actors",
        path="actorfilms.csv",
    )
)

with zipfile.ZipFile(actors_path, "r") as zip_ref:
    zip_ref.extract("actorfilms.csv", RAW_LOCATION)

(RAW_LOCATION / "actorfilms.csv").rename(RAW_LOCATION / "actorfilms.raw.csv")

actors_dataset = pl.scan_csv(
    RAW_LOCATION / "actorfilms.raw.csv",
    encoding="utf8-lossy",
    truncate_ragged_lines=True,
    ignore_errors=True,
    infer_schema_length=100000,
    quote_char='"',  # Enable quote handling for fields with commas
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
actors_dataset.sink_csv(RAW_LOCATION / "actorfilms.csv")
(RAW_LOCATION / "actorfilms.raw.csv").unlink()
