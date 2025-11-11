import shutil
from pathlib import Path

import kagglehub

RAW_LOCATION = Path(__file__).parent / "raw"
RAW_LOCATION.mkdir(parents=True, exist_ok=True)

# Load the latest version
large_movie_path = kagglehub.dataset_download(
    handle="chaitanyahivlekar/large-movie-dataset",
    path="movies_dataset.csv",
)

shutil.copy(large_movie_path, RAW_LOCATION / "large_movie_dataset.csv")

rotten_tomatoes_path = kagglehub.dataset_download(
    handle="stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset",
    path="rotten_tomatoes_critic_reviews.csv",
)

shutil.copy(rotten_tomatoes_path, RAW_LOCATION / "rotten_tomatoes_critic_reviews.csv")

actors_path = kagglehub.dataset_download(
    handle="darinhawley/imdb-films-by-actor-for-10k-actors",
    path="actorfilms.csv",
)

shutil.copy(actors_path, RAW_LOCATION / "actorfilms.csv")
