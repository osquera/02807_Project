import kagglehub
from kagglehub import KaggleDatasetAdapter


# Load the latest version
large_movie_dataset = kagglehub.dataset_load(
  adapter=KaggleDatasetAdapter.POLARS,
  handle="chaitanyahivlekar/large-movie-dataset",
  path="movies_dataset.csv"
)

large_movie_dataset.write_csv("raw/large_movie_dataset.csv")
  
rotten_tomatoes_dataset = kagglehub.dataset_load(
  adapter=KaggleDatasetAdapter.POLARS,
  handle="stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset",
  path="rotten_tomatoes_critic_reviews.csv"
)

rotten_tomatoes_dataset.write_csv("raw/rotten_tomatoes_critic_reviews.csv")

actors_dataset = kagglehub.dataset_load(
  adapter=KaggleDatasetAdapter.POLARS,
  handle="darinhawley/imdb-films-by-actor-for-10k-actors",
  path="actorfilms.csv"
)

actors_dataset.write_csv("raw/actorfilms.csv")


