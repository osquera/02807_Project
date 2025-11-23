# 02807 Project


## Taskfile Installation

Taskfile is a task runner that helps automate repetitive tasks using a simple YAML configuration. To get Taskfile running, follow the installation instructions from [taskfile.dev](https://taskfile.dev).

Once installed, you can run tasks defined in `Taskfile.yml` using the `task` command.

## Running Tasks

To list all available tasks:
```
task --list
```

**Passing parameters to tasks:**

Some tasks accept command-line arguments. Use `--` to separate task arguments from script arguments:
```
task run-frequent-items -- --min-support 0.15 --min-confidence 0.70
```

To see available parameters for a task:
```
task run-frequent-items -- --help
```

### Data Management Tasks

**Complete data setup (recommended for first-time setup):**
```
task setup-data
```
Downloads, processes, and cleans all datasets in one command.

**Individual data tasks:**
```
task sync
```
Syncs Python dependencies using uv.

```
task download-data
```
Downloads and saves all datasets locally from Kaggle.

```
task clean-data
```
Cleans the downloaded datasets by removing nulls, invalid ratings, and mapping review scores to numeric values.

```
task scrape-rt-movies
```
Scrapes movie titles and descriptions from Rotten Tomatoes using the movie IDs from the reviews dataset. Uses 10 concurrent workers with rate limiting and supports resuming if interrupted. Creates `rotten_tomatoes_movie_details.csv` in `data/raw/`.

```
task retry-failed-scrapes
```
Retries scraping movies that previously failed during the main scraping process. Updates existing entries in the output file instead of creating duplicates - failed entries are replaced with successful results when possible.

```
task merge-data
```
Merges all cleaned datasets into a single comprehensive dataset using normalized movie titles as the merge key. Handles inconsistencies like case variations and year annotations (e.g., "Movie (2020)"). Aggregates reviews and actors into lists. Creates `movies_merged.csv` in `data/merged/`.

### Data Exploration Tasks

**Explore all datasets (raw and cleaned):**
```
task explore-data
```
Provides statistics and descriptions for both raw and cleaned versions of all datasets.

**Explore only raw datasets:**
```
task explore-data-raw
```
Provides statistics and descriptions for raw datasets only.

**Explore only cleaned datasets:**
```
task explore-data-clean
```
Provides statistics and descriptions for cleaned datasets only.
