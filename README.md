# 02807 Project


## Taskfile Installation

Taskfile is a task runner that helps automate repetitive tasks using a simple YAML configuration. To get Taskfile running, follow the installation instructions from [taskfile.dev](https://taskfile.dev).

Once installed, you can run tasks defined in `Taskfile.yml` using the `task` command.

## Running Tasks

To list all available tasks:
```
task --list
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
