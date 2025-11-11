# student-project-template

Golden image for student projects leveraging private Google Artifact Registry packages and common ML/EEG toolkits.

## Overview

This template project comes preconfigured to:

- Use private Python packages from Google Artifact Registry via `uv`.
- Include common ML/Electrophysiology dependencies (MNE, PyTorch, etc.).
- Provide handy Taskfile recipes for auth & sync operations.

## Prerequisites

1. A Google service account JSON key named google-service-account-key.json placed in your project root.
2. [Taskfile](https://taskfile.dev/) v3 installed (used to run predefined tasks).

## Getting Started

1. **Install dependencies & Taskfile**  
   Run the setup script to validate the service account key, install `task` (if missing), and run an initial “hello” check:

   ```shell
   bash scripts/setup.sh
   ```

2. **Authenticate & Sync**  
   Once setup completes, authenticate to GCP and sync Python artifacts with:

   ```shell
   task sync
   ```
   This will:
   - Activate your service account using `gcloud`  
   - Write an authenticated `.netrc` entry for Artifact Registry  
   - Run `uv sync` to pull private packages

## Directory Structure

This repository uses a src/ layout. The key files are:

```
.
├── google-service-account-key.json  # your GCP SA key (git-ignored)
├── Taskfile.yml                     # Task recipes
├── scripts/
│   ├── gcloud_auth.sh               # auth + netrc writer
│   └── setup.sh                     # initial install & checks
├── src/
│   └── student-project-template/
│       ├── __init__.py
│       └── main.py                  # “Hello” CLI entrypoint
├── pyproject.toml                   # project metadata & uv config
└── README.md
```

## Scripts & Tasks

- setup.sh – verifies google-service-account-key.json, installs Taskfile.yml, and runs a sample “hello” task.
- `task sync` – defined in Taskfile.yml, it:
  1. Executes `bash scripts/gcloud_auth.sh`  
  2. Runs `uv sync` to pull private indexes

## Usage

After running the setup:

```shell
task sync
uv run src/student_project_template/main.py
# ⇒ Hello from student-project-template!
```

## Contributing

Feel free to open issues or submit PRs to enhance this template.