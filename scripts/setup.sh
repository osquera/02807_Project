#!/usr/bin/env bash
set -euo pipefail

# Determine project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY_FILE="$ROOT_DIR/google-service-account-key.json"

# 1. Check for Google service account key
if [ ! -f "$KEY_FILE" ]; then
  echo "Error: '$KEY_FILE' not found."
  echo "Please add your service account JSON (named google-service-account-key.json) to the project root."
  exit 1
fi
echo "Found service account key."

# 2. Ensure Taskfile is installed
if ! command -v task >/dev/null 2>&1; then
  echo "Taskfile CLI not found. Installing..."
  curl -sSL https://taskfile.dev/install.sh | sh
  echo "Taskfile installed."
else
  echo "Taskfile is already installed."
fi

# 3. Run an initial “hello” task (ensure Taskfile.yml has a hello target)
echo "Running initial 'hello' check..."
task hello

echo "Setup complete! You can now run 'task sync'."