#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is required but not installed." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is required but not installed." >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "Error: docker compose (or docker-compose) is required but not installed." >&2
  exit 1
fi

for remote in origin upstream; do
  if ! git remote get-url "${remote}" >/dev/null 2>&1; then
    echo "Error: git remote '${remote}' is not configured." >&2
    exit 1
  fi
done

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: working tree is not clean. Commit or stash your changes before running this script." >&2
  exit 1
fi

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${current_branch}" != "main" ]]; then
  echo "Switching branch: ${current_branch} -> main"
  git checkout main
fi

echo "Fetching upstream..."
git fetch upstream --prune

echo "Rebasing local main onto upstream/main..."
git rebase upstream/main

echo "Pushing updated main to origin..."
git push --force-with-lease origin main

echo "Rebuilding Docker services..."
"${COMPOSE_CMD[@]}" -f ./docker/docker-compose.yml build

echo "Starting Docker services..."
"${COMPOSE_CMD[@]}" -f ./docker/docker-compose.yml up -d

echo "Current service status:"
"${COMPOSE_CMD[@]}" -f ./docker/docker-compose.yml ps

echo "Update completed."
