#!/usr/bin/env bash
set -euo pipefail

# Creates an orphan branch snapshot containing only current working tree files.
# Useful when a provider still reports binary-history issues on an existing PR branch.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <new-branch-name>" >&2
  exit 1
fi

NEW_BRANCH="$1"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [[ -n $(git status --porcelain) ]]; then
  echo "❌ Working tree is dirty. Commit/stash changes first." >&2
  exit 1
fi

echo "Creating orphan branch: ${NEW_BRANCH}"
git checkout --orphan "$NEW_BRANCH"

# Remove index/worktree tracked content then restore snapshot from original branch tip.
git rm -rf --cached . >/dev/null 2>&1 || true
git clean -fdx >/dev/null 2>&1 || true

git checkout "$CURRENT_BRANCH" -- .

git add -A
git commit -m "Create clean snapshot branch without historical baggage"

echo "✅ Created ${NEW_BRANCH}."
echo "Next: git push -u origin ${NEW_BRANCH}"
echo "Then open a PR from ${NEW_BRANCH}."
