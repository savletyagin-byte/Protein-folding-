#!/usr/bin/env bash
set -euo pipefail

# Rewrites local history to remove model/binary artifacts that some git hosts reject.
# Use on your feature branch only.

PATTERNS=(
  "*.npz"
  "*.pt"
  "*.bin"
  "*.onnx"
)

echo "[1/5] Removing blocked artifact paths from git history..."
FILTER=""
for p in "${PATTERNS[@]}"; do
  FILTER+="git rm -f --cached --ignore-unmatch -- \"$p\"; "
done
FILTER+="git rm -f --cached --ignore-unmatch -- trained_models/*.npz trained_models/*.pt trained_models/*.bin trained_models/*.onnx;"

git filter-branch --force --index-filter "$FILTER" --prune-empty --tag-name-filter cat -- --all

echo "[2/5] Removing original refs and expiring reflogs..."
python - <<'PY'
from pathlib import Path
import shutil
p = Path('.git/refs/original')
if p.exists():
    shutil.rmtree(p)
PY

git reflog expire --expire=now --all

echo "[3/5] Running git gc..."
git gc --prune=now

echo "[4/5] Verifying no blocked files remain in reachable history..."
./scripts/ensure_no_binary_history.sh

echo "[5/5] Next step: force-push rewritten branch"
echo "    git push --force-with-lease origin <your-branch-name>"
