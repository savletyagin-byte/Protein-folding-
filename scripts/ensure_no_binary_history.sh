#!/usr/bin/env bash
set -euo pipefail

PATTERNS='\.npz$|\.pt$|\.bin$|\.onnx$'

matches=$(git rev-list --objects --all | rg -n "$PATTERNS" || true)
if [[ -n "$matches" ]]; then
  echo "❌ Binary/model artifacts found in reachable git history:" >&2
  echo "$matches" >&2
  echo "Run history cleanup and force-push rewritten branch." >&2
  exit 1
fi

echo "✅ No blocked binary/model artifacts found in reachable git history."

# Also flag binary blobs by content in reachable history.
while read -r oid path; do
  [[ -z "${path:-}" ]] && continue
  type=$(git cat-file -t "$oid" 2>/dev/null || true)
  [[ "$type" != "blob" ]] && continue
  if ! git cat-file -p "$oid" | LC_ALL=C grep -q "^[[:print:][:space:]]*$"; then
    case "$path" in
      *.md|*.txt|*.py|*.sh|*.pdb|*.gitignore|.gitkeep|*.yml|*.yaml|*.json) ;;
      *)
        echo "❌ Potential binary blob detected: $path ($oid)" >&2
        exit 1
        ;;
    esac
  fi
done < <(git rev-list --objects --all)

echo "✅ Reachable blob content appears text-safe for tracked source/artifact files."
