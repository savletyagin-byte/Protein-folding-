# Branch update fix for binary-file restrictions

If your provider says _"binary files aren’t supported"_ while updating a branch/PR, the binary usually exists in commit history (not only in the latest commit).

## 1) Verify history is clean

```bash
./scripts/ensure_no_binary_history.sh
```

## 2) If this still fails on remote, force-push rewritten history

The local branch has been rewritten to remove blocked artifacts. Update the remote branch with:

```bash
git push --force-with-lease origin <your-branch-name>
```

## 3) Prevent reintroducing binary artifacts

- Keep model artifacts ignored in `.gitignore`.
- Use JSON checkpoints for branch-safe demos.
- Re-run `./scripts/ensure_no_binary_history.sh` before pushing.

## 4) Optional cleanup if your local still sees stale refs

```bash
git reflog expire --expire=now --all
git gc --prune=now
```


## 5) One-command local repair script

You can run the included repair helper to rewrite history, cleanup refs, and verify:

```bash
./scripts/repair_binary_history.sh
```

After it finishes, push the rewritten branch:

```bash
git push --force-with-lease origin <your-branch-name>
```

## 6) If provider still blocks the same PR branch

Some providers keep stale PR refs. In that case, create a fresh branch from current HEAD and push it:

```bash
git checkout -b clean-binary-safe-branch
git push -u origin clean-binary-safe-branch
```

Then open a new PR from that fresh branch.


## 7) Nuclear option: publish a clean snapshot branch

If a provider still insists the PR branch has blocked history, create a brand-new orphan snapshot branch (single clean commit, no prior history):

```bash
./scripts/create_clean_branch_snapshot.sh clean-snapshot-branch
git push -u origin clean-snapshot-branch
```

Open a new PR from `clean-snapshot-branch`.
