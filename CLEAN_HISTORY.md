# Binary History Cleanup

To resolve branch update failures on platforms that reject binary files, the branch history was rewritten to remove:

- `trained_models/demo_realdata_model.npz`

Verification command:

```bash
git rev-list --objects --all | rg 'demo_realdata_model\.npz'
```

Expected output: no matches.

Current workflow:
- Keep model artifacts out of git via `.gitignore`.
- Use text-safe JSON checkpoints for demos if needed.
