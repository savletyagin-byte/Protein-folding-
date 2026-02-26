# Trainable Protein Folding Prototype (Autograd + BioPython)

This project now uses **more than NumPy**:
- **BioPython** for real PDB structure ingestion/parsing
- **Autograd** for differentiable training and gradient-based optimization

## What this adds

- Real-data training pipeline from:
  - local PDB files
  - RCSB PDB IDs (download + parse)
- Trainable model that predicts:
  - C-alpha coordinates
  - torsion logits
  - distogram logits
  - confidence scores
- CLI for train + infer workflows

## Train on real PDB IDs + infer

```bash
python protein_folding.py ACDEFGHIK \
  --train-pdb-ids 1ubq 1crn \
  --epochs 10 --batch-size 2 --lr 0.01
```

## Train on local PDB files

```bash
python protein_folding.py ACDEFGHIK \
  --train-pdb-files ./data/1ubq.pdb ./data/1crn.pdb
```

## Inference only

```bash
python protein_folding.py ACDEFGHIK
```

## Save and load trained weights

```bash
python protein_folding.py ACDEFGHIK --train-pdb-files ./data/1ubq.pdb --save-model ./trained_models/demo.json
python protein_folding.py ACDEFGHIK --load-model ./trained_models/demo.json
```

## Tests

```bash
pytest -q
```

## Notes

- This remains a compact prototype and not a production-grade, benchmarked system.
- RCSB download success depends on network availability.


- Note: binary model artifacts are not committed. Use JSON checkpoints (`.json`) for branch-friendly text artifacts.


## Branch update fix

If your host rejects branch updates due to binary files, run:

```bash
./scripts/ensure_no_binary_history.sh
```

Then force-push the rewritten branch history:

```bash
git push --force-with-lease origin <your-branch-name>
```

See `BRANCH_UPDATE_FIX.md` for details.


For persistent remote errors, run `./scripts/repair_binary_history.sh` then force-push.


If remote checks still fail, create a clean snapshot branch with `./scripts/create_clean_branch_snapshot.sh <branch>` and open a new PR from it.
