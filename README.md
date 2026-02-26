# Advanced Trainable Protein Folding Prototype (Autograd + BioPython)

This project provides a trainable real-structure prototype with a stronger optimization/evaluation stack.

## Advanced capabilities

- Real PDB ingestion via BioPython:
  - local PDB files
  - RCSB PDB IDs (network permitting)
- Differentiable training with Autograd
- **Adam optimizer** with:
  - weight decay
  - gradient clipping
  - validation split
  - early stopping + best-checkpoint restoration
- Multi-objective supervision:
  - coordinate regression
  - distogram supervision
  - bond-length regularization
  - confidence regularization
- Structural evaluation metrics:
  - Kabsch-aligned RMSD
  - contact precision
- Ensemble inference (`predict_ensemble`) with coordinate variance outputs
- JSON/NPZ model save/load support

## Training + inference

```bash
python protein_folding.py ACDEFGHIK \
  --train-pdb-files data/sample_train.pdb \
  --epochs 20 --batch-size 2 --lr 0.005 \
  --val-split 0.2 --patience 6 \
  --save-model trained_models/demo.json
```

## Ensemble inference

```bash
python protein_folding.py ACDEFGHIK --load-model trained_models/demo.json --ensemble-size 8 --json
```

## Tests

```bash
pytest -q
```

## Notes

- This is still a compact prototype (not production-scale SOTA).
- Binary model artifacts should remain untracked.
- If your host reports binary-history branch issues, see `BRANCH_UPDATE_FIX.md`.
