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

## Tests

```bash
pytest -q
```

## Notes

- This remains a compact prototype and not a production-grade, benchmarked system.
- RCSB download success depends on network availability.
