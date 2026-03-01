#!/usr/bin/env bash
set -euo pipefail

# One-command local demo: install deps, train on sample PDB, run ensemble inference.

python -m pip install -q numpy autograd biopython
mkdir -p trained_models

python protein_folding.py ACDEFGHIK \
  --train-pdb-files data/sample_train.pdb \
  --epochs 8 --batch-size 1 --lr 0.005 \
  --val-split 0.5 --patience 3 \
  --save-model trained_models/demo.json

python protein_folding.py ACDEFGHIK \
  --load-model trained_models/demo.json \
  --ensemble-size 6 \
  --json
