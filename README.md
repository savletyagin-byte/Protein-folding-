# Ultra-Advanced Trainable Protein Folding Prototype (Autograd + BioPython)

This repository now includes a significantly enhanced research pipeline for trainable protein structure prediction on real PDB-derived data.

## Hyper-advanced features

- Real PDB ingestion (local files + RCSB IDs)
- Residual MLP architecture for sequence-to-structure features
- Advanced optimization stack:
  - Adam optimizer
  - cosine LR schedule with warmup
  - gradient norm clipping
  - weight decay
- Training stabilization and generalization:
  - validation split
  - early stopping
  - best-checkpoint restoration
  - EMA weights
  - SWA-like late-phase parameter averaging
  - curriculum batching by sequence length
  - coordinate augmentation noise
- Rich evaluation metrics:
  - Kabsch RMSD
  - contact precision / recall / F1
  - pairwise distance MAE
- Uncertainty-aware inference:
  - ensemble prediction
  - MC-dropout-style stochastic forward passes
  - coordinate/confidence variance outputs
- Checkpointing:
  - JSON and NPZ save/load
  - JSON checkpoint metadata

## Train + infer

```bash
python protein_folding.py ACDEFGHIK \
  --train-pdb-files data/sample_train.pdb \
  --epochs 20 --batch-size 2 --lr 0.005 \
  --val-split 0.2 --patience 6 \
  --save-model trained_models/demo.json
```

## Ensemble / uncertainty inference

```bash
python protein_folding.py ACDEFGHIK --load-model trained_models/demo.json --ensemble-size 8 --json
```

## Tests

```bash
pytest -q
```

## Notes

- This is still a compact prototype and not a production-scale benchmarked foundation model.
- Keep binary model artifacts untracked.
- For binary-history branch issues, see `BRANCH_UPDATE_FIX.md`.

## One-command "do it for me" run

If you want the full flow automated (install deps, train on included sample, then run ensemble inference):

```bash
./scripts/run_demo.sh
```

This writes a JSON checkpoint to `trained_models/demo.json`.


## Export generated protein as PDB (so you can view it)

```bash
python protein_folding.py ACDEFGHIK   --load-model trained_models/demo.json   --save-pred-pdb outputs/predicted_ca_trace.pdb
```

Then open `outputs/predicted_ca_trace.pdb` in a molecular viewer (e.g., PyMOL/ChimeraX) to see the generated structure.


## See the generated protein structure

Export the predicted structure as a PDB file:

```bash
python protein_folding.py ACDEFGHIK   --load-model trained_models/demo.json   --save-pred-pdb outputs/predicted_ca_trace.pdb
```

Then open `outputs/predicted_ca_trace.pdb` in PyMOL/ChimeraX/VMD.


## Generate a picture for each analyzed protein

During training/evaluation, export one PNG per analyzed structure:

```bash
python protein_folding.py ACDEFGHIK   --train-pdb-files data/sample_train.pdb   --export-analysis-images-dir outputs/analysis_images
```

For a single prediction image:

```bash
python protein_folding.py ACDEFGHIK   --load-model trained_models/demo.json   --save-pred-image outputs/predicted_structure.png
```


## Multi-view renderer and animated rotation GIF

Single prediction exports:

```bash
python protein_folding.py ACDEFGHIK   --load-model trained_models/demo.json   --save-pred-image outputs/pred.png   --save-pred-multiview outputs/pred_multiview.png   --save-pred-gif outputs/pred_rotate.gif
```

For every analyzed protein in a dataset:

```bash
python protein_folding.py ACDEFGHIK   --train-pdb-files data/sample_train.pdb   --export-analysis-images-dir outputs/analysis_images   --export-analysis-multiview   --export-analysis-gif
```
