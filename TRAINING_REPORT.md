# Training Report

A trained checkpoint was generated in this environment (artifact can be regenerated locally):

- Checkpoint path used: `trained_models/demo_realdata_model.json` (not committed)
- Training data: local PDB-format structure `data/sample_train.pdb` (6 residues, C-alpha trace)
- Command used:

```bash
python protein_folding.py ACDEFG --train-pdb-files data/sample_train.pdb --epochs 8 --batch-size 1 --lr 0.005 --save-model trained_models/demo_realdata_model.json
```

Observed terminal summary:

- `Trained: True`
- `Final train loss: 4.2907`
- `Mean confidence: 50.41`

I also attempted direct RCSB training by PDB ID:

```bash
python protein_folding.py ACDEFG --train-pdb-ids 1ubq 1crn --epochs 1 --batch-size 1 --lr 0.005
```

In this environment, RCSB download was blocked by network proxy (`403 Forbidden` tunnel), so local PDB-file training was used.


Repository note: the binary `.npz` checkpoint was removed from git history on this branch tip to avoid binary-file restrictions when updating branches.
