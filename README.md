# OmegaFold-Ultra (Most-Advanced NumPy Protein Folding Prototype)

This repository contains an expanded **OmegaFold-Ultra** research scaffold with a very broad feature set.

## What’s included

- Sequence embedding
  - token + positional + biochemical channels
  - allosteric-state conditioning
- MSA encoder
  - axial row/column mixing
  - stochastic MSA dropout
  - coupling-map extraction
- Pair representation
  - sequence/MSA fusion
  - relative/log/inverse separation priors
  - template/contact/band priors and coupling priors
- Structural core
  - triangle multiplicative + triangle attention updates
  - recycling with convergence criterion
- Geometry stack
  - IPA-like SE(3)-equivariant refinement
  - diffusion-like denoising refinement
  - annealing-based relaxation
- Prediction heads
  - torsion logits + decoded angles
  - distogram logits
  - pLDDT-like confidence
  - PAE-like matrix
- Post-processing
  - uncertainty calibration
  - physical quality metrics (clash/bond/Rg/compactness)
  - self-consistency score
  - ensemble ranking and consensus
- State-space analysis
  - allosteric landscape generation across configured states

## Important note

This remains a NumPy research prototype with random initialization and no learned weights.

## Usage

```bash
python protein_folding.py ACDEFGHIK --msa ACDEFGHIK ACDEYGHIK ACDEFGHVK --state active
```

All states:

```bash
python protein_folding.py ACDEFGHIK --state all
```

JSON output:

```bash
python protein_folding.py ACDEFGHIK --json
```

## Tests

```bash
pytest -q
```
