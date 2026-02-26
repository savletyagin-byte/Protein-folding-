# OmegaFold-Prototype-X (Maximum-Feature Protein Folding Prototype)

This repository now ships a **maximal NumPy-based architecture prototype** that packs many modern ideas into one scaffold.

## Included modules

- Sequence embedding
  - token + position + biochemical channels + allosteric state conditioning
- MSA encoder
  - axial-style row/column mixing
  - MSA dropout for ensemble diversity
  - coupling extraction
- Pair representation
  - sequence/MSA fusion
  - relative-position + inverse-separation bias
  - template prior + contact prior + coupling priors
- Evoformer-like structural core
  - triangle multiplicative (incoming/outgoing)
  - triangle attention
  - recycling with early stopping
- Geometry refinement
  - IPA-like SE(3)-equivariant coordinate updates
  - diffusion-like denoising refinement
- Heads
  - torsion angle logits + decoded angles
  - distogram logits
  - pLDDT-like confidence
  - PAE-like pairwise aligned error
- Quality and ranking
  - clash/bond/compactness quality metrics
  - ranking score over confidence + physical plausibility + diversity
- Sampling and state-space
  - ensemble sampling
  - consensus statistics
  - allosteric landscape generation across states

## Important note

This is still an educational/research prototype (no trained weights), so it is intended for experimentation and architecture exploration.

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
