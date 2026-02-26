# HyperFold-X (Advanced Protein Folding Architecture Prototype)

This repository now contains a **substantially expanded architecture prototype** for protein structure modeling.

## What is included

- Sequence embedding with positional/state conditioning
- MSA encoder with axial-style row/column mixing
- Pair representation with relative position, MSA coupling, and template prior
- Evoformer-like recycling stack with triangle multiplicative + attention updates
- Invariant-point-like SE(3)-equivariant coordinate refinement
- Diffusion-style denoising refinement over coordinates
- Geometric heads:
  - torsion logits/angles (phi, psi, omega)
  - distogram logits
  - pLDDT-like confidence scores
- Ensemble sampling and allosteric landscape modeling

## Important note

This is a NumPy prototype and **not a trained SOTA model**, so it should be treated as a research scaffold rather than a replacement for production systems.

## Usage

```bash
python protein_folding.py ACDEFGHIK --msa ACDEFGHIK ACDEYGHIK ACDEFGHVK --state active
```

All allosteric states:

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
