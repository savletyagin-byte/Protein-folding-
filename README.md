# Advanced Protein Folding Architecture Prototype

This repository now provides a **modern, architecture-level protein folding prototype** with the requested advanced components:

- Sequence embedding
- MSA encoder
- Residue–residue pair representation
- AlphaFold-style triangle updates
- SE(3)-equivariant coordinate refinement
- Torsion angle head
- Ensemble sampling
- Allosteric state modeling

> Implementation note: this is a NumPy-only educational/research prototype focused on architecture and geometric principles, not a production-trained model.

## Quick start

```bash
python protein_folding.py ACDEFGHIK --msa ACDEFGHIK ACDEYGHIK ACDEFGHVK --state active
```

### JSON output

```bash
python protein_folding.py ACDEFGHIK --json
```

### All allosteric states at once

```bash
python protein_folding.py ACDEFGHIK --state all --json
```

## CLI flags

- `--msa ...` : provide MSA rows; defaults to sequence-only.
- `--state` : one of `inactive`, `active`, `intermediate`, or `all`.
- `--ensemble` : number of ensemble samples.
- `--recycles` : number of triangle-recycling passes.
- `--json` : machine-readable output.

## Tests

```bash
pytest -q
```
