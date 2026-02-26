# Protein Folding (HP Lattice + Simulated Annealing)

This repository contains a compact protein folding implementation using a **3D lattice HP model**:

- Amino acids are modeled as `H` (hydrophobic) or `P` (polar).
- A conformation is a self-avoiding walk on a cubic lattice.
- The energy function rewards non-consecutive neighboring `H-H` contacts (`-1` per contact).
- Search is performed with simulated annealing and local moves (corner flip, end move, crankshaft).

## Run

```bash
python protein_folding.py HPPHHPHPPHH
```

Optional tuning:

```bash
python protein_folding.py HPPHHPHPPHH --seed 1 --t0 10 --tf 0.05 --cooling 0.995 --steps 500
```

## Test

```bash
pytest -q
```

## Notes

This is a simplified educational model, not an atomistic physics simulation.
