# Advanced Protein Folding (HP Lattice + Parallel Tempering)

This project provides an **advanced lattice-based protein folding optimizer** for HP sequences.

## Advanced features

- **2D or 3D lattice** folding (`--dimensions`).
- **Self-avoiding walk** constraints and backbone connectivity checks.
- **Multi-term energy model**:
  - H-H, H-P, P-P non-bonded contact terms,
  - bend penalty,
  - compactness bonus using radius of gyration.
- **Advanced move set**:
  - end move,
  - corner flip,
  - crankshaft,
  - pivot rotation move.
- **Parallel tempering** replica exchange + cooling schedule.
- **Multi-restart global search** with stagnation kick.
- Rich outputs: best coordinates, contact map, radius of gyration, acceptance rate, restart summaries, optional JSON.

## Run

```bash
python protein_folding.py HPPHHPHPPHH --restarts 6 --replicas 8 --steps 300
```

JSON output:

```bash
python protein_folding.py HPPHHPHPPHH --json
```

2D mode:

```bash
python protein_folding.py HPPHHPHPPHH --dimensions 2
```

## Tests

```bash
pytest -q
```

## Notes

This is still a simplified educational computational model and not an atomistic molecular dynamics engine.
