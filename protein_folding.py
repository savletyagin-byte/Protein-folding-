"""Protein folding in a lattice HP model using simulated annealing.

This module provides:
- A `ProteinFoldingConfig` dataclass to tune the search.
- A `ProteinFolder` class implementing lattice moves and scoring.
- A command-line interface for running folding experiments.

The implementation uses a simplified hydrophobic-polar (HP) energy function:
- Every non-consecutive H-H contact contributes -1 energy.
- Lower energy is better.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

Coord = Tuple[int, int, int]


@dataclass
class ProteinFoldingConfig:
    """Hyperparameters for the simulated annealing search."""

    initial_temperature: float = 8.0
    final_temperature: float = 0.05
    cooling_rate: float = 0.995
    steps_per_temperature: int = 300
    random_seed: Optional[int] = 42


class ProteinFolder:
    """Lattice protein folder for HP sequences."""

    _NEIGHBORS: Tuple[Coord, ...] = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )

    def __init__(self, sequence: str, config: Optional[ProteinFoldingConfig] = None):
        if not sequence:
            raise ValueError("Sequence cannot be empty.")
        if any(c not in {"H", "P"} for c in sequence.upper()):
            raise ValueError("Sequence must contain only 'H' and 'P' characters.")

        self.sequence = sequence.upper()
        self.config = config or ProteinFoldingConfig()
        self.rng = random.Random(self.config.random_seed)

    def _initial_linear_conformation(self) -> List[Coord]:
        return [(i, 0, 0) for i in range(len(self.sequence))]

    @staticmethod
    def _add(a: Coord, b: Coord) -> Coord:
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    @staticmethod
    def _sub(a: Coord, b: Coord) -> Coord:
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    @staticmethod
    def _manhattan(a: Coord, b: Coord) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    def _is_self_avoiding(self, coords: Sequence[Coord]) -> bool:
        return len(set(coords)) == len(coords)

    def score(self, coords: Sequence[Coord]) -> int:
        """Compute HP lattice energy (lower is better)."""
        if len(coords) != len(self.sequence):
            raise ValueError("Coordinates length must match sequence length.")

        energy = 0
        for i, aa_i in enumerate(self.sequence):
            if aa_i != "H":
                continue
            for j in range(i + 2, len(self.sequence)):
                if self.sequence[j] != "H":
                    continue
                if self._manhattan(coords[i], coords[j]) == 1:
                    energy -= 1
        return energy

    def _valid_backbone(self, coords: Sequence[Coord]) -> bool:
        for i in range(len(coords) - 1):
            if self._manhattan(coords[i], coords[i + 1]) != 1:
                return False
        return True

    def _attempt_end_move(self, coords: List[Coord]) -> Optional[List[Coord]]:
        n = len(coords)
        end_idx = 0 if self.rng.random() < 0.5 else n - 1
        anchor_idx = 1 if end_idx == 0 else n - 2

        anchor = coords[anchor_idx]
        candidates = [self._add(anchor, d) for d in self._NEIGHBORS]
        self.rng.shuffle(candidates)

        used = set(coords)
        used.remove(coords[end_idx])
        for c in candidates:
            if c in used:
                continue
            trial = coords.copy()
            trial[end_idx] = c
            if self._valid_backbone(trial) and self._is_self_avoiding(trial):
                return trial
        return None

    def _attempt_corner_flip(self, coords: List[Coord]) -> Optional[List[Coord]]:
        if len(coords) < 3:
            return None
        i = self.rng.randrange(1, len(coords) - 1)
        prev_c, cur_c, next_c = coords[i - 1], coords[i], coords[i + 1]

        if self._manhattan(prev_c, next_c) != 2:
            return None

        delta1 = self._sub(prev_c, cur_c)
        delta2 = self._sub(next_c, cur_c)
        candidate = self._add(cur_c, self._add(delta1, delta2))

        if candidate in set(coords):
            return None

        trial = coords.copy()
        trial[i] = candidate
        if self._valid_backbone(trial) and self._is_self_avoiding(trial):
            return trial
        return None

    def _attempt_crankshaft(self, coords: List[Coord]) -> Optional[List[Coord]]:
        if len(coords) < 4:
            return None
        i = self.rng.randrange(1, len(coords) - 2)

        a = coords[i - 1]
        b = coords[i]
        c = coords[i + 1]
        d = coords[i + 2]

        if self._manhattan(a, d) != 2:
            return None

        candidates_b = [self._add(a, v) for v in self._NEIGHBORS if self._manhattan(self._add(a, v), d) == 1]
        candidates_c = [self._add(d, v) for v in self._NEIGHBORS if self._manhattan(self._add(d, v), a) == 1]
        self.rng.shuffle(candidates_b)
        self.rng.shuffle(candidates_c)

        occupied = set(coords)
        occupied.remove(b)
        occupied.remove(c)

        for nb in candidates_b:
            for nc in candidates_c:
                if nb == nc:
                    continue
                if self._manhattan(nb, nc) != 1:
                    continue
                if nb in occupied or nc in occupied:
                    continue
                trial = coords.copy()
                trial[i] = nb
                trial[i + 1] = nc
                if self._valid_backbone(trial) and self._is_self_avoiding(trial):
                    return trial
        return None

    def _propose_move(self, coords: List[Coord]) -> Optional[List[Coord]]:
        moves = [self._attempt_corner_flip, self._attempt_end_move, self._attempt_crankshaft]
        self.rng.shuffle(moves)
        for move in moves:
            candidate = move(coords)
            if candidate is not None:
                return candidate
        return None

    def fold(self) -> Dict[str, object]:
        """Run simulated annealing and return best conformation found."""
        current = self._initial_linear_conformation()
        current_energy = self.score(current)

        best = current.copy()
        best_energy = current_energy

        temperature = self.config.initial_temperature
        while temperature > self.config.final_temperature:
            for _ in range(self.config.steps_per_temperature):
                candidate = self._propose_move(current)
                if candidate is None:
                    continue

                candidate_energy = self.score(candidate)
                delta = candidate_energy - current_energy

                accept = delta <= 0
                if not accept:
                    prob = math.exp(-delta / max(temperature, 1e-9))
                    accept = self.rng.random() < prob

                if accept:
                    current = candidate
                    current_energy = candidate_energy
                    if current_energy < best_energy:
                        best = current.copy()
                        best_energy = current_energy

            temperature *= self.config.cooling_rate

        return {
            "sequence": self.sequence,
            "best_energy": best_energy,
            "best_coordinates": best,
            "final_energy": current_energy,
            "final_coordinates": current,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fold an HP protein sequence on a 3D lattice.")
    parser.add_argument("sequence", help="Protein sequence containing only H/P letters.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--t0", type=float, default=8.0, help="Initial temperature.")
    parser.add_argument("--tf", type=float, default=0.05, help="Final temperature.")
    parser.add_argument("--cooling", type=float, default=0.995, help="Cooling rate in (0,1).")
    parser.add_argument("--steps", type=int, default=300, help="Steps per temperature.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = ProteinFoldingConfig(
        initial_temperature=args.t0,
        final_temperature=args.tf,
        cooling_rate=args.cooling,
        steps_per_temperature=args.steps,
        random_seed=args.seed,
    )
    folder = ProteinFolder(args.sequence, cfg)
    result = folder.fold()

    print(f"Sequence:       {result['sequence']}")
    print(f"Best energy:    {result['best_energy']}")
    print(f"Final energy:   {result['final_energy']}")
    print("Best coords:")
    for i, c in enumerate(result["best_coordinates"]):
        print(f"  {i:3d}: {c}")


if __name__ == "__main__":
    main()
