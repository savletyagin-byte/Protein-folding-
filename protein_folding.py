"""Advanced lattice protein folding engine.

Features
--------
- 2D/3D self-avoiding lattice representations.
- Multi-term energy model (pair contacts, bend penalty, compactness term).
- Advanced Monte Carlo move set: end move, corner flip, crankshaft, pivot move.
- Parallel tempering (replica exchange) + simulated annealing cooling schedule.
- Multi-start optimization and deterministic reproducibility.
- Contact-map and structural metrics utilities.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Coord = Tuple[int, int, int]


@dataclass
class EnergyWeights:
    """Energy term coefficients."""

    hh_contact: float = -1.0
    hp_contact: float = 0.15
    pp_contact: float = 0.05
    bend_penalty: float = 0.02
    compactness_bonus: float = -0.01


@dataclass
class MoveWeights:
    """Relative probabilities used to sample lattice moves."""

    corner_flip: float = 1.0
    end_move: float = 1.0
    crankshaft: float = 1.0
    pivot: float = 2.5


@dataclass
class ProteinFoldingConfig:
    """Global optimization controls."""

    dimensions: int = 3
    # Annealing / tempering
    initial_temperature: float = 8.0
    final_temperature: float = 0.04
    cooling_rate: float = 0.995
    steps_per_temperature: int = 200
    replicas: int = 6
    exchange_interval: int = 15
    # Global search
    restarts: int = 4
    max_stagnation_steps: int = 1600
    random_seed: Optional[int] = 42
    track_history: bool = True
    history_stride: int = 20
    # Models
    energy: EnergyWeights = EnergyWeights()
    move_weights: MoveWeights = MoveWeights()


class ProteinFolder:
    """Advanced HP lattice protein folder."""

    _NEIGHBORS_3D: Tuple[Coord, ...] = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    _NEIGHBORS_2D: Tuple[Coord, ...] = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
    )

    # Integer orthogonal transforms preserving cubic lattice (subset for practical pivoting).
    _ROTATIONS_3D: Tuple[Tuple[Coord, Coord, Coord], ...] = (
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
        ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
        ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
        ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
        ((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
        ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, 0, -1), (0, -1, 0)),
        ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
        ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    )

    _ROTATIONS_2D: Tuple[Tuple[Coord, Coord, Coord], ...] = (
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    )

    def __init__(self, sequence: str, config: Optional[ProteinFoldingConfig] = None):
        if not sequence:
            raise ValueError("Sequence cannot be empty.")
        seq = sequence.upper()
        if any(c not in {"H", "P"} for c in seq):
            raise ValueError("Sequence must contain only 'H' and 'P' characters.")

        self.sequence = seq
        self.config = config or ProteinFoldingConfig()
        if self.config.dimensions not in {2, 3}:
            raise ValueError("dimensions must be 2 or 3")

        self.neighbors = self._NEIGHBORS_2D if self.config.dimensions == 2 else self._NEIGHBORS_3D
        self.rotations = self._ROTATIONS_2D if self.config.dimensions == 2 else self._ROTATIONS_3D
        self.rng = random.Random(self.config.random_seed)
        self._move_fns = {
            "corner_flip": self._attempt_corner_flip,
            "end_move": self._attempt_end_move,
            "crankshaft": self._attempt_crankshaft,
            "pivot": self._attempt_pivot,
        }

    @staticmethod
    def _add(a: Coord, b: Coord) -> Coord:
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    @staticmethod
    def _sub(a: Coord, b: Coord) -> Coord:
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    @staticmethod
    def _manhattan(a: Coord, b: Coord) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    @staticmethod
    def _sqdist(a: Coord, b: Coord) -> int:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2

    def _apply_rotation(self, vec: Coord, matrix: Tuple[Coord, Coord, Coord]) -> Coord:
        r0, r1, r2 = matrix
        return (
            vec[0] * r0[0] + vec[1] * r0[1] + vec[2] * r0[2],
            vec[0] * r1[0] + vec[1] * r1[1] + vec[2] * r1[2],
            vec[0] * r2[0] + vec[1] * r2[1] + vec[2] * r2[2],
        )

    def _initial_linear_conformation(self) -> List[Coord]:
        return [(i, 0, 0) for i in range(len(self.sequence))]

    def _valid_backbone(self, coords: Sequence[Coord]) -> bool:
        return all(self._manhattan(coords[i], coords[i + 1]) == 1 for i in range(len(coords) - 1))

    def _is_self_avoiding(self, coords: Sequence[Coord]) -> bool:
        return len(set(coords)) == len(coords)

    def contact_map(self, coords: Sequence[Coord]) -> List[List[int]]:
        """Binary non-bonded contact map based on Manhattan adjacency."""
        n = len(coords)
        cmap = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 2, n):
                if self._manhattan(coords[i], coords[j]) == 1:
                    cmap[i][j] = 1
                    cmap[j][i] = 1
        return cmap

    def radius_of_gyration(self, coords: Sequence[Coord]) -> float:
        """Compute radius of gyration (Rg)."""
        n = len(coords)
        cx = sum(c[0] for c in coords) / n
        cy = sum(c[1] for c in coords) / n
        cz = sum(c[2] for c in coords) / n
        rg2 = sum((c[0] - cx) ** 2 + (c[1] - cy) ** 2 + (c[2] - cz) ** 2 for c in coords) / n
        return math.sqrt(max(0.0, rg2))

    def score(self, coords: Sequence[Coord]) -> float:
        """Total energy = contact energy + bend term + compactness term."""
        if len(coords) != len(self.sequence):
            raise ValueError("Coordinates length must match sequence length.")

        e = self.config.energy
        energy = 0.0

        # Pair-contact terms for non-consecutive adjacent residues.
        for i in range(len(self.sequence)):
            ai = self.sequence[i]
            for j in range(i + 2, len(self.sequence)):
                if self._manhattan(coords[i], coords[j]) != 1:
                    continue
                aj = self.sequence[j]
                if ai == "H" and aj == "H":
                    energy += e.hh_contact
                elif ai == "P" and aj == "P":
                    energy += e.pp_contact
                else:
                    energy += e.hp_contact

        # Bend penalty to mildly discourage high-curvature conformations.
        for i in range(1, len(coords) - 1):
            v1 = self._sub(coords[i], coords[i - 1])
            v2 = self._sub(coords[i + 1], coords[i])
            if v1 != v2:
                energy += e.bend_penalty

        # Compactness bonus from centroid distance, weighted by hydrophobic content.
        rg = self.radius_of_gyration(coords)
        h_frac = self.sequence.count("H") / len(self.sequence)
        energy += e.compactness_bonus * h_frac * len(self.sequence) / max(rg, 1e-6)
        return energy

    # ---------- move proposals ----------
    def _attempt_end_move(self, coords: List[Coord]) -> Optional[List[Coord]]:
        n = len(coords)
        if n < 2:
            return None
        end_idx = 0 if self.rng.random() < 0.5 else n - 1
        anchor_idx = 1 if end_idx == 0 else n - 2
        anchor = coords[anchor_idx]

        candidates = [self._add(anchor, d) for d in self.neighbors]
        self.rng.shuffle(candidates)

        occupied = set(coords)
        occupied.remove(coords[end_idx])
        for candidate in candidates:
            if candidate in occupied:
                continue
            trial = coords.copy()
            trial[end_idx] = candidate
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

        delta = self._add(self._sub(prev_c, cur_c), self._sub(next_c, cur_c))
        candidate = self._add(cur_c, delta)

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
        a, b, c, d = coords[i - 1], coords[i], coords[i + 1], coords[i + 2]
        if self._manhattan(a, d) != 2:
            return None

        cand_b = [self._add(a, v) for v in self.neighbors if self._manhattan(self._add(a, v), d) == 1]
        cand_c = [self._add(d, v) for v in self.neighbors if self._manhattan(self._add(d, v), a) == 1]
        self.rng.shuffle(cand_b)
        self.rng.shuffle(cand_c)

        occupied = set(coords)
        occupied.remove(b)
        occupied.remove(c)
        for nb in cand_b:
            for nc in cand_c:
                if nb == nc or self._manhattan(nb, nc) != 1:
                    continue
                if nb in occupied or nc in occupied:
                    continue
                trial = coords.copy()
                trial[i], trial[i + 1] = nb, nc
                if self._valid_backbone(trial) and self._is_self_avoiding(trial):
                    return trial
        return None

    def _attempt_pivot(self, coords: List[Coord]) -> Optional[List[Coord]]:
        if len(coords) < 4:
            return None
        pivot = self.rng.randrange(1, len(coords) - 1)
        side_right = self.rng.random() < 0.5
        rot = self.rng.choice(self.rotations)

        new_coords = coords.copy()
        anchor = coords[pivot]
        if side_right:
            span = range(pivot + 1, len(coords))
        else:
            span = range(0, pivot)

        for idx in span:
            rel = self._sub(coords[idx], anchor)
            rotated = self._apply_rotation(rel, rot)
            new_coords[idx] = self._add(anchor, rotated)

        if self._valid_backbone(new_coords) and self._is_self_avoiding(new_coords):
            return new_coords
        return None

    def _choose_move(self) -> str:
        w = self.config.move_weights
        items = [
            ("corner_flip", w.corner_flip),
            ("end_move", w.end_move),
            ("crankshaft", w.crankshaft),
            ("pivot", w.pivot),
        ]
        total = sum(weight for _, weight in items)
        if total <= 0:
            raise ValueError("At least one move weight must be > 0.")
        r = self.rng.random() * total
        acc = 0.0
        for name, weight in items:
            acc += weight
            if r <= acc:
                return name
        return items[-1][0]

    # ---------- optimization ----------
    def _replica_temperatures(self) -> List[float]:
        t0 = self.config.initial_temperature
        tf = self.config.final_temperature
        if self.config.replicas == 1:
            return [t0]
        return [t0 * (tf / t0) ** (i / (self.config.replicas - 1)) for i in range(self.config.replicas)]

    def _metropolis_accept(self, delta: float, temp: float) -> bool:
        if delta <= 0:
            return True
        return self.rng.random() < math.exp(-delta / max(temp, 1e-9))

    def _attempt_exchange(self, replicas: List[Dict[str, object]], temps: List[float]) -> None:
        for i in range(len(replicas) - 1):
            e1 = float(replicas[i]["energy"])
            e2 = float(replicas[i + 1]["energy"])
            t1, t2 = temps[i], temps[i + 1]
            criterion = (1.0 / t1 - 1.0 / t2) * (e2 - e1)
            accept = criterion >= 0.0 or self.rng.random() < math.exp(criterion)
            if accept:
                replicas[i], replicas[i + 1] = replicas[i + 1], replicas[i]

    def _run_single_restart(self) -> Dict[str, object]:
        temperatures = self._replica_temperatures()
        replicas: List[Dict[str, object]] = []
        for _ in range(self.config.replicas):
            coords = self._initial_linear_conformation()
            replicas.append({"coords": coords, "energy": self.score(coords)})

        best_coords = replicas[0]["coords"].copy()
        best_energy = float(replicas[0]["energy"])
        accepted_moves = 0
        attempted_moves = 0
        stagnation = 0
        history: List[Dict[str, float]] = []

        current_t_scale = 1.0
        temperature_step = 0
        while temperatures[0] * current_t_scale > self.config.final_temperature:
            for _ in range(self.config.steps_per_temperature):
                temperature_step += 1
                for ridx, rep in enumerate(replicas):
                    coords = rep["coords"]
                    temperature = temperatures[ridx] * current_t_scale
                    move_name = self._choose_move()
                    candidate = self._move_fns[move_name](coords)
                    attempted_moves += 1
                    if candidate is None:
                        continue

                    e_new = self.score(candidate)
                    e_old = float(rep["energy"])
                    if self._metropolis_accept(e_new - e_old, temperature):
                        rep["coords"] = candidate
                        rep["energy"] = e_new
                        accepted_moves += 1

                        if e_new < best_energy:
                            best_energy = e_new
                            best_coords = candidate.copy()
                            stagnation = 0

                stagnation += 1
                if self.config.track_history and temperature_step % self.config.history_stride == 0:
                    mean_e = sum(float(r["energy"]) for r in replicas) / len(replicas)
                    history.append({
                        "step": float(temperature_step),
                        "best_energy": best_energy,
                        "mean_replica_energy": mean_e,
                        "temperature_scale": current_t_scale,
                    })

                if temperature_step % self.config.exchange_interval == 0 and len(replicas) > 1:
                    self._attempt_exchange(replicas, [t * current_t_scale for t in temperatures])

                if stagnation > self.config.max_stagnation_steps:
                    # controlled random kick from best structure
                    kick = best_coords.copy()
                    for _ in range(min(8, len(self.sequence))):
                        move_name = self._choose_move()
                        trial = self._move_fns[move_name](kick)
                        if trial is not None:
                            kick = trial
                    replicas[0] = {"coords": kick, "energy": self.score(kick)}
                    stagnation = 0

            current_t_scale *= self.config.cooling_rate

        final_replica_energies = [float(r["energy"]) for r in replicas]
        return {
            "best_coordinates": best_coords,
            "best_energy": best_energy,
            "acceptance_rate": accepted_moves / max(1, attempted_moves),
            "history": history,
            "final_replica_energies": final_replica_energies,
        }

    def fold(self) -> Dict[str, object]:
        """Run multi-restart advanced optimization and return rich diagnostics."""
        global_best: Optional[Dict[str, object]] = None
        restart_summaries: List[Dict[str, float]] = []

        for restart_id in range(self.config.restarts):
            # Decorrelate restarts while keeping determinism for fixed base seed.
            if self.config.random_seed is not None:
                self.rng.seed(self.config.random_seed + restart_id * 7919)
            result = self._run_single_restart()
            restart_summaries.append(
                {
                    "restart": float(restart_id),
                    "best_energy": float(result["best_energy"]),
                    "acceptance_rate": float(result["acceptance_rate"]),
                }
            )
            if global_best is None or float(result["best_energy"]) < float(global_best["best_energy"]):
                global_best = result

        assert global_best is not None
        best_coords = global_best["best_coordinates"]

        return {
            "sequence": self.sequence,
            "config": asdict(self.config),
            "best_energy": global_best["best_energy"],
            "best_coordinates": best_coords,
            "radius_of_gyration": self.radius_of_gyration(best_coords),
            "contact_map": self.contact_map(best_coords),
            "acceptance_rate": global_best["acceptance_rate"],
            "restart_summaries": restart_summaries,
            "history": global_best["history"],
            "final_replica_energies": global_best["final_replica_energies"],
        }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Advanced HP lattice protein folding")
    p.add_argument("sequence", help="Protein sequence containing H/P only")
    p.add_argument("--dimensions", type=int, default=3, choices=[2, 3], help="Lattice dimensionality")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--t0", type=float, default=8.0, help="Initial temperature")
    p.add_argument("--tf", type=float, default=0.04, help="Final temperature")
    p.add_argument("--cooling", type=float, default=0.995, help="Cooling rate")
    p.add_argument("--steps", type=int, default=200, help="Steps per temperature level")
    p.add_argument("--replicas", type=int, default=6, help="Parallel tempering replicas")
    p.add_argument("--exchange-interval", type=int, default=15)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--max-stagnation", type=int, default=1600)
    p.add_argument("--json", action="store_true", help="Output full result as JSON")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = ProteinFoldingConfig(
        dimensions=args.dimensions,
        initial_temperature=args.t0,
        final_temperature=args.tf,
        cooling_rate=args.cooling,
        steps_per_temperature=args.steps,
        replicas=args.replicas,
        exchange_interval=args.exchange_interval,
        restarts=args.restarts,
        max_stagnation_steps=args.max_stagnation,
        random_seed=args.seed,
    )

    folder = ProteinFolder(args.sequence, cfg)
    result = folder.fold()

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Sequence:           {result['sequence']}")
    print(f"Best energy:        {result['best_energy']:.4f}")
    print(f"Radius of gyration: {result['radius_of_gyration']:.4f}")
    print(f"Acceptance rate:    {result['acceptance_rate']:.3f}")
    print("Best coordinates:")
    for i, c in enumerate(result["best_coordinates"]):
        print(f"  {i:3d}: {tuple(c)}")


if __name__ == "__main__":
    main()
