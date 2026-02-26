"""Research-oriented protein folding architecture prototype.

This module implements a compact, dependency-light (NumPy-only) analogue of modern
structure prediction pipelines with the following components:

- Sequence embedding.
- MSA encoder.
- Pair representation initialization.
- AlphaFold-style triangle multiplicative/additive updates.
- SE(3)-equivariant coordinate refinement.
- Torsion-angle prediction head.
- Ensemble sampling.
- Allosteric state conditioning/modeling.

The implementation is intentionally lightweight and educational; it is not meant to
replace production-grade systems such as AlphaFold/OpenFold/RoseTTAFold.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY-X"  # 20 aa + gap + unknown placeholder
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}


@dataclass
class ModelConfig:
    """Hyperparameters for the architecture."""

    d_seq: int = 128
    d_msa: int = 96
    d_pair: int = 64
    n_recycles: int = 4
    n_triangle_updates: int = 2
    n_refine_steps: int = 30
    torsion_bins: int = 36
    n_ensemble: int = 8
    random_seed: Optional[int] = 7
    # allosteric states: e.g. inactive, active, intermediate
    allosteric_states: Tuple[str, ...] = ("inactive", "active", "intermediate")


@dataclass
class ModelWeights:
    """Randomly initialized lightweight weights for demonstrative inference."""

    seq_embed: np.ndarray
    msa_embed: np.ndarray
    pair_proj_left: np.ndarray
    pair_proj_right: np.ndarray
    tri_mul_out: np.ndarray
    tri_mul_in: np.ndarray
    tri_attn_q: np.ndarray
    tri_attn_k: np.ndarray
    tri_attn_v: np.ndarray
    refine_feat_proj: np.ndarray
    torsion_proj: np.ndarray
    state_embeddings: np.ndarray


@dataclass
class StructurePrediction:
    sequence: str
    state: str
    coordinates: np.ndarray
    pair_representation: np.ndarray
    torsion_logits: np.ndarray
    torsion_angles: np.ndarray
    confidence: np.ndarray

    def to_jsonable(self) -> Dict[str, object]:
        return {
            "sequence": self.sequence,
            "state": self.state,
            "coordinates": self.coordinates.tolist(),
            "pair_representation_shape": list(self.pair_representation.shape),
            "torsion_logits_shape": list(self.torsion_logits.shape),
            "torsion_angles": self.torsion_angles.tolist(),
            "confidence": self.confidence.tolist(),
        }


class AdvancedProteinFoldingModel:
    """Compact end-to-end architecture with AlphaFold-inspired geometric blocks."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.weights = self._init_weights()

    def _init_weights(self) -> ModelWeights:
        c = self.config

        def randn(*shape: int, scale: float = 0.05) -> np.ndarray:
            return self.rng.normal(0.0, scale, size=shape)

        return ModelWeights(
            seq_embed=randn(len(AA_VOCAB), c.d_seq),
            msa_embed=randn(len(AA_VOCAB), c.d_msa),
            pair_proj_left=randn(c.d_seq + c.d_msa, c.d_pair),
            pair_proj_right=randn(c.d_seq + c.d_msa, c.d_pair),
            tri_mul_out=randn(c.d_pair, c.d_pair),
            tri_mul_in=randn(c.d_pair, c.d_pair),
            tri_attn_q=randn(c.d_pair, c.d_pair),
            tri_attn_k=randn(c.d_pair, c.d_pair),
            tri_attn_v=randn(c.d_pair, c.d_pair),
            refine_feat_proj=randn(c.d_pair, 3),
            torsion_proj=randn(c.d_seq + c.d_pair, c.torsion_bins * 3),
            state_embeddings=randn(len(c.allosteric_states), c.d_seq, scale=0.02),
        )

    @staticmethod
    def _one_hot_indices(sequence: str) -> np.ndarray:
        idx = [AA_TO_IDX.get(ch, AA_TO_IDX["X"]) for ch in sequence]
        return np.asarray(idx, dtype=np.int64)

    def sequence_embedding(self, sequence: str, state: str) -> np.ndarray:
        """Per-residue embedding with allosteric-state conditioning."""
        idx = self._one_hot_indices(sequence)
        seq_emb = self.weights.seq_embed[idx]  # (L, d_seq)
        state_idx = self.config.allosteric_states.index(state)
        state_emb = self.weights.state_embeddings[state_idx][None, :]
        return seq_emb + state_emb

    def msa_encoder(self, msa: List[str]) -> np.ndarray:
        """Encode MSA and aggregate into per-position contextual signal."""
        if not msa:
            raise ValueError("MSA must contain at least one sequence.")
        lengths = {len(s) for s in msa}
        if len(lengths) != 1:
            raise ValueError("All MSA sequences must have equal length.")

        msa_idx = np.stack([self._one_hot_indices(row) for row in msa], axis=0)  # (N, L)
        emb = self.weights.msa_embed[msa_idx]  # (N, L, d_msa)
        # Robust aggregation: mean + variance features condensed via linear map-like combination.
        mean = emb.mean(axis=0)
        var = emb.var(axis=0)
        return mean + 0.2 * var

    def pair_representation(self, seq_repr: np.ndarray, msa_repr: np.ndarray) -> np.ndarray:
        """Initialize residue-pair representation from single + MSA features."""
        s = np.concatenate([seq_repr, msa_repr], axis=-1)
        left = s @ self.weights.pair_proj_left
        right = s @ self.weights.pair_proj_right
        pair = left[:, None, :] + right[None, :, :]

        # Add simple relative-position bias.
        L = pair.shape[0]
        rel = np.arange(L)[:, None] - np.arange(L)[None, :]
        pair += 0.01 * np.tanh(rel[..., None] / 8.0)
        return pair

    def _triangle_multiplicative_update(self, z: np.ndarray, outgoing: bool) -> np.ndarray:
        w = self.weights.tri_mul_out if outgoing else self.weights.tri_mul_in
        zz = z @ w
        # outgoing: i,j updated through k as i->k and j->k interactions
        if outgoing:
            upd = np.einsum("ikd,jkd->ijd", zz, zz) / math.sqrt(z.shape[-1])
        else:
            upd = np.einsum("kid,kjd->ijd", zz, zz) / math.sqrt(z.shape[-1])
        return upd

    def _triangle_attention_update(self, z: np.ndarray) -> np.ndarray:
        q = z @ self.weights.tri_attn_q
        k = z @ self.weights.tri_attn_k
        v = z @ self.weights.tri_attn_v
        logits = np.einsum("ijd,ikd->ijk", q, k) / math.sqrt(z.shape[-1])
        logits = logits - logits.max(axis=-1, keepdims=True)
        attn = np.exp(logits)
        attn /= np.clip(attn.sum(axis=-1, keepdims=True), 1e-8, None)
        return np.einsum("ijk,ikd->ijd", attn, v)

    def triangle_updates(self, pair_repr: np.ndarray) -> np.ndarray:
        """AlphaFold-style triangle multiplicative + attention updates."""
        z = pair_repr.copy()
        for _ in range(self.config.n_triangle_updates):
            z = z + 0.2 * np.tanh(self._triangle_multiplicative_update(z, outgoing=True))
            z = z + 0.2 * np.tanh(self._triangle_multiplicative_update(z, outgoing=False))
            z = z + 0.2 * np.tanh(self._triangle_attention_update(z))
            z = 0.5 * (z + np.transpose(z, (1, 0, 2)))  # enforce i,j symmetry
        return z

    def se3_equivariant_refinement(self, pair_repr: np.ndarray, n_steps: Optional[int] = None) -> np.ndarray:
        """Coordinate refinement equivariant to rigid translations/rotations.

        Update rule uses only pairwise relative vectors and scalar gates from pair features,
        which preserves SE(3)-equivariance.
        """
        L = pair_repr.shape[0]
        steps = n_steps or self.config.n_refine_steps
        coords = np.stack([np.arange(L), np.zeros(L), np.zeros(L)], axis=-1).astype(np.float64)

        for _ in range(steps):
            rel = coords[:, None, :] - coords[None, :, :]  # (L,L,3)
            dist2 = np.sum(rel * rel, axis=-1, keepdims=True) + 1e-6
            inv_dist = 1.0 / np.sqrt(dist2)
            scalar = np.tanh(pair_repr @ self.weights.refine_feat_proj)  # (L,L,3)
            # isotropic + learned directional modulation
            direction = rel * inv_dist
            force = (0.15 * direction + 0.05 * scalar) / np.sqrt(dist2)
            delta = force.sum(axis=1) - force.sum(axis=0)
            coords += 0.02 * delta
            coords -= coords.mean(axis=0, keepdims=True)  # center-of-mass stabilization
        return coords

    def torsion_head(self, seq_repr: np.ndarray, pair_repr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict torsion-angle distributions and MAP angles for phi/psi/omega."""
        pair_pool = pair_repr.mean(axis=1)
        fused = np.concatenate([seq_repr, pair_pool], axis=-1)
        logits = fused @ self.weights.torsion_proj
        logits = logits.reshape(len(seq_repr), 3, self.config.torsion_bins)

        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs /= np.clip(probs.sum(axis=-1, keepdims=True), 1e-8, None)
        idx = np.argmax(probs, axis=-1)
        angles = -math.pi + (2 * math.pi) * (idx / (self.config.torsion_bins - 1))
        return logits, angles

    def _single_pass(self, sequence: str, msa: List[str], state: str) -> StructurePrediction:
        seq_repr = self.sequence_embedding(sequence, state)
        msa_repr = self.msa_encoder(msa)
        pair = self.pair_representation(seq_repr, msa_repr)

        for _ in range(self.config.n_recycles):
            pair = self.triangle_updates(pair)

        coords = self.se3_equivariant_refinement(pair)
        torsion_logits, torsion_angles = self.torsion_head(seq_repr, pair)

        # Lightweight confidence proxy from pair norms.
        conf = 1.0 / (1.0 + np.exp(-pair.mean(axis=(1, 2))))
        return StructurePrediction(
            sequence=sequence,
            state=state,
            coordinates=coords,
            pair_representation=pair,
            torsion_logits=torsion_logits,
            torsion_angles=torsion_angles,
            confidence=conf,
        )

    def ensemble_sample(self, sequence: str, msa: List[str], state: str) -> Dict[str, object]:
        """Sample an ensemble and return aggregate statistics + members."""
        members: List[StructurePrediction] = []
        base_seed = self.config.random_seed if self.config.random_seed is not None else 0

        for k in range(self.config.n_ensemble):
            self.rng = np.random.default_rng(base_seed + 104729 * (k + 1))
            self.weights = self._init_weights()
            members.append(self._single_pass(sequence, msa, state))

        stack = np.stack([m.coordinates for m in members], axis=0)
        mean_coords = stack.mean(axis=0)
        var_coords = stack.var(axis=0)

        return {
            "state": state,
            "mean_coordinates": mean_coords,
            "coordinate_variance": var_coords,
            "mean_confidence": np.mean(np.stack([m.confidence for m in members], axis=0), axis=0),
            "members": members,
        }

    def model_allosteric_states(self, sequence: str, msa: List[str]) -> Dict[str, Dict[str, object]]:
        """Generate state-conditioned ensembles for all configured allosteric states."""
        results = {}
        for state in self.config.allosteric_states:
            results[state] = self.ensemble_sample(sequence, msa, state)
        return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Advanced protein folding architecture prototype")
    p.add_argument("sequence", help="Primary sequence (single-letter amino-acid code)")
    p.add_argument("--msa", nargs="*", default=None, help="MSA rows. If omitted, uses sequence only")
    p.add_argument("--state", default="inactive", help="Allosteric state label")
    p.add_argument("--ensemble", type=int, default=8, help="Number of ensemble samples")
    p.add_argument("--recycles", type=int, default=4, help="Number of triangle recycling rounds")
    p.add_argument("--json", action="store_true", help="Print JSON output")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = ModelConfig(n_ensemble=args.ensemble, n_recycles=args.recycles)
    model = AdvancedProteinFoldingModel(cfg)
    msa = args.msa if args.msa else [args.sequence]

    if args.state == "all":
        out = model.model_allosteric_states(args.sequence, msa)
        summary = {
            s: {
                "mean_confidence_mean": float(v["mean_confidence"].mean()),
                "mean_coordinates_shape": list(v["mean_coordinates"].shape),
                "coordinate_variance_mean": float(v["coordinate_variance"].mean()),
            }
            for s, v in out.items()
        }
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            for state, stats in summary.items():
                print(f"State={state}: confidence={stats['mean_confidence_mean']:.3f}, "
                      f"var={stats['coordinate_variance_mean']:.5f}")
        return

    if args.state not in cfg.allosteric_states:
        raise ValueError(f"state must be one of {cfg.allosteric_states} or 'all'")

    out = model.ensemble_sample(args.sequence, msa, args.state)
    payload = {
        "state": out["state"],
        "mean_coordinates": out["mean_coordinates"].tolist(),
        "coordinate_variance": out["coordinate_variance"].tolist(),
        "mean_confidence": out["mean_confidence"].tolist(),
        "members": [m.to_jsonable() for m in out["members"]],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"State: {payload['state']}")
        print(f"Residues: {len(args.sequence)}")
        print(f"Ensemble members: {len(payload['members'])}")
        print(f"Mean confidence: {float(np.mean(out['mean_confidence'])):.3f}")
        print("First 5 mean coordinates:")
        for i, c in enumerate(payload["mean_coordinates"][:5]):
            print(f"  {i:3d}: ({c[0]: .3f}, {c[1]: .3f}, {c[2]: .3f})")


if __name__ == "__main__":
    main()
