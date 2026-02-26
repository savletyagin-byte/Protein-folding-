"""HyperFold-X: advanced architecture-level protein structure prototype.

This module is a research/educational implementation that incorporates many design ideas
from modern structure systems while remaining lightweight (NumPy-only):

- Token + positional + allosteric-conditioned sequence embedding
- MSA encoder with row/column mixing and coevolution projection
- Residue-pair representation (with relative position + MSA coupling)
- Template/distogram feature injection
- Evoformer-style recycling with triangle multiplicative + triangle attention updates
- Invariant-point-like SE(3)-equivariant coordinate refinement
- Diffusion-style denoising refinement loop
- Multi-head geometric predictions (torsions, distogram, pLDDT-like confidence)
- Ensemble sampling and allosteric state manifold exploration

Note: This is not a trained SOTA model and cannot legitimately outperform AlphaFold.
It is, however, a substantially richer prototyping scaffold.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY-X"
AA_TO_IDX = {a: i for i, a in enumerate(AA_VOCAB)}


@dataclass
class ModelConfig:
    d_seq: int = 192
    d_msa: int = 128
    d_pair: int = 96
    n_evo_blocks: int = 4
    n_recycles: int = 3
    n_triangle_updates: int = 2
    n_refine_steps: int = 24
    n_diffusion_steps: int = 16
    torsion_bins: int = 48
    dist_bins: int = 64
    n_ensemble: int = 6
    random_seed: Optional[int] = 17
    allosteric_states: Tuple[str, ...] = ("inactive", "active", "intermediate", "agonist-bound")


@dataclass
class Prediction:
    sequence: str
    state: str
    coordinates: np.ndarray
    torsion_logits: np.ndarray
    torsion_angles: np.ndarray
    distogram_logits: np.ndarray
    plddt: np.ndarray
    pair: np.ndarray

    def to_jsonable(self) -> Dict[str, object]:
        return {
            "sequence": self.sequence,
            "state": self.state,
            "coordinates": self.coordinates.tolist(),
            "torsion_angles": self.torsion_angles.tolist(),
            "torsion_logits_shape": list(self.torsion_logits.shape),
            "distogram_logits_shape": list(self.distogram_logits.shape),
            "plddt": self.plddt.tolist(),
            "pair_shape": list(self.pair.shape),
        }


class HyperFoldX:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.W = self._init_weights()

    def _init_weights(self) -> Dict[str, np.ndarray]:
        c = self.config

        def n(*shape: int, scale: float = 0.03) -> np.ndarray:
            return self.rng.normal(0.0, scale, size=shape)

        return {
            "seq_embed": n(len(AA_VOCAB), c.d_seq),
            "pos_embed": n(2048, c.d_seq),
            "state_embed": n(len(c.allosteric_states), c.d_seq, scale=0.02),
            "msa_embed": n(len(AA_VOCAB), c.d_msa),
            "msa_row_mix": n(c.d_msa, c.d_msa),
            "msa_col_mix": n(c.d_msa, c.d_msa),
            "msa_to_pair_l": n(c.d_msa, c.d_pair),
            "msa_to_pair_r": n(c.d_msa, c.d_pair),
            "seq_to_pair_l": n(c.d_seq, c.d_pair),
            "seq_to_pair_r": n(c.d_seq, c.d_pair),
            "pair_bias_proj": n(4, c.d_pair),  # relpos, sep, template, coupling
            "tri_mul_out": n(c.d_pair, c.d_pair),
            "tri_mul_in": n(c.d_pair, c.d_pair),
            "tri_q": n(c.d_pair, c.d_pair),
            "tri_k": n(c.d_pair, c.d_pair),
            "tri_v": n(c.d_pair, c.d_pair),
            "ipa_pair_to_force": n(c.d_pair, 3),
            "ipa_pair_gate": n(c.d_pair, 1),
            "diffusion_cond": n(c.d_pair, 3),
            "torsion_head": n(c.d_seq + c.d_pair, c.torsion_bins * 3),
            "dist_head": n(c.d_pair, c.dist_bins),
            "plddt_head": n(c.d_seq + c.d_pair, 1),
        }

    @staticmethod
    def _idx(seq: str) -> np.ndarray:
        return np.asarray([AA_TO_IDX.get(s, AA_TO_IDX["X"]) for s in seq], dtype=np.int64)

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-9, None)

    def sequence_embedding(self, sequence: str, state: str) -> np.ndarray:
        idx = self._idx(sequence)
        if len(sequence) > self.W["pos_embed"].shape[0]:
            raise ValueError("Sequence too long for positional table.")
        x = self.W["seq_embed"][idx] + self.W["pos_embed"][np.arange(len(sequence))]
        sidx = self.config.allosteric_states.index(state)
        x = x + self.W["state_embed"][sidx][None, :]
        return x

    def msa_encoder(self, msa: Sequence[str]) -> np.ndarray:
        if not msa:
            raise ValueError("MSA cannot be empty.")
        L = len(msa[0])
        if any(len(m) != L for m in msa):
            raise ValueError("All MSA rows must have equal length.")

        idx = np.stack([self._idx(m) for m in msa], axis=0)  # (N,L)
        m = self.W["msa_embed"][idx]  # (N,L,d_msa)

        # Axial-like mixing (row then column contexts)
        row_ctx = np.tanh(m @ self.W["msa_row_mix"])
        col_ctx = np.tanh(np.transpose(np.transpose(m, (1, 0, 2)) @ self.W["msa_col_mix"], (1, 0, 2)))
        fused = 0.6 * m + 0.2 * row_ctx + 0.2 * col_ctx
        return fused.mean(axis=0)  # (L,d_msa)

    def template_feature(self, L: int) -> np.ndarray:
        # Synthetic template prior: favors i~i+3.6 helix-like periodic contacts.
        ii = np.arange(L)[:, None]
        jj = np.arange(L)[None, :]
        sep = np.abs(ii - jj)
        return np.exp(-((sep - 4.0) ** 2) / 18.0)

    def pair_representation(self, seq_repr: np.ndarray, msa_repr: np.ndarray) -> np.ndarray:
        L = seq_repr.shape[0]
        seq_l = seq_repr @ self.W["seq_to_pair_l"]
        seq_r = seq_repr @ self.W["seq_to_pair_r"]
        msa_l = msa_repr @ self.W["msa_to_pair_l"]
        msa_r = msa_repr @ self.W["msa_to_pair_r"]
        z = seq_l[:, None, :] + seq_r[None, :, :] + msa_l[:, None, :] + msa_r[None, :, :]

        ii = np.arange(L)[:, None]
        jj = np.arange(L)[None, :]
        rel = np.tanh((ii - jj) / 16.0)
        sep = np.log1p(np.abs(ii - jj))
        tpl = self.template_feature(L)
        coupling = np.tanh((msa_repr @ msa_repr.T) / msa_repr.shape[-1])

        bias_feat = np.stack([rel, sep / (sep.max() + 1e-6), tpl, coupling], axis=-1)
        z = z + bias_feat @ self.W["pair_bias_proj"]
        z = 0.5 * (z + np.transpose(z, (1, 0, 2)))
        return z

    def _triangle_mul(self, z: np.ndarray, outgoing: bool) -> np.ndarray:
        w = self.W["tri_mul_out"] if outgoing else self.W["tri_mul_in"]
        h = np.tanh(z @ w)
        if outgoing:
            return np.einsum("ikd,jkd->ijd", h, h) / math.sqrt(z.shape[-1])
        return np.einsum("kid,kjd->ijd", h, h) / math.sqrt(z.shape[-1])

    def _triangle_attn(self, z: np.ndarray) -> np.ndarray:
        q = z @ self.W["tri_q"]
        k = z @ self.W["tri_k"]
        v = z @ self.W["tri_v"]
        logits = np.einsum("ijd,ikd->ijk", q, k) / math.sqrt(z.shape[-1])
        a = self._softmax(logits, axis=-1)
        return np.einsum("ijk,ikd->ijd", a, v)

    def evoformer_recycle(self, pair: np.ndarray) -> np.ndarray:
        z = pair
        for _ in range(self.config.n_evo_blocks):
            for _ in range(self.config.n_triangle_updates):
                z = z + 0.2 * np.tanh(self._triangle_mul(z, outgoing=True))
                z = z + 0.2 * np.tanh(self._triangle_mul(z, outgoing=False))
                z = z + 0.15 * np.tanh(self._triangle_attn(z))
                z = 0.5 * (z + np.transpose(z, (1, 0, 2)))
        return z

    def _ipa_refine(self, pair: np.ndarray, n_steps: int) -> np.ndarray:
        L = pair.shape[0]
        coords = np.stack([np.arange(L), np.zeros(L), np.zeros(L)], axis=-1).astype(np.float64)

        for _ in range(n_steps):
            rel = coords[:, None, :] - coords[None, :, :]
            d2 = np.sum(rel * rel, axis=-1, keepdims=True) + 1e-6
            unit = rel / np.sqrt(d2)
            gate = 1.0 / (1.0 + np.exp(-(pair @ self.W["ipa_pair_gate"])))
            force_feat = np.tanh(pair @ self.W["ipa_pair_to_force"])
            force = gate * (0.18 * unit + 0.04 * force_feat) / np.sqrt(d2)
            delta = force.sum(axis=1) - force.sum(axis=0)
            coords += 0.03 * delta
            coords -= coords.mean(axis=0, keepdims=True)
        return coords

    def _diffusion_refine(self, coords: np.ndarray, pair: np.ndarray, steps: int) -> np.ndarray:
        x = coords.copy()
        cond = np.tanh(pair @ self.W["diffusion_cond"]).mean(axis=1)
        for t in range(steps, 0, -1):
            sigma = 0.08 * (t / steps)
            eps = self.rng.normal(0.0, sigma, size=x.shape)
            x_noisy = x + eps

            # Denoiser predicts correction from pair-conditioned drift + backbone smoothness.
            smooth = np.zeros_like(x)
            smooth[1:-1] = 0.5 * (x_noisy[:-2] + x_noisy[2:]) - x_noisy[1:-1]
            denoise = 0.55 * smooth + 0.45 * cond
            x = x_noisy + 0.25 * denoise
            x -= x.mean(axis=0, keepdims=True)
        return x

    def torsion_head(self, seq_repr: np.ndarray, pair: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pooled = pair.mean(axis=1)
        f = np.concatenate([seq_repr, pooled], axis=-1)
        logits = (f @ self.W["torsion_head"]).reshape(len(seq_repr), 3, self.config.torsion_bins)
        probs = self._softmax(logits, axis=-1)
        idx = probs.argmax(axis=-1)
        angles = -math.pi + (2 * math.pi) * idx / max(1, (self.config.torsion_bins - 1))
        return logits, angles

    def distogram_head(self, pair: np.ndarray) -> np.ndarray:
        return np.einsum("ijd,db->ijb", pair, self.W["dist_head"])

    def plddt_head(self, seq_repr: np.ndarray, pair: np.ndarray) -> np.ndarray:
        pooled = pair.mean(axis=1)
        f = np.concatenate([seq_repr, pooled], axis=-1)
        logits = (f @ self.W["plddt_head"]).squeeze(-1)
        return 100.0 / (1.0 + np.exp(-logits))

    def _single(self, sequence: str, msa: Sequence[str], state: str) -> Prediction:
        seq_repr = self.sequence_embedding(sequence, state)
        msa_repr = self.msa_encoder(msa)
        pair = self.pair_representation(seq_repr, msa_repr)

        for _ in range(self.config.n_recycles):
            pair = self.evoformer_recycle(pair)

        coords = self._ipa_refine(pair, self.config.n_refine_steps)
        coords = self._diffusion_refine(coords, pair, self.config.n_diffusion_steps)

        torsion_logits, torsion_angles = self.torsion_head(seq_repr, pair)
        dist_logits = self.distogram_head(pair)
        plddt = self.plddt_head(seq_repr, pair)
        return Prediction(sequence, state, coords, torsion_logits, torsion_angles, dist_logits, plddt, pair)

    def ensemble_sample(self, sequence: str, msa: Sequence[str], state: str) -> Dict[str, object]:
        members: List[Prediction] = []
        seed0 = self.config.random_seed or 0

        for i in range(self.config.n_ensemble):
            self.rng = np.random.default_rng(seed0 + 8191 * (i + 1))
            self.W = self._init_weights()
            members.append(self._single(sequence, msa, state))

        coords = np.stack([m.coordinates for m in members], axis=0)
        plddt = np.stack([m.plddt for m in members], axis=0)

        return {
            "state": state,
            "members": members,
            "mean_coordinates": coords.mean(axis=0),
            "coordinate_variance": coords.var(axis=0),
            "mean_plddt": plddt.mean(axis=0),
            "ensemble_diversity": float(coords.var(axis=0).mean()),
        }

    def allosteric_landscape(self, sequence: str, msa: Sequence[str]) -> Dict[str, Dict[str, object]]:
        landscape: Dict[str, Dict[str, object]] = {}
        for state in self.config.allosteric_states:
            landscape[state] = self.ensemble_sample(sequence, msa, state)
        return landscape


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HyperFold-X architecture prototype")
    p.add_argument("sequence")
    p.add_argument("--msa", nargs="*", default=None)
    p.add_argument("--state", default="inactive")
    p.add_argument("--ensemble", type=int, default=6)
    p.add_argument("--recycles", type=int, default=3)
    p.add_argument("--evo-blocks", type=int, default=4)
    p.add_argument("--json", action="store_true")
    return p


def main() -> None:
    args = _parser().parse_args()
    cfg = ModelConfig(n_ensemble=args.ensemble, n_recycles=args.recycles, n_evo_blocks=args.evo_blocks)
    model = HyperFoldX(cfg)
    msa = args.msa if args.msa else [args.sequence]

    if args.state == "all":
        landscape = model.allosteric_landscape(args.sequence, msa)
        summary = {
            k: {
                "mean_plddt": float(v["mean_plddt"].mean()),
                "diversity": float(v["ensemble_diversity"]),
                "shape": list(v["mean_coordinates"].shape),
            }
            for k, v in landscape.items()
        }
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            for s, d in summary.items():
                print(f"State={s:14s} pLDDT={d['mean_plddt']:.2f} diversity={d['diversity']:.6f}")
        return

    if args.state not in cfg.allosteric_states:
        raise ValueError(f"state must be in {cfg.allosteric_states} or 'all'")

    out = model.ensemble_sample(args.sequence, msa, args.state)
    payload = {
        "state": out["state"],
        "mean_coordinates": out["mean_coordinates"].tolist(),
        "coordinate_variance": out["coordinate_variance"].tolist(),
        "mean_plddt": out["mean_plddt"].tolist(),
        "ensemble_diversity": out["ensemble_diversity"],
        "members": [m.to_jsonable() for m in out["members"]],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"State: {payload['state']}")
        print(f"Length: {len(args.sequence)}")
        print(f"Ensemble: {len(payload['members'])}")
        print(f"Mean pLDDT: {float(np.mean(out['mean_plddt'])):.2f}")
        print(f"Ensemble diversity: {payload['ensemble_diversity']:.6f}")
        print("First 5 residues (mean coordinates):")
        for i, c in enumerate(payload["mean_coordinates"][:5]):
            print(f"  {i:3d}: ({c[0]: .3f}, {c[1]: .3f}, {c[2]: .3f})")


if __name__ == "__main__":
    main()
