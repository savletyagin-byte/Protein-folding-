"""OmegaFold-Prototype-X: maximal architecture-style protein folding scaffold.

This is a NumPy-only, research-oriented prototype that aggregates a broad set of modern
protein-structure modeling concepts in one executable codebase.

Included capabilities:
- Sequence embedding (token + position + biochemical channels + allosteric state context)
- MSA encoder (axial mixing + dropout + coevolution coupling extraction)
- Pair representation (sequence/MSA/template/contact priors + relative geometry bias)
- Evoformer-like triangle blocks with recycling and early-stop criteria
- SE(3)-equivariant refinement (IPA-like force updates)
- Diffusion-like denoising refinement
- Geometric heads: torsions, distograms, confidence (pLDDT-like), PAE-like matrix
- Physical plausibility penalties: clash, bond smoothness, compactness
- Ensemble generation, consensus/ranking, diversity clustering
- Allosteric landscape modeling across multiple states

Important: this is still an untrained prototype and cannot be claimed to beat SOTA models.
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
    d_seq: int = 224
    d_msa: int = 160
    d_pair: int = 128
    n_evo_blocks: int = 6
    n_recycles: int = 4
    n_triangle_updates: int = 2
    n_refine_steps: int = 28
    n_diffusion_steps: int = 18
    torsion_bins: int = 64
    dist_bins: int = 96
    n_ensemble: int = 8
    msa_dropout: float = 0.1
    recycle_tol: float = 1e-4
    random_seed: Optional[int] = 23
    allosteric_states: Tuple[str, ...] = (
        "inactive",
        "active",
        "intermediate",
        "agonist-bound",
        "inhibitor-bound",
    )


@dataclass
class Prediction:
    sequence: str
    state: str
    coordinates: np.ndarray
    torsion_logits: np.ndarray
    torsion_angles: np.ndarray
    distogram_logits: np.ndarray
    plddt: np.ndarray
    pae: np.ndarray
    pair: np.ndarray
    quality: Dict[str, float]

    def to_jsonable(self) -> Dict[str, object]:
        return {
            "sequence": self.sequence,
            "state": self.state,
            "coordinates": self.coordinates.tolist(),
            "torsion_angles": self.torsion_angles.tolist(),
            "torsion_logits_shape": list(self.torsion_logits.shape),
            "distogram_logits_shape": list(self.distogram_logits.shape),
            "plddt": self.plddt.tolist(),
            "pae": self.pae.tolist(),
            "pair_shape": list(self.pair.shape),
            "quality": self.quality,
        }


class OmegaFoldPrototypeX:
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
            "pos_embed": n(4096, c.d_seq),
            "state_embed": n(len(c.allosteric_states), c.d_seq, scale=0.02),
            "bio_embed": n(5, c.d_seq),  # hydrophobicity/charge/size/aromaticity/special
            "msa_embed": n(len(AA_VOCAB), c.d_msa),
            "msa_row_mix": n(c.d_msa, c.d_msa),
            "msa_col_mix": n(c.d_msa, c.d_msa),
            "msa_to_pair_l": n(c.d_msa, c.d_pair),
            "msa_to_pair_r": n(c.d_msa, c.d_pair),
            "seq_to_pair_l": n(c.d_seq, c.d_pair),
            "seq_to_pair_r": n(c.d_seq, c.d_pair),
            "pair_bias_proj": n(6, c.d_pair),
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
            "pae_head": n(c.d_pair, 1),
            "rank_head": n(6, 1),
        }

    @staticmethod
    def _idx(seq: str) -> np.ndarray:
        return np.asarray([AA_TO_IDX.get(s, AA_TO_IDX["X"]) for s in seq], dtype=np.int64)

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-9, None)

    @staticmethod
    def _bio_features(sequence: str) -> np.ndarray:
        hydrophobic = set("AILMFWVY")
        positive = set("KRH")
        negative = set("DE")
        aromatic = set("FWYH")
        special = set("CGP")
        feats = []
        for aa in sequence:
            feats.append(
                [
                    1.0 if aa in hydrophobic else 0.0,
                    1.0 if aa in positive else 0.0,
                    1.0 if aa in negative else 0.0,
                    1.0 if aa in aromatic else 0.0,
                    1.0 if aa in special else 0.0,
                ]
            )
        return np.asarray(feats, dtype=np.float64)

    def sequence_embedding(self, sequence: str, state: str) -> np.ndarray:
        idx = self._idx(sequence)
        if len(sequence) > self.W["pos_embed"].shape[0]:
            raise ValueError("Sequence too long for positional embeddings.")
        seq = self.W["seq_embed"][idx] + self.W["pos_embed"][np.arange(len(sequence))]

        sidx = self.config.allosteric_states.index(state)
        seq = seq + self.W["state_embed"][sidx][None, :]

        bio = self._bio_features(sequence) @ self.W["bio_embed"]
        return seq + 0.3 * bio

    def msa_encoder(self, msa: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        if not msa:
            raise ValueError("MSA cannot be empty.")
        L = len(msa[0])
        if any(len(m) != L for m in msa):
            raise ValueError("All MSA rows must share same length.")

        idx = np.stack([self._idx(m) for m in msa], axis=0)
        m = self.W["msa_embed"][idx]  # (N,L,d_msa)

        # stochastic masking for ensemble diversity
        if self.config.msa_dropout > 0:
            mask = self.rng.random(size=m.shape[:2]) > self.config.msa_dropout
            m = m * mask[..., None]

        row = np.tanh(m @ self.W["msa_row_mix"])
        col = np.transpose(np.transpose(m, (1, 0, 2)) @ self.W["msa_col_mix"], (1, 0, 2))
        fused = 0.5 * m + 0.25 * row + 0.25 * np.tanh(col)

        msa_repr = fused.mean(axis=0)
        coupling = np.tanh((msa_repr @ msa_repr.T) / max(1, msa_repr.shape[-1]))
        return msa_repr, coupling

    @staticmethod
    def _template_prior(L: int) -> np.ndarray:
        i = np.arange(L)[:, None]
        j = np.arange(L)[None, :]
        sep = np.abs(i - j)
        helix = np.exp(-((sep - 4.0) ** 2) / 16.0)
        sheet = np.exp(-((sep - 8.0) ** 2) / 20.0)
        return 0.6 * helix + 0.4 * sheet

    def pair_representation(self, seq_repr: np.ndarray, msa_repr: np.ndarray, coupling: np.ndarray) -> np.ndarray:
        L = seq_repr.shape[0]
        z = (
            (seq_repr @ self.W["seq_to_pair_l"])[:, None, :]
            + (seq_repr @ self.W["seq_to_pair_r"])[None, :, :]
            + (msa_repr @ self.W["msa_to_pair_l"])[:, None, :]
            + (msa_repr @ self.W["msa_to_pair_r"])[None, :, :]
        )

        ii = np.arange(L)[:, None]
        jj = np.arange(L)[None, :]
        rel = np.tanh((ii - jj) / 24.0)
        sep = np.log1p(np.abs(ii - jj))
        inv_sep = 1.0 / (1.0 + np.abs(ii - jj))
        template = self._template_prior(L)
        contact_prior = np.exp(-np.abs(ii - jj) / 12.0)

        bias = np.stack(
            [
                rel,
                sep / (sep.max() + 1e-6),
                inv_sep,
                template,
                coupling,
                contact_prior,
            ],
            axis=-1,
        )
        z = z + bias @ self.W["pair_bias_proj"]
        z = 0.5 * (z + np.transpose(z, (1, 0, 2)))
        return z

    def _triangle_mul(self, z: np.ndarray, outgoing: bool) -> np.ndarray:
        w = self.W["tri_mul_out"] if outgoing else self.W["tri_mul_in"]
        h = np.tanh(z @ w)
        if outgoing:
            return np.einsum("ikd,jkd->ijd", h, h) / math.sqrt(z.shape[-1])
        return np.einsum("kid,kjd->ijd", h, h) / math.sqrt(z.shape[-1])

    def _triangle_attn(self, z: np.ndarray) -> np.ndarray:
        q, k, v = z @ self.W["tri_q"], z @ self.W["tri_k"], z @ self.W["tri_v"]
        logits = np.einsum("ijd,ikd->ijk", q, k) / math.sqrt(z.shape[-1])
        a = self._softmax(logits, axis=-1)
        return np.einsum("ijk,ikd->ijd", a, v)

    def evoformer_recycle(self, pair: np.ndarray) -> np.ndarray:
        z = pair.copy()
        for _ in range(self.config.n_evo_blocks):
            for _ in range(self.config.n_triangle_updates):
                z_new = z
                z_new = z_new + 0.22 * np.tanh(self._triangle_mul(z_new, outgoing=True))
                z_new = z_new + 0.22 * np.tanh(self._triangle_mul(z_new, outgoing=False))
                z_new = z_new + 0.18 * np.tanh(self._triangle_attn(z_new))
                z_new = 0.5 * (z_new + np.transpose(z_new, (1, 0, 2)))

                delta = float(np.mean((z_new - z) ** 2))
                z = z_new
                if delta < self.config.recycle_tol:
                    break
        return z

    def _ipa_refine(self, pair: np.ndarray, steps: int) -> np.ndarray:
        L = pair.shape[0]
        coords = np.stack([np.arange(L), np.zeros(L), np.zeros(L)], axis=-1).astype(np.float64)

        for _ in range(steps):
            rel = coords[:, None, :] - coords[None, :, :]
            d2 = np.sum(rel * rel, axis=-1, keepdims=True) + 1e-6
            unit = rel / np.sqrt(d2)
            gate = 1.0 / (1.0 + np.exp(-(pair @ self.W["ipa_pair_gate"])))
            force_feat = np.tanh(pair @ self.W["ipa_pair_to_force"])
            force = gate * (0.20 * unit + 0.05 * force_feat) / np.sqrt(d2)

            smooth = np.zeros_like(coords)
            smooth[1:-1] = 0.5 * (coords[:-2] + coords[2:]) - coords[1:-1]

            delta = (force.sum(axis=1) - force.sum(axis=0)) + 0.08 * smooth
            coords += 0.03 * delta
            coords -= coords.mean(axis=0, keepdims=True)
        return coords

    def _diffusion_refine(self, coords: np.ndarray, pair: np.ndarray, steps: int) -> np.ndarray:
        x = coords.copy()
        cond = np.tanh(pair @ self.W["diffusion_cond"]).mean(axis=1)
        for t in range(steps, 0, -1):
            sigma = 0.08 * (t / max(1, steps))
            eps = self.rng.normal(0.0, sigma, size=x.shape)
            x_noisy = x + eps

            smooth = np.zeros_like(x)
            smooth[1:-1] = 0.5 * (x_noisy[:-2] + x_noisy[2:]) - x_noisy[1:-1]
            denoise = 0.6 * smooth + 0.4 * cond
            x = x_noisy + 0.24 * denoise
            x -= x.mean(axis=0, keepdims=True)
        return x

    def torsion_head(self, seq_repr: np.ndarray, pair: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pooled = pair.mean(axis=1)
        f = np.concatenate([seq_repr, pooled], axis=-1)
        logits = (f @ self.W["torsion_head"]).reshape(len(seq_repr), 3, self.config.torsion_bins)
        probs = self._softmax(logits, axis=-1)
        idx = probs.argmax(axis=-1)
        angles = -math.pi + (2 * math.pi) * idx / max(1, self.config.torsion_bins - 1)
        return logits, angles

    def distogram_head(self, pair: np.ndarray) -> np.ndarray:
        return np.einsum("ijd,db->ijb", pair, self.W["dist_head"])

    def plddt_head(self, seq_repr: np.ndarray, pair: np.ndarray) -> np.ndarray:
        pooled = pair.mean(axis=1)
        f = np.concatenate([seq_repr, pooled], axis=-1)
        logits = (f @ self.W["plddt_head"]).squeeze(-1)
        return 100.0 / (1.0 + np.exp(-logits))

    def pae_head(self, pair: np.ndarray) -> np.ndarray:
        logits = (pair @ self.W["pae_head"]).squeeze(-1)
        pae = 30.0 / (1.0 + np.exp(-logits))
        return 0.5 * (pae + pae.T)

    @staticmethod
    def _physical_quality(coords: np.ndarray) -> Dict[str, float]:
        rel = coords[:, None, :] - coords[None, :, :]
        d = np.sqrt(np.sum(rel * rel, axis=-1) + 1e-8)
        L = len(coords)

        clash_mask = (d < 1.0) & (~np.eye(L, dtype=bool))
        clash = float(clash_mask.sum()) / max(1, L * L)

        bond = np.linalg.norm(coords[1:] - coords[:-1], axis=-1)
        bond_dev = float(np.mean((bond - 3.8) ** 2))

        rg = float(np.sqrt(np.mean(np.sum((coords - coords.mean(axis=0)) ** 2, axis=-1))))
        compact = 1.0 / (1.0 + rg)
        return {
            "clash_fraction": clash,
            "bond_deviation": bond_dev,
            "radius_of_gyration": rg,
            "compactness": compact,
        }

    def _rank_score(self, plddt: np.ndarray, pae: np.ndarray, quality: Dict[str, float], diversity: float) -> float:
        feats = np.asarray(
            [
                float(plddt.mean()) / 100.0,
                1.0 / (1.0 + float(pae.mean()) / 30.0),
                1.0 - quality["clash_fraction"],
                1.0 / (1.0 + quality["bond_deviation"]),
                quality["compactness"],
                1.0 / (1.0 + diversity),
            ],
            dtype=np.float64,
        )
        return float((feats @ self.W["rank_head"]).squeeze())

    def _single(self, sequence: str, msa: Sequence[str], state: str) -> Prediction:
        seq_repr = self.sequence_embedding(sequence, state)
        msa_repr, coupling = self.msa_encoder(msa)
        pair = self.pair_representation(seq_repr, msa_repr, coupling)

        for _ in range(self.config.n_recycles):
            updated = self.evoformer_recycle(pair)
            delta = float(np.mean((updated - pair) ** 2))
            pair = updated
            if delta < self.config.recycle_tol:
                break

        coords = self._ipa_refine(pair, self.config.n_refine_steps)
        coords = self._diffusion_refine(coords, pair, self.config.n_diffusion_steps)

        torsion_logits, torsion_angles = self.torsion_head(seq_repr, pair)
        dist_logits = self.distogram_head(pair)
        plddt = self.plddt_head(seq_repr, pair)
        pae = self.pae_head(pair)
        quality = self._physical_quality(coords)

        return Prediction(
            sequence=sequence,
            state=state,
            coordinates=coords,
            torsion_logits=torsion_logits,
            torsion_angles=torsion_angles,
            distogram_logits=dist_logits,
            plddt=plddt,
            pae=pae,
            pair=pair,
            quality=quality,
        )

    def ensemble_sample(self, sequence: str, msa: Sequence[str], state: str) -> Dict[str, object]:
        members: List[Prediction] = []
        seed0 = self.config.random_seed or 0

        for i in range(self.config.n_ensemble):
            self.rng = np.random.default_rng(seed0 + 104729 * (i + 1))
            self.W = self._init_weights()
            members.append(self._single(sequence, msa, state))

        coords = np.stack([m.coordinates for m in members], axis=0)
        plddt = np.stack([m.plddt for m in members], axis=0)
        pae = np.stack([m.pae for m in members], axis=0)

        # simple clustering-by-median-distance marker
        flat = coords.reshape(coords.shape[0], -1)
        center = np.median(flat, axis=0)
        dist = np.linalg.norm(flat - center[None, :], axis=1)
        diversity = float(coords.var(axis=0).mean())

        ranked = []
        for i, m in enumerate(members):
            r = self._rank_score(m.plddt, m.pae, m.quality, diversity)
            ranked.append((r, i))
        ranked.sort(reverse=True)

        best_idx = ranked[0][1]
        return {
            "state": state,
            "members": members,
            "ranked_indices": [i for _, i in ranked],
            "best_member_index": best_idx,
            "mean_coordinates": coords.mean(axis=0),
            "coordinate_variance": coords.var(axis=0),
            "mean_plddt": plddt.mean(axis=0),
            "mean_pae": pae.mean(axis=0),
            "ensemble_diversity": diversity,
            "distance_to_center": dist.tolist(),
        }

    def allosteric_landscape(self, sequence: str, msa: Sequence[str]) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for s in self.config.allosteric_states:
            out[s] = self.ensemble_sample(sequence, msa, s)
        return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OmegaFold-Prototype-X")
    p.add_argument("sequence")
    p.add_argument("--msa", nargs="*", default=None)
    p.add_argument("--state", default="inactive")
    p.add_argument("--ensemble", type=int, default=8)
    p.add_argument("--recycles", type=int, default=4)
    p.add_argument("--evo-blocks", type=int, default=6)
    p.add_argument("--json", action="store_true")
    return p


def main() -> None:
    args = _parser().parse_args()
    cfg = ModelConfig(n_ensemble=args.ensemble, n_recycles=args.recycles, n_evo_blocks=args.evo_blocks)
    model = OmegaFoldPrototypeX(cfg)
    msa = args.msa if args.msa else [args.sequence]

    if args.state == "all":
        landscape = model.allosteric_landscape(args.sequence, msa)
        summary = {
            k: {
                "mean_plddt": float(v["mean_plddt"].mean()),
                "mean_pae": float(v["mean_pae"].mean()),
                "diversity": float(v["ensemble_diversity"]),
                "best_member_index": int(v["best_member_index"]),
                "shape": list(v["mean_coordinates"].shape),
            }
            for k, v in landscape.items()
        }
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            for s, d in summary.items():
                print(
                    f"State={s:16s} pLDDT={d['mean_plddt']:.2f} "
                    f"PAE={d['mean_pae']:.2f} diversity={d['diversity']:.6f}"
                )
        return

    if args.state not in cfg.allosteric_states:
        raise ValueError(f"state must be in {cfg.allosteric_states} or 'all'")

    out = model.ensemble_sample(args.sequence, msa, args.state)
    payload = {
        "state": out["state"],
        "ranked_indices": out["ranked_indices"],
        "best_member_index": out["best_member_index"],
        "mean_coordinates": out["mean_coordinates"].tolist(),
        "coordinate_variance": out["coordinate_variance"].tolist(),
        "mean_plddt": out["mean_plddt"].tolist(),
        "mean_pae": out["mean_pae"].tolist(),
        "ensemble_diversity": out["ensemble_diversity"],
        "distance_to_center": out["distance_to_center"],
        "members": [m.to_jsonable() for m in out["members"]],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"State: {payload['state']}")
        print(f"Length: {len(args.sequence)}")
        print(f"Ensemble: {len(payload['members'])}")
        print(f"Best member index: {payload['best_member_index']}")
        print(f"Mean pLDDT: {float(np.mean(out['mean_plddt'])):.2f}")
        print(f"Mean PAE: {float(np.mean(out['mean_pae'])):.2f}")
        print(f"Ensemble diversity: {payload['ensemble_diversity']:.6f}")
        print("First 5 residues (mean coordinates):")
        for i, c in enumerate(payload["mean_coordinates"][:5]):
            print(f"  {i:3d}: ({c[0]: .3f}, {c[1]: .3f}, {c[2]: .3f})")


if __name__ == "__main__":
    main()
