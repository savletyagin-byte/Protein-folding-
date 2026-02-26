"""OmegaFold-Ultra: an extensive NumPy protein-structure research prototype.

This code intentionally combines many architecture ideas into one runnable scaffold:
- sequence/state/biochemical embeddings
- MSA encoder with axial mixing + dropout + coupling map
- pair representation with multiple priors
- triangle updates + recycling
- SE(3)-equivariant IPA-like refinement
- diffusion denoising refinement
- torsion/distogram/pLDDT/PAE heads
- uncertainty calibration and self-consistency scoring
- lightweight torsion-space annealing relaxation
- ensemble ranking, clustering, and allosteric landscape modeling

It is an untrained prototype intended for experimentation and education.
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
    d_seq: int = 256
    d_msa: int = 192
    d_pair: int = 144
    n_evo_blocks: int = 7
    n_recycles: int = 5
    n_triangle_updates: int = 2
    n_refine_steps: int = 32
    n_diffusion_steps: int = 20
    torsion_bins: int = 72
    dist_bins: int = 96
    n_ensemble: int = 10
    msa_dropout: float = 0.1
    recycle_tol: float = 8e-5
    anneal_steps: int = 120
    anneal_temp0: float = 1.5
    random_seed: Optional[int] = 31
    allosteric_states: Tuple[str, ...] = (
        "inactive",
        "active",
        "intermediate",
        "agonist-bound",
        "inhibitor-bound",
        "gprotein-coupled",
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
    self_consistency: float

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
            "self_consistency": self.self_consistency,
        }


class OmegaFoldUltra:
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
            "pos_embed": n(8192, c.d_seq),
            "state_embed": n(len(c.allosteric_states), c.d_seq, scale=0.02),
            "bio_embed": n(6, c.d_seq),
            "msa_embed": n(len(AA_VOCAB), c.d_msa),
            "msa_row_mix": n(c.d_msa, c.d_msa),
            "msa_col_mix": n(c.d_msa, c.d_msa),
            "msa_to_pair_l": n(c.d_msa, c.d_pair),
            "msa_to_pair_r": n(c.d_msa, c.d_pair),
            "seq_to_pair_l": n(c.d_seq, c.d_pair),
            "seq_to_pair_r": n(c.d_seq, c.d_pair),
            "pair_bias_proj": n(8, c.d_pair),
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
            "rank_head": n(8, 1),
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
        small = set("AGSCTPDN")
        out = []
        for aa in sequence:
            out.append([
                aa in hydrophobic,
                aa in positive,
                aa in negative,
                aa in aromatic,
                aa in special,
                aa in small,
            ])
        return np.asarray(out, dtype=np.float64)

    def sequence_embedding(self, sequence: str, state: str) -> np.ndarray:
        idx = self._idx(sequence)
        if len(sequence) > self.W["pos_embed"].shape[0]:
            raise ValueError("Sequence too long for positional table")

        seq = self.W["seq_embed"][idx] + self.W["pos_embed"][np.arange(len(sequence))]
        state_idx = self.config.allosteric_states.index(state)
        seq = seq + self.W["state_embed"][state_idx][None, :]
        seq = seq + 0.25 * (self._bio_features(sequence) @ self.W["bio_embed"])
        return seq

    def msa_encoder(self, msa: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        if not msa:
            raise ValueError("MSA cannot be empty")
        L = len(msa[0])
        if any(len(r) != L for r in msa):
            raise ValueError("MSA rows must have same length")

        idx = np.stack([self._idx(r) for r in msa], axis=0)
        m = self.W["msa_embed"][idx]

        if self.config.msa_dropout > 0:
            keep = self.rng.random(m.shape[:2]) > self.config.msa_dropout
            m = m * keep[..., None]

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
        helix = np.exp(-((sep - 4.0) ** 2) / 14.0)
        sheet = np.exp(-((sep - 8.0) ** 2) / 18.0)
        long_range = np.exp(-((sep - 14.0) ** 2) / 64.0)
        return 0.45 * helix + 0.35 * sheet + 0.20 * long_range

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
        sep = np.abs(ii - jj)
        rel = np.tanh((ii - jj) / 24.0)
        log_sep = np.log1p(sep)
        inv_sep = 1.0 / (1.0 + sep)
        template = self._template_prior(L)
        contact_prior = np.exp(-sep / 12.0)
        band_prior = (sep <= 4).astype(float)
        anti_band = (sep >= 12).astype(float)

        bias = np.stack(
            [
                rel,
                log_sep / (log_sep.max() + 1e-6),
                inv_sep,
                template,
                coupling,
                contact_prior,
                band_prior,
                anti_band,
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
        q = z @ self.W["tri_q"]
        k = z @ self.W["tri_k"]
        v = z @ self.W["tri_v"]
        logits = np.einsum("ijd,ikd->ijk", q, k) / math.sqrt(z.shape[-1])
        a = self._softmax(logits, axis=-1)
        return np.einsum("ijk,ikd->ijd", a, v)

    def evoformer_recycle(self, pair: np.ndarray) -> np.ndarray:
        z = pair.copy()
        for _ in range(self.config.n_evo_blocks):
            for _ in range(self.config.n_triangle_updates):
                z_new = z + 0.22 * np.tanh(self._triangle_mul(z, True))
                z_new = z_new + 0.22 * np.tanh(self._triangle_mul(z_new, False))
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
            force = gate * (0.22 * unit + 0.04 * force_feat) / np.sqrt(d2)

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
            noise = self.rng.normal(0.0, sigma, size=x.shape)
            xn = x + noise
            smooth = np.zeros_like(x)
            smooth[1:-1] = 0.5 * (xn[:-2] + xn[2:]) - xn[1:-1]
            x = xn + 0.22 * (0.58 * smooth + 0.42 * cond)
            x -= x.mean(axis=0, keepdims=True)
        return x

    def _anneal_relax(self, coords: np.ndarray) -> np.ndarray:
        x = coords.copy()
        T = self.config.anneal_temp0

        def energy(c: np.ndarray) -> float:
            b = np.linalg.norm(c[1:] - c[:-1], axis=-1)
            bond = np.mean((b - 3.8) ** 2)
            rg = np.sqrt(np.mean(np.sum((c - c.mean(axis=0)) ** 2, axis=-1)))
            rel = c[:, None, :] - c[None, :, :]
            d = np.sqrt(np.sum(rel * rel, axis=-1) + 1e-8)
            clash = np.mean((d < 1.2) & (~np.eye(len(c), dtype=bool)))
            return float(2.5 * bond + 0.6 / (rg + 1e-6) + 8.0 * clash)

        e = energy(x)
        for step in range(self.config.anneal_steps):
            i = self.rng.integers(0, len(x))
            proposal = x.copy()
            proposal[i] += self.rng.normal(0.0, 0.2, size=3)
            proposal -= proposal.mean(axis=0, keepdims=True)
            en = energy(proposal)
            de = en - e
            if de <= 0 or self.rng.random() < math.exp(-de / max(1e-6, T)):
                x, e = proposal, en
            T = self.config.anneal_temp0 * (1 - (step + 1) / self.config.anneal_steps)
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
        p = 30.0 / (1.0 + np.exp(-logits))
        return 0.5 * (p + p.T)

    @staticmethod
    def _physical_quality(coords: np.ndarray) -> Dict[str, float]:
        rel = coords[:, None, :] - coords[None, :, :]
        d = np.sqrt(np.sum(rel * rel, axis=-1) + 1e-8)
        L = len(coords)
        clash = float(((d < 1.1) & (~np.eye(L, dtype=bool))).sum()) / max(1, L * L)
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

    @staticmethod
    def _self_consistency(pair: np.ndarray, dist_logits: np.ndarray) -> float:
        # compare implied short-distance probability with pair norm signal
        probs = np.exp(dist_logits - dist_logits.max(axis=-1, keepdims=True))
        probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-9, None)
        short_prob = probs[..., :8].mean(axis=-1)
        pair_signal = np.tanh(np.linalg.norm(pair, axis=-1) / max(1e-6, pair.shape[-1]))
        return float(1.0 - np.mean(np.abs(short_prob - pair_signal)))

    def _calibrate_uncertainty(self, plddt: np.ndarray, pae: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        plddt_cal = np.clip(0.85 * plddt + 0.15 * (100 - pae.mean(axis=-1)), 0, 100)
        pae_cal = np.clip(0.85 * pae + 0.15 * (30 - np.minimum(30, plddt[:, None] / 3.0)), 0, 30)
        pae_cal = 0.5 * (pae_cal + pae_cal.T)
        return plddt_cal, pae_cal

    def _rank_score(self, pred: Prediction, diversity: float) -> float:
        q = pred.quality
        feats = np.asarray(
            [
                float(pred.plddt.mean()) / 100.0,
                1.0 / (1.0 + float(pred.pae.mean()) / 30.0),
                1.0 - q["clash_fraction"],
                1.0 / (1.0 + q["bond_deviation"]),
                q["compactness"],
                pred.self_consistency,
                1.0 / (1.0 + diversity),
                1.0 / (1.0 + q["radius_of_gyration"]),
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
        coords = self._anneal_relax(coords)

        torsion_logits, torsion_angles = self.torsion_head(seq_repr, pair)
        dist_logits = self.distogram_head(pair)
        plddt = self.plddt_head(seq_repr, pair)
        pae = self.pae_head(pair)
        plddt, pae = self._calibrate_uncertainty(plddt, pae)

        quality = self._physical_quality(coords)
        consistency = self._self_consistency(pair, dist_logits)

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
            self_consistency=consistency,
        )

    def ensemble_sample(self, sequence: str, msa: Sequence[str], state: str) -> Dict[str, object]:
        members: List[Prediction] = []
        seed0 = self.config.random_seed or 0

        for i in range(self.config.n_ensemble):
            self.rng = np.random.default_rng(seed0 + 100003 * (i + 1))
            self.W = self._init_weights()
            members.append(self._single(sequence, msa, state))

        coords = np.stack([m.coordinates for m in members], axis=0)
        plddt = np.stack([m.plddt for m in members], axis=0)
        pae = np.stack([m.pae for m in members], axis=0)

        flat = coords.reshape(coords.shape[0], -1)
        center = np.median(flat, axis=0)
        dist_center = np.linalg.norm(flat - center[None, :], axis=1)
        diversity = float(coords.var(axis=0).mean())

        ranked = []
        for i, m in enumerate(members):
            ranked.append((self._rank_score(m, diversity), i))
        ranked.sort(reverse=True)

        return {
            "state": state,
            "members": members,
            "ranked_indices": [i for _, i in ranked],
            "best_member_index": ranked[0][1],
            "mean_coordinates": coords.mean(axis=0),
            "coordinate_variance": coords.var(axis=0),
            "mean_plddt": plddt.mean(axis=0),
            "mean_pae": pae.mean(axis=0),
            "mean_self_consistency": float(np.mean([m.self_consistency for m in members])),
            "ensemble_diversity": diversity,
            "distance_to_center": dist_center.tolist(),
        }

    def allosteric_landscape(self, sequence: str, msa: Sequence[str]) -> Dict[str, Dict[str, object]]:
        out = {}
        for state in self.config.allosteric_states:
            out[state] = self.ensemble_sample(sequence, msa, state)
        return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OmegaFold-Ultra prototype")
    p.add_argument("sequence")
    p.add_argument("--msa", nargs="*", default=None)
    p.add_argument("--state", default="inactive")
    p.add_argument("--ensemble", type=int, default=10)
    p.add_argument("--recycles", type=int, default=5)
    p.add_argument("--evo-blocks", type=int, default=7)
    p.add_argument("--json", action="store_true")
    return p


def main() -> None:
    args = _parser().parse_args()
    cfg = ModelConfig(n_ensemble=args.ensemble, n_recycles=args.recycles, n_evo_blocks=args.evo_blocks)
    model = OmegaFoldUltra(cfg)
    msa = args.msa if args.msa else [args.sequence]

    if args.state == "all":
        landscape = model.allosteric_landscape(args.sequence, msa)
        summary = {
            s: {
                "mean_plddt": float(v["mean_plddt"].mean()),
                "mean_pae": float(v["mean_pae"].mean()),
                "mean_self_consistency": float(v["mean_self_consistency"]),
                "diversity": float(v["ensemble_diversity"]),
                "best_member_index": int(v["best_member_index"]),
                "shape": list(v["mean_coordinates"].shape),
            }
            for s, v in landscape.items()
        }
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            for s, d in summary.items():
                print(
                    f"State={s:18s} pLDDT={d['mean_plddt']:.2f} PAE={d['mean_pae']:.2f} "
                    f"SC={d['mean_self_consistency']:.3f} div={d['diversity']:.6f}"
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
        "mean_self_consistency": out["mean_self_consistency"],
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
        print(f"Mean self-consistency: {payload['mean_self_consistency']:.3f}")
        print(f"Ensemble diversity: {payload['ensemble_diversity']:.6f}")
        print("First 5 residues (mean coordinates):")
        for i, c in enumerate(payload["mean_coordinates"][:5]):
            print(f"  {i:3d}: ({c[0]: .3f}, {c[1]: .3f}, {c[2]: .3f})")


if __name__ == "__main__":
    main()
