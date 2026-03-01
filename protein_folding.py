"""Geometric protein folding prototype with equivariant recycling.

Implements (compactly):
- Residue graphs (kNN / radius)
- Node + pair embeddings and message passing
- Long-range attention
- SE(3)-equivariant style updates using spherical-harmonic-inspired features
- Pair triangle updates (multiplicative + attention-like)
- Iterative recycling of structure/representations
- Multi-loss training: FAPE-like, distogram CE, torsion, lDDT-like, clash, bond
- Confidence heads: pLDDT-like and pairwise error (PAE-like)
- Visualization exports: single view, multiview, rotation GIF
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import autograd.numpy as anp
from autograd import grad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from Bio.PDB import PDBList, PDBParser, is_aa

try:
    import torch
    from e3nn import o3
except Exception:  # optional dependency
    torch = None
    o3 = None

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY-X"
AA_TO_IDX = {a: i for i, a in enumerate(AA_VOCAB)}
RES3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
RES1_TO_3 = {v: k for k, v in RES3_TO_1.items()}
RES1_TO_3.update({"X": "UNK", "-": "GLY"})


@dataclass
class ModelConfig:
    # backwards-compatible alias used by existing tests/callers
    d_hidden: Optional[int] = None
    d_node: int = 128
    d_pair: int = 64
    torsion_bins: int = 36
    dist_bins: int = 32
    dropout_rate: float = 0.0
    curriculum: bool = False
    use_swa: bool = False
    use_e3nn: bool = True
    e3nn_step_scale: float = 0.05

    # geometry
    knn_k: int = 8
    radius_cutoff: float = 10.0
    use_radius_graph: bool = False

    # trunk
    n_message_layers: int = 2
    n_triangle_layers: int = 2
    n_recycles: int = 4

    # train
    lr: float = 5e-3
    epochs: int = 20
    batch_size: int = 2
    random_seed: int = 7
    weight_decay: float = 1e-5
    grad_clip_norm: float = 5.0
    val_split: float = 0.2
    early_stopping_patience: int = 6

    def __post_init__(self) -> None:
        if self.d_hidden is not None:
            self.d_node = int(self.d_hidden)
            # keep pair width proportional unless explicitly overridden
            if self.d_pair == 64:
                self.d_pair = max(8, int(self.d_hidden) // 2)


@dataclass
class StructureExample:
    pdb_id: str
    sequence: str
    coords: np.ndarray


@dataclass
class Prediction:
    sequence: str
    coords: np.ndarray
    torsion_logits: np.ndarray
    distogram_logits: np.ndarray
    confidence: np.ndarray
    pair_error: np.ndarray

    def to_pdb(self, chain_id: str = "A") -> str:
        lines = []
        for i, (aa, c) in enumerate(zip(self.sequence, self.coords), start=1):
            resname = RES1_TO_3.get(aa, "UNK")
            x, y, z = map(float, c)
            lines.append(
                f"ATOM  {i:5d}  CA  {resname:>3s} {chain_id}{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
            )
        return "\n".join(lines + ["TER", "END"]) + "\n"


@dataclass
class TrainingResult:
    train_losses: List[float]
    val_losses: List[float]
    best_val_loss: float
    best_epoch: int
    epochs_ran: int
    lrs: List[float]


class RealStructureDataset:
    def __init__(self, examples: List[StructureExample]):
        self.examples = examples

    @staticmethod
    def _extract_from_structure(structure, pdb_id: str, min_len: int = 20, max_len: int = 400) -> Optional[StructureExample]:
        for model in structure:
            for chain in model:
                seq, coords = [], []
                for residue in chain:
                    if not is_aa(residue, standard=True) or "CA" not in residue:
                        continue
                    aa = RES3_TO_1.get(residue.get_resname().upper(), "X")
                    seq.append(aa)
                    coords.append(np.asarray(residue["CA"].get_coord(), dtype=np.float64))
                if min_len <= len(seq) <= max_len:
                    return StructureExample(pdb_id=pdb_id, sequence="".join(seq), coords=np.stack(coords, axis=0))
        return None

    @classmethod
    def from_local_pdbs(cls, pdb_paths: Sequence[str], min_len: int = 20, max_len: int = 400) -> "RealStructureDataset":
        parser = PDBParser(QUIET=True)
        out: List[StructureExample] = []
        for p in pdb_paths:
            structure = parser.get_structure(Path(p).stem, str(p))
            ex = cls._extract_from_structure(structure, Path(p).stem, min_len=min_len, max_len=max_len)
            if ex is not None:
                out.append(ex)
        return cls(out)

    @classmethod
    def from_rcsb_ids(cls, pdb_ids: Sequence[str], cache_dir: str = "pdb_cache", min_len: int = 20, max_len: int = 400) -> "RealStructureDataset":
        os.makedirs(cache_dir, exist_ok=True)
        pdbl = PDBList(verbose=False)
        parser = PDBParser(QUIET=True)
        out: List[StructureExample] = []
        for pid in pdb_ids:
            try:
                local_file = pdbl.retrieve_pdb_file(pid.lower(), pdir=cache_dir, file_format="pdb", overwrite=False)
                ex = cls._extract_from_structure(parser.get_structure(pid, local_file), pid, min_len=min_len, max_len=max_len)
                if ex is not None:
                    out.append(ex)
            except Exception:
                continue
        return cls(out)


def save_structure_image(coords: np.ndarray, out_path: str, title: str = "Predicted Protein") -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    ax.plot(x, y, z, linewidth=1.8)
    ax.scatter(x, y, z, s=12, c=np.arange(len(coords)), cmap="viridis")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_multiview_image(coords: np.ndarray, out_path: str, title: str = "Predicted Protein") -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 4), dpi=140)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    for i, (e, a, name) in enumerate([(20, 30, "iso"), (20, 120, "side"), (90, 0, "top")], start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot(x, y, z, linewidth=1.8)
        ax.scatter(x, y, z, s=10, c=np.arange(len(coords)), cmap="viridis")
        ax.view_init(elev=e, azim=a)
        ax.set_title(name)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_rotation_gif(coords: np.ndarray, out_path: str, title: str = "Predicted Protein", frames: int = 60, fps: int = 20) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 5), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    ax.plot(x, y, z, linewidth=1.8)
    ax.scatter(x, y, z, s=12, c=np.arange(len(coords)), cmap="viridis")
    ax.set_title(title)

    def _u(f: int):
        ax.view_init(elev=24, azim=360.0 * f / max(1, frames))
        return ()

    ani = animation.FuncAnimation(fig, _u, frames=frames, interval=int(1000 / max(1, fps)))
    ani.save(str(out), writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


class TrainableProteinModel:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.params = self._init_params()
        self._init_e3nn()

    def _init_e3nn(self) -> None:
        self._e3nn_ready = False
        self._e3nn_warning = None
        if not self.config.use_e3nn:
            return
        if torch is None or o3 is None:
            self._e3nn_warning = "e3nn/torch unavailable; falling back to numpy coordinate updates."
            return
        try:
            self._e3nn_irreps_in = o3.Irreps("1x0e + 1x1o")
            self._e3nn_vec = o3.Linear(self._e3nn_irreps_in, o3.Irreps("1x1o"), internal_weights=True, shared_weights=True)
            self._e3nn_gate_w = torch.tensor(self.rng.normal(0.0, 0.05, size=(self.config.d_node + self.config.d_pair,)), dtype=torch.float32)
            self._e3nn_ready = True
        except Exception as exc:
            self._e3nn_warning = f"e3nn init failed ({exc}); using numpy fallback."

    def _init_params(self) -> Dict[str, np.ndarray]:
        d_in = len(AA_VOCAB) + 6
        dn, dp = self.config.d_node, self.config.d_pair
        tb, db = self.config.torsion_bins * 3, self.config.dist_bins

        def n(*shape: int, scale: float = 0.05) -> np.ndarray:
            return self.rng.normal(0.0, scale, size=shape)

        return {
            # node trunk
            "W_in": n(d_in, dn), "b_in": np.zeros(dn),
            "W_msg": n(dn + dp + 9, dn), "b_msg": np.zeros(dn),
            "W_attn_q": n(dn, dn), "W_attn_k": n(dn, dn), "W_attn_v": n(dn, dn),
            # pair trunk + triangles
            "W_pair_init": n(dn, dp),
            "W_tri_mul": n(dp, dp),
            "W_tri_attn_q": n(dp, dp), "W_tri_attn_k": n(dp, dp), "W_tri_attn_v": n(dp, dp),
            # se(3)-style update
            "W_coord_gate": n(dn + dp, 1),
            # heads
            "W_coord": n(dn, 3), "b_coord": np.zeros(3),
            "W_tors": n(dn, tb), "b_tors": np.zeros(tb),
            "W_dist": n(dp, db), "b_dist": np.zeros(db),
            "W_conf": n(dn, 1), "b_conf": np.zeros(1),
            "W_pae": n(dp, 1), "b_pae": np.zeros(1),
        }

    @staticmethod
    def _bio_features(sequence: str) -> np.ndarray:
        hydrophobic = set("AILMFWVY")
        positive = set("KRH")
        negative = set("DE")
        aromatic = set("FWYH")
        special = set("CGP")
        small = set("AGSCTPDN")
        return np.asarray([[float(a in hydrophobic), float(a in positive), float(a in negative), float(a in aromatic), float(a in special), float(a in small)] for a in sequence], dtype=np.float64)

    @staticmethod
    def _encode(sequence: str) -> np.ndarray:
        oh = np.zeros((len(sequence), len(AA_VOCAB)), dtype=np.float64)
        for i, aa in enumerate(sequence):
            oh[i, AA_TO_IDX.get(aa, AA_TO_IDX["X"])] = 1.0
        return np.concatenate([oh, TrainableProteinModel._bio_features(sequence)], axis=-1)

    @staticmethod
    def _softmax(x: anp.ndarray, axis: int = -1) -> anp.ndarray:
        x = x - anp.max(x, axis=axis, keepdims=True)
        e = anp.exp(x)
        return e / anp.clip(anp.sum(e, axis=axis, keepdims=True), 1e-9, None)

    def _pairwise(self, coords: anp.ndarray) -> Tuple[anp.ndarray, anp.ndarray, anp.ndarray]:
        rel = coords[:, None, :] - coords[None, :, :]
        dist = anp.sqrt(anp.sum(rel * rel, axis=-1) + 1e-8)
        unit = rel / anp.expand_dims(dist + 1e-8, -1)
        return rel, dist, unit

    def _graph_mask(self, coords: anp.ndarray) -> anp.ndarray:
        _, dist, _ = self._pairwise(coords)
        n = dist.shape[0]
        if self.config.use_radius_graph:
            m = dist < self.config.radius_cutoff
            m = m * (1 - anp.eye(n))
            return m.astype(anp.float64)

        # kNN graph
        idx = np.argsort(np.asarray(dist), axis=1)[:, 1:self.config.knn_k + 1]
        m = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            m[i, idx[i]] = 1.0
        return anp.asarray(np.maximum(m, m.T))

    def _spherical_harmonics_feat(self, unit: anp.ndarray) -> anp.ndarray:
        x, y, z = unit[..., 0], unit[..., 1], unit[..., 2]
        y00 = anp.ones_like(x)
        # l=1 real
        y1m = anp.stack([x, y, z], axis=-1)
        # l=2 real (compact basis)
        y2 = anp.stack([x * y, y * z, z * x, x * x - y * y, 3 * z * z - 1], axis=-1)
        return anp.concatenate([anp.expand_dims(y00, -1), y1m, y2], axis=-1)  # [...,9]

    def _triangle_update(self, pair: anp.ndarray) -> anp.ndarray:
        # multiplicative-ish
        h = anp.tanh(anp.einsum("ijd,dk->ijk", pair, self.params["W_tri_mul"]))
        mul = anp.einsum("ikd,kjd->ijd", h, h) / anp.sqrt(pair.shape[-1])

        q = anp.einsum("ijd,dk->ijk", pair, self.params["W_tri_attn_q"])
        k = anp.einsum("ijd,dk->ijk", pair, self.params["W_tri_attn_k"])
        v = anp.einsum("ijd,dk->ijk", pair, self.params["W_tri_attn_v"])
        logits = anp.einsum("ijd,ikd->ijk", q, k) / anp.sqrt(pair.shape[-1])
        attn = self._softmax(logits, axis=-1)
        attn_out = anp.einsum("ijk,ikd->ijd", attn, v)

        out = pair + 0.2 * anp.tanh(mul) + 0.2 * anp.tanh(attn_out)
        return 0.5 * (out + anp.transpose(out, (1, 0, 2)))

    def _long_range_attention(self, node: anp.ndarray, pair: anp.ndarray) -> anp.ndarray:
        q = anp.dot(node, self.params["W_attn_q"])
        k = anp.dot(node, self.params["W_attn_k"])
        v = anp.dot(node, self.params["W_attn_v"])
        logits = anp.dot(q, k.T) / anp.sqrt(node.shape[-1]) + 0.05 * anp.mean(pair, axis=-1)
        a = self._softmax(logits, axis=-1)
        return node + anp.dot(a, v)

    def _message_passing(self, node: anp.ndarray, pair: anp.ndarray, coords: anp.ndarray, graph: anp.ndarray) -> anp.ndarray:
        rel, _, unit = self._pairwise(coords)
        sh = self._spherical_harmonics_feat(unit)
        n = node.shape[0]

        msgs = []
        for i in range(n):
            feat_ij = anp.concatenate([
                anp.repeat(anp.expand_dims(node[i], 0), n, axis=0),
                pair[i],
                sh[i],
            ], axis=-1)
            m_ij = anp.tanh(anp.dot(feat_ij, self.params["W_msg"]) + self.params["b_msg"])
            m_i = anp.sum(m_ij * anp.expand_dims(graph[i], -1), axis=0) / (anp.sum(graph[i]) + 1e-8)
            msgs.append(m_i)
        return node + anp.stack(msgs, axis=0)

    def _equivariant_coord_update(self, node: anp.ndarray, pair: anp.ndarray, coords: anp.ndarray, graph: anp.ndarray) -> anp.ndarray:
        if self._e3nn_ready:
            try:
                return self._equivariant_coord_update_e3nn(node, pair, coords, graph)
            except Exception:
                pass

        rel, dist, unit = self._pairwise(coords)
        n = coords.shape[0]
        dcoords = []
        for i in range(n):
            ni = anp.repeat(anp.expand_dims(node[i], 0), n, axis=0)
            feat = anp.concatenate([ni, pair[i]], axis=-1)
            gate = 1.0 / (1.0 + anp.exp(-anp.squeeze(anp.dot(feat, self.params["W_coord_gate"]), axis=-1)))
            w = gate * graph[i] / (dist[i] + 1e-6)
            dc = anp.sum(anp.expand_dims(w, -1) * unit[i], axis=0)
            dcoords.append(dc)
        new_coords = coords + 0.1 * anp.stack(dcoords, axis=0)
        return new_coords - anp.mean(new_coords, axis=0, keepdims=True)

    def _equivariant_coord_update_e3nn(self, node: anp.ndarray, pair: anp.ndarray, coords: anp.ndarray, graph: anp.ndarray) -> anp.ndarray:
        # True O(3)-equivariant edge updates using e3nn irreps and linear maps.
        c = torch.tensor(np.asarray(coords), dtype=torch.float32)
        n = c.shape[0]
        gi, gj = np.where(np.asarray(graph) > 0.0)
        if len(gi) == 0:
            return coords

        src = torch.tensor(gi, dtype=torch.long)
        dst = torch.tensor(gj, dtype=torch.long)
        edge = c[dst] - c[src]
        r = torch.norm(edge, dim=-1, keepdim=True).clamp_min(1e-6)
        unit = edge / r

        # irreps: 0e (scalar gate) + 1o (vector direction)
        gate_feat = np.concatenate([np.asarray(node)[gi], np.asarray(pair)[gi, gj]], axis=-1)
        gate = torch.sigmoid(torch.tensor(gate_feat, dtype=torch.float32) @ self._e3nn_gate_w)[:, None]
        x = torch.cat([gate, unit], dim=-1)

        vec_msg = self._e3nn_vec(x) / r
        agg = torch.zeros((n, 3), dtype=torch.float32)
        agg.index_add_(0, src, vec_msg)

        new_coords = c + float(self.config.e3nn_step_scale) * torch.tanh(agg)
        new_coords = new_coords - new_coords.mean(dim=0, keepdim=True)
        return anp.asarray(new_coords.detach().cpu().numpy())

    def _forward(self, params: Dict[str, anp.ndarray], sequence: str, recycle_override: Optional[int] = None) -> Tuple[anp.ndarray, anp.ndarray, anp.ndarray, anp.ndarray, anp.ndarray]:
        # bind params for autograd
        old = self.params
        self.params = params

        x = anp.asarray(self._encode(sequence))
        node = anp.tanh(anp.dot(x, self.params["W_in"]) + self.params["b_in"])  # NxD
        coords = anp.concatenate([anp.arange(len(sequence))[:, None], anp.zeros((len(sequence), 2))], axis=1)
        coords = coords - anp.mean(coords, axis=0, keepdims=True)
        pair = anp.repeat(anp.expand_dims(anp.dot(node, self.params["W_pair_init"]), 1), len(sequence), axis=1)
        pair = 0.5 * (pair + anp.transpose(pair, (1, 0, 2)))

        n_recycles = self.config.n_recycles if recycle_override is None else recycle_override
        for _ in range(n_recycles):
            graph = self._graph_mask(coords)
            for _ in range(self.config.n_message_layers):
                node = self._message_passing(node, pair, coords, graph)
                node = self._long_range_attention(node, pair)
                coords = self._equivariant_coord_update(node, pair, coords, graph)
            for _ in range(self.config.n_triangle_layers):
                pair = self._triangle_update(pair)

        tors = anp.dot(node, self.params["W_tors"]) + self.params["b_tors"]
        tors = tors.reshape((len(sequence), 3, self.config.torsion_bins))
        dist_logits = anp.einsum("ijd,dk->ijk", pair, self.params["W_dist"]) + self.params["b_dist"]
        conf = 100.0 / (1.0 + anp.exp(-(anp.dot(node, self.params["W_conf"]) + self.params["b_conf"]))).reshape((len(sequence),))
        pae = anp.squeeze(anp.einsum("ijd,dk->ijk", pair, self.params["W_pae"]) + self.params["b_pae"], axis=-1)
        pae = 0.5 * (pae + pae.T)

        self.params = old
        return coords, tors, dist_logits, conf, pae

    def _fape_like(self, pred: anp.ndarray, true: anp.ndarray) -> anp.ndarray:
        pred_c = pred - anp.mean(pred, axis=0, keepdims=True)
        true_c = true - anp.mean(true, axis=0, keepdims=True)
        return anp.mean(anp.sqrt(anp.sum((pred_c - true_c) ** 2, axis=-1) + 1e-8))

    def _lddt_like(self, pred: anp.ndarray, true: anp.ndarray) -> anp.ndarray:
        pd = anp.sqrt(anp.sum((pred[:, None, :] - pred[None, :, :]) ** 2, axis=-1) + 1e-8)
        td = anp.sqrt(anp.sum((true[:, None, :] - true[None, :, :]) ** 2, axis=-1) + 1e-8)
        diff = anp.abs(pd - td)
        score = (diff < 0.5) + (diff < 1.0) + (diff < 2.0) + (diff < 4.0)
        return 1.0 - anp.mean(score / 4.0)

    def _example_loss(self, params: Dict[str, anp.ndarray], ex: StructureExample) -> anp.ndarray:
        y = anp.asarray(ex.coords)
        coords, tors, dist_logits, conf, pae = self._forward(params, ex.sequence)

        fape = self._fape_like(coords, y)

        td = anp.sqrt(anp.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1) + 1e-8)
        dbin = anp.clip(anp.floor(td / 1.5), 0, self.config.dist_bins - 1).astype(int)
        probs = self._softmax(dist_logits, axis=-1)
        ii = anp.arange(len(y))[:, None]
        jj = anp.arange(len(y))[None, :]
        dist_ce = -anp.mean(anp.log(anp.clip(probs[ii, jj, dbin], 1e-9, 1.0)))

        bond = anp.sqrt(anp.sum((coords[1:] - coords[:-1]) ** 2, axis=-1) + 1e-8)
        bond_loss = anp.mean((bond - 3.8) ** 2)

        # clash penalty
        pd = anp.sqrt(anp.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1) + 1e-8)
        mask = 1 - anp.eye(len(y))
        clash = anp.mean(anp.maximum(0.0, 1.2 - pd) * mask)

        # torsion regularization (proxy)
        tors_loss = 0.001 * anp.mean(tors ** 2)

        lddt = self._lddt_like(coords, y)
        conf_loss = anp.mean((conf - 70.0) ** 2) / 1000.0

        # pairwise error supervision
        pae_target = anp.abs(pd - td)
        pae_loss = anp.mean((pae - pae_target) ** 2) / 25.0

        wd = self.config.weight_decay * sum(anp.mean(v * v) for k, v in params.items() if k.startswith("W"))
        return 1.0 * fape + 0.4 * dist_ce + 0.1 * bond_loss + 0.1 * clash + 0.15 * lddt + tors_loss + 0.05 * conf_loss + 0.1 * pae_loss + wd

    def _batch_loss(self, params: Dict[str, anp.ndarray], batch: Sequence[StructureExample]) -> anp.ndarray:
        return anp.mean(anp.stack([self._example_loss(params, ex) for ex in batch]))

    def fit(self, dataset: RealStructureDataset) -> TrainingResult:
        if not dataset.examples:
            raise ValueError("Dataset is empty.")

        examples = dataset.examples.copy()
        self.rng.shuffle(examples)
        n_val = max(1, int(len(examples) * self.config.val_split)) if len(examples) > 2 else 0
        val_examples = examples[:n_val]
        train_examples = examples[n_val:] if n_val > 0 else examples

        gfun = grad(self._batch_loss)
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}

        train_hist, val_hist, lr_hist = [], [], []
        best_val, best_epoch = float("inf"), 0
        best_params = copy.deepcopy(self.params)
        patience = 0
        t = 0

        for epoch in range(self.config.epochs):
            self.rng.shuffle(train_examples)
            batches = [train_examples[i:i + self.config.batch_size] for i in range(0, len(train_examples), self.config.batch_size)]
            epoch_losses = []

            for batch in batches:
                t += 1
                g = gfun(self.params, batch)
                norm = np.sqrt(sum(np.sum(np.asarray(gk) ** 2) for gk in g.values()))
                scale = min(1.0, self.config.grad_clip_norm / max(norm, 1e-8))
                for k in self.params:
                    grad_k = np.asarray(g[k]) * scale
                    m[k] = 0.9 * m[k] + 0.1 * grad_k
                    v[k] = 0.999 * v[k] + 0.001 * (grad_k ** 2)
                    mhat = m[k] / (1 - 0.9 ** t)
                    vhat = v[k] / (1 - 0.999 ** t)
                    self.params[k] = self.params[k] - self.config.lr * mhat / (np.sqrt(vhat) + 1e-8)
                epoch_losses.append(float(self._batch_loss(self.params, batch)))

            train_loss = float(np.mean(epoch_losses))
            val_loss = float(self._batch_loss(self.params, val_examples)) if val_examples else train_loss
            train_hist.append(train_loss)
            val_hist.append(val_loss)
            lr_hist.append(float(self.config.lr))

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_params = copy.deepcopy(self.params)
                patience = 0
            else:
                patience += 1
            if patience >= self.config.early_stopping_patience:
                break

        self.params = best_params
        return TrainingResult(train_hist, val_hist, best_val, best_epoch, len(train_hist), lr_hist)

    def train(self, dataset: RealStructureDataset) -> List[float]:
        return self.fit(dataset).train_losses

    @staticmethod
    def _kabsch_rmsd(pred: np.ndarray, true: np.ndarray) -> float:
        p = pred - pred.mean(axis=0)
        t = true - true.mean(axis=0)
        h = p.T @ t
        u, _, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = vt.T @ u.T
        return float(np.sqrt(np.mean(np.sum((p @ r - t) ** 2, axis=-1))))

    def evaluate(self, dataset: RealStructureDataset) -> Dict[str, float]:
        losses, rmsd, cp, cr, mae = [], [], [], [], []
        for ex in dataset.examples:
            pred = self.predict(ex.sequence)
            losses.append(float(self._batch_loss(self.params, [ex])))
            rmsd.append(self._kabsch_rmsd(pred.coords, ex.coords))
            pd = np.sqrt(np.sum((pred.coords[:, None, :] - pred.coords[None, :, :]) ** 2, axis=-1) + 1e-8)
            td = np.sqrt(np.sum((ex.coords[:, None, :] - ex.coords[None, :, :]) ** 2, axis=-1) + 1e-8)
            mae.append(float(np.mean(np.abs(pd - td))))
            m = ~np.eye(len(pred.coords), dtype=bool)
            p = (pd < 8.0) & m
            t = (td < 8.0) & m
            cp.append(float((p & t).sum() / max(p.sum(), 1)))
            cr.append(float((p & t).sum() / max(t.sum(), 1)))
        cpm = float(np.mean(cp)) if cp else 0.0
        crm = float(np.mean(cr)) if cr else 0.0
        f1 = (2.0 * cpm * crm / max(cpm + crm, 1e-8)) if (cpm + crm) > 0 else 0.0
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "rmsd": float(np.mean(rmsd)) if rmsd else 0.0,
            "contact_precision": cpm,
            "contact_recall": crm,
            "contact_f1": float(f1),
            "distance_mae": float(np.mean(mae)) if mae else 0.0,
        }

    def evaluate_and_export_images(self, dataset: RealStructureDataset, out_dir: str, export_multiview: bool = False, export_gif: bool = False) -> Dict[str, float]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics = self.evaluate(dataset)
        for ex in dataset.examples:
            pred = self.predict(ex.sequence)
            save_structure_image(pred.coords, str(out / f"{ex.pdb_id}.png"), title=f"Predicted {ex.pdb_id}")
            if export_multiview:
                save_multiview_image(pred.coords, str(out / f"{ex.pdb_id}_multiview.png"), title=f"Predicted {ex.pdb_id}")
            if export_gif:
                save_rotation_gif(pred.coords, str(out / f"{ex.pdb_id}_rotate.gif"), title=f"Predicted {ex.pdb_id}")
        return metrics

    def predict(self, sequence: str) -> Prediction:
        coords, tors, dist, conf, pae = self._forward(self.params, sequence)
        return Prediction(sequence, np.asarray(coords), np.asarray(tors), np.asarray(dist), np.asarray(conf), np.asarray(pae))

    def predict_ensemble(
        self,
        sequence: str,
        n_members: int = 8,
        recycle_jitter: bool = True,
        mc_dropout: Optional[bool] = None,
    ) -> Dict[str, object]:
        # `mc_dropout` kept for API compatibility; this compact model uses recycle jitter
        if mc_dropout is not None:
            recycle_jitter = bool(mc_dropout)
        preds = []
        for _ in range(max(1, n_members)):
            recycles = self.config.n_recycles + (self.rng.integers(-1, 2) if recycle_jitter else 0)
            recycles = max(1, int(recycles))
            coords, tors, dist, conf, pae = self._forward(self.params, sequence, recycle_override=recycles)
            preds.append((np.asarray(coords), np.asarray(conf), np.asarray(pae)))
        c = np.stack([p[0] for p in preds], axis=0)
        conf = np.stack([p[1] for p in preds], axis=0)
        pae = np.stack([p[2] for p in preds], axis=0)
        return {
            "members": len(preds),
            "mean_coords": c.mean(axis=0),
            "coord_var": c.var(axis=0),
            "mean_confidence": conf.mean(axis=0),
            "confidence_var": conf.var(axis=0),
            "mean_pair_error": pae.mean(axis=0),
        }

    def save(self, path: str) -> None:
        out = Path(path)
        if out.suffix.lower() == ".json":
            out.write_text(json.dumps({"config": self.config.__dict__, "params": {k: v.tolist() for k, v in self.params.items()}}))
        else:
            np.savez(path, **self.params)

    def load(self, path: str) -> None:
        src = Path(path)
        if src.suffix.lower() == ".json":
            payload = json.loads(src.read_text())
            payload = payload.get("params", payload)
            self.params = {k: np.asarray(v) for k, v in payload.items()}
        else:
            z = np.load(path)
            self.params = {k: z[k] for k in z.files}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Geometric protein prototype with optional true e3nn SE(3) equivariance")
    p.add_argument("sequence", nargs="?", help="Sequence for inference")
    p.add_argument("--train-pdb-ids", nargs="*", default=None)
    p.add_argument("--train-pdb-files", nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--recycles", type=int, default=4)
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--radius-cutoff", type=float, default=10.0)
    p.add_argument("--use-radius-graph", action="store_true")
    p.add_argument("--no-e3nn", action="store_true", help="Disable e3nn equivariant coordinate updates")
    p.add_argument("--ensemble-size", type=int, default=0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--save-model", default=None)
    p.add_argument("--load-model", default=None)
    p.add_argument("--save-pred-pdb", default=None)
    p.add_argument("--save-pred-image", default=None)
    p.add_argument("--save-pred-multiview", default=None)
    p.add_argument("--save-pred-gif", default=None)
    p.add_argument("--export-analysis-images-dir", default=None)
    p.add_argument("--export-analysis-multiview", action="store_true")
    p.add_argument("--export-analysis-gif", action="store_true")
    return p


def main() -> None:
    args = _parser().parse_args()
    cfg = ModelConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        early_stopping_patience=args.patience,
        n_recycles=args.recycles,
        knn_k=args.knn_k,
        radius_cutoff=args.radius_cutoff,
        use_radius_graph=args.use_radius_graph,
        use_e3nn=(not args.no_e3nn),
    )
    model = TrainableProteinModel(cfg)
    if args.load_model:
        model.load(args.load_model)

    trained = False
    fit_result: Optional[TrainingResult] = None
    eval_metrics: Optional[Dict[str, float]] = None

    for mode in ["files", "ids"]:
        if mode == "files" and args.train_pdb_files:
            ds = RealStructureDataset.from_local_pdbs(args.train_pdb_files, min_len=5)
        elif mode == "ids" and args.train_pdb_ids:
            ds = RealStructureDataset.from_rcsb_ids(args.train_pdb_ids, min_len=5)
        else:
            continue

        if ds.examples:
            fit_result = model.fit(ds)
            if args.export_analysis_images_dir:
                eval_metrics = model.evaluate_and_export_images(
                    ds,
                    args.export_analysis_images_dir,
                    export_multiview=args.export_analysis_multiview,
                    export_gif=args.export_analysis_gif,
                )
            else:
                eval_metrics = model.evaluate(ds)
            trained = True

    if args.save_model and trained:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.save_model)

    if args.sequence:
        pred = model.predict(args.sequence)
        if args.save_pred_pdb:
            Path(args.save_pred_pdb).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_pred_pdb).write_text(pred.to_pdb())
        if args.save_pred_image:
            save_structure_image(pred.coords, args.save_pred_image, title=f"Predicted {pred.sequence}")
        if args.save_pred_multiview:
            save_multiview_image(pred.coords, args.save_pred_multiview, title=f"Predicted {pred.sequence}")
        if args.save_pred_gif:
            save_rotation_gif(pred.coords, args.save_pred_gif, title=f"Predicted {pred.sequence}")

        out = {
            "sequence": pred.sequence,
            "trained": trained,
            "fit": {
                "train_losses": fit_result.train_losses if fit_result else [],
                "val_losses": fit_result.val_losses if fit_result else [],
                "best_val_loss": fit_result.best_val_loss if fit_result else None,
                "best_epoch": fit_result.best_epoch if fit_result else None,
                "epochs_ran": fit_result.epochs_ran if fit_result else 0,
            },
            "eval_metrics": eval_metrics,
            "coords": pred.coords.tolist(),
            "torsion_logits_shape": list(pred.torsion_logits.shape),
            "distogram_logits_shape": list(pred.distogram_logits.shape),
            "confidence": pred.confidence.tolist(),
            "pair_error_shape": list(pred.pair_error.shape),
            "e3nn_enabled": bool(model._e3nn_ready),
            "saved_model": args.save_model if (args.save_model and trained) else None,
            "saved_prediction_pdb": args.save_pred_pdb,
            "saved_prediction_image": args.save_pred_image,
            "saved_prediction_multiview": args.save_pred_multiview,
            "saved_prediction_gif": args.save_pred_gif,
            "analysis_images_dir": args.export_analysis_images_dir,
        }

        if args.ensemble_size > 0:
            ens = model.predict_ensemble(args.sequence, n_members=args.ensemble_size)
            out["ensemble"] = {
                "members": ens["members"],
                "mean_coords": ens["mean_coords"].tolist(),
                "coord_var": ens["coord_var"].tolist(),
                "mean_confidence": ens["mean_confidence"].tolist(),
                "confidence_var": ens["confidence_var"].tolist(),
                "mean_pair_error": ens["mean_pair_error"].tolist(),
            }

        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"Sequence length: {len(pred.sequence)}")
            print(f"Trained: {trained}")
            if fit_result:
                print(f"Epochs ran: {fit_result.epochs_ran}; Best epoch: {fit_result.best_epoch}; Best val: {fit_result.best_val_loss:.4f}")
            if eval_metrics:
                print(f"Eval: RMSD={eval_metrics['rmsd']:.4f}, ContactP={eval_metrics['contact_precision']:.4f}")
            print(f"Mean confidence: {float(np.mean(pred.confidence)):.2f}")


if __name__ == "__main__":
    main()
