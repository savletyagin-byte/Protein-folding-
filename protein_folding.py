"""Trainable protein-structure prototype with real PDB data support.

Key upgrades:
- Uses BioPython for real PDB structure ingestion (not NumPy-only).
- Uses Autograd to train model parameters with gradient descent.
- Includes dataset builder from RCSB PDB IDs and local PDB files.
- Predicts C-alpha coordinates, torsion logits, distogram logits, and confidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import autograd.numpy as anp
from autograd import grad
import numpy as np
from Bio.PDB import PDBList, PDBParser, is_aa

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY-X"
AA_TO_IDX = {a: i for i, a in enumerate(AA_VOCAB)}
RES3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass
class ModelConfig:
    d_hidden: int = 128
    torsion_bins: int = 36
    dist_bins: int = 32
    lr: float = 1e-2
    epochs: int = 25
    batch_size: int = 4
    random_seed: int = 7


@dataclass
class StructureExample:
    pdb_id: str
    sequence: str
    coords: np.ndarray  # (L,3) C-alpha


@dataclass
class Prediction:
    sequence: str
    coords: np.ndarray
    torsion_logits: np.ndarray
    distogram_logits: np.ndarray
    confidence: np.ndarray


class RealStructureDataset:
    """Build training examples from local or downloaded PDB structures."""

    def __init__(self, examples: List[StructureExample]):
        self.examples = examples

    @staticmethod
    def _extract_from_structure(structure, pdb_id: str, min_len: int = 20, max_len: int = 400) -> Optional[StructureExample]:
        for model in structure:
            for chain in model:
                seq = []
                coords = []
                for residue in chain:
                    if not is_aa(residue, standard=True):
                        continue
                    res3 = residue.get_resname().upper()
                    aa = RES3_TO_1.get(res3, "X")
                    if "CA" not in residue:
                        continue
                    seq.append(aa)
                    coords.append(np.asarray(residue["CA"].get_coord(), dtype=np.float64))
                if min_len <= len(seq) <= max_len:
                    return StructureExample(pdb_id=pdb_id, sequence="".join(seq), coords=np.stack(coords, axis=0))
        return None

    @classmethod
    def from_local_pdbs(cls, pdb_paths: Sequence[str], min_len: int = 20, max_len: int = 400) -> "RealStructureDataset":
        parser = PDBParser(QUIET=True)
        examples: List[StructureExample] = []
        for p in pdb_paths:
            path = Path(p)
            structure = parser.get_structure(path.stem, str(path))
            ex = cls._extract_from_structure(structure, path.stem, min_len=min_len, max_len=max_len)
            if ex is not None:
                examples.append(ex)
        return cls(examples)

    @classmethod
    def from_rcsb_ids(cls, pdb_ids: Sequence[str], cache_dir: str = "pdb_cache", min_len: int = 20, max_len: int = 400) -> "RealStructureDataset":
        os.makedirs(cache_dir, exist_ok=True)
        pdbl = PDBList(verbose=False)
        parser = PDBParser(QUIET=True)
        examples: List[StructureExample] = []

        for pid in pdb_ids:
            pid = pid.lower()
            try:
                local_file = pdbl.retrieve_pdb_file(pid, pdir=cache_dir, file_format="pdb", overwrite=False)
                structure = parser.get_structure(pid, local_file)
                ex = cls._extract_from_structure(structure, pid, min_len=min_len, max_len=max_len)
                if ex is not None:
                    examples.append(ex)
            except Exception:
                continue

        return cls(examples)


class TrainableProteinModel:
    """Autograd-trained baseline for coordinate regression + geometric heads."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.params = self._init_params()

    def _init_params(self) -> Dict[str, np.ndarray]:
        d_in = len(AA_VOCAB) + 6  # one-hot + biochemical
        h = self.config.d_hidden
        t = self.config.torsion_bins * 3
        db = self.config.dist_bins

        def n(*shape, scale=0.05):
            return self.rng.normal(0.0, scale, size=shape)

        return {
            "W1": n(d_in, h),
            "b1": np.zeros(h),
            "W2": n(h, h),
            "b2": np.zeros(h),
            "W_coord": n(h, 3),
            "b_coord": np.zeros(3),
            "W_tors": n(h, t),
            "b_tors": np.zeros(t),
            "W_conf": n(h, 1),
            "b_conf": np.zeros(1),
            "W_pair": n(h, db),
            "b_pair": np.zeros(db),
        }

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
                1.0 if aa in hydrophobic else 0.0,
                1.0 if aa in positive else 0.0,
                1.0 if aa in negative else 0.0,
                1.0 if aa in aromatic else 0.0,
                1.0 if aa in special else 0.0,
                1.0 if aa in small else 0.0,
            ])
        return np.asarray(out, dtype=np.float64)

    @staticmethod
    def _encode(sequence: str) -> np.ndarray:
        oh = np.zeros((len(sequence), len(AA_VOCAB)), dtype=np.float64)
        for i, a in enumerate(sequence):
            oh[i, AA_TO_IDX.get(a, AA_TO_IDX["X"])] = 1.0
        bio = TrainableProteinModel._bio_features(sequence)
        return np.concatenate([oh, bio], axis=-1)

    @staticmethod
    def _softmax(x: anp.ndarray, axis: int = -1) -> anp.ndarray:
        x = x - anp.max(x, axis=axis, keepdims=True)
        e = anp.exp(x)
        return e / anp.clip(anp.sum(e, axis=axis, keepdims=True), 1e-9, None)

    def _forward_autograd(self, params: Dict[str, anp.ndarray], x: anp.ndarray) -> Tuple[anp.ndarray, anp.ndarray, anp.ndarray, anp.ndarray]:
        h1 = anp.tanh(anp.dot(x, params["W1"]) + params["b1"])
        h2 = anp.tanh(anp.dot(h1, params["W2"]) + params["b2"])

        coords = anp.dot(h2, params["W_coord"]) + params["b_coord"]
        coords = coords - anp.mean(coords, axis=0, keepdims=True)

        tors = anp.dot(h2, params["W_tors"]) + params["b_tors"]
        tors = tors.reshape((x.shape[0], 3, self.config.torsion_bins))

        conf = 100.0 / (1.0 + anp.exp(-(anp.dot(h2, params["W_conf"]) + params["b_conf"]))).reshape((x.shape[0],))

        pair_f = h2[:, None, :] + h2[None, :, :]
        dist_logits = anp.einsum("ijd,db->ijb", pair_f, params["W_pair"]) + params["b_pair"]
        return coords, tors, dist_logits, conf

    def _example_loss(self, params: Dict[str, anp.ndarray], ex: StructureExample) -> anp.ndarray:
        x = anp.asarray(self._encode(ex.sequence))
        y = anp.asarray(ex.coords)

        pred_coords, tors, dist_logits, conf = self._forward_autograd(params, x)

        # coordinate loss
        coord_loss = anp.mean((pred_coords - (y - anp.mean(y, axis=0, keepdims=True))) ** 2)

        # smooth backbone prior
        b = pred_coords[1:] - pred_coords[:-1]
        bond_len = anp.sqrt(anp.sum(b * b, axis=-1) + 1e-8)
        bond_loss = anp.mean((bond_len - 3.8) ** 2)

        # distogram supervision from true distances
        d_true = anp.sqrt(anp.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1) + 1e-8)
        d_bin = anp.clip(anp.floor(d_true / 1.5), 0, self.config.dist_bins - 1).astype(int)
        probs = self._softmax(dist_logits, axis=-1)
        idx_i = anp.arange(y.shape[0])[:, None]
        idx_j = anp.arange(y.shape[0])[None, :]
        picked = probs[idx_i, idx_j, d_bin]
        dist_loss = -anp.mean(anp.log(anp.clip(picked, 1e-9, 1.0)))

        # confidence regularization
        conf_loss = anp.mean((conf - 70.0) ** 2) / 1000.0

        return coord_loss + 0.1 * bond_loss + 0.2 * dist_loss + 0.05 * conf_loss + 0.001 * anp.mean(tors ** 2)

    def _batch_loss(self, params: Dict[str, anp.ndarray], batch: Sequence[StructureExample]) -> anp.ndarray:
        losses = [self._example_loss(params, ex) for ex in batch]
        return anp.mean(anp.stack(losses))

    def train(self, dataset: RealStructureDataset) -> List[float]:
        if not dataset.examples:
            raise ValueError("Dataset is empty. Provide valid PDB structures.")

        loss_grad = grad(self._batch_loss)
        history: List[float] = []

        for epoch in range(self.config.epochs):
            self.rng.shuffle(dataset.examples)
            batches = [
                dataset.examples[i: i + self.config.batch_size]
                for i in range(0, len(dataset.examples), self.config.batch_size)
            ]

            epoch_losses = []
            for batch in batches:
                g = loss_grad(self.params, batch)
                for k in self.params:
                    self.params[k] = self.params[k] - self.config.lr * np.asarray(g[k])
                epoch_losses.append(float(self._batch_loss(self.params, batch)))

            history.append(float(np.mean(epoch_losses)))
        return history

    def predict(self, sequence: str) -> Prediction:
        x = self._encode(sequence)
        coords, tors, dist, conf = self._forward_autograd(self.params, anp.asarray(x))
        return Prediction(
            sequence=sequence,
            coords=np.asarray(coords),
            torsion_logits=np.asarray(tors),
            distogram_logits=np.asarray(dist),
            confidence=np.asarray(conf),
        )

    def save(self, path: str) -> None:
        out = Path(path)
        if out.suffix.lower() == ".json":
            payload = {k: v.tolist() for k, v in self.params.items()}
            out.write_text(json.dumps(payload))
        else:
            np.savez(path, **self.params)

    def load(self, path: str) -> None:
        src = Path(path)
        if src.suffix.lower() == ".json":
            payload = json.loads(src.read_text())
            self.params = {k: np.asarray(v) for k, v in payload.items()}
        else:
            loaded = np.load(path)
            self.params = {k: loaded[k] for k in loaded.files}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trainable protein prototype with real PDB ingestion")
    p.add_argument("sequence", nargs="?", help="Sequence for inference")
    p.add_argument("--train-pdb-ids", nargs="*", default=None, help="RCSB PDB IDs for training")
    p.add_argument("--train-pdb-files", nargs="*", default=None, help="Local PDB files for training")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--json", action="store_true")
    p.add_argument("--save-model", default=None, help="Path to save trained weights (.json or .npz)")
    p.add_argument("--load-model", default=None, help="Path to load weights (.json or .npz)")
    return p


def main() -> None:
    args = _parser().parse_args()
    cfg = ModelConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    model = TrainableProteinModel(cfg)
    if args.load_model:
        model.load(args.load_model)

    trained = False
    history = []

    if args.train_pdb_files:
        ds = RealStructureDataset.from_local_pdbs(args.train_pdb_files, min_len=5)
        if ds.examples:
            history = model.train(ds)
            trained = True

    if args.train_pdb_ids:
        ds = RealStructureDataset.from_rcsb_ids(args.train_pdb_ids, min_len=5)
        if ds.examples:
            history = model.train(ds)
            trained = True

    if args.save_model and trained:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.save_model)

    if args.sequence:
        pred = model.predict(args.sequence)
        out = {
            "sequence": pred.sequence,
            "trained": trained,
            "train_loss_history": history,
            "coords": pred.coords.tolist(),
            "torsion_logits_shape": list(pred.torsion_logits.shape),
            "distogram_logits_shape": list(pred.distogram_logits.shape),
            "confidence": pred.confidence.tolist(),
            "saved_model": args.save_model if (args.save_model and trained) else None,
        }
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"Sequence length: {len(pred.sequence)}")
            print(f"Trained: {trained}")
            if history:
                print(f"Final train loss: {history[-1]:.4f}")
            print(f"Mean confidence: {float(np.mean(pred.confidence)):.2f}")
            print("First 5 C-alpha coordinates:")
            for i, c in enumerate(pred.coords[:5]):
                print(f"  {i:3d}: ({c[0]: .3f}, {c[1]: .3f}, {c[2]: .3f})")


if __name__ == "__main__":
    main()
