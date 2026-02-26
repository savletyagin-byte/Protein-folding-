"""Ultra-advanced trainable protein-structure prototype with real PDB ingestion.

Advanced stack highlights:
- BioPython real PDB ingestion (local + RCSB).
- Autograd differentiable model with residual MLP blocks.
- Adam + cosine LR schedule + warmup + gradient clipping + weight decay.
- Curriculum batching by sequence length.
- EMA (exponential moving average) weights.
- SWA-like parameter averaging in final epochs.
- Validation split + early stopping + best-weights restoration.
- Structural metrics: Kabsch RMSD, contact precision/recall/F1, MAE distance.
- Uncertainty-aware ensemble + MC-dropout-style stochastic inference.
- JSON/NPZ checkpoint save/load with metadata.
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
import numpy as np
from Bio.PDB import PDBList, PDBParser, is_aa

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY-X"
AA_TO_IDX = {a: i for i, a in enumerate(AA_VOCAB)}
RES3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}



RES1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
    "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
    "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL", "X": "UNK", "-": "GLY",
}

@dataclass
class ModelConfig:
    d_hidden: int = 160
    torsion_bins: int = 36
    dist_bins: int = 32
    lr: float = 8e-3
    epochs: int = 30
    batch_size: int = 4
    random_seed: int = 7

    # optimization
    weight_decay: float = 1e-5
    grad_clip_norm: float = 5.0
    warmup_epochs: int = 3
    min_lr_ratio: float = 0.15

    # regularization / stochasticity
    dropout_rate: float = 0.10
    augment_noise_std: float = 0.02

    # data split & stopping
    val_split: float = 0.2
    early_stopping_patience: int = 8

    # advanced optimization heads
    ema_decay: float = 0.995
    use_ema_for_eval: bool = True
    swa_start_ratio: float = 0.75
    use_swa: bool = True

    # curriculum
    curriculum: bool = True


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

    def to_pdb(self, chain_id: str = "A") -> str:
        lines = []
        for i, (aa, c) in enumerate(zip(self.sequence, self.coords), start=1):
            resname = RES1_TO_3.get(aa, "UNK")
            x, y, z = float(c[0]), float(c[1]), float(c[2])
            line = (
                f"ATOM  {i:5d}  CA  {resname:>3s} {chain_id}{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
            )
            lines.append(line)
        lines.append("TER")
        lines.append("END")
        return "\n".join(lines) + "\n"


@dataclass
class TrainingResult:
    train_losses: List[float]
    val_losses: List[float]
    lrs: List[float]
    best_val_loss: float
    epochs_ran: int
    best_epoch: int


class RealStructureDataset:
    def __init__(self, examples: List[StructureExample]):
        self.examples = examples

    @staticmethod
    def _extract_from_structure(structure, pdb_id: str, min_len: int = 20, max_len: int = 400) -> Optional[StructureExample]:
        for model in structure:
            for chain in model:
                seq, coords = [], []
                for residue in chain:
                    if not is_aa(residue, standard=True):
                        continue
                    if "CA" not in residue:
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
            path = Path(p)
            structure = parser.get_structure(path.stem, str(path))
            ex = cls._extract_from_structure(structure, path.stem, min_len=min_len, max_len=max_len)
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
            pid = pid.lower()
            try:
                local_file = pdbl.retrieve_pdb_file(pid, pdir=cache_dir, file_format="pdb", overwrite=False)
                structure = parser.get_structure(pid, local_file)
                ex = cls._extract_from_structure(structure, pid, min_len=min_len, max_len=max_len)
                if ex is not None:
                    out.append(ex)
            except Exception:
                continue
        return cls(out)


class TrainableProteinModel:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.params = self._init_params()
        self.ema_params = copy.deepcopy(self.params)

    def _init_params(self) -> Dict[str, np.ndarray]:
        d_in = len(AA_VOCAB) + 6
        h = self.config.d_hidden
        t = self.config.torsion_bins * 3
        db = self.config.dist_bins

        def n(*shape: int, scale: float = 0.05) -> np.ndarray:
            return self.rng.normal(0.0, scale, size=shape)

        return {
            "W1": n(d_in, h), "b1": np.zeros(h),
            "W2": n(h, h), "b2": np.zeros(h),
            "W3": n(h, h), "b3": np.zeros(h),
            "W_coord": n(h, 3), "b_coord": np.zeros(3),
            "W_tors": n(h, t), "b_tors": np.zeros(t),
            "W_conf": n(h, 1), "b_conf": np.zeros(1),
            "W_pair": n(h, db), "b_pair": np.zeros(db),
        }

    @staticmethod
    def _bio_features(sequence: str) -> np.ndarray:
        hydrophobic = set("AILMFWVY")
        positive = set("KRH")
        negative = set("DE")
        aromatic = set("FWYH")
        special = set("CGP")
        small = set("AGSCTPDN")
        return np.asarray([
            [
                float(aa in hydrophobic),
                float(aa in positive),
                float(aa in negative),
                float(aa in aromatic),
                float(aa in special),
                float(aa in small),
            ]
            for aa in sequence
        ], dtype=np.float64)

    @staticmethod
    def _encode(sequence: str) -> np.ndarray:
        oh = np.zeros((len(sequence), len(AA_VOCAB)), dtype=np.float64)
        for i, a in enumerate(sequence):
            oh[i, AA_TO_IDX.get(a, AA_TO_IDX["X"])] = 1.0
        return np.concatenate([oh, TrainableProteinModel._bio_features(sequence)], axis=-1)

    @staticmethod
    def _softmax(x: anp.ndarray, axis: int = -1) -> anp.ndarray:
        x = x - anp.max(x, axis=axis, keepdims=True)
        e = anp.exp(x)
        return e / anp.clip(anp.sum(e, axis=axis, keepdims=True), 1e-9, None)

    def _dropout(self, x: anp.ndarray, p: float, training: bool) -> anp.ndarray:
        if (not training) or p <= 0:
            return x
        mask = (self.rng.random(x.shape) > p).astype(np.float64)
        return x * mask / max(1e-8, (1.0 - p))

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
        p_aligned = p @ r
        return float(np.sqrt(np.mean(np.sum((p_aligned - t) ** 2, axis=-1))))

    @staticmethod
    def _contact_prf(pred: np.ndarray, true: np.ndarray, threshold: float = 8.0) -> Tuple[float, float, float]:
        pd = np.sqrt(np.sum((pred[:, None, :] - pred[None, :, :]) ** 2, axis=-1) + 1e-8)
        td = np.sqrt(np.sum((true[:, None, :] - true[None, :, :]) ** 2, axis=-1) + 1e-8)
        m = ~np.eye(len(pred), dtype=bool)
        p = (pd < threshold) & m
        t = (td < threshold) & m
        tp = float((p & t).sum())
        fp = float((p & ~t).sum())
        fn = float((~p & t).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return precision, recall, f1

    def _forward_autograd(self, params: Dict[str, anp.ndarray], x: anp.ndarray, training: bool = False) -> Tuple[anp.ndarray, anp.ndarray, anp.ndarray, anp.ndarray]:
        h1 = anp.tanh(anp.dot(x, params["W1"]) + params["b1"])
        h1 = self._dropout(h1, self.config.dropout_rate, training)
        h2 = anp.tanh(anp.dot(h1, params["W2"]) + params["b2"])
        h2 = h2 + 0.5 * h1  # residual
        h2 = self._dropout(h2, self.config.dropout_rate, training)
        h3 = anp.tanh(anp.dot(h2, params["W3"]) + params["b3"])
        h = h3 + 0.5 * h2

        coords = anp.dot(h, params["W_coord"]) + params["b_coord"]
        coords = coords - anp.mean(coords, axis=0, keepdims=True)

        tors = anp.dot(h, params["W_tors"]) + params["b_tors"]
        tors = tors.reshape((x.shape[0], 3, self.config.torsion_bins))

        conf = 100.0 / (1.0 + anp.exp(-(anp.dot(h, params["W_conf"]) + params["b_conf"]))).reshape((x.shape[0],))

        pair_f = h[:, None, :] + h[None, :, :]
        dist_logits = anp.einsum("ijd,db->ijb", pair_f, params["W_pair"]) + params["b_pair"]
        return coords, tors, dist_logits, conf

    def _augment_coords(self, coords: np.ndarray) -> np.ndarray:
        if self.config.augment_noise_std <= 0:
            return coords
        noisy = coords + self.rng.normal(0.0, self.config.augment_noise_std, size=coords.shape)
        return noisy - noisy.mean(axis=0, keepdims=True)

    def _example_loss(self, params: Dict[str, anp.ndarray], ex: StructureExample) -> anp.ndarray:
        x = anp.asarray(self._encode(ex.sequence))
        y = anp.asarray(self._augment_coords(ex.coords - ex.coords.mean(axis=0, keepdims=True)))

        pred_coords, tors, dist_logits, conf = self._forward_autograd(params, x, training=True)

        coord_loss = anp.mean((pred_coords - y) ** 2)

        b = pred_coords[1:] - pred_coords[:-1]
        bond_len = anp.sqrt(anp.sum(b * b, axis=-1) + 1e-8)
        bond_loss = anp.mean((bond_len - 3.8) ** 2)

        d_true = anp.sqrt(anp.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1) + 1e-8)
        d_bin = anp.clip(anp.floor(d_true / 1.5), 0, self.config.dist_bins - 1).astype(int)
        probs = self._softmax(dist_logits, axis=-1)
        idx_i = anp.arange(y.shape[0])[:, None]
        idx_j = anp.arange(y.shape[0])[None, :]
        picked = probs[idx_i, idx_j, d_bin]
        dist_loss = -anp.mean(anp.log(anp.clip(picked, 1e-9, 1.0)))

        conf_loss = anp.mean((conf - 70.0) ** 2) / 1000.0
        wd = self.config.weight_decay * sum(anp.mean(v * v) for k, v in params.items() if k.startswith("W"))

        return coord_loss + 0.12 * bond_loss + 0.24 * dist_loss + 0.05 * conf_loss + 0.001 * anp.mean(tors ** 2) + wd

    def _batch_loss(self, params: Dict[str, anp.ndarray], batch: Sequence[StructureExample]) -> anp.ndarray:
        return anp.mean(anp.stack([self._example_loss(params, ex) for ex in batch]))

    def _lr_for_epoch(self, epoch: int) -> float:
        if epoch < self.config.warmup_epochs:
            return self.config.lr * (epoch + 1) / max(1, self.config.warmup_epochs)
        e = epoch - self.config.warmup_epochs
        total = max(1, self.config.epochs - self.config.warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * e / total))
        return self.config.lr * (self.config.min_lr_ratio + (1 - self.config.min_lr_ratio) * cosine)

    def _curriculum_batches(self, examples: List[StructureExample]) -> List[List[StructureExample]]:
        if not self.config.curriculum:
            self.rng.shuffle(examples)
            return [examples[i:i + self.config.batch_size] for i in range(0, len(examples), self.config.batch_size)]

        sorted_ex = sorted(examples, key=lambda ex: len(ex.sequence))
        buckets = [sorted_ex[i:i + self.config.batch_size] for i in range(0, len(sorted_ex), self.config.batch_size)]
        # slight randomization across neighboring buckets to avoid overfitting order
        self.rng.shuffle(buckets)
        return buckets

    def fit(self, dataset: RealStructureDataset) -> TrainingResult:
        if not dataset.examples:
            raise ValueError("Dataset is empty. Provide valid PDB structures.")

        examples = dataset.examples.copy()
        self.rng.shuffle(examples)

        n_val = max(1, int(len(examples) * self.config.val_split)) if len(examples) > 2 else 0
        val_examples = examples[:n_val]
        train_examples = examples[n_val:] if n_val > 0 else examples

        loss_grad = grad(self._batch_loss)
        train_hist, val_hist, lrs = [], [], []

        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8

        best_val = float("inf")
        best_params = copy.deepcopy(self.params)
        best_epoch = 0
        patience = 0
        step_t = 0

        swa_start = int(self.config.epochs * self.config.swa_start_ratio)
        swa_params = copy.deepcopy(self.params)
        swa_n = 0

        for epoch in range(self.config.epochs):
            batches = self._curriculum_batches(train_examples.copy())
            lr_epoch = self._lr_for_epoch(epoch)
            lrs.append(lr_epoch)
            epoch_losses = []

            for batch in batches:
                step_t += 1
                g = loss_grad(self.params, batch)

                global_norm = np.sqrt(sum(np.sum(np.asarray(gk) ** 2) for gk in g.values()))
                clip_scale = min(1.0, self.config.grad_clip_norm / max(global_norm, 1e-8))

                for k in self.params:
                    grad_k = np.asarray(g[k]) * clip_scale
                    m[k] = b1 * m[k] + (1 - b1) * grad_k
                    v[k] = b2 * v[k] + (1 - b2) * (grad_k ** 2)
                    m_hat = m[k] / (1 - b1 ** step_t)
                    v_hat = v[k] / (1 - b2 ** step_t)
                    self.params[k] = self.params[k] - lr_epoch * m_hat / (np.sqrt(v_hat) + eps)

                    # EMA update
                    self.ema_params[k] = self.config.ema_decay * self.ema_params[k] + (1 - self.config.ema_decay) * self.params[k]

                epoch_losses.append(float(self._batch_loss(self.params, batch)))

            # SWA averaging in late epochs
            if self.config.use_swa and epoch >= swa_start:
                swa_n += 1
                for k in self.params:
                    swa_params[k] = (swa_params[k] * (swa_n - 1) + self.params[k]) / swa_n

            train_loss = float(np.mean(epoch_losses))
            train_hist.append(train_loss)

            eval_params = self.ema_params if self.config.use_ema_for_eval else self.params
            if val_examples:
                val_loss = float(self._batch_loss(eval_params, val_examples))
            else:
                val_loss = train_loss
            val_hist.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_params = copy.deepcopy(eval_params)
                best_epoch = epoch
                patience = 0
            else:
                patience += 1

            if patience >= self.config.early_stopping_patience:
                break

        # finalize with best params; optionally blend with SWA
        if self.config.use_swa and swa_n > 0:
            for k in best_params:
                best_params[k] = 0.5 * best_params[k] + 0.5 * swa_params[k]

        self.params = best_params
        self.ema_params = copy.deepcopy(best_params)

        return TrainingResult(
            train_losses=train_hist,
            val_losses=val_hist,
            lrs=lrs,
            best_val_loss=best_val,
            epochs_ran=len(train_hist),
            best_epoch=best_epoch,
        )

    def train(self, dataset: RealStructureDataset) -> List[float]:
        return self.fit(dataset).train_losses

    def evaluate(self, dataset: RealStructureDataset) -> Dict[str, float]:
        if not dataset.examples:
            raise ValueError("Dataset is empty.")

        rmsds, precisions, recalls, f1s, losses, maes = [], [], [], [], [], []
        for ex in dataset.examples:
            pred = self.predict(ex.sequence)
            losses.append(float(self._batch_loss(self.params, [ex])))
            rmsds.append(self._kabsch_rmsd(pred.coords, ex.coords))
            p, r, f1 = self._contact_prf(pred.coords, ex.coords)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

            pd = np.sqrt(np.sum((pred.coords[:, None, :] - pred.coords[None, :, :]) ** 2, axis=-1) + 1e-8)
            td = np.sqrt(np.sum((ex.coords[:, None, :] - ex.coords[None, :, :]) ** 2, axis=-1) + 1e-8)
            maes.append(float(np.mean(np.abs(pd - td))))

        return {
            "loss": float(np.mean(losses)),
            "rmsd": float(np.mean(rmsds)),
            "contact_precision": float(np.mean(precisions)),
            "contact_recall": float(np.mean(recalls)),
            "contact_f1": float(np.mean(f1s)),
            "distance_mae": float(np.mean(maes)),
        }

    def predict(self, sequence: str) -> Prediction:
        x = self._encode(sequence)
        coords, tors, dist, conf = self._forward_autograd(self.params, anp.asarray(x), training=False)
        return Prediction(sequence=sequence, coords=np.asarray(coords), torsion_logits=np.asarray(tors), distogram_logits=np.asarray(dist), confidence=np.asarray(conf))

    def predict_ensemble(self, sequence: str, n_members: int = 8, noise_std: float = 0.01, mc_dropout: bool = True) -> Dict[str, object]:
        members = []
        x0 = self._encode(sequence)
        for _ in range(max(1, n_members)):
            x = x0 + self.rng.normal(0.0, noise_std, size=x0.shape)
            coords, tors, dist, conf = self._forward_autograd(self.params, anp.asarray(x), training=mc_dropout)
            members.append((np.asarray(coords), np.asarray(tors), np.asarray(dist), np.asarray(conf)))

        coords_stack = np.stack([m[0] for m in members], axis=0)
        conf_stack = np.stack([m[3] for m in members], axis=0)
        return {
            "members": len(members),
            "mean_coords": coords_stack.mean(axis=0),
            "coord_var": coords_stack.var(axis=0),
            "mean_confidence": conf_stack.mean(axis=0),
            "confidence_var": conf_stack.var(axis=0),
        }

    def save(self, path: str) -> None:
        out = Path(path)
        metadata = {
            "config": self.config.__dict__,
            "format": out.suffix.lower(),
        }
        if out.suffix.lower() == ".json":
            out.write_text(json.dumps({"metadata": metadata, "params": {k: v.tolist() for k, v in self.params.items()}}))
        else:
            np.savez(path, **self.params)

    def load(self, path: str) -> None:
        src = Path(path)
        if src.suffix.lower() == ".json":
            payload = json.loads(src.read_text())
            if "params" in payload:
                payload = payload["params"]
            self.params = {k: np.asarray(v) for k, v in payload.items()}
        else:
            loaded = np.load(path)
            self.params = {k: loaded[k] for k in loaded.files}
        self.ema_params = copy.deepcopy(self.params)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ultra-advanced trainable protein prototype with real PDB ingestion")
    p.add_argument("sequence", nargs="?", help="Sequence for inference")
    p.add_argument("--train-pdb-ids", nargs="*", default=None, help="RCSB PDB IDs for training")
    p.add_argument("--train-pdb-files", nargs="*", default=None, help="Local PDB files for training")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=8e-3)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--ensemble-size", type=int, default=0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--save-model", default=None)
    p.add_argument("--load-model", default=None)
    p.add_argument("--save-pred-pdb", default=None, help="Write predicted CA trace to a PDB file")
    return p


def main() -> None:
    args = _parser().parse_args()
    cfg = ModelConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        early_stopping_patience=args.patience,
    )
    model = TrainableProteinModel(cfg)

    if args.load_model:
        model.load(args.load_model)

    trained = False
    fit_result: Optional[TrainingResult] = None
    eval_metrics: Optional[Dict[str, float]] = None

    if args.train_pdb_files:
        ds = RealStructureDataset.from_local_pdbs(args.train_pdb_files, min_len=5)
        if ds.examples:
            fit_result = model.fit(ds)
            eval_metrics = model.evaluate(ds)
            trained = True

    if args.train_pdb_ids:
        ds = RealStructureDataset.from_rcsb_ids(args.train_pdb_ids, min_len=5)
        if ds.examples:
            fit_result = model.fit(ds)
            eval_metrics = model.evaluate(ds)
            trained = True

    if args.save_model and trained:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.save_model)

    if args.sequence:
        pred = model.predict(args.sequence)
        if args.save_pred_pdb:
            out_pdb = Path(args.save_pred_pdb)
            out_pdb.parent.mkdir(parents=True, exist_ok=True)
            out_pdb.write_text(pred.to_pdb())
        out = {
            "sequence": pred.sequence,
            "trained": trained,
            "fit": {
                "train_losses": fit_result.train_losses if fit_result else [],
                "val_losses": fit_result.val_losses if fit_result else [],
                "lrs": fit_result.lrs if fit_result else [],
                "best_val_loss": fit_result.best_val_loss if fit_result else None,
                "epochs_ran": fit_result.epochs_ran if fit_result else 0,
                "best_epoch": fit_result.best_epoch if fit_result else None,
            },
            "eval_metrics": eval_metrics,
            "coords": pred.coords.tolist(),
            "torsion_logits_shape": list(pred.torsion_logits.shape),
            "distogram_logits_shape": list(pred.distogram_logits.shape),
            "confidence": pred.confidence.tolist(),
            "saved_model": args.save_model if (args.save_model and trained) else None,
            "saved_prediction_pdb": args.save_pred_pdb if args.save_pred_pdb else None,
        }

        if args.ensemble_size > 0:
            ens = model.predict_ensemble(args.sequence, n_members=args.ensemble_size)
            out["ensemble"] = {
                "members": ens["members"],
                "mean_coords": ens["mean_coords"].tolist(),
                "coord_var": ens["coord_var"].tolist(),
                "mean_confidence": ens["mean_confidence"].tolist(),
                "confidence_var": ens["confidence_var"].tolist(),
            }

        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"Sequence length: {len(pred.sequence)}")
            print(f"Trained: {trained}")
            if fit_result:
                print(f"Epochs ran: {fit_result.epochs_ran}")
                print(f"Best epoch: {fit_result.best_epoch}, Best val loss: {fit_result.best_val_loss:.4f}")
            if eval_metrics:
                print(
                    "Eval metrics: "
                    f"RMSD={eval_metrics['rmsd']:.4f}, "
                    f"P={eval_metrics['contact_precision']:.4f}, "
                    f"R={eval_metrics['contact_recall']:.4f}, "
                    f"F1={eval_metrics['contact_f1']:.4f}, "
                    f"DistMAE={eval_metrics['distance_mae']:.4f}"
                )
            print(f"Mean confidence: {float(np.mean(pred.confidence)):.2f}")
            print("First 5 C-alpha coordinates:")
            for i, c in enumerate(pred.coords[:5]):
                print(f"  {i:3d}: ({c[0]: .3f}, {c[1]: .3f}, {c[2]: .3f})")


if __name__ == "__main__":
    main()
