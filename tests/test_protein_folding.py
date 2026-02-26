from pathlib import Path

import numpy as np

from protein_folding import RealStructureDataset, TrainableProteinModel, ModelConfig


PDB_TEXT = """\
ATOM      1  CA  ALA A   1      11.104  13.207   8.630  1.00 20.00           C
ATOM      2  CA  GLY A   2      12.600  13.500   9.050  1.00 20.00           C
ATOM      3  CA  SER A   3      13.900  12.800   9.800  1.00 20.00           C
ATOM      4  CA  LEU A   4      15.300  13.300  10.200  1.00 20.00           C
ATOM      5  CA  TYR A   5      16.700  12.600  10.850  1.00 20.00           C
ATOM      6  CA  VAL A   6      18.000  13.100  11.300  1.00 20.00           C
TER
END
"""


def test_local_pdb_dataset_parsing(tmp_path: Path):
    pdb_file = tmp_path / "mini.pdb"
    pdb_file.write_text(PDB_TEXT)

    ds = RealStructureDataset.from_local_pdbs([str(pdb_file)], min_len=5)
    assert len(ds.examples) == 1
    ex = ds.examples[0]
    assert ex.sequence == "AGSLYV"
    assert ex.coords.shape == (6, 3)


def test_fit_evaluate_and_metrics(tmp_path: Path):
    pdb_file = tmp_path / "mini.pdb"
    pdb_file.write_text(PDB_TEXT)

    ds = RealStructureDataset.from_local_pdbs([str(pdb_file)], min_len=5)
    model = TrainableProteinModel(
        ModelConfig(
            epochs=4,
            batch_size=1,
            lr=5e-3,
            d_hidden=32,
            torsion_bins=16,
            dist_bins=16,
            val_split=0.5,
            early_stopping_patience=2,
            dropout_rate=0.05,
            curriculum=True,
            use_swa=True,
        )
    )
    result = model.fit(ds)

    assert result.epochs_ran >= 1
    assert np.isfinite(result.best_val_loss)
    assert len(result.lrs) == result.epochs_ran

    metrics = model.evaluate(ds)
    assert set(metrics.keys()) == {"loss", "rmsd", "contact_precision", "contact_recall", "contact_f1", "distance_mae"}
    assert np.isfinite(metrics["loss"])


def test_predict_shapes_and_ensemble_outputs():
    model = TrainableProteinModel(ModelConfig(d_hidden=32, torsion_bins=18, dist_bins=20))
    pred = model.predict("ACDEFG")

    assert pred.coords.shape == (6, 3)
    assert pred.torsion_logits.shape == (6, 3, 18)
    assert pred.distogram_logits.shape == (6, 6, 20)
    assert pred.confidence.shape == (6,)
    assert np.all((pred.confidence >= 0) & (pred.confidence <= 100))

    ens = model.predict_ensemble("ACDEFG", n_members=3, mc_dropout=True)
    assert ens["mean_coords"].shape == (6, 3)
    assert ens["coord_var"].shape == (6, 3)
    assert ens["mean_confidence"].shape == (6,)
    assert ens["confidence_var"].shape == (6,)
    assert ens["members"] == 3


def test_save_and_load_roundtrip(tmp_path: Path):
    model = TrainableProteinModel(ModelConfig(d_hidden=16, torsion_bins=12, dist_bins=12))
    out_path = tmp_path / "weights.json"
    model.save(str(out_path))

    model2 = TrainableProteinModel(ModelConfig(d_hidden=16, torsion_bins=12, dist_bins=12))
    model2.load(str(out_path))

    p1 = model.predict("ACDEF")
    p2 = model2.predict("ACDEF")
    assert np.allclose(p1.coords, p2.coords)
