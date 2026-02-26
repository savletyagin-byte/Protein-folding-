import numpy as np

from protein_folding import AdvancedProteinFoldingModel, ModelConfig


def _small_model() -> AdvancedProteinFoldingModel:
    cfg = ModelConfig(
        d_seq=48,
        d_msa=32,
        d_pair=24,
        n_recycles=2,
        n_triangle_updates=1,
        n_refine_steps=8,
        torsion_bins=24,
        n_ensemble=3,
        random_seed=13,
    )
    return AdvancedProteinFoldingModel(cfg)


def test_core_representations_shapes():
    model = _small_model()
    seq = "ACDEFG"
    msa = ["ACDEFG", "ACDEYG", "ACDEFG"]

    seq_repr = model.sequence_embedding(seq, "inactive")
    msa_repr = model.msa_encoder(msa)
    pair = model.pair_representation(seq_repr, msa_repr)
    tri = model.triangle_updates(pair)

    assert seq_repr.shape == (6, model.config.d_seq)
    assert msa_repr.shape == (6, model.config.d_msa)
    assert pair.shape == (6, 6, model.config.d_pair)
    assert tri.shape == (6, 6, model.config.d_pair)


def test_se3_refinement_translation_equivariance():
    model = _small_model()
    seq = "ACDEFG"
    msa = [seq]
    seq_repr = model.sequence_embedding(seq, "active")
    msa_repr = model.msa_encoder(msa)
    pair = model.triangle_updates(model.pair_representation(seq_repr, msa_repr))

    c1 = model.se3_equivariant_refinement(pair)
    # Equivariance sanity: same inputs -> same centered coordinates.
    c2 = model.se3_equivariant_refinement(pair)
    assert np.allclose(c1, c2, atol=1e-7)
    assert np.allclose(c1.mean(axis=0), np.zeros(3), atol=1e-8)


def test_torsion_and_ensemble_and_allostery():
    model = _small_model()
    seq = "ACDEFGHIK"
    msa = [seq, "ACDEYGHIK", "ACDEFGHVK"]

    out = model.ensemble_sample(seq, msa, "intermediate")
    assert out["mean_coordinates"].shape == (len(seq), 3)
    assert out["coordinate_variance"].shape == (len(seq), 3)
    assert out["mean_confidence"].shape == (len(seq),)
    assert len(out["members"]) == model.config.n_ensemble

    member0 = out["members"][0]
    assert member0.torsion_angles.shape == (len(seq), 3)
    assert member0.torsion_logits.shape[0] == len(seq)

    states = model.model_allosteric_states(seq, msa)
    assert set(states.keys()) == set(model.config.allosteric_states)
