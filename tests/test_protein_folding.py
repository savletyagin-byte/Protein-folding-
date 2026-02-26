import numpy as np

from protein_folding import ModelConfig, OmegaFoldPrototypeX


def _model() -> OmegaFoldPrototypeX:
    cfg = ModelConfig(
        d_seq=64,
        d_msa=48,
        d_pair=40,
        n_evo_blocks=2,
        n_recycles=2,
        n_triangle_updates=1,
        n_refine_steps=8,
        n_diffusion_steps=6,
        torsion_bins=24,
        dist_bins=32,
        n_ensemble=3,
        msa_dropout=0.05,
        random_seed=5,
        allosteric_states=("inactive", "active", "intermediate"),
    )
    return OmegaFoldPrototypeX(cfg)


def test_embedding_msa_and_pair_shapes():
    model = _model()
    seq = "ACDEFGH"
    msa = ["ACDEFGH", "ACDEYGH", "ACDEFGH"]

    s = model.sequence_embedding(seq, "inactive")
    m, coupling = model.msa_encoder(msa)
    p = model.pair_representation(s, m, coupling)

    assert s.shape == (len(seq), model.config.d_seq)
    assert m.shape == (len(seq), model.config.d_msa)
    assert coupling.shape == (len(seq), len(seq))
    assert p.shape == (len(seq), len(seq), model.config.d_pair)
    assert np.allclose(p, np.transpose(p, (1, 0, 2)), atol=1e-6)


def test_single_prediction_outputs_and_quality():
    model = _model()
    seq = "ACDEFGHIK"
    msa = [seq, "ACDEYGHIK", "ACDEFGHVK"]

    pred = model._single(seq, msa, "active")
    assert pred.coordinates.shape == (len(seq), 3)
    assert np.allclose(pred.coordinates.mean(axis=0), np.zeros(3), atol=1e-6)
    assert pred.torsion_logits.shape == (len(seq), 3, model.config.torsion_bins)
    assert pred.distogram_logits.shape == (len(seq), len(seq), model.config.dist_bins)
    assert pred.plddt.shape == (len(seq),)
    assert pred.pae.shape == (len(seq), len(seq))
    assert np.all((pred.plddt >= 0) & (pred.plddt <= 100))
    assert pred.quality["radius_of_gyration"] > 0


def test_ensemble_and_allosteric_landscape():
    model = _model()
    seq = "ACDEFGHIK"
    msa = [seq, "ACDEYGHIK"]

    out = model.ensemble_sample(seq, msa, "intermediate")
    assert len(out["members"]) == model.config.n_ensemble
    assert out["mean_coordinates"].shape == (len(seq), 3)
    assert out["mean_pae"].shape == (len(seq), len(seq))
    assert len(out["ranked_indices"]) == model.config.n_ensemble
    assert 0 <= out["best_member_index"] < model.config.n_ensemble

    landscape = model.allosteric_landscape(seq, msa)
    assert set(landscape.keys()) == set(model.config.allosteric_states)
