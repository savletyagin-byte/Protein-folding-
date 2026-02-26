import numpy as np

from protein_folding import ModelConfig, OmegaFoldUltra


def _model() -> OmegaFoldUltra:
    cfg = ModelConfig(
        d_seq=64,
        d_msa=48,
        d_pair=40,
        n_evo_blocks=2,
        n_recycles=2,
        n_triangle_updates=1,
        n_refine_steps=6,
        n_diffusion_steps=4,
        torsion_bins=24,
        dist_bins=32,
        n_ensemble=3,
        anneal_steps=20,
        msa_dropout=0.05,
        random_seed=5,
        allosteric_states=("inactive", "active", "intermediate"),
    )
    return OmegaFoldUltra(cfg)


def test_embeddings_msa_pair_shapes():
    m = _model()
    seq = "ACDEFGH"
    msa = ["ACDEFGH", "ACDEYGH", "ACDEFGH"]

    s = m.sequence_embedding(seq, "inactive")
    ms, coupling = m.msa_encoder(msa)
    p = m.pair_representation(s, ms, coupling)

    assert s.shape == (len(seq), m.config.d_seq)
    assert ms.shape == (len(seq), m.config.d_msa)
    assert coupling.shape == (len(seq), len(seq))
    assert p.shape == (len(seq), len(seq), m.config.d_pair)
    assert np.allclose(p, np.transpose(p, (1, 0, 2)), atol=1e-6)


def test_single_prediction_has_new_advanced_outputs():
    m = _model()
    seq = "ACDEFGHIK"
    msa = [seq, "ACDEYGHIK", "ACDEFGHVK"]
    pred = m._single(seq, msa, "active")

    assert pred.coordinates.shape == (len(seq), 3)
    assert np.allclose(pred.coordinates.mean(axis=0), np.zeros(3), atol=1e-6)
    assert pred.torsion_logits.shape == (len(seq), 3, m.config.torsion_bins)
    assert pred.distogram_logits.shape == (len(seq), len(seq), m.config.dist_bins)
    assert pred.plddt.shape == (len(seq),)
    assert pred.pae.shape == (len(seq), len(seq))
    assert 0 <= pred.self_consistency <= 1
    assert np.all((pred.plddt >= 0) & (pred.plddt <= 100))
    assert pred.quality["radius_of_gyration"] > 0


def test_ensemble_ranking_and_landscape():
    m = _model()
    seq = "ACDEFGHIK"
    msa = [seq, "ACDEYGHIK"]

    out = m.ensemble_sample(seq, msa, "intermediate")
    assert len(out["members"]) == m.config.n_ensemble
    assert out["mean_coordinates"].shape == (len(seq), 3)
    assert out["mean_pae"].shape == (len(seq), len(seq))
    assert len(out["ranked_indices"]) == m.config.n_ensemble
    assert 0 <= out["best_member_index"] < m.config.n_ensemble
    assert 0 <= out["mean_self_consistency"] <= 1

    landscape = m.allosteric_landscape(seq, msa)
    assert set(landscape.keys()) == set(m.config.allosteric_states)
