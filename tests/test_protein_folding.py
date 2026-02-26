import numpy as np

from protein_folding import HyperFoldX, ModelConfig


def _model() -> HyperFoldX:
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
        random_seed=5,
        allosteric_states=("inactive", "active", "intermediate"),
    )
    return HyperFoldX(cfg)


def test_embedding_msa_pair_shapes():
    m = _model()
    seq = "ACDEFGH"
    msa = ["ACDEFGH", "ACDEYGH", "ACDEFGH"]

    s = m.sequence_embedding(seq, "inactive")
    a = m.msa_encoder(msa)
    p = m.pair_representation(s, a)

    assert s.shape == (len(seq), m.config.d_seq)
    assert a.shape == (len(seq), m.config.d_msa)
    assert p.shape == (len(seq), len(seq), m.config.d_pair)
    assert np.allclose(p, np.transpose(p, (1, 0, 2)), atol=1e-6)


def test_refinement_and_heads_shapes():
    m = _model()
    seq = "ACDEFGHIK"
    msa = [seq, "ACDEYGHIK", "ACDEFGHVK"]

    pred = m._single(seq, msa, "active")
    assert pred.coordinates.shape == (len(seq), 3)
    assert np.allclose(pred.coordinates.mean(axis=0), np.zeros(3), atol=1e-6)
    assert pred.torsion_logits.shape == (len(seq), 3, m.config.torsion_bins)
    assert pred.distogram_logits.shape == (len(seq), len(seq), m.config.dist_bins)
    assert pred.plddt.shape == (len(seq),)
    assert np.all((pred.plddt >= 0) & (pred.plddt <= 100))


def test_ensemble_and_allosteric_landscape():
    m = _model()
    seq = "ACDEFGHIK"
    msa = [seq, "ACDEYGHIK"]

    out = m.ensemble_sample(seq, msa, "intermediate")
    assert out["mean_coordinates"].shape == (len(seq), 3)
    assert out["coordinate_variance"].shape == (len(seq), 3)
    assert out["mean_plddt"].shape == (len(seq),)
    assert len(out["members"]) == m.config.n_ensemble

    landscape = m.allosteric_landscape(seq, msa)
    assert set(landscape.keys()) == set(m.config.allosteric_states)
