from protein_folding import EnergyWeights, ProteinFolder, ProteinFoldingConfig


def test_score_counts_nonconsecutive_hh_contacts_with_weight():
    folder = ProteinFolder("HHHH", ProteinFoldingConfig(energy=EnergyWeights(hh_contact=-2.0)))
    coords = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
    ]
    # only non-consecutive contact is (0,3)
    assert folder.score(coords) < -1.5


def test_fold_returns_valid_structure_and_metrics():
    cfg = ProteinFoldingConfig(
        dimensions=3,
        initial_temperature=5.0,
        final_temperature=0.2,
        cooling_rate=0.97,
        steps_per_temperature=80,
        replicas=4,
        restarts=2,
        random_seed=7,
        track_history=True,
        history_stride=5,
    )
    folder = ProteinFolder("HPPHHPH", cfg)
    result = folder.fold()

    coords = result["best_coordinates"]
    assert len(coords) == 7
    assert len(set(coords)) == 7

    for i in range(len(coords) - 1):
        manhattan = sum(abs(coords[i][k] - coords[i + 1][k]) for k in range(3))
        assert manhattan == 1

    cmap = result["contact_map"]
    assert len(cmap) == 7
    assert all(len(row) == 7 for row in cmap)
    assert result["radius_of_gyration"] > 0
    assert 0 <= result["acceptance_rate"] <= 1


def test_2d_mode_uses_planar_coordinates():
    cfg = ProteinFoldingConfig(
        dimensions=2,
        initial_temperature=4.0,
        final_temperature=0.2,
        cooling_rate=0.96,
        steps_per_temperature=50,
        replicas=3,
        restarts=1,
        random_seed=11,
    )
    folder = ProteinFolder("HPPHHPP", cfg)
    result = folder.fold()
    for _, _, z in result["best_coordinates"]:
        assert z == 0


def test_invalid_sequence_raises():
    try:
        ProteinFolder("ABCD")
    except ValueError as exc:
        assert "H' and 'P" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid sequence")
