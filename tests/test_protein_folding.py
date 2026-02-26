from protein_folding import ProteinFolder, ProteinFoldingConfig


def test_score_counts_nonconsecutive_hh_contacts():
    folder = ProteinFolder("HHHH")
    coords = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
    ]
    assert folder.score(coords) == -1


def test_fold_returns_valid_structure():
    folder = ProteinFolder(
        "HPPHHPH",
        ProteinFoldingConfig(
            initial_temperature=6.0,
            final_temperature=0.2,
            cooling_rate=0.98,
            steps_per_temperature=80,
            random_seed=7,
        ),
    )
    result = folder.fold()

    coords = result["best_coordinates"]
    assert len(coords) == 7
    assert len(set(coords)) == 7

    for i in range(len(coords) - 1):
        manhattan = sum(abs(coords[i][k] - coords[i + 1][k]) for k in range(3))
        assert manhattan == 1


def test_invalid_sequence_raises():
    try:
        ProteinFolder("ABCD")
    except ValueError as exc:
        assert "H' and 'P" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid sequence")
