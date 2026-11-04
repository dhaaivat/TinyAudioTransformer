from dataset import EXPECTED_SIZES, official_splits, speaker_id


def test_official_split_sizes(sc_root):
    splits, classes = official_splits(sc_root)
    assert len(classes) == 35
    assert {s: len(v) for s, v in splits.items()} == EXPECTED_SIZES


def test_speaker_disjoint(sc_root):
    splits, _ = official_splits(sc_root)
    speakers = {s: {speaker_id(r) for r, _ in v} for s, v in splits.items()}
    assert not speakers["train"] & speakers["val"]
    assert not speakers["train"] & speakers["test"]
    assert not speakers["val"] & speakers["test"]


def test_every_split_has_every_class(sc_root):
    # The old 80/10/10 slice of a label-sorted list put only the last label(s) in test
    splits, classes = official_splits(sc_root)
    for items in splits.values():
        assert {label for _, label in items} == set(range(len(classes)))


def test_no_background_noise_clips(sc_root):
    splits, _ = official_splits(sc_root)
    assert not any(r.startswith("_") for items in splits.values() for r, _ in items)
