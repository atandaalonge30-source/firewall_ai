import argparse

import pandas as pd

from src.train import data_profile, load_additional_examples, load_data


def test_load_data_maps_target_labels_as_documented(tmp_path):
    data_path = tmp_path / "news.csv"
    pd.DataFrame({
        "title": ["A real headline", "A fake headline"],
        "content": ["A sufficiently long real article for training.", "A sufficiently long fake article for training."],
        "target": [0, 1],
    }).to_csv(data_path, index=False)
    args = argparse.Namespace(data=data_path, fake_csv=None, true_csv=None, min_characters=1)

    result = load_data(args)

    assert result.set_index("text")["label"].to_dict() == {
        "A real headline A sufficiently long real article for training.": "real",
        "A fake headline A sufficiently long fake article for training.": "fake",
    }


def test_load_data_removes_normalized_duplicates_before_splitting(tmp_path):
    data_path = tmp_path / "news.csv"
    pd.DataFrame({
        "text": [
            "A fake article with enough content to be used as a training example.",
            "A fake article with enough content to be used as a training example!",
            "A real article with enough content to be used as a training example.",
        ],
        "label": ["fake", "fake", "real"],
    }).to_csv(data_path, index=False)
    args = argparse.Namespace(data=data_path, fake_csv=None, true_csv=None, min_characters=1)

    result = load_data(args)

    assert len(result) == 2
    assert result.attrs["deduplication"] == {
        "rows_before_deduplication": 3,
        "duplicate_texts_removed": 1,
        "conflicting_duplicate_groups_removed": 0,
        "conflicting_duplicate_rows_removed": 0,
    }


def test_load_data_discards_all_copies_with_conflicting_labels(tmp_path):
    data_path = tmp_path / "news.csv"
    pd.DataFrame({
        "text": [
            "One repeated article with a sufficiently long text for testing.",
            "One repeated article with a sufficiently long text for testing!",
            "A unique real article with sufficient text for the training sample.",
            "A unique fake article with sufficient text for the training sample.",
        ],
        "label": ["fake", "real", "real", "fake"],
    }).to_csv(data_path, index=False)
    args = argparse.Namespace(data=data_path, fake_csv=None, true_csv=None, min_characters=1)

    result = load_data(args)

    assert len(result) == 2
    assert not result["text"].str.startswith("One repeated").any()
    assert result.attrs["deduplication"]["conflicting_duplicate_groups_removed"] == 1
    assert result.attrs["deduplication"]["conflicting_duplicate_rows_removed"] == 2


def test_data_profile_summarizes_class_counts_and_duplicate_removal():
    data = pd.DataFrame({
        "text": ["A repeated article", "A repeated article", "Another story"],
        "label": ["fake", "fake", "real"],
    })

    profile = data_profile(data, "news.csv")

    assert profile["source"] == "news.csv"
    assert profile["rows"] == 3
    assert profile["class_counts"] == {"fake": 2, "real": 1}
    assert profile["class_percentages"] == {"fake": 66.7, "real": 33.3}
    assert profile["duplicate_texts_removed"] == 1
    assert profile["dataset_attribution"]["license"] == "Check the source dataset's license"


def test_additional_real_examples_are_training_only_and_conflicts_are_rejected(tmp_path):
    examples_path = tmp_path / "additional.csv"
    pd.DataFrame({
        "text": [
            "A user-labelled real story with enough words to pass the configured minimum.",
            "A known fake story repeated with a different label from the main data.",
        ],
        "label": ["real", "real"],
    }).to_csv(examples_path, index=False)
    base_data = pd.DataFrame({
        "text": [
            "An existing real story with enough words to pass the configured minimum.",
            "A known fake story repeated with a different label from the main data.",
        ],
        "label": ["real", "fake"],
    })

    try:
        load_additional_examples(examples_path, base_data, min_characters=1, expected_label="real")
    except ValueError as error:
        assert "conflict" in str(error)
    else:
        raise AssertionError("Conflicting REAL examples should be rejected.")


def test_additional_real_examples_accept_only_real_labels_and_skip_existing_records(tmp_path):
    examples_path = tmp_path / "additional.csv"
    pd.DataFrame({
        "text": [
            "A story already in the primary data and labeled as real.",
            "A new user-labelled real story with enough unique text.",
        ],
        "label": ["real", "real"],
    }).to_csv(examples_path, index=False)
    base_data = pd.DataFrame({
        "text": ["A story already in the primary data and labeled as real."],
        "label": ["real"],
    })

    result = load_additional_examples(
        examples_path, base_data, min_characters=1, expected_label="real"
    )

    assert result.tolist() == ["A new user-labelled real story with enough unique text."]


def test_additional_examples_reject_labels_outside_the_requested_class(tmp_path):
    examples_path = tmp_path / "additional.csv"
    pd.DataFrame({
        "text": ["A fake-labelled claim with enough text to pass minimum length."],
        "label": ["fake"],
    }).to_csv(examples_path, index=False)
    base_data = pd.DataFrame({
        "text": ["An existing real story with enough words for this example."],
        "label": ["real"],
    })

    try:
        load_additional_examples(
            examples_path, base_data, min_characters=1, expected_label="real"
        )
    except ValueError as error:
        assert "only 'real' labels" in str(error)
    else:
        raise AssertionError("Examples with the wrong label should be rejected.")
