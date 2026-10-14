"""Train, evaluate and save a fake-news text classification model.

Example:
python -m src.train --fake-csv data/Fake.csv --true-csv data/True.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.text_processing import clean_text

RANDOM_STATE = 42
LABELS = ["fake", "real"]
ADDITIONAL_EXAMPLE_WEIGHT = 50.0


def combine_title_and_text(frame: pd.DataFrame) -> pd.Series:
    """Return one document per record, accepting common Kaggle column names."""
    columns = {str(c).lower().strip(): c for c in frame.columns}
    title_col = columns.get("title")
    text_col = columns.get("text") or columns.get("content") or columns.get("article")
    if text_col is None and title_col is None:
        raise ValueError("Expected a title, text, content, or article column.")
    title = frame[title_col].fillna("").astype(str) if title_col else ""
    body = frame[text_col].fillna("").astype(str) if text_col else ""
    return (title + " " + body).str.strip()


def load_data(args: argparse.Namespace, require_both_labels: bool = True) -> pd.DataFrame:
    """Load either one labelled CSV or the usual Fake.csv / True.csv pair."""
    if args.data:
        data = pd.read_csv(args.data)
        cols = {str(c).lower().strip(): c for c in data.columns}
        label_col = cols.get("label")
        target_col = cols.get("target")
        if label_col is None and target_col is None:
            raise ValueError("The --data file must have a 'label' column or a supported 'target' column.")
        if label_col is None:
            result_label = data[target_col].replace({"0": "real", "1": "fake", 0: "real", 1: "fake"})
        else:
            result_label = data[label_col]
        result = pd.DataFrame({"text": combine_title_and_text(data), "label": result_label})
    else:
        if not (args.fake_csv and args.true_csv):
            raise ValueError("Provide --data OR both --fake-csv and --true-csv.")
        fake = pd.read_csv(args.fake_csv)
        real = pd.read_csv(args.true_csv)
        result = pd.concat([
            pd.DataFrame({"text": combine_title_and_text(fake), "label": "fake"}),
            pd.DataFrame({"text": combine_title_and_text(real), "label": "real"}),
        ], ignore_index=True)

    result = result.dropna(subset=["text", "label"]).copy()
    result["text"] = result["text"].astype(str).str.strip()
    result = result[result["text"].str.len() >= args.min_characters]
    result["label"] = result["label"].astype(str).str.strip().str.lower()
    result["label"] = result["label"].replace({"0": "fake", "1": "real", "false": "fake", "true": "real"})
    unknown = set(result["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Labels must be fake/real or 0/1; found {sorted(unknown)}")
    result["_normalized_text"] = result["text"].map(clean_text)
    result = result[result["_normalized_text"].str.len() > 0].copy()
    if result.empty:
        raise ValueError("No usable examples remain after cleaning.")
    if require_both_labels and result["label"].nunique() != 2:
        raise ValueError("Training needs examples for both fake and real labels.")

    rows_before_deduplication = len(result)
    labels_per_text = result.groupby("_normalized_text", sort=False)["label"].nunique()
    conflicting_texts = set(labels_per_text[labels_per_text > 1].index)
    conflicting_rows = int(result["_normalized_text"].isin(conflicting_texts).sum())
    result = result[~result["_normalized_text"].isin(conflicting_texts)].copy()
    duplicate_texts_removed = int(result.duplicated("_normalized_text").sum())
    result = result.drop_duplicates("_normalized_text", keep="first")
    result = result.drop(columns="_normalized_text")
    if require_both_labels and result["label"].nunique() != 2:
        raise ValueError("Deduplication left fewer than two classes; check the dataset labels.")
    result = result.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    result.attrs["deduplication"] = {
        "rows_before_deduplication": rows_before_deduplication,
        "duplicate_texts_removed": duplicate_texts_removed,
        "conflicting_duplicate_groups_removed": len(conflicting_texts),
        "conflicting_duplicate_rows_removed": conflicting_rows,
    }
    return result


def load_additional_examples(
    path: Path, data: pd.DataFrame, min_characters: int, expected_label: str
) -> pd.Series:
    """Load user-labelled examples for training without changing evaluation data."""
    if expected_label not in LABELS:
        raise ValueError(f"Expected label must be one of {LABELS}.")
    args = argparse.Namespace(
        data=path, fake_csv=None, true_csv=None, min_characters=min_characters
    )
    examples = load_data(args, require_both_labels=False)
    if not examples["label"].eq(expected_label).all():
        raise ValueError(
            f"The additional {expected_label.upper()} data file must contain only "
            f"'{expected_label}' labels."
        )

    labels_by_text = data.assign(_normalized_text=data["text"].map(clean_text))
    labels_by_text = labels_by_text.groupby("_normalized_text")["label"].agg(set)
    normalized_examples = examples["text"].map(clean_text)
    conflicts = normalized_examples.map(
        lambda text: bool(labels_by_text.get(text, set()) - {expected_label})
    )
    if conflicts.any():
        raise ValueError(
            f"Additional {expected_label.upper()} examples conflict with "
            "oppositely-labelled training records."
        )

    already_present = normalized_examples.isin(labels_by_text.index)
    return examples.loc[~already_present, "text"].reset_index(drop=True)


def split_data(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create reproducible 70/15/15 train, validation and final test sets."""
    train, holdout = train_test_split(data, test_size=0.30, stratify=data.label, random_state=RANDOM_STATE)
    validation, test = train_test_split(holdout, test_size=0.50, stratify=holdout.label, random_state=RANDOM_STATE)
    return train.text, train.label, validation.text, validation.label, test.text, test.label


def make_pipeline(max_features: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, stop_words="english", ngram_range=(1, 2),
                                  min_df=2, max_df=0.95, max_features=max_features, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def metrics_for(y_true: pd.Series, predictions) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, labels=LABELS, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "fake": {"precision": round(float(precision[0]), 4), "recall": round(float(recall[0]), 4), "f1": round(float(f1[0]), 4)},
        "real": {"precision": round(float(precision[1]), 4), "recall": round(float(recall[1]), 4), "f1": round(float(f1[1]), 4)},
    }


def data_profile(data: pd.DataFrame, source: str) -> dict:
    """Summarize the cleaned input data for the workflow report."""
    class_counts = data["label"].value_counts().reindex(LABELS, fill_value=0)
    total = len(data)
    deduplication = data.attrs.get("deduplication", {})
    duplicate_texts = int(
        deduplication.get("duplicate_texts_removed", data["text"].map(clean_text).duplicated().sum())
    )
    source_is_reference_dataset = source == "fake_real_news.csv"
    return {
        "source": source,
        "rows": total,
        "rows_before_deduplication": int(deduplication.get("rows_before_deduplication", total)),
        "class_counts": {label: int(class_counts[label]) for label in LABELS},
        "class_percentages": {
            label: round(float(class_counts[label] / total * 100), 1) for label in LABELS
        },
        "duplicate_texts_removed": duplicate_texts,
        "conflicting_duplicate_groups_removed": int(
            deduplication.get("conflicting_duplicate_groups_removed", 0)
        ),
        "conflicting_duplicate_rows_removed": int(
            deduplication.get("conflicting_duplicate_rows_removed", 0)
        ),
        "median_characters": int(data["text"].str.len().median()),
        "dataset_attribution": {
            "name": "Fake and Real News Dataset" if source_is_reference_dataset else "User-provided labelled dataset",
            "publisher": "Gilchrist (Hugging Face mirror)" if source_is_reference_dataset else "Not supplied",
            "source_url": (
                "https://huggingface.co/datasets/haoxianc/gilchr_fake-and-real-news-dataset"
                if source_is_reference_dataset else ""
            ),
            "license": "MIT (as stated by the dataset mirror)" if source_is_reference_dataset else "Check the source dataset's license",
            "accessed": "2026-10-13" if source_is_reference_dataset else "",
        },
        "preprocessing_summary": "Records were normalized and deduplicated before the stratified split to keep repeated normalized articles from crossing between splits.",
        "preprocessing_steps": [
            "Combine available title and article text.",
            "Remove blank records and entries shorter than the configured minimum.",
            "Normalize case, URLs, HTML, punctuation and whitespace for duplicate detection and feature extraction.",
            "Remove duplicate normalized articles; discard all copies of any duplicate with conflicting labels.",
            "Extract English-stop-word-filtered TF-IDF unigrams and bigrams.",
        ],
    }


def feature_analysis(model: Pipeline, limit: int = 12) -> dict:
    """Return strongest signed linear-model terms for each learned class."""
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    terms = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    negative_label, positive_label = str(classifier.classes_[0]), str(classifier.classes_[1])

    def strongest(indices) -> list[dict]:
        return [
            {"term": str(terms[index]), "weight": round(float(coefficients[index]), 4)}
            for index in indices
        ]

    return {
        negative_label: strongest(coefficients.argsort()[:limit][::-1]),
        positive_label: strongest(coefficients.argsort()[-limit:][::-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fake news NLP model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="CSV with text/title and label columns")
    source.add_argument("--fake-csv", type=Path, help="CSV containing fake articles")
    parser.add_argument("--true-csv", type=Path, help="CSV containing genuine articles (required with --fake-csv)")
    parser.add_argument(
        "--additional-real-data",
        type=Path,
        help="Optional CSV of user-labelled REAL examples added only to the training split",
    )
    parser.add_argument(
        "--additional-fake-data",
        type=Path,
        help="Optional CSV of user-labelled FAKE examples added only to the training split",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--min-characters", type=int, default=40)
    parser.add_argument("--max-features", type=int, default=100000)
    args = parser.parse_args()

    data = load_data(args)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(data)
    additional_example_counts = {}
    known_training_data = data.copy()
    for path, label in (
        (args.additional_real_data, "real"),
        (args.additional_fake_data, "fake"),
    ):
        if path:
            examples = load_additional_examples(
                path, known_training_data, args.min_characters, label
            )
            x_train = pd.concat([x_train, examples], ignore_index=True)
            y_train = pd.concat([
                y_train,
                pd.Series([label] * len(examples), name="label"),
            ], ignore_index=True)
            known_training_data = pd.concat([
                known_training_data,
                pd.DataFrame({"text": examples, "label": label}),
            ], ignore_index=True)
            additional_example_counts[label] = len(examples)
    model = make_pipeline(args.max_features)
    additional_count = sum(additional_example_counts.values())
    sample_weights = [1.0] * (len(y_train) - additional_count)
    sample_weights.extend([ADDITIONAL_EXAMPLE_WEIGHT] * additional_count)
    model.fit(x_train, y_train, classifier__sample_weight=sample_weights)
    validation_metrics = metrics_for(y_val, model.predict(x_val))
    test_predictions = model.predict(x_test)
    test_metrics = metrics_for(y_test, test_predictions)
    test_confusion = confusion_matrix(y_test, test_predictions, labels=LABELS)
    test_errors = {
        "fake_articles_missed": int(test_confusion[0, 1]),
        "real_articles_flagged_fake": int(test_confusion[1, 0]),
        "total": int((y_test.to_numpy() != test_predictions).sum()),
    }
    source = args.data.name if args.data else f"{args.fake_csv.name} + {args.true_csv.name}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "fake_news_model.joblib")
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "dataset_rows": int(len(data)),
        "additional_real_training_examples": int(additional_example_counts.get("real", 0)),
        "additional_fake_training_examples": int(additional_example_counts.get("fake", 0)),
        "data_profile": data_profile(data, source),
        "splits": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "feature_analysis": feature_analysis(model),
        "test_errors": test_errors,
        "validation": validation_metrics, "test": test_metrics,
        "classification_report": classification_report(y_test, test_predictions, labels=LABELS, zero_division=0, output_dict=True),
    }, indent=2), encoding="utf-8")
    pd.DataFrame({"text": x_test, "actual": y_test, "predicted": test_predictions}).to_csv(args.output_dir / "test_predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, test_predictions, labels=LABELS), display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Fake News Detection: Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(args.output_dir / "confusion_matrix.png", dpi=180)
    print(json.dumps({"validation": validation_metrics, "test": test_metrics}, indent=2))
    print(f"Saved model and reports to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
