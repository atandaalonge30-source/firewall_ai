import json

from app import app


def test_home_page_loads():
    response = app.test_client().get("/")

    assert response.status_code == 200
    assert b"Fake News Detection System" in response.data


def test_health_reports_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr("app.MODEL_PATH", tmp_path / "missing-model.joblib")

    response = app.test_client().get("/health")

    assert response.status_code == 503
    assert response.json == {"status": "model_missing"}


def test_prediction_form_explains_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr("app.MODEL_PATH", tmp_path / "missing-model.joblib")

    response = app.test_client().post("/", data={"text": "A long enough news article for this test."})

    assert response.status_code == 200
    assert b"Trained model not found" in response.data


def test_prediction_form_displays_model_prediction(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.touch()

    class FakeScores(list):
        def argmax(self):
            return max(range(len(self)), key=self.__getitem__)

    class FakeModel:
        classes_ = ["fake", "real"]

        def predict_proba(self, texts):
            return [FakeScores([0.2, 0.8])]

    monkeypatch.setattr("app.MODEL_PATH", model_path)
    monkeypatch.setattr("app.joblib.load", lambda _: FakeModel())

    response = app.test_client().post("/", data={"text": "A long enough news article for this test."})

    assert response.status_code == 200
    assert b"Prediction: REAL" in response.data
    assert b"80.0%" in response.data


def test_workflow_page_displays_report_data(monkeypatch, tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({
        "data_profile": {
            "source": "news.csv",
            "rows": 100,
            "rows_before_deduplication": 110,
            "class_counts": {"fake": 50, "real": 50},
            "class_percentages": {"fake": 50.0, "real": 50.0},
            "duplicate_texts_removed": 10,
            "conflicting_duplicate_groups_removed": 0,
            "conflicting_duplicate_rows_removed": 0,
            "median_characters": 240,
            "dataset_attribution": {
                "name": "User-provided labelled dataset",
                "publisher": "Not supplied",
                "source_url": "",
                "license": "Check the source dataset's license",
                "accessed": "",
            },
            "preprocessing_summary": "Cleaned text.",
            "preprocessing_steps": ["Normalize text."],
        },
        "dataset_rows": 100,
        "additional_real_training_examples": 12,
        "additional_fake_training_examples": 13,
        "splits": {"train": 95, "validation": 15, "test": 15},
        "validation": {
            "accuracy": 0.9,
            "fake": {"precision": 0.9, "recall": 0.9, "f1": 0.9},
            "real": {"precision": 0.9, "recall": 0.9, "f1": 0.9},
        },
        "test": {
            "accuracy": 0.8,
            "fake": {"precision": 0.8, "recall": 0.8, "f1": 0.8},
            "real": {"precision": 0.8, "recall": 0.8, "f1": 0.8},
        },
        "feature_analysis": {
            "fake": [{"term": "fabricated claim", "weight": -1.2}],
            "real": [{"term": "official report", "weight": 1.4}],
        },
        "test_errors": {
            "fake_articles_missed": 2,
            "real_articles_flagged_fake": 1,
            "total": 3,
        },
    }), encoding="utf-8")
    monkeypatch.setattr("app.METRICS_PATH", metrics_path)

    response = app.test_client().get("/workflow")

    assert response.status_code == 200
    assert b"Data input" in response.data
    assert b"news.csv" in response.data
    assert b"Training &amp; evaluation" in response.data
    assert b"90.0%" in response.data
    assert b"Additional REAL training examples" in response.data
    assert b"Additional FAKE training examples" in response.data
    assert b">12</div>" in response.data
    assert b">13</div>" in response.data
    assert b"fabricated claim" in response.data
    assert b"Duplicate records removed" in response.data
    assert b"3 incorrect predictions" in response.data


def test_workflow_page_reports_missing_training_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr("app.METRICS_PATH", tmp_path / "missing-metrics.json")

    response = app.test_client().get("/workflow")

    assert response.status_code == 503
    assert b"Training report not found" in response.data


def test_confusion_matrix_route_serves_only_report_image(monkeypatch, tmp_path):
    (tmp_path / "confusion_matrix.png").write_bytes(b"image data")
    monkeypatch.setattr("app.ARTIFACT_DIR", tmp_path)

    response = app.test_client().get("/artifacts/confusion_matrix.png")

    assert response.status_code == 200
    assert response.data == b"image data"
