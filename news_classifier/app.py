"""Flask interface for news prediction and model workflow reports."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
from flask import Flask, abort, jsonify, render_template_string, request, send_from_directory

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ARTIFACT_DIR / "fake_news_model.joblib"))
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
app = Flask(__name__)

BASE_STYLE = """
<style>
*{box-sizing:border-box}body{font-family:Arial,sans-serif;max-width:1080px;margin:32px auto;padding:0 22px;color:#19212b;line-height:1.5}
nav{display:flex;gap:18px;border-bottom:1px solid #dce4e9;padding:12px 0 18px;margin-bottom:28px}
a{color:#146c94}nav a{font-weight:700;text-decoration:none}nav a.active{color:#172b3c;border-bottom:3px solid #146c94;padding-bottom:7px}
h1{font-size:32px;margin-bottom:6px}.muted{color:#536575}.notice{padding:14px;background:#fff7df;border-left:4px solid #d29b22}
.panel,.card{border:1px solid #dce4e9;border-radius:9px;padding:20px;background:#fff}.panel{margin:20px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}
.card h3{font-size:14px;margin:0 0 8px;color:#536575}.value{font-size:25px;font-weight:700}.small{font-size:13px;color:#536575}
textarea{width:100%;min-height:180px;padding:12px;font-size:16px;border:1px solid #aebbc4;border-radius:5px}button{margin-top:14px;background:#146c94;color:white;border:0;padding:12px 19px;border-radius:5px;font-weight:700;cursor:pointer}
.result{margin-top:22px;padding:18px;background:#edf6fa;border-left:5px solid #146c94}.result strong{font-size:18px}.error{color:#a11}
.steps{display:flex;flex-wrap:wrap;gap:8px}.step{background:#edf6fa;padding:9px 12px;border-radius:5px;font-weight:700}
.bar{height:10px;background:#e6edf1;border-radius:8px;overflow:hidden}.bar span{display:block;height:100%;background:#146c94}
table{border-collapse:collapse;width:100%;margin-top:12px}th,td{text-align:left;padding:10px;border-bottom:1px solid #e3e9ed}th{color:#536575}
.terms{display:flex;flex-wrap:wrap;gap:8px}.term{background:#edf6fa;border-radius:4px;padding:6px 9px;font-size:14px}
img{max-width:100%;height:auto}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
@media(max-width:600px){body{margin:18px auto;padding:0 14px}h1{font-size:26px}}
</style>
"""

NAV = """
<nav><a class="{{ 'active' if active == 'predict' else '' }}" href="/">Check a story</a>
<a class="{{ 'active' if active == 'workflow' else '' }}" href="/workflow">Project workflow &amp; results</a></nav>
"""

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fake News Detection System</title>""" + BASE_STYLE + """</head><body>""" + NAV + """
<h1>Fake News Detection System</h1>
<p class="muted">Paste a news headline or article. This is a screening model, not a fact-checking authority.</p>
<form method="post"><textarea name="text" minlength="20" required placeholder="Paste news text here...">{{ text }}</textarea><br><button>Analyse text</button></form>
{% if result %}<div class="result"><strong>Prediction: {{ result.label|upper }}</strong><br>Model confidence: {{ result.confidence }}<br><small>Always verify important claims with credible sources.</small></div>{% endif %}
{% if error %}<p class="error">{{ error }}</p>{% endif %}
<p class="small">See <a href="/workflow">Project workflow &amp; results</a> for the dataset summary, training stages, evaluation and feature analysis.</p>
</body></html>"""

WORKFLOW_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Project Workflow and Results</title>""" + BASE_STYLE + """</head><body>""" + NAV + """
{% macro metrics_table(metrics) %}
<table><thead><tr><th>Metric</th><th>Score</th></tr></thead><tbody>
<tr><td>Accuracy</td><td>{{ "%.1f"|format(metrics.accuracy * 100) }}%</td></tr>
{% for label in ("fake", "real") %}
{% for name in ("precision", "recall", "f1") %}
<tr><td>{{ label|title }} {{ name|title }}</td><td>{{ "%.1f"|format(metrics[label][name] * 100) }}%</td></tr>
{% endfor %}{% endfor %}</tbody></table>
{% endmacro %}
<h1>Project Workflow &amp; Results</h1>
<p class="muted">Summary of the dataset and the trained model currently deployed.</p>
<div class="steps"><span class="step">1. Data input &amp; EDA</span><span class="step">2. Preprocessing</span>
<span class="step">3. Stratified data split</span><span class="step">4. Training &amp; evaluation</span>
<span class="step">5. Result reporting</span><span class="step">6. Feature analysis</span></div>

<section class="panel"><h2>1. Data input &amp; exploratory analysis</h2>
<p>Training data file: <strong>{{ data.source }}</strong></p>
<p>Dataset: <strong>{{ data.dataset_attribution.name }}</strong>
{% if data.dataset_attribution.source_url %} — <a href="{{ data.dataset_attribution.source_url }}" rel="noopener noreferrer">source and dataset card</a>{% endif %}</p>
<p>Attribution: {{ data.dataset_attribution.publisher }}. License: {{ data.dataset_attribution.license }}.
{% if data.dataset_attribution.accessed %}Accessed {{ data.dataset_attribution.accessed }}.{% endif %}</p>
<div class="grid">
<div class="card"><h3>Records after deduplication</h3><div class="value">{{ "{:,}".format(data.rows) }}</div></div>
<div class="card"><h3>Records before deduplication</h3><div class="value">{{ "{:,}".format(data.rows_before_deduplication) }}</div></div>
<div class="card"><h3>Duplicate records removed</h3><div class="value">{{ "{:,}".format(data.duplicate_texts_removed) }}</div></div>
<div class="card"><h3>Median article length</h3><div class="value">{{ "{:,}".format(data.median_characters) }} chars</div></div>
<div class="card"><h3>Additional REAL training examples</h3><div class="value">{{ "{:,}".format(additional_real_examples) }}</div><div class="small">User-labelled; not included in held-out evaluation.</div></div>
<div class="card"><h3>Additional FAKE training examples</h3><div class="value">{{ "{:,}".format(additional_fake_examples) }}</div><div class="small">User-labelled; not included in held-out evaluation.</div></div>
</div>
{% if data.conflicting_duplicate_groups_removed %}<p class="notice">{{ data.conflicting_duplicate_groups_removed }} repeated normalized texts had conflicting labels ({{ data.conflicting_duplicate_rows_removed }} rows); all copies were excluded.</p>{% endif %}
<h3>Class distribution</h3>
{% for label, count in data.class_counts.items() %}
<p><strong>{{ label|upper }}</strong> — {{ "{:,}".format(count) }} ({{ "%.1f"|format(data.class_percentages[label]) }}%)</p>
<div class="bar"><span style="width:{{ data.class_percentages[label] }}%"></span></div>
{% endfor %}
</section>

<section class="panel"><h2>2. Data preprocessing</h2>
<p>{{ data.preprocessing_summary }}</p>
<ul>{% for item in data.preprocessing_steps %}<li>{{ item }}</li>{% endfor %}</ul>
</section>

<section class="panel"><h2>3. Data splitting / resampling</h2>
<p>Stratified split keeps class proportions similar across the training, validation and held-out test sets.</p>
<div class="grid">
{% for split, count in splits.items() %}
<div class="card"><h3>{{ split|title }}</h3><div class="value">{{ "{:,}".format(count) }}</div>
<div class="small">{{ "%.1f"|format(count / total_records * 100) }}% of usable records</div></div>
{% endfor %}
</div>
<p class="small">No records were synthetically resampled. Class-weight balancing is used during model fitting.</p>
</section>

<section class="panel"><h2>4. Training &amp; evaluation</h2>
<div class="grid"><div class="card"><h3>Text features</h3><div class="value">TF-IDF</div><div class="small">Unigrams and bigrams</div></div>
<div class="card"><h3>Classifier</h3><div class="value">Logistic Regression</div><div class="small">Class-weight balanced</div></div>
<div class="card"><h3>Random seed</h3><div class="value">42</div><div class="small">Reproducible split</div></div></div>
<h3>Validation set</h3>{{ metrics_table(validation) }}
<h3>Held-out test set</h3>{{ metrics_table(test) }}
<h3>Test-set error analysis</h3>
<p>{{ test_errors.total }} incorrect predictions out of {{ splits.test }} held-out examples.</p>
<table><thead><tr><th>Error type</th><th>Count</th></tr></thead><tbody>
<tr><td>Fake articles predicted real (missed fake)</td><td>{{ test_errors.fake_articles_missed }}</td></tr>
<tr><td>Real articles predicted fake (false alarm)</td><td>{{ test_errors.real_articles_flagged_fake }}</td></tr>
</tbody></table>
</section>

<section class="panel"><h2>5. Result reporting</h2>
<p>Confusion matrix for the held-out test set (actual labels are rows; predicted labels are columns).</p>
<img src="/artifacts/confusion_matrix.png" alt="Confusion matrix for held-out test predictions">
<p class="small">The test predictions CSV and numeric metrics are generated by the training script. The CSV is not published because it contains article text.</p>
</section>

<section class="panel"><h2>6. Feature analysis</h2>
<p>These are terms with the strongest learned association for each class in the linear classifier. They are model patterns, not evidence that a story is true or false.</p>
<h3>Terms associated with FAKE predictions</h3><div class="terms">{% for item in features.fake %}<span class="term">{{ item.term }}</span>{% endfor %}</div>
<h3>Terms associated with REAL predictions</h3><div class="terms">{% for item in features.real %}<span class="term">{{ item.term }}</span>{% endfor %}</div>
</section>
<p class="notice"><strong>Important limitations:</strong> this is a text-classification demonstration, not a fact-checker. Exact normalized duplicates are removed before splitting, but a random split can still share publisher, topic, or time-period patterns between train and test. The dataset is an educational benchmark with inherited label and source biases; independent, publisher-held-out and time-based evaluation would be needed to establish broader generalization.</p>
<p class="small">For the full method, results, citations, and school-project discussion, see the <a href="https://github.com/atandaalonge30-source/fake-news-detection-system/blob/main/PROJECT_REPORT.md" rel="noopener noreferrer">project report</a>.</p>
</body></html>"""


@app.route("/", methods=["GET", "POST"])
def index():
    text, result, error = "", None, None
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if not MODEL_PATH.is_file():
            error = (
                "Trained model not found. Train it using the commands in README.md "
                "and deploy artifacts/fake_news_model.joblib."
            )
        elif len(text) < 20:
            error = "Please enter at least 20 characters."
        else:
            model = joblib.load(MODEL_PATH)
            scores = model.predict_proba([text])[0]
            position = scores.argmax()
            result = {"label": model.classes_[position], "confidence": f"{scores[position]:.1%}"}
    return render_template_string(PAGE, active="predict", text=text, result=result, error=error)


@app.get("/workflow")
def workflow():
    if not METRICS_PATH.is_file():
        return (
            render_template_string(
                PAGE,
                active="workflow",
                text="",
                result=None,
                error="Training report not found. Retrain the model and deploy artifacts/metrics.json.",
            ),
            503,
        )
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        report = json.load(metrics_file)
    return render_template_string(
        WORKFLOW_PAGE,
        active="workflow",
        data=report["data_profile"],
        splits=report["splits"],
        total_records=report.get("dataset_rows", sum(report["splits"].values()))
        + report.get("additional_real_training_examples", 0)
        + report.get("additional_fake_training_examples", 0),
        additional_real_examples=report.get("additional_real_training_examples", 0),
        additional_fake_examples=report.get("additional_fake_training_examples", 0),
        validation=report["validation"],
        test=report["test"],
        test_errors=report["test_errors"],
        features=report["feature_analysis"],
    )


@app.get("/artifacts/confusion_matrix.png")
def confusion_matrix_image():
    image_path = ARTIFACT_DIR / "confusion_matrix.png"
    if not image_path.is_file():
        abort(404)
    return send_from_directory(ARTIFACT_DIR, "confusion_matrix.png")


@app.get("/health")
def health():
    if not MODEL_PATH.is_file():
        return jsonify(status="model_missing"), 503
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
