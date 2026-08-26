"""
app.py
------
Intelligent Firewall Rule Recommendation System - Flask web application.

Routes:
  /                -> Dashboard (model performance overview)
  /analyze         -> Manually enter a single traffic record to classify
  /upload          -> Upload a CSV of traffic records for batch analysis
  /results         -> Shows classification results + generated rules
  /rules           -> History of all generated rule recommendations
"""

import os
import json
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session

from utils.features import Predictor
from utils.rules import recommend_rule

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

predictor = Predictor()

with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

RULE_LOG = []  # in-memory demo log of generated rules (most-recent-first)


@app.route("/")
def dashboard():
    total_rules = len(RULE_LOG)
    denied = len([r for r in RULE_LOG if r["action"] == "DENY"])
    allowed = len([r for r in RULE_LOG if r["action"] == "ALLOW"])
    return render_template(
        "dashboard.html",
        metrics=METRICS,
        total_rules=total_rules,
        denied=denied,
        allowed=allowed,
        recent=RULE_LOG[:5],
    )


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        record = {
            "duration": request.form.get("duration"),
            "protocol_type": request.form.get("protocol_type"),
            "service": request.form.get("service"),
            "src_bytes": request.form.get("src_bytes"),
            "dst_bytes": request.form.get("dst_bytes"),
            "count": request.form.get("count"),
            "srv_count": request.form.get("srv_count"),
            "serror_rate": request.form.get("serror_rate"),
            "same_srv_rate": request.form.get("same_srv_rate"),
            "dst_host_count": request.form.get("dst_host_count"),
            "src_ip": request.form.get("src_ip"),
        }
        pred, conf = predictor.predict_one(record)
        rule = recommend_rule(record, pred, conf)
        RULE_LOG.insert(0, rule)
        return render_template("results.html", single=True, record=record,
                                prediction=pred, confidence=round(conf * 100, 1), rule=rule)
    return render_template("analyze.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("csv_file")
        if not file or file.filename == "":
            return render_template("upload.html", error="Please choose a CSV file to upload.")
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render_template("upload.html", error=f"Could not read CSV: {e}")

        results = predictor.predict_dataframe(df)
        batch = []
        for record, pred, conf in results:
            rule = recommend_rule(record, pred, conf)
            RULE_LOG.insert(0, rule)
            batch.append({"record": record, "prediction": pred,
                           "confidence": round(conf * 100, 1), "rule": rule})
        return render_template("results.html", single=False, batch=batch)
    return render_template("upload.html")


@app.route("/rules")
def rules():
    return render_template("rules.html", rules=RULE_LOG)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
