"""
features.py
------------
Shared preprocessing helpers so app.py can transform raw input (a form
submission or an uploaded CSV row) into the exact feature vector the trained
model expects.
"""

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

NUMERIC_FEATURES = ["duration", "src_bytes", "dst_bytes", "count", "srv_count",
                     "serror_rate", "same_srv_rate", "dst_host_count"]
CATEGORICAL_FEATURES = ["protocol_type", "service"]


class Predictor:
    def __init__(self):
        self.model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
        self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        self.encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
        self.label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        self.feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

    def _safe_encode(self, col, value):
        le = self.encoders[col]
        if value not in le.classes_:
            value = le.classes_[0]
        return le.transform([value])[0]

    def predict_one(self, record: dict):
        row = {}
        for col in NUMERIC_FEATURES:
            row[col] = float(record.get(col, 0) or 0)
        for col in CATEGORICAL_FEATURES:
            row[col] = self._safe_encode(col, record.get(col, self.encoders[col].classes_[0]))

        X = pd.DataFrame([row])[self.feature_cols]
        X_scaled = self.scaler.transform(X)

        pred_encoded = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
        confidence = float(np.max(proba))

        return pred_label, confidence

    def predict_dataframe(self, df: pd.DataFrame):
        results = []
        for _, row in df.iterrows():
            record = row.to_dict()
            pred, conf = self.predict_one(record)
            results.append((record, pred, conf))
        return results
