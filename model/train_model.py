"""
train_model.py
---------------
Trains the machine learning model used by the Intelligent Firewall Rule
Recommendation System.

Pipeline: data cleaning -> feature encoding -> train/test split ->
RandomForestClassifier -> evaluation (accuracy, precision, recall, F1) ->
persist model + encoders to disk with joblib.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "traffic_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

NUMERIC_FEATURES = ["duration", "src_bytes", "dst_bytes", "count", "srv_count",
                     "serror_rate", "same_srv_rate", "dst_host_count"]
CATEGORICAL_FEATURES = ["protocol_type", "service"]

def train():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["label"])  # malicious=0/normal=1 (alphabetical)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df["label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    malicious_label = label_encoder.transform(["malicious"])[0]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label=malicious_label), 4),
        "recall": round(recall_score(y_test, y_pred, pos_label=malicious_label), 4),
        "f1_score": round(f1_score(y_test, y_pred, pos_label=malicious_label), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics["confusion_matrix"] = cm
    metrics["classes"] = list(label_encoder.inverse_transform(sorted(set(y_test))))

    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model trained and saved.")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    train()
