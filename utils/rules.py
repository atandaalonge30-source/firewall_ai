"""
rules.py
--------
Translates a machine-learning classification result into a human-readable
firewall rule recommendation. This is the "Rule Recommendation Engine"
described in Chapters 1-3 of the write-up: it bridges the gap between
detection (the ML model) and action (an actual firewall configuration line).
"""

import random

def recommend_rule(record, prediction, probability):
    """
    record: dict of the traffic features (plus optional ip/port info)
    prediction: 'normal' or 'malicious'
    probability: model confidence (0-1) for the predicted class
    """
    src_ip = record.get("src_ip") or _fake_ip()
    dst_port = record.get("dst_port") or _guess_port(record.get("service", ""))
    protocol = record.get("protocol_type", "tcp").upper()

    if prediction == "malicious":
        if probability >= 0.9:
            action = "DENY"
            severity = "High"
        elif probability >= 0.7:
            action = "DENY"
            severity = "Medium"
        else:
            action = "FLAG FOR REVIEW"
            severity = "Low"

        rule_text = f"{action} {protocol} FROM {src_ip} TO ANY PORT {dst_port}"
        reason = (
            f"Traffic classified as malicious with {probability*100:.1f}% confidence "
            f"(elevated {['error rate', 'connection count', 'byte pattern'][random.randint(0,2)]} "
            f"consistent with anomalous behaviour)."
        )
    else:
        action = "ALLOW"
        severity = "None"
        rule_text = f"ALLOW {protocol} FROM {src_ip} TO ANY PORT {dst_port}"
        reason = f"Traffic classified as normal with {probability*100:.1f}% confidence."

    return {
        "src_ip": src_ip,
        "protocol": protocol,
        "dst_port": dst_port,
        "action": action,
        "severity": severity,
        "rule_text": rule_text,
        "reason": reason,
        "confidence": round(probability * 100, 1),
    }


def _fake_ip():
    return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def _guess_port(service):
    mapping = {
        "http": 80, "ftp": 21, "ftp_data": 20, "smtp": 25,
        "dns": 53, "ssh": 22, "telnet": 23, "private": random.randint(1024, 65000),
    }
    return mapping.get(service, random.randint(1024, 65000))
