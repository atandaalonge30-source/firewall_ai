"""Reusable, deliberately conservative text cleaning for news classification."""
from __future__ import annotations

import html
import re
from typing import Any


def clean_text(value: Any) -> str:
    """Convert raw news text to normalized text without removing useful wording.

    The vectorizer tokenizes the result. We keep negations and numbers because both
    can be meaningful in a news claim. This function is defined in a module (not a
    training notebook) so the saved scikit-learn model can be loaded by the web app.
    """
    if value is None:
        return ""
    text = html.unescape(str(value)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
