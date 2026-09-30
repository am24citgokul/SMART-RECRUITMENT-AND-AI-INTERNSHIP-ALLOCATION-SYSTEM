"""Small, dependency-light explainability helpers for the V2 prototype."""
from __future__ import annotations
import numpy as np

def feature_importance(model, features: list[str]) -> list[dict]:
    values = getattr(model, "feature_importances_", None)
    if values is None: raise ValueError("Model does not expose feature_importances_")
    return [{"feature": f, "importance": round(float(v), 6)} for f, v in sorted(zip(features, values), key=lambda x: x[1], reverse=True)]

def explain_student(model, feature_row: dict, features: list[str], top_n: int = 3) -> str:
    """Return a local, plain-language heuristic based on feature importance and feature level."""
    ranked = feature_importance(model, features)
    positive = []
    for item in ranked:
        value = float(feature_row.get(item["feature"], 0))
        if value >= 0 and len(positive) < top_n:
            positive.append(item["feature"].replace("_", " "))
    return f"The strongest model signals for this estimate are {', '.join(positive)}. Their influence is based on this model's learned feature importance, not a causal claim."

def local_contributors(model, feature_row: dict, features: list[str], top_n: int = 3) -> list[str]:
    return [x["feature"] for x in feature_importance(model, features)[:top_n]]
