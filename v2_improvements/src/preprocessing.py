"""Shared feature definitions and safe preprocessing for the V2 models."""
from __future__ import annotations

import pandas as pd

CORE_FEATURES = [
    "CGPA", "Hackathons_Participated", "Research_Papers_Published",
    "Projects_Done", "Number_of_Certifications", "certificate_score",
    "academic_score", "LeetCode_Problems_Solved",
]
REGRESSION_FEATURES = CORE_FEATURES + ["CodeChef_Rating", "Weekly_Aptitude_Avg", "Avg_Communication_Score"]
SCHEME_FEATURES = ["CGPA", "Projects_Done", "Hackathons_Participated", "Number_of_Certifications", "final_skill_index"]
RADAR_FEATURES = [
    "CGPA", "Hackathons_Participated", "Research_Papers_Published",
    "Projects_Done", "Number_of_Certifications", "certificate_score",
    "academic_score", "final_skill_index", "LeetCode_Problems_Solved",
]

def prepare_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return finite numeric model features, with median imputation."""
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
    result = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    return result.fillna(result.median()).fillna(0.0)

def student_feature_row(frame: pd.DataFrame, index: int = 0) -> dict:
    """Return one clean feature row suitable for prediction endpoints."""
    features = list(dict.fromkeys(REGRESSION_FEATURES + RADAR_FEATURES))
    return prepare_features(frame.iloc[[index]], features).iloc[0].to_dict()
