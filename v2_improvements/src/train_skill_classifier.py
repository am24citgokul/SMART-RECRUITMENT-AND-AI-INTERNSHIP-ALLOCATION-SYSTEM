from pathlib import Path
import joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocessing import CORE_FEATURES, prepare_features

ROOT = Path(__file__).resolve().parents[1]
def train(frame: pd.DataFrame):
    X, y = prepare_features(frame, CORE_FEATURES), frame["skill_level"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    joblib.dump({"model": model, "features": CORE_FEATURES}, ROOT / "models" / "skill_classifier.joblib")
    return model, X_test, y_test
