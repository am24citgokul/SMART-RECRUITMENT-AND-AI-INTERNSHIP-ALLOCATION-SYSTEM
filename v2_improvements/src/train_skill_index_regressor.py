from pathlib import Path
import joblib, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from preprocessing import REGRESSION_FEATURES, prepare_features

ROOT = Path(__file__).resolve().parents[1]
def train(frame: pd.DataFrame):
    X, y = prepare_features(frame, REGRESSION_FEATURES), frame["skill_index"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    model = RandomForestRegressor(n_estimators=350, min_samples_leaf=4, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump({"model": model, "features": REGRESSION_FEATURES}, ROOT / "models" / "skill_index_regressor.joblib")
    return model, X_test, y_test
