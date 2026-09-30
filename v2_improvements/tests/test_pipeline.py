from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_synthetic_data import generate_synthetic_data
from preprocessing import CORE_FEATURES, prepare_features
from evaluate import evaluate_regressor
from sklearn.ensemble import RandomForestRegressor
def test_preprocessing_has_clean_core_features():
    frame=generate_synthetic_data(rows=800,seed=7)
    prepared=prepare_features(frame,CORE_FEATURES)
    assert prepared.shape==(800,len(CORE_FEATURES))
    assert prepared.isna().sum().sum()==0
def test_synthetic_targets_are_present():
    frame=generate_synthetic_data(rows=800,seed=8)
    assert {"skill_level","skill_index","scheme_eligible"}.issubset(frame.columns)
    assert frame["skill_level"].nunique() >= 3
def test_evaluation_returns_heldout_regression_metrics(tmp_path, monkeypatch):
    frame=generate_synthetic_data(rows=800,seed=9)
    X=prepare_features(frame,CORE_FEATURES)
    model=RandomForestRegressor(n_estimators=5,random_state=42).fit(X.iloc[:640],frame["skill_index"].iloc[:640])
    import evaluate
    monkeypatch.setattr(evaluate,"PLOTS",tmp_path)
    metrics=evaluate_regressor(model,X.iloc[640:],frame["skill_index"].iloc[640:],CORE_FEATURES)
    assert {"r2","mae","rmse","feature_importance"}.issubset(metrics)
