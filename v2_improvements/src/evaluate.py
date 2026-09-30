from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, r2_score, mean_absolute_error, mean_squared_error, roc_auc_score, precision_score, recall_score
from explainability import feature_importance

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "reports" / "plots"

def _plot_importance(model, features, filename, title):
    ranked = feature_importance(model, features)
    plt.figure(figsize=(8, 4.8)); plt.barh([x['feature'] for x in ranked][::-1], [x['importance'] for x in ranked][::-1], color="#2563eb")
    plt.title(title); plt.xlabel("Impurity-based importance"); plt.tight_layout(); plt.savefig(PLOTS / filename, dpi=150); plt.close()
    return ranked

def evaluate_classifier(model, X_test, y_test, features):
    predicted = model.predict(X_test); labels = sorted(y_test.unique())
    p, r, f, s = precision_recall_fscore_support(y_test, predicted, labels=labels, zero_division=0)
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, predicted, labels=labels), display_labels=labels)
    disp.plot(cmap="Blues"); plt.tight_layout(); plt.savefig(PLOTS / "skill_classifier_confusion_matrix.png", dpi=150); plt.close()
    return {"accuracy": float(accuracy_score(y_test, predicted)), "precision_macro": float(precision_recall_fscore_support(y_test,predicted,average="macro",zero_division=0)[0]), "recall_macro": float(precision_recall_fscore_support(y_test,predicted,average="macro",zero_division=0)[1]), "f1_macro": float(precision_recall_fscore_support(y_test,predicted,average="macro",zero_division=0)[2]), "per_class": {label: {"precision": float(a), "recall": float(b), "f1": float(c), "support": int(d)} for label,a,b,c,d in zip(labels,p,r,f,s)}, "feature_importance": _plot_importance(model, features, "skill_classifier_feature_importance.png", "Skill classifier feature importance")}

def evaluate_regressor(model, X_test, y_test, features):
    predicted = model.predict(X_test)
    return {"r2": float(r2_score(y_test,predicted)), "mae": float(mean_absolute_error(y_test,predicted)), "rmse": float(mean_squared_error(y_test,predicted)**.5), "feature_importance": _plot_importance(model, features, "skill_index_regressor_feature_importance.png", "Skill index regressor feature importance")}

def evaluate_scheme(model, X_test, y_test, features):
    predicted = model.predict(X_test); probability = model.predict_proba(X_test)[:, 1]
    return {"accuracy": float(accuracy_score(y_test,predicted)), "roc_auc": float(roc_auc_score(y_test,probability)), "precision": float(precision_score(y_test,predicted,zero_division=0)), "recall": float(recall_score(y_test,predicted,zero_division=0)), "feature_importance": _plot_importance(model, features, "scheme_recommender_feature_importance.png", "Scheme recommender feature importance")}

def save_metrics(metrics: dict):
    (ROOT / "reports").mkdir(exist_ok=True); PLOTS.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
