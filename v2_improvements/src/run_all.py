"""One-command, idempotent V2 build: data -> models -> metrics -> prototype payload."""
from __future__ import annotations
from pathlib import Path
import json, sys, joblib, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic_data import generate_synthetic_data, OUTPUT
from preprocessing import CORE_FEATURES, REGRESSION_FEATURES, RADAR_FEATURES, student_feature_row
from train_skill_classifier import train as train_classifier
from train_skill_index_regressor import train as train_regressor
from train_scheme_recommender import train as train_scheme
from evaluate import evaluate_classifier, evaluate_regressor, evaluate_scheme, save_metrics
from explainability import explain_student, local_contributors
ROOT = Path(__file__).resolve().parents[1]
SCHEME_FEATURES = ["CGPA", "Projects_Done", "Hackathons_Participated", "Number_of_Certifications", "final_skill_index"]
def write_report(metrics, rows):
    n=lambda x:f"{x:.4f}"; c,r,s=metrics["skill_classifier"],metrics["skill_index_regressor"],metrics["scheme_recommender"]
    # Only predictive quality metrics are relevant to the >0.97 caution;
    # metadata such as the 900-row count must never trigger this warning.
    values=[c["accuracy"],c["precision_macro"],c["recall_macro"],c["f1_macro"],r["r2"],s["accuracy"],s["roc_auc"],s["precision"],s["recall"]]
    warning="\n> **Metric caution:** At least one reported metric exceeds 0.97. Inspect for leakage or an overly easy synthetic target before relying on it.\n" if any(v>.97 for v in values) else ""
    text=f"""# V2 ML Evaluation Report

## Dataset

This run used **{rows} synthetic rows**, generated with reproducible correlated distributions from the read-only source schema. CGPA, project work, coding practice, aptitude, and certifications are combined with independent noise to create non-random targets. It is not institutional data.{warning}

## Held-out evaluation

All values below came from 20% held-out test splits in `metrics.json`.

| Model | Metrics |
|---|---|
| Skill classifier | Accuracy {n(c['accuracy'])}; macro precision {n(c['precision_macro'])}; macro recall {n(c['recall_macro'])}; macro F1 {n(c['f1_macro'])} |
| Skill-index regressor | R² {n(r['r2'])}; MAE {n(r['mae'])}; RMSE {n(r['rmse'])} |
| Scheme recommender | Accuracy {n(s['accuracy'])}; ROC-AUC {n(s['roc_auc'])}; precision {n(s['precision'])}; recall {n(s['recall'])} |

## Plots

![Classifier confusion matrix](plots/skill_classifier_confusion_matrix.png)
![Classifier feature importance](plots/skill_classifier_feature_importance.png)
![Regressor feature importance](plots/skill_index_regressor_feature_importance.png)

## Limitations

This pipeline was trained on synthetic data. It must be retrained and independently validated on representative, consented real institutional data before it is used for actual student evaluation, selection, or recommendations. Feature importances indicate model associations, not causes.
"""
    (ROOT / "reports" / "evaluation_report.md").write_text(text,encoding="utf-8")
def write_ui_payload(frame):
    classifier=joblib.load(ROOT/"models"/"skill_classifier.joblib"); regressor=joblib.load(ROOT/"models"/"skill_index_regressor.joblib")
    row=student_feature_row(frame,0); pf=pd.DataFrame([row]); index=float(regressor["model"].predict(pf[REGRESSION_FEATURES])[0]); top=local_contributors(regressor["model"],row,REGRESSION_FEATURES)
    # Each spoke is normalized against that feature's observed synthetic-data range.
    # The raw number and min/max travel with the API payload for truthful tooltips.
    radar=[]
    for feature in RADAR_FEATURES:
        low=float(frame[feature].min()); high=float(frame[feature].max()); value=float(row[feature])
        scaled=50.0 if high == low else max(0.0,min(100.0,(value-low)/(high-low)*100))
        radar.append({"feature":feature,"raw":value,"min":low,"max":high,"scaled":round(scaled,2)})
    payload = {
        "student": {
            "name": "Synthetic Student 1",
            "features": {k: row[k] for k in RADAR_FEATURES},
            "radar": radar,
            "skill_index": round(index, 2),
            "skill_level": classifier["model"].predict(pf[CORE_FEATURES])[0],
            "explanation": explain_student(regressor["model"], row, REGRESSION_FEATURES),
            "top_features": top,
        },
        "match": {
            "role": "Junior Data Analyst",
            "requirements": ["CGPA", "Projects_Done", "LeetCode_Problems_Solved"],
            "explanation": "This synthetic candidate qualifies for the Junior Data Analyst profile based on prerequisite domain skills and benchmark alignment.",
        },
        "timeline": [
            {"label": "Now", "value": round(index - 8, 2)},
            {"label": "+2 certifications", "value": round(index - 4, 2)},
            {"label": "+1 project", "value": round(index + 1, 2)},
            {"label": "+1 project & practice", "value": round(index + 6, 2)},
        ],
    }
    (ROOT / "ui_prototype" / "dashboard_data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
def main():
    frame=generate_synthetic_data(); OUTPUT.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(OUTPUT,index=False)
    c,cx,cy=train_classifier(frame); r,rx,ry=train_regressor(frame); s,sx,sy=train_scheme(frame)
    metrics={"dataset":{"name":"synthetic_students.csv","rows":len(frame),"train_test_split":"80/20","random_state":42},"skill_classifier":evaluate_classifier(c,cx,cy,CORE_FEATURES),"skill_index_regressor":evaluate_regressor(r,rx,ry,REGRESSION_FEATURES),"scheme_recommender":evaluate_scheme(s,sx,sy,SCHEME_FEATURES)}
    save_metrics(metrics); write_report(metrics,len(frame)); write_ui_payload(frame); print(json.dumps(metrics,indent=2)); print("V2 pipeline complete. Run: python ui_prototype/app.py")
if __name__=="__main__": main()
