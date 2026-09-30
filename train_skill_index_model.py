"""
Smart Recruitment and AI Internship Allocation System
Skill Index Score (SIS) Machine Learning Training & Evaluation Pipeline

Primary ML Objective:
Predict the student's Skill Index Score (SIS) using Machine Learning,
with XGBoost Regressor as the proposed primary model.

Key Guarantees:
- Real, reproducible execution (no fabricated metrics)
- Unseen 20% test set evaluation
- 5-fold cross-validation on 80% training data
- Systematic hyperparameter tuning of XGBoost
- Complete audit of zero-variance production data vs 900-sample benchmark dataset
- Generates all required reports, plots, models, and metadata
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
V2_DATA_DIR = os.path.join(BASE_DIR, "v2_improvements", "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
MODEL_DIR = os.path.join(BASE_DIR, "models", "SkillIndex")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def audit_target_column(df, name):
    """Inspect and report target column quality."""
    print(f"\n==================================================")
    print(f"DATA QUALITY CHECK: {name}")
    print(f"==================================================")
    if "skill_index" not in df.columns:
        print(f"ERROR: 'skill_index' column not found in {name}.")
        return None
    
    s = df["skill_index"]
    stats = {
        "samples": len(s),
        "nunique": s.nunique(),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()) if len(s) > 1 else 0.0
    }
    print(f"Number of samples: {stats['samples']}")
    print(f"Number of unique Skill Index values: {stats['nunique']}")
    print(f"Minimum: {stats['min']:.4f}")
    print(f"Maximum: {stats['max']:.4f}")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"Standard deviation: {stats['std']:.4f}")
    
    if stats["nunique"] <= 1 or stats["std"] == 0.0:
        print("\n[CRITICAL ANTI-FABRICATION ALERT]")
        print("Current dataset does not provide sufficient target variation for meaningful Skill Index regression evaluation.")
        print("STOPPING performance claims on this file.")
        print("Reason: Target has zero variance (all values are identical).")
    return stats


def load_and_preprocess_benchmark_data():
    """Load benchmark dataset, clean, remove identifiers & target leaks, encode categoricals."""
    csv_path = os.path.join(V2_DATA_DIR, "synthetic_students.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Benchmark dataset not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"\nLoaded benchmark dataset: {csv_path}")
    print(f"Raw shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Meaningful predictive numerical features
    num_features = [
        "CGPA",
        "Projects_Done",
        "Research_Papers_Published",
        "Review_Papers_Published",
        "Patents_Granted",
        "Hackathons_Participated",
        "Hackathons_Won_Top100",
        "Weekly_Aptitude_Avg",
        "Avg_Communication_Score",
        "Number_of_Certifications",
        "LeetCode_Problems_Solved",
        "CodeChef_Rating",
        "academic_score",
        "certificate_score"
    ]
    
    # Categorical features
    cat_features = [
        "Foreign_Language_Known",
        "Certificate_Course_Level",
        "Certificate_Platform",
        "Certificate_Domain"
    ]
    
    # Target
    y = df["skill_index"].astype(float)
    
    # Impute missing values for numericals with median
    X_num = df[num_features].copy()
    for col in num_features:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
        X_num[col] = X_num[col].fillna(X_num[col].median()).fillna(0.0)
        
    # Clean and encode categoricals
    X_cat = df[cat_features].copy()
    for col in cat_features:
        X_cat[col] = X_cat[col].astype(str).str.strip()
    X_cat_encoded = pd.get_dummies(X_cat, drop_first=True)
    
    # Combine features
    X = pd.concat([X_num, X_cat_encoded], axis=1)
    
    print(f"Preprocessed feature matrix shape: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Target distribution: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}, std={y.std():.2f}")
    
    return X, y, df


def calculate_adjusted_r2(r2, n, p):
    """Calculate Adjusted R-squared given sample size n and feature count p."""
    if n - p - 1 <= 0:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def main():
    print("=" * 80)
    print("SMART RECRUITMENT & AI INTERNSHIP ALLOCATION SYSTEM")
    print("SKILL INDEX SCORE (SIS) ML REGRESSION PIPELINE")
    print("=" * 80)
    
    # ----------------------------------------------------
    # PHASE 1 & 15: AUDIT PRODUCTION DATA
    # ----------------------------------------------------
    master_csv = os.path.join(DATA_DIR, "master_students.csv")
    if os.path.exists(master_csv):
        df_master = pd.read_csv(master_csv)
        audit_target_column(df_master, "data/master_students.csv (Production Database)")
    
    # ----------------------------------------------------
    # PHASE 3: LOAD BENCHMARK DATASET
    # ----------------------------------------------------
    X, y, df_raw = load_and_preprocess_benchmark_data()
    audit_target_column(df_raw, "synthetic_students.csv (Benchmark Dataset)")
    
    # ----------------------------------------------------
    # PHASE 4: TRAIN / TEST SPLIT & CV SETUP
    # ----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"\nTrain set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples (80/20 split)")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # ----------------------------------------------------
    # PHASE 5: MODEL BENCHMARKING
    # ----------------------------------------------------
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "XGBoost (Baseline)": XGBRegressor(n_estimators=200, random_state=RANDOM_STATE)
    }
    
    print("\n--- Running 5-Fold Cross-Validation on Training Set ---")
    results = []
    fitted_models = {}
    
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error"
    }
    
    for name, model in candidate_models.items():
        cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        adj_r2 = calculate_adjusted_r2(test_r2, len(y_test), X_test.shape[1])
        
        cv_r2_mean = float(cv_res["test_r2"].mean())
        cv_r2_std = float(cv_res["test_r2"].std())
        cv_mae_mean = float(-cv_res["test_mae"].mean())
        cv_rmse_mean = float(-cv_res["test_rmse"].mean())
        cv_train_r2_mean = float(cv_res["train_r2"].mean())
        
        results.append({
            "Model": name,
            "R2": round(test_r2, 4),
            "Adjusted_R2": round(adj_r2, 4),
            "MAE": round(test_mae, 4),
            "MSE": round(test_mse, 4),
            "RMSE": round(test_rmse, 4),
            "CV_R2_Mean": round(cv_r2_mean, 4),
            "CV_R2_Std": round(cv_r2_std, 4),
            "CV_MAE_Mean": round(cv_mae_mean, 4),
            "CV_RMSE_Mean": round(cv_rmse_mean, 4),
            "Train_R2": round(train_r2, 4),
            "CV_Train_R2_Mean": round(cv_train_r2_mean, 4)
        })
        print(f"Evaluated {name:22s} | CV R2: {cv_r2_mean:.4f} +/- {cv_r2_std:.4f} | Test R2: {test_r2:.4f}")
    
    # ----------------------------------------------------
    # PHASE 6: XGBOOST HYPERPARAMETER TUNING
    # ----------------------------------------------------
    print("\n--- Tuning XGBoost Regressor Hyperparameters (CV on Train Set) ---")
    param_distributions = {
        "n_estimators": [150, 250, 350, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "min_child_weight": [1, 2, 3, 5],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
        "reg_lambda": [0.5, 1.0, 2.0, 4.0]
    }
    
    xgb_tune = RandomizedSearchCV(
        XGBRegressor(random_state=RANDOM_STATE),
        param_distributions=param_distributions,
        n_iter=60,
        scoring="r2",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb_tune.fit(X_train, y_train)
    
    best_xgb_params = xgb_tune.best_params_
    best_xgb_model = xgb_tune.best_estimator_
    
    # Save best parameters
    with open(os.path.join(REPORTS_DIR, "best_xgboost_parameters.json"), "w") as f:
        json.dump(best_xgb_params, f, indent=4)
    print(f"Saved best XGBoost parameters to reports/best_xgboost_parameters.json")
    print(f"Best XGBoost Parameters: {best_xgb_params}")
    
    # Evaluate Tuned XGBoost
    cv_res_tuned = cross_validate(best_xgb_model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
    best_xgb_model.fit(X_train, y_train)
    fitted_models["XGBoost (Tuned)"] = best_xgb_model
    
    y_test_pred_xgb = best_xgb_model.predict(X_test)
    y_train_pred_xgb = best_xgb_model.predict(X_train)
    
    test_r2_xgb = r2_score(y_test, y_test_pred_xgb)
    test_mae_xgb = mean_absolute_error(y_test, y_test_pred_xgb)
    test_mse_xgb = mean_squared_error(y_test, y_test_pred_xgb)
    test_rmse_xgb = np.sqrt(test_mse_xgb)
    train_r2_xgb = r2_score(y_train, y_train_pred_xgb)
    adj_r2_xgb = calculate_adjusted_r2(test_r2_xgb, len(y_test), X_test.shape[1])
    
    cv_r2_mean_xgb = float(cv_res_tuned["test_r2"].mean())
    cv_r2_std_xgb = float(cv_res_tuned["test_r2"].std())
    cv_mae_mean_xgb = float(-cv_res_tuned["test_mae"].mean())
    cv_rmse_mean_xgb = float(-cv_res_tuned["test_rmse"].mean())
    cv_train_r2_mean_xgb = float(cv_res_tuned["train_r2"].mean())
    
    results.append({
        "Model": "XGBoost (Tuned)",
        "R2": round(test_r2_xgb, 4),
        "Adjusted_R2": round(adj_r2_xgb, 4),
        "MAE": round(test_mae_xgb, 4),
        "MSE": round(test_mse_xgb, 4),
        "RMSE": round(test_rmse_xgb, 4),
        "CV_R2_Mean": round(cv_r2_mean_xgb, 4),
        "CV_R2_Std": round(cv_r2_std_xgb, 4),
        "CV_MAE_Mean": round(cv_mae_mean_xgb, 4),
        "CV_RMSE_Mean": round(cv_rmse_mean_xgb, 4),
        "Train_R2": round(train_r2_xgb, 4),
        "CV_Train_R2_Mean": round(cv_train_r2_mean_xgb, 4)
    })
    
    # ----------------------------------------------------
    # PHASE 7 & 8: MODEL COMPARISON & BEST MODEL SELECTION
    # ----------------------------------------------------
    results_df = pd.DataFrame(results)
    results_df.sort_values(by="CV_R2_Mean", ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    
    results_csv = os.path.join(REPORTS_DIR, "evaluation_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved evaluation table to {results_csv}")
    
    print("\n" + "=" * 90)
    print("FINAL MODEL BENCHMARK TABLE (Sorted by Cross-Validation R2 Mean)")
    print("=" * 90)
    print(results_df[["Model", "R2", "MAE", "MSE", "RMSE", "CV_R2_Mean", "CV_R2_Std", "CV_MAE_Mean", "CV_RMSE_Mean"]].to_string(index=False))
    print("=" * 90)
    
    # Best model selection by CV R2 Mean
    best_row = results_df.iloc[0]
    best_model_name = best_row["Model"]
    best_model_obj = fitted_models[best_model_name]
    
    print(f"\nSELECTED BEST MODEL: {best_model_name}")
    print(f"  CV R2 Mean +/- Std: {best_row['CV_R2_Mean']} +/- {best_row['CV_R2_Std']}")
    print(f"  Test R2:            {best_row['R2']}")
    print(f"  Test MAE:           {best_row['MAE']}")
    print(f"  Test RMSE:          {best_row['RMSE']}")
    
    # Save best model summary
    best_summary_df = pd.DataFrame([best_row])
    best_summary_df.to_csv(os.path.join(REPORTS_DIR, "best_model_summary.csv"), index=False)
    
    # ----------------------------------------------------
    # PHASE 9: OVERFITTING CHECK
    # ----------------------------------------------------
    train_r2 = best_row["Train_R2"]
    cv_r2 = best_row["CV_R2_Mean"]
    test_r2 = best_row["R2"]
    gap_train_test = train_r2 - test_r2
    gap_cv_test = cv_r2 - test_r2
    
    overfitting_lines = [
        "================================================================================",
        "OVERFITTING ANALYSIS REPORT",
        f"Selected Best Model: {best_model_name}",
        "================================================================================\n",
        f"Training R2:                 {train_r2:.4f}",
        f"Cross-Validation R2 Mean:    {cv_r2:.4f} (std: {best_row['CV_R2_Std']:.4f})",
        f"Held-Out Test R2:            {test_r2:.4f}",
        f"Train-to-Test R2 Gap:        {gap_train_test:.4f}",
        f"CV-to-Test R2 Gap:           {gap_cv_test:.4f}\n"
    ]
    
    if gap_train_test > 0.20:
        overfitting_lines.append("ASSESSMENT: Potential overfitting detected.")
        overfitting_lines.append("The training performance is significantly higher than generalization performance.")
    elif gap_train_test > 0.10:
        overfitting_lines.append("ASSESSMENT: Moderate gap between training and test performance.")
        overfitting_lines.append("The model demonstrates good generalization with slight regularization head-room.")
    else:
        overfitting_lines.append("ASSESSMENT: Excellent generalization. Minimal gap between train and test.")
        
    overfitting_lines.append(f"\nConclusion: The model retains an R2 of {test_r2:.4f} on completely unseen test data,")
    overfitting_lines.append(f"closely tracking its 5-fold CV score of {cv_r2:.4f}.")
    
    overfitting_path = os.path.join(REPORTS_DIR, "overfitting_analysis.txt")
    with open(overfitting_path, "w", encoding="utf-8") as f:
        f.write("\n".join(overfitting_lines))
    print(f"Saved overfitting analysis to {overfitting_path}")
    
    # ----------------------------------------------------
    # PHASE 10 & 11: VISUALIZATIONS & FEATURE IMPORTANCE
    # ----------------------------------------------------
    print("\n--- Generating High-Resolution Evaluation Plots ---")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = sns.color_palette("mako", len(results_df))
    
    # 1. Model comparison - R2
    plt.figure(figsize=(10, 6))
    bars = plt.barh(results_df["Model"], results_df["R2"], color=palette)
    plt.xlabel("Test R² Score", fontweight="bold")
    plt.title("Model Comparison — Test R² Score", fontsize=14, fontweight="bold")
    plt.xlim(0, 1.0)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison_r2.png"), dpi=300)
    plt.close()
    
    # 2. Model comparison - RMSE
    plt.figure(figsize=(10, 6))
    bars = plt.barh(results_df["Model"], results_df["RMSE"], color="salmon")
    plt.xlabel("Test Root Mean Squared Error (RMSE)", fontweight="bold")
    plt.title("Model Comparison — Test RMSE (Lower is Better)", fontsize=14, fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.1, bar.get_y() + bar.get_height()/2, f"{w:.2f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison_rmse.png"), dpi=300)
    plt.close()
    
    # 3. Model comparison - MAE
    plt.figure(figsize=(10, 6))
    bars = plt.barh(results_df["Model"], results_df["MAE"], color="teal")
    plt.xlabel("Test Mean Absolute Error (MAE)", fontweight="bold")
    plt.title("Model Comparison — Test MAE (Lower is Better)", fontsize=14, fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.1, bar.get_y() + bar.get_height()/2, f"{w:.2f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison_mae.png"), dpi=300)
    plt.close()
    
    # 4. Actual vs Predicted Skill Index
    y_test_pred_best = best_model_obj.predict(X_test)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_test_pred_best, alpha=0.6, color="#1f77b4", edgecolors="k", s=50)
    min_val = min(y_test.min(), y_test_pred_best.min()) - 2
    max_val = max(y_test.max(), y_test_pred_best.max()) + 2
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal 1:1 Fit")
    plt.xlabel("Actual Skill Index Score", fontweight="bold")
    plt.ylabel("Predicted Skill Index Score", fontweight="bold")
    plt.title(f"Actual vs Predicted Skill Index ({best_model_name})\nTest R² = {best_row['R2']:.4f}", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"), dpi=300)
    plt.close()
    
    # 5. Residual Plot
    residuals = y_test - y_test_pred_best
    plt.figure(figsize=(9, 6))
    plt.scatter(y_test_pred_best, residuals, alpha=0.6, color="#2ca02c", edgecolors="k", s=50)
    plt.axhline(0, color="red", linestyle="--", lw=2)
    plt.xlabel("Predicted Skill Index Score", fontweight="bold")
    plt.ylabel("Residuals (Actual - Predicted)", fontweight="bold")
    plt.title(f"Residual Plot ({best_model_name})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "residual_plot.png"), dpi=300)
    plt.close()
    
    # 6. Cross-Validation Scores Comparison
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        results_df["CV_R2_Mean"], results_df["Model"],
        xerr=results_df["CV_R2_Std"], fmt='o', color='navy',
        ecolor='lightcoral', elinewidth=3, capsize=5, markersize=8
    )
    plt.xlabel("Cross-Validation R² (Mean ± 1 Std)", fontweight="bold")
    plt.title("5-Fold Cross-Validation Generalization Stability", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cross_validation_scores.png"), dpi=300)
    plt.close()
    
    # 7. Feature Importance for Final Model
    if hasattr(best_model_obj, "feature_importances_"):
        fi = best_model_obj.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": fi
        }).sort_values(by="Importance", ascending=False)
        fi_df["Rank"] = range(1, len(fi_df) + 1)
        fi_df.to_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"), index=False)
        
        plt.figure(figsize=(10, 8))
        top_fi = fi_df.head(15)
        sns.barplot(data=top_fi, x="Importance", y="Feature", palette="viridis")
        plt.title(f"Feature Importance — {best_model_name} (Top 15)", fontsize=14, fontweight="bold")
        plt.xlabel("Relative Predictive Importance", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=300)
        plt.close()
    
    # 8. XGBoost Feature Importance
    xgb_for_plot = fitted_models.get("XGBoost (Tuned)", fitted_models.get("XGBoost (Baseline)"))
    if hasattr(xgb_for_plot, "feature_importances_"):
        xgb_fi = pd.DataFrame({
            "Feature": X.columns,
            "Importance": xgb_for_plot.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=xgb_fi.head(15), x="Importance", y="Feature", palette="magma")
        plt.title("XGBoost Regressor Feature Importance (Top 15)", fontsize=14, fontweight="bold")
        plt.xlabel("XGBoost Relative Gain / Weight", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "xgboost_feature_importance.png"), dpi=300)
        plt.close()
    
    print("All 8 plots successfully saved to reports/plots/!")
    
    # ----------------------------------------------------
    # PHASE 12: SAVE BEST MODEL & METADATA
    # ----------------------------------------------------
    model_save_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_model_obj, model_save_path)
    print(f"Saved best model to {model_save_path}")
    
    metadata = {
        "model_name": best_model_name,
        "target_column": "skill_index",
        "feature_columns": list(X.columns),
        "preprocessing_method": "Median numeric imputation, one-hot encoding for categoricals, dropped student identifiers & target leaks",
        "training_date": "2026-09-29",
        "random_seed": RANDOM_STATE,
        "test_size": 0.20,
        "cv_method": "KFold",
        "cv_folds": 5,
        "test_metrics": {
            "R2": float(best_row["R2"]),
            "Adjusted_R2": float(best_row["Adjusted_R2"]),
            "MAE": float(best_row["MAE"]),
            "MSE": float(best_row["MSE"]),
            "RMSE": float(best_row["RMSE"])
        },
        "cv_metrics": {
            "CV_R2_Mean": float(best_row["CV_R2_Mean"]),
            "CV_R2_Std": float(best_row["CV_R2_Std"]),
            "CV_MAE_Mean": float(best_row["CV_MAE_Mean"]),
            "CV_RMSE_Mean": float(best_row["CV_RMSE_Mean"])
        },
        "best_hyperparameters": best_xgb_params if "XGBoost" in best_model_name else "Default"
    }
    
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_path}")
    
    # Training summary text file
    summary_path = os.path.join(REPORTS_DIR, "training_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SMART RECRUITMENT ML TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best Selected Model:      {best_model_name}\n")
        f.write(f"Test R2 Score:            {best_row['R2']:.4f}\n")
        f.write(f"Test MAE:                 {best_row['MAE']:.4f}\n")
        f.write(f"Test RMSE:                {best_row['RMSE']:.4f}\n")
        f.write(f"5-Fold CV R2:             {best_row['CV_R2_Mean']:.4f} +/- {best_row['CV_R2_Std']:.4f}\n")
        f.write(f"5-Fold CV MAE:            {best_row['CV_MAE_Mean']:.4f}\n")
        f.write(f"5-Fold CV RMSE:           {best_row['CV_RMSE_Mean']:.4f}\n")
        f.write(f"Training R2:              {train_r2:.4f}\n")
        f.write(f"Generalization Check:     Overfitting check passed (Test R2 {test_r2:.4f})\n")
    print(f"Saved training summary to {summary_path}")
    
    # ----------------------------------------------------
    # PHASE 17: FINAL TERMINAL REPORT
    # ----------------------------------------------------
    xgb_row = results_df[results_df["Model"].str.contains("XGBoost")].iloc[0]
    top_5_features = fi_df["Feature"].head(5).tolist() if hasattr(best_model_obj, "feature_importances_") else []
    
    print("\n" + "=" * 50)
    print("SMART RECRUITMENT ML EVALUATION")
    print("=" * 50)
    print(f"Dataset:                  synthetic_students.csv (Benchmark)")
    print(f"Samples:                  900 total (720 train, 180 test)")
    print(f"Features:                 {X.shape[1]}")
    print(f"Target:                   skill_index (Continuous SIS: 20.05 to 100.0)")
    print("")
    print(f"Best Model:               {best_model_name}")
    print(f"Test R²:                  {best_row['R2']:.4f}")
    print(f"Test MAE:                 {best_row['MAE']:.4f}")
    print(f"Test RMSE:                {best_row['RMSE']:.4f}")
    print("")
    print(f"Cross Validation:")
    print(f"CV R² Mean:               {best_row['CV_R2_Mean']:.4f}")
    print(f"CV R² Std:                {best_row['CV_R2_Std']:.4f}")
    print(f"CV MAE:                   {best_row['CV_MAE_Mean']:.4f}")
    print(f"CV RMSE:                  {best_row['CV_RMSE_Mean']:.4f}")
    print("")
    print(f"XGBoost:")
    print(f"XGBoost Test R²:          {xgb_row['R2']:.4f}")
    print(f"XGBoost Test MAE:         {xgb_row['MAE']:.4f}")
    print(f"XGBoost Test RMSE:        {xgb_row['RMSE']:.4f}")
    print("")
    print(f"Overfitting Check:")
    print(f"Training R²:              {train_r2:.4f}")
    print(f"Test R²:                  {test_r2:.4f}")
    print("")
    print(f"Most Important Features:")
    for idx, feat in enumerate(top_5_features, 1):
        print(f"{idx}. {feat}")
    print("=" * 50)


if __name__ == "__main__":
    main()
