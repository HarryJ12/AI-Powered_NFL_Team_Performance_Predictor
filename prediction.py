import os
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, f1_score, brier_score_loss, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Load CSV into DataFrame; return None on failure.
def load_data(path: str = "theOne.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError as exc:
        raise ValueError(f"Invalid CSV format: {path}") from exc

# 'possession_time_min' gets converted to seconds for use
# Engineered features 'redzone_efficiency' and 'turnover_impact' are computed
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df
    def _to_minutes(v: object) -> float:
        if pd.isna(v):
            return 0.0
        if isinstance(v, str) and ":" in v:
            try:
                m, s = map(int, v.split(":"))
                return float(m) + float(s) / 60.0
            except Exception:
                return 0.0
        try:
            return float(v)
        except Exception:
            return 0.0
    d["possession_time_min"] = d["possession_time"].apply(_to_minutes)

    d["redzone_efficiency"] = d["redzone_comp"] / d["redzone_att"].clip(lower=1)
    d["turnover_impact"] = d["turnover_diff_pct"] * d["possession_time_min"]

    return d

# Adjust score prediction using win probability (Vegas-style).
def vegas_meta(score_pred: float, win_prob: float) -> float:
    return float(np.clip(score_pred + (win_prob - 0.5) * 2.0, 0, None))

# Train and evaluate regression, logistic, and meta models.
def run_models(df: pd.DataFrame) -> Tuple[dict, dict, dict]:
    df = engineer_features(df.copy())
    
    # REGRESSION MODEL 
    selected_features = [
        "win_numeric",
        "score_against",
        "first_downs",
        "third_down_comp",
        "yards",
        "pass_yards",
        "rush_att",
        "rush_yards",
        "redzone_comp",
        "redzone_att",
        "sacks_num",
        "turnovers_forced",
        "turnover_diff_pct",
        "possession_time_min",
        "redzone_efficiency",
        "third_down_efficiency",
        "yards_per_play",
        "pass_completion_pct",
    ]
    
    regression_target = "score_for"
    
    # Prepare Data
    train_df = df[df["season"].isin([2022, 2023])]
    test_df = df[df["season"] == 2024]
    
    X_train = train_df[selected_features].copy()
    X_test = test_df[selected_features].copy()
    
    y_train = train_df[regression_target].copy()
    y_test = test_df[regression_target].copy()
    
    # # Convert all to numeric and fill NA with median
    # X_train = X_train.apply(pd.to_numeric, errors="coerce")
    # X_test = X_test.apply(pd.to_numeric, errors="coerce")
    # for col in X_train.columns:
    #     med = X_train[col].median()
    #     if np.isnan(med):
    #         med = 0.0
    #     X_train[col] = X_train[col].fillna(med)
    #     X_test[col] = X_test[col].fillna(med)
    
    # Train Model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    # Clip negative predictions
    y_pred_lr = np.clip(y_pred_lr, 0, None)
    
    # Evaluate metrics
    reg_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    reg_r2 = r2_score(y_test, y_pred_lr)
    
    reg_results = {
        "metrics": {"rmse": float(reg_rmse), "r2": float(reg_r2)},
        "model": lr,
        "predictions": y_pred_lr,
        "y_test": y_test,
        "features": selected_features,
    }
    
    # ========== LOGISTIC MODEL ==========
    features = [
        "turnovers_forced",
        "turnover_diff_pct",
        "redzone_comp",
        "redzone_att",
        "possession_time_min",
        "pass_att",
        "rush_att",
        "location",
        "redzone_efficiency",
        "turnover_impact",
    ]
    
    target = "win_numeric"
    
    # Split train and test data
    train_df_clf = df[df["season"].isin([2022, 2023])]
    test_df_clf = df[df["season"] == 2024]
    
    # Get available features
    available_features = [f for f in features if f in train_df_clf.columns]
    
    X_train_clf = train_df_clf[available_features].copy()
    y_train_clf = train_df_clf[target].copy()
    
    X_test_clf = test_df_clf[available_features].copy()
    y_test_clf = test_df_clf[target].copy()
    
    # # Convert all to numeric and fill NA with median
    # X_train_clf = X_train_clf.apply(pd.to_numeric, errors="coerce")
    # X_test_clf = X_test_clf.apply(pd.to_numeric, errors="coerce")
    # for col in X_train_clf.columns:
    #     med = X_train_clf[col].median()
    #     if np.isnan(med):
    #         med = 0.0
    #     X_train_clf[col] = X_train_clf[col].fillna(med)
    #     X_test_clf[col] = X_test_clf[col].fillna(med)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clf)
    X_test_scaled = scaler.transform(X_test_clf)
    
    # Train Model
    lr2 = LogisticRegression(max_iter=5000, class_weight="balanced")
    lr2.fit(X_train_scaled, y_train_clf)
    
    # Get predictions and probabilities
    logi_proba = lr2.predict_proba(X_test_scaled)[:, 1]
    logi_preds = (logi_proba >= 0.5).astype(int)
    
    # Evaluate metrics
    clf_accuracy = accuracy_score(y_test_clf, logi_preds)
    clf_f1 = f1_score(y_test_clf, logi_preds)
    clf_brier = brier_score_loss(y_test_clf, logi_proba)
    clf_roc_auc = roc_auc_score(y_test_clf, logi_proba)
    
    clf_results = {
        "metrics": {"accuracy": float(clf_accuracy), "f1": float(clf_f1), "brier": float(clf_brier), "roc_auc": float(clf_roc_auc)},
        "model": lr2,
        "probabilities": logi_proba,
        "y_test": y_test_clf,
        "scaler": scaler,
        "features": available_features,
    }
    
    # META MODEL 
    
    # Get probability of the positive class (win = 1) from logistic regression
    cl = logi_proba
    
    # Apply meta-model to each (score prediction, win probability) pair
    meta_scores = np.array([vegas_meta(score, prob) for score, prob in zip(y_pred_lr, cl)])
    
    # Evaluate metrics
    meta_rmse = np.sqrt(mean_squared_error(y_test, meta_scores))
    meta_r2 = r2_score(y_test, meta_scores)
    
    meta_results = {
        "metrics": {"rmse": float(meta_rmse), "r2": float(meta_r2)},
        "predictions": meta_scores,
    }
    
    return reg_results, clf_results, meta_results

# Predict a matchup using averaged 3-year stats per team.
def predict_matchup(team1: str, team2: str, df: pd.DataFrame, reg_model, clf_model, scaler, reg_features: list, clf_features: list):
    df = engineer_features(df.copy())
    
    # 3-year history (2022–2024)
    t1_games = df[df["team"] == team1]
    t2_games = df[df["team"] == team2]

    # REGRESSION INPUTS: average numeric features across history
    t1_reg = t1_games[reg_features].apply(pd.to_numeric, errors="coerce").fillna(0).mean().to_frame().T
    t2_reg = t2_games[reg_features].apply(pd.to_numeric, errors="coerce").fillna(0).mean().to_frame().T

    t1_score_raw = float(np.clip(reg_model.predict(t1_reg)[0], 0, None))
    t2_score_raw = float(np.clip(reg_model.predict(t2_reg)[0], 0, None))

    # CLASSIFIER INPUTS: average classifier features
    t1_clf_df = t1_games[clf_features].apply(pd.to_numeric, errors="coerce").fillna(0).mean().to_frame().T
    t2_clf_df = t2_games[clf_features].apply(pd.to_numeric, errors="coerce").fillna(0).mean().to_frame().T

    # Transform using fitted scaler
    t1_scaled = scaler.transform(t1_clf_df)
    t2_scaled = scaler.transform(t2_clf_df)
    t1_win_raw = clf_model.predict_proba(t1_scaled)[0, 1]
    t2_win_raw = clf_model.predict_proba(t2_scaled)[0, 1]

    # META-ADJUSTED SCORES
    t1_score = vegas_meta(t1_score_raw, t1_win_raw)
    t2_score = vegas_meta(t2_score_raw, t2_win_raw)

    # VEGAS SPREAD (WIN PROBABILITY)
    spread = t1_score - t2_score
    t1_prob = float(norm.cdf(spread / 13.86))
    t2_prob = 1 - t1_prob

    return {
        "matchup": f"{team1} vs {team2}",
        "predicted_scores": {team1: round(t1_score, 1), team2: round(t2_score, 1)},
        "win_probabilities": {team1: f"{round(t1_prob * 100):02d}%", team2: f"{round(t2_prob * 100):02d}%"},
    }
