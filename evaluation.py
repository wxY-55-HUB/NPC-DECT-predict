from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             mean_absolute_error, mean_squared_error, roc_auc_score,
                             roc_curve)

def youden_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, probability)
    finite = np.isfinite(thresholds)
    index = np.argmax((tpr - fpr)[finite])
    return float(thresholds[finite][index])

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x); ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float); out[order] = ranks
    return out

def _fast_delong(predictions_sorted: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = label_1_count; n = predictions_sorted.shape[1] - m
    positive = predictions_sorted[:, :m]; negative = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(positive[r]); ty[r] = _compute_midrank(negative[r]); tz[r] = _compute_midrank(predictions_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    if k == 1:
        sx = np.array([[float(sx)]]); sy = np.array([[float(sy)]])
    return aucs, sx / m + sy / n

def delong_auc_ci(y: np.ndarray, probability: np.ndarray, alpha: float = .95) -> tuple[float, float, float, float]:
    y = np.asarray(y, dtype=int); probability = np.asarray(probability, dtype=float)
    order = np.argsort(-y); label_1_count = int(y.sum())
    aucs, covariance = _fast_delong(probability[None, order], label_1_count)
    auc = float(aucs[0]); variance = float(covariance[0, 0]); se = math.sqrt(max(variance, 0))
    z = stats.norm.ppf(1 - (1 - alpha) / 2)
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se), se

def delong_paired_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int); order = np.argsort(-y)
    aucs, covariance = _fast_delong(np.vstack([p1, p2])[:, order], int(y.sum()))
    contrast = np.array([1.0, -1.0]); variance = float(contrast @ covariance @ contrast)
    z = float((aucs[0] - aucs[1]) / math.sqrt(variance)) if variance > 0 else np.nan
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "auc_difference": float(aucs[0] - aucs[1]), "z": z, "p_value": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan}

def calibration_stats(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    import statsmodels.api as sm
    import warnings
    eps = 1e-6; logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.Logit(y, sm.add_constant(logit)).fit(disp=0)
        return float(model.params[0]), float(model.params[1])
    except Exception:
        return np.nan, np.nan

def platt_calibrate_oof_predictions(
    oof: pd.DataFrame,
    temporal: pd.DataFrame,
    seed: int,
    folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if oof.empty:
        return oof.copy(), temporal.copy(), pd.DataFrame()
    keys = [c for c in ("analysis_id", "strategy", "algorithm") if c in oof.columns]
    calibrated_oof_parts: list[pd.DataFrame] = []
    calibrated_temporal_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []

    def _fit_calibrator(y_fit: np.ndarray, p_fit: np.ndarray) -> LogisticRegression:
        x_fit = np.log(np.clip(p_fit, 1e-6, 1 - 1e-6) /
                       np.clip(1 - p_fit, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=seed)
        model.fit(x_fit, y_fit.astype(int))
        return model

    for group_key, dev in oof.groupby(keys, dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        identity = dict(zip(keys, group_key))
        dev = dev.copy().reset_index(drop=True)
        y = dev["y_true"].to_numpy(dtype=int)
        p = dev["probability"].to_numpy(dtype=float)
        x = np.log(np.clip(p, 1e-6, 1 - 1e-6) /
                   np.clip(1 - p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        calibrated_dev = np.full(len(dev), np.nan, dtype=float)

        if len(np.unique(y)) < 2:
            calibrated_dev[:] = p
            final_cal = None
            intercept = np.nan
            slope = np.nan
            status = "skipped_single_class"
        else:
            n_splits = min(int(folds), int(np.bincount(y).min()))
            if n_splits >= 2:
                from sklearn.model_selection import StratifiedKFold
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                for fit_idx, holdout_idx in cv.split(x, y):
                    cal = _fit_calibrator(y[fit_idx], p[fit_idx])
                    calibrated_dev[holdout_idx] = cal.predict_proba(x[holdout_idx])[:, 1]
            else:
                calibrated_dev[:] = p
            final_cal = _fit_calibrator(y, p)
            calibrated_dev = np.where(np.isfinite(calibrated_dev), calibrated_dev, p)
            intercept = float(final_cal.intercept_[0])
            slope = float(final_cal.coef_[0, 0])
            status = "cross_fitted_development_oof"

        dev_out = dev.copy()
        dev_out["probability_raw"] = dev_out["probability"]
        dev_out["probability"] = calibrated_dev
        dev_out["calibration_method"] = "platt_logistic"
        dev_out["calibration_fit_cohort"] = "development_oof"
        calibrated_oof_parts.append(dev_out)

        temp = temporal.copy()
        if keys:
            mask = np.ones(len(temp), dtype=bool)
            for key, value in identity.items():
                mask &= temp[key].eq(value).to_numpy()
            temp = temp.loc[mask].copy()
        if not temp.empty:
            temp_out = temp.copy()
            temp_out["probability_raw"] = temp_out["probability"]
            if final_cal is None:
                temp_out["probability"] = temp_out["probability_raw"]
            else:
                p_temp = temp_out["probability_raw"].to_numpy(dtype=float)
                x_temp = np.log(np.clip(p_temp, 1e-6, 1 - 1e-6) /
                                np.clip(1 - p_temp, 1e-6, 1 - 1e-6)).reshape(-1, 1)
                temp_out["probability"] = final_cal.predict_proba(x_temp)[:, 1]
            temp_out["calibration_method"] = "platt_logistic"
            temp_out["calibration_fit_cohort"] = "development_oof"
            calibrated_temporal_parts.append(temp_out)

        parameter_rows.append({**identity, "calibration_method": "platt_logistic",
                               "fit_cohort": "development_oof",
                               "development_n": int(len(dev)),
                               "development_positive_n": int(y.sum()),
                               "calibration_intercept": intercept,
                               "calibration_slope": slope,
                               "development_assessment": status})

    return (
        pd.concat(calibrated_oof_parts, ignore_index=True) if calibrated_oof_parts else pd.DataFrame(),
        pd.concat(calibrated_temporal_parts, ignore_index=True) if calibrated_temporal_parts else pd.DataFrame(),
        pd.DataFrame(parameter_rows),
    )

def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float); pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc, low, high, se = delong_auc_ci(y, p)
    intercept, slope = calibration_stats(y, p)
    return {
        "n": len(y), "positive_n": int(y.sum()), "negative_n": int((1-y).sum()),
        "threshold": threshold, "AUC": auc, "AUC_CI_low": low, "AUC_CI_high": high, "AUC_SE_DeLong": se,
        "accuracy": accuracy_score(y, pred), "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan, "PPV": tp / (tp + fp) if tp + fp else np.nan,
        "NPV": tn / (tn + fn) if tn + fn else np.nan, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Brier": brier_score_loss(y, p), "calibration_intercept": intercept, "calibration_slope": slope,
    }

def stratified_bootstrap_metrics(y: np.ndarray, p: np.ndarray, threshold: float, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed); y = np.asarray(y, int); p = np.asarray(p, float)
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0); rows = []
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        m = binary_metrics(y[idx], p[idx], threshold)
        m["bootstrap_iteration"] = i; rows.append(m)
    return pd.DataFrame(rows)

def bootstrap_summary(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        values = pd.to_numeric(table[col], errors="coerce").dropna()
        rows.append({"metric": col, "bootstrap_median": values.median(), "bootstrap_CI_low": values.quantile(.025), "bootstrap_CI_high": values.quantile(.975)})
    return pd.DataFrame(rows)

def paired_bootstrap_difference(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed); y = np.asarray(y, int); pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    return {"bootstrap_auc_difference": float(np.mean(diffs)), "bootstrap_CI_low": float(np.quantile(diffs, .025)), "bootstrap_CI_high": float(np.quantile(diffs, .975))}

def evaluate_prediction_sets(oof: pd.DataFrame, temporal: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = []; bootstrap_summaries = []; classified = []
    n_boot = int(cfg["validation"]["bootstrap_iterations"]); seed = int(cfg["project"]["random_seed"])
    keys = [c for c in ["analysis_id", "strategy", "algorithm"] if c in oof.columns]
    for group_key, dev in oof.groupby(keys, dropna=False):
        if not isinstance(group_key, tuple): group_key = (group_key,)
        identity = dict(zip(keys, group_key))
        threshold = youden_threshold(dev["y_true"].to_numpy(), dev["probability"].to_numpy())
        for cohort, frame in (("development_repeated_nested_oof", dev), ("internal_temporal_validation", temporal.merge(dev[keys].drop_duplicates(), on=keys, how="inner") if keys else temporal)):
            if frame.empty or frame["y_true"].isna().any() or frame["y_true"].nunique() < 2:
                continue
            result = {**identity, "cohort": cohort, **binary_metrics(frame["y_true"].to_numpy(), frame["probability"].to_numpy(), threshold)}
            metrics.append(result)
            boot = stratified_bootstrap_metrics(frame["y_true"].to_numpy(), frame["probability"].to_numpy(), threshold, n_boot, seed)
            summary = bootstrap_summary(boot, ["accuracy", "sensitivity", "specificity", "PPV", "NPV", "Brier", "calibration_intercept", "calibration_slope"])
            for col, value in identity.items(): summary[col] = value
            summary["cohort"] = cohort; bootstrap_summaries.append(summary)
            temp = frame.copy(); temp["threshold"] = threshold; temp["predicted_class"] = (temp["probability"] >= threshold).astype(int); temp["cohort"] = cohort
            classified.append(temp)
    return pd.DataFrame(metrics), pd.concat(bootstrap_summaries, ignore_index=True) if bootstrap_summaries else pd.DataFrame(), pd.concat(classified, ignore_index=True) if classified else pd.DataFrame()

def continuous_metrics(predictions: pd.DataFrame, cohort: str) -> dict[str, float | int | str]:
    y = predictions["y_true"].to_numpy(float); p = predictions["prediction"].to_numpy(float)
    rho = stats.spearmanr(y, p).statistic
    return {"analysis_id": predictions["analysis_id"].iloc[0], "cohort": cohort, "n": len(y), "Spearman_rho": float(rho), "MAE": mean_absolute_error(y, p), "RMSE": math.sqrt(mean_squared_error(y, p))}

def decision_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y, int); n = len(y); prevalence = y.mean(); rows = []
    for threshold in np.arange(.05, .96, .01):
        pred = p >= threshold; tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        weight = threshold / (1 - threshold)
        rows.append({"threshold": threshold, "net_benefit_model": tp / n - fp / n * weight,
                     "net_benefit_all": prevalence - (1 - prevalence) * weight, "net_benefit_none": 0.0})
    return pd.DataFrame(rows)

def compare_models(common: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    names = sorted(common["analysis_id"].unique()); rows = []
    for a, b in itertools.combinations(names, 2):
        x = common[common["analysis_id"] == a][["PatientID", "y_true", "probability"]].rename(columns={"probability": "p1"})
        z = common[common["analysis_id"] == b][["PatientID", "probability"]].rename(columns={"probability": "p2"})
        paired = x.merge(z, on="PatientID", how="inner")
        if len(paired) < 4 or paired["y_true"].nunique() < 2: continue
        result = {"model_1": a, "model_2": b, "n": len(paired), **delong_paired_test(paired["y_true"].to_numpy(), paired["p1"].to_numpy(), paired["p2"].to_numpy()), **paired_bootstrap_difference(paired["y_true"].to_numpy(), paired["p1"].to_numpy(), paired["p2"].to_numpy(), n_boot, seed)}
        rows.append(result)
    return pd.DataFrame(rows)

import itertools
