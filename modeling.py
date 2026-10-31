from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, KFold

META_COLUMNS = {"PatientID", "Ki67", "Ki67_ge50", "Ki67_gt50", "ExamDate", "split", "strict_triplicate"}

def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return 0.0
    x_valid, y_valid = x[mask], y[mask]
    if np.all(x_valid == x_valid[0]) or np.all(y_valid == y_valid[0]):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", stats.ConstantInputWarning)
        try:
            value = stats.spearmanr(x_valid, y_valid).statistic
        except stats.ConstantInputWarning:
            return 0.0
    return float(abs(value)) if np.isfinite(value) else 0.0

def _safe_signed_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return 0.0
    x_valid, y_valid = x[mask], y[mask]
    if np.all(x_valid == x_valid[0]) or np.all(y_valid == y_valid[0]):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", stats.ConstantInputWarning)
        try:
            value = stats.spearmanr(x_valid, y_valid).statistic
        except stats.ConstantInputWarning:
            return 0.0
    return float(value) if np.isfinite(value) else 0.0

@dataclass
class FoldPreprocessor:
    missing_rate_max: float = 0.2
    correlation_abs_max: float = 0.9
    max_correlation_candidates: int = 300
    log1p_name_tokens: tuple[str, ...] = ("Volume", "VoxelCount", "Count", "ConnectedComponents")

    def fit(self, X: pd.DataFrame, y: np.ndarray, top_k: int | None = None) -> "FoldPreprocessor":
        work = self._coerce_and_indicators(X)
        all_columns = list(work.columns)
        all_values = work.to_numpy(dtype=float, copy=False)
        finite = np.isfinite(all_values)
        counts = finite.sum(axis=0)
        missing = 1.0 - counts / max(1, all_values.shape[0])
        sums = np.nansum(all_values, axis=0)
        means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)
        squared = np.where(finite, (all_values - means) ** 2, 0.0).sum(axis=0)
        variances = np.divide(squared, counts, out=np.full_like(squared, np.nan, dtype=float), where=counts > 0)
        candidate_indices = np.flatnonzero((missing <= self.missing_rate_max) & np.isfinite(variances) & (variances > 1e-14))
        candidates = [all_columns[i] for i in candidate_indices]
        values = all_values[:, candidate_indices].copy()
        minima = np.nanmin(values, axis=0)
        self.log_columns_ = [c for c, minimum in zip(candidates, minima) if any(token.lower() in c.lower() for token in self.log1p_name_tokens) and np.isfinite(minimum) and minimum >= 0]
        candidate_index = {c: i for i, c in enumerate(candidates)}
        for col in self.log_columns_:
            idx = candidate_index[col]
            values[:, idx] = np.log1p(values[:, idx])
        medians = np.nanmedian(values, axis=0)
        medians[~np.isfinite(medians)] = 0.0
        missing_rows, missing_cols = np.where(~np.isfinite(values))
        values[missing_rows, missing_cols] = medians[missing_cols]
        self.medians_ = pd.Series(medians, index=candidates)
        value_ranks = stats.rankdata(values, axis=0, method="average")
        y_ranks = stats.rankdata(np.asarray(y, float), method="average")
        value_centered = value_ranks - value_ranks.mean(axis=0, keepdims=True)
        y_centered = y_ranks - y_ranks.mean()
        denom = np.sqrt(np.sum(value_centered ** 2, axis=0) * np.sum(y_centered ** 2))
        correlations = np.divide(np.sum(value_centered * y_centered[:, None], axis=0), denom,
                                 out=np.zeros(len(candidates), dtype=float), where=denom > 0)
        scores = {c: float(abs(s)) for c, s in zip(candidates, correlations)}
        ranked = sorted(candidates, key=lambda c: (-scores[c], c))[: self.max_correlation_candidates]
        ranked_indices = [candidates.index(c) for c in ranked]
        ranked_values = value_ranks[:, ranked_indices]
        corr_matrix = np.atleast_2d(np.corrcoef(ranked_values, rowvar=False))
        selected_indices: list[int] = []
        for index, _col in enumerate(ranked):
            if not selected_indices or not np.any(np.abs(corr_matrix[index, selected_indices]) > self.correlation_abs_max):
                selected_indices.append(index)
        selected = [ranked[index] for index in selected_indices]
        if top_k is not None:
            selected = selected[: min(top_k, len(selected))]
        if not selected:
            raise ValueError("No usable features after fold-specific preprocessing")
        self.feature_scores_ = scores
        self.selected_features_ = selected
        selected_frame = pd.DataFrame(values[:, [candidate_index[c] for c in selected]], columns=selected, index=X.index)
        self.centers_ = selected_frame.median(axis=0)
        q1 = selected_frame.quantile(0.25)
        q3 = selected_frame.quantile(0.75)
        scales = q3 - q1
        self.scales_ = scales.where(scales.abs() > 1e-12, 1.0)
        return self

    def _coerce_and_indicators(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            work = X.astype(float, copy=True)
        except (TypeError, ValueError):
            work = X.apply(pd.to_numeric, errors="coerce")
        habitat_regions = sorted({part for col in work.columns for part in str(col).split("__") if part in {"H1", "H2", "H3", "H4"}})
        for region in habitat_regions:
            region_cols = [c for c in work if f"__{region}__" in str(c) or str(c).startswith(f"Habitat__{region}__")]
            if region_cols:
                work[f"HabitatPresence__{region}"] = work[region_cols].notna().any(axis=1).astype(float)
        return work

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        work = self._coerce_and_indicators(X)
        for col in self.selected_features_:
            if col not in work:
                work[col] = np.nan
        work = work[self.selected_features_].copy()
        for col in self.log_columns_:
            if col in work:
                values = pd.to_numeric(work[col], errors="coerce")
                work[col] = np.log1p(values.where(values >= 0))
        work = work.fillna(self.medians_)
        return ((work - self.centers_) / self.scales_).to_numpy(float)

def feature_columns(frame: pd.DataFrame, model_type: str, ablation: str = "full_raw") -> list[str]:
    base = [c for c in frame.columns if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(frame[c])]
    if model_type == "quantitative":
        predictive = [c for c in base if c.startswith("A_mean__") or c.startswith("V_mean__")]
        if ablation == "arterial":
            return [c for c in predictive if c.startswith("A_mean__")]
        if ablation == "venous":
            return [c for c in predictive if c.startswith("V_mean__")]
        if ablation == "raw_two_phase":
            return [c for c in predictive if c.startswith("A_mean__") or c.startswith("V_mean__")]
        return predictive
    if model_type == "three_slice":
        raw_prefix = "ThreeSlice__"
        core: list[str] = []
    elif model_type == "three_d":
        raw_prefix = "Radiomics__"
        core = []
    elif model_type == "habitat":
        raw_prefix = "Habitat__"
        core = [c for c in base if c == "TumorVolume_mm3" or (c.startswith("Habitat__") and not any(f"__{channel}__" in c for channel in ("A100", "A150", "V100", "V150")))]
    else:
        raise ValueError(model_type)
    raw = [c for c in base if c.startswith(raw_prefix)]
    if model_type == "habitat":
        raw = [c for c in raw if any(f"__{channel}__" in c for channel in ("A100", "A150", "V100", "V150"))]
    channels = {channel: [c for c in raw if f"__{channel}__" in c] for channel in ("A100", "A150", "V100", "V150")}
    if ablation in channels:
        return core + channels[ablation]
    pairs = {
        "arterial_dual_energy": ("A100", "A150"),
        "venous_dual_energy": ("V100", "V150"),
        "dual_phase_100": ("A100", "V100"),
        "dual_phase_150": ("A150", "V150"),
    }
    if ablation in pairs:
        return core + [c for ch in pairs[ablation] for c in channels[ch]]
    if ablation == "full_raw":
        return core + raw
    if ablation in {"full_raw", "full_contrasts"}:
        return core + raw
    raise ValueError(f"Unknown ablation without change features: {ablation}")

def _make_preprocessor(cfg: dict[str, Any]) -> FoldPreprocessor:
    p = cfg["preprocessing"]
    return FoldPreprocessor(float(p["missing_rate_max"]), float(p["correlation_abs_max"]), int(p["max_correlation_candidates"]), tuple(p["log1p_name_tokens"]))

def _logistic(C: float, l1_ratio: float, seed: int) -> LogisticRegression:
    return LogisticRegression(solver="saga", C=C, l1_ratio=l1_ratio,
                              class_weight="balanced", max_iter=3000, tol=1e-3, random_state=seed)

def _parameter_grid(cfg: dict[str, Any], strategy: str, quick: bool) -> list[dict[str, Any]]:
    C = cfg["models"]["C_grid"][:2] if quick else cfg["models"]["C_grid"]
    ratios = cfg["models"]["l1_ratio_grid"][:2] if quick else cfg["models"]["l1_ratio_grid"]
    if strategy == "topk":
        C = [0.1] if quick else [0.1, 1.0]
        ratios = [0.5]
    ks: Iterable[int | None] = [None] if strategy == "elasticnet_direct" else (cfg["models"]["top_k_grid"][:2] if quick else cfg["models"]["top_k_grid"])
    return [{"C": float(c), "l1_ratio": float(r), "top_k": k} for c, r, k in itertools.product(C, ratios, ks)]

def _tune_classifier(X: pd.DataFrame, y: np.ndarray, cfg: dict[str, Any], strategy: str, seed: int, quick: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    grid = _parameter_grid(cfg, strategy, quick)
    n_splits = min(int(cfg["validation"]["inner_folds"]), int(np.bincount(y).min()))
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=seed)
    splits = list(cv.split(X, y))
    cache: dict[tuple[int, int | None], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    top_k_values = sorted({None if p["top_k"] is None else int(p["top_k"]) for p in grid}, key=lambda v: -1 if v is None else v)
    for fold, (tr, va) in enumerate(splits):
        for top_k in top_k_values:
            prep = _make_preprocessor(cfg).fit(X.iloc[tr], y[tr], top_k=top_k)
            cache[(fold, top_k)] = (prep.transform(X.iloc[tr]), prep.transform(X.iloc[va]), y[tr], y[va])
    rows = []
    for params in grid:
        fold_scores = []
        top_k = None if params["top_k"] is None else int(params["top_k"])
        for fold, _split in enumerate(splits):
            train, valid, y_train, y_valid = cache[(fold, top_k)]
            model = _logistic(params["C"], params["l1_ratio"], seed + fold)
            model.fit(train, y_train)
            pred = model.predict_proba(valid)[:, 1]
            fold_scores.append(roc_auc_score(y_valid, pred) if len(np.unique(y_valid)) == 2 else np.nan)
        rows.append({**params, "mean_inner_auc": float(np.nanmean(fold_scores)), "sd_inner_auc": float(np.nanstd(fold_scores))})
    table = pd.DataFrame(rows).sort_values(["mean_inner_auc", "sd_inner_auc", "C"], ascending=[False, True, True]).reset_index(drop=True)
    return table.iloc[0][["C", "l1_ratio", "top_k"]].to_dict(), table

def nested_classifier(
    frame: pd.DataFrame,
    features: list[str],
    temporal_cutoff: pd.Timestamp,
    cfg: dict[str, Any],
    strategy: str,
    analysis_id: str,
    quick: bool = False,
    outer_repeats_override: int | None = None,
    label_column: str = "Ki67_ge50",
) -> dict[str, Any]:
    seed = int(cfg["project"]["random_seed"])
    dev = frame[frame["ExamDate"] < temporal_cutoff].copy().reset_index(drop=True)
    temporal = frame[frame["ExamDate"] >= temporal_cutoff].copy().reset_index(drop=True)
    y = dev[label_column].astype(int).to_numpy()
    X = dev[features]
    repeats = 1 if quick else int(outer_repeats_override or cfg["validation"]["outer_repeats"])
    folds = min(int(cfg["validation"]["outer_folds"]), int(np.bincount(y).min()))
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    pred_rows = []
    selection_rows = []
    tuning_rows = []
    for outer_index, (tr, va) in enumerate(cv.split(X, y)):
        repeat = outer_index // folds
        fold = outer_index % folds
        best, tune = _tune_classifier(X.iloc[tr], y[tr], cfg, strategy, seed + outer_index * 101, quick)
        top_k = None if pd.isna(best["top_k"]) else int(best["top_k"])
        prep = _make_preprocessor(cfg).fit(X.iloc[tr], y[tr], top_k=top_k)
        model = _logistic(float(best["C"]), float(best["l1_ratio"]), seed + outer_index)
        model.fit(prep.transform(X.iloc[tr]), y[tr])
        pred = model.predict_proba(prep.transform(X.iloc[va]))[:, 1]
        for idx, probability in zip(va, pred):
            pred_rows.append({"analysis_id": analysis_id, "strategy": strategy, "PatientID": dev.loc[idx, "PatientID"], "repeat": repeat, "outer_fold": fold, "y_true": int(y[idx]), "probability": float(probability)})
        coef = model.coef_.ravel()
        for name, value in zip(prep.selected_features_, coef):
            selection_rows.append({"analysis_id": analysis_id, "strategy": strategy, "repeat": repeat, "outer_fold": fold, "feature": name, "coefficient": float(value), "selected_nonzero": bool(abs(value) > 1e-10)})
        tune.insert(0, "outer_fold", fold); tune.insert(0, "repeat", repeat); tune.insert(0, "strategy", strategy); tune.insert(0, "analysis_id", analysis_id)
        tuning_rows.append(tune)
    long = pd.DataFrame(pred_rows)
    oof = long.groupby(["analysis_id", "strategy", "PatientID", "y_true"], as_index=False)["probability"].mean()
    best, final_tune = _tune_classifier(X, y, cfg, strategy, seed + 900000, quick)
    final_top_k = None if pd.isna(best["top_k"]) else int(best["top_k"])
    prep = _make_preprocessor(cfg).fit(X, y, top_k=final_top_k)
    final_model = _logistic(float(best["C"]), float(best["l1_ratio"]), seed)
    final_model.fit(prep.transform(X), y)
    temporal_pred = final_model.predict_proba(prep.transform(temporal[features]))[:, 1] if len(temporal) else np.array([])
    temporal_out = pd.DataFrame({"analysis_id": analysis_id, "strategy": strategy, "PatientID": temporal["PatientID"], "y_true": temporal[label_column] if label_column in temporal else np.nan, "probability": temporal_pred})
    final_coef = pd.DataFrame({"analysis_id": analysis_id, "strategy": strategy, "feature": prep.selected_features_, "coefficient": final_model.coef_.ravel()})
    return {"oof": oof, "oof_long": long, "temporal": temporal_out, "selections": pd.DataFrame(selection_rows), "tuning": pd.concat(tuning_rows, ignore_index=True), "final_tuning": final_tune, "final_coefficients": final_coef, "preprocessor": prep, "model": final_model}

def continuous_elasticnet(frame: pd.DataFrame, features: list[str], temporal_cutoff: pd.Timestamp, cfg: dict[str, Any], analysis_id: str, quick: bool = False) -> dict[str, pd.DataFrame]:
    seed = int(cfg["project"]["random_seed"])
    dev = frame[frame["ExamDate"] < temporal_cutoff].copy().reset_index(drop=True)
    temporal = frame[frame["ExamDate"] >= temporal_cutoff].copy().reset_index(drop=True)
    y = dev["Ki67"].to_numpy(float)
    X = dev[features]
    repeats = 1 if quick else int(cfg["validation"]["outer_repeats"])
    folds = min(int(cfg["validation"]["outer_folds"]), len(dev))
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    selection = []
    for repeat in range(repeats):
        cv = KFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for fold, (tr, va) in enumerate(cv.split(X)):
            prep = _make_preprocessor(cfg).fit(X.iloc[tr], y[tr], top_k=None)
            best_score = -np.inf; best = None
            alphas = cfg["models"]["continuous_alpha_grid"][:2] if quick else cfg["models"]["continuous_alpha_grid"]
            ratios = cfg["models"]["l1_ratio_grid"][:2] if quick else cfg["models"]["l1_ratio_grid"]
            inner = KFold(n_splits=min(int(cfg["validation"]["inner_folds"]), len(tr)), shuffle=True, random_state=seed + fold)
            for alpha, ratio in itertools.product(alphas, ratios):
                scores = []
                for itr, iva in inner.split(X.iloc[tr]):
                    inner_prep = _make_preprocessor(cfg).fit(X.iloc[tr].iloc[itr], y[tr][itr])
                    model = ElasticNet(alpha=float(alpha), l1_ratio=float(ratio), max_iter=20000, tol=1e-3, random_state=seed)
                    model.fit(inner_prep.transform(X.iloc[tr].iloc[itr]), y[tr][itr])
                    pred = model.predict(inner_prep.transform(X.iloc[tr].iloc[iva]))
                    scores.append(_safe_signed_spearman(y[tr][iva], pred))
                score = float(np.mean(scores)) if scores else 0.0
                if score > best_score:
                    best_score, best = score, (float(alpha), float(ratio))
            model = ElasticNet(alpha=best[0], l1_ratio=best[1], max_iter=20000, tol=1e-3, random_state=seed)
            model.fit(prep.transform(X.iloc[tr]), y[tr])
            pred = model.predict(prep.transform(X.iloc[va]))
            for idx, value in zip(va, pred):
                rows.append({"analysis_id": analysis_id, "PatientID": dev.loc[idx, "PatientID"], "repeat": repeat, "outer_fold": fold, "y_true": y[idx], "prediction": float(value)})
            for name, coef in zip(prep.selected_features_, model.coef_):
                selection.append({"analysis_id": analysis_id, "repeat": repeat, "outer_fold": fold, "feature": name, "coefficient": float(coef), "selected_nonzero": bool(abs(coef) > 1e-10)})
    long = pd.DataFrame(rows)
    oof = long.groupby(["analysis_id", "PatientID", "y_true"], as_index=False)["prediction"].mean()
    prep = _make_preprocessor(cfg).fit(X, y)
    best_score = -np.inf; best = None
    inner = KFold(n_splits=min(int(cfg["validation"]["inner_folds"]), len(dev)), shuffle=True, random_state=seed)
    for alpha, ratio in itertools.product(cfg["models"]["continuous_alpha_grid"], cfg["models"]["l1_ratio_grid"]):
        scores = []
        for tr, va in inner.split(X):
            p = _make_preprocessor(cfg).fit(X.iloc[tr], y[tr])
            m = ElasticNet(alpha=float(alpha), l1_ratio=float(ratio), max_iter=20000, tol=1e-3, random_state=seed)
            m.fit(p.transform(X.iloc[tr]), y[tr]); pred = m.predict(p.transform(X.iloc[va]))
            scores.append(_safe_signed_spearman(y[va], pred))
        score = float(np.mean(scores)) if scores else 0.0
        if score > best_score:
            best_score, best = score, (float(alpha), float(ratio))
    final = ElasticNet(alpha=best[0], l1_ratio=best[1], max_iter=20000, tol=1e-3, random_state=seed).fit(prep.transform(X), y)
    temporal_out = pd.DataFrame({"analysis_id": analysis_id, "PatientID": temporal["PatientID"], "y_true": temporal["Ki67"], "prediction": final.predict(prep.transform(temporal[features])) if len(temporal) else []})
    return {"oof": oof, "oof_long": long, "temporal": temporal_out, "selections": pd.DataFrame(selection)}

def supplemental_tree_oof(frame: pd.DataFrame, features: list[str], temporal_cutoff: pd.Timestamp, cfg: dict[str, Any], analysis_id: str, algorithm: str, quick: bool = False) -> dict[str, pd.DataFrame]:
    seed = int(cfg["project"]["random_seed"])
    dev = frame[frame["ExamDate"] < temporal_cutoff].copy().reset_index(drop=True)
    temporal = frame[frame["ExamDate"] >= temporal_cutoff].copy().reset_index(drop=True)
    y = dev["Ki67_ge50"].astype(int).to_numpy(); X = dev[features]
    cv = StratifiedKFold(n_splits=min(5, int(np.bincount(y).min())), shuffle=True, random_state=seed)
    rows = []; importances = []
    for fold, (tr, va) in enumerate(cv.split(X, y)):
        prep = _make_preprocessor(cfg).fit(X.iloc[tr], y[tr])
        if algorithm == "random_forest":
            model = RandomForestClassifier(n_estimators=200 if quick else 500, max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=seed + fold, n_jobs=cfg["runtime"]["n_jobs"])
        else:
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=100 if quick else 250, max_depth=2, learning_rate=.03, subsample=.8, colsample_bytree=.7, reg_lambda=2, objective="binary:logistic", eval_metric="logloss", random_state=seed + fold, n_jobs=cfg["runtime"]["n_jobs"])
        model.fit(prep.transform(X.iloc[tr]), y[tr]); pred = model.predict_proba(prep.transform(X.iloc[va]))[:, 1]
        for idx, value in zip(va, pred):
            rows.append({"analysis_id": analysis_id, "algorithm": algorithm, "PatientID": dev.loc[idx, "PatientID"], "outer_fold": fold, "y_true": int(y[idx]), "probability": float(value)})
        values = getattr(model, "feature_importances_", np.zeros(len(prep.selected_features_)))
        for name, value in zip(prep.selected_features_, values):
            importances.append({"analysis_id": analysis_id, "algorithm": algorithm, "outer_fold": fold, "feature": name, "importance": float(value)})
    long = pd.DataFrame(rows)
    return {"oof": long.groupby(["analysis_id", "algorithm", "PatientID", "y_true"], as_index=False)["probability"].mean(), "oof_long": long, "importances": pd.DataFrame(importances)}

def nested_fusion_matched_classifier(
    frame: pd.DataFrame,
    features: list[str],
    temporal_cutoff: pd.Timestamp,
    cfg: dict[str, Any],
    analysis_id: str,
    algorithm: str,
    quick: bool = False,
    outer_repeats_override: int | None = None,
    label_column: str = "Ki67_ge50",
) -> dict[str, Any]:
    seed = int(cfg["project"]["random_seed"])
    fusion_algorithm = "randomforest" if algorithm == "random_forest" else algorithm
    if fusion_algorithm not in {"elasticnet", "randomforest", "xgboost"}:
        raise ValueError(f"Unsupported matched algorithm: {algorithm}")
    top_k = int(cfg.get("models", {}).get(
        "fusion_top_k_quantitative" if "quantitative" in analysis_id else "fusion_top_k_radiomics",
        8 if "quantitative" in analysis_id else 12,
    ))
    from .multimodel_fusion import _radiomics_channel_groups, make_pipeline
    channel_groups = (
        _radiomics_channel_groups(list(features))
        if ("three_slice" in analysis_id or "three_d" in analysis_id)
        else ()
    )
    dev = frame[frame["ExamDate"] < temporal_cutoff].copy().reset_index(drop=True)
    temporal = frame[frame["ExamDate"] >= temporal_cutoff].copy().reset_index(drop=True)
    if label_column not in dev.columns:
        raise ValueError(f"Outcome column {label_column!r} is missing from {analysis_id}")
    y = pd.to_numeric(dev[label_column], errors="coerce").astype(int).to_numpy()
    X = dev[features]
    if len(dev) < 4 or len(np.unique(y)) < 2:
        raise ValueError(f"{analysis_id}/{algorithm} development cohort or class distribution is insufficient")

    strategy = "fusion_matched_nested"
    repeats = 1 if quick else int(outer_repeats_override or cfg["validation"]["outer_repeats"])
    folds = min(int(cfg["validation"]["outer_folds"]), int(np.bincount(y).min()))
    cv = RepeatedStratifiedKFold(n_splits=max(2, folds), n_repeats=repeats, random_state=seed)
    pred_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for outer_index, (tr, va) in enumerate(cv.split(X, y)):
        repeat = outer_index // max(2, folds)
        outer_fold = outer_index % max(2, folds)
        fold_seed = seed + outer_index
        pipe = make_pipeline(fusion_algorithm, top_k, fold_seed, channel_groups)
        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict_proba(X.iloc[va])[:, 1]
        for idx, probability in zip(va, pred):
            pred_rows.append({
                "analysis_id": analysis_id,
                "strategy": strategy,
                "algorithm": algorithm,
                "PatientID": dev.loc[idx, "PatientID"],
                "repeat": repeat,
                "outer_fold": outer_fold,
                "y_true": int(y[idx]),
                "probability": float(probability),
            })
        selector = pipe.named_steps["selector"]
        selected_indices = np.asarray(selector.support_, dtype=int)
        selected_names = [features[i] for i in selected_indices]
        base_model = pipe.named_steps["model"]
        values = getattr(base_model, "feature_importances_", getattr(base_model, "coef_", np.zeros(len(selected_names))))
        values = np.asarray(values).ravel()
        for name, value in zip(selected_names, values):
            importance_rows.append({
                "analysis_id": analysis_id,
                "strategy": strategy,
                "algorithm": algorithm,
                "repeat": repeat,
                "outer_fold": outer_fold,
                "feature": name,
                "importance": float(value),
            })

    long = pd.DataFrame(pred_rows)
    oof = long.groupby(
        ["analysis_id", "strategy", "algorithm", "PatientID", "y_true"],
        as_index=False,
    )["probability"].mean()

    final_pipe = make_pipeline(fusion_algorithm, top_k, seed + 900000, channel_groups)
    final_pipe.fit(X, y)
    temporal_probability = (
        final_pipe.predict_proba(temporal[features])[:, 1]
        if len(temporal) else np.array([], dtype=float)
    )
    temporal_out = pd.DataFrame({
        "analysis_id": analysis_id,
        "strategy": strategy,
        "algorithm": algorithm,
        "PatientID": temporal["PatientID"],
        "y_true": temporal[label_column] if label_column in temporal else np.nan,
        "probability": temporal_probability,
    })
    final_selector = final_pipe.named_steps["selector"]
    final_indices = np.asarray(final_selector.support_, dtype=int)
    final_names = [features[i] for i in final_indices]
    final_base_model = final_pipe.named_steps["model"]
    final_values = getattr(final_base_model, "feature_importances_", getattr(final_base_model, "coef_", np.zeros(len(final_names))))
    final_values = np.asarray(final_values).ravel()
    final_importance = pd.DataFrame({
        "analysis_id": analysis_id,
        "strategy": strategy,
        "algorithm": algorithm,
        "feature": final_names,
        "importance": final_values,
        "cohort": "development_final_fit",
    })
    importance = pd.concat([pd.DataFrame(importance_rows), final_importance], ignore_index=True)
    return {
        "oof": oof,
        "oof_long": long,
        "temporal": temporal_out,
        "importances": importance,
        "pipeline": final_pipe,
    }
