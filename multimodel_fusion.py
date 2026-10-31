from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

LOGGER = logging.getLogger("reviewer_multimodel_fusion")

@dataclass
class Config:

    project_dir: str = str(Path(__file__).resolve().parents[1])
    quantitative_file: str = "data/quantitative_matrix.csv"
    three_slice_file: str = "data/three_slice_matrix.csv"
    three_d_file: str = "data/three_d_matrix.csv"
    output_dir: str = "results/model_level_fusion_optimized_v2"
    random_state: int = 20261021
    temporal_cutoff: str = "2023-08-01"
    n_repeats: int = 5
    outer_folds: int = 5
    inner_folds: int = 4
    ki67_threshold: float = 0.50
    top_k_quantitative: int = 8
    top_k_radiomics: int = 12
    top_k_fused: int = 20
    fixed_classification_threshold: float = 0.50
    bootstrap_iterations: int = 2000
    models: Tuple[str, ...] = ("elasticnet", "randomforest")
    periods: Tuple[str, ...] = (
        "A100+A150", "V100+V150", "A100+V100", "A150+V150",
        "A100+A150+V100", "A100+A150+V150", "A100+V100+V150", "A150+V100+V150",
        "A100+A150+V100+V150",
    )
    fusion_methods: Tuple[str, ...] = ("late_stack",)

PERIOD_COMBINATIONS: List[Tuple[str, Tuple[str, ...]]] = [
    ("A100", ("A100",)), ("A150", ("A150",)), ("V100", ("V100",)), ("V150", ("V150",)),
    ("A100+A150", ("A100", "A150")), ("V100+V150", ("V100", "V150")),
    ("A100+V100", ("A100", "V100")), ("A150+V150", ("A150", "V150")),
    ("A100+A150+V100", ("A100", "A150", "V100")),
    ("A100+A150+V150", ("A100", "A150", "V150")),
    ("A100+V100+V150", ("A100", "V100", "V150")),
    ("A150+V100+V150", ("A150", "V100", "V150")),
    ("A100+A150+V100+V150", ("A100", "A150", "V100", "V150")),
]

def _normalise_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

def _read_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Project matrix not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if "PatientID" not in frame.columns:
        raise ValueError(f"Matrix is missing PatientID: {path}")
    frame["PatientID"] = _normalise_id(frame["PatientID"])
    if frame["PatientID"].duplicated().any():
        raise ValueError(f"Matrix must contain exactly one row per patient: {path}")
    return frame

def load_project_inputs(config: Config) -> Dict[str, pd.DataFrame]:

    root = Path(config.project_dir)
    frames = {
        "quantitative": _read_matrix(root / config.quantitative_file),
        "three_slice": _read_matrix(root / config.three_slice_file),
        "three_d": _read_matrix(root / config.three_d_file),
    }
    for frame in frames.values():
        if "Ki67" not in frame.columns:
            raise ValueError("Project matrix is missing the Ki67 column")
        frame["Ki67"] = pd.to_numeric(frame["Ki67"], errors="coerce")
        if "ExamDate" not in frame.columns:
            frame["ExamDate"] = pd.to_datetime(frame["PatientID"].str[:8], format="%Y%m%d", errors="coerce")
    return frames

def _numeric_columns(frame: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [c for c in columns if pd.api.types.is_numeric_dtype(frame[c]) and frame[c].notna().any() and frame[c].std(skipna=True) > 1e-12]

def _image_feature_columns(frame: pd.DataFrame, domain: str, periods: Sequence[str]) -> List[str]:

    prefix = "ThreeSlice" if domain == "three_slice" else "Radiomics"
    raw_prefix = prefix + "__"
    cols = [c for c in frame.columns if str(c).startswith(raw_prefix) and any(f"__{p}__" in str(c) for p in periods)]
    return _numeric_columns(frame, cols)

def _quantitative_feature_columns(frame: pd.DataFrame, periods: Sequence[str]) -> List[str]:

    phases = {p[0] for p in periods}
    cols: List[str] = []
    for phase in sorted(phases): cols.extend(c for c in frame.columns if str(c).startswith(f"{phase}_mean__"))
    return _numeric_columns(frame, cols)

def _label_series(frame: pd.DataFrame, threshold: float) -> pd.Series:
    if "Ki67_ge50" in frame:
        return pd.to_numeric(frame["Ki67_ge50"], errors="coerce").astype("Int64")
    return (pd.to_numeric(frame["Ki67"], errors="coerce") >= threshold).astype("Int64")

def build_fusion_matrices(frames: Dict[str, pd.DataFrame], domain: str, periods: Sequence[str], threshold: float):

    image, quant = frames[domain], frames["quantitative"]
    qcols, rcols = _quantitative_feature_columns(quant, periods), _image_feature_columns(image, domain, periods)
    if not qcols or not rcols: raise ValueError(f"No usable quantitative or radiomics features for {domain}/{periods}")
    ids = sorted(set(image["PatientID"]) & set(quant["PatientID"]))
    image_i, quant_i = image.set_index("PatientID").loc[ids], quant.set_index("PatientID").loc[ids]
    Xq = quant_i[qcols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    Xr = image_i[rcols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    y = _label_series(image_i, threshold).reindex(ids)
    keep = y.notna().to_numpy(); kept_ids = np.asarray(ids)[keep]
    Xq, Xr, y = Xq.loc[keep].reset_index(drop=True), Xr.loc[keep].reset_index(drop=True), y.loc[keep].astype(int).reset_index(drop=True)
    channel_groups = _radiomics_channel_groups(rcols)
    audit = pd.DataFrame({"PatientID": kept_ids, "Ki67": image_i.loc[kept_ids, "Ki67"].to_numpy(), "ExamDate": image_i.loc[kept_ids, "ExamDate"].to_numpy(), "n_quantitative_features": len(qcols), "n_radiomics_features": len(rcols), "n_radiomics_channels": len(channel_groups), "domain": domain, "period_combination": "+".join(periods)})
    return Xq, Xr, y, audit

class TopKMutualInfo(BaseEstimator, TransformerMixin):

    def __init__(self, k: int = 20, random_state: int = 42): self.k, self.random_state = k, random_state

    def fit(self, X, y):
        values = np.asarray(X, dtype=float); k = min(max(1, int(self.k)), values.shape[1])
        candidate_limit = min(values.shape[1], max(50, 20 * k))
        if values.shape[1] > candidate_limit:
            y_values = np.asarray(y, dtype=float)
            x_centered = values - values.mean(axis=0, keepdims=True)
            y_centered = y_values - y_values.mean()
            denom = np.sqrt((x_centered * x_centered).sum(axis=0) * (y_centered * y_centered).sum())
            corr = np.divide(np.abs(x_centered.T @ y_centered), denom, out=np.zeros(values.shape[1]), where=denom > 0)
            candidate_idx = np.argsort(-corr, kind="mergesort")[:candidate_limit]
        else:
            candidate_idx = np.arange(values.shape[1])
        mi = np.nan_to_num(mutual_info_classif(values[:, candidate_idx], np.asarray(y), random_state=self.random_state), nan=0.0, posinf=0.0, neginf=0.0)
        scores = np.zeros(values.shape[1], dtype=float)
        scores[candidate_idx] = mi
        self.support_ = np.sort(candidate_idx[np.argsort(-mi, kind="mergesort")[:k]])
        self.scores_ = scores; self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X): return np.asarray(X, dtype=float)[:, self.support_]

class ChannelBalancedTopK(BaseEstimator, TransformerMixin):

    def __init__(self, k: int = 12, channel_groups: Sequence[Sequence[int]] = (), random_state: int = 42):
        self.k = k
        self.channel_groups = channel_groups
        self.random_state = random_state

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        groups = [tuple(int(i) for i in group) for group in self.channel_groups if group]
        if not groups:
            raise ValueError("ChannelBalancedTopK requires at least one channel group")
        total = min(max(1, int(self.k)), values.shape[1])
        if total >= len(groups):
            base, remainder = divmod(total, len(groups))
            quotas = [base + int(i < remainder) for i in range(len(groups))]
        else:
            quotas = [1 if i < total else 0 for i in range(len(groups))]
        support = []
        for group_index, (group, quota) in enumerate(zip(groups, quotas)):
            if quota <= 0:
                continue
            local = TopKMutualInfo(quota, self.random_state + group_index)
            local.fit(values[:, list(group)], y)
            support.extend(np.asarray(group, dtype=int)[local.support_].tolist())
        self.support_ = np.asarray(sorted(set(support)), dtype=int)
        self.n_features_in_ = values.shape[1]
        self.channel_quotas_ = quotas
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]

def _radiomics_channel_groups(columns: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    channels = ("A100", "A150", "V100", "V150")
    return tuple(
        tuple(i for i, column in enumerate(columns) if f"__{channel}__" in str(column))
        for channel in channels
        if any(f"__{channel}__" in str(column) for column in columns)
    )

def make_estimator(name: str, seed: int):
    if name == "elasticnet":
        return LogisticRegression(solver="saga", C=0.1, l1_ratio=0.5, class_weight="balanced", max_iter=4000, random_state=seed)
    if name == "randomforest":
        return RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "xgboost":
        return XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.03, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=1, random_state=seed)
    raise ValueError(f"Unsupported model: {name}")

def make_pipeline(name: str, top_k: int, seed: int, channel_groups: Sequence[Sequence[int]] = ()) -> Pipeline:
    selector = ChannelBalancedTopK(top_k, channel_groups, seed) if channel_groups else TopKMutualInfo(top_k, seed)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler()), ("selector", selector), ("model", make_estimator(name, seed))])

def _balanced_domain_k(cfg: Config, n_quantitative: int, n_radiomics: int) -> Tuple[int, int]:
    target = max(2, int(cfg.top_k_fused))
    qk = min(max(1, int(cfg.top_k_quantitative)), max(1, int(n_quantitative)))
    rk = min(max(1, int(cfg.top_k_radiomics)), max(1, int(n_radiomics)))
    if qk + rk <= target:
        return qk, rk
    qk = max(1, int(round(target * qk / (qk + rk))))
    rk = max(1, target - qk)
    while qk + rk > target:
        if rk > qk and rk > 1:
            rk -= 1
        elif qk > 1:
            qk -= 1
        else:
            break
    return qk, rk

def _fit_predict_balanced_feature_level(
    Xq_train: pd.DataFrame,
    Xr_train: pd.DataFrame,
    y_train: np.ndarray,
    Xq_test: pd.DataFrame,
    Xr_test: pd.DataFrame,
    name: str,
    cfg: Config,
    seed: int,
    radiomics_channel_groups: Sequence[Sequence[int]],
) -> np.ndarray:
    qk, rk = _balanced_domain_k(cfg, Xq_train.shape[1], Xr_train.shape[1])
    q_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("selector", TopKMutualInfo(qk, random_state=seed)),
    ])
    r_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("selector", ChannelBalancedTopK(rk, radiomics_channel_groups, random_state=seed + 1)),
    ])
    q_train = q_pipe.fit_transform(Xq_train, y_train)
    r_train = r_pipe.fit_transform(Xr_train, y_train)
    q_test = q_pipe.transform(Xq_test)
    r_test = r_pipe.transform(Xr_test)
    model = make_estimator(name, seed)
    model.fit(np.hstack([q_train, r_train]), y_train)
    return model.predict_proba(np.hstack([q_test, r_test]))[:, 1]

def _n_splits(y: np.ndarray, requested: int) -> int:
    return max(2, min(requested, int(np.bincount(np.asarray(y, dtype=int)).min())))

def _fit_predict(pipe: Pipeline, X_train, y_train, X_test) -> np.ndarray:
    fitted = clone(pipe); fitted.fit(X_train, y_train); return fitted.predict_proba(X_test)[:, 1]

def _inner_oof(X, y, name: str, top_k: int, cfg: Config, seed: int, channel_groups: Sequence[Sequence[int]] = ()) -> np.ndarray:
    pred = np.full(len(y), np.nan, dtype=float); cv = StratifiedKFold(_n_splits(y, cfg.inner_folds), shuffle=True, random_state=seed)
    for tr, va in cv.split(X, y): pred[va] = _fit_predict(make_pipeline(name, top_k, seed, channel_groups), X.iloc[tr], y[tr], X.iloc[va])
    return pred

def _late_predict(Xq_train, Xr_train, y_train, Xq_test, Xr_test, name: str, cfg: Config, seed: int, radiomics_channel_groups: Sequence[Sequence[int]]):
    q_oof = _inner_oof(Xq_train, y_train, name, cfg.top_k_quantitative, cfg, seed); r_oof = _inner_oof(Xr_train, y_train, name, cfg.top_k_radiomics, cfg, seed + 1, radiomics_channel_groups)
    meta = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=seed).fit(np.column_stack([q_oof, r_oof]), y_train)
    q_test = _fit_predict(make_pipeline(name, cfg.top_k_quantitative, seed), Xq_train, y_train, Xq_test); r_test = _fit_predict(make_pipeline(name, cfg.top_k_radiomics, seed + 1, radiomics_channel_groups), Xr_train, y_train, Xr_test)
    stacked = meta.predict_proba(np.column_stack([q_test, r_test]))[:, 1]; best_w, best_auc = 0.5, -np.inf
    for w in np.linspace(0, 1, 21):
        auc = roc_auc_score(y_train, w * q_oof + (1 - w) * r_oof)
        if auc > best_auc: best_w, best_auc = float(w), auc
    return stacked, best_w * q_test + (1 - best_w) * r_test, best_w

def _metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else np.nan, "pr_auc": float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else np.nan, "accuracy": float(accuracy_score(y, pred)), "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan, "specificity": float(tn / (tn + fp)) if tn + fp else np.nan, "brier": float(brier_score_loss(y, prob))}

def _bootstrap_ci(y: np.ndarray, prob: np.ndarray, threshold: float, metric: str,
                  iterations: int, seed: int) -> tuple[float, float]:
    y = np.asarray(y, dtype=int); prob = np.asarray(prob, dtype=float)
    if len(y) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(iterations)):
        idx = rng.integers(0, len(y), len(y))
        yy, pp = y[idx], prob[idx]
        if metric == "auc" and len(np.unique(yy)) < 2:
            continue
        values.append(float(_metrics(yy, pp, threshold)[metric]))
    if not values:
        return (float("nan"), float("nan"))
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))

def _youden_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, prob)
    j = tpr - fpr
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    best = np.flatnonzero(finite)[int(np.nanargmax(j[finite]))]
    return float(np.clip(thresholds[best], 0.0, 1.0))

def run_model_optimized(Xq: pd.DataFrame, Xr: pd.DataFrame, y: pd.Series, audit: pd.DataFrame, label: str, name: str, cfg: Config):
    dates = pd.to_datetime(audit["ExamDate"], errors="coerce")
    cutoff = pd.Timestamp(cfg.temporal_cutoff)
    dev_idx = np.flatnonzero((dates < cutoff).to_numpy())
    temporal_idx = np.flatnonzero((dates >= cutoff).to_numpy())
    if len(dev_idx) < 30 or len(np.unique(y.iloc[dev_idx])) < 2:
        raise ValueError(f"{label} development cohort or class distribution is insufficient")

    methods = tuple(cfg.fusion_methods)
    invalid_methods = set(methods) - {"feature_level", "late_stack", "soft_vote"}
    if invalid_methods:
        raise ValueError(f"Unsupported fusion methods: {sorted(invalid_methods)}")
    fold_scores = {method: [] for method in methods}
    summaries, dev_predictions, temporal_predictions = [], [], []
    y_np = y.to_numpy(dtype=int)
    radiomics_channel_groups = _radiomics_channel_groups(list(Xr.columns))

    for repeat in range(cfg.n_repeats):
        stable = sum((i + 1) * ord(c) for i, c in enumerate(label + name)) % 997
        seed = cfg.random_state + repeat * 1000 + stable
        cv = StratifiedKFold(_n_splits(y_np[dev_idx], cfg.outer_folds), shuffle=True, random_state=seed)
        for fold, (tr_local, va_local) in enumerate(cv.split(Xq.iloc[dev_idx], y_np[dev_idx]), 1):
            tr, va = dev_idx[tr_local], dev_idx[va_local]
            probabilities = {}
            if "feature_level" in methods:
                probabilities["feature_level"] = _fit_predict_balanced_feature_level(
                    Xq.iloc[tr], Xr.iloc[tr], y_np[tr], Xq.iloc[va], Xr.iloc[va], name, cfg, seed, radiomics_channel_groups
                )
            if "late_stack" in methods or "soft_vote" in methods:
                stacked, vote, weight = _late_predict(
                    Xq.iloc[tr], Xr.iloc[tr], y_np[tr], Xq.iloc[va], Xr.iloc[va], name, cfg, seed, radiomics_channel_groups
                )
                probabilities.update({"late_stack": stacked, "soft_vote": vote})
            else:
                weight = np.nan
            for method in methods:
                prob = probabilities[method]
                score = _metrics(y_np[va], prob, cfg.fixed_classification_threshold)
                score.update({"repeat": repeat + 1, "fold": fold, "soft_vote_quant_weight": weight if method == "soft_vote" else np.nan})
                fold_scores[method].append(score)
                for row_idx, p in zip(va, prob):
                    dev_predictions.append({
                        "period_combination": label, "model": name, "fusion_method": method,
                        "repeat": repeat + 1, "fold": fold, "PatientID": audit.iloc[row_idx]["PatientID"],
                        "Ki67": audit.iloc[row_idx]["Ki67"], "y_true": int(y_np[row_idx]),
                        "probability": float(p), "cohort": "development_oof",
                    })

    temporal_probabilities: Dict[str, np.ndarray] = {}
    if len(temporal_idx):
        seed = cfg.random_state + 900000 + sum(ord(c) for c in label + name)
        if "feature_level" in methods:
            temporal_probabilities["feature_level"] = _fit_predict_balanced_feature_level(
                Xq.iloc[dev_idx], Xr.iloc[dev_idx], y_np[dev_idx],
                Xq.iloc[temporal_idx], Xr.iloc[temporal_idx], name, cfg, seed, radiomics_channel_groups
            )
        if "late_stack" in methods or "soft_vote" in methods:
            stacked, vote, _weight = _late_predict(
                Xq.iloc[dev_idx], Xr.iloc[dev_idx], y_np[dev_idx],
                Xq.iloc[temporal_idx], Xr.iloc[temporal_idx], name, cfg, seed, radiomics_channel_groups
            )
            temporal_probabilities.update({"late_stack": stacked, "soft_vote": vote})

    dev_frame = pd.DataFrame(dev_predictions)
    dev_thresholds: Dict[str, float] = {}
    dev_pooled: Dict[str, pd.DataFrame] = {}
    for method in methods:
        method_frame = dev_frame[dev_frame["fusion_method"] == method].copy()
        pooled = method_frame.groupby(["PatientID", "y_true"], as_index=False)["probability"].mean()
        dev_pooled[method] = pooled
        dev_thresholds[method] = _youden_threshold(pooled["y_true"].to_numpy(int), pooled["probability"].to_numpy(float))

    for method in methods:
        fd = pd.DataFrame(fold_scores[method])
        threshold = dev_thresholds[method]
        pooled = dev_pooled[method]
        dev_score = _metrics(pooled["y_true"].to_numpy(int), pooled["probability"].to_numpy(float), threshold)
        selection_basis = (
            "prespecified_full_four_channel_no_temporal_selection"
            if label.endswith("A100+A150+V100+V150")
            else "explicit_sensitivity_run_not_primary"
        )
        summary = {
            "period_combination": label, "model": name, "fusion_method": method,
            "n_patients": int(len(y)), "n_development": int(len(dev_idx)), "n_temporal": int(len(temporal_idx)),
            "development_outer_auc_mean": float(fd.auc.mean()),
            "development_outer_auc_sd": float(fd.auc.std(ddof=1)),
            "development_oof_auc": float(dev_score["auc"]),
            "development_accuracy_mean": float(dev_score["accuracy"]),
            "development_sensitivity_mean": float(dev_score["sensitivity"]),
            "development_specificity_mean": float(dev_score["specificity"]),
            "development_brier_mean": float(dev_score["brier"]),
            "temporal_auc": np.nan, "temporal_accuracy": np.nan, "temporal_sensitivity": np.nan,
            "temporal_specificity": np.nan, "temporal_brier": np.nan,
            "fixed_threshold": threshold,
            "threshold_source": "development_nested_oof_youden",
            "period_selection_basis": selection_basis,
            "mean_soft_vote_quant_weight": float(fd.soft_vote_quant_weight.mean()) if method == "soft_vote" else np.nan,
        }
        ci_seed = cfg.random_state + 700000 + sum(ord(c) for c in label + name + method)
        for metric in ("auc", "accuracy", "sensitivity", "specificity", "brier"):
            low, high = _bootstrap_ci(
                pooled["y_true"].to_numpy(int), pooled["probability"].to_numpy(float),
                threshold, metric, cfg.bootstrap_iterations, ci_seed + len(metric),
            )
            summary[f"development_{metric}_ci_low"] = low
            summary[f"development_{metric}_ci_high"] = high
        if len(temporal_idx):
            temporal_prob = temporal_probabilities[method]
            tm = _metrics(y_np[temporal_idx], temporal_prob, threshold)
            summary.update({f"temporal_{k}": v for k, v in tm.items() if k in {"auc", "accuracy", "sensitivity", "specificity", "brier"}})
            for metric in ("auc", "accuracy", "sensitivity", "specificity", "brier"):
                low, high = _bootstrap_ci(
                    y_np[temporal_idx], temporal_prob, threshold, metric,
                    cfg.bootstrap_iterations, ci_seed + 100 + len(metric),
                )
                summary[f"temporal_{metric}_ci_low"] = low
                summary[f"temporal_{metric}_ci_high"] = high
            for row_idx, p in zip(temporal_idx, temporal_prob):
                temporal_predictions.append({
                    "period_combination": label, "model": name, "fusion_method": method,
                    "PatientID": audit.iloc[row_idx]["PatientID"], "Ki67": audit.iloc[row_idx]["Ki67"],
                    "y_true": int(y_np[row_idx]), "probability": float(p),
                    "threshold": threshold, "cohort": "internal_temporal_validation",
                })
        summaries.append(summary)
    return summaries, dev_predictions, temporal_predictions

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Multimodel fusion for the reviewer_reanalysis project")
    parser.add_argument("--quick", action="store_true", help="Smoke check with one repeat, three outer folds and two inner folds")
    parser.add_argument("--models", default="elasticnet,randomforest", help="elasticnet,randomforest,xgboost")
    parser.add_argument("--periods", default=None, help="Use for prespecified sensitivity analyses; by default, run all dual-, triple- and full-channel combinations")
    parser.add_argument("--fusion-methods", default=None, help="Supplementary comparison methods; the default primary strategy is late_stack")
    parser.add_argument("--output-dir", default=None, help="Output directory relative to the project root")
    args = parser.parse_args(); cfg = Config(models=tuple(x.strip().lower() for x in args.models.split(",") if x.strip()))
    if args.periods:
        cfg.periods = tuple(x.strip() for x in args.periods.split(",") if x.strip())
    if args.fusion_methods:
        cfg.fusion_methods = tuple(x.strip() for x in args.fusion_methods.split(",") if x.strip())
    if args.quick:
        cfg.n_repeats, cfg.outer_folds, cfg.inner_folds = 1, 3, 2
        if not cfg.periods:
            cfg.periods = ("A100+A150+V100+V150",)
    if args.output_dir: cfg.output_dir = args.output_dir
    return cfg

def run_fusion(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    root = Path(cfg.project_dir)
    output = root / cfg.output_dir
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with (output / "run_config.json").open("w", encoding="utf-8") as handle: json.dump(asdict(cfg), handle, ensure_ascii=False, indent=2)
    LOGGER.info("Loading configured project data matrices"); frames = load_project_inputs(cfg)
    all_summary, all_dev, all_temporal, audits = [], [], [], []
    period_combinations = [item for item in PERIOD_COMBINATIONS if item[0] in cfg.periods]
    if not period_combinations:
        raise ValueError(f"No matching phase/energy combinations: {cfg.periods}")
    if cfg.fusion_methods == ():
        raise ValueError("At least one fusion method must be selected")
    for domain in ("three_slice", "three_d"):
        for label, periods in period_combinations:
            LOGGER.info("Running %s / %s", domain, label); Xq, Xr, y, audit = build_fusion_matrices(frames, domain, periods, cfg.ki67_threshold); audits.append(audit)
            for model_name in cfg.models:
                summary, dev_pred, temp_pred = run_model_optimized(Xq, Xr, y, audit, f"{domain}:{label}", model_name, cfg); all_summary.extend(summary); all_dev.extend(dev_pred); all_temporal.extend(temp_pred)
    summary_df = pd.DataFrame(all_summary).sort_values("development_outer_auc_mean", ascending=False)
    dev_df, temp_df = pd.DataFrame(all_dev), pd.DataFrame(all_temporal)
    audit_df = pd.concat(audits, ignore_index=True).drop_duplicates(["domain", "period_combination", "PatientID"])
    ci_rows = []
    for row in summary_df.to_dict("records"):
        for cohort in ("development", "temporal"):
            for metric in ("auc", "accuracy", "sensitivity", "specificity", "brier"):
                low_key = f"{cohort}_{metric}_ci_low"; high_key = f"{cohort}_{metric}_ci_high"
                ci_rows.append({
                    "period_combination": row["period_combination"], "model": row["model"],
                    "fusion_method": row["fusion_method"], "cohort": cohort,
                    "metric": metric, "estimate": row.get(f"{cohort}_{metric}", np.nan),
                    "ci_low": row.get(low_key, np.nan), "ci_high": row.get(high_key, np.nan),
                    "bootstrap_iterations": int(cfg.bootstrap_iterations),
                })
    ci_df = pd.DataFrame(ci_rows)
    summary_df.to_csv(tables / "fusion_performance.csv", index=False, encoding="utf-8-sig")
    dev_df.to_csv(tables / "fusion_development_oof_predictions.csv", index=False, encoding="utf-8-sig")
    temp_df.to_csv(tables / "fusion_temporal_predictions.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(tables / "fusion_input_audit.csv", index=False, encoding="utf-8-sig")
    ci_df.to_csv(tables / "fusion_bootstrap_metric_intervals.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(tables / "fusion_performance.xlsx", engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Performance", index=False); ci_df.to_excel(writer, sheet_name="Bootstrap_CI", index=False); dev_df.to_excel(writer, sheet_name="Development_OOF", index=False); temp_df.to_excel(writer, sheet_name="Temporal", index=False); audit_df.to_excel(writer, sheet_name="Input_audit", index=False)
    LOGGER.info("Completed; results saved to %s", output)

def main() -> None:
    run_fusion(parse_args())

if __name__ == "__main__": main()
