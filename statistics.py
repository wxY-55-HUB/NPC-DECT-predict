from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

def ki67_distribution(audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = audit[audit["has_complete_imaging"] & audit["Ki67"].notna()].copy()
    values = cohort["Ki67"] * 100
    summary = pd.DataFrame([{
        "n": len(values), "median_percent": values.median(), "IQR_low": values.quantile(.25), "IQR_high": values.quantile(.75),
        "minimum": values.min(), "maximum": values.max(), "unique_values": values.nunique(), "exactly_50_n": int(np.isclose(values, 50).sum()),
        "ge50_positive_n": int((values >= 50).sum()), "gt50_positive_n": int((values > 50).sum()),
        "ge40_positive_n": int((values >= 40).sum()), "ge60_positive_n": int((values >= 60).sum()),
    }])
    discrete = values.value_counts().sort_index().rename_axis("Ki67_percent").reset_index(name="n")
    return summary, discrete

def _rank_biserial_auc(values: np.ndarray, y: np.ndarray) -> float:
    pos = values[y == 1]; neg = values[y == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    auc = u / (len(pos) * len(neg))
    return float(2 * auc - 1)

def univariate_feature_statistics(frame: pd.DataFrame, features: list[str], domain: str) -> pd.DataFrame:
    y = frame["Ki67_ge50"].astype(int).to_numpy(); continuous = frame["Ki67"].to_numpy(float); rows = []
    for col in features:
        x = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(continuous)
        if mask.sum() < 10 or np.unique(x[mask]).size < 2: continue
        binary_mask = mask & np.isfinite(y)
        try:
            test = stats.mannwhitneyu(x[binary_mask & (y == 1)], x[binary_mask & (y == 0)], alternative="two-sided")
            p_binary = float(test.pvalue); effect = _rank_biserial_auc(x[binary_mask], y[binary_mask])
        except Exception:
            p_binary = np.nan; effect = np.nan
        spear = stats.spearmanr(x[mask], continuous[mask])
        rows.append({"domain": domain, "feature": col, "n": int(mask.sum()), "missing_n": int((~mask).sum()), "median": float(np.nanmedian(x)), "IQR": float(np.nanquantile(x,.75)-np.nanquantile(x,.25)),
                     "mannwhitney_p": p_binary, "rank_biserial_effect": effect, "spearman_rho_continuous": float(spear.statistic), "spearman_p_continuous": float(spear.pvalue)})
    table = pd.DataFrame(rows)
    if not table.empty:
        for pcol, qcol in (("mannwhitney_p", "mannwhitney_BH_FDR"), ("spearman_p_continuous", "spearman_BH_FDR")):
            valid = table[pcol].notna(); table.loc[valid, qcol] = multipletests(table.loc[valid, pcol], method="fdr_bh")[1]
    return table

def bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y); x=x[mask]; y=y[mask]; rho = stats.spearmanr(x,y).statistic
    rng=np.random.default_rng(seed); values=[]
    for _ in range(n_boot):
        idx=rng.integers(0,len(x),len(x)); val=stats.spearmanr(x[idx],y[idx]).statistic
        if np.isfinite(val): values.append(val)
    return float(rho), float(np.quantile(values,.025)), float(np.quantile(values,.975))

def top_continuous_correlations(feature_stats: pd.DataFrame, frames: dict[str, pd.DataFrame], n_boot: int, seed: int, top_n: int = 20) -> pd.DataFrame:
    rows=[]
    for domain, group in feature_stats.groupby("domain"):
        if domain not in frames: continue
        frame=frames[domain]
        for _, rec in group.reindex(group["spearman_rho_continuous"].abs().sort_values(ascending=False).index).head(top_n).iterrows():
            col=rec["feature"]; rho,low,high=bootstrap_spearman_ci(pd.to_numeric(frame[col],errors="coerce").to_numpy(float), frame["Ki67"].to_numpy(float), n_boot, seed)
            rows.append({"domain":domain,"feature":col,"n":int(rec["n"]),"Spearman_rho":rho,"bootstrap_CI_low":low,"bootstrap_CI_high":high,"BH_FDR":rec.get("spearman_BH_FDR",np.nan)})
    return pd.DataFrame(rows)

def clinical_descriptive_and_stage(clinical: pd.DataFrame, audit: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    merged=audit[audit["has_complete_imaging"]].merge(clinical,on="PatientID",how="left",suffixes=("","_clinical"))
    rows=[]
    for split, group in merged.groupby("split"):
        rows.append({"split":split,"n":len(group),"age_median":pd.to_numeric(group["Age"],errors="coerce").median(),"age_IQR_low":pd.to_numeric(group["Age"],errors="coerce").quantile(.25),"age_IQR_high":pd.to_numeric(group["Age"],errors="coerce").quantile(.75),"male_n":int((group["Sex"]=="M").sum()),"female_n":int(group["Sex"].isin(["F", "W"]).sum()),"T_stage_missing":int(group["T_stage"].isna().sum()),"N_stage_missing":int(group["N_stage"].isna().sum())})
    tests=[]
    for variable in ("T_stage","N_stage"):
        sub=merged[[variable,"Ki67","Ki67_ge50"]].dropna()
        rho=stats.spearmanr(sub[variable],sub["Ki67"])
        table=pd.crosstab(sub[variable],sub["Ki67_ge50"])
        chi=stats.chi2_contingency(table) if table.shape[0]>1 and table.shape[1]>1 else (np.nan,np.nan,np.nan,np.nan)
        tests.append({"variable":variable,"n":len(sub),"Spearman_rho_with_continuous_Ki67":rho.statistic,"Spearman_p":rho.pvalue,"chi_square":chi[0],"chi_square_p":chi[1]})
    return pd.DataFrame(rows),pd.DataFrame(tests)

def stage_adjusted_scores(predictions: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.api as sm
    data=predictions.merge(clinical[["PatientID","T_stage","N_stage"]],on="PatientID",how="left").dropna(subset=["y_true","probability","T_stage","N_stage"])
    rows=[]
    grouping=[c for c in ("analysis_id","strategy","cohort") if c in data.columns]
    for key,group in data.groupby(grouping,dropna=False):
        if not isinstance(key,tuple): key=(key,)
        ident=dict(zip(grouping,key))
        if group["y_true"].nunique()<2 or len(group)<10: continue
        X=sm.add_constant(group[["probability","T_stage","N_stage"]].astype(float))
        try:
            fit=sm.Logit(group["y_true"].astype(int),X).fit(disp=0)
            rows.append({**ident,"n":len(group),"score_adjusted_OR":float(np.exp(fit.params["probability"])),"score_adjusted_CI_low":float(np.exp(fit.conf_int().loc["probability",0])),"score_adjusted_CI_high":float(np.exp(fit.conf_int().loc["probability",1])),"score_adjusted_p":float(fit.pvalues["probability"]),"T_coefficient":float(fit.params["T_stage"]),"N_coefficient":float(fit.params["N_stage"])})
        except Exception as exc:
            rows.append({**ident,"n":len(group),"error":type(exc).__name__})
    return pd.DataFrame(rows)
