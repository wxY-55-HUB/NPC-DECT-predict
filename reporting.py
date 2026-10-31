from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150, "savefig.dpi": 300})

def save_ki67_distribution(audit: pd.DataFrame, path: Path) -> None:
    setup_style(); data=audit.loc[audit["has_complete_imaging"],"Ki67"].dropna()*100
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].hist(data,bins=np.arange(0,105,5),color=COLORS[0],edgecolor="white")
    axes[0].axvline(50,color=COLORS[1],ls="--",lw=1.5,label="Primary threshold: 50%")
    axes[0].set(xlabel="Ki-67 (%)",ylabel="Patients",title="Ki-67 distribution"); axes[0].legend(frameon=False)
    counts=data.value_counts().sort_index(); axes[1].bar(counts.index,counts.values,width=3.5,color=COLORS[2])
    axes[1].set(xlabel="Observed Ki-67 value (%)",ylabel="Patients",title="Discrete reported values")
    fig.tight_layout(); fig.savefig(path,bbox_inches="tight"); plt.close(fig)

def save_roc(predictions: pd.DataFrame, path: Path, cohort: str) -> None:
    setup_style(); subset=predictions[predictions["cohort"]==cohort]
    fig,ax=plt.subplots(figsize=(6,5))
    for color,(name,group) in zip(COLORS,subset.groupby("analysis_id")):
        if group["y_true"].nunique()<2: continue
        fpr,tpr,_=roc_curve(group["y_true"],group["probability"]); ax.plot(fpr,tpr,color=color,lw=2,label=f"{name} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0,1],[0,1],"--",color="0.5"); ax.set(xlabel="1 - specificity",ylabel="Sensitivity",title=cohort.replace("_"," ").title()); ax.legend(frameon=False,fontsize=7)
    fig.tight_layout(); fig.savefig(path,bbox_inches="tight"); plt.close(fig)

def save_calibration(predictions: pd.DataFrame, path: Path, cohort: str) -> None:
    setup_style(); subset=predictions[predictions["cohort"]==cohort]
    fig,ax=plt.subplots(figsize=(6,5))
    for color,(name,group) in zip(COLORS,subset.groupby("analysis_id")):
        if group["y_true"].nunique()<2: continue
        obs,pred=calibration_curve(group["y_true"],group["probability"],n_bins=min(8,max(3,len(group)//8)),strategy="quantile")
        ax.plot(pred,obs,"o-",color=color,label=name)
    ax.plot([0,1],[0,1],"--",color="0.5"); ax.set(xlabel="Predicted probability",ylabel="Observed proportion",title=f"Calibration: {cohort.replace('_',' ')}"); ax.legend(frameon=False,fontsize=7)
    fig.tight_layout(); fig.savefig(path,bbox_inches="tight"); plt.close(fig)

def save_dca(dca: pd.DataFrame, path: Path, cohort: str) -> None:
    setup_style(); fig,ax=plt.subplots(figsize=(7,5)); subset=dca[dca["cohort"]==cohort]
    for color,(name,group) in zip(COLORS,subset.groupby("analysis_id")):
        ax.plot(group["threshold"],group["net_benefit_model"],color=color,lw=1.8,label=name)
    if not subset.empty:
        first=subset.groupby("threshold",as_index=False).first(); ax.plot(first["threshold"],first["net_benefit_all"],"--",color="0.4",label="Treat all"); ax.axhline(0,color="black",lw=1,label="Treat none")
    ax.set(xlabel="Threshold probability",ylabel="Net benefit",title=f"Exploratory DCA: {cohort.replace('_',' ')}",ylim=(-.15,.5)); ax.legend(frameon=False,fontsize=7,ncol=2)
    fig.tight_layout(); fig.savefig(path,bbox_inches="tight"); plt.close(fig)

def save_three_slice_vs_3d(metrics: pd.DataFrame, path: Path) -> None:
    setup_style(); subset=metrics[(metrics["analysis_id"].isin(["three_slice_fair","three_d_fair","three_slice_max","three_d_max"])) & (metrics.get("strategy","")=="elasticnet_direct")]
    if subset.empty: return
    fig,ax=plt.subplots(figsize=(8,4)); subset=subset.copy(); subset["label"]=subset["analysis_id"]+"\n"+subset["cohort"].str.replace("_"," ")
    ax.bar(np.arange(len(subset)),subset["AUC"],color=[COLORS[i%len(COLORS)] for i in range(len(subset))]); ax.errorbar(np.arange(len(subset)),subset["AUC"],yerr=[subset["AUC"]-subset["AUC_CI_low"],subset["AUC_CI_high"]-subset["AUC"]],fmt="none",ecolor="black",capsize=3)
    ax.set_xticks(np.arange(len(subset)),subset["label"],rotation=25,ha="right"); ax.set(ylabel="AUC",ylim=(0,1),title="Three-slice versus whole-tumor 3D radiomics")
    fig.tight_layout(); fig.savefig(path,bbox_inches="tight"); plt.close(fig)

def save_flow_figure(audit: pd.DataFrame, path: Path) -> None:
    setup_style(); img=audit[audit["has_complete_imaging"]]; fair=audit[audit["fair_comparison_eligible"]]
    lines=[f"Clinical source records\n(n={audit['has_clinical'].sum()})",f"Complete four-channel imaging\n(n={len(img)})",f"Development before 2024\n(n={(img['split']=='development').sum()})",f"Internal temporal validation in 2024\n(n={(img['split']=='temporal_validation').sum()})",f"Strict quantitative triplicates + imaging\n(n={len(fair)})",f"Fair comparison: development / temporal\n(n={(fair['split']=='development').sum()} / {(fair['split']=='temporal_validation').sum()})"]
    fig,ax=plt.subplots(figsize=(9,7)); ax.axis("off")
    positions=[(.5,.9),(.5,.72),(.25,.52),(.75,.52),(.5,.31),(.5,.12)]
    for text,(x,y) in zip(lines,positions): ax.text(x,y,text,ha="center",va="center",bbox=dict(boxstyle="round,pad=.5",fc="#EAF2F8",ec="#0072B2"),transform=ax.transAxes)
    arrows=[((.5,.84),(.5,.78)),((.46,.66),(.28,.58)),((.54,.66),(.72,.58)),((.25,.46),(.45,.37)),((.75,.46),(.55,.37)),((.5,.25),(.5,.18))]
    for a,b in arrows: ax.annotate("",xy=b,xytext=a,xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",color="0.35"))
    fig.savefig(path,bbox_inches="tight"); plt.close(fig)

def literature_evidence_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"topic":"NPC Ki-67 cutoff","citation":"An MRI-based radiomics nomogram for preoperative prediction of Ki-67 index in NPC: a two-center study","year":2025,"identifier":"PMID 39759139; PMCID PMC11695239","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC11695239/","relevance":"Used >=50% for high expression; supports a prespecified primary boundary but not a universal standard."},
        {"topic":"NPC DECT radiomics","citation":"Radiomics based on dual-energy CT for noninvasive prediction of cervical lymph node metastases in NPC","year":2025,"identifier":"DOI 10.1016/j.radi.2025.102989","url":"https://doi.org/10.1016/j.radi.2025.102989","relevance":"Supports multi-phase/dual-energy feature fusion; acquisition semantics must follow the present raw channel metadata."},
        {"topic":"DECT change features","citation":"Dual-energy CT-based radiomics for predicting invasiveness of lung adenocarcinoma appearing as GGNs","year":2023,"identifier":"DOI 10.3389/fonc.2023.1208758; PMID 37637058","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC10449576/","relevance":"Supports prespecified delta radiomics across DECT energy representations."},
        {"topic":"CT habitat stability","citation":"Identification of Precise 3D CT Radiomics for Habitat Computation by Machine Learning in Cancer","year":2024,"identifier":"PMID 38294307; DOI 10.1148/ryai.230118","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC10982821/","relevance":"Supports stability-first habitat features and reporting sensitivity to perturbation parameters."},
        {"topic":"Radiomics standardization","citation":"The Image Biomarker Standardization Initiative","year":2020,"identifier":"PMID 32154773; DOI 10.1148/radiol.2020191145","url":"https://pubmed.ncbi.nlm.nih.gov/32154773/","relevance":"Feature definitions and preprocessing reporting framework."},
        {"topic":"Prediction reporting","citation":"TRIPOD+AI statement","year":2024,"identifier":"DOI 10.1136/bmj-2023-078378","url":"https://www.bmj.com/content/385/bmj-2023-078378","relevance":"Reporting of development, internal validation, performance and open science."},
        {"topic":"Imaging AI reporting","citation":"CLAIM: 2024 Update","year":2024,"identifier":"PMID 38809149; DOI 10.1148/ryai.240300","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/","relevance":"Imaging-specific reporting checklist."},
    ])

def write_methods_and_summary(cfg: dict[str, Any], metrics: pd.DataFrame, audit: pd.DataFrame, path: Path) -> None:
    fair=audit[audit["fair_comparison_eligible"]]; img=audit[audit["has_complete_imaging"]]
    if metrics.empty:
        metric_text="Results have not been generated."
    else:
        shown=metrics.copy()
        cols=list(shown.columns)
        header="| " + " | ".join(cols) + " |"
        rule="| " + " | ".join(["---"]*len(cols)) + " |"
        body=[]
        for row in shown.itertuples(index=False,name=None):
            body.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.4g}" if isinstance(v,float) else str(v)) for v in row) + " |")
        metric_text="\n".join([header,rule,*body])
    labels={"quantitative":"Quantitative parameters","three_slice":"Three-slice radiomics","three_d":"Whole-tumor 3D radiomics"}
    enabled=cfg.get("models",{}).get("enabled_model_types",["quantitative","three_slice","three_d"])
    model_text=", ".join(labels.get(name,name) for name in enabled)
    cutoff = cfg["project"]["temporal_cutoff"]
    lines=["# Analysis methods and results summary","","## Frozen design","",f"The chronological cutoff was {cutoff}. The imaging-complete cohort included {len(img)} patients, and the fair-comparison cohort included {len(fair)} patients.","","The primary endpoint was Ki-67 >=50%. A threshold above 50% was prespecified for boundary sensitivity analysis; 40% and 60% were exploratory thresholds. No threshold was selected by maximizing cohort AUC. Imputation, transformations, scaling, feature filtering, feature selection, hyperparameter tuning and classification-threshold selection were restricted to development data.","",f"The configured model domains were: {model_text}. The source channels A100, A150, V100 and V150 refer to the corresponding arterial/venous acquisition and tube-voltage data as defined by the input matrices.","","Development evaluation used patient-level nested cross-validation. The classification threshold was selected from development out-of-fold predictions and then fixed for temporal validation. Temporal-validation outcomes were not used for model fitting or selection.","","AUC confidence intervals and paired AUC comparisons use the method reported in the corresponding result tables. Bootstrap intervals for other metrics are reported separately.","","## Results","",metric_text]
    path.write_text("\n".join(lines),encoding="utf-8")
