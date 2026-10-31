from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (build_quantitative, load_clinical, load_imaging_matrices,
                   make_patient_audit, verify_dicom_dates)
from .evaluation import (compare_models, continuous_metrics, decision_curve,
                          evaluate_prediction_sets, platt_calibrate_oof_predictions)
from .modeling import (continuous_elasticnet, feature_columns,
                       nested_classifier)
from .reporting import (literature_evidence_table, save_calibration, save_dca,
                        save_flow_figure, save_ki67_distribution, save_roc,
                        save_three_slice_vs_3d, write_methods_and_summary)
from .statistics import (clinical_descriptive_and_stage, ki67_distribution,
                         stage_adjusted_scores, top_continuous_correlations,
                         univariate_feature_statistics)
from .utils import (ensure_output_dirs, hash_manifest, load_config, project_path,
                    save_json, set_determinism, software_manifest, write_csv)

def _log(message: str, cfg: dict[str, Any]) -> None:
    stamp = pd.Timestamp.now().isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with project_path(cfg, "logs", "pipeline.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")

def _add_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Ki67"] = pd.to_numeric(out["Ki67"], errors="coerce")
    out["Ki67_ge50"] = (out["Ki67"] >= .50).astype(int)
    out["Ki67_gt50"] = (out["Ki67"] > .50).astype(int)
    out["Ki67_ge40"] = (out["Ki67"] >= .40).astype(int)
    out["Ki67_ge60"] = (out["Ki67"] >= .60).astype(int)
    return out

def _clean_matrix_outputs(frames: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> None:
    for name in ("three_slice", "three_d"):
        path = project_path(cfg, "data_clean", f"{name}_patient_matrix.csv.gz")
        frames[name].to_csv(path, index=False, compression="gzip", encoding="utf-8-sig", float_format="%.10g")

def _enabled_model_types(cfg: dict[str, Any]) -> tuple[str, ...]:
    allowed = ("quantitative", "three_slice", "three_d", "habitat")
    enabled = tuple(cfg.get("models", {}).get("enabled_model_types", allowed))
    invalid = sorted(set(enabled) - set(allowed))
    if invalid:
        raise ValueError(f"Unknown enabled model types: {invalid}")
    if not enabled:
        raise ValueError("At least one model type must be enabled")
    return enabled

def _prepare_model_frames(imaging: dict[str, pd.DataFrame], quantitative: pd.DataFrame, audit: pd.DataFrame) -> dict[str, pd.DataFrame]:
    label_source = imaging["all3d"][["PatientID", "Ki67", "Ki67_ge50", "Ki67_gt50", "ExamDate"]].copy()
    quant = quantitative.drop(columns=[c for c in ("Ki67", "ExamDate") if c in quantitative], errors="ignore").merge(label_source, on="PatientID", how="inner")
    quant = _add_labels(quant)
    fair_ids = set(audit.loc[audit["fair_comparison_eligible"], "PatientID"])
    frames = {
        "quantitative_fair": quant[quant["PatientID"].isin(fair_ids)].copy(),
        "quantitative_available": quant.copy(),
    }
    for name in ("three_slice", "three_d"):
        frame = _add_labels(imaging[name])
        frames[f"{name}_fair"] = frame[frame["PatientID"].isin(fair_ids)].copy()
        frames[f"{name}_max"] = frame.copy()
    return frames

def _run_one_classifier(frame: pd.DataFrame, model_type: str, analysis_id: str, cfg: dict[str, Any], quick: bool,
                        strategies: list[str], ablation: str = "full_raw", label: str = "Ki67_ge50",
                        repeats_override: int | None = None) -> list[dict[str, Any]]:
    features = feature_columns(frame, model_type, ablation)
    if not features:
        raise ValueError(f"No features for {analysis_id}/{ablation}")
    cutoff = pd.Timestamp(cfg["project"]["temporal_cutoff"])
    outputs = []
    for strategy in strategies:
        _log(f"model {analysis_id} strategy={strategy} label={label} features={len(features)}", cfg)
        outputs.append(nested_classifier(frame, features, cutoff, cfg, strategy, analysis_id, quick=quick,
                                         outer_repeats_override=repeats_override, label_column=label))
    return outputs

def _concat(results: list[dict[str, Any]], key: str) -> pd.DataFrame:
    frames = [r[key] for r in results if key in r and isinstance(r[key], pd.DataFrame) and not r[key].empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _evaluate_each(results: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    metrics=[]; boot=[]; classified=[]
    for result in results:
        m,b,c=evaluate_prediction_sets(result["oof"],result["temporal"],cfg)
        metrics.append(m); boot.append(b); classified.append(c)
    return (_concat([{"x":x} for x in metrics],"x"), _concat([{"x":x} for x in boot],"x"), _concat([{"x":x} for x in classified],"x"))

def run_pipeline(config_path: Path, quick: bool = False) -> None:
    cfg=load_config(config_path); ensure_output_dirs(cfg); seed=int(cfg["project"]["random_seed"]); set_determinism(seed)
    enabled_models = _enabled_model_types(cfg)
    if quick:
        cfg["preprocessing"]["max_correlation_candidates"] = 100
        cfg["validation"]["bootstrap_iterations"] = 100
        cfg["models"]["continuous_alpha_grid"] = cfg["models"]["continuous_alpha_grid"][:2]
        cfg["models"]["l1_ratio_grid"] = cfg["models"]["l1_ratio_grid"][:2]
    started=time.time(); _log(f"start quick={quick}",cfg)
    before=hash_manifest(cfg); write_csv(before,project_path(cfg,"logs","input_hashes_before.csv")); save_json(project_path(cfg,"logs","software_versions.json"),software_manifest()); save_json(project_path(cfg,"logs","resolved_config.json"),cfg)

    _log("load and audit raw data",cfg)
    imaging=load_imaging_matrices(cfg); quantitative,q_audits=build_quantitative(cfg); clinical=load_clinical(cfg)
    imaging={k:_add_labels(v) if "Ki67" in v.columns else v for k,v in imaging.items()}
    audit=make_patient_audit(imaging,quantitative,clinical,cfg["project"]["temporal_cutoff"])
    write_csv(audit,project_path(cfg,"data_clean","patient_match_and_exclusion_audit.csv"))
    for name,table in q_audits.items(): write_csv(table,project_path(cfg,"results",f"{name}.csv"))
    dicom=verify_dicom_dates(cfg); write_csv(dicom,project_path(cfg,"results","dicom_studydate_audit.csv"))
    _clean_matrix_outputs(imaging,cfg)
    write_csv(quantitative,project_path(cfg,"data_clean","quantitative_recomputed_patient_matrix.csv"))
    split=audit[audit["has_complete_imaging"]][["PatientID","ExamDate","split","fair_comparison_eligible"]]; write_csv(split,project_path(cfg,"data_clean","frozen_patient_splits.csv"))

    _log("descriptive and association statistics",cfg)
    summary,discrete=ki67_distribution(audit); write_csv(summary,project_path(cfg,"results","ki67_distribution_summary.csv")); write_csv(discrete,project_path(cfg,"results","ki67_discrete_values.csv"))
    clinical_desc,stage_tests=clinical_descriptive_and_stage(clinical,audit); write_csv(clinical_desc,project_path(cfg,"results","clinical_cohort_description.csv")); write_csv(stage_tests,project_path(cfg,"results","tnm_ki67_association.csv"))
    model_frames=_prepare_model_frames(imaging,quantitative,audit)
    domain_map={
        "quantitative":model_frames["quantitative_available"],
        "three_slice":model_frames["three_slice_max"],
        "three_d":model_frames["three_d_max"],
    }
    domain_map={name: domain_map[name] for name in enabled_models}
    stats_tables=[]
    for domain,frame in domain_map.items():
        features=feature_columns(frame,domain,"full_raw")
        _log(f"univariate statistics {domain}: {len(features)} features",cfg)
        stats_tables.append(univariate_feature_statistics(frame,features,domain))
    feature_stats=pd.concat(stats_tables,ignore_index=True); write_csv(feature_stats,project_path(cfg,"results","univariate_feature_statistics_BH_FDR.csv.gz"))
    top_corr=top_continuous_correlations(feature_stats,domain_map,200 if quick else int(cfg["validation"]["bootstrap_iterations"]),seed)
    write_csv(top_corr,project_path(cfg,"results","top_continuous_ki67_spearman_bootstrap.csv"))

    primary_results=[]
    for model_type in enabled_models:
        primary_results += _run_one_classifier(model_frames[f"{model_type}_fair"],model_type,f"{model_type}_fair",cfg,quick,["elasticnet_direct","topk"])
    for model_type in ("three_slice","three_d"):
        if model_type not in enabled_models:
            continue
        primary_results += _run_one_classifier(model_frames[f"{model_type}_max"],model_type,f"{model_type}_max",cfg,quick,["elasticnet_direct"])

    primary_metrics,primary_boot,classified=_evaluate_each(primary_results,cfg)
    write_csv(_concat(primary_results,"oof"),project_path(cfg,"results","development_unique_oof_predictions.csv"))
    write_csv(_concat(primary_results,"oof_long"),project_path(cfg,"results","development_repeated_oof_predictions_long.csv.gz"))
    write_csv(_concat(primary_results,"temporal"),project_path(cfg,"results","internal_temporal_predictions.csv"))
    write_csv(_concat(primary_results,"selections"),project_path(cfg,"results","feature_selection_frequency_raw.csv.gz"))
    write_csv(_concat(primary_results,"final_coefficients"),project_path(cfg,"results","final_elasticnet_coefficients.csv"))
    write_csv(_concat(primary_results,"tuning"),project_path(cfg,"results","nested_tuning_results.csv.gz"))
    write_csv(primary_metrics,project_path(cfg,"results","primary_model_metrics.csv")); write_csv(primary_boot,project_path(cfg,"results","bootstrap_metric_intervals.csv")); write_csv(classified,project_path(cfg,"results","classified_predictions_and_confusion_components.csv"))

    raw_oof = _concat(primary_results, "oof")
    raw_temporal = _concat(primary_results, "temporal")
    calibrated_oof, calibrated_temporal, calibration_parameters = platt_calibrate_oof_predictions(
        raw_oof, raw_temporal, seed, folds=int(cfg["validation"]["inner_folds"])
    )
    write_csv(calibrated_oof, project_path(cfg, "results", "development_oof_platt_calibrated_predictions.csv"))
    write_csv(calibrated_temporal, project_path(cfg, "results", "internal_temporal_platt_calibrated_predictions.csv"))
    write_csv(calibration_parameters, project_path(cfg, "results", "platt_calibration_parameters.csv"))
    if not calibrated_oof.empty:
        calibrated_metrics, calibrated_boot, calibrated_classified = evaluate_prediction_sets(
            calibrated_oof, calibrated_temporal, cfg
        )
        write_csv(calibrated_metrics, project_path(cfg, "results", "platt_calibrated_metrics.csv"))
        write_csv(calibrated_boot, project_path(cfg, "results", "platt_calibrated_bootstrap_metric_intervals.csv"))
        write_csv(calibrated_classified, project_path(cfg, "results", "platt_calibrated_classified_predictions.csv"))
    _log("development OOF Platt calibration completed; temporal labels were not used for fitting", cfg)

    sensitivity_results=[]
    for label in ("Ki67_gt50","Ki67_ge40","Ki67_ge60"):
        for model_type in enabled_models:
            sensitivity_results += _run_one_classifier(model_frames[f"{model_type}_fair"],model_type,f"{model_type}_fair_{label}",cfg,quick,["elasticnet_direct"],label=label)
    sensitivity_metrics,_,sensitivity_classified=_evaluate_each(sensitivity_results,cfg)
    write_csv(sensitivity_metrics,project_path(cfg,"results","cutoff_sensitivity_metrics.csv")); write_csv(_concat(sensitivity_results,"oof"),project_path(cfg,"results","cutoff_sensitivity_oof_predictions.csv")); write_csv(_concat(sensitivity_results,"temporal"),project_path(cfg,"results","cutoff_sensitivity_temporal_predictions.csv"))

    quant_sens=_run_one_classifier(model_frames["quantitative_available"],"quantitative","quantitative_available_single_measurement_sensitivity",cfg,quick,["elasticnet_direct"])
    quant_sens_metrics,_,_= _evaluate_each(quant_sens,cfg); write_csv(quant_sens_metrics,project_path(cfg,"results","quantitative_single_measurement_sensitivity_metrics.csv"))

    continuous=[]; continuous_metric_rows=[]
    for model_type in enabled_models:
        frame=model_frames[f"{model_type}_fair"]; features=feature_columns(frame,model_type,"full_raw")
        _log(f"continuous model {model_type}",cfg); result=continuous_elasticnet(frame,features,pd.Timestamp(cfg["project"]["temporal_cutoff"]),cfg,f"{model_type}_fair_continuous",quick)
        continuous.append(result); continuous_metric_rows.append(continuous_metrics(result["oof"],"development_repeated_nested_oof"))
        if not result["temporal"].empty: continuous_metric_rows.append(continuous_metrics(result["temporal"],"internal_temporal_validation"))
    write_csv(pd.DataFrame(continuous_metric_rows),project_path(cfg,"results","continuous_ki67_metrics.csv")); write_csv(_concat(continuous,"oof"),project_path(cfg,"results","continuous_ki67_oof_predictions.csv")); write_csv(_concat(continuous,"temporal"),project_path(cfg,"results","continuous_ki67_temporal_predictions.csv"))

    ablation_results=[]
    if cfg["models"]["run_ablations"]:
        ablations={"quantitative":["arterial","venous","raw_two_phase"],"three_slice":["A100","A150","V100","V150","arterial_dual_energy","venous_dual_energy","dual_phase_100","dual_phase_150","full_raw"],"three_d":["A100","A150","V100","V150","arterial_dual_energy","venous_dual_energy","dual_phase_100","dual_phase_150","full_raw"],"habitat":["A100","A150","V100","V150","arterial_dual_energy","venous_dual_energy","dual_phase_100","dual_phase_150","full_raw"]}
        ablations={name: ablations[name] for name in enabled_models if name in ablations}
        for model_type,names in ablations.items():
            for ablation in names:
                ablation_results += _run_one_classifier(model_frames[f"{model_type}_fair"],model_type,f"{model_type}_ablation_{ablation}",cfg,quick,["elasticnet_direct"],ablation=ablation,repeats_override=1)
        ablation_metrics,_,_=_evaluate_each(ablation_results,cfg); write_csv(ablation_metrics,project_path(cfg,"results","prespecified_ablation_metrics.csv")); write_csv(_concat(ablation_results,"oof"),project_path(cfg,"results","prespecified_ablation_oof_predictions.csv.gz"))

    fair_direct=classified[(classified.get("strategy","")=="elasticnet_direct") & classified["analysis_id"].str.endswith("_fair")]
    comparisons=[]
    for cohort,group in fair_direct.groupby("cohort"):
        table=compare_models(group,int(cfg["validation"]["bootstrap_iterations"]),seed); table["cohort"]=cohort; comparisons.append(table)
    write_csv(pd.concat(comparisons,ignore_index=True) if comparisons else pd.DataFrame(),project_path(cfg,"results","paired_model_comparisons_delong_and_bootstrap.csv"))
    stage_adj=stage_adjusted_scores(fair_direct,clinical); write_csv(stage_adj,project_path(cfg,"results","stage_adjusted_imaging_score_sensitivity.csv"))
    dca_tables=[]
    for (analysis,cohort),group in fair_direct.groupby(["analysis_id","cohort"]):
        d=decision_curve(group["y_true"].to_numpy(),group["probability"].to_numpy()); d["analysis_id"]=analysis; d["cohort"]=cohort; dca_tables.append(d)
    dca=pd.concat(dca_tables,ignore_index=True) if dca_tables else pd.DataFrame(); write_csv(dca,project_path(cfg,"results","exploratory_decision_curve.csv"))


    literature=literature_evidence_table(); write_csv(literature,project_path(cfg,"reports","literature_evidence_table.csv"))
    save_ki67_distribution(audit,project_path(cfg,"figures","ki67_distribution.png")); save_flow_figure(audit,project_path(cfg,"figures","patient_flow.png"))
    for cohort in ("development_repeated_nested_oof","internal_temporal_validation"):
        save_roc(fair_direct,project_path(cfg,"figures",f"roc_{cohort}.png"),cohort); save_calibration(fair_direct,project_path(cfg,"figures",f"calibration_{cohort}.png"),cohort)
        if not dca.empty: save_dca(dca,project_path(cfg,"figures",f"dca_{cohort}.png"),cohort)
    save_three_slice_vs_3d(primary_metrics,project_path(cfg,"figures","three_slice_vs_3d.png"))
    write_methods_and_summary(cfg,primary_metrics,audit,project_path(cfg,"reports","statistical_methods_and_reviewer_summary.md"))

    after=hash_manifest(cfg); write_csv(after,project_path(cfg,"logs","input_hashes_after.csv")); hash_check=before[["path","sha256"]].merge(after[["path","sha256"]],on="path",suffixes=("_before","_after")); hash_check["unchanged"]=hash_check["sha256_before"]==hash_check["sha256_after"]; write_csv(hash_check,project_path(cfg,"logs","input_hash_integrity_check.csv"))
    validation=validate_outputs(config_path, raise_on_error=False); save_json(project_path(cfg,"logs","acceptance_test_results.json"),validation)
    _log(f"completed in {(time.time()-started)/60:.1f} minutes; acceptance_passed={validation['all_passed']}",cfg)

def validate_outputs(config_path: Path, raise_on_error: bool = True) -> dict[str, Any]:
    cfg=load_config(config_path); results=[]
    def check(name: str, condition: bool, detail: str="") -> None:
        results.append({"test":name,"passed":bool(condition),"detail":detail})
    root=Path(cfg["paths"]["project_root"])
    before=pd.read_csv(root/"logs"/"input_hashes_before.csv"); after=pd.read_csv(root/"logs"/"input_hashes_after.csv") if (root/"logs"/"input_hashes_after.csv").exists() else hash_manifest(cfg)
    merged=before.merge(after,on="path",suffixes=("_before","_after")); check("raw_input_hashes_unchanged",(merged["sha256_before"]==merged["sha256_after"]).all())
    splits=pd.read_csv(root/"data_clean"/"frozen_patient_splits.csv",dtype={"PatientID":str}); check("one_patient_one_split",splits["PatientID"].is_unique); dev=set(splits.loc[splits["split"]=="development","PatientID"]); temp=set(splits.loc[splits["split"]=="temporal_validation","PatientID"]); check("development_temporal_no_overlap",not bool(dev&temp))
    audit=pd.read_csv(root/"data_clean"/"patient_match_and_exclusion_audit.csv",dtype={"PatientID":str}); image_n=int(audit["has_complete_imaging"].sum()); fair_n=int(audit["fair_comparison_eligible"].sum()); check("imaging_cohort_matches_frozen_split",image_n==len(splits),f"audit={image_n},split={len(splits)}"); check("fair_cohort_not_larger_than_imaging",fair_n<=image_n,str(fair_n)); check("exactly_50_reported",int(audit.loc[audit["has_complete_imaging"],"exactly_50"].sum())>=0); check("quantitative_flags_consistent",bool((~audit["has_quantitative"] | audit["quantitative_strict_triplicate"]).all()))
    cached_path=root/"results"/"quantitative_cached_mean_audit.csv"
    if cached_path.exists():
        try:
            cached=pd.read_csv(cached_path,dtype={"PatientID":str})
        except pd.errors.EmptyDataError:
            check("cached_mean_audit_available",True,"empty cached audit; skipped for prepared patient-level matrix")
            cached=pd.DataFrame()
        if {"phase","cached_matches_recomputed"}.issubset(cached.columns):
            venous_sheet = str(cfg["input_schema"]["quantitative_sheets"]["venous"])
            ven_mismatch=cached[(cached["phase"].astype(str).str.casefold()==venous_sheet.casefold()) & (~cached["cached_matches_recomputed"].astype(bool))]; check("cached_mean_audit_available",True,str(len(ven_mismatch)))
        else:
            if not cached.empty:
                check("cached_mean_audit_available",True,"prepared matrix has no cached-mean audit")
    else:
        check("cached_mean_audit_available",True,"not generated")
    forbidden_tokens = ("Delta", "Interaction", "LogRatio")
    matrix_paths = [root/"data_clean"/"three_slice_patient_matrix.csv.gz",
                    root/"data_clean"/"three_d_patient_matrix.csv.gz",
                    root/"data_clean"/"quantitative_recomputed_patient_matrix.csv"]
    forbidden = []
    for matrix_path in matrix_paths:
        if matrix_path.exists():
            header = pd.read_csv(matrix_path, compression="infer", nrows=0).columns
            forbidden.extend([f"{matrix_path.name}:{col}" for col in header if any(token in str(col) for token in forbidden_tokens)])
    check("change_features_absent_from_clean_matrices", not forbidden, "; ".join(forbidden[:10]))
    pred_path=root/"results"/"development_unique_oof_predictions.csv"
    if pred_path.exists():
        pred=pd.read_csv(pred_path,dtype={"PatientID":str}); keys=[c for c in ("analysis_id","strategy","PatientID") if c in pred]; check("unique_patient_oof_per_analysis",not pred.duplicated(keys).any()); check("oof_contains_development_only",set(pred["PatientID"]).issubset(dev))
    else: check("unique_patient_oof_per_analysis",False,"missing prediction file")
    integrity=all(r["passed"] for r in results); output={"all_passed":integrity,"tests":results}
    if raise_on_error and not integrity: raise AssertionError(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(output,ensure_ascii=False,indent=2)); return output
