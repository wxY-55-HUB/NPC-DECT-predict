from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .data import (build_quantitative, load_clinical, load_imaging_matrices,
                   make_patient_audit)
from .evaluation import evaluate_prediction_sets
from .modeling import (feature_columns, nested_fusion_matched_classifier)
from .pipeline import _add_labels, _prepare_model_frames
from .utils import load_config, save_json, set_determinism, write_csv

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only the matched-algorithm independent-model analysis")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to the project configuration file")
    parser.add_argument(
        "--output-dir",
        default="results/independent_fusion_matched_only",
        help="Output directory for the standalone supplementary analysis; relative paths use project_root",
    )
    parser.add_argument(
        "--algorithms",
        default=None,
        help="Comma-separated algorithms; defaults to models.fusion_matched_algorithms",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model domains; defaults to enabled_model_types; habitat is not supported",
    )
    parser.add_argument("--quick", action="store_true", help="Run a quick smoke check; do not use for final results")
    return parser.parse_args()

def _output_path(project_root: Path, output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = project_root / path
    path.mkdir(parents=True, exist_ok=True)
    return path

def _enabled_models(cfg: dict[str, Any], override: str | None) -> list[str]:
    names = [x.strip() for x in override.split(",") if x.strip()] if override else list(
        cfg.get("models", {}).get("enabled_model_types", ["quantitative", "three_slice", "three_d"])
    )
    allowed = {"quantitative", "three_slice", "three_d"}
    invalid = sorted(set(names) - allowed)
    if invalid:
        raise ValueError(f"Unsupported model domains for the standalone analysis: {invalid}")
    if not names:
        raise ValueError("At least one model domain must be selected")
    return names

def _algorithms(cfg: dict[str, Any], override: str | None) -> list[str]:
    names = [x.strip() for x in override.split(",") if x.strip()] if override else list(
        cfg.get("models", {}).get("fusion_matched_algorithms", ["elasticnet", "randomforest"])
    )
    aliases = {"random_forest": "randomforest"}
    names = [aliases.get(x, x) for x in names]
    allowed = {"elasticnet", "randomforest", "xgboost"}
    invalid = sorted(set(names) - allowed)
    if invalid:
        raise ValueError(f"Unsupported matched algorithms: {invalid}")
    if not names:
        raise ValueError("At least one algorithm must be selected")
    return names

def run_standalone(config_path: Path, output_dir: str, algorithms: str | None = None,
                   models: str | None = None, quick: bool = False) -> Path:
    cfg = load_config(config_path)
    seed = int(cfg["project"]["random_seed"])
    set_determinism(seed)
    if quick:
        cfg["validation"]["bootstrap_iterations"] = min(100, int(cfg["validation"]["bootstrap_iterations"]))

    project_root = Path(cfg["paths"]["project_root"])
    out = _output_path(project_root, output_dir)
    model_names = _enabled_models(cfg, models)
    algorithm_names = _algorithms(cfg, algorithms)
    started = time.time()

    imaging = load_imaging_matrices(cfg)
    quantitative, _ = build_quantitative(cfg)
    clinical = load_clinical(cfg)
    imaging = {k: _add_labels(v) if "Ki67" in v.columns else v for k, v in imaging.items()}
    audit = make_patient_audit(imaging, quantitative, clinical, cfg["project"]["temporal_cutoff"])
    frames = _prepare_model_frames(imaging, quantitative, audit)

    audit_out = audit[[
        "PatientID", "ExamDate", "split", "has_complete_imaging",
        "has_quantitative", "quantitative_strict_triplicate",
        "fair_comparison_eligible",
    ]].copy()
    write_csv(audit_out, out / "independent_fusion_matched_input_audit.csv")

    results: list[dict[str, Any]] = []
    for model_type in model_names:
        frame_key = f"{model_type}_fair"
        if frame_key not in frames:
            raise ValueError(f"Fair-comparison cohort not found: {frame_key}")
        frame = frames[frame_key]
        features = feature_columns(frame, model_type, "full_raw")
        if not features:
            raise ValueError(f"No usable features for model domain: {model_type}")
        for algorithm in algorithm_names:
            analysis_id = f"{model_type}_fair"
            print(
                f"[independent-matched] {model_type} / {algorithm}; "
                f"n={len(frame)}, features={len(features)}",
                flush=True,
            )
            results.append(nested_fusion_matched_classifier(
                frame=frame,
                features=features,
                temporal_cutoff=pd.Timestamp(cfg["project"]["temporal_cutoff"]),
                cfg=cfg,
                analysis_id=analysis_id,
                algorithm=algorithm,
                quick=quick,
            ))

    oof = pd.concat([r["oof"] for r in results], ignore_index=True)
    oof_long = pd.concat([r["oof_long"] for r in results], ignore_index=True)
    temporal = pd.concat([r["temporal"] for r in results], ignore_index=True)
    importance = pd.concat([r["importances"] for r in results], ignore_index=True)
    metrics, bootstrap, classified = evaluate_prediction_sets(oof, temporal, cfg)

    write_csv(oof, out / "independent_fusion_matched_oof_predictions.csv")
    write_csv(oof_long, out / "independent_fusion_matched_oof_predictions_long.csv")
    write_csv(temporal, out / "independent_fusion_matched_temporal_predictions.csv")
    write_csv(metrics, out / "independent_fusion_matched_metrics.csv")
    write_csv(bootstrap, out / "independent_fusion_matched_bootstrap_metric_intervals.csv")
    write_csv(classified, out / "independent_fusion_matched_classified_predictions.csv")
    write_csv(importance, out / "independent_fusion_matched_feature_importance.csv")

    manifest = {
        "analysis_type": "standalone_independent_fusion_matched",
        "full_pipeline_called": False,
        "config": str(config_path.resolve()),
        "output_dir": str(out),
        "models": model_names,
        "algorithms": algorithm_names,
        "temporal_cutoff": cfg["project"]["temporal_cutoff"],
        "development_n": int((audit["split"] == "development").sum()),
        "temporal_validation_n": int((audit["split"] == "temporal_validation").sum()),
        "quick": bool(quick),
        "random_seed": seed,
        "elapsed_minutes": round((time.time() - started) / 60, 3),
    }
    save_json(out / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return out

def main() -> None:
    args = _parse_args()
    run_standalone(args.config, args.output_dir, args.algorithms, args.models, args.quick)

if __name__ == "__main__":
    main()
