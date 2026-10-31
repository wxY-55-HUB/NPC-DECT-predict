from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .utils import normalize_patient_id, parse_ki67, patient_date

LABEL_COLUMNS = ["PatientID", "Ki67", "Ki67_ge50", "Ki67_gt50"]

def _read_sheet_by_signature(path: Path, required_column: str, preferred: str | None = None) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    if preferred and preferred in workbook.sheet_names:
        return pd.read_excel(path, sheet_name=preferred)
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet, nrows=2)
        if required_column in frame.columns:
            return pd.read_excel(path, sheet_name=sheet)
    raise ValueError(f"No sheet in {path} contains {required_column!r}")

def load_imaging_matrices(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    three_d_path = Path(cfg["paths"]["three_d_matrix"])
    three_slice_path = Path(cfg["paths"]["three_slice_matrix"])

    if three_d_path.suffix.lower() in {".csv", ".gz"}:
        all3d = pd.read_csv(three_d_path, compression="infer", low_memory=False)
        keep = [c for c in all3d.columns if c in LABEL_COLUMNS + ["ExamDate"] or str(c).startswith("Radiomics__")]
        all3d = all3d.loc[:, keep].copy()
    else:
        all3d = None
    if three_d_path.suffix.lower() not in {".csv", ".gz"}:
        for sheet in pd.ExcelFile(three_d_path).sheet_names:
            header = list(pd.read_excel(three_d_path, sheet_name=sheet, nrows=0).columns)
            if "PatientID" not in header:
                continue
            keep = [c for c in header if c in LABEL_COLUMNS or str(c).startswith("Radiomics__")]
            if keep:
                all3d = pd.read_excel(three_d_path, sheet_name=sheet, usecols=keep)
                break
    if all3d is None:
        raise ValueError("Three-dimensional radiomics feature sheet not found")
    if three_slice_path.suffix.lower() in {".csv", ".gz"}:
        three_slice = pd.read_csv(three_slice_path, compression="infer", low_memory=False)
        keep = [c for c in three_slice.columns if c in LABEL_COLUMNS + ["ExamDate"] or str(c).startswith("ThreeSlice__")]
        three_slice = three_slice.loc[:, keep].copy()
    else:
        three_slice = None
        for sheet in pd.ExcelFile(three_slice_path).sheet_names:
            header = list(pd.read_excel(three_slice_path, sheet_name=sheet, nrows=0).columns)
            if any(str(c).startswith("ThreeSlice__") for c in header):
                keep = [c for c in header if c in LABEL_COLUMNS or str(c).startswith("ThreeSlice__")]
                three_slice = pd.read_excel(three_slice_path, sheet_name=sheet, usecols=keep)
                break
    if three_slice is None:
        raise ValueError("Three-slice feature sheet not found")
    for frame in (all3d, three_slice):
        frame["PatientID"] = frame["PatientID"].map(normalize_patient_id)
        frame["Ki67"] = frame["Ki67"].map(parse_ki67)
        frame["ExamDate"] = frame["PatientID"].map(patient_date)
    radiomics3d_cols = LABEL_COLUMNS + ["ExamDate"] + [c for c in all3d if str(c).startswith("Radiomics__")]
    radiomics3d = all3d.loc[:, radiomics3d_cols].copy()
    return {
        "three_slice": three_slice,
        "three_d": radiomics3d.copy(),
        "all3d": radiomics3d,
    }

def _phase_measurements(path: Path, phase: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(path, sheet_name=phase)
    name_col, id_col, ki_col = raw.columns[:3]
    parameter_cols = list(raw.columns[3:])
    measurement_rows: list[dict[str, Any]] = []
    cached_rows: list[dict[str, Any]] = []
    current_id: str | None = None; current_ki = np.nan; block: list[dict[str, Any]] = []
    def flush() -> None:
        nonlocal block
        if not block: return
        if len(block) == 4:
            measurement_rows.extend(block[:3]); cached_rows.append(block[3])
        else:
            measurement_rows.extend(block)
        block = []
    for idx, row in raw.iterrows():
        row_id = normalize_patient_id(row[id_col])
        if row_id and current_id and row_id != current_id:
            flush()
        if row_id:
            current_id = row_id
        parsed = parse_ki67(row[ki_col])
        if np.isfinite(parsed): current_ki = parsed
        if current_id is None or not any(pd.notna(row[col]) for col in parameter_cols):
            continue
        rec = {"PatientID": current_id, "Ki67": current_ki, "source_row": int(idx) + 2}
        rec.update({str(col): pd.to_numeric(row[col], errors="coerce") for col in parameter_cols})
        block.append(rec)
    flush()
    measurements = pd.DataFrame(measurement_rows)
    cached = pd.DataFrame(cached_rows)
    if not measurements.empty:
        measurements = measurements.sort_values("source_row", ascending=False)
        keep_rank = measurements.groupby("PatientID", sort=False).cumcount()
        measurements = measurements.loc[keep_rank < 3].sort_values("source_row").reset_index(drop=True)
    if not cached.empty:
        cached = cached.sort_values("source_row").groupby("PatientID", as_index=False, sort=False).tail(1)
    measurements["replicate"] = measurements.groupby("PatientID").cumcount() + 1
    measurements["phase"] = phase
    return measurements, cached

def _icc_absolute(matrix: np.ndarray) -> tuple[float, float]:
    matrix = np.asarray(matrix, dtype=float)
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    n, k = matrix.shape if matrix.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return np.nan, np.nan
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    msr = k * np.sum((row_means - grand) ** 2) / (n - 1)
    msc = n * np.sum((col_means - grand) ** 2) / (k - 1)
    residual = matrix - row_means[:, None] - col_means[None, :] + grand
    mse = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    icc1 = (msr - mse) / denom if denom else np.nan
    icck = (msr - mse) / (msr + (msc - mse) / n) if (msr + (msc - mse) / n) else np.nan
    return float(icc1), float(icck)

def build_quantitative(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    path = Path(cfg["paths"]["quantitative"])
    if path.suffix.lower() in {".csv", ".gz"}:
        frame = pd.read_csv(path, compression="infer", low_memory=False)
        forbidden_tokens = ("Delta", "Interaction", "LogRatio")
        frame = frame.loc[:, [c for c in frame.columns if not any(token in str(c) for token in forbidden_tokens)]].copy()
        required = {"PatientID", "Ki67"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Prepared quantitative matrix is missing columns: {sorted(missing)}")
        frame["PatientID"] = frame["PatientID"].map(normalize_patient_id)
        frame["Ki67"] = frame["Ki67"].map(parse_ki67)
        if "ExamDate" not in frame:
            frame["ExamDate"] = frame["PatientID"].map(patient_date)
        if "strict_triplicate" not in frame:
            frame["strict_triplicate"] = True
        empty = pd.DataFrame()
        return frame, {"quantitative_measurements": empty, "quantitative_cached_mean_audit": empty, "quantitative_icc": empty}
    phase_sheets = cfg["input_schema"]["quantitative_sheets"]
    phase_map = {phase_sheets["arterial"]: "A", phase_sheets["venous"]: "V"}
    phase_summaries = []
    measurement_audits = []
    cached_audits = []
    icc_rows = []
    for sheet, short in phase_map.items():
        measurements, cached = _phase_measurements(path, sheet)
        measurement_audits.append(measurements)
        parameters = [c for c in measurements if c not in {"PatientID", "Ki67", "source_row", "replicate", "phase"}]
        grouped_rows = []
        for pid, group in measurements.groupby("PatientID", sort=False):
            rec: dict[str, Any] = {"PatientID": pid, f"{short}_n_measurements": len(group), "Ki67": group["Ki67"].dropna().iloc[0] if group["Ki67"].notna().any() else np.nan}
            for col in parameters:
                vals = pd.to_numeric(group[col], errors="coerce")
                rec[f"{short}_mean__{col}"] = vals.mean()
                rec[f"{short}_sd__{col}"] = vals.std(ddof=1)
                rec[f"{short}_cv__{col}"] = vals.std(ddof=1) / abs(vals.mean()) if np.isfinite(vals.mean()) and vals.mean() != 0 else np.nan
            grouped_rows.append(rec)
        summary = pd.DataFrame(grouped_rows)
        phase_summaries.append(summary)

        cached_lookup = cached.set_index("PatientID") if not cached.empty else pd.DataFrame()
        for _, rec in summary.iterrows():
            pid = rec["PatientID"]
            if not cached.empty and pid in cached_lookup.index:
                cached_row = cached_lookup.loc[pid]
                if isinstance(cached_row, pd.DataFrame):
                    cached_row = cached_row.iloc[0]
                diffs = []
                for col in parameters:
                    a = rec[f"{short}_mean__{col}"]
                    b = cached_row[col]
                    if np.isfinite(a) and np.isfinite(b):
                        diffs.append(abs(a - b))
                cached_audits.append({"PatientID": pid, "phase": sheet, "max_abs_difference": max(diffs, default=np.nan), "cached_matches_recomputed": max(diffs, default=0) < 1e-8})

        trip = measurements[measurements["PatientID"].map(measurements["PatientID"].value_counts()) == 3]
        for col in parameters:
            pivot = trip.pivot(index="PatientID", columns="replicate", values=col).reindex(columns=[1, 2, 3])
            icc1, icc3 = _icc_absolute(pivot.to_numpy())
            icc_rows.append({"phase": sheet, "parameter": col, "n_patients": int(pivot.dropna().shape[0]), "ICC_A1": icc1, "ICC_A3": icc3})

    merged = phase_summaries[0].merge(phase_summaries[1], on="PatientID", how="outer", suffixes=("_Aki", "_Vki"))
    merged["Ki67"] = merged.pop("Ki67_Aki").combine_first(merged.pop("Ki67_Vki"))
    merged["ExamDate"] = merged["PatientID"].map(patient_date)
    merged["strict_triplicate"] = (merged["A_n_measurements"] == 3) & (merged["V_n_measurements"] == 3)
    audits = {
        "quantitative_measurements": pd.concat(measurement_audits, ignore_index=True),
        "quantitative_cached_mean_audit": pd.DataFrame(cached_audits),
        "quantitative_icc": pd.DataFrame(icc_rows),
    }
    return merged, audits

def load_clinical(cfg: dict[str, Any]) -> pd.DataFrame:
    path = Path(cfg["paths"]["clinical"])
    schema = cfg["input_schema"]
    columns = schema["clinical_columns"]
    raw = pd.read_excel(path, sheet_name=schema["clinical_sheet"])
    required = list(columns.values())
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Clinical workbook is missing configured columns: {missing}")
    stage_text = raw[columns["clinical_stage"]].fillna("").astype(str)
    imaging_text = raw[columns["imaging_stage"]].fillna("").astype(str)
    combined = stage_text.where(stage_text.str.len() > 0, imaging_text)
    result = pd.DataFrame({
        "PatientID": raw[columns["id"]].map(normalize_patient_id),
        "Name": raw[columns["name"]],
        "Age": raw[columns["age"]],
        "Sex": raw[columns["sex"]],
        "Weight": raw[columns["weight"]],
        "Height": raw[columns["height"]],
        "Ki67_raw": raw[columns["ki67"]],
        "Imaging_stage": raw[columns["imaging_stage"]],
        "Clinical_stage": raw[columns["clinical_stage"]],
    })
    sex_values = schema.get("clinical_sex_values", {"male": "M", "female": "F"})
    sex_map = {str(sex_values["male"]).strip().casefold(): "M",
               str(sex_values["female"]).strip().casefold(): "F"}
    result["Sex"] = result["Sex"].map(lambda value: sex_map.get(str(value).strip().casefold(), value))
    result["Ki67"] = result["Ki67_raw"].map(parse_ki67)
    result["T_stage"] = pd.to_numeric(combined.str.extract(r"T(\d)", expand=False), errors="coerce")
    result["N_stage"] = pd.to_numeric(combined.str.extract(r"N(\d)", expand=False), errors="coerce")
    result["M_stage"] = pd.to_numeric(combined.str.extract(r"M(\d)", expand=False), errors="coerce")
    return result

def verify_dicom_dates(cfg: dict[str, Any]) -> pd.DataFrame:
    source = pd.read_csv(cfg["paths"]["radiomics_source"])
    patient_ids = source["Patient"].astype(str).str.extract(r"(\d{12})", expand=False).rename("PatientID")
    source = pd.concat([source.drop(columns=["PatientID"], errors="ignore"), patient_ids], axis=1)
    rows = []
    try:
        import pydicom
    except Exception as exc:
        return pd.DataFrame([{"status": "pydicom_unavailable", "detail": str(exc)}])
    for pid, group in source.groupby("PatientID"):
        expected = pid[:8]
        dates = set()
        errors = []
        for folder in group["ImageDirectory"].dropna().unique():
            try:
                mapped = Path(str(folder))
                for source_root, target_root in cfg["paths"].get("path_mappings", {}).items():
                    mapped = Path(str(mapped).replace(str(source_root), str(target_root)))
                candidates = [p for p in mapped.rglob("*") if p.is_file() and p.name.upper() != "DICOMDIR"]
                if not candidates:
                    errors.append(f"no_files:{folder}")
                    continue
                ds = pydicom.dcmread(str(candidates[0]), stop_before_pixels=True, force=True)
                date = str(getattr(ds, "StudyDate", ""))
                if date:
                    dates.add(date)
            except Exception as exc:
                errors.append(f"{folder}:{type(exc).__name__}")
        rows.append({"PatientID": pid, "id_date": expected, "dicom_study_dates": ";".join(sorted(dates)), "date_match": bool(dates) and dates == {expected}, "read_errors": " | ".join(errors)})
    return pd.DataFrame(rows)

def make_patient_audit(imaging: dict[str, pd.DataFrame], quantitative: pd.DataFrame, clinical: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    image_ids = set(imaging["all3d"]["PatientID"])
    quant_ids = set(quantitative["PatientID"])
    clinical_ids = set(clinical["PatientID"].dropna())
    all_ids = sorted(image_ids | quant_ids | clinical_ids)
    q = quantitative.set_index("PatientID")
    k = imaging["all3d"].set_index("PatientID")["Ki67"]
    rows = []
    temporal_cutoff = pd.Timestamp(cutoff)
    for pid in all_ids:
        exam = patient_date(pid)
        ki = k.get(pid, q.loc[pid, "Ki67"] if pid in q.index else np.nan)
        strict = bool(q.loc[pid, "strict_triplicate"]) if pid in q.index else False
        rows.append({"PatientID": pid, "ExamDate": exam, "split": "temporal_validation" if exam >= temporal_cutoff else "development",
                     "Ki67": ki, "Ki67_ge50": int(ki >= .5) if np.isfinite(ki) else np.nan,
                     "Ki67_gt50": int(ki > .5) if np.isfinite(ki) else np.nan,
                     "exactly_50": bool(np.isclose(ki, .5)) if np.isfinite(ki) else False,
                     "has_complete_imaging": pid in image_ids, "has_quantitative": pid in quant_ids,
                     "quantitative_strict_triplicate": strict, "has_clinical": pid in clinical_ids,
                     "fair_comparison_eligible": pid in image_ids and strict,
                     "exclusion_reason_fair": "" if pid in image_ids and strict else ("missing_imaging" if pid not in image_ids else "missing_or_nontriplicate_quantitative")})
    return pd.DataFrame(rows)
