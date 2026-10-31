from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path.resolve())
    return cfg

def project_path(cfg: dict[str, Any], *parts: str) -> Path:
    return Path(cfg["paths"]["project_root"]).joinpath(*parts)

def ensure_output_dirs(cfg: dict[str, Any]) -> None:
    for name in ("data_clean", "results", "figures", "reports", "logs"):
        project_path(cfg, name).mkdir(parents=True, exist_ok=True)

def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def input_paths(cfg: dict[str, Any]) -> list[Path]:
    keys = ["quantitative", "clinical", "three_d_matrix", "three_slice_matrix",
            "radiomics_source"]
    return [Path(cfg["paths"][key]) for key in keys]

def hash_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for path in input_paths(cfg):
        stat = path.stat()
        rows.append({"path": str(path), "size_bytes": stat.st_size,
                     "modified_time": pd.Timestamp(stat.st_mtime, unit="s").isoformat(),
                     "sha256": sha256_file(path)})
    return pd.DataFrame(rows)

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, default=str)

def software_manifest() -> dict[str, Any]:
    import scipy
    import sklearn
    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    for package in ("statsmodels", "matplotlib", "seaborn", "xgboost", "shap", "pydicom", "openpyxl"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "installed")
        except Exception:
            versions[package] = None
    return versions

def normalize_patient_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits if len(digits) >= 8 else None

def patient_date(patient_id: str) -> pd.Timestamp:
    return pd.to_datetime(patient_id[:8], format="%Y%m%d", errors="coerce")

def parse_ki67(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        return number if number <= 1 else number / 100.0
    text = str(value).strip().replace("％", "%").replace("+", "")
    import re
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return np.nan
    number = float(match.group())
    return number / 100.0 if "%" in text or number > 1 else number

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")
