# NPC dual-energy CT and Ki-67 reanalysis code

This repository contains the analysis code and a portable configuration template for a retrospective nasopharyngeal carcinoma study. Patient-level data, prediction records, fitted-model artifacts, logs and figure outputs are excluded.

## Contents

- `reviewer_reanalysis/`: data processing, nested model development, temporal evaluation, statistics, reporting and fusion implementation.
- `config.example.yaml`: portable configuration with relative paths and frozen analysis settings.
- `requirements.txt`: Python dependencies.
- `data/radiomics_source.example.csv`: header-only placeholder.

Chinese and English Python comments and docstrings were removed from the upload copy. Machine-specific image-root paths were replaced with configurable path mappings. No trained model artifacts are included.

## Data required

Patient-level imaging, clinical and feature-matrix data are not distributed because they contain identifying information and are subject to institutional restrictions. Before running, place approved local input files in a secure location and update the paths in `config.example.yaml`. Required inputs are a quantitative-measurement workbook (arterial and venous phase sheets) or prepared quantitative CSV, a clinical workbook, a three-dimensional radiomics matrix with `PatientID`, `Ki67` and `Radiomics__*` columns, and a three-slice matrix with `PatientID`, `Ki67` and `ThreeSlice__*` columns. Workbook sheet names, clinical column names and raw sex labels are configurable under `input_schema`; map these settings to the labels in your local files. The analysis converts the clinical workbook fields to standardized internal names. DICOM-date auditing remains configurable through local image-index paths. ROI perturbation code was omitted because it is not part of the reported analysis.

Do not commit source images, masks, clinical workbooks, patient-level feature matrices, patient-level predictions, logs, or derived outputs. Review staged files before publishing.

## Environment and execution

Use Python 3.10 or newer. From the repository root, install dependencies and run:

```bash
python -m venv .venv
Activate the environment, then install dependencies:
python -m pip install -r requirements.txt
python -m reviewer_reanalysis run --config config.example.yaml
```

The analysis uses the 1 August 2023 chronological cutoff. Preprocessing, feature selection, tuning and threshold selection are fit using development data; the later cohort is treated as internal temporal validation. Output folders are ignored by Git.

## License

No reuse license has been assigned. Select and add a license before release if reuse permissions are intended.
