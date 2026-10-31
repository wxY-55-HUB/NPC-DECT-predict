# NPC dual-energy CT and Ki-67  code

This repository contains the analysis code and a portable configuration template for a retrospective nasopharyngeal carcinoma study. Patient-level data, prediction records, fitted-model artifacts, logs and figure outputs are excluded.

## Contents

- `reviewer_reanalysis/`: data processing, nested model development, temporal evaluation, statistics, reporting and fusion implementation.
- `config.example.yaml`: portable configuration with relative paths and frozen analysis settings.
- `requirements.txt`: Python dependencies.

Chinese and English Python comments and docstrings were removed from the upload copy. Machine-specific image-root paths were replaced with configurable path mappings. No trained model artifacts are included.

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
