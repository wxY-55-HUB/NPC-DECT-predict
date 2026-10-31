from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline, validate_outputs

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NPC DECT Ki-67 reviewer reanalysis")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the complete frozen pipeline")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--quick", action="store_true", help="small deterministic smoke run")
    val = sub.add_parser("validate", help="run acceptance checks on frozen outputs")
    val.add_argument("--config", required=True, type=Path)
    fusion = sub.add_parser("fusion", help="run the independent quantitative-radiomics fusion analysis")
    fusion.add_argument("--quick", action="store_true", help="small deterministic smoke run")
    fusion.add_argument("--models", default="elasticnet,randomforest", help="elasticnet,randomforest,xgboost")
    fusion.add_argument("--periods", default=None, help="comma-separated phase/energy combinations")
    fusion.add_argument("--fusion-methods", default=None, help="feature_level,late_stack,soft_vote")
    fusion.add_argument("--output-dir", default=None, help="output directory relative to the project root")
    return parser

def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run_pipeline(args.config, quick=args.quick)
    elif args.command == "validate":
        validate_outputs(args.config)
    else:
        from .multimodel_fusion import Config as FusionConfig, run_fusion

        cfg = FusionConfig(models=tuple(x.strip().lower() for x in args.models.split(",") if x.strip()))
        if args.periods:
            cfg.periods = tuple(x.strip() for x in args.periods.split(",") if x.strip())
        if args.fusion_methods:
            cfg.fusion_methods = tuple(x.strip() for x in args.fusion_methods.split(",") if x.strip())
        if args.quick:
            cfg.n_repeats, cfg.outer_folds, cfg.inner_folds = 1, 3, 2
            if not cfg.periods:
                cfg.periods = ("A100+A150", "V100+V150", "A100+V100", "A100+A150+V100+V150")
        if args.output_dir:
            cfg.output_dir = args.output_dir
        run_fusion(cfg)
