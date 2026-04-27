#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_PREDS = (
    ROOT
    / "outputs"
    / "20260330"
    / "3class"
    / "ttime_suite_physionetmi3_publicsrc_seed0_v1"
    / "predictions_all_methods.csv"
)
DEFAULT_PREDS = CANONICAL_PREDS if CANONICAL_PREDS.is_file() else None
DEFAULT_CANDIDATE_METHODS = "tent,t3a,cotta,shot,coral"


def _alpha_tag(value: float) -> str:
    return f"{float(value):.2f}"


def _default_out_dir(date_prefix: str, risk_alpha: float) -> Path:
    alpha = _alpha_tag(risk_alpha)
    return (
        ROOT
        / "outputs"
        / str(date_prefix)
        / "3class"
        / f"physio_safe_tts_d3_evi_physionetmi3_alpha{alpha}_v1"
    )


def _default_method_name(risk_alpha: float) -> str:
    alpha = _alpha_tag(risk_alpha)
    return f"safe-tts-d3-evidential-physionetmi3-alpha{alpha}-v1"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Canonical D3 wrapper for the current Safe-TTS PhysioNetMI 3-class selector."
    )
    parser.add_argument(
        "--preds",
        type=Path,
        default=DEFAULT_PREDS,
        help=(
            "Canonical PhysioNetMI 3-class predictions_all_methods.csv. "
            "If omitted, uses outputs/20260330/3class/ttime_suite_physionetmi3_publicsrc_seed0_v1/predictions_all_methods.csv "
            "when that file exists."
        ),
    )
    parser.add_argument("--risk-alpha", type=float, default=0.40)
    parser.add_argument("--anchor-method", type=str, default="eegnet_ea")
    parser.add_argument("--candidate-methods", type=str, default=DEFAULT_CANDIDATE_METHODS)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--calib-fraction", type=float, default=0.25)
    parser.add_argument("--guard-gray-margin", type=float, default=0.02)
    parser.add_argument("--calibration-protocol", type=str, default="paper_oof_dev_cal")
    parser.add_argument("--selector-model", type=str, default="evidential")
    parser.add_argument("--selector-views", type=str, default="stats,decision,relative")
    parser.add_argument("--selector-hidden-dim", type=int, default=32)
    parser.add_argument("--selector-epochs", type=int, default=50)
    parser.add_argument("--selector-outcome-delta", type=float, default=0.02)
    parser.add_argument("--date-prefix", type=str, default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--method-name", type=str, default=None)
    parser.add_argument("--with-diagnostics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> int:
    args, passthrough = parse_args()
    if args.preds is None:
        raise SystemExit(
            "No default PhysioNetMI 3-class predictions file found. "
            "Run scripts/safe_tts/run_physionetmi3_tta_suite.py first, or pass --preds explicitly."
        )
    out_dir = Path(args.out_dir) if args.out_dir is not None else _default_out_dir(args.date_prefix, args.risk_alpha)
    method_name = str(args.method_name) if args.method_name is not None else _default_method_name(args.risk_alpha)

    target = ROOT / "scripts" / "offline_safe_tta_multi_select_crc_from_predictions.py"
    cmd = [
        sys.executable,
        str(target),
        "--preds",
        str(args.preds),
        "--anchor-method",
        str(args.anchor_method),
        "--candidate-methods",
        str(args.candidate_methods),
        "--risk-alpha",
        str(args.risk_alpha),
        "--delta",
        str(args.delta),
        "--n-splits",
        str(args.n_splits),
        "--calib-fraction",
        str(args.calib_fraction),
        "--guard-gray-margin",
        str(args.guard_gray_margin),
        "--calibration-protocol",
        str(args.calibration_protocol),
        "--selector-model",
        str(args.selector_model),
        "--selector-views",
        str(args.selector_views),
        "--selector-hidden-dim",
        str(args.selector_hidden_dim),
        "--selector-epochs",
        str(args.selector_epochs),
        "--selector-outcome-delta",
        str(args.selector_outcome_delta),
        "--method-name",
        method_name,
        "--date-prefix",
        str(args.date_prefix),
        "--out-dir",
        str(out_dir),
    ]
    if not bool(args.with_diagnostics):
        cmd.append("--no-diagnostics")
    cmd.extend(passthrough)

    if bool(args.dry_run):
        print(" ".join(cmd))
        return 0

    subprocess.run(cmd, check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
