#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "outputs" / "20260330" / "3class" / "ttime_suite_physionetmi3_publicsrc_seed0_v1"
DEFAULT_METHODS = "eegnet_ea,tent,t3a,cotta,shot,coral"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Canonical wrapper for the current Safe-TTS PhysioNetMI 3-class TTA suite."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="DeepTransferEEG export directory.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", type=str, default=DEFAULT_METHODS)
    parser.add_argument("--target-subject-idxs", type=str, default="ALL")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> int:
    args, passthrough = parse_args()
    target = ROOT / "scripts" / "ttime_suite" / "run_suite_loso.py"
    cmd = [
        sys.executable,
        str(target),
        "--data-dir",
        str(args.data_dir),
        "--out-dir",
        str(args.out_dir),
        "--seed",
        str(args.seed),
        "--methods",
        str(args.methods),
        "--target-subject-idxs",
        str(args.target_subject_idxs),
        "--lr",
        str(args.lr),
        "--max-epochs",
        str(args.max_epochs),
        "--batch-size",
        str(args.batch_size),
        "--test-batch",
        str(args.test_batch),
        "--num-workers",
        str(args.num_workers),
        "--torch-threads",
        str(args.torch_threads),
    ]
    cmd.append("--resume" if bool(args.resume) else "--no-resume")
    cmd.extend(passthrough)

    if bool(args.dry_run):
        print(" ".join(cmd))
        return 0

    subprocess.run(cmd, check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
