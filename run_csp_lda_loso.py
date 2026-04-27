from __future__ import annotations

import argparse
from datetime import datetime
import gc
from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd
import mne

from csp_lda.config import ExperimentConfig, ModelConfig, PreprocessingConfig
from csp_lda.data import MoabbMotorImageryLoader, split_by_subject
from csp_lda.evaluation import compute_metrics, loso_cross_subject_evaluation
from csp_lda.plots import (
    plot_confusion_matrix,
    plot_csp_patterns,
    plot_method_comparison_bar,
)
from csp_lda.reporting import today_yyyymmdd, write_results_txt_multi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSP+LDA LOSO on MOABB MotorImagery datasets.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Output root directory. Run outputs go to OUT_DIR/YYYYMMDD/<N>class/HHMMSS_* (no overwrite).",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run subfolder name under OUT_DIR/YYYYMMDD/. Default: current time HHMMSS.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="BNCI2014_001",
        help=(
            "MOABB dataset name (e.g., BNCI2014_001, BNCI2014_002, Cho2017, PhysionetMI, Schirrmeister2017). "
            "Default: BNCI2014_001 (BCI IV 2a)."
        ),
    )
    p.add_argument("--fmin", type=float, default=8.0)
    p.add_argument("--fmax", type=float, default=30.0)
    # Align with He & Wu (EA) paper for BCI IV 2a:
    # use 0.5–3.5s after cue appearance.
    p.add_argument("--tmin", type=float, default=0.5)
    p.add_argument("--tmax", type=float, default=3.5)
    p.add_argument("--resample", type=float, default=250.0)
    p.add_argument("--n-components", type=int, default=4, help="CSP components (n_components).")
    p.add_argument(
        "--fbcsp-multiclass-strategy",
        type=str,
        default="multiclass",
        choices=["auto", "multiclass", "ovo", "ovr"],
        help=(
            "For fbcsp-lda / ea-fbcsp-lda (and stack candidate family 'fbcsp'): multiclass CSP strategy inside "
            "FilterBankCSP. 'ovo' can be very slow for K>2 (fits CSP per class pair per band)."
        ),
    )
    p.add_argument(
        "--preprocess",
        choices=["moabb", "paper_fir"],
        default="moabb",
        help="Preprocessing pipeline: 'moabb' (default) or 'paper_fir' (causal 50-order FIR Hamming).",
    )
    p.add_argument(
        "--car",
        action="store_true",
        help=(
            "Apply common average reference (CAR) after temporal filtering and before alignment/feature extraction. "
            "This is an unsupervised per-trial channel re-referencing: X <- X - mean_c X."
        ),
    )
    p.add_argument("--fir-order", type=int, default=50, help="FIR order for paper_fir mode.")
    p.add_argument(
        "--events",
        type=str,
        default="left_hand,right_hand",
        help="Comma-separated events/classes (e.g., left_hand,right_hand).",
    )
    p.add_argument(
        "--sessions",
        type=str,
        default="0train",
        help="Comma-separated MOABB session names to include (e.g., 0train). Use 'ALL' to include all.",
    )
    p.add_argument(
        "--test-subjects",
        type=str,
        default="ALL",
        help=(
            "Optional comma-separated test subject ids to evaluate (subset of dataset subjects). "
            "Training in each fold still uses all remaining subjects (strict LOSO protocol). "
            "Use 'ALL' (default) to evaluate all subjects. This is useful for chunking large LOSO runs."
        ),
    )
    p.add_argument(
        "--methods",
        type=str,
        default="csp-lda,ea-csp-lda",
	        help=(
	            "Comma-separated methods to run: csp-lda, fbcsp-lda, ea-csp-lda, ea-fbcsp-lda, lea-csp-lda, lea-rot-csp-lda, "
                "deep4net, "
                "atcnet, "
                "tcformer, "
	            "riemann-mdm, rpa-mdm, rpa-rot-mdm, ts-lr, rpa-ts-lr, ea-ts-lr, "
	            "ea-stack-multi-safe-csp-lda, "
	            "ea-mm-safe, "
	            "oea-cov-csp-lda, oea-csp-lda, "
            "oea-zo-csp-lda, oea-zo-ent-csp-lda, oea-zo-im-csp-lda, oea-zo-pce-csp-lda, oea-zo-conf-csp-lda, "
            "oea-zo-imr-csp-lda, "
            "ea-si-csp-lda, ea-si-zo-csp-lda, ea-si-zo-ent-csp-lda, ea-si-zo-im-csp-lda, ea-si-zo-imr-csp-lda, "
            "ea-si-zo-pce-csp-lda, ea-si-zo-conf-csp-lda, "
            "ea-si-chan-csp-lda, "
            "ea-si-chan-safe-csp-lda, "
            "ea-si-chan-multi-safe-csp-lda, "
            "ea-si-chan-spsa-safe-csp-lda, "
            "raw-zo-csp-lda, raw-zo-ent-csp-lda, raw-zo-im-csp-lda, raw-zo-imr-csp-lda, raw-zo-pce-csp-lda, raw-zo-conf-csp-lda, "
            "ea-zo-csp-lda, ea-zo-ent-csp-lda, ea-zo-im-csp-lda, ea-zo-pce-csp-lda, ea-zo-conf-csp-lda, "
            "ea-zo-imr-csp-lda"
        ),
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving plots (confusion matrices / CSP patterns / comparison bar).",
    )
    p.add_argument(
        "--deep-max-epochs",
        type=int,
        default=50,
        help=(
            "Max epochs for deep baselines (deep4net/atcnet/tcformer). "
            "Note: many deep MI baselines require larger values; use this to scale compute."
        ),
    )
    p.add_argument(
        "--deep-patience",
        type=int,
        default=10,
        help="Early-stopping patience (valid_loss) for deep baselines (deep4net/atcnet/tcformer).",
    )
    p.add_argument(
        "--oea-eps",
        type=float,
        default=1e-10,
        help="Numeric stability epsilon used by EA/OEA (eigenvalue floor via eps*max_eig).",
    )
    p.add_argument(
        "--oea-shrinkage",
        type=float,
        default=0.0,
        help="Optional covariance shrinkage in EA/OEA (0 means disabled).",
    )
    p.add_argument(
        "--si-subject-lambda",
        type=float,
        default=1.0,
        help=(
            "For ea-si-* methods: subject invariance strength λ (>=0) in the HSIC-style projector "
            "(larger enforces more subject invariance)."
        ),
    )
    p.add_argument(
        "--si-ridge",
        type=float,
        default=1e-6,
        help="For ea-si-* methods: ridge (>0) used in the generalized eigen-problem for the projector.",
    )
    p.add_argument(
        "--si-proj-dim",
        type=int,
        default=0,
        help="For ea-si-* methods: projector output dimension r (0 keeps full CSP feature dimension).",
    )
    p.add_argument(
        "--si-chan-ranks",
        type=str,
        default="",
        help=(
            "For ea-si-chan-multi-safe-csp-lda: comma-separated candidate ranks for the channel projector "
            "A=QQᵀ (e.g. '18,19,20,21'). Empty falls back to --si-proj-dim."
        ),
    )
    p.add_argument(
        "--si-chan-lambdas",
        type=str,
        default="",
        help=(
            "For ea-si-chan-multi-safe-csp-lda: comma-separated candidate subject invariance strengths λ "
            "(e.g. '0.5,1,2'). Empty falls back to --si-subject-lambda."
        ),
    )
    p.add_argument(
        "--oea-pseudo-iters",
        type=int,
        default=2,
        help="For oea-csp-lda only: number of pseudo-label iterations on the target subject.",
    )
    p.add_argument(
        "--oea-pseudo-mode",
        choices=["hard", "soft"],
        default="hard",
        help="For oea-csp-lda only: pseudo-label mode for target Q_t selection ('hard' or 'soft').",
    )
    p.add_argument(
        "--oea-pseudo-confidence",
        type=float,
        default=0.0,
        help=(
            "For oea-csp-lda only (hard mode) and oea-zo-* with objective=pseudo_ce: "
            "minimum confidence to keep a pseudo-labeled trial (0 disables)."
        ),
    )
    p.add_argument(
        "--oea-pseudo-topk-per-class",
        type=int,
        default=0,
        help=(
            "For oea-csp-lda only (hard mode) and oea-zo-* with objective=pseudo_ce: "
            "keep top-k confident trials per class (0 disables)."
        ),
    )
    p.add_argument(
        "--oea-pseudo-balance",
        action="store_true",
        help=(
            "For oea-csp-lda only (hard mode) and oea-zo-* with objective=pseudo_ce: "
            "balance pseudo-labeled trials per class (uses min count)."
        ),
    )
    p.add_argument(
        "--oea-zo-objective",
        choices=["entropy", "infomax", "pseudo_ce", "confidence", "lda_nll", "entropy_bilevel", "infomax_bilevel"],
        default="entropy",
        help="For oea-zo-* methods: zero-order objective on target unlabeled data.",
    )
    p.add_argument(
        "--oea-zo-transform",
        choices=["orthogonal", "rot_scale", "local_mix", "local_mix_then_ea", "local_affine_then_ea"],
        default="orthogonal",
        help=(
            "For oea-zo-* methods: channel-space transform family. "
            "'orthogonal' uses Q∈O(C) (pure rotation); "
            "'rot_scale' uses A=diag(exp(s))·Q (rotation + per-channel scaling); "
            "'local_mix' uses a row-stochastic local mixing A (each channel mixes only itself + neighbors); "
            "'local_mix_then_ea' applies EA whitening *after* the local mixing (A→EA), i.e. Q_eff = EA(A·X)·A; "
            "'local_affine_then_ea' is a signed local linear mixing (neighbor-sparse) followed by EA whitening."
        ),
    )
    p.add_argument(
        "--oea-zo-localmix-neighbors",
        type=int,
        default=4,
        help="For oea-zo-* methods with transform=local_mix: k nearest neighbors per channel (k>=0).",
    )
    p.add_argument(
        "--oea-zo-localmix-self-bias",
        type=float,
        default=3.0,
        help=(
            "For oea-zo-* methods with transform=local_mix: non-negative logit bias for the self-weight "
            "(larger keeps A closer to identity)."
        ),
    )
    p.add_argument(
        "--oea-zo-infomax-lambda",
        type=float,
        default=1.0,
        help="For oea-zo-* methods with objective=infomax: weight λ for H(mean p) term (must be > 0).",
    )
    p.add_argument(
        "--oea-zo-marginal-mode",
        choices=["none", "l2_uniform", "kl_uniform", "hinge_uniform", "hard_min", "kl_prior"],
        default="none",
        help=(
            "For oea-zo-* methods: add a class-marginal balance penalty on the predicted marginal p_bar "
            "(none disables)."
        ),
    )
    p.add_argument(
        "--oea-zo-marginal-beta",
        type=float,
        default=0.0,
        help="For oea-zo-* methods: class-marginal penalty weight β (>=0).",
    )
    p.add_argument(
        "--oea-zo-marginal-tau",
        type=float,
        default=0.05,
        help="For oea-zo-* methods with marginal_mode=hinge_uniform: lower-bound threshold τ in [0,1].",
    )
    p.add_argument(
        "--oea-zo-marginal-prior",
        choices=["uniform", "source", "anchor_pred"],
        default="uniform",
        help=(
            "For oea-zo-* methods with marginal_mode=kl_prior: choose π used in KL(π||p_bar). "
            "uniform uses π=1/K; source uses training-label empirical prior; anchor_pred uses "
            "the target marginal predicted at Q=I (EA)."
        ),
    )
    p.add_argument(
        "--oea-zo-marginal-prior-mix",
        type=float,
        default=0.0,
        help="For oea-zo-* methods with marginal_mode=kl_prior: mix π with uniform (0=no mix, 1=all uniform).",
    )
    p.add_argument(
        "--oea-zo-bilevel-iters",
        type=int,
        default=5,
        help="For oea-zo-* methods with objective=*_bilevel: lower-level iterations for (w,q) solver.",
    )
    p.add_argument(
        "--oea-zo-bilevel-temp",
        type=float,
        default=1.0,
        help="For oea-zo-* methods with objective=*_bilevel: temperature for soft labels q (T>0).",
    )
    p.add_argument(
        "--oea-zo-bilevel-step",
        type=float,
        default=1.0,
        help="For oea-zo-* methods with objective=*_bilevel: prior-matching step size (>=0).",
    )
    p.add_argument(
        "--oea-zo-bilevel-coverage-target",
        type=float,
        default=0.5,
        help="For oea-zo-* methods with objective=*_bilevel: coverage target in (0,1] for gating prior strength.",
    )
    p.add_argument(
        "--oea-zo-bilevel-coverage-power",
        type=float,
        default=1.0,
        help="For oea-zo-* methods with objective=*_bilevel: power (>=0) applied to coverage gating.",
    )
    p.add_argument(
        "--oea-zo-reliable-metric",
        choices=["none", "confidence", "entropy"],
        default="none",
        help=(
            "For oea-zo-* methods: optional reliability weighting metric used inside entropy/infomax/confidence "
            "objectives (none disables)."
        ),
    )
    p.add_argument(
        "--oea-zo-reliable-threshold",
        type=float,
        default=0.7,
        help=(
            "For oea-zo-* methods with reliable_metric != none: threshold for reliability weighting. "
            "If metric=confidence, must be in [0,1]. If metric=entropy, must be >=0."
        ),
    )
    p.add_argument(
        "--oea-zo-reliable-alpha",
        type=float,
        default=10.0,
        help="For oea-zo-* methods with reliable_metric != none: sigmoid sharpness (alpha > 0).",
    )
    p.add_argument(
        "--oea-zo-trust-lambda",
        type=float,
        default=0.0,
        help=(
            "For oea-zo-* methods: trust-region penalty weight ρ for ||Q - Q0||_F^2 "
            "(0 disables)."
        ),
    )
    p.add_argument(
        "--oea-zo-trust-q0",
        choices=["identity", "delta"],
        default="identity",
        help="For oea-zo-* methods: trust-region anchor Q0 (identity|delta).",
    )
    p.add_argument(
        "--oea-zo-drift-mode",
        choices=["none", "penalty", "hard"],
        default="none",
        help=(
            "For oea-zo-* methods: prediction-drift guard relative to the EA anchor (Q=I). "
            "none disables; penalty adds γ * mean KL(p0||pQ); hard enforces drift<=δ at selection time."
        ),
    )
    p.add_argument(
        "--oea-zo-drift-gamma",
        type=float,
        default=0.0,
        help="For oea-zo-* methods with drift_mode=penalty: penalty weight γ (>=0).",
    )
    p.add_argument(
        "--oea-zo-drift-delta",
        type=float,
        default=0.0,
        help="For oea-zo-* methods with drift_mode=hard: drift threshold δ (>=0).",
    )
    p.add_argument(
        "--oea-zo-selector",
        choices=[
            "objective",
            "dev",
            "evidence",
            "probe_mixup",
            "probe_mixup_hard",
            "iwcv",
            "iwcv_ucb",
            "calibrated_ridge",
            "calibrated_guard",
            "calibrated_ridge_guard",
            "calibrated_stack_ridge",
            "calibrated_stack_ridge_guard",
            "calibrated_stack_ridge_guard_borda",
            "calibrated_stack_ridge_guard_borda3",
            "calibrated_stack_bandit_guard",
            "prefer_fbcsp",
            "oracle",
        ],
        default="objective",
        help=(
            "For oea-zo-* methods: how to select Q_t from the candidate set. "
            "objective selects by the unlabeled objective (plus optional drift guard); "
            "evidence selects by LDA evidence (-log p(z)) under the frozen CSP+LDA model; "
            "probe_mixup selects by a MixUp-style probe score in CSP feature space; "
            "probe_mixup_hard selects by a MixUp-style probe with hard-major pseudo labels (MixVal-style λ>0.5); "
            "dev selects by a DEV-style control-variate IW certificate on labeled source; "
            "iwcv selects by importance-weighted NLL on labeled source (covariate-shift certificate); "
            "iwcv_ucb selects by IWCV-UCB (IWCV-NLL plus kappa*SE via n_eff); "
            "calibrated_ridge learns a regressor on source subjects to predict improvement; "
            "calibrated_guard learns a binary guard to reject likely negative transfer; "
            "calibrated_ridge_guard uses the learned guard to filter candidates, then selects by ridge-predicted improvement; "
            "calibrated_stack_ridge learns a ridge regressor on stacked certificate features (objective+evidence+probe+drift) to predict improvement; "
            "calibrated_stack_ridge_guard uses a guard + ridge on stacked certificate features; "
            "calibrated_stack_ridge_guard_borda uses a guard + ridge (stacked features) for calibration, but selects by Borda rank aggregation "
            "of (ridge_pred_improve, probe_hard_improve) after safety gates; "
            "calibrated_stack_ridge_guard_borda3 adds IWCV-UCB improvement as a third Borda signal; "
            "calibrated_stack_bandit_guard trains a softmax contextual bandit policy on stacked certificate features (full-information Δacc on pseudo-targets) "
            "to select a candidate, with the same guard/fallback safety; "
            "prefer_fbcsp is a lightweight policy for ea-stack-multi-safe-csp-lda: prefer the FBCSP candidate when present, "
            "then rely on family-specific safety gates (e.g., pred_disagree) to accept/fallback to EA; "
            "oracle selects by true accuracy (analysis-only upper bound; uses labels)."
        ),
    )
    p.add_argument(
        "--oea-zo-iwcv-kappa",
        type=float,
        default=1.0,
        help="For selector=iwcv_ucb: UCB penalty strength kappa (>=0).",
    )
    p.add_argument(
        "--oea-zo-calib-ridge-alpha",
        type=float,
        default=1.0,
        help="For oea-zo-* methods with selector=calibrated_ridge: Ridge alpha (>0).",
    )
    p.add_argument(
        "--oea-zo-calib-max-subjects",
        type=int,
        default=0,
        help="For oea-zo-* methods with selector=calibrated_ridge: limit pseudo-target subjects per fold (0=all).",
    )
    p.add_argument(
        "--oea-zo-calib-seed",
        type=int,
        default=0,
        help="For oea-zo-* methods with selector=calibrated_ridge: random seed for sampling pseudo-targets.",
    )
    p.add_argument(
        "--oea-zo-calib-guard-c",
        type=float,
        default=1.0,
        help="For oea-zo-* methods with selector=calibrated_guard: LogisticRegression C (>0).",
    )
    p.add_argument(
        "--oea-zo-calib-guard-threshold",
        type=float,
        default=0.5,
        help="For oea-zo-* methods with selector=calibrated_guard: keep candidates with P(pos) >= threshold.",
    )
    p.add_argument(
        "--oea-zo-calib-guard-margin",
        type=float,
        default=0.0,
        help="For oea-zo-* methods with selector=calibrated_guard: label as positive if improvement > margin.",
    )
    p.add_argument(
        "--mm-safe-mdm-guard-threshold",
        type=float,
        default=-1.0,
        help=(
            "For method=ea-mm-safe only: additional (family-specific) guard threshold for the MDM candidate. "
            "If set >=0, the effective MDM threshold is max(--oea-zo-calib-guard-threshold, this). "
            "Use this to treat MDM as a higher-risk candidate family."
        ),
    )
    p.add_argument(
        "--mm-safe-mdm-min-pred-improve",
        type=float,
        default=0.0,
        help=(
            "For method=ea-mm-safe only: require the MDM candidate's ridge-predicted improvement to be at least this value "
            "before it can be selected (>=0)."
        ),
    )
    p.add_argument(
        "--mm-safe-mdm-drift-delta",
        type=float,
        default=0.0,
        help=(
            "For method=ea-mm-safe only: additional hard drift guard (mean KL(p_anchor||p_mdm) <= delta) applied only "
            "to the MDM candidate. 0 disables."
        ),
    )
    p.add_argument(
        "--stack-safe-fbcsp-guard-threshold",
        type=float,
        default=-1.0,
        help=(
            "For method=ea-stack-multi-safe-csp-lda only: additional (family-specific) guard threshold for the FBCSP candidate. "
            "If set >=0, the effective FBCSP threshold is max(--oea-zo-calib-guard-threshold, this). "
            "Use this to treat FBCSP as a higher-risk candidate family."
        ),
    )
    p.add_argument(
        "--stack-safe-fbcsp-min-pred-improve",
        type=float,
        default=0.0,
        help=(
            "For method=ea-stack-multi-safe-csp-lda only: require the FBCSP candidate's ridge-predicted improvement "
            "to be at least this value before it can be selected (>=0)."
        ),
    )
    p.add_argument(
        "--stack-safe-fbcsp-drift-delta",
        type=float,
        default=0.0,
        help=(
            "For method=ea-stack-multi-safe-csp-lda only: additional hard drift guard (mean KL(p_anchor||p_fbcsp) <= delta) "
            "applied only to the FBCSP candidate. 0 disables."
        ),
    )
    p.add_argument(
        "--stack-safe-fbcsp-max-pred-disagree",
        type=float,
        default=-1.0,
        help=(
            "For method=ea-stack-multi-safe-csp-lda only: additional hard gate applied only to the FBCSP candidate. "
            "Requires pred_disagree <= tau, where pred_disagree is the fraction of target trials whose argmax prediction "
            "differs from the EA anchor. Set tau in [0,1]. Use -1 to disable."
        ),
    )
    p.add_argument(
        "--stack-safe-tsa-guard-threshold",
        type=float,
        default=-1.0,
        help=(
            "For method=ea-stack-multi-safe-csp-lda only: additional (family-specific) guard threshold for the TSA candidate. "
            "If set >=0, the effective TSA threshold is max(--oea-zo-calib-guard-threshold, this). "
            "Use this to treat TSA as a higher-risk candidate family."
        ),
    )
    p.add_argument(
        "--stack-safe-tsa-min-pred-improve",
        type=float,
        default=0.0,
        help=(
            "For method=ea-stack-multi-safe-csp-lda only: require the TSA candidate's ridge-predicted improvement "
            "to be at least this value before it can be selected (>=0)."
        ),
    )
    p.add_argument(
        "--stack-safe-tsa-drift-delta",
        type=float,
        default=0.0,
        help=(
            "For method=ea-stack-multi-safe-csp-lda only: additional hard drift guard (mean KL(p_anchor||p_tsa) <= delta) "
            "applied only to the TSA candidate. 0 disables."
        ),
    )
    p.add_argument(
        "--stack-safe-anchor-guard-delta",
        type=float,
        default=0.0,
        help=(
            "For method=ea-stack-multi-safe-csp-lda only: require a candidate's calibrated guard probability to exceed "
            "the EA(anchor) guard probability by at least delta (P_pos(cand) >= P_pos(anchor) + delta), in addition to "
            "--oea-zo-calib-guard-threshold. 0 disables."
        ),
    )
    p.add_argument(
        "--stack-safe-anchor-probe-hard-worsen",
        type=float,
        default=-1.0,
        help=(
            "For method=ea-stack-multi-safe-csp-lda only: additional EA-anchor-relative gate using the MixVal-style "
            "hard-major probe score h=probe_mixup_hard_best (smaller is better). "
            "If set > -1, require h(cand) <= h(EA_anchor) + eps for non-identity candidates. "
            "This supports two modes: eps>=0 is 'do-not-worsen' (tolerance), eps<0 is 'min-improve' "
            "(require improvement by |eps|). Use -1 to disable."
        ),
    )
    p.add_argument(
        "--stack-safe-min-pred-improve",
        type=float,
        default=0.0,
        help=(
            "For method=ea-stack-multi-safe-csp-lda only: global gate applied to all non-identity candidates. "
            "Require the (blended) ridge-predicted improvement to be at least this value before a candidate can be selected (>=0)."
        ),
    )
    p.add_argument(
        "--stack-calib-per-family",
        action="store_true",
        help=(
            "For method=ea-stack-multi-safe-csp-lda only: train separate calibrated ridge/guard models per candidate family "
            "(e.g., fbcsp/rpa/tsa/chan) on pseudo-target subjects, and use the family-specific models for selection "
            "when available (falls back to the global model otherwise)."
        ),
    )
    p.add_argument(
        "--stack-calib-per-family-mode",
        choices=["hard", "blend"],
        default="hard",
        help=(
            "For method=ea-stack-multi-safe-csp-lda only: how to use per-family calibrated models. "
            "'hard' uses the per-family model whenever available; "
            "'blend' uses shrinkage/partial-pooling: prediction = (1-w)*global + w*family with w=n/(n+K). "
            "Effective only when --stack-calib-per-family is enabled."
        ),
    )
    p.add_argument(
        "--stack-calib-per-family-shrinkage",
        type=float,
        default=20.0,
        help=(
            "For method=ea-stack-multi-safe-csp-lda only: shrinkage K used by --stack-calib-per-family-mode blend "
            "(w = n / (n + K)). Larger K => more conservative (closer to global). Must be >= 0."
        ),
    )
    p.add_argument(
        "--stack-feature-set",
        type=str,
        default="stacked",
        choices=["base", "base_delta", "stacked", "stacked_delta"],
        help=(
            "For method=ea-stack-multi-safe-csp-lda only: feature representation used by calibrated stack selectors. "
            "'base' uses the unlabeled base features only (no extra certificates like evidence/probe); "
            "'base_delta' uses anchor-relative (candidate - EA) base features (no extra certificates like evidence/probe); "
            "'stacked' uses absolute unlabeled features; "
            "'stacked_delta' uses anchor-relative (candidate - EA) features for all non-meta features."
        ),
    )
    p.add_argument(
        "--stack-candidate-families",
        type=str,
        default="ea,fbcsp,rpa,tsa,chan",
        help=(
            "For method=ea-stack-multi-safe-csp-lda only: comma-separated candidate families to include. "
            "Supported: ea (anchor), fbcsp, rpa(=LEA view), tsa(=LEA+rot view), lea(alias of rpa), lea_rot(alias of tsa), "
            "chan, ts_svc, tsa_ts_svc, fgmdm. "
            "Note: 'tsa' requires 'rpa'."
        ),
    )
    p.add_argument(
        "--oea-zo-min-improvement",
        type=float,
        default=0.0,
        help=(
            "For oea-zo-* methods: require at least this much holdout-objective improvement over identity "
            "before accepting an adapted Q_t (0 disables)."
        ),
    )
    p.add_argument(
        "--oea-zo-holdout-fraction",
        type=float,
        default=0.0,
        help=(
            "For oea-zo-* methods: holdout fraction in [0,1) used for best-iterate selection "
            "(updates use the remaining trials). 0 disables."
        ),
    )
    p.add_argument(
        "--oea-zo-warm-start",
        choices=["none", "delta"],
        default="none",
        help="For oea-zo-* methods: initialization strategy for SPSA (none|delta).",
    )
    p.add_argument(
        "--oea-zo-warm-iters",
        type=int,
        default=1,
        help="For oea-zo-* methods with warm_start=delta: number of pseudo-Δ refinement iterations.",
    )
    p.add_argument(
        "--oea-zo-fallback-min-marginal-entropy",
        type=float,
        default=0.0,
        help=(
            "For oea-zo-* methods: if >0, enable an unlabeled safety fallback when the predicted "
            "class-marginal entropy H(mean p) falls below this threshold (nats)."
        ),
    )
    p.add_argument(
        "--oea-zo-iters",
        type=int,
        default=30,
        help="For oea-zo-* methods: SPSA iterations for optimizing Q_t.",
    )
    p.add_argument(
        "--oea-zo-lr",
        type=float,
        default=0.5,
        help="For oea-zo-* methods: SPSA learning rate (base).",
    )
    p.add_argument(
        "--oea-zo-mu",
        type=float,
        default=0.1,
        help="For oea-zo-* methods: SPSA perturbation size.",
    )
    p.add_argument(
        "--oea-zo-k",
        type=int,
        default=50,
        help="For oea-zo-* methods: number of Givens rotations (low-dim Q parameterization).",
    )
    p.add_argument(
        "--oea-zo-seed",
        type=int,
        default=0,
        help="For oea-zo-* methods: random seed for Givens planes and SPSA directions.",
    )
    p.add_argument(
        "--oea-zo-l2",
        type=float,
        default=0.0,
        help="For oea-zo-* methods: L2 regularization on Givens angles.",
    )
    p.add_argument(
        "--oea-q-blend",
        type=float,
        default=1.0,
        help="Blend factor in [0,1] to control how aggressive the selected Q is (0=I, 1=full Q).",
    )
    p.add_argument(
        "--diagnose-subjects",
        type=str,
        default="",
        help="Optional comma-separated subject ids to write ZO diagnostics (e.g. '4').",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mne.set_log_level("WARNING")
    warnings.filterwarnings("ignore", message=r"warnEpochs.*")
    warnings.filterwarnings(
        "ignore", message=r"Concatenation of Annotations within Epochs is not supported yet.*"
    )
    date_prefix = today_yyyymmdd()
    out_root = Path(args.out_dir)

    events = tuple([e.strip() for e in str(args.events).split(",") if e.strip()])
    if len(events) < 2:
        raise ValueError("--events must contain at least two classes.")
    task_dir = f"{len(events)}class"
    base_dir = out_root / date_prefix / task_dir
    dataset_slug = re.sub(r"[^0-9a-zA-Z]+", "", str(args.dataset).strip().lower()) or "dataset"
    run_name = args.run_name or f"{datetime.now().strftime('%H%M%S')}_{dataset_slug}"
    out_dir = base_dir / run_name
    i = 1
    while out_dir.exists():
        out_dir = base_dir / f"{run_name}_{i:02d}"
        i += 1

    sessions_raw = str(args.sessions).strip()
    sessions = None if sessions_raw.upper() == "ALL" else tuple(
        [s.strip() for s in sessions_raw.split(",") if s.strip()]
    )
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    method_aliases = {
        # Historical names (kept for backward compatibility): these are *not* paper-faithful RPA/TSA.
        "rpa-csp-lda": "lea-csp-lda",
        "tsa-csp-lda": "lea-rot-csp-lda",
    }
    methods_canon: list[str] = []
    for m in methods:
        canon = method_aliases.get(m, m)
        if canon != m:
            print(f"[DEPRECATED] method '{m}' is now '{canon}' (paper-faithful naming).")
        methods_canon.append(canon)
    # Preserve order but de-duplicate (aliases can cause duplicates).
    methods = list(dict.fromkeys(methods_canon))

    stack_fams_raw = [s.strip() for s in str(args.stack_candidate_families).split(",") if s.strip()]
    stack_fam_aliases = {
        # Paper-faithful naming aliases (internal code uses historical family ids).
        "lea": "rpa",
        "lea_rot": "tsa",
        "lea-rot": "tsa",
    }
    stack_candidate_families_canon: list[str] = []
    for fam in stack_fams_raw:
        canon = stack_fam_aliases.get(fam, fam)
        if canon != fam:
            print(f"[INFO] stack family alias '{fam}' -> '{canon}'.")
        stack_candidate_families_canon.append(canon)
    stack_candidate_families = tuple(dict.fromkeys(stack_candidate_families_canon))
    deep_device = "cpu"
    if any(m in {"deep4net", "atcnet", "tcformer"} for m in methods):
        try:
            import torch

            deep_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            deep_device = "cpu"

    def _parse_csv_ints(raw: str) -> list[int]:
        raw = str(raw).strip()
        if not raw:
            return []
        out: list[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(int(part))
        return out

    def _parse_csv_floats(raw: str) -> list[float]:
        raw = str(raw).strip()
        if not raw:
            return []
        out: list[float] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(float(part))
        return out

    si_chan_candidate_ranks = _parse_csv_ints(str(args.si_chan_ranks))
    si_chan_candidate_lambdas = _parse_csv_floats(str(args.si_chan_lambdas))

    diagnose_subjects: tuple[int, ...] = ()
    if str(args.diagnose_subjects).strip():
        diagnose_subjects = tuple(
            sorted({int(x) for x in str(args.diagnose_subjects).split(",") if str(x).strip()})
        )

    test_subjects: tuple[int, ...] | None = None
    test_subjects_raw = str(args.test_subjects).strip()
    if test_subjects_raw and test_subjects_raw.upper() != "ALL":
        test_subjects = tuple(
            sorted({int(x) for x in test_subjects_raw.split(",") if str(x).strip()})
        )

    preprocessing = PreprocessingConfig(
        fmin=float(args.fmin),
        fmax=float(args.fmax),
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        resample=float(args.resample),
        events=events,
        sessions=tuple(sessions) if sessions is not None else (),
        preprocess=str(args.preprocess),
        car=bool(args.car),
        paper_fir_order=int(args.fir_order),
    )
    model_cfg = ModelConfig(csp_n_components=int(args.n_components))
    loader = MoabbMotorImageryLoader(
        dataset=str(args.dataset),
        fmin=preprocessing.fmin,
        fmax=preprocessing.fmax,
        tmin=preprocessing.tmin,
        tmax=preprocessing.tmax,
        resample=preprocessing.resample,
        events=preprocessing.events,
        sessions=sessions,
        preprocess=preprocessing.preprocess,
        car=preprocessing.car,
        paper_fir_order=preprocessing.paper_fir_order,
        paper_fir_window=preprocessing.paper_fir_window,
    )
    config = ExperimentConfig(out_dir=out_dir, dataset=f"MOABB {loader.dataset_id}", preprocessing=preprocessing, model=model_cfg)

    try:
        X, y, meta = loader.load_arrays(dtype="float32")
    except RuntimeError as e:
        if sessions is not None and "No trials left after filtering sessions=" in str(e):
            print(f"[WARN] {e} Falling back to sessions=ALL.")
            sessions = None
            preprocessing = PreprocessingConfig(
                fmin=float(args.fmin),
                fmax=float(args.fmax),
                tmin=float(args.tmin),
                tmax=float(args.tmax),
                resample=float(args.resample),
                events=events,
                sessions=(),
                preprocess=str(args.preprocess),
                car=bool(args.car),
                paper_fir_order=int(args.fir_order),
            )
            loader = MoabbMotorImageryLoader(
                dataset=str(args.dataset),
                fmin=preprocessing.fmin,
                fmax=preprocessing.fmax,
                tmin=preprocessing.tmin,
                tmax=preprocessing.tmax,
                resample=preprocessing.resample,
                events=preprocessing.events,
                sessions=None,
                preprocess=preprocessing.preprocess,
                car=preprocessing.car,
                paper_fir_order=preprocessing.paper_fir_order,
                paper_fir_window=preprocessing.paper_fir_window,
            )
            config = ExperimentConfig(
                out_dir=out_dir, dataset=f"MOABB {loader.dataset_id}", preprocessing=preprocessing, model=model_cfg
            )
            X, y, meta = loader.load_arrays(dtype="float32")
        else:
            raise
    subject_data = split_by_subject(X, y, meta)
    # IMPORTANT (memory): split_by_subject uses boolean masks, which materialize copies of X/y slices.
    # On large datasets (e.g., PhysionetMI), keeping the original concatenated X/y/meta alongside per-subject
    # copies can double peak RSS before the LOSO loop even starts. Drop the global arrays early.
    try:
        del X
        del y
        del meta
    except Exception:
        pass
    gc.collect()
    info = loader.load_epochs_info()

    metric_columns = ["accuracy", "precision", "recall", "f1", "auc", "kappa"]
    class_order = list(config.preprocessing.events)

    results_by_method: dict[str, pd.DataFrame] = {}
    overall_by_method: dict[str, dict[str, float]] = {}
    predictions_by_method: dict[str, tuple] = {}
    trial_predictions_by_method: dict[str, pd.DataFrame] = {}
    method_details: dict[str, str] = {}

    for method in methods:
        # Per-method override for OEA-ZO objectives (so we can compare multiple ZO objectives in one run).
        zo_objective_override: str | None = None

        if method == "csp-lda":
            alignment = "none"
            method_details[method] = "No alignment."
        elif method == "fbcsp-lda":
            alignment = "fbcsp"
            fb_n_components = max(2, min(4, int(config.model.csp_n_components)))
            method_details[method] = (
                "FBCSP+LDA: FilterBank-CSP over sub-bands within 8–30 Hz "
                f"(n_components_per_band={fb_n_components}, selector=MI@24, lda=shrinkage_auto, "
                f"multiclass_strategy={str(args.fbcsp_multiclass_strategy)})."
            )
        elif method == "ea-csp-lda":
            alignment = "ea"
            method_details[method] = f"EA: per-subject whitening (eps={args.oea_eps}, shrinkage={args.oea_shrinkage})."
        elif method == "ea-fbcsp-lda":
            alignment = "ea_fbcsp"
            fb_n_components = max(2, min(4, int(config.model.csp_n_components)))
            method_details[method] = (
                "EA + FBCSP+LDA: per-subject EA whitening, then FilterBank-CSP over sub-bands within 8–30 Hz "
                f"(n_components_per_band={fb_n_components}, selector=MI@24, lda=shrinkage_auto, "
                f"multiclass_strategy={str(args.fbcsp_multiclass_strategy)}, "
                f"eps={args.oea_eps}, shrinkage={args.oea_shrinkage})."
            )
        elif method == "lea-csp-lda":
            alignment = "rpa"
            method_details[method] = (
                "LEA: per-subject log-Euclidean whitening "
                f"(eps={args.oea_eps}, shrinkage={args.oea_shrinkage})."
            )
        elif method == "lea-rot-csp-lda":
            alignment = "tsa"
            method_details[method] = (
                "LEA + closed-form pseudo-anchor target rotation: "
                f"pseudo_mode={args.oea_pseudo_mode}, pseudo_iters={args.oea_pseudo_iters}, q_blend={args.oea_q_blend}, "
                f"pseudo_conf={args.oea_pseudo_confidence}, topk={args.oea_pseudo_topk_per_class}, balance={bool(args.oea_pseudo_balance)} "
                f"(eps={args.oea_eps}, shrinkage={args.oea_shrinkage})."
            )
        elif method == "riemann-mdm":
            alignment = "riemann_mdm"
            method_details[method] = (
                "pyRiemann baseline: MDM(metric='riemann') on per-trial SPD covariances "
                f"(cov eps={args.oea_eps}, shrinkage={args.oea_shrinkage})."
            )
        elif method == "rpa-mdm":
            alignment = "rpa_mdm"
            method_details[method] = (
                "pyRiemann transfer baseline: TLCenter+TLStretch then MDM(metric='riemann') on covariances "
                f"(cov eps={args.oea_eps}, shrinkage={args.oea_shrinkage})."
            )
        elif method == "rpa-rot-mdm":
            alignment = "rpa_rot_mdm"
            method_details[method] = (
                "pyRiemann transfer baseline: TLCenter+TLStretch+TLRotate (pseudo-labels) then "
                "MDM(metric='riemann') "
                f"(cov eps={args.oea_eps}, shrinkage={args.oea_shrinkage})."
            )
        elif method == "ts-lr":
            alignment = "ts_lr"
            method_details[method] = (
                "pyRiemann baseline: TangentSpace(metric='riemann') + LogisticRegression "
                "on per-trial SPD covariances "
                f"(cov eps={args.oea_eps}, shrinkage={args.oea_shrinkage})."
            )
        elif method == "rpa-ts-lr":
            alignment = "rpa_ts_lr"
            method_details[method] = (
                "pyRiemann transfer baseline: TLCenter+TLStretch then TangentSpace(metric='riemann') + LogisticRegression "
                "on covariances "
                f"(cov eps={args.oea_eps}, shrinkage={args.oea_shrinkage})."
            )
        elif method == "ea-ts-lr":
            alignment = "ea_ts_lr"
            method_details[method] = (
                "EA + TangentSpace-LR: apply EA whitening on time series, then "
                "TangentSpace(metric='riemann') + LogisticRegression on per-trial covariances "
                f"(cov eps={args.oea_eps}, shrinkage={args.oea_shrinkage})."
            )
        elif method == "ts-svc":
            alignment = "ts_svc"
            method_details[method] = (
                "Riemannian baseline: TangentSpace(metric='riemann') + linear SVC(probability=True) "
                "on per-trial SPD covariances "
                f"(cov eps={args.oea_eps}, shrinkage={args.oea_shrinkage})."
            )
        elif method == "tsa-ts-svc":
            alignment = "tsa_ts_svc"
            method_details[method] = (
                "TSA (tangent-space alignment) + linear SVC: recenter+rescale per domain in tangent space, "
                "then pseudo-label Procrustes rotation on target and classify with linear SVC(probability=True)."
            )
        elif method == "fgmdm":
            alignment = "fgmdm"
            method_details[method] = (
                "Riemannian baseline: FgMDM(metric='riemann') on per-trial SPD covariances "
                f"(cov eps={args.oea_eps}, shrinkage={args.oea_shrinkage})."
            )
        elif method == "deep4net":
            alignment = "deep4net"
            method_details[method] = (
                "Braindecode Deep4Net baseline (trialwise): train per LOSO fold on pooled source subjects; "
                "per-channel z-score standardization fit on training fold only; "
                f"optimizer=AdamW(lr=1e-2, wd=5e-4); early stop on valid loss "
                f"(max_epochs={int(args.deep_max_epochs)}, patience={int(args.deep_patience)})."
            )
        elif method == "atcnet":
            alignment = "atcnet"
            method_details[method] = (
                "Braindecode ATCNet baseline (trialwise): train per LOSO fold on pooled source subjects; "
                "per-channel z-score standardization fit on training fold only; "
                f"optimizer=AdamW(lr=1e-3); early stop on valid loss "
                f"(max_epochs={int(args.deep_max_epochs)}, patience={int(args.deep_patience)})."
            )
        elif method == "tcformer":
            alignment = "tcformer"
            method_details[method] = (
                "TCFormer baseline (trialwise): train per LOSO fold on pooled source subjects; "
                "per-channel z-score standardization fit on training fold only; "
                f"optimizer=AdamW(lr=9e-4, wd=1e-3); early stop on valid loss "
                f"(max_epochs={int(args.deep_max_epochs)}, patience={int(args.deep_patience)})."
            )
        elif method == "ea-stack-multi-safe-csp-lda":
            alignment = "ea_stack_multi_safe"
            ranks_str = str(args.si_chan_ranks).strip() or str(args.si_proj_dim)
            lambdas_str = str(args.si_chan_lambdas).strip() or str(args.si_subject_lambda)
            fbcsp_gate_str = ""
            if float(args.stack_safe_fbcsp_guard_threshold) >= 0.0:
                fbcsp_gate_str += f", fbcsp_guard_thr={float(args.stack_safe_fbcsp_guard_threshold)}"
            if float(args.stack_safe_fbcsp_min_pred_improve) > 0.0:
                fbcsp_gate_str += f", fbcsp_min_pred={float(args.stack_safe_fbcsp_min_pred_improve)}"
            if float(args.stack_safe_fbcsp_drift_delta) > 0.0:
                fbcsp_gate_str += f", fbcsp_drift_delta={float(args.stack_safe_fbcsp_drift_delta)}"
            if float(args.stack_safe_fbcsp_max_pred_disagree) >= 0.0:
                fbcsp_gate_str += f", fbcsp_max_pred_disagree={float(args.stack_safe_fbcsp_max_pred_disagree)}"
            tsa_gate_str = ""
            if float(args.stack_safe_tsa_guard_threshold) >= 0.0:
                tsa_gate_str += f", tsa_guard_thr={float(args.stack_safe_tsa_guard_threshold)}"
            if float(args.stack_safe_tsa_min_pred_improve) > 0.0:
                tsa_gate_str += f", tsa_min_pred={float(args.stack_safe_tsa_min_pred_improve)}"
            if float(args.stack_safe_tsa_drift_delta) > 0.0:
                tsa_gate_str += f", tsa_drift_delta={float(args.stack_safe_tsa_drift_delta)}"
            per_family_str = ""
            if bool(args.stack_calib_per_family):
                per_family_str = f", per_family_calib=1(mode={args.stack_calib_per_family_mode},K={args.stack_calib_per_family_shrinkage})"
            anchor_delta_str = ""
            if float(args.stack_safe_anchor_guard_delta) > 0.0:
                anchor_delta_str = f", anchor_guard_delta={float(args.stack_safe_anchor_guard_delta)}"
            probe_gate_str = ""
            if float(args.stack_safe_anchor_probe_hard_worsen) > -1.0:
                probe_gate_str = f", anchor_probe_hard_worsen={float(args.stack_safe_anchor_probe_hard_worsen)}"
            global_min_pred_str = ""
            if float(args.stack_safe_min_pred_improve) > 0.0:
                global_min_pred_str = f", min_pred={float(args.stack_safe_min_pred_improve)}"
            feat_set_str = f", feat_set={str(args.stack_feature_set)}"
            fams = [s.strip() for s in str(args.stack_candidate_families).split(",") if s.strip()]
            fams_str = ",".join(fams) if fams else "ea,fbcsp,rpa,tsa,chan"
            method_details[method] = (
                "EA-STACK-MULTI-SAFE: multi-family candidate selection with safe fallback to EA. "
                f"Candidate families={fams_str} (EA anchor always included). "
                f"(ranks={ranks_str}, lambdas={lambdas_str}, si_ridge={args.si_ridge}); "
                f"selector={args.oea_zo_selector} "
                f"(ridge_alpha={args.oea_zo_calib_ridge_alpha}, guard_C={args.oea_zo_calib_guard_c}, "
                f"guard_thr={args.oea_zo_calib_guard_threshold}, guard_margin={args.oea_zo_calib_guard_margin}, "
                f"max_subjects={args.oea_zo_calib_max_subjects}, seed={args.oea_zo_calib_seed}; "
                f"drift_mode={args.oea_zo_drift_mode}, drift_delta={args.oea_zo_drift_delta}{global_min_pred_str}{fbcsp_gate_str}{tsa_gate_str}; "
                f"fallback_Hbar<{args.oea_zo_fallback_min_marginal_entropy}{per_family_str}{feat_set_str}{anchor_delta_str}{probe_gate_str})."
            )
        elif method == "ea-mm-safe":
            alignment = "ea_mm_safe"
            ranks_str = str(args.si_chan_ranks).strip() or str(args.si_proj_dim)
            lambdas_str = str(args.si_chan_lambdas).strip() or str(args.si_subject_lambda)
            mdm_gate_str = ""
            if float(args.mm_safe_mdm_guard_threshold) >= 0.0:
                mdm_gate_str += f", mdm_guard_thr={float(args.mm_safe_mdm_guard_threshold)}"
            if float(args.mm_safe_mdm_min_pred_improve) > 0.0:
                mdm_gate_str += f", mdm_min_pred={float(args.mm_safe_mdm_min_pred_improve)}"
            if float(args.mm_safe_mdm_drift_delta) > 0.0:
                mdm_gate_str += f", mdm_drift_delta={float(args.mm_safe_mdm_drift_delta)}"
            method_details[method] = (
                "EA-MM-SAFE: multi-model candidate selection with safe fallback to EA anchor. "
                "Candidates include EA(anchor CSP+LDA), EA-SI-CHAN channel projectors, and MDM(RPA: TLCenter+TLStretch) "
                f"(ranks={ranks_str}, lambdas={lambdas_str}, si_ridge={args.si_ridge}); "
                f"selector={args.oea_zo_selector} "
                f"(ridge_alpha={args.oea_zo_calib_ridge_alpha}, guard_C={args.oea_zo_calib_guard_c}, "
                f"guard_thr={args.oea_zo_calib_guard_threshold}, guard_margin={args.oea_zo_calib_guard_margin}, "
                f"max_subjects={args.oea_zo_calib_max_subjects}, seed={args.oea_zo_calib_seed}; "
                f"drift_mode={args.oea_zo_drift_mode}, drift_delta={args.oea_zo_drift_delta}{mdm_gate_str})."
            )
        elif method == "oea-cov-csp-lda":
            alignment = "oea_cov"
            method_details[method] = (
                "OEA (cov-eig selection): choose Q_s from EA solution set using covariance eigen-basis "
                f"(eps={args.oea_eps}, shrinkage={args.oea_shrinkage}, q_blend={args.oea_q_blend})."
            )
        elif method == "oea-csp-lda":
            alignment = "oea"
            method_details[method] = (
                "OEA (discriminative optimistic selection): choose Q_s by aligning a covariance signature to a reference "
                "(binary: Δ=Cov(c1)-Cov(c0); multiclass: between-class covariance scatter); "
                f"target uses {args.oea_pseudo_iters} pseudo-label iters "
                f"(eps={args.oea_eps}, shrinkage={args.oea_shrinkage}, q_blend={args.oea_q_blend}, "
                f"pseudo_mode={args.oea_pseudo_mode}, pseudo_conf={args.oea_pseudo_confidence}, "
                f"topk={args.oea_pseudo_topk_per_class}, balance={bool(args.oea_pseudo_balance)})."
            )
        elif method in {
            "oea-zo-csp-lda",
            "oea-zo-ent-csp-lda",
            "oea-zo-im-csp-lda",
            "oea-zo-imr-csp-lda",
            "oea-zo-pce-csp-lda",
            "oea-zo-conf-csp-lda",
        }:
            alignment = "oea_zo"
            if method == "oea-zo-ent-csp-lda":
                zo_objective_override = "entropy"
            elif method == "oea-zo-im-csp-lda":
                zo_objective_override = "infomax"
            elif method == "oea-zo-imr-csp-lda":
                zo_objective_override = "infomax_bilevel"
            elif method == "oea-zo-pce-csp-lda":
                zo_objective_override = "pseudo_ce"
            elif method == "oea-zo-conf-csp-lda":
                zo_objective_override = "confidence"

            zo_obj = zo_objective_override or str(args.oea_zo_objective)
            marginal_prior_str = ""
            if str(args.oea_zo_marginal_mode) == "kl_prior":
                marginal_prior_str = (
                    f", prior={args.oea_zo_marginal_prior}, mix={args.oea_zo_marginal_prior_mix}"
                )
            drift_str = ""
            if str(args.oea_zo_drift_mode) != "none":
                drift_str = (
                    f"; drift={args.oea_zo_drift_mode} "
                    f"(gamma={args.oea_zo_drift_gamma}, delta={args.oea_zo_drift_delta})"
                )
            selector_str = f"; selector={args.oea_zo_selector}"
            if str(args.oea_zo_selector) == "calibrated_ridge":
                selector_str += (
                    f"(alpha={args.oea_zo_calib_ridge_alpha}, "
                    f"max_subjects={args.oea_zo_calib_max_subjects}, seed={args.oea_zo_calib_seed})"
                )
            elif str(args.oea_zo_selector) == "calibrated_guard":
                selector_str += (
                    f"(C={args.oea_zo_calib_guard_c}, "
                    f"thr={args.oea_zo_calib_guard_threshold}, margin={args.oea_zo_calib_guard_margin}, "
                    f"max_subjects={args.oea_zo_calib_max_subjects}, seed={args.oea_zo_calib_seed})"
                )
            method_details[method] = (
                "OEA-ZO (target optimistic selection): source uses covariance-signature alignment for Q_s "
                "(binary Δ; multiclass scatter); "
                "target optimizes Q_t by zero-order SPSA on unlabeled data "
                f"(objective={zo_obj}, transform={args.oea_zo_transform}, iters={args.oea_zo_iters}, lr={args.oea_zo_lr}, mu={args.oea_zo_mu}, "
                f"k={args.oea_zo_k}, seed={args.oea_zo_seed}, l2={args.oea_zo_l2}, q_blend={args.oea_q_blend}; "
                f"infomax_lambda={args.oea_zo_infomax_lambda}; "
                f"marginal={args.oea_zo_marginal_mode}*{args.oea_zo_marginal_beta} (tau={args.oea_zo_marginal_tau}{marginal_prior_str}); "
                f"{drift_str}{selector_str} "
                f"holdout={args.oea_zo_holdout_fraction}; "
                f"warm_start={args.oea_zo_warm_start}x{args.oea_zo_warm_iters}; "
                f"fallback_Hbar<{args.oea_zo_fallback_min_marginal_entropy}; "
                f"reliable={args.oea_zo_reliable_metric}@{args.oea_zo_reliable_threshold} (alpha={args.oea_zo_reliable_alpha}); "
                f"trust=||Q-Q0||^2*{args.oea_zo_trust_lambda} (Q0={args.oea_zo_trust_q0}); "
                f"min_improve={args.oea_zo_min_improvement}; "
                f"pseudo_conf={args.oea_pseudo_confidence}, topk={args.oea_pseudo_topk_per_class}, balance={bool(args.oea_pseudo_balance)})."
            )
        elif method == "ea-si-csp-lda":
            alignment = "ea_si"
            method_details[method] = (
                "EA-SI: source trains on EA-whitened data with a subject-invariant linear projector (HSIC-style), "
                f"(si_lambda={args.si_subject_lambda}, si_ridge={args.si_ridge}, si_proj_dim={args.si_proj_dim})."
            )
        elif method == "ea-si-chan-csp-lda":
            alignment = "ea_si_chan"
            method_details[method] = (
                "EA-SI-CHAN: learn a low-rank channel projector (rank=si_proj_dim) to reduce inter-subject shift "
                "before CSP, then train CSP+LDA on projected signals "
                f"(si_lambda={args.si_subject_lambda}, si_ridge={args.si_ridge}, si_proj_dim={args.si_proj_dim})."
            )
        elif method == "ea-si-chan-safe-csp-lda":
            alignment = "ea_si_chan_safe"
            method_details[method] = (
                "EA-SI-CHAN-SAFE: binary selection between EA anchor (A=I) and the channel projector candidate "
                "(A=QQᵀ) using a fold-local calibrated guard trained on pseudo-target subjects; fallback to EA when "
                "not accepted. "
                f"(rank={args.si_proj_dim}, si_lambda={args.si_subject_lambda}, si_ridge={args.si_ridge}; "
                f"guard_C={args.oea_zo_calib_guard_c}, guard_thr={args.oea_zo_calib_guard_threshold}, "
                f"guard_margin={args.oea_zo_calib_guard_margin}, max_subjects={args.oea_zo_calib_max_subjects}, "
                f"seed={args.oea_zo_calib_seed}; drift_mode={args.oea_zo_drift_mode}, drift_delta={args.oea_zo_drift_delta}; "
                f"fallback_Hbar<{args.oea_zo_fallback_min_marginal_entropy})."
            )
        elif method == "ea-si-chan-multi-safe-csp-lda":
            alignment = "ea_si_chan_multi_safe"
            ranks_str = str(args.si_chan_ranks).strip() or str(args.si_proj_dim)
            lambdas_str = str(args.si_chan_lambdas).strip() or str(args.si_subject_lambda)
            method_details[method] = (
                "EA-SI-CHAN-MULTI-SAFE: multi-candidate selection among EA anchor (A=I) and multiple "
                "channel projectors A=QQᵀ learned with different (rank,λ), using a fold-local calibrated selector "
                f"(selector={args.oea_zo_selector}); fallback to EA when selection is not confident/positive. "
                f"(ranks={ranks_str}, lambdas={lambdas_str}, si_ridge={args.si_ridge}; "
                f"ridge_alpha={args.oea_zo_calib_ridge_alpha}; "
                f"guard_C={args.oea_zo_calib_guard_c}, guard_thr={args.oea_zo_calib_guard_threshold}, "
                f"guard_margin={args.oea_zo_calib_guard_margin}, max_subjects={args.oea_zo_calib_max_subjects}, "
                f"seed={args.oea_zo_calib_seed}; drift_mode={args.oea_zo_drift_mode}, drift_delta={args.oea_zo_drift_delta}; "
                f"fallback_Hbar<{args.oea_zo_fallback_min_marginal_entropy})."
            )
        elif method == "ea-si-chan-spsa-safe-csp-lda":
            alignment = "ea_si_chan_spsa_safe"
            ranks_str = str(args.si_chan_ranks).strip() or str(args.si_proj_dim)
            lambdas_str = str(args.si_chan_lambdas).strip() or str(args.si_subject_lambda)
            method_details[method] = (
                "EA-SI-CHAN-SPSA-SAFE: continuous λ search (SPSA on log λ) for the SI-CHAN projector on the target "
                "subject, using a fold-local calibrated ridge/guard to score candidates; fallback to EA when not "
                "confident/positive. "
                f"(selector={args.oea_zo_selector}; ranks_grid={ranks_str}, lambdas_grid={lambdas_str}, "
                f"si_lambda_init={args.si_subject_lambda}, si_ridge={args.si_ridge}; "
                f"spsa iters={args.oea_zo_iters}, lr={args.oea_zo_lr}, mu={args.oea_zo_mu}, seed={args.oea_zo_seed}; "
                f"ridge_alpha={args.oea_zo_calib_ridge_alpha}; "
                f"guard_C={args.oea_zo_calib_guard_c}, guard_thr={args.oea_zo_calib_guard_threshold}, "
                f"guard_margin={args.oea_zo_calib_guard_margin}, max_subjects={args.oea_zo_calib_max_subjects}, "
                f"calib_seed={args.oea_zo_calib_seed}; drift_mode={args.oea_zo_drift_mode}, drift_delta={args.oea_zo_drift_delta}; "
                f"fallback_Hbar<{args.oea_zo_fallback_min_marginal_entropy})."
            )
        elif method in {
            "ea-si-zo-csp-lda",
            "ea-si-zo-ent-csp-lda",
            "ea-si-zo-im-csp-lda",
            "ea-si-zo-imr-csp-lda",
            "ea-si-zo-pce-csp-lda",
            "ea-si-zo-conf-csp-lda",
        }:
            alignment = "ea_si_zo"
            if method == "ea-si-zo-ent-csp-lda":
                zo_objective_override = "entropy"
            elif method == "ea-si-zo-im-csp-lda":
                zo_objective_override = "infomax"
            elif method == "ea-si-zo-imr-csp-lda":
                zo_objective_override = "infomax_bilevel"
            elif method == "ea-si-zo-pce-csp-lda":
                zo_objective_override = "pseudo_ce"
            elif method == "ea-si-zo-conf-csp-lda":
                zo_objective_override = "confidence"

            zo_obj = zo_objective_override or str(args.oea_zo_objective)
            marginal_prior_str = ""
            if str(args.oea_zo_marginal_mode) == "kl_prior":
                marginal_prior_str = (
                    f", prior={args.oea_zo_marginal_prior}, mix={args.oea_zo_marginal_prior_mix}"
                )
            drift_str = ""
            if str(args.oea_zo_drift_mode) != "none":
                drift_str = (
                    f"; drift={args.oea_zo_drift_mode} "
                    f"(gamma={args.oea_zo_drift_gamma}, delta={args.oea_zo_drift_delta})"
                )
            selector_str = f"; selector={args.oea_zo_selector}"
            if str(args.oea_zo_selector) == "calibrated_ridge":
                selector_str += (
                    f"(alpha={args.oea_zo_calib_ridge_alpha}, "
                    f"max_subjects={args.oea_zo_calib_max_subjects}, seed={args.oea_zo_calib_seed})"
                )
            elif str(args.oea_zo_selector) == "calibrated_guard":
                selector_str += (
                    f"(C={args.oea_zo_calib_guard_c}, "
                    f"thr={args.oea_zo_calib_guard_threshold}, margin={args.oea_zo_calib_guard_margin}, "
                    f"max_subjects={args.oea_zo_calib_max_subjects}, seed={args.oea_zo_calib_seed})"
                )
            method_details[method] = (
                "EA-SI-ZO: source trains on EA-whitened data with a subject-invariant linear projector (HSIC-style), "
                f"(si_lambda={args.si_subject_lambda}, si_ridge={args.si_ridge}, si_proj_dim={args.si_proj_dim}); "
                "target optimizes Q_t by zero-order SPSA on unlabeled data "
                f"(objective={zo_obj}, transform={args.oea_zo_transform}, iters={args.oea_zo_iters}, lr={args.oea_zo_lr}, mu={args.oea_zo_mu}, "
                f"k={args.oea_zo_k}, seed={args.oea_zo_seed}, l2={args.oea_zo_l2}, q_blend={args.oea_q_blend}; "
                f"infomax_lambda={args.oea_zo_infomax_lambda}; "
                f"marginal={args.oea_zo_marginal_mode}*{args.oea_zo_marginal_beta} (tau={args.oea_zo_marginal_tau}{marginal_prior_str}); "
                f"{drift_str}{selector_str} "
                f"holdout={args.oea_zo_holdout_fraction}; "
                f"warm_start={args.oea_zo_warm_start}x{args.oea_zo_warm_iters}; "
                f"fallback_Hbar<{args.oea_zo_fallback_min_marginal_entropy}; "
                f"reliable={args.oea_zo_reliable_metric}@{args.oea_zo_reliable_threshold} (alpha={args.oea_zo_reliable_alpha}); "
                f"trust=||Q-Q0||^2*{args.oea_zo_trust_lambda} (Q0={args.oea_zo_trust_q0}); "
                f"min_improve={args.oea_zo_min_improvement}; "
                f"pseudo_conf={args.oea_pseudo_confidence}, topk={args.oea_pseudo_topk_per_class}, balance={bool(args.oea_pseudo_balance)})."
            )
        elif method in {
            "ea-zo-ent-csp-lda",
            "ea-zo-im-csp-lda",
            "ea-zo-imr-csp-lda",
            "ea-zo-pce-csp-lda",
            "ea-zo-conf-csp-lda",
            "ea-zo-csp-lda",
        }:
            alignment = "ea_zo"
            if method == "ea-zo-ent-csp-lda":
                zo_objective_override = "entropy"
            elif method == "ea-zo-im-csp-lda":
                zo_objective_override = "infomax"
            elif method == "ea-zo-imr-csp-lda":
                zo_objective_override = "infomax_bilevel"
            elif method == "ea-zo-pce-csp-lda":
                zo_objective_override = "pseudo_ce"
            elif method == "ea-zo-conf-csp-lda":
                zo_objective_override = "confidence"

            zo_obj = zo_objective_override or str(args.oea_zo_objective)
            marginal_prior_str = ""
            if str(args.oea_zo_marginal_mode) == "kl_prior":
                marginal_prior_str = (
                    f", prior={args.oea_zo_marginal_prior}, mix={args.oea_zo_marginal_prior_mix}"
                )
            drift_str = ""
            if str(args.oea_zo_drift_mode) != "none":
                drift_str = (
                    f"; drift={args.oea_zo_drift_mode} "
                    f"(gamma={args.oea_zo_drift_gamma}, delta={args.oea_zo_drift_delta})"
                )
            selector_str = f"; selector={args.oea_zo_selector}"
            if str(args.oea_zo_selector) == "calibrated_ridge":
                selector_str += (
                    f"(alpha={args.oea_zo_calib_ridge_alpha}, "
                    f"max_subjects={args.oea_zo_calib_max_subjects}, seed={args.oea_zo_calib_seed})"
                )
            method_details[method] = (
                "EA-ZO (target optimistic selection): source trains on EA-whitened data (no Q_s selection); "
                "target optimizes Q_t by zero-order SPSA on unlabeled data "
                f"(objective={zo_obj}, transform={args.oea_zo_transform}, iters={args.oea_zo_iters}, lr={args.oea_zo_lr}, mu={args.oea_zo_mu}, "
                f"k={args.oea_zo_k}, seed={args.oea_zo_seed}, l2={args.oea_zo_l2}, q_blend={args.oea_q_blend}; "
                f"infomax_lambda={args.oea_zo_infomax_lambda}; "
                f"marginal={args.oea_zo_marginal_mode}*{args.oea_zo_marginal_beta} (tau={args.oea_zo_marginal_tau}{marginal_prior_str}); "
                f"{drift_str}{selector_str} "
                f"holdout={args.oea_zo_holdout_fraction}; "
                f"warm_start={args.oea_zo_warm_start}x{args.oea_zo_warm_iters}; "
                f"fallback_Hbar<{args.oea_zo_fallback_min_marginal_entropy}; "
                f"reliable={args.oea_zo_reliable_metric}@{args.oea_zo_reliable_threshold} (alpha={args.oea_zo_reliable_alpha}); "
                f"trust=||Q-Q0||^2*{args.oea_zo_trust_lambda} (Q0={args.oea_zo_trust_q0}); "
                f"min_improve={args.oea_zo_min_improvement}; "
                f"pseudo_conf={args.oea_pseudo_confidence}, topk={args.oea_pseudo_topk_per_class}, balance={bool(args.oea_pseudo_balance)})."
            )
        elif method in {
            "raw-zo-ent-csp-lda",
            "raw-zo-im-csp-lda",
            "raw-zo-imr-csp-lda",
            "raw-zo-pce-csp-lda",
            "raw-zo-conf-csp-lda",
            "raw-zo-csp-lda",
        }:
            alignment = "raw_zo"
            if method == "raw-zo-ent-csp-lda":
                zo_objective_override = "entropy"
            elif method == "raw-zo-im-csp-lda":
                zo_objective_override = "infomax"
            elif method == "raw-zo-imr-csp-lda":
                zo_objective_override = "infomax_bilevel"
            elif method == "raw-zo-pce-csp-lda":
                zo_objective_override = "pseudo_ce"
            elif method == "raw-zo-conf-csp-lda":
                zo_objective_override = "confidence"

            zo_obj = zo_objective_override or str(args.oea_zo_objective)
            marginal_prior_str = ""
            if str(args.oea_zo_marginal_mode) == "kl_prior":
                marginal_prior_str = (
                    f", prior={args.oea_zo_marginal_prior}, mix={args.oea_zo_marginal_prior_mix}"
                )
            drift_str = ""
            if str(args.oea_zo_drift_mode) != "none":
                drift_str = (
                    f"; drift={args.oea_zo_drift_mode} "
                    f"(gamma={args.oea_zo_drift_gamma}, delta={args.oea_zo_drift_delta})"
                )
            selector_str = f"; selector={args.oea_zo_selector}"
            if str(args.oea_zo_selector) == "calibrated_ridge":
                selector_str += (
                    f"(alpha={args.oea_zo_calib_ridge_alpha}, "
                    f"max_subjects={args.oea_zo_calib_max_subjects}, seed={args.oea_zo_calib_seed})"
                )
            elif str(args.oea_zo_selector) == "calibrated_guard":
                selector_str += (
                    f"(C={args.oea_zo_calib_guard_c}, "
                    f"thr={args.oea_zo_calib_guard_threshold}, margin={args.oea_zo_calib_guard_margin}, "
                    f"max_subjects={args.oea_zo_calib_max_subjects}, seed={args.oea_zo_calib_seed})"
                )
            method_details[method] = (
                "RAW-ZO: source trains on raw (preprocessed) signals (no EA whitening); "
                "target optimizes a channel-space transform by zero-order SPSA on unlabeled data "
                f"(objective={zo_obj}, transform={args.oea_zo_transform}, iters={args.oea_zo_iters}, lr={args.oea_zo_lr}, mu={args.oea_zo_mu}, "
                f"k={args.oea_zo_k}, seed={args.oea_zo_seed}, l2={args.oea_zo_l2}, q_blend={args.oea_q_blend}; "
                f"infomax_lambda={args.oea_zo_infomax_lambda}; "
                f"marginal={args.oea_zo_marginal_mode}*{args.oea_zo_marginal_beta} (tau={args.oea_zo_marginal_tau}{marginal_prior_str}); "
                f"{drift_str}{selector_str} "
                f"holdout={args.oea_zo_holdout_fraction}; "
                f"warm_start={args.oea_zo_warm_start}x{args.oea_zo_warm_iters}; "
                f"fallback_Hbar<{args.oea_zo_fallback_min_marginal_entropy}; "
                f"reliable={args.oea_zo_reliable_metric}@{args.oea_zo_reliable_threshold} (alpha={args.oea_zo_reliable_alpha}); "
                f"trust=||Q-Q0||^2*{args.oea_zo_trust_lambda} (Q0={args.oea_zo_trust_q0}); "
                f"min_improve={args.oea_zo_min_improvement}; "
                f"pseudo_conf={args.oea_pseudo_confidence}, topk={args.oea_pseudo_topk_per_class}, balance={bool(args.oea_pseudo_balance)})."
            )
        else:
            raise ValueError(
                "Unknown method "
                f"'{method}'. Supported: csp-lda, ea-csp-lda, oea-cov-csp-lda, oea-csp-lda, "
                "fbcsp-lda, ea-fbcsp-lda, lea-csp-lda, lea-rot-csp-lda, deep4net, atcnet, tcformer, riemann-mdm, rpa-mdm, rpa-rot-mdm, "
                "ea-stack-multi-safe-csp-lda, "
                "ea-mm-safe, "
                "oea-zo-csp-lda, oea-zo-ent-csp-lda, oea-zo-im-csp-lda, oea-zo-imr-csp-lda, "
                "oea-zo-pce-csp-lda, oea-zo-conf-csp-lda, "
                "ea-si-csp-lda, ea-si-zo-csp-lda, ea-si-zo-ent-csp-lda, ea-si-zo-im-csp-lda, ea-si-zo-imr-csp-lda, "
                "ea-si-zo-pce-csp-lda, ea-si-zo-conf-csp-lda, "
                "ea-si-chan-csp-lda, ea-si-chan-safe-csp-lda, ea-si-chan-multi-safe-csp-lda, ea-si-chan-spsa-safe-csp-lda, "
                "raw-zo-csp-lda, raw-zo-ent-csp-lda, raw-zo-im-csp-lda, raw-zo-imr-csp-lda, raw-zo-pce-csp-lda, raw-zo-conf-csp-lda, "
                "ea-zo-csp-lda, ea-zo-ent-csp-lda, ea-zo-im-csp-lda, ea-zo-imr-csp-lda, "
                "ea-zo-pce-csp-lda, ea-zo-conf-csp-lda"
            )

        if alignment == "deep4net":
            # Lazy import to avoid importing torch/CUDA for non-deep runs (saves memory on large datasets).
            from csp_lda.deep_baselines import Deep4NetParams, loso_deep4net_evaluation

            (
                results_df,
                pred_df,
                y_true_all,
                y_pred_all,
                y_proba_all,
                _class_order,
                _models_by_subject,
            ) = loso_deep4net_evaluation(
                subject_data,
                class_order=class_order,
                test_subjects=test_subjects,
                average=config.metrics_average,
                sfreq=float(args.resample),
                params=Deep4NetParams(
                    max_epochs=int(args.deep_max_epochs),
                    early_stop_patience=int(args.deep_patience),
                    device=str(deep_device),
                ),
            )
        elif alignment == "atcnet":
            # Lazy import to avoid importing torch/CUDA for non-deep runs (saves memory on large datasets).
            from csp_lda.deep_baselines import ATCNetParams, loso_atcnet_evaluation

            (
                results_df,
                pred_df,
                y_true_all,
                y_pred_all,
                y_proba_all,
                _class_order,
                _models_by_subject,
            ) = loso_atcnet_evaluation(
                subject_data,
                class_order=class_order,
                test_subjects=test_subjects,
                average=config.metrics_average,
                sfreq=float(args.resample),
                params=ATCNetParams(
                    max_epochs=int(args.deep_max_epochs),
                    early_stop_patience=int(args.deep_patience),
                    device=str(deep_device),
                ),
            )
        elif alignment == "tcformer":
            # Lazy import to avoid importing torch/CUDA for non-deep runs (saves memory on large datasets).
            from csp_lda.deep_baselines import TCFormerParams, loso_tcformer_evaluation

            (
                results_df,
                pred_df,
                y_true_all,
                y_pred_all,
                y_proba_all,
                _class_order,
                _models_by_subject,
            ) = loso_tcformer_evaluation(
                subject_data,
                class_order=class_order,
                test_subjects=test_subjects,
                average=config.metrics_average,
                sfreq=float(args.resample),
                params=TCFormerParams(
                    max_epochs=int(args.deep_max_epochs),
                    early_stop_patience=int(args.deep_patience),
                    device=str(deep_device),
                ),
            )
        else:
            (
                results_df,
                pred_df,
                y_true_all,
                y_pred_all,
                y_proba_all,
                _class_order,
                _models_by_subject,
            ) = loso_cross_subject_evaluation(
                subject_data,
                class_order=class_order,
                test_subjects=test_subjects,
                channel_names=list(info["ch_names"]),
                n_components=config.model.csp_n_components,
                average=config.metrics_average,
                alignment=alignment,
                inplace_pre_align=bool(len(methods) == 1 and alignment == "ea_fbcsp"),
                fbcsp_multiclass_strategy=str(args.fbcsp_multiclass_strategy),
                sfreq=float(args.resample),
                oea_eps=float(args.oea_eps),
                oea_shrinkage=float(args.oea_shrinkage),
                oea_pseudo_iters=int(args.oea_pseudo_iters),
                oea_q_blend=float(args.oea_q_blend),
                oea_pseudo_mode=str(args.oea_pseudo_mode),
                oea_pseudo_confidence=float(args.oea_pseudo_confidence),
                oea_pseudo_topk_per_class=int(args.oea_pseudo_topk_per_class),
                oea_pseudo_balance=bool(args.oea_pseudo_balance),
                oea_zo_objective=str(zo_objective_override or args.oea_zo_objective),
                oea_zo_transform=str(args.oea_zo_transform),
                oea_zo_localmix_neighbors=int(args.oea_zo_localmix_neighbors),
                oea_zo_localmix_self_bias=float(args.oea_zo_localmix_self_bias),
                oea_zo_infomax_lambda=float(args.oea_zo_infomax_lambda),
                oea_zo_marginal_mode=str(args.oea_zo_marginal_mode),
                oea_zo_marginal_beta=float(args.oea_zo_marginal_beta),
                oea_zo_marginal_tau=float(args.oea_zo_marginal_tau),
                oea_zo_marginal_prior=str(args.oea_zo_marginal_prior),
                oea_zo_marginal_prior_mix=float(args.oea_zo_marginal_prior_mix),
                oea_zo_bilevel_iters=int(args.oea_zo_bilevel_iters),
                oea_zo_bilevel_temp=float(args.oea_zo_bilevel_temp),
                oea_zo_bilevel_step=float(args.oea_zo_bilevel_step),
                oea_zo_bilevel_coverage_target=float(args.oea_zo_bilevel_coverage_target),
                oea_zo_bilevel_coverage_power=float(args.oea_zo_bilevel_coverage_power),
                oea_zo_reliable_metric=str(args.oea_zo_reliable_metric),
                oea_zo_reliable_threshold=float(args.oea_zo_reliable_threshold),
                oea_zo_reliable_alpha=float(args.oea_zo_reliable_alpha),
                oea_zo_trust_lambda=float(args.oea_zo_trust_lambda),
                oea_zo_trust_q0=str(args.oea_zo_trust_q0),
                oea_zo_drift_mode=str(args.oea_zo_drift_mode),
                oea_zo_drift_gamma=float(args.oea_zo_drift_gamma),
                oea_zo_drift_delta=float(args.oea_zo_drift_delta),
                oea_zo_selector=str(args.oea_zo_selector),
                oea_zo_iwcv_kappa=float(args.oea_zo_iwcv_kappa),
                oea_zo_calib_ridge_alpha=float(args.oea_zo_calib_ridge_alpha),
                oea_zo_calib_max_subjects=int(args.oea_zo_calib_max_subjects),
                oea_zo_calib_seed=int(args.oea_zo_calib_seed),
                oea_zo_calib_guard_c=float(args.oea_zo_calib_guard_c),
                oea_zo_calib_guard_threshold=float(args.oea_zo_calib_guard_threshold),
                oea_zo_calib_guard_margin=float(args.oea_zo_calib_guard_margin),
                oea_zo_min_improvement=float(args.oea_zo_min_improvement),
                oea_zo_holdout_fraction=float(args.oea_zo_holdout_fraction),
                oea_zo_warm_start=str(args.oea_zo_warm_start),
                oea_zo_warm_iters=int(args.oea_zo_warm_iters),
                oea_zo_fallback_min_marginal_entropy=float(args.oea_zo_fallback_min_marginal_entropy),
                oea_zo_iters=int(args.oea_zo_iters),
                oea_zo_lr=float(args.oea_zo_lr),
                oea_zo_mu=float(args.oea_zo_mu),
                oea_zo_k=int(args.oea_zo_k),
                oea_zo_seed=int(args.oea_zo_seed),
                oea_zo_l2=float(args.oea_zo_l2),
                mm_safe_mdm_guard_threshold=float(args.mm_safe_mdm_guard_threshold),
                mm_safe_mdm_min_pred_improve=float(args.mm_safe_mdm_min_pred_improve),
                mm_safe_mdm_drift_delta=float(args.mm_safe_mdm_drift_delta),
                stack_safe_fbcsp_guard_threshold=float(args.stack_safe_fbcsp_guard_threshold),
                stack_safe_fbcsp_min_pred_improve=float(args.stack_safe_fbcsp_min_pred_improve),
                stack_safe_fbcsp_drift_delta=float(args.stack_safe_fbcsp_drift_delta),
                stack_safe_fbcsp_max_pred_disagree=float(args.stack_safe_fbcsp_max_pred_disagree),
                stack_safe_tsa_guard_threshold=float(args.stack_safe_tsa_guard_threshold),
                stack_safe_tsa_min_pred_improve=float(args.stack_safe_tsa_min_pred_improve),
                stack_safe_tsa_drift_delta=float(args.stack_safe_tsa_drift_delta),
                stack_safe_anchor_guard_delta=float(args.stack_safe_anchor_guard_delta),
                stack_safe_anchor_probe_hard_worsen=float(args.stack_safe_anchor_probe_hard_worsen),
                stack_safe_min_pred_improve=float(args.stack_safe_min_pred_improve),
                stack_calib_per_family=bool(args.stack_calib_per_family),
                stack_calib_per_family_mode=str(args.stack_calib_per_family_mode),
                stack_calib_per_family_shrinkage=float(args.stack_calib_per_family_shrinkage),
                stack_feature_set=str(args.stack_feature_set),
                stack_candidate_families=stack_candidate_families,
                si_subject_lambda=float(args.si_subject_lambda),
                si_ridge=float(args.si_ridge),
                si_proj_dim=int(args.si_proj_dim),
                si_chan_candidate_ranks=si_chan_candidate_ranks,
                si_chan_candidate_lambdas=si_chan_candidate_lambdas,
                diagnostics_dir=out_dir if diagnose_subjects else None,
                diagnostics_subjects=diagnose_subjects,
                diagnostics_tag=method,
            )
        results_by_method[method] = results_df
        trial_predictions_by_method[method] = pred_df
        overall_by_method[method] = compute_metrics(
            y_true=y_true_all,
            y_pred=y_pred_all,
            y_proba=y_proba_all,
            class_order=class_order,
            average=config.metrics_average,
        )
        predictions_by_method[method] = (y_true_all, y_pred_all)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{date_prefix}_results.txt"
    write_results_txt_multi(
        results_by_method,
        config=config,
        output_path=results_path,
        metric_columns=metric_columns,
        overall_metrics_by_method=overall_by_method,
        method_details_by_method=method_details,
        protocol_name="LOSO",
        command_line=" ".join(sys.argv),
    )

    # Small, reproducible method-comparison table (mean / worst-subject / negative-transfer vs EA).
    base_method = "ea-csp-lda"
    base_df = results_by_method.get(base_method)
    base_acc = base_df.set_index("subject")["accuracy"].astype(float) if base_df is not None else None
    rows = []

    def _rankdata(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(x.size, dtype=np.float64)
        return ranks

    for method, df in sorted(results_by_method.items()):
        acc = df["accuracy"].astype(float)
        row = {
            "method": method,
            "n_subjects": int(df.shape[0]),
            "mean_accuracy": float(acc.mean()),
            "worst_accuracy": float(acc.min()),
        }
        # Optional: certificate/guard diagnostics (present for some safe selectors).
        if "chan_safe_guard_pos" in df.columns and "chan_safe_improve" in df.columns:
            p = df["chan_safe_guard_pos"].astype(float).to_numpy()
            imp = df["chan_safe_improve"].astype(float).to_numpy()
            mask = np.isfinite(p) & np.isfinite(imp)
            if int(np.sum(mask)) >= 2:
                row["guard_improve_pearson"] = float(np.corrcoef(p[mask], imp[mask])[0, 1])
                row["guard_improve_spearman"] = float(
                    np.corrcoef(_rankdata(p[mask]), _rankdata(imp[mask]))[0, 1]
                )
            else:
                row["guard_improve_pearson"] = float("nan")
                row["guard_improve_spearman"] = float("nan")
        if "chan_multi_guard_pos" in df.columns and "chan_multi_improve" in df.columns:
            p = df["chan_multi_guard_pos"].astype(float).to_numpy()
            imp = df["chan_multi_improve"].astype(float).to_numpy()
            mask = np.isfinite(p) & np.isfinite(imp)
            if int(np.sum(mask)) >= 2:
                row["guard_improve_pearson"] = float(np.corrcoef(p[mask], imp[mask])[0, 1])
                row["guard_improve_spearman"] = float(
                    np.corrcoef(_rankdata(p[mask]), _rankdata(imp[mask]))[0, 1]
                )
            else:
                row["guard_improve_pearson"] = float("nan")
                row["guard_improve_spearman"] = float("nan")
        if "chan_spsa_guard_pos" in df.columns and "chan_spsa_improve" in df.columns:
            p = df["chan_spsa_guard_pos"].astype(float).to_numpy()
            imp = df["chan_spsa_improve"].astype(float).to_numpy()
            mask = np.isfinite(p) & np.isfinite(imp)
            if int(np.sum(mask)) >= 2:
                row["guard_improve_pearson"] = float(np.corrcoef(p[mask], imp[mask])[0, 1])
                row["guard_improve_spearman"] = float(
                    np.corrcoef(_rankdata(p[mask]), _rankdata(imp[mask]))[0, 1]
                )
            else:
                row["guard_improve_pearson"] = float("nan")
                row["guard_improve_spearman"] = float("nan")
        if "stack_multi_guard_pos" in df.columns and "stack_multi_improve" in df.columns:
            p = df["stack_multi_guard_pos"].astype(float).to_numpy()
            imp = df["stack_multi_improve"].astype(float).to_numpy()
            mask = np.isfinite(p) & np.isfinite(imp)
            if int(np.sum(mask)) >= 2:
                row["guard_improve_pearson"] = float(np.corrcoef(p[mask], imp[mask])[0, 1])
                row["guard_improve_spearman"] = float(
                    np.corrcoef(_rankdata(p[mask]), _rankdata(imp[mask]))[0, 1]
                )
            else:
                row["guard_improve_pearson"] = float("nan")
                row["guard_improve_spearman"] = float("nan")
        if "mm_safe_guard_pos" in df.columns and "mm_safe_improve" in df.columns:
            p = df["mm_safe_guard_pos"].astype(float).to_numpy()
            imp = df["mm_safe_improve"].astype(float).to_numpy()
            mask = np.isfinite(p) & np.isfinite(imp)
            if int(np.sum(mask)) >= 2:
                row["guard_improve_pearson"] = float(np.corrcoef(p[mask], imp[mask])[0, 1])
                row["guard_improve_spearman"] = float(
                    np.corrcoef(_rankdata(p[mask]), _rankdata(imp[mask]))[0, 1]
                )
            else:
                row["guard_improve_pearson"] = float("nan")
                row["guard_improve_spearman"] = float("nan")
        if "chan_multi_ridge_pred_improve" in df.columns and "chan_multi_improve" in df.columns:
            p = df["chan_multi_ridge_pred_improve"].astype(float).to_numpy()
            imp = df["chan_multi_improve"].astype(float).to_numpy()
            mask = np.isfinite(p) & np.isfinite(imp)
            if int(np.sum(mask)) >= 2:
                row["cert_improve_pearson"] = float(np.corrcoef(p[mask], imp[mask])[0, 1])
                row["cert_improve_spearman"] = float(
                    np.corrcoef(_rankdata(p[mask]), _rankdata(imp[mask]))[0, 1]
                )
            else:
                row["cert_improve_pearson"] = float("nan")
                row["cert_improve_spearman"] = float("nan")
        if "chan_spsa_ridge_pred_improve" in df.columns and "chan_spsa_improve" in df.columns:
            p = df["chan_spsa_ridge_pred_improve"].astype(float).to_numpy()
            imp = df["chan_spsa_improve"].astype(float).to_numpy()
            mask = np.isfinite(p) & np.isfinite(imp)
            if int(np.sum(mask)) >= 2:
                row["cert_improve_pearson"] = float(np.corrcoef(p[mask], imp[mask])[0, 1])
                row["cert_improve_spearman"] = float(
                    np.corrcoef(_rankdata(p[mask]), _rankdata(imp[mask]))[0, 1]
                )
            else:
                row["cert_improve_pearson"] = float("nan")
                row["cert_improve_spearman"] = float("nan")
        if "stack_multi_ridge_pred_improve" in df.columns and "stack_multi_improve" in df.columns:
            p = df["stack_multi_ridge_pred_improve"].astype(float).to_numpy()
            imp = df["stack_multi_improve"].astype(float).to_numpy()
            mask = np.isfinite(p) & np.isfinite(imp)
            if int(np.sum(mask)) >= 2:
                row["cert_improve_pearson"] = float(np.corrcoef(p[mask], imp[mask])[0, 1])
                row["cert_improve_spearman"] = float(
                    np.corrcoef(_rankdata(p[mask]), _rankdata(imp[mask]))[0, 1]
                )
            else:
                row["cert_improve_pearson"] = float("nan")
                row["cert_improve_spearman"] = float("nan")
        if "mm_safe_ridge_pred_improve" in df.columns and "mm_safe_improve" in df.columns:
            p = df["mm_safe_ridge_pred_improve"].astype(float).to_numpy()
            imp = df["mm_safe_improve"].astype(float).to_numpy()
            mask = np.isfinite(p) & np.isfinite(imp)
            if int(np.sum(mask)) >= 2:
                row["cert_improve_pearson"] = float(np.corrcoef(p[mask], imp[mask])[0, 1])
                row["cert_improve_spearman"] = float(
                    np.corrcoef(_rankdata(p[mask]), _rankdata(imp[mask]))[0, 1]
                )
            else:
                row["cert_improve_pearson"] = float("nan")
                row["cert_improve_spearman"] = float("nan")
        if "chan_safe_accept" in df.columns:
            row["accept_rate"] = float(np.mean(df["chan_safe_accept"].astype(float)))
        if "chan_multi_accept" in df.columns:
            row["accept_rate"] = float(np.mean(df["chan_multi_accept"].astype(float)))
        if "chan_spsa_accept" in df.columns:
            row["accept_rate"] = float(np.mean(df["chan_spsa_accept"].astype(float)))
        if "stack_multi_accept" in df.columns:
            row["accept_rate"] = float(np.mean(df["stack_multi_accept"].astype(float)))
        if "mm_safe_accept" in df.columns:
            row["accept_rate"] = float(np.mean(df["mm_safe_accept"].astype(float)))
        if "chan_safe_guard_train_auc" in df.columns:
            row["guard_train_auc_mean"] = float(np.nanmean(df["chan_safe_guard_train_auc"].astype(float)))
        if "chan_multi_guard_train_auc" in df.columns:
            row["guard_train_auc_mean"] = float(np.nanmean(df["chan_multi_guard_train_auc"].astype(float)))
        if "chan_spsa_guard_train_auc" in df.columns:
            row["guard_train_auc_mean"] = float(np.nanmean(df["chan_spsa_guard_train_auc"].astype(float)))
        if "stack_multi_guard_train_auc" in df.columns:
            row["guard_train_auc_mean"] = float(np.nanmean(df["stack_multi_guard_train_auc"].astype(float)))
        if "mm_safe_guard_train_auc" in df.columns:
            row["guard_train_auc_mean"] = float(np.nanmean(df["mm_safe_guard_train_auc"].astype(float)))
        if "chan_safe_guard_train_spearman" in df.columns:
            row["guard_train_spearman_mean"] = float(np.nanmean(df["chan_safe_guard_train_spearman"].astype(float)))
        if "chan_multi_guard_train_spearman" in df.columns:
            row["guard_train_spearman_mean"] = float(
                np.nanmean(df["chan_multi_guard_train_spearman"].astype(float))
            )
        if "chan_spsa_guard_train_spearman" in df.columns:
            row["guard_train_spearman_mean"] = float(
                np.nanmean(df["chan_spsa_guard_train_spearman"].astype(float))
            )
        if "stack_multi_guard_train_spearman" in df.columns:
            row["guard_train_spearman_mean"] = float(
                np.nanmean(df["stack_multi_guard_train_spearman"].astype(float))
            )
        if "mm_safe_guard_train_spearman" in df.columns:
            row["guard_train_spearman_mean"] = float(
                np.nanmean(df["mm_safe_guard_train_spearman"].astype(float))
            )
        if base_acc is not None and method != base_method:
            m_acc = df.set_index("subject")["accuracy"].astype(float)
            common = m_acc.index.intersection(base_acc.index)
            if len(common) > 0:
                delta = m_acc.loc[common] - base_acc.loc[common]
                row["mean_delta_vs_ea"] = float(delta.mean())
                row["neg_transfer_rate_vs_ea"] = float(np.mean(delta < -1e-12))
            else:
                row["mean_delta_vs_ea"] = float("nan")
                row["neg_transfer_rate_vs_ea"] = float("nan")
        rows.append(row)
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(out_dir / f"{date_prefix}_method_comparison.csv", index=False)

    # Detailed per-trial predictions (for per-subject analysis).
    all_pred_parts: list[pd.DataFrame] = []
    for method, pred_df in trial_predictions_by_method.items():
        out_path = out_dir / f"{date_prefix}_{method}_predictions.csv"
        pred_df.to_csv(out_path, index=False)
        df2 = pred_df.copy()
        df2.insert(0, "method", method)
        all_pred_parts.append(df2)
    if all_pred_parts:
        pd.concat(all_pred_parts, axis=0, ignore_index=True).to_csv(
            out_dir / f"{date_prefix}_predictions_all_methods.csv", index=False
        )

    # Plots
    if args.no_plots:
        print("Skipping plots (--no-plots).")
        return

    # 1) CSP patterns (fit on full data) + confusion matrices for each method
    from csp_lda.model import fit_csp_lda
    from csp_lda.alignment import (
        EuclideanAligner,
        apply_spatial_transform,
        blend_with_identity,
        class_cov_diff,
        orthogonal_align_symmetric,
        sorted_eigh,
    )

    for method in results_by_method.keys():
        if method.endswith("-mdm"):
            # MDM-based methods do not use CSP; only plot confusion matrix.
            y_true_all, y_pred_all = predictions_by_method[method]
            plot_confusion_matrix(
                y_true_all,
                y_pred_all,
                labels=class_order,
                output_path=out_dir / f"{date_prefix}_{method}_confusion_matrix.png",
                title=f"{method} confusion matrix (LOSO, all subjects)",
            )
            continue
        if method in {"deep4net", "atcnet", "tcformer"} or method.endswith("-svc") or method.endswith("-lr"):
            # Non-CSP baselines: only plot confusion matrix.
            y_true_all, y_pred_all = predictions_by_method[method]
            plot_confusion_matrix(
                y_true_all,
                y_pred_all,
                labels=class_order,
                output_path=out_dir / f"{date_prefix}_{method}_confusion_matrix.png",
                title=f"{method} confusion matrix (LOSO, all subjects)",
            )
            continue

        if method == "ea-csp-lda" or method.startswith("ea-zo"):
            # Align each subject independently, then concatenate for a representative visualization.
            X_parts = []
            y_parts = []
            for s, sd in subject_data.items():
                X_parts.append(
                    EuclideanAligner(eps=float(args.oea_eps), shrinkage=float(args.oea_shrinkage)).fit_transform(
                        sd.X
                    )
                )
                y_parts.append(sd.y)
            X_fit = np.concatenate(X_parts, axis=0)
            y_fit = np.concatenate(y_parts, axis=0)
        elif method in {
            "ea-si-chan-csp-lda",
            "ea-si-chan-safe-csp-lda",
            "ea-si-chan-multi-safe-csp-lda",
            "ea-si-chan-spsa-safe-csp-lda",
        }:
            # Visualization-only: learn a single channel projector on EA-whitened full data (using true labels),
            # then fit CSP+LDA on the projected signals.
            # Note: SAFE variants perform per-fold selection; this is only a representative visualization.
            from csp_lda.subject_invariant import ChannelProjectorParams, learn_subject_invariant_channel_projector

            class_labels = tuple([str(c) for c in class_order])
            X_parts = []
            y_parts = []
            subj_parts = []
            for s, sd in subject_data.items():
                z = EuclideanAligner(eps=float(args.oea_eps), shrinkage=float(args.oea_shrinkage)).fit_transform(
                    sd.X
                )
                X_parts.append(z)
                y_parts.append(sd.y)
                subj_parts.append(np.full(sd.y.shape[0], int(s), dtype=int))

            X_all = np.concatenate(X_parts, axis=0)
            y_all = np.concatenate(y_parts, axis=0)
            subj_all = np.concatenate(subj_parts, axis=0)

            chan_params = ChannelProjectorParams(
                subject_lambda=float(args.si_subject_lambda),
                ridge=float(args.si_ridge),
                n_components=(int(args.si_proj_dim) if int(args.si_proj_dim) > 0 else None),
            )
            A = learn_subject_invariant_channel_projector(
                X=X_all,
                y=y_all,
                subjects=subj_all,
                class_order=class_labels,
                eps=float(args.oea_eps),
                shrinkage=float(args.oea_shrinkage),
                params=chan_params,
            )
            X_fit = apply_spatial_transform(A, X_all)
            y_fit = y_all
        elif method == "oea-cov-csp-lda":
            # Visualization-only: build U_ref from all subjects.
            covs = []
            ea_by_subject = {}
            for s, sd in subject_data.items():
                ea = EuclideanAligner(eps=float(args.oea_eps), shrinkage=float(args.oea_shrinkage)).fit(sd.X)
                ea_by_subject[int(s)] = ea
                covs.append(ea.cov_)
            c_ref = np.mean(np.stack(covs, axis=0), axis=0)
            _evals_ref, u_ref = sorted_eigh(c_ref)

            X_parts = []
            y_parts = []
            for s, sd in subject_data.items():
                ea = ea_by_subject[int(s)]
                z = ea.transform(sd.X)
                q = u_ref @ ea.eigvecs_.T
                q = blend_with_identity(q, float(args.oea_q_blend))
                X_parts.append(apply_spatial_transform(q, z))
                y_parts.append(sd.y)
            X_fit = np.concatenate(X_parts, axis=0)
            y_fit = np.concatenate(y_parts, axis=0)
        elif method == "oea-csp-lda" or method.startswith("oea-zo"):
            # Visualization-only: use true labels to compute a covariance-signature reference and Q_s for all subjects.
            class_labels = tuple([str(c) for c in class_order])

            ea_by_subject = {}
            z_by_subject = {}
            diffs = []
            for s, sd in subject_data.items():
                ea = EuclideanAligner(eps=float(args.oea_eps), shrinkage=float(args.oea_shrinkage)).fit(sd.X)
                ea_by_subject[int(s)] = ea
                z = ea.transform(sd.X)
                z_by_subject[int(s)] = z
                diffs.append(
                    class_cov_diff(
                        z,
                        sd.y,
                        class_order=class_labels,
                        eps=float(args.oea_eps),
                        shrinkage=float(args.oea_shrinkage),
                    )
                )
            d_ref = np.mean(np.stack(diffs, axis=0), axis=0)

            X_parts = []
            y_parts = []
            for s, sd in subject_data.items():
                d_s = class_cov_diff(
                    z_by_subject[int(s)],
                    sd.y,
                    class_order=class_labels,
                    eps=float(args.oea_eps),
                    shrinkage=float(args.oea_shrinkage),
                )
                q_s = orthogonal_align_symmetric(d_s, d_ref)
                q_s = blend_with_identity(q_s, float(args.oea_q_blend))
                X_parts.append(apply_spatial_transform(q_s, z_by_subject[int(s)]))
                y_parts.append(sd.y)
            X_fit = np.concatenate(X_parts, axis=0)
            y_fit = np.concatenate(y_parts, axis=0)
        else:
            X_fit, y_fit = X, y

        final_model = fit_csp_lda(X_fit, y_fit, n_components=config.model.csp_n_components)
        plot_csp_patterns(
            final_model.csp,
            info,
            output_path=out_dir / f"{date_prefix}_{method}_csp_patterns.png",
            title=f"{method} CSP patterns (n_components={config.model.csp_n_components})",
        )

        y_true_all, y_pred_all = predictions_by_method[method]
        plot_confusion_matrix(
            y_true_all,
            y_pred_all,
            labels=class_order,
            output_path=out_dir / f"{date_prefix}_{method}_confusion_matrix.png",
            title=f"{method} confusion matrix (LOSO, all subjects)",
        )

    # 2) Model performance comparison bar (per-subject accuracy)
    plot_method_comparison_bar(
        results_by_method,
        metric="accuracy",
        output_path=out_dir / f"{date_prefix}_model_compare_accuracy.png",
        title="LOSO accuracy by subject (model comparison)",
    )

    # Console short summary
    pd.set_option("display.width", 120)
    for method in results_by_method.keys():
        print(f"\n=== {method} ===")
        print(results_by_method[method])
    print("\nSaved:", results_path)


if __name__ == "__main__":
    main()
