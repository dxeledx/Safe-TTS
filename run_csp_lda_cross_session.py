from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd

from csp_lda.config import ExperimentConfig, ModelConfig, PreprocessingConfig
from csp_lda.data import MoabbMotorImageryLoader, split_by_subject_session
from csp_lda.evaluation import compute_metrics, cross_session_within_subject_evaluation
from csp_lda.plots import plot_confusion_matrix, plot_method_comparison_bar
from csp_lda.reporting import today_yyyymmdd, write_results_txt_multi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CSP+LDA within-subject cross-session on MOABB MotorImagery datasets."
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Output root directory. Run outputs go to OUT_DIR/YYYYMMDD/<N>class/cross_session/HHMMSS_*.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run subfolder name. Default: current time HHMMSS.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="BNCI2014_001",
        help=(
            "MOABB dataset name (e.g., BNCI2014_001, Cho2017, PhysionetMI, Schirrmeister2017). "
            "Default: BNCI2014_001 (BCI IV 2a)."
        ),
    )
    p.add_argument("--fmin", type=float, default=8.0)
    p.add_argument("--fmax", type=float, default=30.0)
    p.add_argument("--tmin", type=float, default=0.5)
    p.add_argument("--tmax", type=float, default=3.5)
    p.add_argument("--resample", type=float, default=250.0)
    p.add_argument("--n-components", type=int, default=4, help="CSP components (n_components).")
    p.add_argument(
        "--preprocess",
        choices=["moabb", "paper_fir"],
        default="moabb",
        help="Preprocessing pipeline: 'moabb' or 'paper_fir' (causal 50-order FIR Hamming).",
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
        default="left_hand,right_hand,feet,tongue",
        help="Comma-separated events/classes.",
    )
    p.add_argument(
        "--train-sessions",
        type=str,
        default="0train",
        help="Comma-separated MOABB session names used for training (default: 0train).",
    )
    p.add_argument(
        "--test-sessions",
        type=str,
        default="1test",
        help="Comma-separated MOABB session names used for testing (default: 1test).",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="ea-csp-lda,ea-zo-imr-csp-lda",
        help=(
            "Comma-separated methods to run: "
            "csp-lda, ea-csp-lda, lea-csp-lda, lea-rot-csp-lda, "
            "oea-cov-csp-lda, oea-csp-lda, "
            "oea-zo-csp-lda, oea-zo-ent-csp-lda, oea-zo-im-csp-lda, oea-zo-imr-csp-lda, "
            "oea-zo-pce-csp-lda, oea-zo-conf-csp-lda, "
            "ea-zo-csp-lda, ea-zo-ent-csp-lda, ea-zo-im-csp-lda, ea-zo-imr-csp-lda, "
            "ea-zo-pce-csp-lda, ea-zo-conf-csp-lda, "
            "rpa-zo-csp-lda, rpa-zo-ent-csp-lda, rpa-zo-im-csp-lda, rpa-zo-imr-csp-lda, "
            "rpa-zo-pce-csp-lda, rpa-zo-conf-csp-lda, "
            "tsa-zo-csp-lda, tsa-zo-ent-csp-lda, tsa-zo-im-csp-lda, tsa-zo-imr-csp-lda, "
            "tsa-zo-pce-csp-lda, tsa-zo-conf-csp-lda"
        ),
    )
    p.add_argument("--oea-eps", type=float, default=1e-10)
    p.add_argument("--oea-shrinkage", type=float, default=0.0)
    p.add_argument("--oea-pseudo-iters", type=int, default=2)
    p.add_argument("--oea-pseudo-mode", choices=["hard", "soft"], default="hard")
    p.add_argument("--oea-pseudo-confidence", type=float, default=0.0)
    p.add_argument("--oea-pseudo-topk-per-class", type=int, default=0)
    p.add_argument("--oea-pseudo-balance", action="store_true")
    p.add_argument("--oea-q-blend", type=float, default=0.3)

    p.add_argument(
        "--oea-zo-objective",
        choices=["entropy", "infomax", "pseudo_ce", "confidence", "lda_nll", "entropy_bilevel", "infomax_bilevel"],
        default="infomax_bilevel",
    )
    p.add_argument(
        "--oea-zo-transform",
        choices=["orthogonal", "rot_scale", "local_mix", "local_mix_then_ea"],
        default="orthogonal",
        help=(
            "Channel-space transform family for ZO. "
            "'orthogonal' uses Q∈O(C); "
            "'rot_scale' uses A=diag(exp(s))·Q; "
            "'local_mix' uses a row-stochastic local mixing A; "
            "'local_mix_then_ea' applies EA whitening after the local mixing (A→EA)."
        ),
    )
    p.add_argument(
        "--oea-zo-localmix-neighbors",
        type=int,
        default=4,
        help="For transform=local_mix/local_mix_then_ea: k nearest neighbors per channel (k>=0).",
    )
    p.add_argument(
        "--oea-zo-localmix-self-bias",
        type=float,
        default=3.0,
        help=(
            "For transform=local_mix/local_mix_then_ea: non-negative logit bias for the self-weight "
            "(larger keeps A closer to identity)."
        ),
    )
    p.add_argument("--oea-zo-infomax-lambda", type=float, default=1.0)
    p.add_argument(
        "--oea-zo-marginal-mode",
        choices=["none", "l2_uniform", "kl_uniform", "hinge_uniform", "hard_min", "kl_prior"],
        default="none",
    )
    p.add_argument("--oea-zo-marginal-beta", type=float, default=0.0)
    p.add_argument("--oea-zo-marginal-tau", type=float, default=0.05)
    p.add_argument("--oea-zo-marginal-prior", choices=["uniform", "source", "anchor_pred"], default="uniform")
    p.add_argument("--oea-zo-marginal-prior-mix", type=float, default=0.0)
    p.add_argument("--oea-zo-bilevel-iters", type=int, default=5)
    p.add_argument("--oea-zo-bilevel-temp", type=float, default=1.0)
    p.add_argument("--oea-zo-bilevel-step", type=float, default=1.0)
    p.add_argument("--oea-zo-bilevel-coverage-target", type=float, default=0.5)
    p.add_argument("--oea-zo-bilevel-coverage-power", type=float, default=1.0)
    p.add_argument(
        "--oea-zo-reliable-metric",
        choices=["none", "confidence", "entropy"],
        default="entropy",
    )
    p.add_argument("--oea-zo-reliable-threshold", type=float, default=0.7)
    p.add_argument("--oea-zo-reliable-alpha", type=float, default=10.0)
    p.add_argument("--oea-zo-trust-lambda", type=float, default=0.0)
    p.add_argument("--oea-zo-trust-q0", choices=["identity", "delta"], default="identity")
    p.add_argument("--oea-zo-drift-mode", choices=["none", "penalty", "hard"], default="none")
    p.add_argument("--oea-zo-drift-gamma", type=float, default=0.0)
    p.add_argument("--oea-zo-drift-delta", type=float, default=0.0)
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
            "oracle",
        ],
        default="objective",
    )
    p.add_argument(
        "--oea-zo-iwcv-kappa",
        type=float,
        default=1.0,
        help="UCB penalty strength for iwcv_ucb selector (kappa>=0).",
    )
    p.add_argument("--oea-zo-calib-ridge-alpha", type=float, default=1.0)
    p.add_argument("--oea-zo-calib-max-subjects", type=int, default=0)
    p.add_argument("--oea-zo-calib-seed", type=int, default=0)
    p.add_argument("--oea-zo-calib-guard-c", type=float, default=1.0)
    p.add_argument("--oea-zo-calib-guard-threshold", type=float, default=0.5)
    p.add_argument("--oea-zo-calib-guard-margin", type=float, default=0.0)
    p.add_argument("--oea-zo-min-improvement", type=float, default=0.0)
    p.add_argument("--oea-zo-holdout-fraction", type=float, default=0.0)
    p.add_argument("--oea-zo-warm-start", choices=["none", "delta"], default="none")
    p.add_argument("--oea-zo-warm-iters", type=int, default=1)
    p.add_argument("--oea-zo-fallback-min-marginal-entropy", type=float, default=0.0)
    p.add_argument("--oea-zo-iters", type=int, default=30)
    p.add_argument("--oea-zo-lr", type=float, default=0.5)
    p.add_argument("--oea-zo-mu", type=float, default=0.1)
    p.add_argument("--oea-zo-k", type=int, default=50)
    p.add_argument("--oea-zo-seed", type=int, default=0)
    p.add_argument("--oea-zo-l2", type=float, default=0.0)

    p.add_argument("--diagnose-subjects", type=str, default="", help="Comma-separated subject ids for diagnostics.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="moabb")

    events = [e.strip() for e in str(args.events).split(",") if e.strip()]
    train_sessions = [s.strip() for s in str(args.train_sessions).split(",") if s.strip()]
    test_sessions = [s.strip() for s in str(args.test_sessions).split(",") if s.strip()]
    sessions = sorted(set(train_sessions + test_sessions))

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
    methods = list(dict.fromkeys(methods_canon))
    date_prefix = today_yyyymmdd()
    n_classes = len(events)
    dataset_slug = re.sub(r"[^0-9a-zA-Z]+", "", str(args.dataset).strip().lower()) or "dataset"
    run_name = args.run_name or f"{datetime.now().strftime('%H%M%S')}_{dataset_slug}"
    out_dir = Path(args.out_dir) / date_prefix / f"{n_classes}class" / "cross_session" / run_name

    preprocessing = PreprocessingConfig(
        fmin=float(args.fmin),
        fmax=float(args.fmax),
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        resample=float(args.resample),
        events=tuple(events),
        sessions=tuple(sessions),
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
    X, y, meta = loader.load_arrays(dtype="float32")
    subject_session_data = split_by_subject_session(X, y, meta)
    info = loader.load_epochs_info()

    metric_columns = ["accuracy", "precision", "recall", "f1", "auc", "kappa"]
    class_order = list(config.preprocessing.events)

    results_by_method: dict[str, pd.DataFrame] = {}
    overall_by_method: dict[str, dict[str, float]] = {}
    predictions_by_method: dict[str, tuple] = {}
    trial_predictions_by_method: dict[str, pd.DataFrame] = {}
    method_details: dict[str, str] = {}

    diagnose_subjects = [int(s) for s in str(args.diagnose_subjects).split(",") if s.strip()]

    for method in methods:
        zo_objective_override: str | None = None

        if method == "csp-lda":
            alignment = "none"
            method_details[method] = "No alignment."
        elif method == "ea-csp-lda":
            alignment = "ea"
            method_details[method] = "EA: session-wise whitening (train/test sessions aligned independently)."
        elif method == "lea-csp-lda":
            alignment = "rpa"
            method_details[method] = "LEA: log-Euclidean session-wise whitening (SPD mean)."
        elif method == "lea-rot-csp-lda":
            alignment = "tsa"
            method_details[method] = "LEA + pseudo-label Procrustes target rotation (closed-form)."
        elif method == "oea-cov-csp-lda":
            alignment = "oea_cov"
            method_details[method] = "OEA (cov-eig): align test eigen-basis to train eigen-basis (within subject)."
        elif method == "oea-csp-lda":
            alignment = "oea"
            method_details[method] = (
                "OEA (signature): use labeled train session(s) signature as reference, "
                f"pseudo iters={args.oea_pseudo_iters} on test session(s)."
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
            method_details[method] = f"OEA-ZO: objective={zo_objective_override or args.oea_zo_objective}."
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
            method_details[method] = f"EA-ZO: objective={zo_objective_override or args.oea_zo_objective}."
        elif method in {
            "rpa-zo-ent-csp-lda",
            "rpa-zo-im-csp-lda",
            "rpa-zo-imr-csp-lda",
            "rpa-zo-pce-csp-lda",
            "rpa-zo-conf-csp-lda",
            "rpa-zo-csp-lda",
        }:
            alignment = "rpa_zo"
            if method == "rpa-zo-ent-csp-lda":
                zo_objective_override = "entropy"
            elif method == "rpa-zo-im-csp-lda":
                zo_objective_override = "infomax"
            elif method == "rpa-zo-imr-csp-lda":
                zo_objective_override = "infomax_bilevel"
            elif method == "rpa-zo-pce-csp-lda":
                zo_objective_override = "pseudo_ce"
            elif method == "rpa-zo-conf-csp-lda":
                zo_objective_override = "confidence"
            method_details[method] = f"RPA-ZO: objective={zo_objective_override or args.oea_zo_objective}."
        elif method in {
            "tsa-zo-ent-csp-lda",
            "tsa-zo-im-csp-lda",
            "tsa-zo-imr-csp-lda",
            "tsa-zo-pce-csp-lda",
            "tsa-zo-conf-csp-lda",
            "tsa-zo-csp-lda",
        }:
            alignment = "tsa_zo"
            if method == "tsa-zo-ent-csp-lda":
                zo_objective_override = "entropy"
            elif method == "tsa-zo-im-csp-lda":
                zo_objective_override = "infomax"
            elif method == "tsa-zo-imr-csp-lda":
                zo_objective_override = "infomax_bilevel"
            elif method == "tsa-zo-pce-csp-lda":
                zo_objective_override = "pseudo_ce"
            elif method == "tsa-zo-conf-csp-lda":
                zo_objective_override = "confidence"
            method_details[method] = f"TSA-ZO: objective={zo_objective_override or args.oea_zo_objective}."
        else:
            raise ValueError(f"Unknown method '{method}'.")

        (
            results_df,
            pred_df,
            y_true_all,
            y_pred_all,
            y_proba_all,
            _class_order,
            _models_by_subject,
        ) = cross_session_within_subject_evaluation(
            subject_session_data,
            train_sessions=train_sessions,
            test_sessions=test_sessions,
            class_order=class_order,
            channel_names=list(info["ch_names"]),
            n_components=config.model.csp_n_components,
            average=config.metrics_average,
            alignment=alignment,
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
            oea_zo_reliable_metric=str(args.oea_zo_reliable_metric),
            oea_zo_reliable_threshold=float(args.oea_zo_reliable_threshold),
            oea_zo_reliable_alpha=float(args.oea_zo_reliable_alpha),
            oea_zo_trust_lambda=float(args.oea_zo_trust_lambda),
            oea_zo_trust_q0=str(args.oea_zo_trust_q0),
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
        protocol_name=f"CrossSession ({','.join(train_sessions)}→{','.join(test_sessions)})",
        command_line=" ".join(sys.argv),
    )

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

    # Plots (aggregate across subjects)
    for method, (y_true_all, y_pred_all) in predictions_by_method.items():
        plot_confusion_matrix(
            y_true_all,
            y_pred_all,
            labels=class_order,
            output_path=out_dir / f"{date_prefix}_{method}_confusion.png",
            title=f"{method} Confusion ({','.join(train_sessions)}→{','.join(test_sessions)})",
        )
    plot_method_comparison_bar(
        results_by_method,
        metric="accuracy",
        output_path=out_dir / f"{date_prefix}_accuracy_by_subject.png",
        title=f"Cross-session accuracy ({','.join(train_sessions)}→{','.join(test_sessions)})",
    )

    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
