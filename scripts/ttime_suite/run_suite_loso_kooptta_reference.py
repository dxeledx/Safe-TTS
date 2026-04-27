from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ttime_suite.kooptta_reference_plain_tta import (
    _make_loader,
    run_adabn_kooptta_reference,
    run_cotta_dteeg_reference,
    run_delta_dteeg_reference,
    run_note_kooptta_reference,
    run_pl_dteeg_reference,
    run_sar_dteeg_reference,
    run_shot_kooptta_reference,
    run_t3a_kooptta_reference,
    run_tent_kooptta_reference,
    run_ttime_dteeg_reference,
    streaming_normalize_trials_reference,
)
from scripts.ttime_suite.run_suite_loso import (
    _TTAArgs,
    _TrialDataset,
    _dataset_tag_from_data_dir,
    _ensure_dir,
    _load_or_compute_ea_whitenings,
    _make_model,
    _parse_subject_idxs,
    _predict_stream_noea,
    _predict_stream_with_iea,
    _train_baseline,
)
from scripts.ttime_suite.write_predictions_csv import load_class_order, write_predictions_csv


DEFAULT_METHODS = (
    "eegnet_noea,tent_kooptta_ref,adabn_kooptta_ref,note_kooptta_ref,t3a_kooptta_ref,shot_kooptta_ref,"
    "pl_dteeg_ref,sar_dteeg_ref,delta_dteeg_ref,cotta_dteeg_ref,ttime_dteeg_ref"
)


def _extract_trials_and_labels(
    *,
    X_mmap: np.ndarray,
    y_all: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    X = np.stack([np.asarray(X_mmap[int(idx)], dtype=np.float32, order="C") for idx in indices.tolist()], axis=0)
    y = np.asarray([int(y_all[int(idx)]) for idx in indices.tolist()], dtype=np.int64)
    return X, y


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run KoopTTA-reference plain TTA methods on strict LOSO exports.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", type=str, default=DEFAULT_METHODS)
    p.add_argument("--target-subject-idxs", type=str, default="ALL")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    p.add_argument("--baseline-train-ea", action="store_true", default=False, help="Optional source-side offline EA during baseline training.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=0)
    p.add_argument("--skip-merge", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--test-batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--tent-lr-scale", type=float, default=0.5)
    p.add_argument("--note-lr-scale", type=float, default=0.5)
    p.add_argument("--adabn-momentum", type=float, default=None)
    p.add_argument("--note-memory-type", type=str, default="fifo", choices=("fifo", "reservoir", "pbrs"))
    p.add_argument("--note-memory-size", type=int, default=50)
    p.add_argument("--note-replay-batch-size", type=int, default=32)
    p.add_argument("--note-update-every-x", type=int, default=8)
    p.add_argument("--note-use-learned-stats", action="store_true")
    p.add_argument("--note-bn-momentum", type=float, default=0.1)
    p.add_argument("--note-temperature", type=float, default=1.0)
    p.add_argument("--note-adapt-epochs", type=int, default=1)
    norm_group = p.add_mutually_exclusive_group()
    norm_group.add_argument(
        "--use-streaming-normalization",
        dest="use_streaming_normalization",
        action="store_true",
        help="Apply KoopTTA-style streaming target normalization before source_only/TTA evaluation.",
    )
    norm_group.add_argument(
        "--no-use-streaming-normalization",
        dest="use_streaming_normalization",
        action="store_false",
        help="Disable KoopTTA-style streaming target normalization.",
    )
    p.set_defaults(use_streaming_normalization=True)
    p.add_argument("--t3a-filter-k", type=int, default=16)
    p.add_argument("--shot-epochs", type=int, default=5)
    p.add_argument("--shot-lr-scale", type=float, default=0.5)
    p.add_argument("--shot-cls-par", type=float, default=0.3)
    p.add_argument("--shot-ent-par", type=float, default=1.0)
    p.add_argument("--shot-threshold", type=int, default=0)
    p.add_argument("--shot-distance", type=str, default="cosine", choices=("cosine", "euclidean"))
    p.add_argument("--shot-pseudo-rounds", type=int, default=1)
    p.add_argument("--sar-rho", type=float, default=0.05)
    p.add_argument("--sar-e0", type=float, default=None)
    p.add_argument("--sar-recovery-e0", type=float, default=0.2)
    p.add_argument("--delta-lambda-z", type=float, default=0.9)
    p.add_argument("--cotta-mt-alpha", type=float, default=0.999)
    p.add_argument("--cotta-rst-m", type=float, default=0.01)
    p.add_argument("--cotta-ap", type=float, default=0.9)
    p.add_argument("--cotta-aug-num", type=int, default=32)
    return p.parse_args(argv)


def main() -> int:
    args_ns = parse_args()
    data_dir = Path(args_ns.data_dir)
    out_dir = Path(args_ns.out_dir)
    _ensure_dir(out_dir)

    class_order = load_class_order(data_dir / "class_order.json")
    export_cfg_path = data_dir / "export_config.json"
    if not export_cfg_path.exists():
        raise RuntimeError(f"Missing export_config.json in {data_dir}")
    with export_cfg_path.open("r", encoding="utf-8") as f:
        export_cfg = json.load(f)
    sample_rate = float(export_cfg.get("resample"))

    meta = pd.read_csv(data_dir / "meta.csv")
    subject_orig = meta["subject_orig"].astype(int).to_numpy()
    subject_idx = np.load(data_dir / "subject_idx.npy").astype(np.int64)
    y_all = np.load(data_dir / "labels.npy").astype(np.int64)
    X_mmap = np.load(data_dir / "X.npy", mmap_mode="r")

    if X_mmap.ndim != 3:
        raise RuntimeError(f"X.npy expected 3D (trials, chn, time), got {X_mmap.shape}")
    if len(subject_idx) != len(y_all) or len(subject_idx) != len(subject_orig):
        raise RuntimeError("subject_idx/labels/meta length mismatch")

    n_trials, chn, time_n = map(int, X_mmap.shape)
    del n_trials
    n_subjects = int(len(set(subject_idx.tolist())))
    target_subject_idxs = _parse_subject_idxs(str(args_ns.target_subject_idxs), max_idx=n_subjects - 1)

    dataset_tag = _dataset_tag_from_data_dir(data_dir)
    ckpt_root = _REPO_ROOT / "runs_deeptransfer" / f"{dataset_tag}_kooptta_reference"
    _ensure_dir(ckpt_root)

    tta_args = _TTAArgs(
        chn=int(chn),
        time_sample_num=int(time_n),
        class_num=int(len(class_order)),
        sample_rate=float(sample_rate),
        lr=float(args_ns.lr),
        max_epochs=int(args_ns.max_epochs),
        batch_size=int(args_ns.batch_size),
        align=True,
        test_batch=int(args_ns.test_batch),
        stride=int(args_ns.stride),
        steps=int(args_ns.steps),
        t=float(args_ns.temp),
        calc_time=False,
        data_env=("gpu" if torch.cuda.is_available() else "local"),
    )

    methods = [m.strip() for m in str(args_ns.methods).split(",") if m.strip()]
    if not methods:
        raise RuntimeError("--methods parsed to empty list")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(int(args_ns.seed))
    np.random.seed(int(args_ns.seed))

    if bool(args_ns.baseline_train_ea):
        cache_path = data_dir / "offline_ea_whiten_by_subject.npy"
        ea_whiten_by_subject = _load_or_compute_ea_whitenings(
            cache_path=cache_path,
            X_mmap=X_mmap,
            subject_idx=subject_idx,
            n_subjects=n_subjects,
            chn=chn,
            recompute=False,
        )
    else:
        ea_whiten_by_subject = {}

    for t in target_subject_idxs:
        t = int(t)
        print(f"[loso] target_subject_idx={t} ({t + 1}/{n_subjects})")
        tar_mask = subject_idx == t
        src_mask = ~tar_mask
        src_indices = np.where(src_mask)[0]
        tar_indices = np.where(tar_mask)[0]
        if tar_indices.size <= 0:
            print(f"[loso] WARN: target_subject_idx={t} has 0 trials; skip.")
            continue

        tar_meta = meta.loc[tar_mask, ["session", "run", "trial"]].copy()
        tar_order = np.lexsort(
            (
                tar_meta["trial"].astype(int).to_numpy(),
                tar_meta["run"].astype(str).to_numpy(),
                tar_meta["session"].astype(str).to_numpy(),
            )
        )
        tar_indices = tar_indices[tar_order]

        subject_orig_id = int(subject_orig[int(tar_indices[0])])
        trials_in_subject = meta.loc[tar_indices, "trial"].astype(int).to_numpy()
        y_true_int = y_all[tar_indices]
        X_source_raw, y_source_raw = _extract_trials_and_labels(X_mmap=X_mmap, y_all=y_all, indices=src_indices)
        X_target_raw, y_target_raw = _extract_trials_and_labels(X_mmap=X_mmap, y_all=y_all, indices=tar_indices)
        if bool(args_ns.use_streaming_normalization):
            X_target_eval = streaming_normalize_trials_reference(X_target_raw)
        else:
            X_target_eval = X_target_raw

        ckpt_path = ckpt_root / f"EEGNet_S{t}_seed{int(args_ns.seed)}.ckpt"
        if (not ckpt_path.exists()) or (not bool(args_ns.resume)):
            train_ds = _TrialDataset(
                X_mmap=X_mmap,
                y=y_all,
                indices=src_indices,
                subject_idx=subject_idx,
                ea_whiten_by_subject=ea_whiten_by_subject,
                apply_ea=bool(args_ns.baseline_train_ea),
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=int(tta_args.batch_size),
                shuffle=True,
                drop_last=True,
                num_workers=int(args_ns.num_workers),
            )
            print(f"[baseline] TRAIN -> {ckpt_path}")
            _train_baseline(
                ckpt_path=ckpt_path,
                train_loader=train_loader,
                args=tta_args,
                device=device,
                torch_threads=int(args_ns.torch_threads) if int(args_ns.torch_threads) > 0 else None,
            )
            torch.manual_seed(int(args_ns.seed))
            np.random.seed(int(args_ns.seed))
        else:
            print(f"[baseline] SKIP ckpt exists: {ckpt_path}")

        target_online_loader = _make_loader(
            X=X_target_eval,
            y=y_target_raw,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        for m in methods:
            pred_csv = out_dir / "predictions" / f"method={m}" / f"subject={subject_orig_id}.csv"
            if pred_csv.exists() and bool(args_ns.resume):
                print(f"[pred] SKIP {m} subject={subject_orig_id} (exists)")
                continue

            print(f"[pred] RUN  method={m} subject={subject_orig_id}")
            model = _make_model(args=tta_args)
            model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu")))
            model = model.to(device)
            model.eval()

            if m == "eegnet_noea":
                proba = _predict_stream_noea(loader=target_online_loader, model=model, tta_args=tta_args)
            elif m == "eegnet_ea":
                proba = _predict_stream_with_iea(loader=target_online_loader, model=model, tta_args=tta_args)
            elif m == "tent_kooptta_ref":
                proba = run_tent_kooptta_reference(
                    X_target=X_target_eval,
                    y_true=y_target_raw,
                    model=model,
                    device=device,
                    batch_size=int(args_ns.test_batch),
                    num_workers=int(args_ns.num_workers),
                    lr=float(args_ns.lr) * float(args_ns.tent_lr_scale),
                )
            elif m == "adabn_kooptta_ref":
                proba = run_adabn_kooptta_reference(
                    X_target=X_target_eval,
                    y_true=y_target_raw,
                    model=model,
                    device=device,
                    batch_size=int(args_ns.test_batch),
                    num_workers=int(args_ns.num_workers),
                    momentum=None if args_ns.adabn_momentum is None else float(args_ns.adabn_momentum),
                )
            elif m == "note_kooptta_ref":
                proba = run_note_kooptta_reference(
                    X_target=X_target_eval,
                    y_true=y_target_raw,
                    model=model,
                    device=device,
                    replay_batch_size=int(args_ns.note_replay_batch_size),
                    memory_size=int(args_ns.note_memory_size),
                    update_every_x=int(args_ns.note_update_every_x),
                    num_workers=int(args_ns.num_workers),
                    lr=float(args_ns.lr) * float(args_ns.note_lr_scale),
                    memory_type=str(args_ns.note_memory_type),
                    use_learned_stats=bool(args_ns.note_use_learned_stats),
                    bn_momentum=float(args_ns.note_bn_momentum),
                    temperature=float(args_ns.note_temperature),
                    adapt_epochs=int(args_ns.note_adapt_epochs),
                )
            elif m == "t3a_kooptta_ref":
                proba = run_t3a_kooptta_reference(
                    X_source=X_source_raw,
                    y_source=y_source_raw,
                    X_target=X_target_eval,
                    y_true=y_target_raw,
                    model=model,
                    device=device,
                    batch_size=int(args_ns.test_batch),
                    num_workers=int(args_ns.num_workers),
                    filter_k=int(args_ns.t3a_filter_k),
                )
            elif m == "shot_kooptta_ref":
                proba = run_shot_kooptta_reference(
                    X_target=X_target_eval,
                    y_true=y_target_raw,
                    model=model,
                    device=device,
                    batch_size=int(args_ns.test_batch),
                    num_workers=int(args_ns.num_workers),
                    epochs=int(args_ns.shot_epochs),
                    base_lr=float(args_ns.lr) * float(args_ns.shot_lr_scale),
                    cls_par=float(args_ns.shot_cls_par),
                    ent_par=float(args_ns.shot_ent_par),
                    threshold=int(args_ns.shot_threshold),
                    distance=str(args_ns.shot_distance),
                    pseudo_rounds=int(args_ns.shot_pseudo_rounds),
                )
            elif m == "pl_dteeg_ref":
                proba = run_pl_dteeg_reference(
                    X_target=X_target_eval,
                    y_true=y_target_raw,
                    model=model,
                    device=device,
                    batch_size=int(args_ns.test_batch),
                    stride=int(args_ns.stride),
                    steps=int(args_ns.steps),
                    lr=float(args_ns.lr),
                )
            elif m == "sar_dteeg_ref":
                proba = run_sar_dteeg_reference(
                    X_target=X_target_eval,
                    y_true=y_target_raw,
                    model=model,
                    device=device,
                    batch_size=int(args_ns.test_batch),
                    stride=int(args_ns.stride),
                    steps=int(args_ns.steps),
                    lr=float(args_ns.lr),
                    rho=float(args_ns.sar_rho),
                    entropy_margin=None if args_ns.sar_e0 is None else float(args_ns.sar_e0),
                    recovery_threshold=float(args_ns.sar_recovery_e0),
                    temperature=float(args_ns.temp),
                )
            elif m == "delta_dteeg_ref":
                proba = run_delta_dteeg_reference(
                    X_target=X_target_eval,
                    y_true=y_target_raw,
                    model=model,
                    device=device,
                    batch_size=int(args_ns.test_batch),
                    stride=int(args_ns.stride),
                    steps=int(args_ns.steps),
                    lr=float(args_ns.lr),
                    temperature=float(args_ns.temp),
                    lambda_z=float(args_ns.delta_lambda_z),
                )
            elif m == "cotta_dteeg_ref":
                proba = run_cotta_dteeg_reference(
                    X_target=X_target_eval,
                    y_true=y_target_raw,
                    model=model,
                    device=device,
                    batch_size=int(args_ns.test_batch),
                    stride=int(args_ns.stride),
                    steps=int(args_ns.steps),
                    lr=float(args_ns.lr),
                    mt_alpha=float(args_ns.cotta_mt_alpha),
                    rst_m=float(args_ns.cotta_rst_m),
                    ap=float(args_ns.cotta_ap),
                    aug_num=int(args_ns.cotta_aug_num),
                )
            elif m == "ttime_dteeg_ref":
                proba = run_ttime_dteeg_reference(
                    X_target=X_target_eval,
                    y_true=y_target_raw,
                    model=model,
                    device=device,
                    batch_size=int(args_ns.test_batch),
                    stride=int(args_ns.stride),
                    steps=int(args_ns.steps),
                    lr=float(args_ns.lr),
                    temperature=float(args_ns.temp),
                )
            else:
                raise ValueError(f"Unknown method: {m}")

            proba = np.asarray(proba, dtype=np.float64).reshape(-1, int(tta_args.class_num))
            if proba.shape[0] != y_true_int.shape[0]:
                raise RuntimeError(f"{m}: proba n_trials mismatch: {proba.shape[0]} vs {y_true_int.shape[0]}")

            write_predictions_csv(
                out_csv=pred_csv,
                method=m,
                subject=subject_orig_id,
                y_true_int=y_true_int,
                proba=proba,
                class_order=class_order,
                trial=trials_in_subject,
            )

    if bool(args_ns.skip_merge):
        print("[done] skip merge: predictions_all_methods.csv not written")
        return 0

    pred_rows: list[pd.DataFrame] = []
    pred_root = out_dir / "predictions"
    for csv_path in sorted(pred_root.rglob("subject=*.csv")):
        pred_rows.append(pd.read_csv(csv_path))
    if not pred_rows:
        raise RuntimeError(f"No per-subject predictions found under: {pred_root}")
    merged = pd.concat(pred_rows, axis=0, ignore_index=True)
    merged.to_csv(out_dir / "predictions_all_methods.csv", index=False)
    print(f"[done] wrote: {out_dir / 'predictions_all_methods.csv'}  rows={len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
