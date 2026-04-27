from __future__ import annotations

"""
Run DeepTransferEEG (T-TIME suite) in strict LOSO protocol and export SAFE_TTA predictions.

This runner is designed to be executed on remote server with the repo synced to:
  /path/to/remote/Safe-TTS

It consumes data exported by:
  scripts/ttime_suite/export_moabb_for_deeptransfer.py

Outputs
-------
<out_dir>/
  predictions/
    method=<method_name>/
      subject=<subject_orig>.csv
  predictions_all_methods.csv

Baseline checkpoints are stored in:
  runs_deeptransfer/<dataset_tag>/EEGNet_S<target_subject_idx>_seed<seed>.ckpt
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.linalg import fractional_matrix_power
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, Dataset

from scripts.ttime_suite.write_predictions_csv import load_class_order, write_predictions_csv
from scripts.ttime_suite.kooptta_align_coral import run_coral_kooptta_aligned
from scripts.ttime_suite.kooptta_align_shot import run_shot_kooptta_aligned
from scripts.ttime_suite.kooptta_align_tent_cotta import (
    run_cotta_note_aligned,
    run_tent_kooptta_aligned,
    run_tent_kooptta_strict,
)


DEFAULT_METHODS = (
    "eegnet_ea,tent,t3a,cotta,shot,coral"
)

_EA_CACHE_FILENAME = "offline_ea_whiten_by_subject.npy"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _deeptransfer_tl_dir() -> Path:
    return _repo_root() / "third_party" / "DeepTransferEEG" / "tl"


def _add_deeptransfer_to_syspath() -> None:
    tl_dir = _deeptransfer_tl_dir()
    sys.path.insert(0, str(tl_dir))


def _load_bn_adapt_module():
    # filename contains a hyphen, so normal import does not work.
    import importlib.machinery
    import importlib.util

    path = _deeptransfer_tl_dir() / "bn-adapt.py"
    if not path.exists():
        raise FileNotFoundError(path)
    loader = importlib.machinery.SourceFileLoader("deeptransfer_bn_adapt", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError("Failed to build import spec for bn-adapt.py")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _parse_subject_idxs(s: str, *, max_idx: int) -> list[int]:
    s = str(s).strip()
    if not s or s.upper() == "ALL":
        return list(range(int(max_idx) + 1))
    if "," in s:
        out = [int(x) for x in s.split(",") if x.strip()]
        return sorted(set(out))
    if "-" in s:
        a, b = s.split("-", 1)
        a_i = int(a)
        b_i = int(b)
        if b_i < a_i:
            raise ValueError(f"Invalid range: {s!r}")
        return list(range(a_i, b_i + 1))
    return [int(s)]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _dataset_tag_from_data_dir(data_dir: Path) -> str:
    # Keep stable across runs, avoid spaces.
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", data_dir.name)


@dataclass(frozen=True)
class _TTAArgs:
    chn: int
    time_sample_num: int
    class_num: int
    sample_rate: float

    # model/training
    lr: float
    max_epochs: int
    batch_size: int

    # stream/TTA
    align: bool
    test_batch: int
    stride: int
    steps: int
    t: float
    calc_time: bool

    # execution env
    data_env: str  # 'local' for CPU


class _TrialDataset(Dataset):
    def __init__(
        self,
        *,
        X_mmap: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        subject_idx: np.ndarray,
        ea_whiten_by_subject: dict[int, np.ndarray] | None,
        apply_ea: bool,
    ) -> None:
        self.X_mmap = X_mmap
        self.y = np.asarray(y, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.subject_idx = np.asarray(subject_idx, dtype=np.int64)
        self.ea_whiten_by_subject = ea_whiten_by_subject or {}
        self.apply_ea = bool(apply_ea)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self.indices[int(i)])
        x = np.asarray(self.X_mmap[idx], dtype=np.float64)  # (chn, time)
        if self.apply_ea:
            s = int(self.subject_idx[idx])
            W = self.ea_whiten_by_subject.get(s)
            if W is None:
                raise KeyError(f"Missing EA whitening for subject_idx={s}")
            x = W @ x
        x = np.asarray(x, dtype=np.float32, order="C")
        xt = torch.from_numpy(x).unsqueeze(0)  # (1, chn, time)
        yt = int(self.y[idx])
        return xt, yt


def _compute_offline_ea_whitening(*, X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    # Match DeepTransferEEG EA: cov per trial via np.cov, then mean, then invsqrt.
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size <= 0:
        raise ValueError("Empty indices for offline EA.")

    first = np.asarray(X[int(indices[0])], dtype=np.float64)
    chn = int(first.shape[0])
    cov_sum = np.zeros((chn, chn), dtype=np.float64)
    for idx in indices.tolist():
        trial = np.asarray(X[int(idx)], dtype=np.float64)
        cov_sum += np.cov(trial)
    ref = cov_sum / float(indices.size)
    # Add a small trace-scaled diagonal for numerical stability (safe, paper-aligned).
    trace = float(np.trace(ref))
    ref = ref + 1e-6 * (trace / float(chn)) * np.eye(chn, dtype=np.float64)
    W = fractional_matrix_power(ref, -0.5)
    return np.asarray(W, dtype=np.float64)


def _precompute_ea_whitenings(
    *,
    X_mmap: np.ndarray,
    subject_idx: np.ndarray,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for s in sorted(set(subject_idx.tolist())):
        idxs = np.where(subject_idx == int(s))[0]
        out[int(s)] = _compute_offline_ea_whitening(X=X_mmap, indices=idxs)
    return out


def _load_or_compute_ea_whitenings(
    *,
    cache_path: Path,
    X_mmap: np.ndarray,
    subject_idx: np.ndarray,
    n_subjects: int,
    chn: int,
    recompute: bool,
) -> dict[int, np.ndarray]:
    cache_path = Path(cache_path)
    lock_path = Path(str(cache_path) + ".lock")

    def _load() -> dict[int, np.ndarray]:
        arr = np.load(cache_path)
        if arr.shape != (int(n_subjects), int(chn), int(chn)):
            raise RuntimeError(
                f"EA cache shape mismatch: {arr.shape} (expected {(int(n_subjects), int(chn), int(chn))}) path={cache_path}"
            )
        return {int(s): np.asarray(arr[int(s)], dtype=np.float64) for s in range(int(n_subjects))}

    if cache_path.exists() and not recompute:
        print(f"[ea] load cache: {cache_path}")
        return _load()

    # Acquire a coarse lock to avoid wasting CPU when running shards concurrently.
    acquired_lock = False
    if cache_path.parent:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not recompute:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired_lock = True
        except FileExistsError:
            acquired_lock = False

        if not acquired_lock:
            t0 = time.time()
            print(f"[ea] wait for cache lock: {lock_path}")
            while True:
                if cache_path.exists():
                    print(f"[ea] cache ready after {time.time() - t0:.1f}s: {cache_path}")
                    return _load()
                # Stale lock: 12h
                try:
                    st = lock_path.stat()
                    if time.time() - st.st_mtime > 12 * 3600:
                        print(f"[ea] WARN: stale lock detected, recomputing cache: {lock_path}")
                        try:
                            lock_path.unlink()
                        except FileNotFoundError:
                            pass
                        break
                except FileNotFoundError:
                    # lock disappeared but cache not present yet; keep waiting a bit.
                    pass
                if time.time() - t0 > 24 * 3600:
                    raise TimeoutError(f"Timed out waiting for EA cache lock: {lock_path}")
                time.sleep(10)

    # Either recompute explicitly, or we acquired the lock, or lock was stale.
    try:
        if not acquired_lock and not recompute:
            # Try to acquire again after stale-lock cleanup.
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired_lock = True

        print(f"[ea] compute offline whitening for {n_subjects} subjects ...")
        t0 = time.time()
        ea = _precompute_ea_whitenings(X_mmap=X_mmap, subject_idx=subject_idx)
        arr = np.stack([ea[int(s)] for s in range(int(n_subjects))], axis=0)
        tmp_path = Path(str(cache_path) + ".tmp.npy")
        np.save(tmp_path, arr)
        os.replace(str(tmp_path), str(cache_path))
        print(f"[ea] wrote cache: {cache_path}  seconds={time.time() - t0:.1f}")
        return ea
    finally:
        if acquired_lock:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _make_model(*, args: _TTAArgs) -> nn.Module:
    _add_deeptransfer_to_syspath()
    from utils.network import backbone_net  # type: ignore

    ns = argparse.Namespace(
        chn=int(args.chn),
        time_sample_num=int(args.time_sample_num),
        sample_rate=float(args.sample_rate),
        class_num=int(args.class_num),
        backbone="EEGNet",
        data_env=str(args.data_env),
    )
    netF, netC = backbone_net(ns, return_type="xy")
    return nn.Sequential(netF, netC)


def _train_baseline(
    *,
    ckpt_path: Path,
    train_loader: DataLoader,
    args: _TTAArgs,
    device: torch.device,
    torch_threads: int | None,
) -> None:
    _ensure_dir(ckpt_path.parent)
    if torch_threads is not None and int(torch_threads) > 0:
        torch.set_num_threads(int(torch_threads))

    model = _make_model(args=args).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    for _epoch in range(int(args.max_epochs)):
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = torch.as_tensor(yb, dtype=torch.long, device=device)
            logits = model(xb)[1]
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    # Save on CPU for portability (load with map_location=cpu is still supported).
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt_path)


def _predict_stream_with_iea(
    *,
    loader: DataLoader,
    model: nn.Module,
    tta_args: _TTAArgs,
) -> np.ndarray:
    # Incremental EA, no parameter updates (baseline "eegnet_ea").
    model.eval()
    device = next(model.parameters()).device
    y_pred: list[np.ndarray] = []
    R = 0
    data_cum = None
    softmax = nn.Softmax(dim=1)

    it = iter(loader)
    for i in range(len(loader)):
        data = next(it)
        inputs = data[0]
        inputs = inputs.reshape(1, 1, inputs.shape[-2], inputs.shape[-1]).cpu()

        if i == 0:
            data_cum = inputs.float().cpu()
        else:
            data_cum = torch.cat((data_cum, inputs.float().cpu()), 0)

        if tta_args.align:
            # Use the same EA_online update schedule as DeepTransferEEG.
            if i == 0:
                sample = data_cum.reshape(int(tta_args.chn), int(tta_args.time_sample_num))
            else:
                sample = data_cum[i].reshape(int(tta_args.chn), int(tta_args.time_sample_num))

            from utils.alg_utils import EA_online  # type: ignore

            R = EA_online(sample.numpy(), R, i)
            sqrt_ref = fractional_matrix_power(R, -0.5)
            sample = np.dot(sqrt_ref, sample.numpy()).reshape(1, 1, int(tta_args.chn), int(tta_args.time_sample_num))
        else:
            sample = data_cum[i].numpy().reshape(1, 1, int(tta_args.chn), int(tta_args.time_sample_num))

        x = torch.from_numpy(sample).to(torch.float32).to(device, non_blocking=True)
        _fea, logits = model(x)
        proba = softmax(logits).detach().cpu().numpy()
        y_pred.append(proba.reshape(-1))

    proba_m = np.asarray(y_pred, dtype=np.float64).reshape(-1, int(tta_args.class_num))
    return proba_m


def _predict_stream_noea(
    *,
    loader: DataLoader,
    model: nn.Module,
    tta_args: _TTAArgs,
) -> np.ndarray:
    # No EA, no parameter update: plain streaming inference.
    model.eval()
    device = next(model.parameters()).device
    softmax = nn.Softmax(dim=1)
    y_pred: list[np.ndarray] = []
    it = iter(loader)
    for _i in range(len(loader)):
        data = next(it)
        inputs = data[0]
        x = inputs.reshape(1, 1, inputs.shape[-2], inputs.shape[-1]).to(torch.float32).to(device, non_blocking=True)
        _fea, logits = model(x)
        proba = softmax(logits).detach().cpu().numpy()
        y_pred.append(proba.reshape(-1))
    proba_m = np.asarray(y_pred, dtype=np.float64).reshape(-1, int(tta_args.class_num))
    return proba_m


def _extract_iea_aligned_trials_and_labels(
    *,
    loader: DataLoader,
    tta_args: _TTAArgs,
) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    aligned_trials: list[np.ndarray] = []
    R = 0 if bool(tta_args.align) else None
    data_cum = None

    it = iter(loader)
    for i in range(len(loader)):
        xb, yb = next(it)
        x = xb.reshape(1, 1, xb.shape[-2], xb.shape[-1]).cpu()
        if i == 0:
            data_cum = x.float().cpu()
        else:
            data_cum = torch.cat((data_cum, x.float().cpu()), 0)

        if bool(tta_args.align):
            if i == 0:
                sample = data_cum.reshape(int(tta_args.chn), int(tta_args.time_sample_num))
            else:
                sample = data_cum[i].reshape(int(tta_args.chn), int(tta_args.time_sample_num))
            from utils.alg_utils import EA_online  # type: ignore

            R = EA_online(sample.numpy(), R, i)
            sqrt_ref = fractional_matrix_power(R, -0.5)
            sample_np = np.dot(sqrt_ref, sample.numpy())
        else:
            sample_np = data_cum[i].numpy().reshape(int(tta_args.chn), int(tta_args.time_sample_num))

        aligned_trials.append(np.asarray(sample_np, dtype=np.float32, order="C"))
        y_true.append(int(float(np.asarray(yb).reshape(-1)[0])))

    X_aligned = np.stack(aligned_trials, axis=0)
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    return X_aligned, y_true_arr


def _extract_offline_ea_trials_and_labels(
    *,
    X_mmap: np.ndarray,
    y_all: np.ndarray,
    indices: np.ndarray,
    subject_idx: np.ndarray,
    ea_whiten_by_subject: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    aligned_trials: list[np.ndarray] = []
    labels: list[int] = []
    for idx in indices.tolist():
        s = int(subject_idx[int(idx)])
        W = ea_whiten_by_subject.get(s)
        if W is None:
            raise KeyError(f"Missing EA whitening for subject_idx={s}")
        x = np.asarray(X_mmap[int(idx)], dtype=np.float64)
        x = np.asarray(W @ x, dtype=np.float32, order="C")
        aligned_trials.append(x)
        labels.append(int(y_all[int(idx)]))
    return np.stack(aligned_trials, axis=0), np.asarray(labels, dtype=np.int64)


def _predict_batch(
    *,
    loader: DataLoader,
    model: nn.Module,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    softmax = nn.Softmax(dim=1)
    all_probs: list[np.ndarray] = []
    it = iter(loader)
    with torch.no_grad():
        for _ in range(len(loader)):
            xb, _yb = next(it)
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)[1]
            probs = softmax(logits).detach().cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def _run_shot_im(
    *,
    target_aligned_loader: DataLoader,
    target_eval_loader: DataLoader,
    model: nn.Module,
    args: _TTAArgs,
) -> np.ndarray:
    """SHOT-IM style information maximization (no pseudo labels), output proba matrix.

    We apply this on *pre-aligned* target data (already IEA-aligned) and therefore run with args.align=False.
    """

    _add_deeptransfer_to_syspath()
    from utils.loss import Entropy  # type: ignore

    model.train()
    device = next(model.parameters()).device

    # Freeze classifier (netC); update feature extractor (netF) only.
    for p in model[1].parameters():
        p.requires_grad = False
    for p in model[0].parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model[0].parameters(), lr=float(args.lr))
    softmax = nn.Softmax(dim=1)

    for _epoch in range(5):
        it = iter(target_aligned_loader)
        for _ in range(len(target_aligned_loader)):
            xb, _yb = next(it)
            xb = xb.to(device, non_blocking=True)
            _fea, logits = model(xb)
            probs = softmax(logits)
            ent = torch.mean(Entropy(probs))
            msoft = probs.mean(dim=0)
            gent = torch.sum(msoft * torch.log(msoft + 1e-5))
            loss = ent + gent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate per-trial proba
    return _predict_batch(loader=target_eval_loader, model=model)


def _parse_comma_ints(s: str) -> list[int]:
    vals: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return vals


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def _sml_weights_multiclass(*, preds: np.ndarray, n_classes: int) -> np.ndarray:
    """Estimate per-model ensemble weights via one-vs-rest SML.

    This follows the idea in DeepTransferEEG's `ttime_ensemble.py` (SML_multiclass),
    but uses the principal eigenvector (largest eigenvalue) for numerical robustness.

    Parameters
    ----------
    preds:
      int array of shape (n_models, n_samples_so_far), each entry is predicted class id.
    """

    preds = np.asarray(preds, dtype=np.int64)
    n_models, n_samples = preds.shape
    if n_models <= 0 or n_samples <= 0:
        raise ValueError("SML: empty preds")

    weights_all: list[np.ndarray] = []
    for k in range(int(n_classes)):
        pred_bin = np.where(preds == int(k), 1.0, -1.0)  # (n_models, n_samples)
        mu = np.mean(pred_bin, axis=1, keepdims=True)
        dev = pred_bin - mu
        denom = float(max(1, n_samples - 1))
        Q = (dev @ dev.T) / denom  # symmetric

        # principal eigenvector
        eigvals, eigvecs = np.linalg.eigh(Q)
        v = np.asarray(eigvecs[:, -1], dtype=np.float64)
        if v[0] <= 0:
            v = -v
        v_sum = float(np.sum(v))
        if abs(v_sum) < 1e-12:
            w = np.ones((n_models,), dtype=np.float64) / float(n_models)
        else:
            w = v / v_sum
        weights_all.append(w)

    weights_final = np.sum(np.stack(weights_all, axis=0), axis=0)  # (n_models,)
    return np.asarray(weights_final, dtype=np.float64).reshape(-1)


def _ttime_ensemble_proba(*, probas: np.ndarray) -> np.ndarray:
    """Streaming ensemble aggregation for T-TIME.

    Parameters
    ----------
    probas: array (n_models, n_trials, n_classes)

    Returns
    -------
    proba: array (n_trials, n_classes)
    """

    probas = np.asarray(probas, dtype=np.float64)
    if probas.ndim != 3:
        raise ValueError(f"ttime_ensemble: expected 3D probas, got {probas.shape}")

    n_models, n_trials, n_classes = map(int, probas.shape)
    pred_ids = np.argmax(probas, axis=2).astype(np.int64)  # (n_models, n_trials)

    out = np.zeros((n_trials, n_classes), dtype=np.float64)
    for i in range(n_trials):
        a = i + 1
        if a < n_models:
            out[i] = np.mean(probas[:, i, :], axis=0)
            continue
        w = _sml_weights_multiclass(preds=pred_ids[:, :a], n_classes=n_classes)  # (n_models,)
        scores = np.tensordot(w, probas[:, i, :], axes=(0, 0))  # (n_classes,)
        out[i] = _softmax_np(scores)
    return out


def _extract_features_and_probs(*, loader: DataLoader, model: nn.Module) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    softmax = nn.Softmax(dim=1)

    fea_all: list[np.ndarray] = []
    prob_all: list[np.ndarray] = []
    it = iter(loader)
    with torch.no_grad():
        for _ in range(len(loader)):
            xb, _yb = next(it)
            xb = xb.to(device, non_blocking=True)
            fea, logits = model(xb)
            probs = softmax(logits)
            fea_all.append(fea.detach().cpu().numpy())
            prob_all.append(probs.detach().cpu().numpy())
    return np.concatenate(fea_all, axis=0), np.concatenate(prob_all, axis=0)


def _shot_obtain_pseudo_labels(*, features: np.ndarray, probs: np.ndarray, threshold: int = 0) -> np.ndarray:
    """SHOT pseudo-labeling (Eq.6 in SHOT paper) — clustering with iterative refinement."""

    features = np.asarray(features, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    n, feat_dim = features.shape
    class_num = int(probs.shape[1])

    # cosine distance with an appended bias (match common SHOT implementation)
    fea = np.concatenate([features, np.ones((n, 1), dtype=np.float64)], axis=1)
    fea = fea / (np.linalg.norm(fea, axis=1, keepdims=True) + 1e-12)

    predict = np.argmax(probs, axis=1).astype(np.int64)
    aff = probs.copy()

    for _ in range(2):
        initc = aff.T.dot(fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(class_num, dtype=np.float64)[predict].sum(axis=0)
        labelset = np.where(cls_count > float(threshold))[0]
        if labelset.size <= 0:
            labelset = np.arange(class_num, dtype=np.int64)

        dd = cdist(fea, initc[labelset], metric="cosine")
        pred_label = np.argmin(dd, axis=1)
        predict = labelset[pred_label].astype(np.int64)
        aff = np.eye(class_num, dtype=np.float64)[predict]

    return predict.astype(np.int64)


def _run_shot_paper(
    *,
    X_aligned: np.ndarray,
    model: nn.Module,
    args: _TTAArgs,
    num_workers: int,
    epochs: int,
    beta: float,
) -> np.ndarray:
    """Paper-aligned SHOT (Algorithm 1 + Eq.7), adapted to per-subject target stream."""

    _add_deeptransfer_to_syspath()
    from utils.loss import Entropy  # type: ignore

    device = next(model.parameters()).device
    class_num = int(args.class_num)
    softmax = nn.Softmax(dim=1)

    # Freeze classifier (netC); update feature extractor (netF) only.
    model.train()
    for p in model[1].parameters():
        p.requires_grad = False
    for p in model[0].parameters():
        p.requires_grad = True

    # Match the public SHOT reference code (tim-learn/SHOT):
    # use SGD + momentum + weight decay, with polynomial LR decay.
    lr0 = float(args.lr)
    optimizer = torch.optim.SGD(
        model[0].parameters(),
        lr=lr0,
        momentum=0.9,
        weight_decay=1e-3,
        nesterov=True,
    )

    def _lr_scheduler(iter_num: int, max_iter: int, gamma: float = 10.0, power: float = 0.75) -> None:
        decay = (1.0 + gamma * float(iter_num) / float(max_iter)) ** (-power)
        lr = lr0 * decay
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    n_trials = int(X_aligned.shape[0])
    base_subject_idx = np.zeros((n_trials,), dtype=np.int64)
    indices = np.arange(n_trials, dtype=np.int64)
    y_dummy = np.zeros((n_trials,), dtype=np.int64)

    eval_ds = _TrialDataset(
        X_mmap=X_aligned,
        y=y_dummy,
        indices=indices,
        subject_idx=base_subject_idx,
        ea_whiten_by_subject=None,
        apply_ea=False,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(args.batch_size) * 3,
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
    )

    it_global = 0
    # Use a fixed max_iter like the reference (epochs * len(target_loader)).
    # With drop_last=True below, each epoch has a stable number of steps.
    max_iter = int(max(1, int(epochs) * max(1, n_trials // max(1, int(args.batch_size)))))

    for _epoch in range(int(epochs)):
        features, probs = _extract_features_and_probs(loader=eval_loader, model=model)
        pseudo = _shot_obtain_pseudo_labels(features=features, probs=probs, threshold=0)

        # Ensure train-mode for adaptation steps (the feature extractor contains dropout/BN).
        model.train()

        train_ds = _TrialDataset(
            X_mmap=X_aligned,
            y=pseudo,
            indices=indices,
            subject_idx=base_subject_idx,
            ea_whiten_by_subject=None,
            apply_ea=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            drop_last=True,
            num_workers=int(num_workers),
        )

        it = iter(train_loader)
        for _ in range(len(train_loader)):
            xb, yb = next(it)
            xb = xb.to(device, non_blocking=True)
            yb = torch.as_tensor(yb, dtype=torch.long, device=device)

            it_global += 1
            _lr_scheduler(iter_num=it_global, max_iter=max_iter)

            _fea, logits = model(xb)
            probs_t = softmax(logits)
            ent = torch.mean(Entropy(probs_t))
            msoft = probs_t.mean(dim=0)
            gent = torch.sum(msoft * torch.log(msoft + 1e-5))
            ce = torch.nn.functional.cross_entropy(logits, yb)
            loss = ent + gent + float(beta) * ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return _predict_batch(loader=eval_loader, model=model)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DeepTransferEEG TTA suite with strict LOSO and export predictions.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", type=str, default=DEFAULT_METHODS)
    p.add_argument("--target-subject-idxs", type=str, default="ALL")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    p.add_argument(
        "--baseline-train-ea",
        dest="baseline_train_ea",
        action="store_true",
        default=True,
        help="Apply offline EA whitening in source baseline training (default: on).",
    )
    p.add_argument(
        "--no-baseline-train-ea",
        dest="baseline_train_ea",
        action="store_false",
        help="Disable offline EA whitening in source baseline training.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=0)
    p.add_argument("--skip-merge", action="store_true", help="Skip writing predictions_all_methods.csv (useful for sharded runs).")
    p.add_argument(
        "--ea-cache",
        type=Path,
        default=None,
        help=f"Offline EA cache path (.npy). Default: <data-dir>/{_EA_CACHE_FILENAME}.",
    )
    p.add_argument("--no-ea-cache", action="store_true", help="Disable EA cache (always recompute offline EA).")
    p.add_argument("--recompute-ea-cache", action="store_true", help="Recompute and overwrite EA cache even if it exists.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--test-batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--tent-lr-scale", type=float, default=0.5, help="Scale factor from base lr for KoopTTA-style TENT.")
    p.add_argument("--coral-epochs", type=int, default=10, help="Deep CORAL adaptation epochs from pretrained checkpoint.")
    p.add_argument("--coral-lr-scale", type=float, default=1.0, help="Scale factor from base lr for CORAL adaptation.")
    p.add_argument("--coral-weight-decay", type=float, default=1e-4, help="Weight decay for CORAL adaptation.")
    p.add_argument("--coral-lambda", type=float, default=0.5, help="CORAL loss weight.")
    p.add_argument("--cotta-update-every-x", type=int, default=0, help="CoTTA update cadence; 0 means use --test-batch.")
    p.add_argument("--cotta-memory-size", type=int, default=0, help="CoTTA FIFO memory size; 0 means use --test-batch.")
    p.add_argument("--cotta-adapt-epochs", type=int, default=1, help="Replay epochs per CoTTA update.")
    p.add_argument("--cotta-ema-factor", type=float, default=0.999, help="CoTTA EMA teacher factor.")
    p.add_argument("--cotta-restoration-factor", type=float, default=0.01, help="CoTTA stochastic restoration factor.")
    p.add_argument("--cotta-aug-threshold", type=float, default=0.9, help="CoTTA anchor confidence threshold for augmentation averaging.")
    p.add_argument("--cotta-aug-num", type=int, default=32, help="Number of CoTTA teacher augmentations when augmentation averaging is enabled.")
    p.add_argument(
        "--ttime-ensemble-seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds for T-TIME ensemble base models (trained independently).",
    )
    p.add_argument("--shot-epochs", type=int, default=5, help="SHOT adaptation epochs.")
    p.add_argument("--shot-lr-scale", type=float, default=0.5, help="Scale factor from base lr for SHOT feature adaptation.")
    p.add_argument("--shot-beta", "--shot-cls-par", dest="shot_beta", type=float, default=0.3, help="SHOT classification loss weight.")
    p.add_argument("--shot-ent-par", type=float, default=1.0, help="SHOT information-maximization weight.")
    p.add_argument("--shot-threshold", type=int, default=0, help="SHOT pseudo-label class-count threshold.")
    p.add_argument("--shot-distance", type=str, default="cosine", choices=("cosine", "euclidean"), help="SHOT pseudo-label refinement distance.")
    return p.parse_args()


def main() -> int:
    args_ns = parse_args()
    data_dir = Path(args_ns.data_dir)
    out_dir = Path(args_ns.out_dir)
    _ensure_dir(out_dir)

    class_order = load_class_order(data_dir / "class_order.json")
    export_cfg_path = data_dir / "export_config.json"
    if not export_cfg_path.exists():
        raise RuntimeError(f"Missing export_config.json in {data_dir} (expected from export_moabb_for_deeptransfer).")
    with export_cfg_path.open("r", encoding="utf-8") as f:
        export_cfg = json.load(f)
    sample_rate = float(export_cfg.get("resample"))

    meta = pd.read_csv(data_dir / "meta.csv")
    subject_orig = meta["subject_orig"].astype(int).to_numpy()
    subject_idx = np.load(data_dir / "subject_idx.npy").astype(np.int64)
    y_all = np.load(data_dir / "labels.npy").astype(np.int64)

    if len(subject_idx) != len(y_all) or len(subject_idx) != len(subject_orig):
        raise RuntimeError("subject_idx/labels/meta length mismatch.")

    X_mmap = np.load(data_dir / "X.npy", mmap_mode="r")
    if X_mmap.ndim != 3:
        raise RuntimeError(f"X.npy expected 3D (trials,chn,time), got {X_mmap.shape}")

    n_trials, chn, time_n = map(int, X_mmap.shape)
    n_subjects = int(len(set(subject_idx.tolist())))
    target_subject_idxs = _parse_subject_idxs(str(args_ns.target_subject_idxs), max_idx=n_subjects - 1)

    dataset_tag = _dataset_tag_from_data_dir(data_dir)
    ckpt_root = _repo_root() / "runs_deeptransfer" / dataset_tag
    _ensure_dir(ckpt_root)

    torch_threads = int(args_ns.torch_threads) if int(args_ns.torch_threads) > 0 else None

    # Precompute offline EA whitening matrices per subject once (independent of LOSO fold).
    _add_deeptransfer_to_syspath()
    if bool(args_ns.no_ea_cache):
        print("[ea] cache disabled; computing offline whitening ...")
        ea_whiten_by_subject = _precompute_ea_whitenings(X_mmap=X_mmap, subject_idx=subject_idx)
    else:
        cache_path = Path(args_ns.ea_cache) if args_ns.ea_cache is not None else (data_dir / _EA_CACHE_FILENAME)
        ea_whiten_by_subject = _load_or_compute_ea_whitenings(
            cache_path=cache_path,
            X_mmap=X_mmap,
            subject_idx=subject_idx,
            n_subjects=n_subjects,
            chn=chn,
            recompute=bool(args_ns.recompute_ea_cache),
        )

    # Load DeepTransferEEG method functions.
    import ttime as mod_ttime  # type: ignore
    import sar as mod_sar  # type: ignore
    import pl as mod_pl  # type: ignore
    import t3a as mod_t3a  # type: ignore
    import isfda as mod_isfda  # type: ignore
    import delta as mod_delta  # type: ignore
    bn_mod = _load_bn_adapt_module()

    methods = [m.strip() for m in str(args_ns.methods).split(",") if m.strip()]
    if not methods:
        raise RuntimeError("--methods parsed to empty list")

    if int(len(class_order)) != 2 and "isfda" in methods:
        raise RuntimeError("Method 'isfda' is only implemented for binary classification; remove it for 4-class runs.")

    ttime_ensemble_seeds: list[int] = []
    if "ttime_ensemble" in methods:
        ttime_ensemble_seeds = sorted(set(_parse_comma_ints(str(args_ns.ttime_ensemble_seeds))))
        if not ttime_ensemble_seeds:
            raise RuntimeError("--ttime-ensemble-seeds parsed to empty list")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Build constant args struct for method functions.
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
        data_env=("gpu" if device.type == "cuda" else "local"),
    )

    # Seed
    torch.manual_seed(int(args_ns.seed))
    np.random.seed(int(args_ns.seed))

    for t in target_subject_idxs:
        t = int(t)
        print(f"[loso] target_subject_idx={t} ({t+1}/{n_subjects})")
        tar_mask = subject_idx == t
        src_mask = ~tar_mask
        src_indices = np.where(src_mask)[0]
        tar_indices = np.where(tar_mask)[0]

        if tar_indices.size <= 0:
            print(f"[loso] WARN: target_subject_idx={t} has 0 trials; skip.")
            continue

        subject_orig_id = int(subject_orig[int(tar_indices[0])])
        trials_in_subject = meta.loc[tar_mask, "trial"].astype(int).to_numpy()
        y_true_int = y_all[tar_indices]

        # Train baseline checkpoints if needed (seed0 + optional ensemble seeds).
        ckpt_path = ckpt_root / f"EEGNet_S{t}_seed{int(args_ns.seed)}.ckpt"
        seeds_to_train: list[int] = []
        if (not ckpt_path.exists()) or (not bool(args_ns.resume)):
            seeds_to_train.append(int(args_ns.seed))
        if ttime_ensemble_seeds:
            for s in ttime_ensemble_seeds:
                p = ckpt_root / f"EEGNet_S{t}_seed{int(s)}.ckpt"
                if (not p.exists()) or (not bool(args_ns.resume)):
                    seeds_to_train.append(int(s))
        seeds_to_train = sorted(set(seeds_to_train))

        if not seeds_to_train:
            print(f"[baseline] SKIP ckpt exists: {ckpt_path}")
        else:
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
            for s in seeds_to_train:
                p = ckpt_root / f"EEGNet_S{t}_seed{int(s)}.ckpt"
                print(f"[baseline] TRAIN seed={s} -> {p}")
                torch.manual_seed(int(s))
                np.random.seed(int(s))
                _train_baseline(
                    ckpt_path=p,
                    train_loader=train_loader,
                    args=tta_args,
                    device=device,
                    torch_threads=torch_threads,
                )
            # Restore global seed for deterministic downstream steps.
            torch.manual_seed(int(args_ns.seed))
            np.random.seed(int(args_ns.seed))

        # Target loaders
        target_raw_ds = _TrialDataset(
            X_mmap=X_mmap,
            y=y_all,
            indices=tar_indices,
            subject_idx=subject_idx,
            ea_whiten_by_subject=ea_whiten_by_subject,
            apply_ea=False,
        )
        target_online_loader = DataLoader(
            target_raw_ds,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        target_batch_loader = DataLoader(
            target_raw_ds,
            batch_size=int(tta_args.batch_size),
            shuffle=True,
            drop_last=True,
            num_workers=int(args_ns.num_workers),
        )
        target_eval_loader = DataLoader(
            target_raw_ds,
            batch_size=int(tta_args.batch_size) * 3,
            shuffle=False,
            drop_last=False,
            num_workers=int(args_ns.num_workers),
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

            # Call per-method
            if m == "eegnet_ea":
                proba = _predict_stream_with_iea(loader=target_online_loader, model=model, tta_args=tta_args)
            elif m == "eegnet_noea":
                proba = _predict_stream_noea(loader=target_online_loader, model=model, tta_args=tta_args)
            elif m == "ttime":
                _score, proba = mod_ttime.TTIME(target_online_loader, model, argparse.Namespace(**tta_args.__dict__), balanced=True)
            elif m == "ttime_ensemble":
                if not ttime_ensemble_seeds:
                    raise RuntimeError("ttime_ensemble requested but ttime_ensemble_seeds is empty")
                proba_models: list[np.ndarray] = []
                for s in ttime_ensemble_seeds:
                    s = int(s)
                    torch.manual_seed(s)
                    np.random.seed(s)
                    if s == int(args_ns.seed):
                        model_s = model
                    else:
                        ckpt_s = ckpt_root / f"EEGNet_S{t}_seed{s}.ckpt"
                        if not ckpt_s.exists():
                            raise RuntimeError(f"Missing ckpt for ttime_ensemble seed={s}: {ckpt_s}")
                        model_s = _make_model(args=tta_args)
                        model_s.load_state_dict(torch.load(ckpt_s, map_location=torch.device("cpu")))
                        model_s = model_s.to(device)
                        model_s.eval()

                    _score, proba_s = mod_ttime.TTIME(
                        target_online_loader,
                        model_s,
                        argparse.Namespace(**tta_args.__dict__),
                        balanced=True,
                    )
                    proba_s = np.asarray(proba_s, dtype=np.float64).reshape(-1, int(tta_args.class_num))
                    proba_models.append(proba_s)

                # restore global seed for downstream methods
                torch.manual_seed(int(args_ns.seed))
                np.random.seed(int(args_ns.seed))

                proba = _ttime_ensemble_proba(probas=np.stack(proba_models, axis=0))
            elif m == "tent":
                X_aligned, y_true_arr = _extract_iea_aligned_trials_and_labels(loader=target_online_loader, tta_args=tta_args)
                proba = run_tent_kooptta_aligned(
                    X_aligned=X_aligned,
                    y_true=y_true_arr,
                    model=model,
                    device=device,
                    batch_size=int(args_ns.test_batch),
                    num_workers=int(args_ns.num_workers),
                    lr=float(args_ns.lr) * float(args_ns.tent_lr_scale),
                    steps=int(args_ns.steps),
                )
            elif m == "tent_kooptta_strict":
                X_aligned, y_true_arr = _extract_iea_aligned_trials_and_labels(loader=target_online_loader, tta_args=tta_args)
                proba = run_tent_kooptta_strict(
                    X_aligned=X_aligned,
                    y_true=y_true_arr,
                    model=model,
                    device=device,
                    batch_size=int(args_ns.test_batch),
                    num_workers=int(args_ns.num_workers),
                    lr=float(args_ns.lr) * float(args_ns.tent_lr_scale),
                )
            elif m == "bn_adapt":
                _score, proba = bn_mod.BN_adapt(target_online_loader, model, argparse.Namespace(**tta_args.__dict__), balanced=True)
            elif m == "pl":
                _score, proba = mod_pl.PL(target_online_loader, model, argparse.Namespace(**tta_args.__dict__), balanced=True)
            elif m == "t3a":
                weight = model[1].fc.weight.detach()
                weights = []
                for k in range(int(tta_args.class_num)):
                    w = weight[k] / torch.norm(weight, dim=1)[k]
                    weights.append([w.cpu()])
                _score, proba = mod_t3a.T3A(target_online_loader, model, argparse.Namespace(**tta_args.__dict__), balanced=True, weights=weights)
            elif m == "coral":
                X_source, y_source = _extract_offline_ea_trials_and_labels(
                    X_mmap=X_mmap,
                    y_all=y_all,
                    indices=src_indices,
                    subject_idx=subject_idx,
                    ea_whiten_by_subject=ea_whiten_by_subject,
                )
                X_target, _y_target = _extract_iea_aligned_trials_and_labels(loader=target_online_loader, tta_args=tta_args)
                proba = run_coral_kooptta_aligned(
                    X_source=X_source,
                    y_source=y_source,
                    X_target=X_target,
                    model=model,
                    device=device,
                    batch_size=int(args_ns.batch_size),
                    num_workers=int(args_ns.num_workers),
                    epochs=int(args_ns.coral_epochs),
                    lr=float(args_ns.lr) * float(args_ns.coral_lr_scale),
                    weight_decay=float(args_ns.coral_weight_decay),
                    lambda_coral=float(args_ns.coral_lambda),
                )
            elif m == "cotta":
                X_aligned, y_true_arr = _extract_iea_aligned_trials_and_labels(loader=target_online_loader, tta_args=tta_args)
                cotta_update_every_x = int(args_ns.cotta_update_every_x) if int(args_ns.cotta_update_every_x) > 0 else int(args_ns.test_batch)
                cotta_memory_size = int(args_ns.cotta_memory_size) if int(args_ns.cotta_memory_size) > 0 else int(args_ns.test_batch)
                proba = run_cotta_note_aligned(
                    X_aligned=X_aligned,
                    y_true=y_true_arr,
                    model=model,
                    device=device,
                    batch_size=int(args_ns.batch_size),
                    update_every_x=cotta_update_every_x,
                    memory_size=cotta_memory_size,
                    adapt_epochs=int(args_ns.cotta_adapt_epochs),
                    num_workers=int(args_ns.num_workers),
                    lr=float(args_ns.lr),
                    ema_factor=float(args_ns.cotta_ema_factor),
                    restoration_factor=float(args_ns.cotta_restoration_factor),
                    aug_threshold=float(args_ns.cotta_aug_threshold),
                    aug_num=int(args_ns.cotta_aug_num),
                )
            elif m == "sar":
                _score, proba = mod_sar.SAR(target_online_loader, model, argparse.Namespace(**tta_args.__dict__), balanced=True)
            elif m == "isfda":
                _score, proba = mod_isfda.ISFDA(target_online_loader, model, argparse.Namespace(**tta_args.__dict__), balanced=True)
            elif m == "delta":
                _score, proba = mod_delta.DELTA(target_online_loader, model, argparse.Namespace(**tta_args.__dict__), balanced=True)
            elif m == "shot":
                X_aligned, y_true_arr = _extract_iea_aligned_trials_and_labels(loader=target_online_loader, tta_args=tta_args)
                proba = run_shot_kooptta_aligned(
                    x_aligned=X_aligned,
                    y_true=y_true_arr,
                    model=model,
                    device=device,
                    batch_size=int(args_ns.batch_size),
                    num_workers=int(args_ns.num_workers),
                    epochs=int(args_ns.shot_epochs),
                    base_lr=float(args_ns.lr) * float(args_ns.shot_lr_scale),
                    cls_par=float(args_ns.shot_beta),
                    ent_par=float(args_ns.shot_ent_par),
                    threshold=int(args_ns.shot_threshold),
                    distance=str(args_ns.shot_distance),
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
        print("[done] skip merge: predictions_all_methods.csv not written (use merge_predictions_all_methods.py)")
        return 0

    # Merge all per-method CSVs.
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
