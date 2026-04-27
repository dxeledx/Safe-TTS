from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _validate_source_target(
    *,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_src = np.asarray(X_source, dtype=np.float32)
    y_src = np.asarray(y_source, dtype=np.int64)
    x_tgt = np.asarray(X_target, dtype=np.float32)
    if x_src.ndim != 3 or x_tgt.ndim != 3:
        raise ValueError(f"CORAL expects 3D trial tensors, got source={x_src.shape}, target={x_tgt.shape}")
    if y_src.ndim != 1:
        raise ValueError(f"y_source must be 1D, got {y_src.shape}")
    if x_src.shape[0] != y_src.shape[0]:
        raise ValueError(f"X_source/y_source length mismatch: {x_src.shape[0]} vs {y_src.shape[0]}")
    if x_src.shape[1:] != x_tgt.shape[1:]:
        raise ValueError(f"source/target trial shape mismatch: {x_src.shape[1:]} vs {x_tgt.shape[1:]}")
    return x_src, y_src, x_tgt


def _forward_features_logits(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(x)
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise RuntimeError("CORAL expects model(x) -> (features, logits)")
    return out[0], out[1]


def _torch_coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("CORAL loss expects 2D feature tensors")
    d = source_features.shape[1]
    src = source_features - source_features.mean(dim=0, keepdim=True)
    tgt = target_features - target_features.mean(dim=0, keepdim=True)
    cov_src = (src.T @ src) / max(src.shape[0] - 1, 1)
    cov_tgt = (tgt.T @ tgt) / max(tgt.shape[0] - 1, 1)
    return ((cov_src - cov_tgt) ** 2).sum() / (4.0 * d * d)


def run_coral_kooptta_aligned(
    *,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_coral: float = 0.5,
) -> np.ndarray:
    """Run KoopTTA-style Deep CORAL from a pretrained EEGNet checkpoint.

    Alignment target:
    - `KoopTTA/kooptta/methods/static.py::run_coral`

    Important scope note:
    - This is a source+target unlabeled adaptation baseline, not a pure online TTA method.
    - We do not use target labels in training.
    - Unlike KoopTTA's full pipeline, this adapter has no target-val split for best-epoch selection;
      it returns the final-epoch target probabilities.
    """

    x_src, y_src, x_tgt = _validate_source_target(X_source=X_source, y_source=y_source, X_target=X_target)
    src_ds = TensorDataset(
        torch.from_numpy(np.expand_dims(x_src, axis=1)),
        torch.from_numpy(y_src),
    )
    tgt_x = torch.from_numpy(np.expand_dims(x_tgt, axis=1))
    tgt_idx = torch.arange(x_tgt.shape[0], dtype=torch.long)
    tgt_train_ds = TensorDataset(tgt_x, tgt_idx)
    tgt_eval_ds = TensorDataset(tgt_x, tgt_idx)

    source_loader = DataLoader(
        src_ds,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
    )
    target_loader = DataLoader(
        tgt_train_ds,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
    )
    eval_loader = DataLoader(
        tgt_eval_ds,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
    )

    adapt_model = deepcopy(model).to(device)
    adapt_model.train()
    optimizer = torch.optim.Adam(
        adapt_model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    criterion = torch.nn.CrossEntropyLoss()

    for _epoch in range(max(1, int(epochs))):
        adapt_model.train()
        target_iter = iter(target_loader)
        for xb_src, yb_src in source_loader:
            try:
                xb_tgt, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                xb_tgt, _ = next(target_iter)

            xb_src = xb_src.to(device, non_blocking=True)
            yb_src = yb_src.to(device, non_blocking=True)
            xb_tgt = xb_tgt.to(device, non_blocking=True)

            optimizer.zero_grad()
            feat_src, logits_src = _forward_features_logits(adapt_model, xb_src)
            feat_tgt, _logits_tgt = _forward_features_logits(adapt_model, xb_tgt)
            cls_loss = criterion(logits_src, yb_src)
            coral_loss = _torch_coral_loss(feat_src, feat_tgt)
            loss = cls_loss + float(lambda_coral) * coral_loss
            loss.backward()
            optimizer.step()

    n_trials = int(x_tgt.shape[0])
    out = None
    adapt_model.eval()
    with torch.no_grad():
        for xb_tgt, idx_tgt in eval_loader:
            xb_tgt = xb_tgt.to(device, non_blocking=True)
            _feat, logits_tgt = _forward_features_logits(adapt_model, xb_tgt)
            probs = torch.softmax(logits_tgt, dim=1).detach().cpu().numpy().astype(np.float64)
            if out is None:
                out = np.zeros((n_trials, probs.shape[1]), dtype=np.float64)
            out[idx_tgt.detach().cpu().numpy().astype(np.int64)] = probs
    if out is None:
        raise RuntimeError("CORAL evaluation produced no outputs")
    return out


__all__ = ["run_coral_kooptta_aligned"]
