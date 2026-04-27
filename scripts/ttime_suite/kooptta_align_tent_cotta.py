from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _add_deeptransfer_tl_to_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    tl_dir = repo_root / "third_party" / "DeepTransferEEG" / "tl"
    tl_dir_str = str(tl_dir)
    if tl_dir_str not in sys.path:
        sys.path.insert(0, tl_dir_str)


def _validate_trials(
    *,
    X_aligned: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(X_aligned)
    y = np.asarray(y_true)
    if x.ndim != 3:
        raise ValueError(f"X_aligned must be (n_trials, chn, time), got {x.shape}")
    if y.ndim != 1:
        raise ValueError(f"y_true must be 1D, got {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X_aligned/y_true length mismatch: {x.shape[0]} vs {y.shape[0]}")
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)


def _forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, (tuple, list)):
        if len(out) < 2:
            raise RuntimeError("Model forward returned tuple/list with <2 elements; expected (features, logits).")
        return out[1]
    return out


def _configure_bn_only_kooptta_strict(model: nn.Module) -> list[nn.Parameter]:
    """Configure a DeepTransferEEG Sequential EEGNet like KoopTTA's `configure_bn_only`.

    KoopTTA semantics we mirror:
    - `model.train()`
    - classifier/head in eval mode
    - freeze all params first
    - only encoder BatchNorm{1,2}d affine params are trainable
    - force batch stats (`track_running_stats=False`, `running_mean/var=None`)
    """

    model.train()
    if isinstance(model, nn.Sequential) and len(model) >= 2:
        model[-1].eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    encoder = model[0] if isinstance(model, nn.Sequential) and len(model) >= 1 else model
    params: list[nn.Parameter] = []
    for module in encoder.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            if module.weight is not None:
                params.append(module.weight)
            if module.bias is not None:
                params.append(module.bias)
    return params


def run_tent_kooptta_aligned(
    *,
    X_aligned: np.ndarray,
    y_true: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 0,
    lr: float = 5e-4,
    steps: int = 1,
) -> np.ndarray:
    """Run TENT with KoopTTA-style online BN-only adaptation.

    Alignment target:
    - `KoopTTA/kooptta/methods/tta.py::run_tent` (predict current batch, then entropy update).
    - BN-only adaptation primitives are from `third_party/DeepTransferEEG/tl/models/tent.py`.

    EEG-specific deviation:
    - Uses EEGNet tuple output `(features, logits)` and pre-aligned input `X_aligned`.
    """

    x_np, y_np = _validate_trials(X_aligned=X_aligned, y_true=y_true)
    x_tensor = torch.from_numpy(np.expand_dims(x_np, axis=1))
    y_tensor = torch.from_numpy(y_np)
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
    )

    _add_deeptransfer_tl_to_syspath()
    from models.tent import collect_params as tent_collect_params  # type: ignore
    from models.tent import configure_model as tent_configure_model  # type: ignore
    from models.tent import softmax_entropy as tent_softmax_entropy  # type: ignore

    adapt_model = deepcopy(model).to(device)
    adapt_model = tent_configure_model(adapt_model)
    params, _ = tent_collect_params(adapt_model)
    if not params:
        raise RuntimeError("TENT requires BatchNorm affine parameters, but none were found.")
    optimizer = torch.optim.Adam(params, lr=float(lr))

    probs_all: list[np.ndarray] = []
    n_steps = max(1, int(steps))
    adapt_model.train()
    for xb, _yb in loader:
        xb = xb.to(device, non_blocking=True)

        logits = _forward_logits(adapt_model, xb)
        probs = torch.softmax(logits, dim=1)
        probs_all.append(probs.detach().cpu().numpy().astype(np.float64, copy=False))

        for step_idx in range(n_steps):
            loss = tent_softmax_entropy(logits).mean(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step_idx + 1 < n_steps:
                logits = _forward_logits(adapt_model, xb)

    return np.concatenate(probs_all, axis=0)


def run_tent_kooptta_strict(
    *,
    X_aligned: np.ndarray,
    y_true: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 0,
    lr: float = 5e-4,
) -> np.ndarray:
    """Run a stricter KoopTTA-style TENT control.

    Differences vs `run_tent_kooptta_aligned`:
    - uses a local `configure_bn_only` that mirrors KoopTTA semantics directly
    - updates only encoder BN affine params
    - forces classifier/head eval mode
    - fixed to one update step per batch (matching KoopTTA `run_tent`)
    """

    x_np, y_np = _validate_trials(X_aligned=X_aligned, y_true=y_true)
    x_tensor = torch.from_numpy(np.expand_dims(x_np, axis=1))
    y_tensor = torch.from_numpy(y_np)
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
    )

    adapt_model = deepcopy(model).to(device)
    params = _configure_bn_only_kooptta_strict(adapt_model)
    if not params:
        raise RuntimeError("Strict KoopTTA TENT requires encoder BatchNorm affine parameters, but none were found.")
    optimizer = torch.optim.Adam(params, lr=float(lr))

    probs_all: list[np.ndarray] = []
    adapt_model.train()
    for xb, _yb in loader:
        xb = xb.to(device, non_blocking=True)

        logits = _forward_logits(adapt_model, xb)
        probs = torch.softmax(logits, dim=1)
        probs_all.append(probs.detach().cpu().numpy().astype(np.float64, copy=False))

        optimizer.zero_grad()
        loss = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
        loss.backward()
        optimizer.step()

    return np.concatenate(probs_all, axis=0)


def run_cotta_note_aligned(
    *,
    X_aligned: np.ndarray,
    y_true: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 32,
    update_every_x: int = 8,
    memory_size: int = 8,
    adapt_epochs: int = 1,
    num_workers: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    ema_factor: float = 0.999,
    restoration_factor: float = 0.01,
    aug_threshold: float = 0.9,
    aug_num: int = 32,
) -> np.ndarray:
    """Run CoTTA with NOTE-style online memory/update schedule.

    Alignment target:
    - Memory/update cadence follows `baseline/NOTE-main/learner/cotta.py`
      (`update_every_x`, replay from memory, online adaptation).
    - CoTTA adaptation core (EMA teacher + stochastic restore + EEG augmentations)
      reuses `third_party/DeepTransferEEG/tl/models/cotta.py::CoTTA`.

    EEG-specific deviation:
    - Uses FIFO memory over pre-aligned EEG trials and EEGNet tuple output `(features, logits)`.
    """

    x_np, y_np = _validate_trials(X_aligned=X_aligned, y_true=y_true)
    del y_np  # kept only for input consistency checks

    x_tensor = torch.from_numpy(np.expand_dims(x_np, axis=1))
    n_trials = int(x_tensor.shape[0])

    _add_deeptransfer_tl_to_syspath()
    from models.cotta import CoTTA as CottaAdapter  # type: ignore
    from models.cotta import configure_model as cotta_configure_model  # type: ignore

    adapt_model = deepcopy(model).to(device)
    adapt_model = cotta_configure_model(adapt_model)
    trainable = [p for p in adapt_model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("CoTTA requires trainable parameters, but none were found after configuration.")
    optimizer = torch.optim.Adam(trainable, lr=float(lr), weight_decay=float(weight_decay))

    cotta_adapter = CottaAdapter(
        adapt_model,
        optimizer,
        steps=1,
        mt_alpha=float(ema_factor),
        rst_m=float(restoration_factor),
        ap=float(aug_threshold),
        aug_num=int(aug_num),
    )

    softmax = nn.Softmax(dim=1)
    memory: deque[torch.Tensor] = deque(maxlen=max(1, int(memory_size)))
    probs_all: list[np.ndarray] = []

    eff_update_every_x = max(1, int(update_every_x))
    eff_batch_size = max(1, int(batch_size))
    eff_epochs = max(1, int(adapt_epochs))
    eff_workers = max(0, int(num_workers))

    for idx in range(n_trials):
        sample_x = x_tensor[idx : idx + 1]
        sample_dev = sample_x.to(device, non_blocking=True)

        adapt_model.eval()
        with torch.no_grad():
            logits = _forward_logits(adapt_model, sample_dev)
            probs = softmax(logits)
        probs_all.append(probs.detach().cpu().numpy().astype(np.float64, copy=False))

        memory.append(sample_x)
        seen = idx + 1
        should_update = (seen % eff_update_every_x == 0) or (seen == n_trials)
        if not should_update:
            continue

        replay_x = torch.cat(list(memory), dim=0)
        replay_loader = DataLoader(
            TensorDataset(replay_x),
            batch_size=eff_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=eff_workers,
        )

        adapt_model.train()
        for _ in range(eff_epochs):
            for (xb,) in replay_loader:
                xb = xb.to(device, non_blocking=True)
                cotta_adapter(xb)

    return np.concatenate(probs_all, axis=0)


__all__ = [
    "run_tent_kooptta_aligned",
    "run_tent_kooptta_strict",
    "run_cotta_note_aligned",
]
