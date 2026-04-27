from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


class _AlignedTrialDataset(Dataset):
    def __init__(self, x_aligned: np.ndarray, y_true: np.ndarray) -> None:
        x_np = np.asarray(x_aligned, dtype=np.float32)
        y_np = np.asarray(y_true, dtype=np.int64)
        if x_np.ndim != 3:
            raise ValueError(f"x_aligned must be 3D (n_trials, chn, time), got {x_np.shape}")
        if y_np.ndim != 1:
            raise ValueError(f"y_true must be 1D (n_trials,), got {y_np.shape}")
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError(f"x_aligned and y_true length mismatch: {x_np.shape[0]} vs {y_np.shape[0]}")
        self._x = x_np
        self._y = y_np

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self._x[int(idx)]).unsqueeze(0)
        index = torch.tensor(int(idx), dtype=torch.long)
        return x, index


def _softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1)


def _marginal_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    mean_probs = probs.mean(dim=0)
    return -(mean_probs * torch.log(mean_probs.clamp_min(1e-6))).sum()


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-8, None)


def _pairwise_distance(features: np.ndarray, centers: np.ndarray, distance: str) -> np.ndarray:
    if distance == "cosine":
        feat_norm = _normalize_rows(features)
        center_norm = _normalize_rows(centers)
        return 1.0 - feat_norm @ center_norm.T
    if distance == "euclidean":
        diff = features[:, None, :] - centers[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))
    raise ValueError(f"Unsupported distance: {distance}")


def _collect_features_probs(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    indices_all: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, index = batch
            x = x.to(device, non_blocking=True)
            index = index.detach().cpu().numpy().astype(np.int64)
            features, logits = model(x)
            features_all.append(F.normalize(features, dim=1).detach().cpu().numpy())
            probs_all.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            indices_all.append(index)

    features_np = np.concatenate(features_all, axis=0)
    probs_np = np.concatenate(probs_all, axis=0)
    indices_np = np.concatenate(indices_all, axis=0)
    return features_np, probs_np, indices_np


def _obtain_shot_pseudo_labels(
    *,
    features: np.ndarray,
    probs: np.ndarray,
    threshold: int,
    distance: str,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    n_classes = int(probs.shape[1])
    predict = np.argmax(probs, axis=1).astype(np.int64)

    feature_bank = features
    if distance == "cosine":
        feature_bank = np.concatenate([feature_bank, np.ones((feature_bank.shape[0], 1), dtype=np.float64)], axis=1)
        feature_bank = _normalize_rows(feature_bank)

    aff = probs.copy()
    for _ in range(2):
        centroids = aff.T @ feature_bank
        centroids = centroids / np.clip(aff.sum(axis=0)[:, None], 1e-8, None)
        class_count = np.eye(n_classes, dtype=np.float64)[predict].sum(axis=0)
        labelset = np.where(class_count > float(threshold))[0]
        if labelset.size == 0:
            labelset = np.arange(n_classes)
        dist_mat = _pairwise_distance(feature_bank, centroids[labelset], distance=distance)
        pred_idx = np.argmin(dist_mat, axis=1)
        predict = labelset[pred_idx].astype(np.int64)
        aff = np.eye(n_classes, dtype=np.float64)[predict]
    return predict


def _predict_probs(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_trials: int,
    n_classes: int,
) -> np.ndarray:
    out = np.zeros((int(n_trials), int(n_classes)), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, indices = batch
            x = x.to(device, non_blocking=True)
            indices = indices.detach().cpu().numpy().astype(np.int64)
            _features, logits = model(x)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)
            out[indices] = probs
    return out


def run_shot_kooptta_aligned(
    *,
    x_aligned: np.ndarray,
    y_true: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    epochs: int,
    base_lr: float,
    cls_par: float,
    ent_par: float,
    threshold: int,
    distance: str,
) -> np.ndarray:
    """Run a KoopTTA/public-source-aligned SHOT adapter and return per-trial probabilities.

    Alignment target:
    - KoopTTA `kooptta/methods/tta.py::run_shot`
    - SHOT reference `baseline/SHOT-master/object/image_target.py`

    EEG-specific deviation:
    - Uses EEG trial tensors shaped `(n_trials, chn, time)` and EEGNet tuple output `(features, logits)`.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if int(num_workers) < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if int(epochs) <= 0:
        raise ValueError(f"epochs must be > 0, got {epochs}")
    if float(base_lr) <= 0:
        raise ValueError(f"base_lr must be > 0, got {base_lr}")
    if str(distance) not in {"cosine", "euclidean"}:
        raise ValueError(f"distance must be 'cosine' or 'euclidean', got {distance}")

    dataset = _AlignedTrialDataset(x_aligned=x_aligned, y_true=y_true)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
    )

    model = model.to(device)
    model.train()
    if not isinstance(model, nn.Sequential) or len(model) < 2:
        raise TypeError("model is expected to be nn.Sequential(netF, netC) for SHOT adaptation")

    for parameter in model[1].parameters():
        parameter.requires_grad = False
    for parameter in model[0].parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.SGD(
        [parameter for parameter in model[0].parameters() if parameter.requires_grad],
        lr=float(base_lr),
        momentum=0.9,
        weight_decay=1e-3,
        nesterov=True,
    )

    n_trials = int(np.asarray(x_aligned).shape[0])
    n_classes = -1

    for _ in range(int(epochs)):
        features_np, probs_np, indices_np = _collect_features_probs(model=model, loader=loader, device=device)
        n_classes = int(probs_np.shape[1])
        refined = _obtain_shot_pseudo_labels(
            features=features_np,
            probs=probs_np,
            threshold=int(threshold),
            distance=str(distance),
        )
        pseudo_labels: Dict[int, int] = {int(i): int(y) for i, y in zip(indices_np.tolist(), refined.tolist())}

        model.train()
        for batch in loader:
            x, indices = batch
            x = x.to(device, non_blocking=True)
            indices = indices.detach().cpu().numpy().astype(np.int64)
            y_pseudo = torch.as_tensor([pseudo_labels[int(i)] for i in indices], dtype=torch.long, device=device)

            optimizer.zero_grad()
            _features, logits = model(x)
            entropy = _softmax_entropy(logits).mean()
            im_loss = float(ent_par) * (entropy - _marginal_entropy(logits))
            loss = float(cls_par) * F.cross_entropy(logits, y_pseudo) + im_loss
            loss.backward()
            optimizer.step()

    if n_classes <= 0:
        with torch.no_grad():
            sample = next(iter(loader))
            x0 = sample[0].to(device, non_blocking=True)
            _f0, logit0 = model(x0)
            n_classes = int(logit0.shape[1])

    return _predict_probs(
        model=model,
        loader=loader,
        device=device,
        n_trials=n_trials,
        n_classes=n_classes,
    )


__all__ = ["run_shot_kooptta_aligned"]
