from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


class _IndexedTrialDataset(Dataset):
    def __init__(self, *, X: np.ndarray, y: np.ndarray) -> None:
        x_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int64)
        if x_np.ndim != 3:
            raise ValueError(f"Expected X with shape (n_trials, chn, time), got {x_np.shape}")
        if y_np.ndim != 1:
            raise ValueError(f"Expected y with shape (n_trials,), got {y_np.shape}")
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError(f"X/y length mismatch: {x_np.shape[0]} vs {y_np.shape[0]}")
        self._x = x_np
        self._y = y_np

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self._x[int(idx)]).unsqueeze(0)
        y = torch.tensor(int(self._y[int(idx)]), dtype=torch.long)
        index = torch.tensor(int(idx), dtype=torch.long)
        return x, y, index


def _make_loader(
    *,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        _IndexedTrialDataset(X=X, y=y),
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=max(0, int(num_workers)),
    )


class _StreamingTrialNormalizerReference:
    def __init__(self, n_channels: int, eps: float = 1e-6) -> None:
        self.n_channels = int(n_channels)
        self.eps = float(eps)
        self.count = 0
        self.sum = np.zeros((1, self.n_channels, 1), dtype=np.float64)
        self.sum_sq = np.zeros((1, self.n_channels, 1), dtype=np.float64)

    def update(self, signals: np.ndarray) -> None:
        if signals.ndim != 3:
            raise ValueError(f"Expected signals with shape (batch, channels, samples), got {signals.shape}")
        if signals.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {signals.shape[1]}")

        samples = int(signals.shape[0]) * int(signals.shape[2])
        self.count += samples
        self.sum += signals.sum(axis=(0, 2), keepdims=True, dtype=np.float64)
        self.sum_sq += np.square(signals, dtype=np.float64).sum(axis=(0, 2), keepdims=True, dtype=np.float64)

    def transform(self, signals: np.ndarray) -> np.ndarray:
        if self.count == 0:
            return np.asarray(signals, dtype=np.float32, order="C").copy()
        mean = self.sum / self.count
        var = self.sum_sq / self.count - np.square(mean)
        std = np.sqrt(np.clip(var, self.eps, None))
        return ((signals - mean) / std).astype(np.float32)


def streaming_normalize_trials_reference(
    X: np.ndarray,
    *,
    session_ids: np.ndarray | None = None,
    run_ids: np.ndarray | None = None,
    trial_ids: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(X, dtype=np.float32, order="C")
    if x.ndim != 3:
        raise ValueError(f"Expected X with shape (n_trials, channels, time), got {x.shape}")

    if session_ids is not None and run_ids is not None and trial_ids is not None:
        order = np.lexsort(
            (
                np.asarray(trial_ids, dtype=np.int64),
                np.asarray(run_ids, dtype=str),
                np.asarray(session_ids, dtype=str),
            )
        )
        ordered = x[order]
    else:
        order = None
        ordered = x

    normalizer = _StreamingTrialNormalizerReference(n_channels=int(ordered.shape[1]))
    normalized_parts: list[np.ndarray] = []
    for idx in range(int(ordered.shape[0])):
        trial = ordered[idx : idx + 1]
        normalized_parts.append(normalizer.transform(trial))
        normalizer.update(trial)

    normalized = np.concatenate(normalized_parts, axis=0).astype(np.float32)
    if order is None:
        return normalized

    inverse = np.empty_like(order)
    inverse[order] = np.arange(int(order.shape[0]))
    return normalized[inverse]


def _forward_features_logits(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(x)
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise RuntimeError("Expected model(x) -> (features, logits)")
    return out[0], out[1]


def _infer_n_classes_from_model(model: nn.Module, fallback: int) -> int:
    if isinstance(model, nn.Sequential) and len(model) >= 2 and hasattr(model[1], "fc"):
        return int(model[1].fc.out_features)
    return int(fallback)


def _disable_dropout(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, _DROPOUT_TYPES):
            child.eval()


def _enable_tta_encoder_mode_reference(model: nn.Module) -> nn.Module:
    model.train()
    if isinstance(model, nn.Sequential) and len(model) >= 2:
        model[-1].eval()
        encoder = model[0]
    else:
        encoder = model
    _disable_dropout(encoder)
    return encoder


def _configure_bn_only_reference(
    model: nn.Module,
    *,
    use_batch_stats: bool,
    momentum: float | None,
) -> list[nn.Parameter]:
    encoder = _enable_tta_encoder_mode_reference(model)
    for parameter in model.parameters():
        parameter.requires_grad = False

    params: list[nn.Parameter] = []
    for module in encoder.modules():
        if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            continue
        module.train()
        module.requires_grad_(True)
        if bool(use_batch_stats):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        else:
            module.track_running_stats = True
            if momentum is not None:
                module.momentum = float(momentum)
        if module.weight is not None:
            params.append(module.weight)
        if module.bias is not None:
            params.append(module.bias)
    return params


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
        for xb, _yb, indices in loader:
            xb = xb.to(device, non_blocking=True)
            _features, logits = _forward_features_logits(model, xb)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)
            out[indices.detach().cpu().numpy().astype(np.int64)] = probs
    return out


def _softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1)


def _marginal_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1).mean(dim=0)
    probs = probs.clamp_min(1e-6)
    return -(probs * probs.log()).sum()


def _trials_to_device(
    x: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    arr = np.asarray(x, dtype=np.float32, order="C")
    if arr.ndim == 2:
        arr = arr[None, None, :, :]
    elif arr.ndim == 3:
        arr = arr[:, None, :, :]
    else:
        raise ValueError(f"Expected trial array with shape (chn, time) or (batch, chn, time), got {arr.shape}")
    return torch.from_numpy(arr).to(device, non_blocking=True)


def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp_min(1e-6)
    return -(probs * probs.log()).sum(dim=1)


class _SAMReference(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho: float = 0.05, adaptive: bool = False, **kwargs):
        if float(rho) < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        defaults = dict(rho=float(rho), adaptive=bool(adaptive), **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                coeff = torch.pow(p, 2) if bool(group["adaptive"]) else 1.0
                p.add_(coeff * p.grad * scale.to(p))
        if bool(zero_grad):
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if bool(zero_grad):
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack(
                [
                    (((torch.abs(p) if bool(group["adaptive"]) else 1.0) * p.grad).norm(p=2)).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def _configure_sar_bn_affine_reference(model: nn.Module) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    params: list[nn.Parameter] = []
    for module in model.modules():
        if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            continue
        for name, parameter in module.named_parameters():
            if name in {"weight", "bias"}:
                parameter.requires_grad = True
                params.append(parameter)
    model.train()
    return params


def _copy_model_and_optimizer_reference(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, torch.Tensor], dict, nn.Module, nn.Module]:
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for parameter in ema_model.parameters():
        parameter.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def _load_model_and_optimizer_reference(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict,
) -> None:
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def _update_ema_variables_reference(ema_model: nn.Module, model: nn.Module, alpha_teacher: float) -> nn.Module:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = float(alpha_teacher) * ema_param[:].data[:] + (1.0 - float(alpha_teacher)) * param[:].data[:]
    return ema_model


class _CoTTAReference(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        steps: int = 1,
        episodic: bool = False,
        mt_alpha: float = 0.999,
        rst_m: float = 0.01,
        ap: float = 0.9,
        aug_num: int = 32,
        noise_std: float = 0.01,
        max_time_shift: int = 8,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        channel_dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = max(1, int(steps))
        self.episodic = bool(episodic)
        self.mt = float(mt_alpha)
        self.rst = float(rst_m)
        self.ap = float(ap)
        self.aug_num = int(aug_num)
        self.noise_std = float(noise_std)
        self.max_time_shift = int(max_time_shift)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.channel_dropout_p = float(channel_dropout_p)
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = _copy_model_and_optimizer_reference(
            self.model,
            self.optimizer,
        )

    def _augment_eeg(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        if self.scale_max > self.scale_min:
            scale = torch.empty((y.shape[0], 1, 1, 1), device=y.device).uniform_(self.scale_min, self.scale_max)
            y = y * scale
        if self.max_time_shift > 0:
            shift = int(torch.randint(-self.max_time_shift, self.max_time_shift + 1, (1,), device="cpu").item())
            if shift != 0:
                y = torch.roll(y, shifts=shift, dims=-1)
        if self.noise_std > 0:
            y = y + self.noise_std * torch.randn_like(y)
        if self.channel_dropout_p > 0:
            keep = (torch.rand((y.shape[0], 1, y.shape[2], 1), device=y.device) > self.channel_dropout_p).to(y.dtype)
            y = y * keep
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if bool(self.episodic):
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        if outputs is None:
            raise RuntimeError("CoTTA reference produced no outputs")
        return outputs

    def reset(self) -> None:
        _load_model_and_optimizer_reference(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = _copy_model_and_optimizer_reference(
            self.model,
            self.optimizer,
        )

    @torch.enable_grad()
    def forward_and_adapt(self, x: torch.Tensor) -> torch.Tensor:
        outputs = _forward_features_logits(self.model, x)[1]
        with torch.no_grad():
            anchor_prob = torch.softmax(_forward_features_logits(self.model_anchor, x)[1], dim=1).max(dim=1)[0]
            standard_ema_logits = _forward_features_logits(self.model_ema, x)[1]
            if self.aug_num > 0 and float(anchor_prob.mean().item()) < self.ap:
                logits_ema = []
                for _ in range(self.aug_num):
                    logits_ema.append(_forward_features_logits(self.model_ema, self._augment_eeg(x))[1].detach())
                outputs_ema = torch.stack(logits_ema, dim=0).mean(dim=0)
            else:
                outputs_ema = standard_ema_logits

        loss = -(outputs_ema.softmax(1) * outputs.log_softmax(1)).sum(1).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model_ema = _update_ema_variables_reference(self.model_ema, self.model, alpha_teacher=self.mt)

        for name, module in self.model.named_modules():
            for param_name, parameter in module.named_parameters(recurse=False):
                if param_name not in {"weight", "bias"} or not parameter.requires_grad:
                    continue
                mask = (torch.rand(parameter.shape, device=parameter.device) < self.rst).to(parameter.dtype)
                state_key = f"{name}.{param_name}" if name else param_name
                with torch.no_grad():
                    parameter.data = self.model_state[state_key].to(parameter.device) * mask + parameter * (1.0 - mask)
        return outputs_ema


def run_tent_kooptta_reference(
    *,
    X_target: np.ndarray,
    y_true: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 0,
    lr: float = 5e-4,
) -> np.ndarray:
    adapt_model = deepcopy(model).to(device)
    params = _configure_bn_only_reference(adapt_model, use_batch_stats=True, momentum=None)
    if not params:
        raise RuntimeError("Tent reference requires encoder BatchNorm parameters, but none were found.")
    optimizer = torch.optim.Adam(params, lr=float(lr))

    loader = _make_loader(X=X_target, y=y_true, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    n_trials = int(np.asarray(X_target).shape[0])
    n_classes = -1
    probs_all: list[np.ndarray] = []

    for xb, _yb, _indices in loader:
        xb = xb.to(device, non_blocking=True)
        _features, logits = _forward_features_logits(adapt_model, xb)
        probs = torch.softmax(logits, dim=1)
        n_classes = int(probs.shape[1])
        probs_all.append(probs.detach().cpu().numpy().astype(np.float64))

        optimizer.zero_grad()
        loss = _softmax_entropy(logits).mean()
        loss.backward()
        optimizer.step()

    out = np.concatenate(probs_all, axis=0)
    if out.shape != (n_trials, n_classes):
        raise RuntimeError(f"Tent reference output shape mismatch: {out.shape}")
    return out


def run_adabn_kooptta_reference(
    *,
    X_target: np.ndarray,
    y_true: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 0,
    momentum: float | None = None,
) -> np.ndarray:
    adapt_model = deepcopy(model).to(device)
    _configure_bn_only_reference(adapt_model, use_batch_stats=False, momentum=momentum)

    loader = _make_loader(X=X_target, y=y_true, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    _enable_tta_encoder_mode_reference(adapt_model)
    with torch.no_grad():
        for xb, _yb, _indices in loader:
            xb = xb.to(device, non_blocking=True)
            _forward_features_logits(adapt_model, xb)

    sample = next(iter(loader))
    _xb, _yb, _idx = sample
    with torch.no_grad():
        _features0, logits0 = _forward_features_logits(adapt_model, _xb.to(device, non_blocking=True))
    n_classes = int(logits0.shape[1])
    return _predict_probs(
        model=adapt_model,
        loader=loader,
        device=device,
        n_trials=int(np.asarray(X_target).shape[0]),
        n_classes=n_classes,
    )


class _NoteFIFO:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self.data: deque[dict[str, torch.Tensor]] = deque(maxlen=self.capacity)

    def add_instance(self, instance: dict[str, torch.Tensor], pseudo_label: int) -> None:
        del pseudo_label
        self.data.append(instance)

    def get_memory(self) -> list[dict[str, torch.Tensor]]:
        return list(self.data)


class _NoteReservoir:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self.data: list[dict[str, torch.Tensor]] = []
        self.counter = 0

    def add_instance(self, instance: dict[str, torch.Tensor], pseudo_label: int) -> None:
        del pseudo_label
        self.counter += 1
        if len(self.data) < self.capacity:
            self.data.append(instance)
            return
        if random.random() <= self.capacity / max(self.counter, 1):
            self.data[random.randrange(len(self.data))] = instance

    def get_memory(self) -> list[dict[str, torch.Tensor]]:
        return list(self.data)


class _NotePBRS:
    def __init__(self, capacity: int, n_classes: int) -> None:
        self.capacity = max(1, int(capacity))
        self.n_classes = int(n_classes)
        self.data: list[list[dict[str, torch.Tensor]]] = [[] for _ in range(self.n_classes)]
        self.counter = [0] * self.n_classes

    def _occupancy_per_class(self) -> list[int]:
        return [len(bucket) for bucket in self.data]

    def _total(self) -> int:
        return sum(self._occupancy_per_class())

    def add_instance(self, instance: dict[str, torch.Tensor], pseudo_label: int) -> None:
        cls = int(pseudo_label)
        self.counter[cls] += 1
        if self._total() < self.capacity:
            self.data[cls].append(instance)
            return

        occupancy = self._occupancy_per_class()
        largest = [idx for idx, size in enumerate(occupancy) if size == max(occupancy)]
        if cls not in largest:
            donor = random.choice(largest)
            self.data[donor].pop(random.randrange(len(self.data[donor])))
            self.data[cls].append(instance)
            return

        current_cls_size = occupancy[cls]
        seen_cls = self.counter[cls]
        if random.random() <= current_cls_size / max(seen_cls, 1):
            self.data[cls].pop(random.randrange(len(self.data[cls])))
            self.data[cls].append(instance)

    def get_memory(self) -> list[dict[str, torch.Tensor]]:
        memory: list[dict[str, torch.Tensor]] = []
        for bucket in self.data:
            memory.extend(bucket)
        return memory


def _build_note_memory(*, memory_type: str, capacity: int, n_classes: int):
    key = str(memory_type).strip().lower()
    if key == "fifo":
        return _NoteFIFO(capacity)
    if key == "reservoir":
        return _NoteReservoir(capacity)
    if key == "pbrs":
        return _NotePBRS(capacity, n_classes=n_classes)
    raise ValueError(f"Unsupported NOTE memory type: {memory_type}")


def _clone_batch_item(x: torch.Tensor, y: torch.Tensor, index: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "x": x.detach().clone(),
        "y": y.detach().clone(),
        "index": index.detach().clone(),
    }


def _stack_note_items(memory_items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    x = torch.cat([item["x"] for item in memory_items], dim=0)
    y = torch.cat([item["y"] for item in memory_items], dim=0)
    index = torch.cat([item["index"] for item in memory_items], dim=0)
    return {"x": x, "y": y, "index": index}


def _note_entropy_loss(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    probs = F.softmax(logits / float(temperature), dim=1)
    return -(probs * probs.clamp_min(1e-6).log()).mean()


def _run_note_updates(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    memory_items: list[dict[str, torch.Tensor]],
    replay_batch_size: int,
    adapt_epochs: int,
    temperature: float,
    device: torch.device,
) -> None:
    if not memory_items:
        return
    replay = _stack_note_items(memory_items)
    dataset = TensorDataset(replay["x"])
    loader = DataLoader(dataset, batch_size=max(1, int(replay_batch_size)), shuffle=True, drop_last=False)
    for _ in range(max(1, int(adapt_epochs))):
        _enable_tta_encoder_mode_reference(model)
        for (x_batch,) in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            _features, logits = _forward_features_logits(model, x_batch)
            loss = _note_entropy_loss(logits, float(temperature))
            loss.backward()
            optimizer.step()


def run_note_kooptta_reference(
    *,
    X_target: np.ndarray,
    y_true: np.ndarray,
    model: nn.Module,
    device: torch.device,
    replay_batch_size: int = 32,
    memory_size: int = 8,
    update_every_x: int = 8,
    num_workers: int = 0,
    lr: float = 5e-4,
    memory_type: str = "fifo",
    use_learned_stats: bool = False,
    bn_momentum: float = 0.1,
    temperature: float = 1.0,
    adapt_epochs: int = 1,
) -> np.ndarray:
    adapt_model = deepcopy(model).to(device)
    params = _configure_bn_only_reference(
        adapt_model,
        use_batch_stats=not bool(use_learned_stats),
        momentum=float(bn_momentum) if bool(use_learned_stats) else None,
    )
    if not params:
        raise RuntimeError("NOTE reference requires encoder BatchNorm parameters, but none were found.")
    optimizer = torch.optim.Adam(params, lr=float(lr))

    loader = _make_loader(X=X_target, y=y_true, batch_size=1, shuffle=False, num_workers=num_workers)
    n_trials = int(np.asarray(X_target).shape[0])
    n_classes = -1
    out: np.ndarray | None = None
    memory = _build_note_memory(
        memory_type=memory_type,
        capacity=memory_size,
        n_classes=_infer_n_classes_from_model(model, fallback=int(np.max(y_true)) + 1),
    )
    eval_fifo: deque[dict[str, torch.Tensor]] = deque(maxlen=max(1, int(update_every_x)))
    seen_samples = 0

    def flush_eval_fifo() -> None:
        nonlocal out, n_classes
        if not eval_fifo:
            return
        eval_batch = _stack_note_items(list(eval_fifo))
        adapt_model.eval()
        with torch.no_grad():
            _features, logits = _forward_features_logits(adapt_model, eval_batch["x"].to(device, non_blocking=True))
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)
        if out is None:
            n_classes = int(probs.shape[1])
            out = np.zeros((n_trials, n_classes), dtype=np.float64)
        out[eval_batch["index"].detach().cpu().numpy().astype(np.int64)] = probs
        eval_fifo.clear()

    for xb, yb, idx in loader:
        sample = _clone_batch_item(xb, yb, idx)
        eval_fifo.append(sample)

        adapt_model.eval()
        with torch.no_grad():
            _features, logits = _forward_features_logits(adapt_model, xb.to(device, non_blocking=True))
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)
        if out is None:
            n_classes = int(probs.shape[1])
            out = np.zeros((n_trials, n_classes), dtype=np.float64)
        if bool(use_learned_stats):
            out[idx.detach().cpu().numpy().astype(np.int64)] = probs

        pseudo_label = int(np.argmax(probs, axis=1)[0])
        memory.add_instance(sample, pseudo_label)
        seen_samples += 1

        is_update_step = (seen_samples % max(1, int(update_every_x)) == 0) or (seen_samples == n_trials)
        if not is_update_step:
            continue

        if not bool(use_learned_stats):
            flush_eval_fifo()
        _run_note_updates(
            model=adapt_model,
            optimizer=optimizer,
            memory_items=memory.get_memory(),
            replay_batch_size=replay_batch_size,
            adapt_epochs=adapt_epochs,
            temperature=temperature,
            device=device,
        )

    if not bool(use_learned_stats):
        flush_eval_fifo()
    if out is None:
        raise RuntimeError("NOTE reference produced no outputs")
    return out


def _initialize_t3a_supports_from_source(
    *,
    model: nn.Module,
    X_source: np.ndarray,
    y_source: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = _make_loader(X=X_source, y=y_source, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    n_classes = _infer_n_classes_from_model(model, fallback=int(np.max(y_source)) + 1)
    class_features: list[list[torch.Tensor]] = [[] for _ in range(n_classes)]

    model.eval()
    with torch.no_grad():
        for xb, yb, _idx in loader:
            xb = xb.to(device, non_blocking=True)
            features, _logits = _forward_features_logits(model, xb)
            features = F.normalize(features, dim=1)
            labels = yb.to(device, non_blocking=True)
            for cls in range(n_classes):
                cls_mask = labels == cls
                if cls_mask.any():
                    class_features[cls].append(features[cls_mask])

    classifier = model[1] if isinstance(model, nn.Sequential) and len(model) >= 2 else None
    supports: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for cls in range(n_classes):
        if class_features[cls]:
            cls_features = torch.cat(class_features[cls], dim=0)
            proto = F.normalize(cls_features.mean(dim=0, keepdim=True), dim=1)
        else:
            if classifier is None or not hasattr(classifier, "fc"):
                raise RuntimeError("T3A reference fallback requires classifier.fc when a class has no source supports.")
            template = classifier.fc.weight[cls : cls + 1].detach().to(device)
            proto = F.normalize(template, dim=1)
        supports.append(proto)
        labels.append(F.one_hot(torch.tensor([cls], device=device), num_classes=n_classes).float())
        entropies.append(torch.zeros(1, device=device))
    return torch.cat(supports, dim=0), torch.cat(labels, dim=0), torch.cat(entropies, dim=0)


def _select_t3a_supports(
    *,
    supports: torch.Tensor,
    labels: torch.Tensor,
    entropies: torch.Tensor,
    filter_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices: list[torch.Tensor] = []
    y_hat = labels.argmax(dim=1)
    for cls in range(labels.shape[1]):
        cls_mask = y_hat == cls
        cls_indices = torch.nonzero(cls_mask, as_tuple=False).flatten()
        if cls_indices.numel() == 0:
            continue
        cls_entropy = entropies[cls_indices]
        order = torch.argsort(cls_entropy)
        if int(filter_k) > 0:
            order = order[: min(int(filter_k), int(order.numel()))]
        indices.append(cls_indices[order])
    keep = torch.cat(indices, dim=0)
    return supports[keep], labels[keep]


def run_t3a_kooptta_reference(
    *,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_true: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 0,
    filter_k: int = 16,
) -> np.ndarray:
    adapt_model = deepcopy(model).to(device)
    adapt_model.eval()
    supports, labels, entropies = _initialize_t3a_supports_from_source(
        model=adapt_model,
        X_source=X_source,
        y_source=y_source,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    loader = _make_loader(X=X_target, y=y_true, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    n_trials = int(np.asarray(X_target).shape[0])
    out: np.ndarray | None = None

    for xb, _yb, indices in loader:
        xb = xb.to(device, non_blocking=True)
        features, _logits = _forward_features_logits(adapt_model, xb)

        sel_supports, sel_labels = _select_t3a_supports(
            supports=supports,
            labels=labels,
            entropies=entropies,
            filter_k=filter_k,
        )
        weights = F.normalize(sel_supports, dim=1).T @ sel_labels
        logits = features @ F.normalize(weights, dim=0)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)

        if out is None:
            out = np.zeros((n_trials, int(probs.shape[1])), dtype=np.float64)
        out[indices.detach().cpu().numpy().astype(np.int64)] = probs

        pseudo = F.one_hot(logits.argmax(dim=1), num_classes=int(probs.shape[1])).float()
        entropy = _softmax_entropy(logits)
        supports = torch.cat([supports, features.detach()], dim=0)
        labels = torch.cat([labels, pseudo.detach()], dim=0)
        entropies = torch.cat([entropies, entropy.detach()], dim=0)

    if out is None:
        raise RuntimeError("T3A reference produced no outputs")
    return out


def _collect_features_probs(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    indices_all: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for xb, _yb, idx in loader:
            xb = xb.to(device, non_blocking=True)
            features, logits = _forward_features_logits(model, xb)
            features_all.append(F.normalize(features, dim=1).detach().cpu().numpy())
            probs_all.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            indices_all.append(idx.detach().cpu().numpy().astype(np.int64))

    return (
        np.concatenate(features_all, axis=0),
        np.concatenate(probs_all, axis=0),
        np.concatenate(indices_all, axis=0),
    )


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


def _obtain_shot_pseudo_labels(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pseudo_rounds: int,
    threshold: int,
    distance: str,
) -> tuple[np.ndarray, np.ndarray]:
    features, probs, indices = _collect_features_probs(model=model, loader=loader, device=device)
    predict = np.argmax(probs, axis=1).astype(np.int64)
    feature_bank = features
    if distance == "cosine":
        feature_bank = np.concatenate([feature_bank, np.ones((feature_bank.shape[0], 1), dtype=np.float64)], axis=1)
        feature_bank = _normalize_rows(feature_bank)

    aff = probs.copy()
    n_classes = int(probs.shape[1])
    for _ in range(max(1, int(pseudo_rounds))):
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
    return indices, predict


def run_shot_kooptta_reference(
    *,
    X_target: np.ndarray,
    y_true: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 0,
    epochs: int = 5,
    base_lr: float = 5e-4,
    cls_par: float = 0.3,
    ent_par: float = 1.0,
    threshold: int = 0,
    distance: str = "cosine",
    pseudo_rounds: int = 1,
) -> np.ndarray:
    adapt_model = deepcopy(model).to(device)
    if isinstance(adapt_model, nn.Sequential) and len(adapt_model) >= 2:
        for parameter in adapt_model[1].parameters():
            parameter.requires_grad = False
        trainable = [parameter for parameter in adapt_model[0].parameters() if parameter.requires_grad]
    else:
        raise TypeError("SHOT reference expects an nn.Sequential(netF, netC) EEGNet model")
    optimizer = torch.optim.SGD(
        trainable,
        lr=float(base_lr),
        momentum=0.9,
        weight_decay=1e-3,
        nesterov=True,
    )

    loader = _make_loader(X=X_target, y=y_true, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for _ in range(max(1, int(epochs))):
        indices, refined = _obtain_shot_pseudo_labels(
            model=adapt_model,
            loader=loader,
            device=device,
            pseudo_rounds=pseudo_rounds,
            threshold=threshold,
            distance=distance,
        )
        pseudo_labels: Dict[int, int] = {int(i): int(y) for i, y in zip(indices.tolist(), refined.tolist())}

        _enable_tta_encoder_mode_reference(adapt_model)
        for xb, _yb, idx in loader:
            xb = xb.to(device, non_blocking=True)
            batch_pseudo = torch.as_tensor(
                [pseudo_labels[int(i)] for i in idx.detach().cpu().numpy().astype(np.int64)],
                dtype=torch.long,
                device=device,
            )
            optimizer.zero_grad()
            _features, logits = _forward_features_logits(adapt_model, xb)
            entropy = _softmax_entropy(logits).mean()
            im_loss = float(ent_par) * (entropy - _marginal_entropy(logits))
            loss = float(cls_par) * F.cross_entropy(logits, batch_pseudo) + im_loss
            loss.backward()
            optimizer.step()

    sample = next(iter(loader))
    _xb, _yb, _idx = sample
    with torch.no_grad():
        _features0, logits0 = _forward_features_logits(adapt_model, _xb.to(device, non_blocking=True))
    return _predict_probs(
        model=adapt_model,
        loader=loader,
        device=device,
        n_trials=int(np.asarray(X_target).shape[0]),
        n_classes=int(logits0.shape[1]),
    )


def run_pl_dteeg_reference(
    *,
    X_target: np.ndarray,
    y_true: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    stride: int = 1,
    steps: int = 1,
    lr: float = 1e-3,
) -> np.ndarray:
    adapt_model = deepcopy(model).to(device)
    optimizer = torch.optim.Adam(adapt_model.parameters(), lr=float(lr))
    x_target = np.asarray(X_target, dtype=np.float32, order="C")
    n_trials = int(x_target.shape[0])
    out = np.zeros((n_trials, _infer_n_classes_from_model(adapt_model, fallback=int(np.max(y_true)) + 1)), dtype=np.float64)

    for idx in range(n_trials):
        adapt_model.eval()
        with torch.no_grad():
            _features, logits = _forward_features_logits(adapt_model, _trials_to_device(x_target[idx], device=device))
            out[idx] = torch.softmax(logits, dim=1).detach().cpu().numpy()[0].astype(np.float64)

        should_update = (idx + 1) >= int(batch_size) and ((idx + 1) % max(1, int(stride)) == 0)
        if not should_update:
            continue

        adapt_model.train()
        batch = _trials_to_device(x_target[idx - int(batch_size) + 1 : idx + 1], device=device)
        for _ in range(max(1, int(steps))):
            _features, logits = _forward_features_logits(adapt_model, batch)
            pseudo_labels = logits.argmax(dim=1)
            loss = F.cross_entropy(logits, pseudo_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return out


def run_ttime_dteeg_reference(
    *,
    X_target: np.ndarray,
    y_true: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    stride: int = 1,
    steps: int = 1,
    lr: float = 1e-3,
    temperature: float = 2.0,
) -> np.ndarray:
    adapt_model = deepcopy(model).to(device)
    optimizer = torch.optim.Adam(adapt_model.parameters(), lr=float(lr))
    x_target = np.asarray(X_target, dtype=np.float32, order="C")
    n_trials = int(x_target.shape[0])
    out = np.zeros((n_trials, _infer_n_classes_from_model(adapt_model, fallback=int(np.max(y_true)) + 1)), dtype=np.float64)

    for idx in range(n_trials):
        adapt_model.eval()
        with torch.no_grad():
            _features, logits = _forward_features_logits(adapt_model, _trials_to_device(x_target[idx], device=device))
            out[idx] = torch.softmax(logits, dim=1).detach().cpu().numpy()[0].astype(np.float64)

        should_update = (idx + 1) >= int(batch_size) and ((idx + 1) % max(1, int(stride)) == 0)
        if not should_update:
            continue

        adapt_model.train()
        batch = _trials_to_device(x_target[idx - int(batch_size) + 1 : idx + 1], device=device)
        for _ in range(max(1, int(steps))):
            _features, logits = _forward_features_logits(adapt_model, batch)
            probs = torch.softmax(logits / float(temperature), dim=1)
            cem_loss = _entropy_from_probs(probs).mean()
            msoftmax = probs.mean(dim=0)
            mdr_loss = torch.sum(msoftmax * torch.log(msoftmax.clamp_min(1e-5)))
            loss = cem_loss + mdr_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return out


def run_delta_dteeg_reference(
    *,
    X_target: np.ndarray,
    y_true: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    stride: int = 1,
    steps: int = 1,
    lr: float = 1e-3,
    temperature: float = 2.0,
    lambda_z: float = 0.9,
) -> np.ndarray:
    adapt_model = deepcopy(model).to(device)
    optimizer = torch.optim.Adam(adapt_model.parameters(), lr=float(lr))
    x_target = np.asarray(X_target, dtype=np.float32, order="C")
    n_trials = int(x_target.shape[0])
    n_classes = _infer_n_classes_from_model(adapt_model, fallback=int(np.max(y_true)) + 1)
    out = np.zeros((n_trials, n_classes), dtype=np.float64)
    z = torch.full((n_classes,), 1.0 / float(n_classes), dtype=torch.float32, device=device)
    eps = 1e-5

    for idx in range(n_trials):
        adapt_model.eval()
        with torch.no_grad():
            _features, logits = _forward_features_logits(adapt_model, _trials_to_device(x_target[idx], device=device))
            out[idx] = torch.softmax(logits, dim=1).detach().cpu().numpy()[0].astype(np.float64)

        should_update = (idx + 1) >= int(batch_size) and ((idx + 1) % max(1, int(stride)) == 0)
        if not should_update:
            continue

        adapt_model.train()
        batch = _trials_to_device(x_target[idx - int(batch_size) + 1 : idx + 1], device=device)
        last_msoftmax = None
        for _ in range(max(1, int(steps))):
            _features, logits = _forward_features_logits(adapt_model, batch)
            probs = torch.softmax(logits / float(temperature), dim=1)
            last_msoftmax = probs.mean(dim=0).detach()
            cem_loss = _entropy_from_probs(probs).mean()
            pseudo = probs.argmax(dim=1)
            weights = 1.0 / (z[pseudo] + eps)
            weights = float(batch_size) * weights / weights.sum().clamp_min(eps)
            weighted = (probs.T @ weights.unsqueeze(1)).squeeze(1) / float(batch_size)
            gentropy_loss = torch.sum(weighted * torch.log(weighted.clamp_min(eps)))
            loss = cem_loss + gentropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (idx + 1) % int(batch_size) == 0 and last_msoftmax is not None:
            z = float(lambda_z) * z + (1.0 - float(lambda_z)) * last_msoftmax.to(device)

    return out


def run_sar_dteeg_reference(
    *,
    X_target: np.ndarray,
    y_true: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    stride: int = 1,
    steps: int = 1,
    lr: float = 1e-3,
    rho: float = 0.05,
    entropy_margin: float | None = None,
    recovery_threshold: float = 0.2,
    temperature: float = 2.0,
) -> np.ndarray:
    adapt_model = deepcopy(model).to(device)
    bn_params = _configure_sar_bn_affine_reference(adapt_model)
    if not bn_params:
        raise RuntimeError("SAR reference requires BatchNorm affine parameters, but none were found.")
    optimizer = _SAMReference(bn_params, torch.optim.Adam, lr=float(lr), rho=float(rho))
    init_params = [parameter.detach().clone() for parameter in bn_params]
    x_target = np.asarray(X_target, dtype=np.float32, order="C")
    n_trials = int(x_target.shape[0])
    n_classes = _infer_n_classes_from_model(adapt_model, fallback=int(np.max(y_true)) + 1)
    out = np.zeros((n_trials, n_classes), dtype=np.float64)
    e0 = float(0.4 * math.log(float(n_classes)) if entropy_margin is None else entropy_margin)
    ema_entropy = 0.0

    def recover() -> None:
        nonlocal optimizer, ema_entropy
        for parameter, init in zip(bn_params, init_params):
            parameter.data.copy_(init)
        optimizer = _SAMReference(bn_params, torch.optim.Adam, lr=float(lr), rho=float(rho))
        ema_entropy = 0.0

    for idx in range(n_trials):
        adapt_model.eval()
        with torch.no_grad():
            _features, logits = _forward_features_logits(adapt_model, _trials_to_device(x_target[idx], device=device))
            out[idx] = torch.softmax(logits, dim=1).detach().cpu().numpy()[0].astype(np.float64)

        should_update = (idx + 1) >= int(batch_size) and ((idx + 1) % max(1, int(stride)) == 0)
        if not should_update:
            continue

        batch = _trials_to_device(x_target[idx - int(batch_size) + 1 : idx + 1], device=device)
        adapt_model.train()
        for _ in range(max(1, int(steps))):
            optimizer.zero_grad()
            _features, logits = _forward_features_logits(adapt_model, batch)
            probs = torch.softmax(logits / float(temperature), dim=1)
            entropies = _entropy_from_probs(probs)
            keep = torch.where(entropies < e0)[0]
            if keep.numel() <= 0:
                continue
            loss_first = entropies[keep].mean(0)
            loss_first.backward()
            optimizer.first_step(zero_grad=True)

            _features_adv, logits_adv = _forward_features_logits(adapt_model, batch)
            probs_adv = torch.softmax(logits_adv / float(temperature), dim=1)
            entropies_adv = _entropy_from_probs(probs_adv)[keep]
            mean_adv = float(entropies_adv.detach().mean(0).item())
            keep_adv = torch.where(entropies_adv < e0)[0]
            if keep_adv.numel() > 0:
                loss_second = entropies_adv[keep_adv].mean(0)
            else:
                loss_second = entropies_adv.mean(0) * 0.0
            loss_second.backward()
            optimizer.second_step(zero_grad=True)

            ema_entropy = 0.9 * ema_entropy + 0.1 * mean_adv if ema_entropy != 0.0 else mean_adv
            if ema_entropy < float(recovery_threshold):
                recover()

    return out


def run_cotta_dteeg_reference(
    *,
    X_target: np.ndarray,
    y_true: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
    stride: int = 1,
    steps: int = 1,
    lr: float = 1e-3,
    mt_alpha: float = 0.999,
    rst_m: float = 0.01,
    ap: float = 0.9,
    aug_num: int = 32,
) -> np.ndarray:
    if int(stride) != 1:
        raise ValueError("CoTTA reference expects stride=1.")

    adapt_model = deepcopy(model).to(device)
    optimizer = torch.optim.Adam(adapt_model.parameters(), lr=float(lr))
    cotta_model: _CoTTAReference | None = None
    x_target = np.asarray(X_target, dtype=np.float32, order="C")
    n_trials = int(x_target.shape[0])
    out = np.zeros((n_trials, _infer_n_classes_from_model(adapt_model, fallback=int(np.max(y_true)) + 1)), dtype=np.float64)

    for idx in range(n_trials):
        if (idx + 1) >= int(batch_size):
            adapt_model.train()
            if cotta_model is None:
                cotta_model = _CoTTAReference(
                    adapt_model,
                    optimizer,
                    steps=int(steps),
                    mt_alpha=float(mt_alpha),
                    rst_m=float(rst_m),
                    ap=float(ap),
                    aug_num=int(aug_num),
                ).to(device)
            batch = _trials_to_device(x_target[idx - int(batch_size) + 1 : idx + 1], device=device)
            outputs = cotta_model(batch)[-1:].detach()
            out[idx] = torch.softmax(outputs, dim=1).cpu().numpy()[0].astype(np.float64)
        else:
            adapt_model.eval()
            with torch.no_grad():
                _features, logits = _forward_features_logits(adapt_model, _trials_to_device(x_target[idx], device=device))
                out[idx] = torch.softmax(logits, dim=1).detach().cpu().numpy()[0].astype(np.float64)

    return out


__all__ = [
    "run_adabn_kooptta_reference",
    "run_cotta_dteeg_reference",
    "run_delta_dteeg_reference",
    "run_note_kooptta_reference",
    "run_pl_dteeg_reference",
    "run_sar_dteeg_reference",
    "run_shot_kooptta_reference",
    "run_t3a_kooptta_reference",
    "run_tent_kooptta_reference",
    "run_ttime_dteeg_reference",
    "streaming_normalize_trials_reference",
]
