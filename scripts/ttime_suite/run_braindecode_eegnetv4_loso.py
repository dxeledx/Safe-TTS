from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class EEGNetv4Params:
    max_epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    valid_split: float = 0.2
    early_stop_patience: int = 15
    seed: int = 0
    standardize: str = "none"  # "none" | "train_zscore"
    device: str = "cuda"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _standardize_train_zscore(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)
    return (X_train - mean) / std, (X_test - mean) / std


def _ensure_probabilities(y_raw: np.ndarray) -> np.ndarray:
    y = np.asarray(y_raw, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"Expected 2D output for probabilities/log-probabilities, got {y.shape}")
    row_sums = y.sum(axis=1)
    if np.all(y >= 0.0) and np.all(y <= 1.0) and np.allclose(row_sums, 1.0, atol=1e-4):
        return y
    y_exp = np.exp(y)
    exp_sums = y_exp.sum(axis=1)
    if np.allclose(exp_sums, 1.0, atol=1e-4):
        return y_exp
    y_shift = y - np.max(y, axis=1, keepdims=True)
    y_softmax = np.exp(y_shift)
    y_softmax /= np.sum(y_softmax, axis=1, keepdims=True)
    return y_softmax


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run strict LOSO baseline with Braindecode EEGNetv4 on exported trials.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subjects", type=str, default="ALL", help="ALL or comma-separated subject_idx list.")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--valid-split", type=float, default=0.2)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--standardize", type=str, default="none", choices=["none", "train_zscore"])
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from braindecode import EEGClassifier
    from braindecode.models import EEGNetv4
    from skorch.callbacks import EarlyStopping
    from skorch.dataset import ValidSplit

    params = EEGNetv4Params(
        max_epochs=int(args.max_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        valid_split=float(args.valid_split),
        early_stop_patience=int(args.early_stop_patience),
        seed=int(args.seed),
        standardize=str(args.standardize),
        device=str(args.device),
    )

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_dir / "X.npy").astype(np.float32, copy=False)
    y = np.load(data_dir / "labels.npy").astype(np.int64, copy=False)
    subject_idx = np.load(data_dir / "subject_idx.npy").astype(np.int64, copy=False)
    meta = pd.read_csv(data_dir / "meta.csv")
    class_order = list(_load_json(data_dir / "class_order.json"))

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n_trials, n_channels, n_times), got {X.shape}")

    all_subjects = sorted(int(s) for s in np.unique(subject_idx).tolist())
    if str(args.subjects).strip().upper() == "ALL":
        subjects = all_subjects
    else:
        wanted = [int(s) for s in str(args.subjects).split(",") if str(s).strip()]
        missing = sorted(set(wanted) - set(all_subjects))
        if missing:
            raise ValueError(f"Unknown subject_idx in --subjects: {missing}")
        subjects = wanted

    fold_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []

    base_seed = int(params.seed)
    for test_subject in subjects:
        train_mask = subject_idx != int(test_subject)
        test_mask = subject_idx == int(test_subject)
        X_train = np.asarray(X[train_mask], dtype=np.float32, order="C")
        y_train = np.asarray(y[train_mask], dtype=np.int64, order="C")
        X_test = np.asarray(X[test_mask], dtype=np.float32, order="C")
        y_test = np.asarray(y[test_mask], dtype=np.int64, order="C")

        print(
            f"[loso] start subject={int(test_subject):03d} "
            f"n_train={int(X_train.shape[0])} n_test={int(X_test.shape[0])}",
            flush=True,
        )

        if params.standardize == "train_zscore":
            X_train, X_test = _standardize_train_zscore(X_train, X_test)
            X_train = X_train.astype(np.float32, copy=False)
            X_test = X_test.astype(np.float32, copy=False)

        fold_seed = base_seed + int(test_subject) * 997
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)

        clf = EEGClassifier(
            module=EEGNetv4,
            module__final_conv_length="auto",
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            optimizer__lr=float(params.lr),
            optimizer__weight_decay=float(params.weight_decay),
            batch_size=int(params.batch_size),
            max_epochs=int(params.max_epochs),
            train_split=ValidSplit(float(params.valid_split), random_state=fold_seed),
            callbacks=[
                (
                    "early_stopping",
                    EarlyStopping(monitor="valid_loss", patience=int(params.early_stop_patience)),
                ),
            ],
            device=str(params.device),
            classes=list(range(int(len(class_order)))),
            verbose=0,
        )
        clf.fit(X_train, y_train)

        y_raw = np.asarray(clf.predict_proba(X_test), dtype=np.float64)
        y_proba = _ensure_probabilities(y_raw)
        y_pred = y_proba.argmax(axis=1).astype(np.int64)
        acc = float((y_pred == y_test).mean())

        fold_rows.append(
            {
                "subject": int(test_subject),
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "accuracy": acc,
            }
        )

        subject_meta = meta.loc[test_mask].reset_index(drop=True)
        pred_df = pd.DataFrame(
            {
                "subject": int(test_subject),
                "trial": subject_meta["trial"].to_numpy(dtype=np.int64),
                "method": "eegnetv4_braindecode",
                "y_true": y_test.astype(np.int64),
                "y_pred": y_pred.astype(np.int64),
            }
        )
        for class_idx, class_name in enumerate(class_order):
            pred_df[f"proba_{class_name}"] = y_proba[:, int(class_idx)]
        pred_rows.append(pred_df)

        print(
            f"[loso] subject={int(test_subject):03d} "
            f"n_train={int(X_train.shape[0])} n_test={int(X_test.shape[0])} acc={acc:.6f}",
            flush=True,
        )

    results_df = pd.DataFrame(fold_rows).sort_values("subject").reset_index(drop=True)
    pred_df = pd.concat(pred_rows, axis=0, ignore_index=True)

    mean_acc = float(results_df["accuracy"].mean())
    worst_acc = float(results_df["accuracy"].min())
    std_acc = float(results_df["accuracy"].std(ddof=0))
    best_subject = int(results_df.loc[results_df["accuracy"].idxmax(), "subject"])
    worst_subject = int(results_df.loc[results_df["accuracy"].idxmin(), "subject"])

    summary = {
        "method": "eegnetv4_braindecode",
        "n_subjects": int(len(results_df)),
        "mean_acc": mean_acc,
        "std_subject": std_acc,
        "worst_acc": worst_acc,
        "best_subject": best_subject,
        "worst_subject": worst_subject,
        "params": asdict(params),
        "data_dir": str(data_dir),
    }

    results_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    pred_df.to_csv(out_dir / "predictions_all_methods.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": "eegnetv4_braindecode",
                "n_subjects": int(len(results_df)),
                "mean_acc": mean_acc,
                "std_subject": std_acc,
                "worst_acc": worst_acc,
            }
        ]
    ).to_csv(out_dir / "method_comparison.csv", index=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[done] wrote: {out_dir / 'fold_metrics.csv'}", flush=True)
    print(f"[done] wrote: {out_dir / 'predictions_all_methods.csv'} rows={len(pred_df)}", flush=True)
    print(f"[summary] mean_acc={mean_acc:.6f} worst_acc={worst_acc:.6f} std_subject={std_acc:.6f}", flush=True)


if __name__ == "__main__":
    main()
