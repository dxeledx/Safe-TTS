from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> np.ndarray:
    lab_to_i = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        if yt not in lab_to_i or yp not in lab_to_i:
            continue
        cm[lab_to_i[yt], lab_to_i[yp]] += 1
    return cm


def _plot_cm(cm: np.ndarray, labels: list[str], *, out: Path, title: str, normalize: bool) -> None:
    cm = np.asarray(cm, dtype=float)
    disp = cm.copy()
    if normalize:
        row_sum = np.maximum(1.0, disp.sum(axis=1, keepdims=True))
        disp = disp / row_sum

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(disp, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate.
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            val = disp[i, j]
            if normalize:
                txt = f"{val:.2f}"
            else:
                txt = f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot confusion matrix from *_predictions_all_methods.csv (offline).")
    ap.add_argument("--predictions-csv", type=Path, required=True)
    ap.add_argument("--method", type=str, required=True)
    ap.add_argument("--out-no-ext", type=Path, required=True, help="Output path without extension.")
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--normalize", action="store_true", help="Row-normalize confusion matrix.")
    ap.add_argument(
        "--label-order",
        type=str,
        default="",
        help="Optional comma-separated label order (default: infer sorted unique labels from y_true).",
    )
    args = ap.parse_args()

    df = pd.read_csv(Path(args.predictions_csv))
    df = df[df["method"].astype(str) == str(args.method)].copy()
    if df.empty:
        raise SystemExit(f"No rows found for method={args.method} in {args.predictions_csv}")

    y_true = df["y_true"].astype(str).to_numpy()
    y_pred = df["y_pred"].astype(str).to_numpy()
    if args.label_order:
        labels = [s.strip() for s in str(args.label_order).split(",") if s.strip()]
    else:
        labels = sorted(pd.unique(df["y_true"].astype(str)))

    cm = _confusion(y_true, y_pred, labels)
    title = args.title.strip() or f"{args.method} confusion ({'norm' if args.normalize else 'counts'})"
    _plot_cm(cm, labels, out=Path(args.out_no_ext), title=title, normalize=bool(args.normalize))


if __name__ == "__main__":
    main()
