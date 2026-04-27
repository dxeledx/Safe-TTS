#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an EA-anchor pred_disagree gate from saved predictions CSVs.")
    p.add_argument("--ea-preds", type=Path, required=True, help="Path to EA predictions_all_methods.csv.")
    p.add_argument("--cand-preds", type=Path, required=True, help="Path to candidate predictions_all_methods.csv.")
    p.add_argument("--tau", type=float, required=True, help="Gate threshold: accept candidate if pred_disagree <= tau.")
    p.add_argument(
        "--sweep-max",
        type=float,
        default=0.6,
        help="Max tau for sweep curve (inclusive). Default: 0.6.",
    )
    p.add_argument(
        "--sweep-steps",
        type=int,
        default=61,
        help="Number of sweep points in [0, sweep_max]. Default: 61.",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for CSVs and figures.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ea = pd.read_csv(Path(args.ea_preds))
    cand = pd.read_csv(Path(args.cand_preds))

    required = {"subject", "trial", "y_true", "y_pred"}
    if not required.issubset(ea.columns):
        raise ValueError(f"EA CSV missing columns {sorted(required - set(ea.columns))}.")
    if not required.issubset(cand.columns):
        raise ValueError(f"Candidate CSV missing columns {sorted(required - set(cand.columns))}.")

    m = ea.merge(cand, on=["subject", "trial", "y_true"], suffixes=("_ea", "_cand"))
    if m.empty:
        raise RuntimeError("Merged dataframe is empty; check that subject/trial/y_true keys align between the two runs.")

    m["disagree_trial"] = (m["y_pred_ea"] != m["y_pred_cand"]).astype(float)
    pred_disagree = m.groupby("subject")["disagree_trial"].mean()

    acc_ea = (m["y_pred_ea"] == m["y_true"]).groupby(m["subject"]).mean()
    acc_cand = (m["y_pred_cand"] == m["y_true"]).groupby(m["subject"]).mean()

    tau = float(args.tau)
    accept = pred_disagree <= tau
    acc_sel = acc_ea.copy()
    acc_sel[accept] = acc_cand[accept]

    per_subject = pd.DataFrame(
        {
            "acc_ea": acc_ea,
            "acc_cand": acc_cand,
            "acc_selected": acc_sel,
            "pred_disagree": pred_disagree,
            "selected_cand": accept,
        }
    ).sort_index()
    per_subject["delta_cand_vs_ea"] = per_subject["acc_cand"] - per_subject["acc_ea"]
    per_subject["delta_selected_vs_ea"] = per_subject["acc_selected"] - per_subject["acc_ea"]

    summary = {
        "tau": tau,
        "n_subjects": int(per_subject.shape[0]),
        "mean_acc_ea": float(per_subject["acc_ea"].mean()),
        "mean_acc_cand": float(per_subject["acc_cand"].mean()),
        "mean_acc_selected": float(per_subject["acc_selected"].mean()),
        "mean_delta_selected_vs_ea": float(per_subject["delta_selected_vs_ea"].mean()),
        "neg_transfer_rate_selected_vs_ea": float((per_subject["delta_selected_vs_ea"] < 0).mean()),
        "accept_rate": float(per_subject["selected_cand"].mean()),
        "worst_acc_ea": float(per_subject["acc_ea"].min()),
        "worst_acc_selected": float(per_subject["acc_selected"].min()),
    }

    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    per_subject.to_csv(out_dir / "per_subject.csv", index=True)

    # Sweep
    sweep_max = float(args.sweep_max)
    sweep_steps = int(args.sweep_steps)
    taus = np.linspace(0.0, sweep_max, sweep_steps)
    rows = []
    for t in taus:
        mask = pred_disagree <= float(t)
        acc_s = acc_ea.copy()
        acc_s[mask] = acc_cand[mask]
        delta = acc_s - acc_ea
        rows.append(
            {
                "tau": float(t),
                "mean_acc": float(acc_s.mean()),
                "mean_delta_vs_ea": float(delta.mean()),
                "neg_transfer_rate": float((delta < 0).mean()),
                "accept_rate": float(mask.mean()),
                "worst_acc": float(acc_s.min()),
            }
        )
    df_sweep = pd.DataFrame(rows)
    df_sweep.to_csv(out_dir / "tau_sweep.csv", index=False)

    # Plots
    import matplotlib.pyplot as plt

    # tau sweep
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(df_sweep["tau"], df_sweep["mean_acc"], label="mean acc", color="C0")
    ax1.set_xlabel("tau (max pred_disagree)")
    ax1.set_ylabel("Mean accuracy", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.axvline(tau, color="C0", linestyle="--", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(df_sweep["tau"], df_sweep["neg_transfer_rate"], label="neg transfer rate", color="C3")
    ax2.set_ylabel("Neg-transfer rate vs EA", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tau_sweep.png", dpi=200)
    plt.close(fig)

    # scatter disagree vs delta
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = per_subject["pred_disagree"].to_numpy(float)
    y = per_subject["delta_cand_vs_ea"].to_numpy(float)
    ax.scatter(x, y, c=np.where(y >= 0, "C0", "C3"))
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.5)
    ax.axvline(tau, color="C0", linestyle="--", alpha=0.7)
    for s, row in per_subject.iterrows():
        ax.annotate(str(s), (float(row["pred_disagree"]), float(row["delta_cand_vs_ea"])), fontsize=8, alpha=0.8)
    ax.set_xlabel("pred_disagree(EA, candidate)")
    ax.set_ylabel("Δacc(candidate - EA)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_disagree_vs_delta_cand.png", dpi=200)
    plt.close(fig)

    # per-subject delta selected
    fig, ax = plt.subplots(figsize=(8, 4.5))
    idx = np.arange(per_subject.shape[0])
    colors = ["C0" if v else "C1" for v in per_subject["selected_cand"].to_numpy(bool)]
    ax.bar(idx, per_subject["delta_selected_vs_ea"].to_numpy(float), color=colors)
    ax.axhline(0.0, color="k", linewidth=1, alpha=0.6)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(s) for s in per_subject.index.tolist()])
    ax.set_xlabel("Subject")
    ax.set_ylabel("Δacc(selected - EA)")
    ax.set_title(f"pred_disagree gate (tau={tau})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bar_delta_selected.png", dpi=200)
    plt.close(fig)

    print("Wrote:", out_dir)
    print("Summary:", summary)


if __name__ == "__main__":
    main()

