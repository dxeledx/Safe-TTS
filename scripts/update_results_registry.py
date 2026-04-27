from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


@dataclass(frozen=True)
class RunContext:
    date: str
    task: str
    run_name: str
    run_dir: Path
    method_comparison_csv: Path
    results_txt: Path | None
    dataset: str | None
    preprocess_line: str | None
    preprocess_mode: str | None
    events: str | None
    sessions: str | None
    n_components: int | None
    metrics_average: str | None
    git_commit: str | None
    command_line: str | None


def _parse_list_str(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()
    # Keep as a stable comma-separated string.
    raw = raw.replace("'", "").replace('"', "")
    return ",".join([x.strip() for x in raw.split(",") if x.strip()])


def _parse_results_txt(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()
    out: dict = {}

    def _find(prefix: str) -> str | None:
        for line in lines:
            if line.startswith(prefix):
                return line.strip()
        return None

    dataset = _find("Dataset:")
    if dataset:
        out["dataset"] = dataset.split(":", 1)[1].strip() or None

    preproc = _find("Preprocessing:")
    out["preprocess_line"] = preproc
    if preproc:
        m = re.search(r"mode=([^,]+)", preproc)
        out["preprocess_mode"] = m.group(1).strip() if m else None
        m = re.search(r"events=(\[[^\]]*\])", preproc)
        out["events"] = _parse_list_str(m.group(1)) if m else None
        m = re.search(r"sessions=(\[[^\]]*\]|ALL)", preproc)
        out["sessions"] = _parse_list_str(m.group(1)) if m else None

    model = _find("Model:")
    if model:
        m = re.search(r"n_components=(\d+)", model)
        out["n_components"] = int(m.group(1)) if m else None

    avg = _find("Metrics average:")
    if avg:
        out["metrics_average"] = avg.split(":", 1)[1].strip()

    git_line = _find("Git commit:")
    if git_line:
        out["git_commit"] = git_line.split(":", 1)[1].strip() or None

    cmd_line = _find("Command:")
    if cmd_line:
        out["command_line"] = cmd_line.split(":", 1)[1].strip() or None

    return out


def _infer_context(*, method_csv: Path) -> RunContext:
    method_csv = Path(method_csv)
    run_dir = method_csv.parent
    run_name = run_dir.name
    task = run_dir.parent.name if run_dir.parent else ""

    date = ""
    m = re.match(r"(\d{8})_method_comparison\.csv$", method_csv.name)
    if m:
        date = m.group(1)
    else:
        # Fallback: take the first 8-digit token from the path.
        m2 = re.search(r"/(\d{8})/", str(method_csv).replace("\\", "/"))
        date = m2.group(1) if m2 else ""

    results_txt = None
    results_txts = sorted(run_dir.glob("*_results.txt"))
    if results_txts:
        results_txt = results_txts[0]

    meta = {}
    if results_txt is not None:
        meta = _parse_results_txt(results_txt)

    return RunContext(
        date=str(date),
        task=str(task),
        run_name=str(run_name),
        run_dir=run_dir,
        method_comparison_csv=method_csv,
        results_txt=results_txt,
        dataset=meta.get("dataset"),
        preprocess_line=meta.get("preprocess_line"),
        preprocess_mode=meta.get("preprocess_mode"),
        events=meta.get("events"),
        sessions=meta.get("sessions"),
        n_components=meta.get("n_components"),
        metrics_average=meta.get("metrics_average"),
        git_commit=meta.get("git_commit"),
        command_line=meta.get("command_line"),
    )


def build_registry(*, outputs_dir: Path) -> pd.DataFrame:
    outputs_dir = Path(outputs_dir)
    csvs = sorted(outputs_dir.glob("*/*/*/*_method_comparison.csv"))
    if not csvs:
        raise RuntimeError(f"No *_method_comparison.csv found under {outputs_dir}")

    rows: list[dict] = []
    for p in csvs:
        ctx = _infer_context(method_csv=p)
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            row = {
                "date": ctx.date,
                "task": ctx.task,
                "run_name": ctx.run_name,
                "run_dir": str(ctx.run_dir),
                "method_comparison_csv": str(ctx.method_comparison_csv),
                "results_txt": (str(ctx.results_txt) if ctx.results_txt is not None else ""),
                "dataset": ctx.dataset or "",
                "preprocess_mode": ctx.preprocess_mode or "",
                "events": ctx.events or "",
                "sessions": ctx.sessions or "",
                "n_components": (int(ctx.n_components) if ctx.n_components is not None else ""),
                "metrics_average": ctx.metrics_average or "",
                "git_commit": ctx.git_commit or "",
                "command_line": ctx.command_line or "",
                "preprocess_line": ctx.preprocess_line or "",
            }
            for col in df.columns:
                row[col] = r.get(col)
            rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["date", "task", "run_name", "method"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a results registry by scanning outputs/*/*/*/*_method_comparison.csv")
    ap.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--out", type=Path, default=Path("docs/experiments/results_registry.csv"))
    args = ap.parse_args()

    df = build_registry(outputs_dir=args.outputs_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
