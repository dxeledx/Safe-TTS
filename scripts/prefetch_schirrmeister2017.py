from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


GIN_BASE = "https://web.gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data"


def _default_mne_data_dir() -> Path:
    # MOABB uses MNE's default cache unless overridden.
    return Path.home() / "mne_data"


def _dest_root(*, mne_data_dir: Path) -> Path:
    # Mirrors moabb.datasets.download.data_dl destination path (MOABB>=1.4):
    # <MNE_DATA>/MNE-schirrmeister2017-data/robintibor/high-gamma-dataset/raw/master/data/{train|test}/{subject}.edf
    # (i.e., URL path appended under the "MNE-<sign>-data" folder.)
    return (
        Path(mne_data_dir)
        / "MNE-schirrmeister2017-data"
        / "robintibor"
        / "high-gamma-dataset"
        / "raw"
        / "master"
        / "data"
    )


def _download_one(*, url: str, dest: Path, tries: int, timeout_s: int, quiet: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "--continue",
        "--no-check-certificate",
        "--timeout",
        str(int(timeout_s)),
        "--read-timeout",
        str(int(timeout_s)),
        "--waitretry",
        "5",
        "--retry-connrefused",
        "--tries",
        str(int(tries)),
        "-O",
        str(dest),
        url,
    ]
    if quiet:
        cmd.insert(1, "-q")
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"wget failed (code={res.returncode}) for {url} -> {dest}")


def main() -> int:
    p = argparse.ArgumentParser(description="Prefetch MOABB Schirrmeister2017 EDFs with resumable downloads (wget -c).")
    p.add_argument("--mne-data-dir", type=str, default="", help="Override MNE data dir (default: ~/mne_data).")
    p.add_argument("--subjects", type=str, default="1-14", help="Subject range, e.g. '1-14' or '1,2,3'.")
    p.add_argument("--tries", type=int, default=20)
    p.add_argument("--timeout-s", type=int, default=30)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    mne_data_dir = Path(args.mne_data_dir).expanduser() if str(args.mne_data_dir).strip() else _default_mne_data_dir()
    root = _dest_root(mne_data_dir=mne_data_dir)

    subj_str = str(args.subjects).strip()
    subjects: list[int] = []
    if "-" in subj_str and "," not in subj_str:
        a, b = subj_str.split("-", 1)
        subjects = list(range(int(a), int(b) + 1))
    else:
        subjects = [int(s) for s in subj_str.split(",") if s.strip()]

    for subj in subjects:
        for split in ("train", "test"):
            url = f"{GIN_BASE}/{split}/{subj}.edf"
            dest = Path(root) / split / f"{subj}.edf"
            print(f"[prefetch] {split}/{subj}.edf -> {dest}")
            _download_one(url=url, dest=dest, tries=int(args.tries), timeout_s=int(args.timeout_s), quiet=bool(args.quiet))
    print("[prefetch] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
