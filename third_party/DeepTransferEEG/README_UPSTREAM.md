# Upstream: DeepTransferEEG (T-TIME official implementation)

This directory vendors a subset of the upstream repository **DeepTransferEEG**,
which contains the official implementation of:

- Li et al., *"T-TIME: Test-Time Information Maximization Ensemble for Plug-and-Play BCIs"*,
  IEEE TBME 2024.

## Source

- Repo: `https://github.com/sylyoung/DeepTransferEEG`
- Commit: `e88dbf0676b53b90658d831f9c07b5360697415e`
- License: MIT (see `LICENSE`)

## Notes on local modifications

We apply minimal, explicitly-scoped changes to make the code runnable in this
repo's experiment harness:

- Optional dependencies are guarded (e.g. `learn2learn`) to avoid import-time failures.
- Multi-class methods are adjusted to export **per-trial class probabilities**
  for downstream SAFE_TTA analysis.
- Dataset export + LOSO runner lives under `scripts/ttime_suite/` (not upstream).

