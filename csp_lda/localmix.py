from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

import numpy as np


def _fallback_index_neighbors(i: int, *, n_channels: int, k: int) -> list[int]:
    left = [j for j in range(i - 1, -1, -1)]
    right = [j for j in range(i + 1, int(n_channels))]
    out: list[int] = []
    for a, b in zip(left, right):
        out.extend([a, b])
    out.extend(left[len(right) :])
    out.extend(right[len(left) :])
    out = [j for j in out if j != i]
    return out[: int(k)]


@lru_cache(maxsize=32)
def knn_channel_neighbors(
    ch_names: tuple[str, ...],
    *,
    k: int = 4,
    montage: str = "standard_1020",
) -> List[List[int]]:
    """Build a locality graph (neighbors per channel) using a template montage.

    This uses only channel names and a *template* montage (no subject-specific coordinates),
    which is appropriate for public datasets like BCI IV-2a where true electrode locations
    are not available.

    Returns
    -------
    neighbors:
        List of length C; neighbors[i] is a list of up to k indices (excluding i).
    """

    if int(k) < 0:
        raise ValueError("k must be >= 0.")
    ch_names = tuple(str(c) for c in ch_names)
    n_channels = len(ch_names)
    if n_channels < 1:
        raise ValueError("Need at least one channel.")
    if int(k) == 0:
        return [[] for _ in range(int(n_channels))]

    import mne

    montage_obj = mne.channels.make_standard_montage(str(montage))
    ch_pos = montage_obj.get_positions().get("ch_pos", {})

    coords = np.full((int(n_channels), 3), np.nan, dtype=np.float64)
    for i, name in enumerate(ch_names):
        pos = ch_pos.get(str(name))
        if pos is None:
            continue
        coords[int(i)] = np.asarray(pos, dtype=np.float64).reshape(3)

    neighbors: List[List[int]] = []
    for i in range(int(n_channels)):
        if not np.all(np.isfinite(coords[int(i)])):
            neighbors.append(_fallback_index_neighbors(int(i), n_channels=int(n_channels), k=int(k)))
            continue

        diff = coords - coords[int(i)].reshape(1, 3)
        d = np.sqrt(np.sum(diff * diff, axis=1))
        d[int(i)] = np.inf
        order = np.argsort(d)
        chosen = [int(j) for j in order[: int(k)] if np.isfinite(d[int(j)])]
        if len(chosen) < int(k):
            extra = _fallback_index_neighbors(int(i), n_channels=int(n_channels), k=int(k))
            for j in extra:
                if j == int(i) or j in chosen:
                    continue
                chosen.append(int(j))
                if len(chosen) >= int(k):
                    break
        neighbors.append(chosen[: int(k)])

    return neighbors


def knn_channel_neighbors_from_names(
    ch_names: Sequence[str],
    *,
    k: int = 4,
    montage: str = "standard_1020",
) -> List[List[int]]:
    return knn_channel_neighbors(tuple(str(c) for c in ch_names), k=int(k), montage=str(montage))

