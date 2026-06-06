"""PK sampler for ReID training.

Samples P identities and K instances per identity in each mini-batch,
which is the standard setup for triplet-based ReID training.
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from typing import Iterator, List

from torch.utils.data import Sampler

from core.reid.datasets.base import ReIDSample


def count_eligible_pids(samples: List[ReIDSample], k: int) -> int:
    pid_to_count: dict[int, int] = defaultdict(int)
    for sample in samples:
        pid_to_count[sample.pid] += 1
    return sum(count >= k for count in pid_to_count.values())


def resolve_pk(
    eligible_count: int,
    p: int = 16,
    k: int = 4,
    fallback_p: int = 8,
) -> tuple[int, int]:
    if eligible_count < p and eligible_count >= fallback_p:
        return fallback_p, k
    return p, k


class PKSampler(Sampler[int]):
    """Randomly samples P identities, then K images per identity.

    Args:
        samples: List of ``ReIDSample`` from the training set.
        p: Number of identities per batch.
        k: Number of instances per identity.
    """

    def __init__(self, samples: List[ReIDSample], p: int = 16, k: int = 4):
        self.samples = samples
        self.p = p
        self.k = k

        self._pid_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, s in enumerate(samples):
            self._pid_to_indices[s.pid].append(idx)
        self._pids = list(self._pid_to_indices.keys())

    def __iter__(self) -> Iterator[int]:
        pids = copy.deepcopy(self._pids)
        random.shuffle(pids)

        batch_indices: List[int] = []
        for pid in pids:
            idxs = copy.deepcopy(self._pid_to_indices[pid])
            if len(idxs) < self.k:
                idxs = idxs * (self.k // len(idxs) + 1)
            random.shuffle(idxs)
            batch_indices.extend(idxs[: self.k])

            if len(batch_indices) >= self.p * self.k:
                yield from batch_indices[: self.p * self.k]
                batch_indices = batch_indices[self.p * self.k :]

        # Yield remaining complete batches
        bs = self.p * self.k
        while len(batch_indices) >= bs:
            yield from batch_indices[:bs]
            batch_indices = batch_indices[bs:]

    def __len__(self) -> int:
        return (len(self._pids) // self.p) * self.p * self.k
