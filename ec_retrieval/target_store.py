"""Pre-materialized TRAIN-only EC supervision for vectorized batch lookup."""
from collections.abc import Sequence
import torch


class ECTargetStore:
    """Move each boolean target table once; unknown entity rows remain all false.

    Accepts only already constructed targets, with no file loading or inference
    input role. The caller is responsible for supplying TRAIN-only annotations.
    """

    def __init__(self, targets_by_level: list[list[tuple[int, ...]]],
                 vocab_sizes: Sequence[int], device: torch.device):
        if not targets_by_level or len(targets_by_level) != len(vocab_sizes):
            raise ValueError("Targets must align with the EC vocabulary levels")
        n = len(targets_by_level[0])
        self.device = device
        self.tables = []
        for targets, size in zip(targets_by_level, vocab_sizes):
            if len(targets) != n or size < 1:
                raise ValueError("Inconsistent target table shape")
            table = torch.zeros((n, size), dtype=torch.bool)
            rows, cols = [], []
            for row, ids in enumerate(targets):
                if any(v < 0 or v >= size for v in ids):
                    raise ValueError("EC target outside TRAIN vocabulary")
                rows.extend([row] * len(ids))
                cols.extend(ids)
            if rows:
                table[rows, cols] = True
            self.tables.append(table.to(device))

    def __getitem__(self, rows):
        indices = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        return [table.index_select(0, indices) for table in self.tables]

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tables)
