"""Batch molecule caches without allocating a dense dataset x max-length tensor."""
from __future__ import annotations

import torch


class MoleculeFeatureStore:
    """Index reaction rows into vectorized, masked component batches.

    IDs are used by the data loader to align rows, never as model features.
    Construction is CPU-only metadata work; each training batch uses tensor
    indexing instead of per-molecule Python processing or graph re-encoding.
    """

    def __init__(self, vectors, component_rows, molecule_weights, global_features, device="cpu"):
        vectors = torch.as_tensor(vectors, dtype=torch.float32)
        molecule_weights = torch.as_tensor(molecule_weights, dtype=torch.float32)
        global_features = torch.as_tensor(global_features, dtype=torch.float32)
        if vectors.ndim != 2 or not torch.isfinite(vectors).all():
            raise ValueError("Finite molecule features [N,F] required")
        if molecule_weights.shape != (len(vectors),) or not torch.isfinite(molecule_weights).all() or (molecule_weights <= 0).any():
            raise ValueError("Finite positive weights required for every molecule")
        if global_features.ndim != 2 or len(global_features) != len(component_rows) or not torch.isfinite(global_features).all():
            raise ValueError("Global features must align with reactions")
        lengths = torch.tensor([len(rows) for rows in component_rows], dtype=torch.long)
        if not len(lengths) or (lengths < 1).any():
            raise ValueError("Every reaction must retain its nonempty component multiset")
        padded = torch.zeros(len(lengths), int(lengths.max()), dtype=torch.long)
        for row, values in enumerate(component_rows):  # Once at load time, outside forward.
            values = torch.as_tensor(values, dtype=torch.long)
            if (values < 0).any() or (values >= len(vectors)).any():
                raise ValueError("Invalid molecule row")
            padded[row, :len(values)] = values
        self.device = torch.device(device)
        self.vectors = vectors.to(self.device)
        self.weights = molecule_weights.to(self.device)
        self.global_features = global_features.to(self.device)
        self.rows = padded.to(self.device)
        self.lengths = lengths  # CPU metadata for choosing each batch's padding width.
        self.molecule_input_dim = vectors.shape[1]
        self.global_input_dim = global_features.shape[1]

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu()
        indices = torch.as_tensor(indices, dtype=torch.long).reshape(-1)
        if not len(indices) or (indices < 0).any() or (indices >= len(self)).any():
            raise ValueError("Nonempty valid reaction rows required")
        lengths = self.lengths[indices]
        width = int(lengths.max())
        rows = self.rows[indices.to(self.device), :width]
        mask = torch.arange(width, device=self.device)[None, :] < lengths.to(self.device)[:, None]
        return {"molecules": self.vectors[rows], "weights": self.weights[rows], "mask": mask,
                "global_features": self.global_features[indices.to(self.device)]}
