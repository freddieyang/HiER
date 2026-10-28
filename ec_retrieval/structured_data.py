"""Batched graph utilities for structured reaction retrieval."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Sequence

import torch


@dataclass
class ReactionGraphBatch:
    """A disconnected atom/motif graph batch with molecule and reaction maps."""

    atom_x: torch.Tensor  # [Na, Fa]
    edge_index: torch.Tensor  # [2, Ea]
    edge_attr: torch.Tensor  # [Ea, Fe]
    atom_to_motif: torch.Tensor  # [Na]
    motif_x: torch.Tensor  # [Nm, Fm]
    motif_edge_index: torch.Tensor  # [2, Em]
    atom_component: torch.Tensor  # [Na]
    motif_component: torch.Tensor  # [Nm]
    component_reaction: torch.Tensor  # [Nc]
    num_reactions: int

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "ReactionGraphBatch":
        payload: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            payload[item.name] = (
                value.to(device, non_blocking=non_blocking)
                if isinstance(value, torch.Tensor)
                else value
            )
        return ReactionGraphBatch(**payload)

    def pin_memory(self) -> "ReactionGraphBatch":
        payload: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            payload[item.name] = value.pin_memory() if isinstance(value, torch.Tensor) else value
        return ReactionGraphBatch(**payload)


class ReactionGraphStore:
    """Lazily materialize complete reaction graphs from reusable molecule graphs.

    The offline feature bundle stores each unique molecule only once.  This
    store preserves that compact representation and caches a reaction after its
    first use, reducing training collation from one loop per molecule to one
    loop per reaction after warm-up.
    """

    def __init__(self, bundle: dict[str, Any]) -> None:
        required = {
            "reaction_ids",
            "reaction_component_rows",
            "component_graphs",
            "atom_feature_dim",
            "edge_feature_dim",
            "motif_feature_dim",
            "feature_contract",
        }
        missing = required - set(bundle)
        if missing:
            raise ValueError(f"Reaction graph bundle lacks {sorted(missing)}")
        contract = bundle["feature_contract"]
        if contract.get("uses_pair_associations") or contract.get("uses_split_labels"):
            raise ValueError("Reaction graph bundle violates the query-only feature contract")
        self.reaction_ids = list(map(str, bundle["reaction_ids"]))
        self.reaction_component_rows = bundle["reaction_component_rows"]
        self.component_graphs = bundle["component_graphs"]
        self.atom_feature_dim = int(bundle["atom_feature_dim"])
        self.edge_feature_dim = int(bundle["edge_feature_dim"])
        self.motif_feature_dim = int(bundle["motif_feature_dim"])
        if len(self.reaction_ids) != len(self.reaction_component_rows):
            raise ValueError("Reaction IDs and component lists have different lengths")
        self._cache: dict[int, dict[str, torch.Tensor | int]] = {}

    def __len__(self) -> int:
        return len(self.reaction_ids)

    def materialize(self, reaction_row: int) -> dict[str, torch.Tensor | int]:
        row = int(reaction_row)
        if row < 0 or row >= len(self):
            raise IndexError(f"Reaction row out of bounds: {row}")
        if row in self._cache:
            return self._cache[row]

        atom_x: list[torch.Tensor] = []
        edge_index: list[torch.Tensor] = []
        edge_attr: list[torch.Tensor] = []
        atom_to_motif: list[torch.Tensor] = []
        motif_x: list[torch.Tensor] = []
        motif_edge_index: list[torch.Tensor] = []
        atom_component: list[torch.Tensor] = []
        motif_component: list[torch.Tensor] = []
        atom_offset = 0
        motif_offset = 0
        component_rows = self.reaction_component_rows[row]
        if not component_rows:
            raise ValueError(f"Reaction {self.reaction_ids[row]} has no components")
        for local_component, component_row in enumerate(component_rows):
            graph = self.component_graphs[int(component_row)]
            component_atom_x = torch.as_tensor(graph["atom_x"]).float()
            component_motif_x = torch.as_tensor(graph["motif_x"]).float()
            component_edges = torch.as_tensor(graph["edge_index"]).long()
            component_motif_edges = torch.as_tensor(graph["motif_edge_index"]).long()
            num_atoms = int(component_atom_x.shape[0])
            num_motifs = int(component_motif_x.shape[0])
            if num_atoms <= 0 or num_motifs <= 0:
                raise ValueError("Component graph contains no atom or motif")
            atom_x.append(component_atom_x)
            edge_index.append(component_edges + atom_offset)
            edge_attr.append(torch.as_tensor(graph["edge_attr"]).float())
            atom_to_motif.append(torch.as_tensor(graph["atom_to_motif"]).long() + motif_offset)
            motif_x.append(component_motif_x)
            motif_edge_index.append(component_motif_edges + motif_offset)
            atom_component.append(torch.full((num_atoms,), local_component, dtype=torch.long))
            motif_component.append(torch.full((num_motifs,), local_component, dtype=torch.long))
            atom_offset += num_atoms
            motif_offset += num_motifs

        packed: dict[str, torch.Tensor | int] = {
            "atom_x": torch.cat(atom_x, dim=0),
            "edge_index": torch.cat(edge_index, dim=1),
            "edge_attr": torch.cat(edge_attr, dim=0),
            "atom_to_motif": torch.cat(atom_to_motif, dim=0),
            "motif_x": torch.cat(motif_x, dim=0),
            "motif_edge_index": torch.cat(motif_edge_index, dim=1),
            "atom_component": torch.cat(atom_component, dim=0),
            "motif_component": torch.cat(motif_component, dim=0),
            "num_components": len(component_rows),
        }
        self._cache[row] = packed
        return packed

    def warm(self, reaction_rows: Sequence[int] | None = None) -> None:
        rows = range(len(self)) if reaction_rows is None else reaction_rows
        for row in rows:
            self.materialize(int(row))

    def collate(self, reaction_rows: Sequence[int] | torch.Tensor) -> ReactionGraphBatch:
        if isinstance(reaction_rows, torch.Tensor):
            rows = [int(value) for value in reaction_rows.detach().cpu().tolist()]
        else:
            rows = [int(value) for value in reaction_rows]
        if not rows:
            raise ValueError("Cannot collate an empty reaction batch")

        atom_x: list[torch.Tensor] = []
        edge_index: list[torch.Tensor] = []
        edge_attr: list[torch.Tensor] = []
        atom_to_motif: list[torch.Tensor] = []
        motif_x: list[torch.Tensor] = []
        motif_edge_index: list[torch.Tensor] = []
        atom_component: list[torch.Tensor] = []
        motif_component: list[torch.Tensor] = []
        component_reaction: list[torch.Tensor] = []
        atom_offset = 0
        motif_offset = 0
        component_offset = 0
        for batch_row, reaction_row in enumerate(rows):
            graph = self.materialize(reaction_row)
            graph_atom_x = graph["atom_x"]
            graph_motif_x = graph["motif_x"]
            if not isinstance(graph_atom_x, torch.Tensor) or not isinstance(
                graph_motif_x, torch.Tensor
            ):
                raise TypeError("Cached graph tensors are invalid")
            num_atoms = graph_atom_x.shape[0]
            num_motifs = graph_motif_x.shape[0]
            num_components = int(graph["num_components"])
            atom_x.append(graph_atom_x)
            edge_index.append(torch.as_tensor(graph["edge_index"]) + atom_offset)
            edge_attr.append(torch.as_tensor(graph["edge_attr"]))
            atom_to_motif.append(torch.as_tensor(graph["atom_to_motif"]) + motif_offset)
            motif_x.append(graph_motif_x)
            motif_edge_index.append(
                torch.as_tensor(graph["motif_edge_index"]) + motif_offset
            )
            atom_component.append(
                torch.as_tensor(graph["atom_component"]) + component_offset
            )
            motif_component.append(
                torch.as_tensor(graph["motif_component"]) + component_offset
            )
            component_reaction.append(
                torch.full((num_components,), batch_row, dtype=torch.long)
            )
            atom_offset += int(num_atoms)
            motif_offset += int(num_motifs)
            component_offset += num_components

        batch = ReactionGraphBatch(
            atom_x=torch.cat(atom_x, dim=0),
            edge_index=torch.cat(edge_index, dim=1),
            edge_attr=torch.cat(edge_attr, dim=0),
            atom_to_motif=torch.cat(atom_to_motif, dim=0),
            motif_x=torch.cat(motif_x, dim=0),
            motif_edge_index=torch.cat(motif_edge_index, dim=1),
            atom_component=torch.cat(atom_component, dim=0),
            motif_component=torch.cat(motif_component, dim=0),
            component_reaction=torch.cat(component_reaction, dim=0),
            num_reactions=len(rows),
        )
        if batch.edge_index.shape[1] != batch.edge_attr.shape[0]:
            raise ValueError("Atom edge indices and attributes are misaligned")
        if batch.atom_x.shape[0] != batch.atom_to_motif.shape[0]:
            raise ValueError("Atom and atom-to-motif rows are misaligned")
        return batch

