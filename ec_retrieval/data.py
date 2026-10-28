"""Data utilities for EC-conditioned enzyme retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

import torch


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MASK_TOKEN = "-"
EC_COMPONENT_PATTERN = re.compile(r"^(?:\d+|n\d+)$", flags=re.IGNORECASE)


def canonicalize_ec(value: object) -> str:
    """Return a four-level EC path while preserving partial-path markers."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise ValueError("EC number is empty")
    parts = [part.strip() for part in text.split(".")]
    if len(parts) != 4:
        raise ValueError(f"EC number must have four levels, got {text!r}")
    if any(not part for part in parts):
        raise ValueError(f"EC number contains an empty level: {text!r}")
    for part in parts:
        if part != MASK_TOKEN and EC_COMPONENT_PATTERN.fullmatch(part) is None:
            raise ValueError(f"Invalid EC component {part!r} in {text!r}")
    first_mask = next((i for i, part in enumerate(parts) if part == MASK_TOKEN), 4)
    if any(part != MASK_TOKEN for part in parts[first_mask:]):
        raise ValueError(f"Partial EC paths must mask a suffix: {text!r}")
    return ".".join(parts)


def expand_ec_values(value: object) -> list[str]:
    """Expand a semicolon-separated set of EC annotations."""
    return [canonicalize_ec(item) for item in str(value).split(";") if item.strip()]


@dataclass(frozen=True)
class ECVocabulary:
    """Per-level vocabulary for exact and partial EC paths."""

    level_to_index: tuple[dict[str, int], ...]

    @classmethod
    def build(cls, ec_numbers: Iterable[object]) -> "ECVocabulary":
        values: list[set[str]] = [set() for _ in range(4)]
        for value in ec_numbers:
            for ec in expand_ec_values(value):
                for level, component in enumerate(ec.split(".")):
                    if component != MASK_TOKEN:
                        values[level].add(component)
        mappings = []
        for level_values in values:
            mapping = {PAD_TOKEN: 0, UNK_TOKEN: 1, MASK_TOKEN: 2}
            def component_order(component: str) -> tuple[int, int | str]:
                if component.isdigit():
                    return 0, int(component)
                if component.lower().startswith("n") and component[1:].isdigit():
                    return 1, int(component[1:])
                return 2, component

            for component in sorted(level_values, key=component_order):
                mapping.setdefault(component, len(mapping))
            mappings.append(mapping)
        return cls(tuple(mappings))

    @property
    def sizes(self) -> tuple[int, int, int, int]:
        return tuple(len(mapping) for mapping in self.level_to_index)  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ECVocabulary":
        levels = payload.get("levels")
        if not isinstance(levels, list) or len(levels) != 4:
            raise ValueError("Serialized EC vocabulary must contain four levels")
        mappings: list[dict[str, int]] = []
        for level in levels:
            if not isinstance(level, dict):
                raise ValueError("Each EC vocabulary level must be a mapping")
            mappings.append({str(key): int(value) for key, value in level.items()})
        return cls(tuple(mappings))

    def encode_one(self, ec_number: object) -> tuple[list[int], list[bool]]:
        parts = canonicalize_ec(ec_number).split(".")
        token_ids: list[int] = []
        valid: list[bool] = []
        masked = False
        for level, component in enumerate(parts):
            masked = masked or component == MASK_TOKEN
            if masked:
                token_ids.append(self.level_to_index[level][MASK_TOKEN])
                valid.append(False)
            else:
                token_ids.append(
                    self.level_to_index[level].get(
                        component, self.level_to_index[level][UNK_TOKEN]
                    )
                )
                valid.append(True)
        if not any(valid):
            raise ValueError("At least the first EC level must be known")
        return token_ids, valid

    def encode_many(self, ec_numbers: Sequence[object]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode_one(value) for value in ec_numbers]
        tokens = torch.tensor([item[0] for item in encoded], dtype=torch.long)
        mask = torch.tensor([item[1] for item in encoded], dtype=torch.bool)
        return tokens, mask

    def to_dict(self) -> dict[str, object]:
        return {"levels": [dict(mapping) for mapping in self.level_to_index]}
