"""Molecule splitting shared by the cache-consuming HiER trainers.

The filename is retained for import compatibility. Feature extraction and
cache requirements are described in docs/data.md.
"""
from __future__ import annotations

# Molecule excluded by ReactZyme's official Uni-Mol preprocessing.
REACTZYME_EXCLUDED_MOLECULE = (
    r"[H]C(=O)C1=C(CCC(=O)[O-])C2=N3/C1=C\C1=C(C)C([C@@H](O)CC/"
    r"C=C(\C)CC/C=C(\C)CCC=C(C)C)=C4/C=C5/C(C)=C(C=C)C6=N5[Fe]3"
    r"(N14)N1C(=C2)C(CCC(=O)[O-])=C(C)/C1=C/6"
)


def split_components(reaction: str) -> tuple[list[str], int]:
    raw = [value for value in reaction.split(".") if value]
    filtered = [value for value in raw if value != REACTZYME_EXCLUDED_MOLECULE]
    excluded = len(raw) - len(filtered)
    if not filtered:
        raise ValueError("Reaction contains no encodable molecules after filtering")
    return filtered, excluded
