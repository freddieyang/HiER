"""Depict externally supplied molecular inventories without inventing atoms."""
from rdkit import Chem


def molecule_for_display(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES in display input")
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 0:
            source_index = atom.GetIsotope()
            atom.SetProp("atomLabel", f"X<sub>{source_index}</sub>" if source_index else "X")
            atom.SetIsotope(0)
    return molecule
