from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles


def is_mol(obj):
    return isinstance(obj, Mol)


def to_mol(smi):
    if isinstance(smi, Mol):
        return smi
    if isinstance(smi, str):
        return MolFromSmiles(smi)
    if hasattr(smi, 'smiles'):
        return MolFromSmiles(smi.smiles)
    return None


def to_mols(smis):
    return [to_mol(smi) for smi in smis]


def is_smi(obj):
    if not isinstance(obj, str):
        return False
    if MolFromSmiles(obj) is not None:
        return True
    return False


def to_smi(smi):
    if isinstance(smi, str):
        return smi
    if isinstance(smi, Mol):
        return MolToSmiles(smi)
    if hasattr(smi, 'smiles'):
        return smi.smiles
    return None


def to_smis(smis):
    return [to_smi(smi) for smi in smis]
