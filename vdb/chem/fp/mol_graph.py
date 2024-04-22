from rdkit import Chem

from vdb.chem.fp.base import ObjectFPFunc
from vdb.chem.utils import to_mol, to_smi

MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'het_atom': ["C", "N", "O", "S", "P", "Cl", "Br", "F", "I"],
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


def _onehot_encoding_unk(value: int or str, choices: list[int or str]) -> list[int]:
    """
    Creates a one-hot encoding from a list of choices
    Will be length `choices` + 1 (last index is wildcard)

    Parameters
    ----------
    value: int or str
        the value to be encoded
    choices: list[int or str]
        the list of choices for the one hot vector
        choices should be ordered

    Returns
    -------
    onehot_vector: list[int]
        onehot encoding vector of length `choices` + 1

    Notes
    -----
    If the value passed is not in choices, then the value at index -1 in the returned list will be set to 1.
    Otherwise, the value set 1 in the returned list will match the index of the value in the passes choices list
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


# TODO add documentation here
def _atom_features(atom: Chem.rdchem.Atom) -> list[int or float or bool]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :return: A list containing the atom features.
    """
    features = (_onehot_encoding_unk(atom.GetSymbol(), ATOM_FEATURES['het_atom']) +
                _onehot_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) +
                _onehot_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) +
                _onehot_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) +
                _onehot_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) +
                _onehot_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) +
                [1 if atom.GetIsAromatic() else 0] +
                [atom.GetMass() * 0.01])  # scaled to about the same range as other features
    return features


# TODO add documentation here
def _bond_features(bond: Chem.rdchem.Bond) -> list[int or float or bool]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += _onehot_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


# TODO add documentation here
class MolGraph:
    def __init__(self, smi: str):
        """
        Computes the graph structure and featurization of a molecule.

        :param smi: A smiles string.
        """
        self.smiles = to_smi(smi)
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond

        # Convert smi to molecule
        mol = to_mol(smi)

        # fake the number of "atoms" if we are collapsing substructures
        self.n_atoms = mol.GetNumAtoms()

        # Get atom features
        for i, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(_atom_features(atom))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = _bond_features(bond)

                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2


class MolGraphFunc(ObjectFPFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._func = MolGraph
        self._dimension: int = 1

