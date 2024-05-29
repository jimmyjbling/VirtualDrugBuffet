import re

from typing import Iterable

from func_timeout import func_timeout

from rdkit.Chem import (Mol, MolFromSmiles, MolToSmiles, AddHs, MolFromSmarts, RemoveStereochemistry, SanitizeMol,
                        RemoveHs, RemoveHsParameters)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule

# get rid of those pesky rdkit logger dumps
from rdkit.rdBase import BlockLogs
BLOCKER = BlockLogs()


NON_ORGANIC = MolFromSmarts("[!#6;!#5;!#8;!#7;!#16;!#15;!F;!Cl;!Br;!I;!Na;!K;!Mg;!Ca;!Li;!#1]")


def is_mol(obj: object, true_on_none: bool = False) -> bool:
    """
    Returns True if the passed object is a rdkit.Mol object

    Parameters
    ----------
    obj: object
        some object to check
    true_on_none: bool, default = False
        if object is None return True
        this could be useful because rdkit will return `None` for Mols that are invalid (violate valance rules)

    Returns
    -------
    bool
    """
    if obj is None and true_on_none:
        return True
    return isinstance(obj, Mol)


def to_mol(smi: str or object) -> Mol or None:
    """
    Convert a mol (or some object containing a `smiles` attribute) into an RDKit.Mol object
    will return None if a passed object cannot be converted to mol (just like rdkit)

    Parameters
    ----------
    smi: str or object
        something to

    Returns
    -------
    rdkit.Chem.Mol
    """
    if isinstance(smi, Mol):
        return smi
    if isinstance(smi, str):
        return MolFromSmiles(smi)
    if hasattr(smi, 'smiles'):
        return MolFromSmiles(smi.smiles)
    return None


def to_mols(smis: Iterable[str or object]) -> list[Mol]:
    """
    Convert a list (or other iterable) into a list of rdkit Mols

    Parameters
    ----------
    smis: iterable
        a of SMILES (or object with `.smiles` attributes)

    Returns
    -------
    list[Mol]

    """
    return [to_mol(smi) for smi in smis]


def is_smi(obj: object) -> bool:
    """
    Returns `True` if the passed object is a valid SMILES string

    Parameters
    ----------
    obj: object
        object to check if is SMILES

    Returns
    -------
    bool

    """
    if not isinstance(obj, str):
        return False
    if MolFromSmiles(obj) is not None:
        return True
    return False


def to_smi(mol: Mol or object) -> str or None:
    """
    Convert a Mol (or object with a `.smiles` attribute) into a SMILES string
    Returns None when a passed object cannot be converted to SMILES

    Parameters
    ----------
    mol: Mol or object
        object to convert to SMILES

    Returns
    -------
    smi or None

    """
    if isinstance(mol, str):
        return mol
    if isinstance(mol, Mol):
        return MolToSmiles(mol)
    if hasattr(mol, 'smiles'):
        return mol.smiles
    return None


def to_smis(mols: Iterable[Mol or object]) -> list[str or None]:
    """
    Convert an iterable of Mols (or objects with `.smiles` attributes) into a list of rdkit Mols
    Will return None for any objects that cannot be converted into SMILES

    Parameters
    ----------
    mols: iterable
        iterable of objects to convert to SMILES

    Returns
    -------
    list[str]
    """
    return [to_smi(smi) for smi in mols]


def add_3d(mol: Mol or None) -> None:
    """
    Given a rdkit Mol, generate a random energy minimized 3D conformer, inplace

    Parameters
    ----------
    mol: Mol
        Mol to add 3D conformer to

    Returns
    -------
    None

    """
    if mol is None:
        return None
    mol = AddHs(mol)
    ps = ETKDGv3()
    ps.useRandomCoords = True
    func_timeout(10, EmbedMolecule, (mol, ps))
    EmbedMolecule(mol, ps)
    mol.GetConformer()


def add_hydrogen(mol: Mol or None) -> Mol or None:
    """
    Given a Mol, return a new Mol with explicit Hydrogen atoms added

    Parameters
    ----------
    mol: Mol
        Mol to add Hs to

    Returns
    -------
    mol: Mol or None
        new Mol object with Hs (None if passed Mol is None)
    """
    if mol is None:
        return None
    return AddHs(mol)


def remove_stereochem(mol: Mol or None) -> None:
    """
    Given a Mol, return a remove stereochemistry (chirality) inplace

    Parameters
    ----------
    mol: Mol
        Mol to remove stereochem from

    Returns
    -------
    None
    """
    if mol is None:
        return None
    RemoveStereochemistry(mol)


def sanitize_mol(mol: Mol or None) -> Mol or None:
    """
    Given a Mol, return a new Mol that has been 'Sanitized' by RDKit

    Parameters
    ----------
    mol: Mol
        Mol to sanitize

    Returns
    -------
    mol: Mol or None
        new sanitized Mol object (None if passed Mol is None)
    """
    if mol is None:
        return None
    return SanitizeMol(mol)


def mol_is_organic(mol: Mol or None) -> bool:
    """
    Return True if Mol only has organic Atoms
    These atoms are:
        H, C, N, O, S, B, P, S, F, Cl, Br, I, Na, K, Ca, Li, Mg

    Notes
    -----
    Here, organic is referring to atoms that are commonly found in abundance in living organism.

    Parameters
    ----------
    mol: Mol
        Mol to check for organic atoms

    Returns
    -------
    bool
        whether Mol has only organic atoms (None input returns False)
    """
    if mol is None:
        return False
    return not mol.HasSubstructMatch(NON_ORGANIC)


def smi_is_organic(smi: str or object or None) -> bool:
    """
    Checks if a passed SMILES (or object with `.smiles` attribute) is organic

    Notes
    -----
    Function warps `mol_is_organic`, see this function for more details

    Parameters
    ----------
    smi: smi or object
        SMILES to check for organic atoms in

    Returns
    -------
    bool
        returns False if SMILES is invalid or None
    """
    return mol_is_organic(to_mol(smi))


def mol_has_boron(mol: Mol or None) -> bool:
    """
    Return True if Mol has a Boron atom

    Parameters
    ----------
    mol: Mol
        Mol to check for Boron atoms

    Returns
    -------
    bool
        whether Mol has only Boron atoms (None input returns False)
    """
    if mol is None:
        return False
    return mol.HasSubstructMatch(MolFromSmarts("[#5]"))


def smi_has_boron(smi: str or object or None):
    """
    Checks if a passed SMILES (or object with `.smiles` attribute) has Boron

    Notes
    -----
    Function warps `mol_has_boron`, see this function for more details

    Parameters
    ----------
    smi: smi or object
        SMILES to check for Boron atoms in

    Returns
    -------
    bool
        returns False if SMILES is invalid or None
    """
    return mol_has_boron(MolFromSmiles(smi))


def atomize_smiles(smi: str) -> list[str]:
    """
    This function will take a SMILES str (must be the string, not some other object) and tokenizes it to a list of atom
     symbols, one for each atom.
     Will ignore explict H atoms

    Parameters
    ----------
    smi: str

    Returns
    -------
    tokens: list[str]
        a list of atom symbols for each atom in the SMILES

    """
    _tokens = [r"\[Ds.*?\]", r"\[Rg.*?\]", r"\[Cn.*?\]", r"\[He.*?\]", r"\[Li.*?\]", r"\[Be.*?\]", r"\[Ne.*?\]",
               r"\[Na.*?\]", r"\[Mg.*?\]", r"\[Al.*?\]", r"\[Si.*?\]", r"\[?Cl.*?\]?", r"\[Ar.*?\]", r"\[Ca.*?\]",
               r"\[Sc.*?\]", r"\[Ti.*?\]", r"\[Cr.*?\]", r"\[Mn.*?\]", r"\[Fe.*?\]", r"\[Co.*?\]", r"\[Ni.*?\]",
               r"\[Cu.*?\]", r"\[Zn.*?\]", r"\[Ga.*?\]", r"\[Ge.*?\]", r"\[As.*?\]", r"\[Se.*?\]", r"\[?Br.*?\]?",
               r"\[Kr.*?\]", r"\[Rb.*?\]", r"\[Sr.*?\]", r"\[Zr.*?\]", r"\[Nb.*?\]", r"\[Mo.*?\]", r"\[Tc.*?\]",
               r"\[Ru.*?\]", r"\[Rh.*?\]", r"\[Pd.*?\]", r"\[Ag.*?\]", r"\[Cd.*?\]", r"\[In.*?\]", r"\[Sn.*?\]",
               r"\[Sb.*?\]", r"\[Te.*?\]", r"\[Xe.*?\]", r"\[Cs.*?\]", r"\[Ba.*?\]", r"\[La.*?\]", r"\[Ce.*?\]",
               r"\[Pr.*?\]", r"\[Nd.*?\]", r"\[Pm.*?\]", r"\[Sm.*?\]", r"\[Eu.*?\]", r"\[Gd.*?\]", r"\[Tb.*?\]",
               r"\[Dy.*?\]", r"\[Ho.*?\]", r"\[Er.*?\]", r"\[Tm.*?\]", r"\[Yb.*?\]", r"\[Lu.*?\]", r"\[Hf.*?\]",
               r"\[Ta.*?\]", r"\[Re.*?\]", r"\[Os.*?\]", r"\[Ir.*?\]", r"\[Pt.*?\]", r"\[Au.*?\]", r"\[Hg.*?\]",
               r"\[Tl.*?\]", r"\[Pb.*?\]", r"\[Bi.*?\]", r"\[Po.*?\]", r"\[At.*?\]", r"\[Rn.*?\]", r"\[Fr.*?\]",
               r"\[Ra.*?\]", r"\[Ac.*?\]", r"\[Th.*?\]", r"\[Pa.*?\]", r"\[Np.*?\]", r"\[Pu.*?\]", r"\[Am.*?\]",
               r"\[Cm.*?\]", r"\[Bk.*?\]", r"\[Cf.*?\]", r"\[Es.*?\]", r"\[Fm.*?\]", r"\[Md.*?\]", r"\[No.*?\]",
               r"\[Lr.*?\]", r"\[Rf.*?\]", r"\[Db.*?\]", r"\[Sg.*?\]", r"\[Bh.*?\]", r"\[Hs.*?\]", r"\[Mt.*?\]",
               r"\[Nh.*?\]", r"\[Fl.*?\]", r"\[Mc.*?\]", r"\[Lv.*?\]", r"\[Ts.*?\]", r"\[Og.*?\]", r"H",
               r"\[B.*?\]|B", r"\[C.*?\]|C", r"\[N.*?\]|N", r"\[O.*?\]O", r"\[F.*?\]|F", r"\[P.*?\]|P", r"\[S.*?\]|S",
               r"\[K.*?\]", r"\[V.*?\]", r"\[Y.*?\]", r"\[?I.*?\]?", r"\[W.*?\]", r"\[U.*?\]", r"c", r"s", r"o",
               r"n", r"p", r"b"]
    _tokens = re.compile(r"|".join(_tokens)).findall(smi)
    return [re.sub(f"[^a-zA-Z]", "", _) for _ in _tokens]


def neutralize_mol(mol: Mol or None) -> None:
    """
    Removes any neutralize-able charge on the molecule in place.
    See https://www.rdkit.org/docs/Cookbook.html#neutralizing-molecules

    Notes
    -----
    Will remove charge regardless of pka and pH

    Parameters
    ----------
    mol: Mol or None
        Mol to neutralize

    Returns
    -------
    None
    """
    if mol is None:
        return None
    pattern = MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),$([!B&-1])!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            h_count = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(h_count - chg)
            atom.UpdatePropertyCache()


def neutralize_smi(smi: str or object or None) -> None:
    """
    Neutralizes a passed SMILES (or object with `.smiles` attribute)

    Notes
    -----
    Function warps `neutralize_mol`, see this function for more details

    Parameters
    ----------
    smi: smi or object
        SMILES to neutralize

    Returns
    -------
    None
    """
    return MolToSmiles(neutralize_mol(MolFromSmiles(smi)))


def generate_scaffold(mol: str or Mol or object or None, include_chirality: bool = False) -> str or None:
    """
    Takes a mol (or SMILES, or object with `.smiles` attribute) and returns its Murko Scaffold as SMILES

    Parameters
    ----------
    mol: str or Mol or object
        Mol (or SMILES) to get scaffold for
    include_chirality: bool, default False
        include stereochemistry in scaffold

    Returns
    -------
    smiles: str or None
        SMILES of scaffold, None if Mol or SMILES is None or invalid
    """
    if isinstance(mol, str):
        mol = MolFromSmiles(mol)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def remove_Hs(mol: Mol or None, remove_zero_degree: bool = True) -> Mol or None:
    """
    Removes the Hs from a Mol and returns as a new Mol object
    When `remove_zero_degree` is True will also remove any zero-degree Hs (like [H+] ions)

    Parameters
    ----------
    mol: Mol or None
        Mol to remove Hs from
    remove_zero_degree: bool, default True
        whether to also remove zero-degree Hs

    Returns
    -------
    mol: Mol or None
        mol without Hs (or None if mol was None
    """
    params = RemoveHsParameters()
    params.removeDegreeZero = True
    mol_nohs = RemoveHs(mol, params)
    return mol_nohs
