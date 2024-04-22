import re

from func_timeout import func_timeout

from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles, AddHs, MolFromSmarts
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule


NON_ORGANIC = MolFromSmarts("[!#6;!#8;!#7;!#16;!#15;!F;!Cl;!Br;!I;!Na;!K;!Mg;!Ca;!Li;!#1]")


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


def add_3d(mol: Mol):
    mol = AddHs(mol)
    ps = ETKDGv3()
    ps.useRandomCoords = True
    func_timeout(10, EmbedMolecule, (mol, ps))
    EmbedMolecule(mol, ps)
    mol.GetConformer()


def add_hydrogen(mol: Mol):
    return AddHs(mol)


def mol_is_organic(mol):
    return not mol.HasSubstructMatch(NON_ORGANIC)


def smi_is_organic(smi):
    return not MolFromSmiles(smi).HasSubstructMatch(NON_ORGANIC)


def mol_has_boron(mol):
    return mol.HasSubstructMatch(MolFromSmarts("[#5]"))


def smi_has_boron(smi):
    return mol_has_boron(MolFromSmiles(smi))


def atomize_smiles(smi):
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


def neutralize_mol(mol):
    if mol is None:
        return None
    pattern = MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
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


def neutralize_smi(smi):
    return MolToSmiles(neutralize_mol(MolFromSmiles(smi)))


def generate_scaffold(mol, include_chirality: bool = False) -> str:
    if isinstance(mol, str):
        mol = MolFromSmiles(mol)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold
