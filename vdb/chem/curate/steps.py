from func_timeout import FunctionTimedOut

import numpy as np
from rdkit.Chem import SANITIZE_NONE

from vdb.chem.curate.base import CurationNote, CurationFunction, CurationIssue
from vdb.chem.utils import (to_mols, add_3d, add_hydrogen, mol_is_organic, mol_has_boron,
                            neutralize_mol, remove_stereochem, sanitize_mol, to_smis)
from vdb.utils import isnan


class CurateRdkit(CurationFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = "smiles"
        self.returns = "mol"
        self.static = False
        self.issue = CurationIssue.rdkit_failed

    def _func(self, smiles, **kwargs):
        _mols = to_mols(smiles)
        bad_idx = np.where(np.vectorize(lambda x: x is None)(_mols) > 0)[0]
        good_idx = np.delete(np.arange(len(smiles)), bad_idx)
        return good_idx, bad_idx, _mols

    @staticmethod
    def get_rank():
        return "rdkit"


class CurateMixtures(CurationFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = "smiles"
        self.static = True
        self.issue = CurationIssue.mixture

    def _func(self, smiles, **kwargs):
        bad_idx = np.flatnonzero(np.core.defchararray.find(np.atleast_1d(smiles).astype(str), ".") != -1)
        good_idx = np.delete(np.arange(len(smiles)), bad_idx)
        return good_idx, bad_idx, None

    @staticmethod
    def get_rank():
        return tuple(["smiles", None, True])


class CurateAdd3D(CurationFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = "mol"
        self.static = False
        self.issue = CurationIssue.failed_gen_random_3d
        self.note = CurationNote.added_random_3d_conformer

    def _func(self, mols, **kwargs):
        bad_idx = []
        for ii, mol in enumerate(np.atleast_1d(mols)):
            try:
                add_3d(mol)
            except (ValueError, FunctionTimedOut):
                bad_idx.append(ii)
                continue
        bad_idx = np.array(bad_idx).astype(int)
        good_idx = np.delete(np.arange(len(mols)), bad_idx)
        return good_idx, bad_idx, None

    @staticmethod
    def get_rank():
        return tuple(["mol", None, False])


class CurateAddH(CurationFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = "mol"
        self.returns = "mol"
        self.static = False
        self.issue = CurationIssue.failed_adding_Hs
        self.note = CurationNote.added_Hs

    def _func(self, mols, **kwargs):
        bad_idx = []
        new_mols = []
        for ii, mol in enumerate(np.atleast_1d(mols)):
            try:
                new_mols.append(add_hydrogen(mol))
            except Exception:  # TODO need to figure out which exceptions this would be
                bad_idx.append(ii)
                new_mols.append(mol)
        bad_idx = np.array(bad_idx).astype(int)
        good_idx = np.delete(np.arange(len(mols)), bad_idx)
        return good_idx, bad_idx, new_mols

    @staticmethod
    def get_rank():
        return tuple(["mol", "mol", False])


class CurateInorganic(CurationFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = "mols"  # if swapping to the non-rdkit func swap this back to smiles
        self.static = True
        self.issue = CurationIssue.inorganic

    # This function is 2x faster than trying to use Mols (below), but mols are created anyway, so use RDKit for now
    # def _func(self, smiles, **kwargs):
    #     bad_idx = np.where(np.vectorize(lambda x: any([_.upper() not in HET_ATOMS for _ in atomize_smiles(x)]))(
    #         np.atleast_1d(smiles)) > 0)[0].astype(int)
    #     good_idx = np.delete(np.arange(len(smiles)), bad_idx)
    #     return good_idx, bad_idx, None

    def _func(self, mols, **kwargs):
        good_idx = np.where(np.vectorize(lambda x: mol_is_organic(x))(np.atleast_1d(mols)) > 0)[0].astype(int)
        bad_idx = np.delete(np.arange(len(mols)), good_idx)
        return good_idx, bad_idx, None

    @staticmethod
    def get_rank():
        return tuple(["smiles", None, True])


class CurateBoron(CurationFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = "mol"  # if swapping to the non-rdkit func swap this back to smiles
        self.static = True
        self.issue = CurationIssue.boron

    # 2x faster than using mols, but mols is more stable
    # def _func(self, smiles, **kwargs):
    #     bad_idx = np.where(np.vectorize(lambda x: any([_.upper() == "B" for _ in atomize_smiles(x)]))(
    #         np.atleast_1d(smiles)) > 0)[0].astype(int)
    #     good_idx = np.delete(np.arange(len(smiles)), bad_idx)
    #     return good_idx, bad_idx, None

    def _func(self, mols, **kwargs):
        bad_idx = np.where(np.vectorize(lambda x: mol_has_boron(x))(np.atleast_1d(mols)) > 0)[0].astype(int)
        good_idx = np.delete(np.arange(len(mols)), bad_idx)
        return good_idx, bad_idx, None

    @staticmethod
    def get_rank():
        return tuple(["smiles", None, True])


class CurateFlatten(CurationFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = "mol"
        self.static = False
        self.issue = CurationIssue.flatten_failed
        self.note = CurationNote.flattened

    def _func(self, mols, **kwargs):
        np.vectorize(lambda x: remove_stereochem(x) if x else None)(np.atleast_1d(mols))
        bad_idx = np.where(np.vectorize(lambda x: x is None)(mols) > 0)[0].astype(int)
        good_idx = np.delete(np.arange(len(mols)), bad_idx)
        return good_idx, bad_idx, None

    @staticmethod
    def get_rank():
        return tuple(["mol", None, False])


class CurateSanitize(CurationFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = "mol"
        self.static = False
        self.issue = CurationIssue.sanitize_failed
        self.note = CurationNote.sanitized

    def _func(self, mols, **kwargs):
        _flags = np.vectorize(lambda x: sanitize_mol(x) if x else None)(np.atleast_1d(mols))
        bad_idx = np.where(np.vectorize(lambda x: x != SANITIZE_NONE)(_flags) > 0)[0].astype(int)
        good_idx = np.delete(np.arange(len(mols)), bad_idx)
        return good_idx, bad_idx, None

    @staticmethod
    def get_rank():
        return tuple(["mol", None, False])


class CurateNeutralize(CurationFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = "mol"
        self.static = False
        self.issue = CurationIssue.neutralize_failed
        self.note = CurationNote.neutralized

    def _func(self, mols, **kwargs):
        np.vectorize(lambda x: neutralize_mol(x) if x else None)(np.atleast_1d(mols))
        bad_idx = np.where(np.vectorize(lambda x: x != SANITIZE_NONE)(mols) > 0)[0].astype(int)
        good_idx = np.delete(np.arange(len(mols)), bad_idx)
        return good_idx, bad_idx, None

    @staticmethod
    def get_rank():
        return tuple(["mol", None, False])


class CurateCanonicalize(CurationFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = "mol"
        self.returns = "smiles"
        self.static = False
        self.issue = CurationIssue.canonical_failed
        self.note = CurationNote.canonical

    def _func(self, mols, **kwargs):
        _smiles = to_smis(np.atleast_1d(mols))
        bad_idx = np.where(np.vectorize(lambda x: x is None)(_smiles) > 0)[0].astype(int)
        good_idx = np.delete(np.arange(len(mols)), bad_idx)
        return good_idx, bad_idx, _smiles

    @staticmethod
    def get_rank():
        return tuple(["mol", "smiles", False])


# any smiles that is repeated, all compounds for that
class CurateRemoveDuplicates(CurationFunction):
    # This assumes that you already have canonical smiles
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = "smiles"
        self.static = True
        self.issue = CurationIssue.duplicate

    def _func(self, smiles, **kwargs):
        idx_sort = np.argsort(np.atleast_1d(smiles))
        _, idx_start, _ = np.unique(np.atleast_1d(smiles)[idx_sort], return_counts=True, return_index=True)
        bad_idx = np.array([__ for _ in np.split(idx_sort, idx_start[1:]) for __ in _ if len(_) > 1]).astype(int)
        good_idx = np.delete(np.arange(len(smiles)), bad_idx)
        return good_idx, bad_idx, None

    @staticmethod
    def get_rank():
        return "dup2"


# keeps the first copy and removes all others
class CurateRemoveDuplicatesGreedy(CurationFunction):
    # This assumes that you already have canonical smiles; keeps one of the duplicates
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = "smiles"
        self.static = True
        self.issue = CurationIssue.duplicate

    def _func(self, smiles, **kwargs):
        idx_sort = np.argsort(np.atleast_1d(smiles))
        _, idx_start, _ = np.unique(np.atleast_1d(smiles)[idx_sort], return_counts=True, return_index=True)
        bad_idx = np.array([__ for _ in np.split(idx_sort, idx_start[1:]) for __ in _[1:]]).astype(int)
        good_idx = np.delete(np.arange(len(smiles)), bad_idx)
        return good_idx, bad_idx, None

    @staticmethod
    def get_rank():
        return "dup2"


class CurateRemoveDisagreeingDuplicates(CurationFunction):
    # This assumes that you already have canonical smiles and numerical labels
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = "smiles"
        self.static = True
        self.issue = CurationIssue.disagreeing_duplicate
        self.labeled = True

    def _func(self, smiles, labels, **kwargs):
        idx_sort = np.argsort(np.atleast_1d(smiles))
        _, idx_start, _ = np.unique(np.atleast_1d(smiles)[idx_sort], return_counts=True, return_index=True)
        np.split(idx_sort, idx_start[1:])
        bad_idx = []
        _labels = np.atleast_1d(labels)
        for _ in np.split(idx_sort, idx_start[1:]):
            if len(_) == 1:
                continue
            _all_labels = [_labels[__] for __ in _]
            relative_err = max(_all_labels) / (min(_all_labels) + 1e-10)  # add noise to the bottom for safe divide
            if relative_err < 0.9:
                for __ in _:
                    bad_idx.append(__)
        bad_idx = np.array(bad_idx).astype(int)
        good_idx = np.delete(np.arange(len(smiles)), bad_idx)
        return good_idx, bad_idx, None

    @staticmethod
    def get_rank():
        return "dup1"


class CurateRemoveMissingLabel(CurationFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = None
        self.static = False
        self.returns = "labels"
        self.issue = CurationIssue.missing_label
        self.labeled = True

    def _func(self, labels, **kwargs):
        _labels = np.atleast_1d(labels)
        _nans = np.isnan(_labels)
        _empties = (_labels == "")
        _nones = (_labels == "None")
        bad_idx = np.where(_nans + _empties + _nones)[0].astype(int)
        good_idx = np.delete(np.arange(len(labels)), bad_idx)
        _labels[bad_idx] = np.nan
        return good_idx, bad_idx, _labels

    @staticmethod
    def get_rank():
        return tuple(["label", "label"])


class CurateMakeNumericLabel(CurationFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = None
        self.static = False
        self.returns = "label"
        self.issue = CurationIssue.non_numeric_label
        self.labeled = True

    def _func(self, labels, **kwargs):
        def _check_numeric(x):
            try:
                x = float(x)
                if np.isnan(x):
                    return False
                return True
            except ValueError or TypeError:
                return False

        _labels = np.atleast_1d(labels).astype(str)
        good_idx = np.where(np.vectorize(lambda x: _check_numeric(x))(_labels))[0].astype(int)
        bad_idx = np.delete(np.arange(len(labels)), good_idx)
        _labels[bad_idx] = np.nan
        _labels = _labels.astype(np.float32)
        return good_idx, bad_idx, _labels

    @staticmethod
    def get_rank():
        return tuple(["label", "label"])


class CurateMakeBinaryLabel(CurationFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uses = None
        self.static = False
        self.returns = "label"
        self.note = CurationNote.label_made_binary
        self.issue = CurationIssue.non_numeric_label
        self.labeled = True

    def _func(self, labels, threshold, greater: bool = True, **kwargs):
        def _binarize(val):
            if isnan(val):
                return np.nan
            return 1 if val > threshold else 0 if greater else 0 if val < threshold else 1

        _labels = np.vectorize(_binarize)(np.atleast_1d(labels))
        bad_idx = np.where(np.vectorize(lambda x: isnan(x))(_labels))[0].astype(int)
        good_idx = np.delete(np.arange(len(labels)), bad_idx)
        _labels = _labels.astype(np.float32)
        return good_idx, bad_idx, _labels

    @staticmethod
    def get_rank():
        return tuple(["numeric_label", "label"])


CURATION_LITERAL_TO_FUNC = {
    "mixture": CurateMixtures,
    "inorganic": CurateInorganic,
    "boron": CurateBoron,
    "rdkit": CurateRdkit,
    "add_h": CurateAddH,
    "add_3d": CurateAdd3D,
    "flatten": CurateFlatten,
    "sanitize": CurateSanitize,
    "neutralize": CurateNeutralize,
    "canonicalize": CurateCanonicalize,
    "remove_duplicates": CurateRemoveDuplicates,
    "remove_disagreeing_duplicates": CurateRemoveDisagreeingDuplicates,
    "remove_duplicates_greedy": CurateRemoveDuplicatesGreedy,
    "missing_label": CurateRemoveMissingLabel,
    "numeric_label": CurateMakeNumericLabel,
    "binary_label": CurateMakeBinaryLabel
}


CURATION_NAME_TO_ISSUE_ENUM = {
    "mixture": CurationIssue.mixture,
    "inorganic": CurationIssue.inorganic,
    "boron": CurationIssue.boron,
    "rdkit": CurationIssue.rdkit_failed,
    "add_h": CurationIssue.failed_adding_Hs,
    "add_3d": CurationIssue.failed_gen_random_3d,
    "flatten": CurationIssue.flatten_failed,
    "sanitize": CurationIssue.sanitize_failed,
    "neutralize": CurationIssue.neutralize_failed,
    "canonicalize": CurationIssue.canonical_failed,
    "remove_duplicates": CurationIssue.duplicate,
    "remove_disagreeing_duplicates": CurationIssue.disagreeing_duplicate,
    "remove_duplicates_greedy": CurationIssue.duplicate,
    "missing_label": CurationIssue.missing_label,
    "numeric_label": CurationIssue.non_numeric_label
}