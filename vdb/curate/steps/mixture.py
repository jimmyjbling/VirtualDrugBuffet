import numpy as np
from rdkit.Chem import GetMolFrags
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser

from vdb.curate.steps.base import CurationStep
from ..notes import CurationNote
from ..issues import CurationIssue
from vdb.base import compile_step
from vdb.chem.utils import remove_Hs


@compile_step
class CurateMixtures(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.mixture

    def _func(self, X, y, **kwargs):
        _X = np.array([remove_Hs(_mol) for _mol in X])
        good_idx = np.where(
            np.vectorize(lambda x: len(GetMolFrags(x)) == 1 if x else False)(np.atleast_1d(_X)) > 0
        )[0].astype(int)
        mask = np.zeros(len(_X), dtype=bool)
        mask[good_idx] = True
        return mask, _X, y

    @staticmethod
    def get_rank():
        return 2


@compile_step
class CurateDemix(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.mixture
        self.note = CurationNote.demixed

    def _func(self, X, y, **kwargs):
        _chooser = LargestFragmentChooser()

        _X = np.array([_chooser.choose(remove_Hs(_mol)) if _mol is not None else None for _mol in X])
        mask = np.vectorize(lambda x: x is not None)(_X) > 0
        return mask, _X, y

    @staticmethod
    def get_rank():
        return 2
