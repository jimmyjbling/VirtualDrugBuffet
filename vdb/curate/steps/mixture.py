import numpy as np
from rdkit.Chem import GetMolFrags
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser

from vdb.curate.steps.base import CurationStep
from ..notes import CurationNote
from ..issues import CurationIssue
from vdb.base import compile_step, prep_curation_input
from vdb.chem.utils import remove_Hs


@compile_step
class CurateMixtures(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.mixture

    @prep_curation_input
    def _func(self, molecule, y):
        molecule = np.vectorize(remove_Hs)(molecule)
        mask = np.vectorize(lambda x: len(GetMolFrags(x)) == 1 if x else False)(molecule)
        return mask, molecule, y

    @staticmethod
    def get_rank():
        return 2


@compile_step
class CurateDemix(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.mixture
        self.note = CurationNote.demixed

    @prep_curation_input
    def _func(self, molecule, y, **kwargs):
        _chooser = LargestFragmentChooser()
        molecule = np.vectorize(lambda x: _chooser.choose(remove_Hs(x)) if x is not None else None)(molecule)
        mask = np.vectorize(lambda x: x is not None)(molecule)
        return mask, molecule, y

    @staticmethod
    def get_rank():
        return 2
