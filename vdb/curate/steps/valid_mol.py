import numpy as np

from vdb.curate.steps.base import CurationStep
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import to_mols
from vdb.base import compile_step, prep_curation_input


@compile_step
class CurateValid(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.rdkit_failed

    @prep_curation_input
    def _func(self, molecules, y):
        _mols = to_mols(molecules)
        mask = np.vectorize(lambda x: x is None)(_mols)
        return mask, np.atleast_1d(_mols), y

    @staticmethod
    def get_rank():
        return 1
