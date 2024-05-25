import numpy as np

from vdb.curate.steps.base import CurationStep
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import to_mols


class CurateValid(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.rdkit_failed

    def _func(self, X, y, **kwargs):
        _mols = to_mols(X)
        bad_idx = np.where(np.vectorize(lambda x: x is None)(_mols) > 0)[0]
        mask = np.ones(len(X), dtype=bool)
        mask[bad_idx] = False
        return mask, np.atleast_1d(_mols), y

    @staticmethod
    def get_rank():
        return 1
