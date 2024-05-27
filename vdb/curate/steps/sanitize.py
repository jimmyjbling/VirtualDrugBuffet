import numpy as np

from rdkit.Chem import SANITIZE_NONE

from vdb.curate.steps.base import CurationStep
from vdb.curate.notes import CurationNote
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import sanitize_mol
from vdb.base import compile_step


@compile_step
class CurateSanitize(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.sanitize_failed
        self.note = CurationNote.sanitized

    def _func(self, X, y, **kwargs):
        _flags = np.vectorize(lambda x: sanitize_mol(x) if x else -1)(np.atleast_1d(X))
        bad_idx = np.where(np.vectorize(lambda x: x != SANITIZE_NONE)(_flags) > 0)[0].astype(int)
        mask = np.ones(len(X), dtype=bool)
        mask[bad_idx] = False
        return mask, X, y

    @staticmethod
    def get_rank():
        return 3
