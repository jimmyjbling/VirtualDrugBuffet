import numpy as np

from vdb.curate.steps.base import CurationStep
from vdb.curate.notes import CurationNote
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import neutralize_mol
from vdb.base import compile_step


@compile_step
class CurateNeutralize(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.neutralize_failed
        self.note = CurationNote.neutralized

    def _func(self, X, y, **kwargs):
        np.vectorize(lambda x: neutralize_mol(x) if x else None)(np.atleast_1d(X))
        bad_idx = np.where(np.vectorize(lambda x: x is None)(X) > 0)[0].astype(int)
        mask = np.ones(len(X), dtype=bool)
        mask[bad_idx] = False
        return mask, X, y

    @staticmethod
    def get_rank():
        return 3
