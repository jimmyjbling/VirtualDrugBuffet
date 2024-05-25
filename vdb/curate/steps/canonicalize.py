import numpy as np

from vdb.curate.steps.base import CurationStep
from vdb.curate.notes import CurationNote
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import to_smis


class CurateCanonicalize(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.canonical_failed
        self.note = CurationNote.canonical

    def _func(self, X, y, **kwargs):
        _smiles = to_smis(np.atleast_1d(X))
        bad_idx = np.where(np.vectorize(lambda x: x is None)(_smiles) > 0)[0].astype(int)
        mask = np.ones(len(X), dtype=bool)
        mask[bad_idx] = False
        return mask, np.atleast_1d(_smiles), y

    @staticmethod
    def get_rank():
        return 6
