import numpy as np

from vdb.curate.steps.base import CurationStep
from vdb.curate.notes import CurationNote
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import remove_stereochem


class CurateFlatten(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.flatten_failed
        self.note = CurationNote.flattened

    def _func(self, X, y, **kwargs):
        np.vectorize(lambda x: remove_stereochem(x) if x else None)(np.atleast_1d(X))
        bad_idx = np.where(np.vectorize(lambda x: x is None)(X) > 0)[0].astype(int)
        mask = np.ones(len(X), dtype=bool)
        mask[bad_idx] = False
        return mask, X, y

    @staticmethod
    def get_rank():
        return 3
