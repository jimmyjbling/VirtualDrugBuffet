import numpy as np
from func_timeout import FunctionTimedOut

from vdb.curate.steps.base import CurationStep
from vdb.curate.notes import CurationNote
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import add_3d
from vdb.base import compile_step


@compile_step
class CurateAdd3D(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.failed_gen_random_3d
        self.note = CurationNote.added_random_3d_conformer

    def _func(self, X, y, **kwargs):
        bad_idx = []
        for ii, mol in enumerate(np.atleast_1d(X)):
            try:
                add_3d(mol)
            except (ValueError, FunctionTimedOut):
                bad_idx.append(ii)
                continue
        bad_idx = np.array(bad_idx).astype(int)
        mask = np.ones(len(X), dtype=bool)
        mask[bad_idx] = False

        return mask, X, y

    @staticmethod
    def get_rank():
        return 3
