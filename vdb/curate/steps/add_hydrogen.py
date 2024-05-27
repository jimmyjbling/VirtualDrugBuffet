import numpy as np

from vdb.curate.steps.base import CurationStep
from vdb.curate.notes import CurationNote
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import add_hydrogen
from vdb.base import compile_step


@compile_step
class CurateAddH(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.failed_adding_Hs
        self.note = CurationNote.added_Hs

    def _func(self, X, y, **kwargs):
        bad_idx = []
        new_mols = []
        for ii, mol in enumerate(np.atleast_1d(X)):
            try:
                new_mols.append(add_hydrogen(mol))
            except Exception:  # cannot check this in a specific manner because it is a Boost Exception :(
                bad_idx.append(ii)
                new_mols.append(mol)
        bad_idx = np.array(bad_idx).astype(int)
        mask = np.ones(len(X), dtype=bool)
        mask[bad_idx] = False
        return mask, np.atleast_1d(new_mols), y

    @staticmethod
    def get_rank():
        return 3
