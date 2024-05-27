import numpy as np
from rdkit.Chem import GetMolFrags

from vdb.curate.steps.base import CurationStep
from vdb.curate.issues import CurationIssue
from vdb.base import compile_step


@compile_step
class CurateMixtures(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.mixture

    def _func(self, X, y, **kwargs):
        good_idx = np.where(np.vectorize(lambda x: len(GetMolFrags(x)) == 1 if x else False)(np.atleast_1d(X)) > 0)[0].astype(int)
        mask = np.zeros(len(X), dtype=bool)
        mask[good_idx] = True
        return mask, X, y

    @staticmethod
    def get_rank():
        return 2
