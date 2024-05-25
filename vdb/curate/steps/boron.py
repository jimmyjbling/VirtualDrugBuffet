import numpy as np

from vdb.curate.steps.base import CurationStep
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import mol_has_boron


class CurateBoron(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.boron

    # 2x faster than using mols, but mols is more stable
    # def _func(self, smiles, **kwargs):
    #     bad_idx = np.where(np.vectorize(lambda x: any([_.upper() == "B" for _ in atomize_smiles(x)]))(
    #         np.atleast_1d(smiles)) > 0)[0].astype(int)
    #     good_idx = np.delete(np.arange(len(smiles)), bad_idx)
    #     return good_idx, bad_idx, None

    def _func(self, X, y, **kwargs):
        bad_idx = np.where(np.vectorize(lambda x: mol_has_boron(x))(np.atleast_1d(X)) > 0)[0].astype(int)
        mask = np.ones(len(X), dtype=bool)
        mask[bad_idx] = False
        return mask, X, y

    @staticmethod
    def get_rank():
        return 2
