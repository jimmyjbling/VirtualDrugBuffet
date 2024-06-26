import numpy as np

from vdb.curate.steps.base import CurationStep
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import mol_is_organic
from vdb.base import compile_step, prep_curation_input


@compile_step
class CurateInorganic(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.inorganic

    # This function is 2x faster than trying to use Mols (below), but mols are created anyway, so use RDKit for now
    # def _func(self, smiles, **kwargs):
    #     bad_idx = np.where(np.vectorize(lambda x: any([_.upper() not in HET_ATOMS for _ in atomize_smiles(x)]))(
    #         np.atleast_1d(smiles)) > 0)[0].astype(int)
    #     good_idx = np.delete(np.arange(len(smiles)), bad_idx)
    #     return good_idx, bad_idx, None

    @prep_curation_input
    def _func(self, molecules, y):
        mask = np.vectorize(lambda x: mol_is_organic(x))(molecules)
        return mask, molecules, y

    @staticmethod
    def get_rank():
        return 4
