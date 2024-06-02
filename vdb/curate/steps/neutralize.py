import numpy as np

from vdb.curate.steps.base import CurationStep
from vdb.curate.notes import CurationNote
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import neutralize_mol
from vdb.base import compile_step, prep_curation_input


@compile_step
class CurateNeutralize(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.neutralize_failed
        self.note = CurationNote.neutralized

    @prep_curation_input
    def _func(self, molecules, y):
        mask = np.vectorize(lambda x: neutralize_mol(x))(molecules)
        return mask, molecules, y

    @staticmethod
    def get_rank():
        return 3
