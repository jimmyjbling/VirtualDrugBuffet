import numpy as np

from rdkit.Chem import SANITIZE_NONE

from vdb.curate.steps.base import CurationStep
from vdb.curate.notes import CurationNote
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import sanitize_mol
from vdb.base import compile_step, prep_curation_input


@compile_step
class CurateSanitize(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.sanitize_failed
        self.note = CurationNote.sanitized

    @prep_curation_input
    def _func(self, molecules, y):
        _flags = np.vectorize(lambda x: sanitize_mol(x) if x else -1)(molecules)
        mask = np.vectorize(lambda x: x != SANITIZE_NONE)(_flags)
        return mask, molecules, y

    @staticmethod
    def get_rank():
        return 3
