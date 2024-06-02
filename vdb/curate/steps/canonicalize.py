import numpy as np

from vdb.curate.steps.base import CurationStep
from vdb.curate.notes import CurationNote
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import to_smis
from vdb.base import compile_step, prep_curation_input


@compile_step
class CurateCanonicalize(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.canonical_failed
        self.note = CurationNote.canonical

        self.dependency = {"CurateValid"}

    @prep_curation_input
    def _func(self, molecules, y):
        _smiles = to_smis(molecules)
        mask = np.vectorize(lambda x: x is not None)(_smiles)
        return mask, np.atleast_1d(_smiles), y

    @staticmethod
    def get_rank():
        return 9
