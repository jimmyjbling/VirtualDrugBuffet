import numpy as np
from func_timeout import FunctionTimedOut

from vdb.curate.steps.base import CurationStep
from vdb.curate.notes import CurationNote
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import add_3d
from vdb.base import compile_step, prep_curation_input


@compile_step
class CurateAdd3D(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.failed_gen_random_3d
        self.note = CurationNote.added_random_3d_conformer
        self.dependency = {"CurateValid"}

    @prep_curation_input
    def _func(self, molecules, y):
        mask = np.ones(len(molecules)).astype(bool)
        for i, mol in enumerate(molecules):
            try:
                add_3d(mol)
            except (ValueError, FunctionTimedOut):
                mask[i] = False
        return mask, molecules, y

    @staticmethod
    def get_rank():
        return 3
