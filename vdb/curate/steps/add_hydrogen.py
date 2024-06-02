import numpy as np

from vdb.curate.steps.base import CurationStep
from vdb.curate.notes import CurationNote
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import add_hydrogen
from vdb.base import compile_step, prep_curation_input


@compile_step
class CurateAddH(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.failed_adding_Hs
        self.note = CurationNote.added_Hs

        self.dependency = {"CurateValid"}

    @prep_curation_input
    def _func(self, molecules, y):
        mask = np.ones(len(molecules)).astype(bool)
        new_molecules = []
        for i, mol in enumerate(np.atleast_1d(molecules)):
            try:
                new_molecules.append(add_hydrogen(mol))
            except Exception:  # cannot check this in a specific manner because it is a Boost Exception :(
                mask[i] = False
                new_molecules.append(mol)
        return mask, np.atleast_1d(new_molecules), y

    @staticmethod
    def get_rank():
        return 3
