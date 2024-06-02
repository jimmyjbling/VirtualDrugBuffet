import abc

import numpy.typing as npt

from ..issues import CurationIssue
from ..notes import CurationNote
from ...base import Step


class CurationStepError(Exception):
    """
    Default exception to throw if there is an error raised with a CurationStep
    """
    pass


class CurationStep(abc.ABC, Step):
    """
    The base abstract class for all CurationSteps.
    While following similar SciKitLearn API rules, CurationStep is not a true transformer, since it is only
     generating boolean mask for each chemical on whether it passed the step, or standardizing them
     If you want to use a series of CurationSteps in a SciKit Learn Pipeline (as a Transformer) you should use the
     CurationWorkflow object, which can merge a series of Curation steps into a single Transformer with a scikit-learn
     compatible API.

    CurationSteps must implement 2 methods:
        1. _func(self, molecules, y, **kwargs) -> mask: NDArray, new_X: NDarray, new_y: NDarray
            - This function is what will be called for the main curation
              It should return a boolean mask defining which inputs pass (True) or failed (False) the curation step
              All CurationSteps operate on rdkit.Mol objects (except of the CurateValid function, which is used to
               start the conversion from SMILES to mols) so molecules should be an array of rdkit.Mol objects
              It should be able to handle None or np.nan objects (and of course rdkit.Mol objects)
              The shape of the mask return must match the shape of molecules (input)
              y is for any function that might need to use a label or extra column when curating
               (the CurateRemoveDisagreeingDuplicates is an example of this)
               y is generally the class or regression value and will always be passes, even if it isn't used
              If the Mol objects changed out-of-place, the new mols should be return as an array the same shape as molecules
               if Mols are not changed or changed in-place, return the input molecules
              The above is same for y (labels) as well
        2. get_rank() -> int
            - This function is used to determine when this CurationStep should occur with respect to other steps.
              The order for ranking is as follows:
                1. RDKit mol creation (NEVER USE THIS! ONLY CurateValid SHOULD BE 1)
                2. Subsets a molecule (e.g. mixture handling)
                3. Changes/Standardizes a molecule (e.g. neutralize, flatten)
                4. Excludes a molecule (e.g. inorganic)
                5. Standardize labels (e.g. make numeric/categorical)
                6. Modify labels (e.g. binarize continuous labels)
                7. Duplicate handling (e.g. greedy duplicate handling
                8. Duplicate handling that changes label values
                9. Canonicalize SMILES (NEVER USE THIS! ONLY CurateCanonicalize SHOULD BE 9)
              excludes means values are not changed, just flagged (in-place or out-of-place)
              changes means mols are both flagged and possibly changed

    You must also set the corresponding CurationIssue and CurationNote for the function. These will be attached to any
     mol that failed the curation step (issue) or is somehow altered by the CurationStep (note). Can be left as None if
     no flagging and or changing occurs during the step

    You will also need to specify if the curation function requires the use of "labels" (the `y` parameter) to
     function (the CurationWorkflow uses this to determine if the workflow will require that `y` must be passed)

    Some curation steps might need extra arguments (for example, generating a binary threshold would require a threshold
     parameter is passed). In this case, you can pass that as kwarg during construction, and it will be saved in a kwarg
     dict for later use by _func. This dict will automatically be pass as kwargs to _func, so when declaring your custom
     _func, you can add it to the signature. You can also add it to the constructor

    Lastly, some CurationSteps require an explict dependency before it can be used. A notable example is
     CurateRemoveDisagreeingDuplicatesMinMax, which requires that labels are numeric. Thus, it requires that the
     CurationStep `CurateStandardizeLabels` is run before it. To help a user understand these dependencies all
     CurationSteps have a `self.dependency` attribute that includes the `__name__` attribute of any CurationStep class
     that it is dependent on. There is also a `self.missing_dependency()` function that, when passed a list of step,
     will return [] if the passed list of steps satisfies all dependency, otherwise it returns all missing dependencies

    Attributes
    ----------
    issue: CurationIssue, default None
        the associated curation issue to attached to mol that gets flagged (None if no flagging occurs)
    note: CurationNote, default None
        the associated CurationNote to attach to a mol that gets changed (None if no change occurs)
    requires_y: bool, default False
        whether the CurationStep requires that the `y` variable is passed during the run
    dependency: set[str], default []
        the set of __name__ attributes for the CurationSteps this CurationStep is dependent on
    """
    def __init__(self):
        self.issue: CurationNote or None = None
        self.note: CurationIssue or None = None
        self.requires_y: bool = False

        self.dependency: set[str] = set()

    @abc.abstractmethod
    def _func(self, molecules, y) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        raise NotImplemented

    def __call__(self, molecules, y=None) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        return self._func(molecules, y)

    def __str__(self):
        return str(self.__class__.__name__)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    @abc.abstractmethod
    def get_rank() -> int:
        """
        Get the rank of the step (lower means higher priority)

        Returns
        -------
        rank: int
            The rank
        """
        raise NotImplemented

    def missing_dependency(self, steps: list) -> set[str]:
        """
        Finds all the missing dependency from a given list of steps for this CurationsStep

        Parameters
        ----------
        steps: list[CurationStep]
            the steps that you want to check for missing dependencies

        Returns
        -------
        missing_dependencies: set[str]
            the set of missing_dependencies (will be empty set is none are missing)

        """
        return self.dependency - set([str(step) for step in steps])
