import abc

import numpy as np

from vdb.curate.steps.base import CurationStep
from vdb.curate.issues import CurationIssue
from vdb.chem.utils import to_smis
from vdb.base import compile_step, prep_curation_input


# TODO this needs documentation to difference all the different methods here


@compile_step
class CurateRemoveDuplicates(CurationStep):
    # This assumes that you already have canonical smiles
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.duplicate
        self.dependency = {"CurateValid"}

    @prep_curation_input
    def _func(self, molecules, y):
        smiles = np.atleast_1d(to_smis(molecules)).reshape(-1, 1).astype(str)
        _sorted = smiles[:, 0].argsort()
        bad_idx = np.array([sub_element for element
                            in np.split(_sorted, np.unique(smiles[_sorted], return_index=True)[1][1:])
                            for sub_element in element if len(element) > 1])
        mask = np.ones(len(smiles), dtype=bool)
        mask[bad_idx] = False
        return mask, molecules, y

    @staticmethod
    def get_rank():
        return 7


# keeps the first copy and removes all others
@compile_step
class CurateRemoveDuplicatesGreedy(CurationStep):
    # This assumes that you already have canonical smiles; keeps one of the duplicates
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.duplicate
        self.dependency = {"CurateValid"}

    @prep_curation_input
    def _func(self, molecules, y):
        smiles = np.atleast_1d(to_smis(molecules)).reshape(-1, 1).astype(str)
        _sorted = smiles[:, 0].argsort()
        bad_idx = np.array([sub_element for element
                            in np.split(_sorted, np.unique(smiles[_sorted], return_index=True)[1][1:])
                            for sub_element in element[1:] if len(element) > 1])
        mask = np.ones(len(smiles), dtype=bool)
        mask[bad_idx] = False
        return mask, molecules, y

    @staticmethod
    def get_rank():
        return 7


class _CurateRemoveDisagreeingDuplicates(CurationStep, abc.ABC):
    # This assumes that you already have numerical labels
    def __init__(self, threshold: float = 0.5, log_scale: bool = False, greater: bool = True):
        super().__init__()

        self._error_func = lambda x: -1
        self._agg_func = lambda x: -1
        self.requires_y = True

        self._threshold: float = threshold
        self._log_scale: bool = log_scale
        self._greater: bool = greater

    @prep_curation_input
    def _func(self, molecules, y):
        smiles = np.atleast_1d(to_smis(molecules)).reshape(-1, 1).astype(str)
        _sorted = smiles[:, 0].argsort()
        bad_idx = []
        for group in np.split(_sorted, np.unique(smiles[_sorted], return_index=True)[1][1:]):
            if len(group) == 1:
                continue
            _all_labels = [np.log10(y[_]) if self._log_scale else y[_] for _ in group]
            _err = self._error_func(_all_labels)
            if (_err >= self._threshold) if self._greater else (_err <= self._threshold):
                for idx in group:
                    bad_idx.append(idx)
            else:  # still need to remove the other ones
                y[group] = self._agg_func(_all_labels)
                for idx in group[1:]:
                    bad_idx.append(idx)
        mask = np.ones(len(smiles), dtype=bool)
        mask[bad_idx] = False
        return mask, molecules, y

    @staticmethod
    def get_rank():
        return 8


@compile_step
class CurateRemoveDisagreeingDuplicatesMinMax(_CurateRemoveDisagreeingDuplicates):
    # This assumes that you already have numerical labels
    def __init__(self, threshold: float = 2, log_scale: bool = False, greater: bool = True):
        super().__init__(threshold=threshold, log_scale=log_scale, greater=greater)
        self.issue = CurationIssue.disagreeing_duplicate
        self._threshold = threshold
        self._log_scale = log_scale
        self._greater = greater
        self.dependency = {"CurateValid", "CurateStandardizeNumericalLabels"}

        def error_func(x) -> float:
            return np.abs(np.nanmax(x)-np.nanmin(x))/2

        self._error_func = error_func

        def agg_func(x) -> float:
            return np.nanmean(x)

        self._agg_func = agg_func


@compile_step
class CurateRemoveDisagreeingDuplicatesStd(_CurateRemoveDisagreeingDuplicates):
    # This assumes that you already have numerical labels
    def __init__(self, threshold: float = 0.5, log_scale: bool = False, greater: bool = True):
        super().__init__(threshold=threshold, log_scale=log_scale, greater=greater)
        self.issue = CurationIssue.disagreeing_duplicate
        self.dependency = {"CurateValid", "CurateStandardizeNumericalLabels"}

        def error_func(x) -> float:
            return np.nanstd(x)

        self._error_func = error_func

        def agg_func(x) -> float:
            return np.nanmean(x)

        self._agg_func = agg_func


@compile_step
class CurateRemoveDisagreeingDuplicatesCategorical(_CurateRemoveDisagreeingDuplicates):
    """
    Purity is defined as the occupancy of the most abundant element in a set, so a set of
    [A, A, B, A, A] has a purity of 4/5 or 0.8. One of [B, B, C, C, E] has a purity of 2/5 or 0.4
    """
    def __init__(self, threshold: float = 0.75, greater: bool = False):
        super().__init__(threshold=threshold, log_scale=False, greater=greater)
        self.issue = CurationIssue.disagreeing_duplicate
        self.dependency = {"CurateValid", "CurateStandardizeCategoricalLabels"}

        def error_func(x) -> float:
            return np.nanmax(np.unique(x, return_counts=True)[1]) / len(x)

        self._error_func = error_func

        def agg_func(x) -> str or int:
            vals, counts = np.unique(x, return_counts=True)
            return vals[np.argmax(counts)]

        self._agg_func = agg_func
