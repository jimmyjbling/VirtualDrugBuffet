import numpy as np

from vdb.curate.steps.base import CurationStep
from vdb.curate.issues import CurationIssue


class CurateRemoveMissingLabel(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.missing_label
        self.requires_y = True

    def _func(self, X, y, **kwargs):
        _labels = np.array(y, dtype=object)

        # I hate the way numpy has theorized its nan type; they define it mathematically but treat it like a NA/Missing
        _nans = []
        for _ in _labels:
            try:
                _nans.append(np.isnan(_))
            except TypeError:
                _nans.append(_ == "nan")
        _empties = (_labels == "")
        _nones = (_labels == "None")
        bad_idx = np.where(_nans + _empties + _nones)[0].astype(int)
        mask = np.ones(len(y)).astype(bool)
        mask[bad_idx] = False
        _labels[bad_idx] = np.nan
        return mask, X, _labels

    @staticmethod
    def get_rank():
        return 3


class CurateMakeNumericLabel(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.non_numeric_label
        self.requires_y = True

    def _func(self, X, y, **kwargs):
        def _check_numeric(x):
            try:
                x = float(x)
                if np.isnan(x):
                    return False
                return True
            except ValueError or TypeError:
                return False

        _labels = np.atleast_1d(y).astype(str)
        good_idx = np.where(np.vectorize(lambda x: _check_numeric(x))(_labels))[0].astype(int)
        bad_idx = np.delete(np.arange(len(y)), good_idx)
        mask = np.ones(len(y)).astype(bool)
        mask[bad_idx] = False
        _labels = _labels.astype(object)
        _labels[bad_idx] = np.nan
        _labels = _labels.astype(np.float32)
        return mask, X, _labels

    @staticmethod
    def get_rank():
        return 3
