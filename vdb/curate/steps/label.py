from typing import Literal

import numpy as np
from numpy import typing as npt

from vdb.curate.steps.base import CurationStep, CurationStepError
from ..issues import CurationIssue
from ..notes import CurationNote
from vdb.base import compile_step, prep_curation_input
from vdb.utils import isnan


@compile_step
class CurateRemoveMissingLabel(CurationStep):
    """
    This set is depreciated
    """
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.missing_label
        self.requires_y = True
        self.dependency = {"CurateMakeLabelNumeric"}

    @prep_curation_input
    def _func(self, molecules, y):
        mask = (~isnan(y)).all(axis=1)
        return mask, molecules, y

    @staticmethod
    def get_rank():
        return 5


@compile_step
class CurateMakeLabelNumeric(CurationStep):
    """
    This set is depreciated
    """
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.non_numeric_label
        self.note = CurationNote.label_made_numeric
        self.requires_y = True

    @prep_curation_input
    def _func(self, molecules, y):
        mask = np.ones(len(y))
        y = y.astype(np.float32)
        return mask, molecules, y

    @staticmethod
    def get_rank():
        return 5


@compile_step
class CurateFilterLabel(CurationStep):
    def __init__(self, value: object,
                 compare: Literal["greater", "less", "greater_equal", "less_equal",
                                  "in", "not_in", "equal", "not_equal"]):
        super().__init__()

        self.dependency = {"CurateMakeLabelNumeric"}

        self._value = value
        self._compare = compare

        if self._compare not in ["greater", "less", "greater_equal", "less_equal",
                                 "in", "not_in", "equal", "not_equal"]:
            raise CurationStepError(f"compare option {self._compare} not valid")

        if isinstance(value, (list, np.ndarray)):
            if compare not in ["in", "not_in"]:
                raise CurationStepError(f"compare value must be 'in' or 'not_in' when value is {type(value)}")
        else:
            if compare in ["in", "not_in"]:
                raise CurationStepError(f"compare value cannot be 'in' or 'not_in' when value is {type(value)}")

    @prep_curation_input
    def _func(self, molecules, y) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        if self._compare == "greater":
            mask = y > self._value
        elif self._compare == "greater_equal":
            mask = y >= self._value
        elif self._compare == "less":
            mask = y < self._value
        elif self._compare == "less_equal":
            mask = y <= self._value
        elif self._compare == "equal":
            mask = y == self._value
        elif self._compare == "not_equal":
            mask = y != self._value
        elif self._compare == "equal":
            mask = y == self._value
        elif self._compare == "not_equal":
            mask = y != self._value
        else:
            raise CurationStepError(f"unrecognized compare step {self._compare}")
        return mask, molecules, y

    @staticmethod
    def get_rank():
        return 5


@compile_step
class CurateStandardizeCategoricalLabels(CurationStep):
    def __init__(self, encoding_map: dict = None):
        super().__init__()
        self.issue = CurationIssue.non_numeric_label
        self.note = CurationNote.label_made_numeric
        self.requires_y = True

        self.cat_dict = encoding_map if encoding_map else {}

        try:
            [int(_) for _ in self.cat_dict.values()]
        except (TypeError, ValueError):
            raise CurationStepError("passed `encoding_map` must map to integer cast-able values")

    @prep_curation_input
    def _func(self, molecules, y):
        def _cat_or_nan(x):
            if isnan(x):
                return np.nan
            _val = self.cat_dict.get(x, None)
            if _val is None:
                _val = len(self.cat_dict)
                self.cat_dict[x] = _val
            return _val

        y = np.vectorize(_cat_or_nan, otypes="f")(y)
        mask = (~isnan(y)).all(axis=1)

        return mask, molecules, y

    @staticmethod
    def get_rank():
        return 5


@compile_step
class CurateStandardizeNumericalLabels(CurationStep):
    def __init__(self):
        super().__init__()
        self.issue = CurationIssue.non_numeric_label
        self.note = CurationNote.label_made_numeric
        self.requires_y = True

    @prep_curation_input
    def _func(self, molecules, y):
        def _float_or_nan(x):
            try:
                return float(x)
            except (ValueError, TypeError):
                return np.nan

        y = np.vectorize(_float_or_nan)(y).astype(np.float32)
        mask = (~isnan(y)).all(axis=1)
        return mask, molecules, y

    @staticmethod
    def get_rank():
        return 5


@compile_step
class CurateDigitizeLabel(CurationStep):
    def __init__(self, thresholds: list[float], greater: bool = True):
        super().__init__()
        self.note = CurationNote.digitized_label
        self.requires_y = True
        self.dependency = {"CurateMakeLabelNumeric"}

        self._thresholds = thresholds
        self._greater = greater

        if not all(a <= b for a, b in zip(self._thresholds, self._thresholds[1:])):
            raise CurationStepError(f"`threshold` must be sorted smallest to largest; got {self._thresholds}")

    @prep_curation_input
    def _func(self, molecules, y):
        mask = ~np.isnan(y)
        y = y.copy()  # don't want to change label inplace
        y[mask] = np.digitize(y[mask], sorted(self._thresholds, reverse=not self._greater))
        return mask.all(axis=1), molecules, y

    @staticmethod
    def get_rank():
        return 6


@compile_step
class CurateBinarizeLabel(CurationStep):
    def __init__(self, threshold: float, greater: bool = True):
        super().__init__()
        self._threshold = threshold
        self._greater = greater

        self.dependency = {"CurateMakeLabelNumeric"}
        self.note = CurationNote.binarize_label

    @prep_curation_input
    def _func(self, molecules, y):
        mask = ~np.isnan(y)
        y = y.copy()  # don't want to change label inplace
        y[mask] = y[mask] > self._threshold if self._greater else y[mask] < self._threshold
        return mask.all(axis=1), molecules, y

    @staticmethod
    def get_rank():
        return 6
