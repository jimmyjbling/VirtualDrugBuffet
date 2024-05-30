import numpy as np

from .base import CurationStep
from ..issues import CurationIssue
from vdb.chem import get_mw


class CurateMW(CurationStep):
    def __init__(self, min_mw: float = 50, max_mw: float = 1000):
        super().__init__(**{"min_mw": min_mw, "max_mw": max_mw})
        self.issue = CurationIssue.wrong_mw
        self._min_mw = min_mw
        self._max_mw = max_mw

        if self._min_mw <= 0:
            raise ValueError(f"min_mw must be greater than 0; got {min_mw}")
        if self._min_mw > self._max_mw:
            raise ValueError(f"min_mw cannot be largers than man_mw; min_mw: {min_mw} max_mw: {max_mw}")

    def _func(self, X, y, **kwargs):
        mask = np.vectorize(lambda x: self._min_mw <= get_mw(x) <= self._max_mw)(np.atleast_1d(X))
        return mask, X, y

    @staticmethod
    def get_rank():
        return 2
