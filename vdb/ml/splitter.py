from abc import ABC, abstractmethod

from sklearn.model_selection import StratifiedGroupKFold

from vdb.chem.cluster import cluster_scaffold, cluster_leader
from vdb.base import Step, compile_step


class BaseScaler(ABC, Step):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, X, y=None, **fit_params):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X, y=None):
        raise NotImplementedError


@compile_step
class StratifiedScaffoldKFold(StratifiedGroupKFold):
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state=None):
        super().__init__(n_splits, shuffle, random_state)

    def split(self, X, y=None, groups=None):
        groups = cluster_scaffold(X)
        for train, test, in super().split(X, y, groups):
            yield train, test


@compile_step
class StratifiedTanimotoClusterKFold(StratifiedGroupKFold):
    def __init__(self, n_splits: int = 5, thresh: float = 0.65, shuffle: bool = True, random_state=None):
        super().__init__(n_splits, shuffle, random_state)
        self.thresh = thresh

    def split(self, X, y=None, groups=None):
        groups = cluster_leader(X, thresh=self.thresh)
        for train, test, in super().split(X, y, groups):
            yield train, test
