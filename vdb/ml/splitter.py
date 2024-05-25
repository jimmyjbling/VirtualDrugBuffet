from sklearn.model_selection import StratifiedGroupKFold

from vdb.chem.cluster import cluster_scaffold, cluster_leader


class StratifiedScaffoldKFold(StratifiedGroupKFold):
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state=None):
        super().__init__(n_splits, shuffle, random_state)

    def split(self, X, y=None, groups=None):
        groups = cluster_scaffold(X)
        for train, test, in super().split(X, y, groups):
            yield train, test


class StratifiedTanimotoClusterKFold(StratifiedGroupKFold):
    def __init__(self, n_splits: int = 5, thresh: float = 0.65, shuffle: bool = True, random_state=None):
        super().__init__(n_splits, shuffle, random_state)
        self.thresh = thresh

    def split(self, X, y=None, groups=None):
        groups = cluster_leader(X, thresh=self.thresh)
        for train, test, in super().split(X, y, groups):
            yield train, test
