from sklearn.model_selection import StratifiedGroupKFold

from vdb.chem.cluster import cluster_scaffold


class StratifiedScaffoldKFold(StratifiedGroupKFold):
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        super().__init__(n_splits, shuffle, random_state)

    def split(self, X, y=None, groups=None):
        groups = cluster_scaffold(X)
        for train, test, in super().split(X, y, groups):
            yield train, test
