from abc import ABC, abstractmethod

from sklearn.feature_selection import VarianceThreshold as SKVarianceThreshold


class FeatureSelector(ABC):
    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X, y=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y)


class VarianceThreshold(SKVarianceThreshold, FeatureSelector):
    def __init__(self, threshold: float = 0):
        super().__init__(threshold=threshold)
