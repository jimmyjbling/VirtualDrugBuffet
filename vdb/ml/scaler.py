from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Literal

from sklearn.preprocessing import MinMaxScaler as SK_MinMaxScaler
from sklearn.preprocessing import Normalizer as SK_Normalizer
from sklearn.preprocessing import MaxAbsScaler as SK_MaxAbsScaler
from sklearn.preprocessing import RobustScaler as SK_RobustScalar
from sklearn.preprocessing import StandardScaler as SK_StandardScaler


class BaseScaler(ABC):
    def __init__(self):
        self._kwargs = {}

    @abstractmethod
    def fit(self, X, y=None):
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, X, y=None):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X, y=None):
        raise NotImplementedError

    def copy(self):
        return deepcopy(self)

    def to_dict(self):
        return {
            "name": self.__class__.__name__,
            "kwargs": self._kwargs
        }


class DummyScaler(BaseScaler):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return None

    def fit_transform(self, X, y=None):
        return X

    def transform(self, X, y=None):
        return X

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._kwargs == other._kwargs
        return False


class MinMaxScaler(SK_MinMaxScaler, BaseScaler):
    def __init__(self, feature_range: tuple[int, int] = (0, 1)):
        super().__init__(feature_range=feature_range, copy=False, clip=False)
        self._kwargs = {'feature_range': feature_range}

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._kwargs == other._kwargs
        return False


class Normalizer(SK_Normalizer, BaseScaler):
    def __init__(self, norm: Literal["l1", "l2", "max"] = "l2"):
        super().__init__(norm=norm, copy=False)
        self._kwargs = {'norm': norm}

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._kwargs == other._kwargs
        return False


class MaxAbsScaler(SK_MaxAbsScaler, BaseScaler):
    def __init__(self):
        super().__init__(copy=False)
        self._kwargs = {}

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._kwargs == other._kwargs
        return False


class RobustScaler(SK_RobustScalar, BaseScaler):
    def __init__(self, with_centering: bool = True, with_scaling: bool = True,
                 quantile_range: tuple[int, int] = (25.0, 75.0)):
        super().__init__(with_centering=with_centering, with_scaling=with_scaling,
                         quantile_range=quantile_range, copy=False)
        self._kwargs = {'with_centering': with_centering, "with_scaling": with_scaling,
                        "quantile_range": quantile_range}

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._kwargs == other._kwargs
        return False


class StandardScaler(SK_StandardScaler, BaseScaler):
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        super().__init__(with_mean=with_mean, with_std=with_std, copy=False)
        self._kwargs = {'with_mean': with_mean, "with_std": with_std}

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._kwargs == other._kwargs
        return False


def get_scaler(name: str or None, **kwargs):
    if name not in globals().keys():
        return DummyScaler()
    return globals()[name](**kwargs)
