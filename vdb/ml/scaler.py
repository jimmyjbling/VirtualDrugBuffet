from abc import ABC, abstractmethod
from typing import Literal

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler as SK_MinMaxScaler
from sklearn.preprocessing import Normalizer as SK_Normalizer
from sklearn.preprocessing import MaxAbsScaler as SK_MaxAbsScaler
from sklearn.preprocessing import RobustScaler as SK_RobustScalar
from sklearn.preprocessing import StandardScaler as SK_StandardScaler

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
class DummyScaler(BaseEstimator, TransformerMixin, BaseScaler):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, **fit_params):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y)

    def transform(self, X, y=None):
        return X


@compile_step
class MinMaxScaler(SK_MinMaxScaler, BaseScaler):
    def __init__(self, feature_range: tuple[int, int] = (0, 1)):
        super().__init__(feature_range=feature_range, copy=False, clip=False)


@compile_step
class Normalizer(SK_Normalizer, BaseScaler):
    def __init__(self, norm: Literal["l1", "l2", "max"] = "l2"):
        super().__init__(norm=norm, copy=False)
        self._kwargs = {'norm': norm}


@compile_step
class MaxAbsScaler(SK_MaxAbsScaler, BaseScaler):
    def __init__(self):
        super().__init__(copy=False)
        self._kwargs = {}


@compile_step
class RobustScaler(SK_RobustScalar, BaseScaler):
    def __init__(self, with_centering: bool = True, with_scaling: bool = True,
                 quantile_range: tuple[int, int] = (25.0, 75.0)):
        super().__init__(with_centering=with_centering, with_scaling=with_scaling,
                         quantile_range=quantile_range, copy=False)
        self._kwargs = {'with_centering': with_centering, "with_scaling": with_scaling,
                        "quantile_range": quantile_range}


@compile_step
class StandardScaler(SK_StandardScaler, BaseScaler):
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        super().__init__(with_mean=with_mean, with_std=with_std, copy=False)
        self._kwargs = {'with_mean': with_mean, "with_std": with_std}


def get_scaler(name: str or None, **kwargs):
    if name not in globals().keys():
        return DummyScaler()
    return globals()[name](**kwargs)
