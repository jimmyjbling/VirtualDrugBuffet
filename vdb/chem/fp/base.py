import importlib
import pickle
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm

from vdb.chem.utils import to_mol
from vdb.utils import isnan


def _handle_fail_nan(fp_func: Callable, dimensions: int = 1, *args, **kwargs):
    """
    Wraps an FP func and returns a None if it fails

    Parameters
    ----------
    fp_func: Callable
        function to wrap
    *args:
        positional arguments to pass to the function
    **kwargs:
        keyword arguments to pass to the function

    Returns
    -------
    FPObject or None:
        The `fp_func` return(s) if it worked of a None if it failed
    """
    try:
        return fp_func(*args, **kwargs)
    except Exception as e:  # just means function failed for some reason, so give it a None
        return [np.nan]*dimensions


class BaseFPFunc(BaseEstimator):
    """
    Base class for all FP functions used in vdb
    It is a child of the sklearn BaseEstimator to make it compatible with the `Pipeline` API
    The `fit` function does nothing, while `fit_transform` and `transform` just wrap `generate_fps`

    Parameters
    ----------
    kwargs : dict
        dictionary of keyword arguments to pass to the fingerprint function in from {`argument`: `value`}

    Attributes
    ----------
    _func : functools.partial object
        The callable FP function instance as a partial with static setting arguments (e.g., 'radius') pre-set
    _dimension : int
        the dimensionality of the fingerprints that will be generated

    Notes
    -----
    When declaring a child of the `BaseFPFunc` class, the `_func`, `_dimension` and `_binary` attributes must be set
    during instantiation of the child.
    """
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._func: Callable = None
        self._dimension: int = -1

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            if self._kwargs == other._kwargs:
                return True
        return False

    @property
    def __name__(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return self.generate_fps(*args, **kwargs)

    def is_binary(self) -> bool:
        return isinstance(self, BinaryFPFunc)

    def get_dimension(self) -> int:
        return self._dimension

    def generate_fps(self, smis: str or list[str], use_tqdm: bool = False, nan_on_fail: bool = True):
        """
        Generate Fingerprints for a set of smi

        Parameters
        ----------
        smis : str or list[str]
            the SMILES (or multiple SMILES) you want to generate a fingerprint(s) for
        use_tqdm : bool, default: False
            have a tqdm task to track progress
        nan_on_fail : bool, default: True
            if True will return `np.nan` for a given SMILES if any exception is raised during fp generation

        Returns
        -------
        np.ndarray
            numerical array of fingerprint (filled with `np.nan` for any SMILES that failed)
        pass_bool: 1-D np.ndarray of bools
            an index-mapped 1-D bool array corresponding to whether the SMILES failed fp generation
        """
        smis = np.atleast_1d(smis)

        # loop through all the smiles and call the fp func
        if nan_on_fail:
            _tmp = [_handle_fail_nan(self._func, self._dimension, s) for s in
                    tqdm(smis, disable=not use_tqdm, desc="fingerprinting")]
        else:
            _tmp = [self._func(s) for s in tqdm(smis, disable=not use_tqdm, desc="fingerprinting")]

        _tmp = np.array(_tmp)

        pass_bool = ~np.any(isnan(_tmp), axis=1)  # determine which fingerprints failed to generate

        return _tmp, pass_bool

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X, y)

    def transform(self, X, y=None):
        return self.generate_fps(X)

    def save(self, file_path: str):
        pickle.dump(self, open(file_path, "wb"))


class BinaryFPFunc(BaseFPFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DiscreteFPFunc(BaseFPFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ContinuousFPFunc(BaseFPFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ObjectFPFunc(BaseFPFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RDKitFPFunc(BaseFPFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self._block = BlockLogs()

    def generate_fps_as_rdkit_objects(self, smis: str or list[str], use_tqdm: bool = False):
        smis = np.atleast_1d(smis)
        return [self._func(m, as_list=False) if m else None for m in
                tqdm(smis, disable=not use_tqdm, desc="making rdkit fingerprints")]


class RdkitWrapper:
    """
    Pybind or otherwise non-python functions (like, for example, RDKit functions that are binding from C++)
    cannot be serialized by pickle, thus cannot be saved with a pickle dump.

    This is a problem since it is much better for us to just dump to a pickle and then load it later.
    This wrapper gets around that issue by importing the required function on the fly at runtime, rather than needing
    to serialize the function for storage in the pickle itself
    """
    def __init__(self, method_name, module_name, **kwargs):
        self.method_name = method_name
        self.module = module_name
        self._kwargs = kwargs

    @property
    def method(self):
        return getattr(importlib.import_module(self.module), self.method_name)

    def __call__(self, smi, as_list: bool = True):
        if as_list:
            return list(self.method(to_mol(smi), **self._kwargs))
        else:
            return self.method(to_mol(smi), **self._kwargs)

    @property
    def __name__(self):
        return self.method_name
