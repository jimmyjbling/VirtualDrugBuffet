import importlib
import pickle
from typing import Callable

import numpy as np
import numpy.typing as npt

from rdkit.rdBase import BlockLogs
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from vdb.base import Step

from vdb.chem.utils import to_mol
from vdb.utils import isnan


def _handle_fail_nan(fp_func: Callable, dimensions: int = 1, *args, **kwargs):
    """
    Wraps an FP func and returns a vector of `np.nan` if it fails for any reason

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
    except Exception:  # just means function failed for some reason, so give it a None
        return [np.nan]*dimensions


class BaseFPFunc(BaseEstimator, TransformerMixin, Step):
    """
    Base class for all FP functions used in the ml pipeline
    It is a child of the sklearn BaseEstimator and TransformerMixin to make it compatible with the `Pipeline` API
    The `fit` function does nothing, while `fit_transform` and `transform` just wrap `generate_fps`

    Parameters
    ----------
    use_tqdm : bool, default: False
        have a tqdm task to track progress of fingerprint generation
    kwargs : dict
        dictionary of keyword arguments to pass to the fingerprint function in from {`argument`: `value`}

    Attributes
    ----------
    _func : functools.partial object
        The callable FP function instance as a partial with static setting arguments (e.g., 'radius') pre-set
    _dimension : int
        the dimensionality of the fingerprints that will be generated
    _kwargs : dict
        the keyword arguments passed to the fingerprint functions (e.g. `nBits` or `radius`)

    Notes
    -----
    When declaring a child of the `BaseFPFunc` class, the `_func`, `_dimension` and `_binary` attributes must be set
    during instantiation of the child.
    """
    def __init__(self, use_tqdm: bool = False, **kwargs):
        self.use_tqdm = use_tqdm
        self._kwargs: dict = kwargs
        self._func: Callable or None = None
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

    def turn_on_tqdm(self):
        self.use_tqdm = True

    def turn_off_tqdm(self):
        self.use_tqdm = False

    def is_binary(self) -> bool:
        return isinstance(self, BinaryFPFunc)

    def get_dimension(self) -> int:
        """
        Gets dimensionality of the Fingerprint

        Returns
        -------
        int
        """
        return self._dimension

    def generate_fps(self, smis: str or list[str], nan_on_fail: bool = True, return_mask: bool = False) \
            -> npt.NDArray or tuple[npt.NDArray, npt.NDArray]:
        """
        Generate Fingerprints for a set of smiles

        Parameters
        ----------
        smis : str or list[str]
            the SMILES (or multiple SMILES) you want to generate a fingerprint(s) for
        nan_on_fail : bool, default: True
            if True will return `np.nan` for a given SMILES if any exception is raised during fp generation
        return_mask: bool, default False
            return a boolean mask of SMILES that FPs were successfully generated for
        Returns
        -------
        np.ndarray
            numerical array of fingerprint (filled with `np.nan` for any SMILES that failed)
        pass_bool: 1-D np.ndarray of bools
            an index-mapped 1-D bool array corresponding to whether the SMILES failed fp generation
        """
        _block = BlockLogs()
        smis = np.atleast_1d(smis)

        # loop through all the smiles and call the fp func
        if nan_on_fail:
            _tmp = [_handle_fail_nan(self._func, self._dimension, s) for s in
                    tqdm(smis, disable=not self.use_tqdm, desc="fingerprinting")]
        else:
            _tmp = [self._func(s) for s in tqdm(smis, disable=not self.use_tqdm, desc="fingerprinting")]

        _tmp = np.array(_tmp)

        mask = ~np.any(isnan(_tmp), axis=1)  # determine which fingerprints failed to generate

        if return_mask:
            return _tmp, mask
        else:
            return _tmp

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None, **kwargs):
        return self.transform(X, y)

    def transform(self, X, y=None):
        return self.generate_fps(X)[0]

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.__name__}_{i+1}" for i in range(self._dimension)])

    def save(self, file_path: str):
        pickle.dump(self, open(file_path, "wb"))


class BinaryFPFunc(BaseFPFunc):
    """
    An abstract class used to declare an FPFunc as generating a binary fingerprint output.
    This means all values in the fingerprint vector are either `0` or `1`.
    For example, an FPFunc wrapping the rdkit GetMorganFingerprintAsBitVec would be a `BinaryFPFunc`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DiscreteFPFunc(BaseFPFunc):
    """
    An abstract class used to declare an FPFunc as generating a discrete fingerprint output.
    This means all values in the fingerprint are whole numbers (can be `0` and negative).
    This is more general than the `BinaryFPFunc`.
    Even if all values are binary except for one which is not (e.g. `2`), it must be a `DiscreteFPFunc`.
    For example, an FPFunc wrapping the rdkit GetHashedMorganFingerprint would be a `DiscreteFPFunc`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ContinuousFPFunc(BaseFPFunc):
    """
    An abstract class used to declare an FPFunc as generating a continuous fingerprint output.
    This means all values in the fingerprint are real floats.
    This is more general than the `DiscreteFPFunc`.
    Even if all values are discrete except for one which is not (e.g. `2.342`), it must be a `ContinuousFPFunc`.
    For example, an FPFunc using the Mordred fingerprinting functions would be a ContinuousFPFunc
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ObjectFPFunc(BaseFPFunc):
    """
    An abstract class used to declare an FPFunc as generating an object-based fingerprint output.
    This means the fingerprinting function does not return the tradition vector of numbers.
    Instead, it returns some type of object (usually an instance of a custom class).
    This should only be used in cases where preprocessing of SMILES in non-standard ways is needed.
    It is most commonly used as the `FPFunc` for GCN or SmilesTransformer, which often require a multitude of inputs.
    For example, an FPFunc for generating the `MolGraph` objects needed for a GCN would be a child of `ObjectFPFunc`
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RDKitFPFunc(BaseFPFunc):
    """
    This is a parent class that all `FPFunc` that wrap any RDKit functions should inherit from.
    It simply gives a `FPFunc` the ability to generate a list of rdkit Vector objects instead of a numpy array
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_fps_as_rdkit_objects(self, smis: str or list[str]) -> list:
        """
        Generates a list of RDKit Vector object (type determined by function) rather than a numpy array of the
        fingerprint vectors.

        Parameters
        ----------
        smis: str or list[str]
            the SMILES to generate the vector objects for

        Returns
        -------
        list[rdkit.DataStruct objects]
            a index mapped list (one object per passed smiles) of the resulting rdkit Vector objects for the FPFunc
        """
        _block = BlockLogs()
        smis = np.atleast_1d(smis)
        return [self._func(m, as_list=False) if m else None for m in
                tqdm(smis, disable=not self.use_tqdm, desc="making rdkit fingerprints")]


class RdkitWrapper:
    """
    Pybind or otherwise non-python functions (like, for example, RDKit functions that are binding from C++)
    cannot be serialized by pickle, thus cannot be saved with a pickle dump.

    This is a problem since it VDB depends on being able to pickle functions to load later for reproducability.
    This wrapper gets around that issue by importing the required function on the fly at runtime, rather than needing
    to serialize the function for storage in the pickle itself

    Notes
    -----
    The only problem with this is that if RDKit versions changes and for some reason the FP function is altered,
     there would be no pickling error, as the function would just be imported.
    Luckily, the VDB Step concept compensates for this, by saving the version of RDKit that was used with the FPFunc.
    However, you will need to manully match this yourself (unless some script exists to check it for you in the future)

    Parameters
    ----------
    method_name: str
        the name of the method to use as a string; should be just the method name, no module
    module_name: str
        the full name of the module that the `method_name` is found in
    **kwargs:
        additional keyword arguments to be passed to the warpped function (if any)

    Examples
    --------
    >>> wrapped_func = RdkitWrapper('MolToSmiles', 'rdkit.Chem')
    >>> wrapped_fp_func = RdkitWrapper("GetHashedMorganFingerprint", "rdkit.Chem.AllChem", radius=2, nBits=512)

    """
    def __init__(self, method_name, module_name, **kwargs):
        self.method_name = method_name
        self.module = module_name
        self._kwargs = kwargs

    @property
    def method(self):
        """
        Import the requested function on the fly and returns it

        Returns
        -------
        function: Callable
            function requested defined by <self.module_name>.<self.method_name>

        """
        return getattr(importlib.import_module(self.module), self.method_name)

    def __call__(self, smi, as_list: bool = True):
        if as_list:
            return list(self.method(to_mol(smi), **self._kwargs))
        else:
            return self.method(to_mol(smi), **self._kwargs)

    @property
    def __name__(self):
        return self.method_name
