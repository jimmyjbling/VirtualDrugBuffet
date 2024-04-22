import numpy as np


def to_list(obj):
    if isinstance(obj, list):
        return obj
    elif hasattr(obj, 'to_list'):
        return obj.to_list()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, str):
        return [obj]
    elif not hasattr(obj, "__iter__"):
        return [obj]
    else:
        return list(obj)


def isnan(arr):
    """
    Once again, the numpy devs fail me and have a very arbitrary yet somehow strict definition of nan,
    such that objects are both somehow not nan but also nan.

    This is the equivalent of np.isnan, except made to also be friendly towards arrays of object/string dtype.
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        if arr.dtype == object:
            is_nan = np.array([1 if (isinstance(x, float) and np.isnan(x)) else 0 for x in arr.ravel()], dtype=bool)
            return is_nan.reshape(arr.shape)
        if arr.dtype.kind in "fci":  # Only [f]loats and [c]omplex numbers and [i]ntegers can be NaN
            return np.isnan(arr)
        return np.zeros(arr.shape, dtype=bool)

    else:
        if isinstance(arr, float):
            return np.isnan(arr)
        else:
            return True
