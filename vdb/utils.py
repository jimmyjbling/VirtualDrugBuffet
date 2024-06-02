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


def isnan(arr: object):
    """
    Once again, the numpy and data science devs fail me and have established NaN as only usable for floats, yet somehow
     also masking it the defacto standard for missing values in a numpy array.
    This is the equivalent of np.isnan, except made to also be friendly towards arrays of object/string dtype.

    Will return True if when an element is a nan (float), None, 'nan', 'null', 'None', 'na' or '' (empty string)
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        if arr.dtype == object:
            is_nan = np.array([1 if ((isinstance(x, float) and np.isnan(x)) or
                                     (isinstance(x, str)) and x.lower() in ["nan", "na", "null", "none", ""]) or
                                    (x is None)
                               else 0 for x in arr.ravel()], dtype=bool)
            return is_nan.reshape(arr.shape)
        if arr.dtype.kind == "U":
            is_nan = np.array([1 if x.lower() in ["nan", "na", "null", "none", ""]
                               else 0 for x in arr.ravel()], dtype=bool)
            return is_nan.reshape(arr.shape)
        if arr.dtype.kind in "fci":  # Only [f]loats and [c]omplex numbers and [i]ntegers can be NaN
            return np.isnan(arr)
        return np.zeros(arr.shape, dtype=bool)
    else:
        if hasattr(arr, "tolist"):
            arr = arr.tolist()
        if isinstance(arr, (float, int)):
            return np.isnan(arr)
        elif isinstance(arr, str):
            return arr.lower() in ["nan", "na", "null", "none", ""]
        else:
            return False
