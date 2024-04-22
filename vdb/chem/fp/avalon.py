from vdb.chem.fp.base import RDKitFPFunc, BinaryFPFunc, DiscreteFPFunc, RdkitWrapper


class Avalon(RDKitFPFunc, DiscreteFPFunc):
    """
    The FP calculation when generating Avalon fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self):
        super().__init__(**{"nBits": 2048})

        self._func = RdkitWrapper("GetAvalonCountFP", "rdkit.Avalon.pyAvalonTools", **self._kwargs)
        self._dimension = 2048


class BinaryAvalon(RDKitFPFunc, BinaryFPFunc):
    """
    The FP calculation when generating Binary Avalon fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self):
        super().__init__(**{"nBits": 2048})

        self._func = RdkitWrapper("GetAvalonFP", "rdkit.Avalon.pyAvalonTools", **self._kwargs)
        self._dimension = 2048
