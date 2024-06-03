from vdb.base import compile_step
from vdb.chem.fp.base import RDKitFPFunc, BinaryFPFunc, DiscreteFPFunc, RdkitWrapper


@compile_step
class Avalon(RDKitFPFunc, DiscreteFPFunc):
    """
    The FP calculation when generating Avalon fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self, use_tqdm: bool = False):
        super().__init__(**{"nBits": 2048, "use_tqdm": use_tqdm})

        self._func = RdkitWrapper("GetAvalonCountFP", "rdkit.Avalon.pyAvalonTools", **self._kwargs)
        self._dimension = 2048


@compile_step
class BinaryAvalon(RDKitFPFunc, BinaryFPFunc):
    """
    The FP calculation when generating Binary Avalon fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self, use_tqdm: bool = False):
        super().__init__(**{"nBits": 2048, "use_tqdm": use_tqdm})

        self._func = RdkitWrapper("GetAvalonFP", "rdkit.Avalon.pyAvalonTools", **self._kwargs)
        self._dimension = 2048
