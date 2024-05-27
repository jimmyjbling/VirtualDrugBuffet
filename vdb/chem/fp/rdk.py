from vdb.base import compile_step
from vdb.chem.fp.base import RDKitFPFunc, BinaryFPFunc, RdkitWrapper


@compile_step
class RDK(RDKitFPFunc, BinaryFPFunc):
    """
    The FP calculation when generating RDK fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self):
        super().__init__(**{"fpSize": 2048})

        self._func = RdkitWrapper("RDKFingerprint", "rdkit.Chem", **self._kwargs)
        self._dimension = 2048
