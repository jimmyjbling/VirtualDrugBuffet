from vdb.chem.fp.base import RDKitFPFunc, BinaryFPFunc, DiscreteFPFunc, RdkitWrapper
from vdb.base import compile_step


@compile_step
class AtomPair(RDKitFPFunc, DiscreteFPFunc):
    """
    The FP calculation when generating AtomPair fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self):
        super().__init__(**{"nBits": 2048})

        self._func = RdkitWrapper("GetHashedAtomPairFingerprint", "rdkit.Chem.rdMolDescriptors", **self._kwargs)
        self._dimension = 2048


@compile_step
class BinaryAtomPair(RDKitFPFunc, BinaryFPFunc):
    """
    The FP calculation when generating Binary AtomPair fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self):
        super().__init__(**{"nBits": 2048})

        self._func = RdkitWrapper("GetHashedAtomPairFingerprintAsBitVect", "rdkit.Chem.rdMolDescriptors", **self._kwargs)
        self._dimension = 2048
