from vdb.chem.fp.base import RDKitFPFunc, BinaryFPFunc, DiscreteFPFunc, RdkitWrapper


class TopTor(RDKitFPFunc, DiscreteFPFunc):
    """
    The FP calculation when generating TopologicalTorsion fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self):
        super().__init__(**{"nBits": 2048})

        self._func = RdkitWrapper("GetHashedTopologicalTorsionFingerprint", "rdkit.Chem.AllChem", **self._kwargs)
        self._dimension = 2048


class BinaryTopTor(RDKitFPFunc, BinaryFPFunc):
    """
    The FP calculation when generating Binary TopologicalTorsion fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self):
        super().__init__(**{"nBits": 2048})

        self._func = RdkitWrapper("GetHashedTopologicalTorsionFingerprintAsBitVect", "rdkit.Chem.AllChem", **self._kwargs)
        self._dimension = 2048
