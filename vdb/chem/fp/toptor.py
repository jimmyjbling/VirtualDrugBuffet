from vdb.base import compile_step
from vdb.chem.fp.base import RDKitFPFunc, BinaryFPFunc, DiscreteFPFunc, RdkitWrapper


@compile_step
class TopTor(RDKitFPFunc, DiscreteFPFunc):
    """
    The FP calculation when generating TopologicalTorsion fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self, use_tqdm: bool = False):
        super().__init__(**{"nBits": 2048, "use_tqdm": use_tqdm})

        self._func = RdkitWrapper("GetHashedTopologicalTorsionFingerprint", "rdkit.Chem.AllChem", **self._kwargs)
        self._dimension = 2048


@compile_step
class BinaryTopTor(RDKitFPFunc, BinaryFPFunc):
    """
    The FP calculation when generating Binary TopologicalTorsion fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self, use_tqdm: bool = False):
        super().__init__(**{"nBits": 2048, "use_tqdm": use_tqdm})

        self._func = RdkitWrapper("GetHashedTopologicalTorsionFingerprintAsBitVect", "rdkit.Chem.AllChem", **self._kwargs)
        self._dimension = 2048
