from vdb.base import compile_step
from vdb.chem.fp.base import RDKitFPFunc, BinaryFPFunc, DiscreteFPFunc, RdkitWrapper


@compile_step
class FCFP4(RDKitFPFunc, DiscreteFPFunc):
    """
    The FP calculation when generating FCFP4 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": True})

        self._func = RdkitWrapper("GetHashedMorganFingerprint", "rdkit.Chem.AllChem", **self._kwargs)
        self._dimension = 2048


@compile_step
class FCFP6(RDKitFPFunc, DiscreteFPFunc):
    """
    The FP calculation when generating FCFP6 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": True})

        self._func = RdkitWrapper("GetHashedMorganFingerprint", "rdkit.Chem.AllChem", **self._kwargs)
        self._dimension = 2048


@compile_step
class BinaryFCFP4(RDKitFPFunc, BinaryFPFunc):
    """
    The FP calculation when generating Binary FCFP4 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self):
        super().__init__(**{"radius": 4, "nBits": 2048, "useFeatures": True})

        self._func = RdkitWrapper("GetMorganFingerprintAsBitVect", "rdkit.Chem.AllChem", **self._kwargs)
        self._dimension = 2048


@compile_step
class BinaryFCFP6(RDKitFPFunc, BinaryFPFunc):
    """
    The FP calculation when generating Binary FCFP6 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": True})

        self._func = RdkitWrapper("GetMorganFingerprintAsBitVect", "rdkit.Chem.AllChem", **self._kwargs)
        self._dimension = 2048
