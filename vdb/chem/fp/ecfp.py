from vdb.base import compile_step
from vdb.chem.fp.base import RDKitFPFunc, BinaryFPFunc, DiscreteFPFunc, RdkitWrapper


@compile_step
class ECFP4(RDKitFPFunc, DiscreteFPFunc):
    """
    The FP calculation when generating ECFP4 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """
    def __init__(self, use_tqdm: bool = False):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": False, "use_tqdm": use_tqdm})

        self._func = RdkitWrapper("GetHashedMorganFingerprint", "rdkit.Chem.AllChem", **self._kwargs)
        self._dimension = 2048


@compile_step
class ECFP6(RDKitFPFunc, DiscreteFPFunc):
    """
    The FP calculation when generating ECFP6 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self, use_tqdm: bool = False):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": False, "use_tqdm": use_tqdm})

        self._func = RdkitWrapper("GetHashedMorganFingerprint", "rdkit.Chem.AllChem", **self._kwargs)
        self._dimension = 2048


@compile_step
class BinaryECFP4(RDKitFPFunc, BinaryFPFunc):
    """
    The FP calculation when generating Binary FCFP4 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self, use_tqdm: bool = False):
        super().__init__(**{"radius": 4, "nBits": 2048, "useFeatures": False, "use_tqdm": use_tqdm})

        self._func = RdkitWrapper("GetMorganFingerprintAsBitVect", "rdkit.Chem.AllChem", **self._kwargs)
        self._dimension = 2048


@compile_step
class BinaryECFP6(RDKitFPFunc, BinaryFPFunc):
    """
    The FP calculation when generating Binary FCFP6 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self, use_tqdm: bool = False):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": False, "use_tqdm": use_tqdm})

        self._func = RdkitWrapper("GetMorganFingerprintAsBitVect", "rdkit.Chem.AllChem", **self._kwargs)
        self._dimension = 2048
