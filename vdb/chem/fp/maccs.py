from vdb.base import compile_step
from vdb.chem.fp.base import RDKitFPFunc, BinaryFPFunc, RdkitWrapper


@compile_step
class MACCS(RDKitFPFunc, BinaryFPFunc):
    """
    The FP calculation when generating MACCS fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.

    """

    def __init__(self, use_tqdm: bool = False):
        super().__init__(**{"use_tqdm": use_tqdm})

        self._func = RdkitWrapper("GetMACCSKeysFingerprint", "rdkit.Chem.rdMolDescriptors", **self._kwargs)
        self._dimension = 167
