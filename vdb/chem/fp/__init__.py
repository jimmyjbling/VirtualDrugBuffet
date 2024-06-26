from .ecfp import (
    ECFP4, ECFP6, BinaryECFP4, BinaryECFP6
)
from .fcfp import (
    FCFP4, FCFP6, BinaryFCFP4, BinaryFCFP6
)
from .atompair import (
    AtomPair, BinaryAtomPair
)
from .avalon import (
    Avalon, BinaryAvalon
)
from .toptor import (
    TopTor, BinaryTopTor
)
from .maccs import MACCS
from .rdk import RDK
from .base import BaseFPFunc


def get_fp_func(class_name: str, **kwargs) -> BaseFPFunc:
    _class = globals().get(class_name)
    if _class:
        return _class(**kwargs)
    raise FPFuncError(f"cannot find FPFunc {class_name}")


class FPFuncError(Exception):
    pass


__all__ = [
    "BaseFPFunc",
    "ECFP4",
    "BinaryECFP4",
    "ECFP6",
    "BinaryECFP6",
    "FCFP4",
    "BinaryFCFP4",
    "FCFP6",
    "BinaryFCFP6",
    "AtomPair",
    "BinaryAtomPair",
    "Avalon",
    "BinaryAvalon",
    "TopTor",
    "BinaryTopTor",
    "MACCS",
    "RDK",
    "get_fp_func",
    "FPFuncError"
]
