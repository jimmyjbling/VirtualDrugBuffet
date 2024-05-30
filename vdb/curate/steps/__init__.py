from .add_3d import CurateAdd3D
from .add_hydrogen import CurateAddH
from .boron import CurateBoron
from .canonicalize import CurateCanonicalize
from .duplicate import (CurateRemoveDuplicates, CurateRemoveDuplicatesGreedy,
                        CurateRemoveDisagreeingDuplicatesCategorical, CurateRemoveDisagreeingDuplicatesStd,
                        CurateRemoveDisagreeingDuplicatesMinMax)
from .flatten import CurateFlatten
from .valid_mol import CurateValid
from .inorganic import CurateInorganic
from .mixture import CurateMixtures, CurateDemix
from .label import CurateRemoveMissingLabel, CurateMakeNumericLabel
from .neutralize import CurateNeutralize
from .sanitize import CurateSanitize
from .mw import CurateMW
from .base import CurationStepError, CurationStep

DEFAULT_CURATION_STEPS = [CurateValid(), CurateCanonicalize()]


def get_curation_step(name: str):
    if name not in globals().keys():
        raise CurationStepError("cannot find curation step {name}".format(name=name))
    return globals()[name]


__all__ = [
    "CurationStep",
    "CurationStepError",
    "CurateMixtures",
    "CurateDemix",
    "CurateInorganic",
    "CurateBoron",
    "CurateValid",
    "CurateAddH",
    "CurateAdd3D",
    "CurateFlatten",
    "CurateSanitize",
    "CurateNeutralize",
    "CurateCanonicalize",
    "CurateRemoveDuplicates",
    "CurateRemoveDisagreeingDuplicatesCategorical",
    "CurateRemoveDisagreeingDuplicatesStd",
    "CurateRemoveDisagreeingDuplicatesMinMax",
    "CurateRemoveDuplicatesGreedy",
    "CurateRemoveMissingLabel",
    "CurateMakeNumericLabel",
    "CurateMW",
    "get_curation_step",
    "DEFAULT_CURATION_STEPS"
]
