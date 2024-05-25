import numpy as np
from tqdm import tqdm

from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.SimDivFilters import rdSimDivPickers

from vdb.chem.fp import BinaryFCFP4
from vdb.chem.utils import generate_scaffold


def cluster_scaffold(smis):
    _scaffolds = {}
    _ticker = 0
    _idxs = []
    for smi in smis:
        _scaffold = generate_scaffold(smi)
        _idx = _scaffolds.get(_scaffold, None)
        if _idx is None:
            _scaffolds[_scaffold] = _ticker
            _idx = _ticker
            _ticker += 1
        _idxs.append(_idx)
    return _idxs


def cluster_leader(smis, thresh: float = 0.65, use_tqdm: bool = False):
    _fps = BinaryFCFP4().generate_fps_as_rdkit_objects(smis)
    lp = rdSimDivPickers.LeaderPicker()

    _centroids = lp.LazyBitVectorPick(_fps, len(_fps), thresh)
    _centroid_fps = [_fps[i] for i in _centroids]

    _cluster_ids = []
    for _fp in tqdm(_fps, disable=not use_tqdm, desc="assigning SMILES to clusters"):
        sims = BulkTanimotoSimilarity(_fp, _centroid_fps)
        _cluster_ids.append(np.argmax(sims))
    return _cluster_ids
