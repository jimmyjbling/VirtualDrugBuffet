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


