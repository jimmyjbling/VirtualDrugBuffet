import abc
import os
from copy import deepcopy

import numpy as np

from vdb.utils import to_list
from vdb.chem.utils import is_mol, to_mol, to_smi


class _BaseVector:
    def __init__(self, items: list):
        self._items = to_list(items)
        self._numpy_cached = np.atleast_1d(self._items)

    def _update_cache(self):
        self._numpy_cached = np.atleast_1d(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._numpy_cached[idx]

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"cannot add object of type {type(other)} to {self.__class__}")
        return self.__class__(self._items + other._items)

    def __iter__(self):
        for n in self._items:
            yield n

    def __delitem__(self, key):
        del self._items[key]
        self._update_cache()

    def __copy__(self):
        return deepcopy(self)

    def copy(self):
        return self.__copy__()

    def to_array(self):
        return self._numpy_cached.copy()

    def to_list(self):
        return self._items


class NameVector(_BaseVector):
    def __init__(self, names: list):
        super().__init__(names)
        self._items = [str(_) for _ in self._items]
        self._update_cache()

    def get_names(self):
        return self._numpy_cached


class SmilesVector(_BaseVector):
    def __init__(self, smiles: list):
        super().__init__(smiles)

        if not all([(_ is None) or isinstance(_, str) for _ in self._items]):
            raise ValueError(f"{self.__class__} can only contain 'None' or 'str' objects")

    def get_smiles(self):
        return self._numpy_cached

    def to_mols(self):
        return MolVector([to_mol(smi) for smi in self])


class MolVector(_BaseVector):
    def __init__(self, mols):
        super().__init__(mols)

        if not all([(_ is None) or is_mol(_) for _ in self._items]):
            raise ValueError(f"{self.__class__} can only contain 'None' or rdkit 'Mol' objects")

    def get_mols(self):
        return self._numpy_cached

    def to_smiles(self):
        return SmilesVector([to_smi(m) for m in self])


class LabelVector(_BaseVector):
    def __init__(self, labels):
        super().__init__(labels)

    def get_labels(self):
        return self._numpy_cached

    def is_binary(self):
        if set(self._items) == {0, 1}:
            return True
        return False

    def to_binary(self, threshold: float or int, greater: bool = True, inplace: bool = False):
        _tmp = self.to_numeric(inplace=False)
        _tmp = [1 if label > threshold else 0 if greater else 0 if label > threshold else 1 for label in _tmp]
        if inplace:
            self._items = _tmp
            self._update_cache()
        else:
            return LabelVector(_tmp)

    def is_numeric(self):
        """
        Determines whether the label array contains values that can be cast to numeric (float).

        Notes
        -----
        Will still return True if the array is of a non-numeric type but cast-able to numeric

        Returns
        -------
        bool
            True if all values can be cast to numeric else False
        """

        # check if already bool (b) unsigned int (u) signed int (i) or float (f)
        if self._numpy_cached.dtype.kind in 'buif':
            return True

        # else try and cast to numeric
        try:
            self._numpy_cached.astype(float)
        except ValueError:
            return False
        return True

    def to_numeric(self, inplace: bool = False):
        # special cases if already a numeric type
        if self._numpy_cached.dtype.kind in 'uif':
            if inplace:
                return
            return self.copy()

        if inplace:
            self._numpy_cached = self._numpy_cached.astype(float)
            self._items = self._numpy_cached.tolist()
        else:
            return LabelVector(self._numpy_cached.astype(float).tolist())


class _Base(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def get_names(self):
        raise NotImplemented

    @abc.abstractmethod
    def get_name_batches(self, batch_size):
        raise NotImplemented


class _BaseSmiles(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def get_smiles_batches(self, batch_size):
        raise NotImplemented

    @abc.abstractmethod
    def get_smiles(self):
        raise NotImplemented

    @abc.abstractmethod
    def get_mol_batches(self, batch_size):
        raise NotImplemented

    @abc.abstractmethod
    def get_mols(self):
        raise NotImplemented


class _BaseLabeled(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._labels = None

    @abc.abstractmethod
    def get_label_batches(self, batch_size):
        raise NotImplemented

    @abc.abstractmethod
    def get_labels(self):
        raise NotImplemented

    def make_binary_labels(self, threshold: float, greater: bool = True):
        try:
            _labels = self._labels.astype(float)
        except ValueError:
            raise ValueError("cannot convert all labels into numeric; use curation to fix")
        self._labels = _labels > threshold if greater else _labels < threshold

    def make_discrete_labels(self, threshold):
        threshold = np.atleast_1d(threshold).flatten()
        try:
            _labels = self._labels.astype(float)
        except ValueError:
            raise ValueError("cannot convert all labels into numeric; use curation to fix")
        threshold.sort()
        self._labels = np.digitize(self._labels, threshold)


class _BaseLoader(_Base, abc.ABC):
    def __init__(self, file_path: str, delimiter: str = ",", header: bool = None):
        super().__init__(self)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"cannot find file {file_path}")
        else:
            self._file_path = file_path

        self._delimiter = delimiter
        self._header = header

        self._col_names = self.get_header() if header else None
        self._num_fields = len(self._col_names) if self._col_names else len(self.get_header())

    def get_header(self):
        return open(self._file_path, "r").readline().strip().split(self._delimiter)

    def from_where(self):
        return self._file_path

    def get_batches(self, batch_size: int):
        _batch_lines = []
        for i, line in enumerate(self):
            _batch_lines.append(line)
            if ((i + 1) % batch_size) == 0:
                yield _batch_lines
                _batch_lines = []

        if len(_batch_lines) > 0:
            yield _batch_lines

    def __iter__(self):
        with open(self._file_path, "r") as f:
            if self._header:
                f.readline()
            for line in f:
                yield line.strip().split(self._delimiter)

    def __len__(self):
        num_lines = 0
        with open(self._file_path, "r") as f:
            for _ in f:
                num_lines += 1
        return num_lines - (1 if self._header else 0)

    @abc.abstractmethod
    def load(self):
        raise NotImplemented

    @abc.abstractmethod
    def load_batches(self, batch_size):
        raise NotImplemented


class _BaseDataset(_Base, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(self)

    @abc.abstractmethod
    def get(self):
        raise NotImplemented

    @abc.abstractmethod
    def get_batches(self, batch_size):
        raise NotImplemented

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplemented
