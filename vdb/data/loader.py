import warnings
from typing import Union, Optional

from vdb.data.dataset import SmilesDataset, LabeledSmilesDataset
from vdb.data.base import _BaseLoader, _BaseSmiles, _BaseLabeled, NameVector, SmilesVector, LabelVector

from vdb.chem.utils import is_smi, to_mols


class SmilesLoader(_BaseLoader, _BaseSmiles):
    def __init__(
            self,
            file_path: str,
            delimiter: str = "\t",
            smiles_col: Union[int, str] = 0,
            name_col: Optional[Union[int, str]] = None,
            header: bool = False
    ):
        """
        Initialize a SmilesLoader

        Loaders are the main way to go from a file to a Dataset object.
        Think of them as parsers.
        They can generate the SMILES objects from these files, either in batches or all at once

        They have all the functionality of a Dataset, but don't load chem into memory.
        This has upsides and downsides:
            upside: reading batches on the fly then discarding is very memory efficient, large files can be used
            downside: the loader explicitly saves nothing, so each time you want the SMILES column you need to
                      re-read the file. This can be time-consuming if the code is not written with this in mind

        This loader requires that the datafile has a SMILES column.
        Optionally, you can specify a column that contains names (think ids) for the SMILES.
        These should be unique, but the loader will not check this explicitly.
        If names are not unique, there could be issues with chem loss or duplications when merging smiles in the future

        Parameters
        ----------
        file_path: str
            location of file to read
            file must have a delimited column containing SMILES
        delimiter: str = "\t"
            delimiter used to separated column in file
        smiles_col: int or str
            if int, the column idx that contains the SMILES
            if str, the name of the column that contains the SMILES (requires header=True)
        name_col: int or str or None
            if int, the column idx that contains the name
            if str, the name of the column that contains the names (requires header=True)
            if None, the names will initialize to
        header: bool = True
            does the file contain a header
            even if using index number to declare smiles_col, needs to be set to True if there is a header otherwise
            header will be read in as chemical

        Returns
        ----------
        SmilesLoader
            the loader linked to this file
        """
        super().__init__(file_path=file_path, header=header, delimiter=delimiter)

        # file parse properties
        self._delimiter = delimiter
        self._split = lambda x: [_.strip() for _ in x.split(delimiter)]

        try:
            smiles_col = int(smiles_col)
        except ValueError:
            pass

        if isinstance(smiles_col, int):
            if len(self._col_names) <= smiles_col:
                raise IndexError(f"detected {len(self._col_names)} columns but got smiles idx of {smiles_col}")
        elif header:
            if smiles_col not in self._col_names:
                raise ValueError(f"smiles_col '{smiles_col}' not in file header")
            smiles_col = self._col_names.index(smiles_col)
        else:
            raise ValueError(f"if header is false, the 'smiles_col' must be column index as int got '{smiles_col}'")
        self._smiles_col = smiles_col

        self._test_for_smiles()  # some quick checks to make raise warnings if SMILES don't look right

        # deal with name_col
        if name_col is None:
            pass
        else:
            with open(self._file_path, "r") as f:
                _first_line = f.readline()
            self._col_names = [_.strip() for _ in self._split(_first_line)]
            if isinstance(name_col, int):
                if len(self._col_names) <= name_col:
                    raise IndexError(f"detected {len(self._col_names)} columns but got name_col idx of {name_col}")
            elif header:
                if name_col not in self._col_names:
                    raise ValueError(f"name_col '{name_col}' not in file header")
                name_col = self._col_names.index(name_col)
            else:
                raise ValueError(f"if header is false 'name_column' must be column index as int got '{name_col}'")
        self._name_col = name_col

    def _test_for_smiles(self):
        """
        internal function to check is column contains valid smiles, or if 'header' was set wrong
        throws warning if detected header when header was false or if it found lots of bad smiles in the first 10 rows
        """
        _first_not_smile = self._header
        bad_smiles = 0
        with open(self._file_path, "r") as f:
            for i, line in enumerate(f):
                smi = self._split(line)[self._smiles_col].strip()
                if i == 10:
                    break
                _is_smiles = is_smi(smi)
                if i == 0 and not _is_smiles:
                    _first_not_smile = True
                else:
                    if not _is_smiles:
                        bad_smiles += 1
        if self._header and (not _first_not_smile):
            warnings.warn("detected no header when 'header' was set to True; check file for header")
        if ((not self._header) and _first_not_smile) and bad_smiles == 0:
            warnings.warn("detected a header when 'header' was set to False; check file for header")
        if bad_smiles > min(i, 5):
            warnings.warn(f"detected {bad_smiles} non-smiles value in first {9 if self._header else 10} "
                          f"entries of smiles column; check smiles col")

    def _load(self, iterator, generate_mols):
        smiles = []
        names = [] if self._name_col else None
        for line in iterator:
            smiles.append(line[self._smiles_col])
            if names:
                names.append(line[self._name_col])

        return SmilesDataset(smiles=SmilesVector(smiles),
                             names=NameVector(names) if names else None,
                             generate_mols=generate_mols)

    def load(self, generate_mols: bool = False):
        return self._load(iterator=self, generate_mols=generate_mols)

    def load_batches(self, batch_size: int, generate_mols: bool = False):
        for batch in self.get_batches(batch_size):
            yield self._load(iterator=batch, generate_mols=generate_mols)

    def get_smiles(self):
        """
        Reads the file in and returns all the SMILES

        Returns
        -------
        smiles: array-like
            shape (num_chemicals, 1)
        names: array-like (optional)
            shape (num_chemicals, 1)
        """
        return [_[self._smiles_col] for _ in self]

    def get_smiles_batches(self, batch_size: int):
        """
        Read in the file as batches of a given number of chemicals; Will yield each batch, reading
        and removing from memory as it progresses.

        Yields batches of SMILES

        Parameters
        ----------
        batch_size: int
            the number of chemicals in each returned batch

        Returns
        -------
        smiles: array-like
            shape (batch_size, 1)
        names: array-like (optional)
            shape (num_chemicals, 1)
        """
        for _batch in self.get_batches(batch_size):
            yield [_[self._smiles_col] for _ in _batch]

    def get_names(self):
        """
        Read in the whole file into memory and return the names for each chemical

        Returns
        -------
        names: array-like
            shape (num_chemicals, 1)
        """
        return [_[self._name_col] for _ in self]

    def get_name_batches(self, batch_size: int):
        """
        Read in the file as batches of a give number of chemicals; Will yield each batch, reading
        and removing from memory as it progresses

        Yields batches of Names

        Parameters
        ----------
        batch_size: int
            the number of chemicals in each returned batch

        Returns
        -------
        names: array-like
            shape (batch_size, 1)
        """
        for _batch in self.get_batches(batch_size):
            yield [_[self._name_col] for _ in _batch]

    def get_mols(self):
        """
        Read in the whole file into memory and return the mols for each chemical

        Returns
        -------
        mols: array-like[ROMol]
            shape (num_chemicals, 1)
        """

        return to_mols(self.get_smiles())

    def get_mol_batches(self, batch_size: int):
        """
        Read in the file as batches of a given number of chemicals; Will yield each batch, reading
        and removing from memory as it progresses.

        Yields batches of Mols

        Parameters
        ----------
        batch_size: int
            the number of chemicals in each returned batch

        Returns
        -------
        smiles: array-like
            shape (batch_size, 1)
        names: array-like (optional)
            shape (num_chemicals, 1)
        """
        for _smiles in self.get_smiles_batches(batch_size):
            yield to_mols(_smiles)

    def get_smiles_col(self):
        return self._smiles_col

    def get_name_col(self):
        return self._name_col


class LabeledSmilesLoader(SmilesLoader, _BaseLabeled):
    def __init__(self,
                 file_path: str,
                 delimiter: str = "\t",
                 smiles_col: Union[int, str] = 0,
                 name_col: Optional[Union[int, str]] = None,
                 label_col: Optional[Union[int, str]] = None,
                 header: bool = False):
        super().__init__(file_path, delimiter=delimiter, smiles_col=smiles_col, name_col=name_col, header=header)

        self._label_col = label_col

        if isinstance(self._label_col, int):
            if len(self._col_names) <= self._label_col:
                raise IndexError(f"detected {len(self._col_names)} columns but got label_col idx of {self._label_col}")
        elif header:
            if self._label_col not in self._col_names:
                raise ValueError(f"label_col '{self._label_col}' not in file header")
            self._label_col = self._col_names.index(label_col)
        else:
            raise ValueError(f"if header is false, the label_col must be column index of labels "
                             f"location as int got '{self._label_col}'")

    def _load(self, iterator, generate_mols):
        smiles = []
        labels = []
        names = [] if self._name_col else None
        for line in iterator:
            smiles.append(line[self._smiles_col])
            labels.append(line[self._label_col])
            if names:
                names.append(line[self._name_col])

        return LabeledSmilesDataset(smiles=SmilesVector(smiles),
                                    labels=LabelVector(labels),
                                    names=NameVector(names) if names else None,
                                    generate_mols=generate_mols)

    def load(self, generate_mols: bool = False):
        return self._load(iterator=self, generate_mols=generate_mols)

    def load_batches(self, batch_size: int, generate_mols: bool = False):
        for batch in self.get_batches(batch_size):
            yield self._load(iterator=batch, generate_mols=generate_mols)

    def label_col_name(self):
        return self._col_names[self._label_col]

    def get_labels(self):
        return [_[self._label_col] for _ in self]

    def get_label_batches(self, batch_size: int):
        for _batch in self.get_batches(batch_size):
            yield [_[self._label_col] for _ in self]
