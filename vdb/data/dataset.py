import numpy as np
import numpy.typing as npt

from rdkit.Chem import Mol

from vdb.data.base import _BaseDataset, _BaseSmiles, _BaseLabeled, SmilesVector, MolVector, NameVector, LabelVector


class SmilesDataset(_BaseDataset, _BaseSmiles):
    def __init__(self, smiles: SmilesVector or list[str] or npt.NDArray,
                 names: NameVector or list[str] or npt.NDArray = None,
                 mols: MolVector or list[Mol] or npt.NDArray = None,
                 generate_mols: bool = False):
        """
        Initialize a SmilesDataset2D object for storing and operating on SMILES datasets

        Parameters
        ----------
        smiles: SmilesVector or list[str]
            the SMILES defining all the chemicals in the dataset
        names: NameVector or list[str], optional
            the ids to associate with each chemical.
            if passed must match the length of smiles, and is index mapped
            if not passed, names defaults to the numerical index of each chemical
        mols: MolVector or list[Mol], optional
            the Mols for the respective smiles, if precomputed.
            will be ignored if `generate_mols` is True
            must match length of smiles, and is index mapped
        generate_mols: bool = False
            whether to create rdkit mol objects for each smile when dataset is created
            for large datasets that will not need rdkit mols setting this to True can slow things down
        """

        super().__init__(self)
        self._mols = None

        self._valid_desc_func = ["smiles", "mol", "any"]

        self._smiles_col = "SMILES"
        self._name_col = "Name"
        self._mol_col = "ROMol"

        self._smiles = smiles if isinstance(smiles, SmilesVector) else SmilesVector(smiles)  # make smiles vector

        # generate mols if asked
        if generate_mols and (mols is None):
            self._mols = self._smiles.to_mols()
        else:
            self._mols = mols if isinstance(mols, MolVector) else MolVector(mols)

        # generate name vector (if needed)
        if names is None:
            self._names = NameVector(list(range(len(self._smiles))))
        else:
            self._names = names if isinstance(names, NameVector) else NameVector(names)

        # check that name, smiles and mols are all the same length
        if len(self._names) != len(self._smiles):
            raise ValueError(f"length of name and smiles mismatched: "
                             f"got {len(self._names)} names and {len(self._smiles)} smiles")

        if len(self._names) != len(self._smiles):
            raise ValueError(f"length of mols and smiles mismatched: "
                             f"got {len(self._mols)} mols and {len(self._smiles)} smiles")

    # python class implements
    def __len__(self):
        return len(self._smiles)

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"cannot add {other.__class__} to 'SmilesDataset' class")
        else:
            return SmilesDataset(smiles=self._smiles + other._smiles,
                                 names=self._names + other._names,
                                 mols=None if ((other._mols is None) or (self._mols is None))
                                 else self._mols + other._mols)

    def __iter__(self):
        if self._mols is None:
            for smi, name in zip(self._smiles, self._names):
                yield smi, None, name
        else:
            for smi, mol, name, fp in zip(self._smiles, self._mols, self._names):
                yield smi, mol, name

    # SMILES functions
    def get_smiles(self):
        return self._smiles.get_smiles()

    def get_smiles_batches(self, batch_size: int):
        for i in range(0, len(self), batch_size):
            yield self._smiles[i:i + batch_size]

    # rdkit Mol functions
    def _check_mols(self):
        if self._mols is None:
            self.make_mols()

    def get_mols(self):
        self._check_mols()
        return self._mols.get_mols()

    def get_mol_batches(self, batch_size: int):
        self._check_mols()
        if self._mols:
            for i in range(0, len(self), batch_size):
                yield self._mols[i:i + batch_size]
        else:
            yield None

    def make_mols(self):
        self._mols = self._smiles.to_mols()

    def del_mols(self):
        self._mols = None

    # name functions
    def get_names(self):
        return self._names.get_names()

    def get_name_batches(self, batch_size):
        for i in range(0, len(self), batch_size):
            yield self._names[i:i + batch_size]

    # generic dataset functions
    def get(self):
        if self._mols is None:
            return self._smiles.get_smiles(), [None]*len(self), self._names.get_names()
        else:
            return self._smiles.get_smiles(), self._mols.get_mols(), self._names.get_names()

    def get_batches(self, batch_size: int):
        for i in range(0, len(self), batch_size):
            if self._mols is None:
                yield self._smiles[i:i + batch_size], [None]*batch_size, self._names[i:i + batch_size]
            else:
                yield self._smiles[i:i + batch_size], self._mols[i:i + batch_size], self._names[i:i + batch_size]

    # conversion functions
    def add_label(self, labels: LabelVector or list[str or int or float] or npt.NDArray):
        if len(labels) != len(self):
            raise ValueError(f"length of labels does not match length of dataset: {len(labels)}, {len(self)}")
        else:
            return LabeledSmilesDataset(smiles=self._smiles, mols=self._mols, names=self._names,
                                        labels=labels, generate_mols=False)

    def curate(self, curation_workflow: CurationWorkflow):
        return CuratedSmilesDataset(smiles=self._smiles, names=self._names, mols=self._mols,
                                    curation_workflow=curation_workflow)

    def to_csv(self, file_loc, header: bool = True, delimiter: str = ","):
        open(file_loc, "w").write(self.to_csv_str(header=header, delimiter=delimiter))

    def to_csv_str(self, header: bool = True, delimiter: str = ","):
        _str = ""
        if header:
            _str += f"SMILES{delimiter}Name\n"
        for smi, _, name in self:
            _str += f"{smi}{delimiter}{name}\n"
        return _str


class LabeledSmilesDataset(SmilesDataset, _BaseLabeled):
    def __init__(self, smiles: SmilesVector or list[str] or npt.NDArray,
                 labels: LabelVector or list[str or int or float] or npt.NDArray,
                 names: NameVector or list[str] or npt.NDArray = None,
                 mols: MolVector or list[Mol] or npt.NDArray = None,
                 generate_mols: bool = True):
        """
        Initialize a LabeledSmilesDataset object for storing and operating on SMILES datasets with labels attached
        to the chemicals

        Parameters
        ----------
        smiles: SmilesVector or list[str] or NDArray
            the SMILES defining all the chemicals in the dataset
        labels: LabelVector or list[str or int or float] or NDArray
            the labels associated with each chemical. Can be numerical or categorical
        names: NameVector or list[str] or NDArray, optional
            the ids to associate with each chemical.
            if passed must match the length of chemicals and is index mapped
            if not passed, names defaults to the numerical index of each chemical
        mols: MolVector or list[Mol], optional
            the Mols for the respective smiles, if precomputed. Will be ignored if `generate_mols` is True
            if passed must match the length of chemicals and is index mapped
        generate_mols: bool = True
            whether to create rdkit mol objects for each smile when dataset is created
            for large datasets that will not need rdkit mols setting this to True can slow things down
        """

        super().__init__(smiles=smiles, names=names, mols=mols, generate_mols=generate_mols)
        self._labels = labels if isinstance(labels, LabelVector) else LabelVector(labels)

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"cannot add {other.__class__} to 'LabeledSmilesDataset' class")
        else:
            return LabeledSmilesDataset(smiles=self._smiles + other._smiles,
                                        labels=self._labels + other._labels,
                                        names=self._names + other._names,
                                        mols=None if ((other._mols is None) or (self._mols is None))
                                        else self._mols + other._mols)

    def __iter__(self):
        if self._mols is None:
            for smi, name, label in zip(self._smiles, self._names, self._labels):
                yield smi, None, name, label
        else:
            for smi, mol, name, label in zip(self._smiles, self._mols, self._names, self._labels):
                yield smi, mol, name, label

    # label functions
    def get_labels(self):
        return self._labels.get_labels()

    def get_label_batches(self, batch_size: int):
        for i in range(0, len(self), batch_size):
            yield self._labels[i:i + batch_size]

    def subset_on_label(self, threshold: float, greater: bool = True):
        _picks = self._labels.to_binary(threshold, greater=greater, inplace=False).to_array().astype(bool)
        return self.__class__(
            smiles=self._smiles.to_array()[_picks],
            labels=self._labels.to_array()[_picks],
            names=self._names.to_array()[_picks],
            mols=self._mols.to_array()[_picks] if self._mols is not None else None
        )

    # override parent get with new gets
    def get(self):
        if self._mols is None:
            return self._smiles, MolVector([None] * len(self)), self._names, self._labels,
        else:
            return self._smiles, self._mols, self._names, self._labels

    def get_batches(self, batch_size: int):
        for i in range(0, len(self), batch_size):
            if self._mols is None:
                yield (self._smiles[i:i + batch_size], MolVector([None] * batch_size), self._names[i:i + batch_size],
                       self._labels[i:i + batch_size])
            else:
                yield (self._smiles[i:i + batch_size], self._mols[i:i + batch_size], self._names[i:i + batch_size],
                       self._labels[i:i + batch_size])

    # conversion functions
    def curate(self, curation_workflow: CurationWorkflow):
        return CuratedLabeledSmilesDataset(smiles=self._smiles, names=self._names, labels=self._labels,
                                           mols=self._mols, curation_workflow=curation_workflow)

    def to_csv(self, file_loc, header: bool = True, delimiter: str = ","):
        f = open(file_loc, "w").write(self.to_csv_str(header=header, delimiter=delimiter))

    def to_csv_str(self, header: bool = True, delimiter: str = ","):
        _str = ""
        if header:
            _str += f"SMILES{delimiter}Name{delimiter}Label\n"
        for smi, _, name, label, _ in self:
            _str += f"{smi}{delimiter}{name}{delimiter}{label}\n"
        return _str
