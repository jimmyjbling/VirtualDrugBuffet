import abc
import enum


class CurationIssue(enum.Enum):
    mixture = "compound is a mixture"
    inorganic = "compound is inorganic"
    boron = "compound has boron"
    rdkit_failed = "rdkit failed to read molecule"
    failed_adding_Hs = "rdkit failed to add H to molecule"
    failed_gen_random_3d = "rdkit failed to generated a random 3d pose for molecule"
    flatten_failed = "compound failed to be flattened"
    sanitize_failed = "compound failed to be sanitized"
    neutralize_failed = "compound failed to be neutralized"
    canonical_failed = "compound failed to be canonicalized"
    duplicate = "compound is duplicate"
    disagreeing_duplicate = "compound has duplicate with disagreeing label"
    missing_label = "label value is missing"
    non_numeric_label = "label value is not numeric"


class CurationNote(enum.Enum):
    flattened = "compound flattened"
    sanitized = "compound sanitized"
    neutralized = "compound neutralized"
    canonical = "compound made canonical"
    added_Hs = "explict H added to compound"
    added_random_3d_conformer = "3d conformer added to compound"
    label_made_numeric = "label made numeric"
    label_made_binary = "label made binary"


class CurationFunction(abc.ABC):
    def __init__(self, **kwargs):
        self.uses = None  # type of chemical object (smiles or mol) used for curation
        self.returns = None  # type of object returned by function (if any) (smiles, mol, label)
        self.static = True  # the chemical object is changed as a result of this curation; includes inplace mol updates
        self.func_args = kwargs
        self.issue = None  # the issue that should be used if curation function fails
        self.note = None  # the note that should be used if curation function acts on molecule
        self.labeled: bool = False  # whether the curation function needs labels to run

    @abc.abstractmethod
    def _func(self, chemicals, labels=None, **kwargs):
        raise NotImplemented

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs, **self.func_args)

    @staticmethod
    @abc.abstractmethod
    def get_rank():
        raise NotImplemented
