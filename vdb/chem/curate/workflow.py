from collections import Counter
from datetime import datetime

import numpy as np

from vdb.chem.curate.base import CurationIssue, CurationNote
from vdb.chem.curate.steps import CURATION_LITERAL_TO_FUNC, CURATION_NAME_TO_ISSUE_ENUM
from vdb.chem.utils import to_smis
from vdb.data.base import SmilesVector, MolVector, LabelVector
from vdb.logger import setup_logger

DEFAULT_CURATION_STEPS = ["missing_label", "rdkit", 'canonicalize']

PRIORITY_RANKING = {
    ("smiles", "smiles", False): 0,
    ("smiles", None, True): 1,
    ("label", "label"): 1,
    ("numeric_label", "label"): 2,
    "rdkit": 2,
    ("smiles", "mol", False): 3,
    ("mol", "mol", False): 4,
    ("mol", None, False): 4,
    ("mol", None, True): 5,
    ("use_label", False): 6,
    ("use_label", True): 7,
    ("mol", "smiles", True): 8,  # '8' steps should result in smiles that if processed by smiles use is the same output
    ("mol", "smiles", False): 8,
    "dup1": 9,
    "dup2": 10
}

CURATION_PRIORITY = {"rdkit": 2}
CURATION_PRIORITY.update({key: PRIORITY_RANKING[val.get_rank()] for key, val in CURATION_LITERAL_TO_FUNC.items()})


class CurationDictionary:
    def __init__(self, size: int, curation_steps: list):
        self._curr_curate_dict = {}
        self._curr_step = curation_steps
        self._size = size

        self._note_keys = None
        self._issue_keys = None
        self._note_values = None
        self._issue_values = None

    def __getitem__(self, item):
        return self._curr_curate_dict[item]

    def __setitem__(self, key, value):
        self._curr_curate_dict[key] = value

    def __len__(self):
        return self._size

    def update(self, other: dict):
        for key, val in other.items():
            if key in self._curr_curate_dict.keys():
                self._curr_curate_dict[key].extend(val)
            else:
                self._curr_curate_dict[key] = val
        self._issue_keys = [key for key, val in self.items() if any([isinstance(_, CurationIssue) for _ in val])]
        self._note_keys = [key for key, val in self.items() if any([isinstance(_, CurationNote) for _ in val])]
        self._note_values = [self._curr_curate_dict[k] for k in self.note_keys()]
        self._issue_values = [self._curr_curate_dict[k] for k in self.issue_keys()]

    def items(self):
        return self._curr_curate_dict.items()

    def keys(self):
        return self._curr_curate_dict.keys()

    def issue_keys(self):
        return self._issue_keys

    def note_keys(self):
        return self._note_keys

    def values(self):
        return self._curr_curate_dict.values()

    def issue_values(self):
        return self._issue_values

    def get_issue(self, key):
        if key in self.issue_keys():
            return [_ for _ in self._curr_curate_dict[key] if isinstance(_, CurationIssue)]
        else:
            return None

    def note_values(self):
        return self._note_values

    def get_note(self, key):
        if key in self.note_keys():
            return [_ for _ in self._curr_curate_dict[key] if isinstance(_, CurationNote)]
        else:
            return None

    def get_failed(self):
        return np.array(self.issue_keys()).astype(int)

    def get_passed(self):
        _tmp = np.ones(self._size).astype(int)
        _tmp[self.get_failed()] = 0
        return np.arange(self._size)[_tmp.astype(bool)]

    def get_steps(self):
        return self._curr_step

    def generate_report(self):
        return CurationReport(self)


class CurationReport:
    def __init__(self, cur_dict: CurationDictionary = None):
        self._steps = cur_dict.get_steps() if cur_dict else None
        self._note_counter = Counter([__ for _ in cur_dict.note_values() for __ in _
                                      if isinstance(__, CurationNote)]) if cur_dict else Counter()
        self._issue_counter = Counter([_[0] for _ in cur_dict.issue_values()]) if cur_dict else Counter()
        self._num_noted = len(cur_dict.note_keys()) if cur_dict else 0
        self._num_failed = len(cur_dict.get_failed()) if cur_dict else 0
        self._num_passed = len(cur_dict.get_passed()) if cur_dict else 0
        self._size = len(cur_dict) if cur_dict else 0

    def to_file(self, file_path: str, append: bool = False):
        if append:
            open(file_path, "a").write("\n"+str(self))
        else:
            open(file_path, "w").write(str(self))

    def __str__(self):
        report = f"curated {self._size} compounds\n\n###NOTES###\n\n"

        for cur_note_key in self._note_counter.keys():
            report += f"altered {self._note_counter[cur_note_key]} compounds due to '{cur_note_key.value}'\n"

        report += f"\n{self._num_noted} compounds altered during curation\n\n###ISSUES###\n\n"

        for step in self._steps:
            if CURATION_NAME_TO_ISSUE_ENUM[step] in self._issue_counter.keys():
                report += (f"removed {self._issue_counter[CURATION_NAME_TO_ISSUE_ENUM[step]]} after step {step} due to "
                           f"'{CURATION_NAME_TO_ISSUE_ENUM[step].value}'\n")

        report += f"\nremoved {self._num_failed} compounds during curation\n"
        report += f"{self._num_passed} compounds passed curation\n"

        return report

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"cannot add object of class {other.__class__} to CurationReport")
        if (self._steps is not None) and (set(self._steps) != set(other._steps)):
            raise ValueError("cannot add CurationReports together when step do not match")
        self._note_counter = self._note_counter + other._note_counter
        self._issue_counter = self._issue_counter + other._issue_counter
        self._num_noted = self._num_noted + other._num_noted
        self._num_failed = self._num_failed + other._num_failed
        self._num_passed = self._num_passed + other._num_passed
        self._size = self._size + other._size
        if self._steps is None:
            self._steps = other._steps


class CurationWorkflow:
    def __init__(self, curation_steps=None, optimize_order: bool = True, override_defaults: bool = False,
                 do_logging: bool = False):
        if curation_steps is None:
            curation_steps = DEFAULT_CURATION_STEPS
        if not isinstance(curation_steps, list):
            curation_steps = [curation_steps]

        self._logger = setup_logger("curation-workflow", dummy=(not do_logging))

        if not override_defaults:
            if "canonicalize" not in curation_steps:
                self._logger.debug("'canonicalize' curation step missing, adding to workflow")
                curation_steps = curation_steps + ["canonicalize"]
            else:
                self._logger.debug("moving 'canonicalize' curation step to end of workflow")
                curation_steps = [_ for _ in curation_steps if _ != "canonicalize"] + ["canonicalize"]

            if "rdkit" not in curation_steps:
                self._logger.debug("'rdkit' curation step missing, adding to workflow")
                curation_steps = ["rdkit"] + curation_steps
            else:
                self._logger.debug("moving 'rdkit' curation step to start of workflow")
                curation_steps = ["rdkit"] + [_ for _ in curation_steps if _ != "rdkit"]

        self._logger.info(f"curation workflow set to {curation_steps}")

        if optimize_order:
            self._logger.info("optimizing curation workflow order")
            _ranked_steps = [(step, CURATION_PRIORITY[step]) for step in curation_steps
                             if step in CURATION_PRIORITY.keys()]
            _ranked_steps.sort(key=lambda x: x[1])
            curation_steps = [_[0] for _ in _ranked_steps]

        self._curation_steps = {}
        for curation_step in curation_steps:
            if curation_step not in CURATION_LITERAL_TO_FUNC.keys():
                self._logger.error(f"unrecognized curation step; skipping this step")
            else:
                self._curation_steps[curation_step] = CURATION_LITERAL_TO_FUNC[curation_step]()

    def run_workflow(self, smiles=None, mols=None, labels=None):
        t0 = datetime.now()

        # some argument checking
        if smiles is None and mols is None:
            raise ValueError("smiles and mols cannot both be 'None'")

        if (labels is None) and self.has_label_steps():
            raise ValueError("cannot use curation workflow with label based step when labels is 'None'")

        if smiles is None:
            self._logger.debug("generating smiles from mols")
            smiles = to_smis(mols)
            skip_rdkit = True
        else:
            if mols is not None:
                self._logger.warning("over-ridding passed mols with mols generated from passed smiles")
            skip_rdkit = False

        curation_dict = CurationDictionary(size=len(smiles), curation_steps=list(self._curation_steps.keys()))

        for step_name, step_func in self._curation_steps.items():
            self._logger.debug(f"running curation step {step_name}")
            if step_name == "rdkit" and skip_rdkit:
                self._logger.debug("skipping explict rdkit curation step; mols already exist")
                bad_idx = np.where(np.vectorize(lambda x: x is None)(mols) > 0)[0]
                _curr_dict = {_: [CurationIssue.rdkit_failed] for _ in bad_idx}
                curation_dict.update(_curr_dict)
                continue

            good_idx, bad_idx, returns = step_func(smiles=smiles, mols=mols, labels=labels)
            _curr_dict = {_: [step_func.issue] for _ in bad_idx}
            if step_func.note is not None:
                _curr_dict.update({_: [step_func.note] for _ in good_idx})

            if step_func.returns == "smiles":
                self._logger.debug(f"overriding smiles with output of step {step_name}")
                smiles = returns
            elif step_func.returns == "mol":
                self._logger.debug(f"overriding mols with output of step {step_name}")
                mols = returns
            elif step_func.returns == "label":
                self._logger.debug(f"overriding labels with output of step {step_name}")
                labels = returns
            curation_dict.update(_curr_dict)
        self._logger.info(f"curation workflow finished in {str(datetime.now() - t0)}")
        self._logger.info(f"{len(curation_dict.issue_values())} out of {len(smiles)} datapoints failed curation")
        return SmilesVector(smiles), MolVector(mols), LabelVector(labels), curation_dict

    def get_steps(self):
        return list(self._curation_steps.keys())

    def has_label_steps(self):
        return any([_.labeled for _ in self._curation_steps.values()])

    def __str__(self):
        return " -> ".join(list(self._curation_steps.keys()))

    def save_workflow(self, out_loc: str):
        with open(out_loc, "w") as f:
            f.write(str(self))


def load_workflow(workflow_file: str, logging: bool = False):
    steps = [_.strip() for _ in open(workflow_file, "r").readline().split(" -> ")]
    return CurationWorkflow(curation_steps=steps, optimize_order=False, do_logging=logging)
