from collections import Counter
import datetime

import numpy as np
import numpy.typing as npt

from sklearn.base import TransformerMixin, BaseEstimator

from .notes import CurationNote
from .issues import CurationIssue
from .steps import DEFAULT_CURATION_STEPS, CurationStep, get_curation_step

from vdb.logger import setup_logger
from vdb.utils import to_list


class CurationWorkflowError(Exception):
    pass


class CurationWorkflow(BaseEstimator, TransformerMixin):
    """
    A SciKit Learn compatible curation transformer

    This is useful for taking a series of CurationSteps and linking them together. CurationSteps do not transform data,
     they only return boolean mask and alter molecules in place to standardize them. CurationWorkflow can merge these
     step such that it will take in a list of SMILES and output either a set of rdkit.Mol or SMILES that passed curation

    Since it is compatible with SciKit Learn, it can be used inside a Pipeline for easy workflow design. See
    `vdb.ml.MoleculeModel` for more details

    Will also do some logging of curation step results and generate a report that tracks which chemicals failed which
     steps. See `CurationReport` for more details on the report

    Notes
    -----
    All CurationWorkflows MUST start with a CurateValid step, as this will convert the SMILES into mols. The
     CurationWorkflow class operates assuming you only give it SMILES, so its default is to get upset when
     CurateValid is missing. On the off chance that you already have a list of Mols that you want to use, you can adjust
     this behavior by setting `use_mols` to True during construction. If you plan on using SMILES but forgot to
     include the CurateValid step, no worries! The default behavior of CurationWorkflow sets `correct_broken` to True,
     which will fix this issue for you (but log a warning to let you know if did so)

    All CurationStep have an optimal order to be run in (See CurationStep for more details). This is to avoid unexpected
     behavior and possible crashes. For example, if you de-duplicate before you neutralized compounds you could end up
     with identical compounds in your output. Others, like CurateRemoveDisagreeingDuplicates require specific dataset
     traits, like numeric labels. If you didn't make you labels numeric, CurateRemoveDisagreeingDuplicates could fail

    To help compensate for dependency, all CurationSteps include a list of dependencies that must occur before a given
     step (if there are any). The workflow will fail to initialize if it detects this dependency tree has been violated.
     However, `correct_broken` will tell the CurationWorkflow to fix any issues like this. For more details on
     CurationStep dependency see CurationStep docs
             TODO: This is missing the ability to have dependencies in the same rank class. Currently not an issue, but
              could become one in the future

    Parameters
    ----------
    steps: list[CurationStep, ...], default `None`
        a list of curation steps that should be taken
    optimize_order: bool, default `False`
        sort the curation steps into an optimal order
        RECOMMEND if the order is not optimized can result in unhandled exceptions
    correct_broken: bool, default `True`
        add in the minimum required steps to allow the workflow to run
        currently this is the "CurateValid" step
        RECOMMEND if missing CurateValid at the start will cause all other curation steps to fail
    do_logging: bool, default = True
        turn on (True) or off (False) logging
    report_path: str, default = None
        where to write the curation report file
        if left as None will not write a curation report
    """

    def __init__(self, steps: list[CurationStep] = None, optimize_order: bool = True, correct_broken: bool = True,
                 use_mols: bool = False, do_logging: bool = True, report_path: str = None):
        super().__init__()

        self._steps = to_list(steps) if steps is not None else DEFAULT_CURATION_STEPS
        self._report = CurationReport()
        self._logger = setup_logger("curation") if do_logging else setup_logger(dummy=True)

        self._report_path = report_path

        _missing_deps = set.union(*[_step.missing_dependency(self._steps) for _step in self._steps])

        # handle dependencies first
        if len(_missing_deps) > 0:
            if correct_broken:
                self._logger.warning(f"workflow has unmet CurationStep dependanies: {_missing_deps}")
                self._steps = self._steps + [get_curation_step(_name)() for _name in _missing_deps]
                self._logger.info(f"added steps {_missing_deps} step to curation workflow")
            else:
                raise CurationWorkflowError(f"workflow has unmet dependanies: {_missing_deps}")

        if optimize_order:
            # sort the steps by their rank; lower values should come first (e.g. 1 is first, 2 is next...)
            self._steps = [_[1] for _ in sorted([(_s.get_rank(), _s) for _s in self._steps], key=lambda x: x[0])]

        # make sure that CurateValid is the first step if not using mols
        if not use_mols:
            if self._steps[0].get_rank() != 1:
                if correct_broken:
                    self._logger.warning("missing `CurateValid` step in curation workflow")
                    self._steps = [get_curation_step("CurateValid")()] + self._steps
                    self._logger.info("added `CurateValid` step to curation workflow")
                else:
                    raise CurationWorkflowError("cannot run `CurationWorkflow` without `CurateValid` step")
        else:  # ignore the `CurateValid step if it is not needed
            self._logger.debug("use_mol mode is active")
            if self._steps[0].get_rank() == 1:
                self._steps = self._steps[1:]
                self._logger.warning(f"ignoring `CurateValid` step")

    def run_workflow(self, X: npt.NDArray, y: npt.NDArray):
        _start = datetime.datetime.now()
        _overall_mask = np.ones(len(X)).astype(bool)
        self._report.set_size(len(X))
        for _step in self._steps:
            _mask, X, y = _step(X=X, y=y)
            self._logger.info(f"{len(X) - np.sum(_mask)} compounds failed step {str(_step)}")
            if self._report_path:  # don't waste time with reporting if we don't need it
                self._report.add_step(_step, _mask)
            _overall_mask = np.all((_overall_mask, _mask), axis=0)
        _stop = datetime.datetime.now()
        self._logger.info(f"{np.sum(_overall_mask)} out of {len(X)} compounds passed curation workflow")
        self._report.write_report(self._report_path, time_delta=_stop - _start)  # write out the report
        return X[_overall_mask], y[_overall_mask]

    def fit(self, X, y, **kwargs):
        pass

    def transform(self, X, y):
        return self.run_workflow(X, y)[0]


class CurationReport:
    def __init__(self):
        self._dictionary = CurationDictionary()
        self._size = -1

    def set_size(self, size: int):
        self._size = size

    def add_step(self, step: CurationStep, mask: npt.NDArray):
        _bad_chemical_idx = np.where(~mask)[0]

        if step.note:
            for _idx in range(len(mask)):
                self._dictionary.add(_idx, note=step.note)

        if step.issue:
            for _idx in _bad_chemical_idx:
                self._dictionary.add(_idx, issue=step.issue)
        self._dictionary.update_step_order(step)

    def write_report(self, path: str or None, time_delta: datetime.timedelta):
        if path is None:
            return

        _issue_counter, _num_removed = self._dictionary.gather_issue_counter()
        _note_counter, _num_altered = self._dictionary.gather_note_counter()
        report = (f"TOTAL PASSING CURATION: {self._size - _num_removed} out of {self._size}\n"
                  f"time taken: {str(time_delta)}\n\n###ISSUES###\n\n")

        for key in self._dictionary.get_step_order():
            report += f"removed {_issue_counter.get(key.issue, 0)} compounds due to '{key}'\n"

        report += f"\nremoved {_num_removed} compounds during curation\n\n###NOTES###\n\n"

        for key in self._dictionary.get_step_order():
            if key.note is not None:
                report += f"altered {_note_counter.get(key.note, 0)} compounds due to '{key}'\n"

        report += f"\naltered {_num_altered} compounds during curation\n"

        open(path, "w").write(report)


class CurationDictionary:
    def __init__(self):
        self._curr_curate_dict = {}

        self._step_order = []

        self._note_keys = set()
        self._issue_keys = set()

    def __getitem__(self, item):
        return self._curr_curate_dict[item]

    def __setitem__(self, key, value):
        self._curr_curate_dict[key] = value

    def add(self, idx: int, issue: CurationIssue = None, note: CurationNote = None):
        if issue:
            self._curr_curate_dict.setdefault(idx, []).append(issue)
            self._issue_keys.add(idx)
        if note:
            self._curr_curate_dict.setdefault(idx, []).append(note)
            self._note_keys.add(idx)

    def get_step_order(self) -> list[CurationStep]:
        return self._step_order

    def update_step_order(self, step: CurationStep):
        self._step_order.append(step)

    def items(self):
        return self._curr_curate_dict.items()

    def keys(self):
        return self._curr_curate_dict.keys()

    def values(self):
        return self._curr_curate_dict.values()

    def issue_keys(self):
        return self._issue_keys

    def note_keys(self):
        return self._note_keys

    def get_issues(self, idx: int):
        return [_ for _ in self._curr_curate_dict.get(idx, []) if isinstance(_, CurationIssue)]

    def get_notes(self, idx: int):
        return [_ for _ in self._curr_curate_dict.get(idx, []) if isinstance(_, CurationNote)]

    def gather_issue_counter(self) -> tuple[Counter, int]:
        _issue_counter = Counter()
        for key in self.issue_keys():
            _issue_counter.update(self.get_issues(key)[:1])  # get first issue only
        return _issue_counter, len(self.issue_keys())

    def gather_note_counter(self) -> tuple[Counter, int]:
        _note_counter = Counter()
        _number_alter = 0
        for key in self.note_keys():
            if len(self.get_issues(key)) > 1:
                continue
            _note_counter.update(self.get_notes(key))
            _number_alter += 1
        return _note_counter, _number_alter
