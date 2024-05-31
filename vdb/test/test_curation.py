import unittest

import numpy as np

from rdkit.Chem import MolFromSmiles, MolToSmiles, Mol
from rdkit.rdBase import BlockLogs

from vdb.curate import *

_BLOCKER = BlockLogs()

TEST_SMILES = []
TEST_LABELS = []
TEST_MOLS = []

with open("./data/BBBP.csv", "r") as f:
    _header = f.readline()
    for line in f:
        splits = line.strip().split(",")
        TEST_SMILES.append(splits[3])
        TEST_LABELS.append(int(splits[2]))
        TEST_MOLS.append(MolFromSmiles(splits[3]))


DUMMY_DUP_SMILES = ["CCCC", "CCCC", "CCCC", "CCCC", "CCO", "CCCO", "CCCC", "CCO", "CCO"]
DUMMY_DUP_MOLS = [MolFromSmiles(_) for _ in DUMMY_DUP_SMILES]
DUMMY_DUP_LABELS = [1, 1, 1, 1, 4.4, 2.1, 0, np.nan, 100]
DUMMY_CAT_LABELS = ["Y", "Y", "Y", "Y", "0", "Y", "0", "0", np.nan]


class TestCurationWorkflow(unittest.TestCase):
    def test_workflow(self):
        workflow = CurationWorkflow(DEFAULT_CURATION_STEPS, do_logging=False)

    def test_missing_deps(self):
        with self.assertRaises(CurationWorkflowError):
            workflow = CurationWorkflow(DEFAULT_CURATION_STEPS+[CurateRemoveDisagreeingDuplicatesStd()],
                                        correct_broken=False, do_logging=False)

    def test_fix_missing_deps(self):
        workflow = CurationWorkflow([CurateValid(), CurateRemoveDisagreeingDuplicatesStd()], do_logging=False)
        self.assertEquals(workflow._steps[1], CurateMakeNumericLabel())

    def test_optimize_order(self):
        workflow = CurationWorkflow([CurateValid(), CurateCanonicalize(), CurateDemix()], do_logging=False)
        self.assertEqual(workflow._steps[0], CurateValid())
        self.assertEqual(workflow._steps[-1], CurateCanonicalize())

    def test_valid_not_first(self):
        with self.assertRaises(CurationWorkflowError):
            workflow = CurationWorkflow([CurateCanonicalize(), CurateDemix()], correct_broken=False, do_logging=False)
        workflow = CurationWorkflow([CurateCanonicalize(), CurateDemix()], correct_broken=False, use_mols=True, do_logging=False)

    def test_fix_missing_valid(self):
        workflow = CurationWorkflow([CurateCanonicalize(), CurateDemix()], correct_broken=True, do_logging=False)
        self.assertEqual(workflow._steps[0], CurateValid())
        workflow = CurationWorkflow([CurateCanonicalize(), CurateDemix()], correct_broken=True, use_mols=True, do_logging=False)
        self.assertNotEqual(workflow._steps[0], CurateValid())

    def test_requires_y(self):
        workflow = CurationWorkflow([CurateCanonicalize(), CurateDemix(), CurateMakeNumericLabel()], do_logging=False)
        self.assertTrue(workflow._requires_y)
        workflow = CurationWorkflow([CurateCanonicalize(), CurateDemix()], do_logging=False)
        self.assertFalse(workflow._requires_y)

    def test_run(self):
        workflow = CurationWorkflow(DEFAULT_CURATION_STEPS, do_logging=False)
        X, y, mask = workflow.run_workflow(np.array(TEST_SMILES), np.array(TEST_LABELS))
        self.assertNotEqual(len(X), len(TEST_SMILES))
        self.assertEqual(mask.sum(), len(X))
        workflow = CurationWorkflow(DEFAULT_CURATION_STEPS, use_mols=True, do_logging=False)
        X, y, mask = workflow.run_workflow(np.array(TEST_MOLS), np.array(TEST_LABELS))
        self.assertNotEqual(len(X), len(TEST_MOLS))
        self.assertEqual(mask.sum(), len(X))

    def test_report(self):
        import os
        workflow = CurationWorkflow(DEFAULT_CURATION_STEPS, do_logging=False, report_path="./TEST_REPORT.txt")
        X, y, mask = workflow.run_workflow(np.array(TEST_SMILES), np.array(TEST_LABELS))
        self.assertTrue(os.path.exists("./TEST_REPORT.txt"))
        os.remove("./TEST_REPORT.txt")

    def test_report_counting(self):
        workflow = CurationWorkflow(DEFAULT_CURATION_STEPS, do_logging=False)
        X, y, mask = workflow.run_workflow(np.array(TEST_SMILES), np.array(TEST_LABELS))
        _, _num_removed = workflow._report._dictionary.gather_issue_counter()
        self.assertEqual(_num_removed, len(mask)-mask.sum())
        _, _num_altered = workflow._report._dictionary.gather_note_counter()
        self.assertEqual(_num_altered, mask.sum())
        self.assertEqual(_num_removed+_num_altered, len(mask))


class TestCurationSteps(unittest.TestCase):
    def test_Add3D(self):
        _curate_function = CurateAdd3D()
        mask, X, y = _curate_function(TEST_MOLS[:10], TEST_LABELS[:10])
        self.assertEqual(TEST_LABELS[:10], y)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS[:10]), len(X))
        self.assertEqual(len(TEST_MOLS[:10]), len(mask))

    def test_AddHs(self):
        _curate_function = CurateAddH()
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(TEST_LABELS, y)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))

    def test_Boron(self):
        _curate_function = CurateBoron()
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(TEST_LABELS, y)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))

    def test_MW(self):
        _curate_function = CurateMW()
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(TEST_LABELS, y)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))

        with self.assertRaises(ValueError):
            CurateMW(min_mw=-1)
            CurateMW(min_mw=100, max_mw=1)
            CurateMW(min_mw=100, max_mw=100)

        _curate_function = CurateMW(min_mw=10, max_mw=20)
        _smiles = ["C", "CC"]
        _mols = [MolFromSmiles(_) for _ in _smiles]
        mask, X, y = _curate_function(_mols)
        self.assertIsNone(y)
        self.assertEqual([True, False], mask.tolist())
        self.assertEqual(_smiles, [MolToSmiles(_) for _ in X])

    def test_Canonicalize(self):
        _curate_function = CurateCanonicalize()
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(TEST_LABELS, y)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))
        self.assertEqual([MolToSmiles(MolFromSmiles(_)) for _ in np.array(TEST_SMILES)[mask]], X[mask].tolist())

    def test_DupDisagree(self):
        _curate_function = CurateRemoveDuplicates()
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(TEST_LABELS, y)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))

        mask, X, y = _curate_function(DUMMY_DUP_MOLS, DUMMY_DUP_LABELS)
        self.assertEqual(mask.tolist(), [False, False, False, False, False, True, False, False, False])

    def test_DupDisagreeGreedy(self):
        _curate_function = CurateRemoveDuplicatesGreedy()
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(TEST_LABELS, y)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))

        mask, X, y = _curate_function(DUMMY_DUP_MOLS, DUMMY_DUP_LABELS)
        self.assertEqual(mask.tolist(), [True, False, False, False, True, True, False, False, False])

    def test_DupDisagreeCat(self):
        _curate_function = CurateRemoveDisagreeingDuplicatesCategorical(threshold=0.75)
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))

        mask, X, y = _curate_function(DUMMY_DUP_MOLS, DUMMY_DUP_LABELS)

        _curate_function = CurateRemoveDisagreeingDuplicatesCategorical(threshold=0.75)
        mask, X, y = _curate_function(DUMMY_DUP_MOLS, DUMMY_CAT_LABELS)
        self.assertEqual(mask.tolist(), [True, False, False, False, False, True, False, False, False])
        self.assertEqual(np.sum(y == np.array(["Y", "Y", "Y", "Y", "0", "Y", "Y", "0", np.nan])), 8)

        _curate_function = CurateRemoveDisagreeingDuplicatesCategorical(threshold=1.01)
        mask, X, y = _curate_function(DUMMY_DUP_MOLS, DUMMY_CAT_LABELS)
        self.assertEqual(mask.tolist(), [False, False, False, False, False, True, False, False, False])
        self.assertEqual(np.sum(y == DUMMY_CAT_LABELS), 8)

        _curate_function = CurateRemoveDisagreeingDuplicatesCategorical(threshold=0.01)
        mask, X, y = _curate_function(DUMMY_DUP_MOLS, DUMMY_CAT_LABELS)
        self.assertEqual(mask.tolist(), [True, False, False, False, True, True, False, False, False])
        self.assertEqual(y.tolist(), ["Y", "Y", "Y", "Y", "0", "Y", "Y", "0", "0"])

    def test_DupDisagreeStd(self):
        _curate_function = CurateRemoveDisagreeingDuplicatesStd(threshold=0.5)
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))

        mask, X, y = _curate_function(DUMMY_DUP_MOLS, DUMMY_DUP_LABELS)

        _curate_function = CurateRemoveDisagreeingDuplicatesStd(threshold=0.5)
        mask, X, y = _curate_function(DUMMY_DUP_MOLS, DUMMY_DUP_LABELS)
        self.assertEqual(mask.tolist(), [True, False, False, False, False, True, False, False, False])
        self.assertEqual(y.tolist(), [0.8, 0.8, 0.8, 0.8, 4.4, 2.1, 0.8, np.nan, 100])

        _curate_function = CurateRemoveDisagreeingDuplicatesStd(threshold=0.01)
        mask, X, y = _curate_function(DUMMY_DUP_MOLS, DUMMY_DUP_LABELS)
        self.assertEqual(mask.tolist(), [False, False, False, False, False, True, False, False, False])
        self.assertEqual(y.tolist(), DUMMY_DUP_LABELS)

        _curate_function = CurateRemoveDisagreeingDuplicatesStd(threshold=100)
        mask, X, y = _curate_function(DUMMY_DUP_MOLS, DUMMY_DUP_LABELS)
        self.assertEqual(mask.tolist(), [True, False, False, False, True, True, False, False, False])
        self.assertEqual(y.tolist(), [0.8, 0.8, 0.8, 0.8, np.mean([4.4, 100]), 2.1, 0.8,
                                      np.mean([4.4, 100]), np.mean([4.4, 100])])

    def test_DupDisagreeMinMax(self):
        _curate_function = CurateRemoveDisagreeingDuplicatesMinMax(threshold=0.5)
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))

        mask, X, y = _curate_function(DUMMY_DUP_MOLS, DUMMY_DUP_LABELS)

        _curate_function = CurateRemoveDisagreeingDuplicatesMinMax(threshold=2)
        mask, X, y = _curate_function(DUMMY_DUP_MOLS, DUMMY_DUP_LABELS)
        self.assertEqual(mask.tolist(), [True, False, False, False, False, True, False, False, False])
        self.assertEqual(y.tolist(), [0.8, 0.8, 0.8, 0.8, 4.4, 2.1, 0.8, np.nan, 100])

        _curate_function = CurateRemoveDisagreeingDuplicatesMinMax(threshold=0.01)
        mask, X, y = _curate_function(DUMMY_DUP_MOLS, DUMMY_DUP_LABELS)
        self.assertEqual(mask.tolist(), [False, False, False, False, False, True, False, False, False])
        self.assertEqual(y.tolist(), DUMMY_DUP_LABELS)

        _curate_function = CurateRemoveDisagreeingDuplicatesMinMax(threshold=100)
        mask, X, y = _curate_function(DUMMY_DUP_MOLS, DUMMY_DUP_LABELS)
        self.assertEqual(mask.tolist(), [True, False, False, False, True, True, False, False, False])
        self.assertEqual(y.tolist(),[0.8, 0.8, 0.8, 0.8, np.mean([4.4, 100]), 2.1, 0.8,
                                      np.mean([4.4, 100]), np.mean([4.4, 100])])

    def test_Flatten(self):
        _curate_function = CurateFlatten()
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(TEST_LABELS, y)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))

    def test_Inorganic(self):
        _curate_function = CurateInorganic()
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(TEST_LABELS, y)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))

    def test_Mixture(self):
        _curate_function = CurateMixtures()
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(TEST_LABELS, y)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))

        dummy_smiles = ["[Cl-].[H+]", "c1ccccc1.[NH2+]"]
        dummy_mols = [MolFromSmiles(_) for _ in dummy_smiles]
        dummy_labels = [1,1]
        mask, X, y = _curate_function(dummy_mols, dummy_labels)
        self.assertEqual([True, False], mask.tolist())
        self.assertEqual(MolToSmiles(X[0]), "[Cl-]")

    def test_demixture(self):
        _curate_function = CurateDemix()
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(TEST_LABELS, y)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))

        dummy_smiles = ["[Cl-].[H+]", "c1ccccc1.[NH2+]", "hello"]
        dummy_mols = [MolFromSmiles(_) for _ in dummy_smiles]
        dummy_labels = [1,1, 1]
        mask, X, y = _curate_function(dummy_mols, dummy_labels)
        self.assertEqual([True, True, False], mask.tolist())
        self.assertEqual(MolToSmiles(X[0]), "[Cl-]")
        self.assertEqual(MolToSmiles(X[1]), "c1ccccc1")

    def test_Neutralize(self):
        _curate_function = CurateNeutralize()
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(TEST_LABELS, y)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))

    def test_Sanitize(self):
        _curate_function = CurateSanitize()
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(TEST_LABELS, y)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))

    def test_Valid(self):
        _curate_function = CurateValid()
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(TEST_LABELS, y)
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))
        self.assertIsInstance(X[0], Mol)

    def test_LabelMissing(self):
        _curate_function = CurateRemoveMissingLabel()
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(TEST_LABELS, y.tolist())
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))
        self.assertIsInstance(X[0], Mol)

        mask, X, y = _curate_function(DUMMY_DUP_SMILES, DUMMY_CAT_LABELS)
        self.assertEqual(mask.tolist(), [True, True, True, True, True, True, True, True, False])

    def test_LabelNumeric(self):
        _curate_function = CurateMakeNumericLabel()
        mask, X, y = _curate_function(TEST_MOLS, TEST_LABELS)
        self.assertEqual(TEST_LABELS, y.tolist())
        self.assertEqual(len(mask), len(X))
        self.assertEqual(len(TEST_MOLS), len(X))
        self.assertEqual(len(TEST_MOLS), len(mask))
        self.assertIsInstance(X[0], Mol)

        mask, X, y = _curate_function(DUMMY_DUP_SMILES, DUMMY_DUP_LABELS)
        self.assertEqual(mask.tolist(), [True, True, True, True, True, True, True, False, True])

        mask, X, y = _curate_function(DUMMY_DUP_SMILES, DUMMY_CAT_LABELS)
        self.assertEqual(mask.tolist(), [False, False, False, False, True, False, True, True, False])


if __name__ == '__main__':
    unittest.main()
