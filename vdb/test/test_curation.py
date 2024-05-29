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
