import pickle
import unittest

import numpy as np

from rdkit.Chem import MolFromSmiles, MolToSmiles, Mol
from rdkit.rdBase import BlockLogs

from rdkit.DataStructs import ExplicitBitVect, UIntSparseIntVect

from vdb.chem.fp import *
from vdb.chem.fp.base import BinaryFPFunc, DiscreteFPFunc, ContinuousFPFunc, ObjectFPFunc, RdkitWrapper
from vdb.chem.cluster import cluster_scaffold, cluster_leader

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


TEST_MOLS = TEST_MOLS[:50] + [None]
TEST_SMILES = TEST_SMILES[:50] + ["helo"]


class TestFPFunc(unittest.TestCase):
    def test_getfp(self):
        fp_func = get_fp_func("ECFP4")
        with self.assertRaises(FPFuncError):
            get_fp_func("ALALALALALALALALALAL")

    def test_equal(self):
        fp_func1 = ECFP4()
        fp_func2 = ECFP4()
        fp_func3 = ECFP6()
        self.assertEqual(fp_func1, fp_func2)
        self.assertNotEqual(fp_func1, fp_func3)

    def test_base(self):
        fp_func1 = ECFP4()
        self.assertIsNone(fp_func1.fit(None, None))
        fp_func1.fit_transform(TEST_MOLS, None)
        fp_func1.get_feature_names_out()
        _name = fp_func1.__name__

        import os
        fp_func1.save("./TEST.pkl")
        fp_func2 = pickle.load(open("./TEST.pkl", "rb"))
        self.assertEqual(fp_func1, fp_func2)
        self.assertTrue(os.path.exists("./TEST.pkl"))
        os.remove("./TEST.pkl")

    def test_rdkit(self):
        fp_func1 = ECFP4()
        _fps = fp_func1.generate_fps_as_rdkit_objects(TEST_SMILES)
        self.assertIsInstance(_fps[0], UIntSparseIntVect)
        self.assertIsNone(_fps[-1])
        fp_func1 = BinaryECFP4()
        _fps = fp_func1.generate_fps_as_rdkit_objects(TEST_SMILES)
        self.assertIsInstance(_fps[0], ExplicitBitVect)
        self.assertIsNone(_fps[-1])

    def test_ECFP4(self):
        fp_func = ECFP4(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        self.assertEqual(fp_func.get_dimension(), 2048)
        self.assertFalse(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, DiscreteFPFunc)

    def test_BinaryECFP4(self):
        fp_func = BinaryECFP4(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        self.assertEqual(fp_func.get_dimension(), 2048)
        self.assertTrue(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, BinaryFPFunc)

    def test_ECFP6(self):
        fp_func = ECFP6(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        self.assertEqual(fp_func.get_dimension(), 2048)
        self.assertFalse(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, DiscreteFPFunc)

    def test_BinaryECFP6(self):
        fp_func = BinaryECFP6(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        self.assertEqual(fp_func.get_dimension(), 2048)
        self.assertTrue(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, BinaryFPFunc)

    def test_FCFP4(self):
        fp_func = FCFP4(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        self.assertEqual(fp_func.get_dimension(), 2048)
        self.assertFalse(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, DiscreteFPFunc)

    def test_BinaryFCFP4(self):
        fp_func = BinaryFCFP4(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        self.assertEqual(fp_func.get_dimension(), 2048)
        self.assertTrue(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, BinaryFPFunc)

    def test_FCFP6(self):
        fp_func = FCFP6(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        self.assertEqual(fp_func.get_dimension(), 2048)
        self.assertFalse(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, DiscreteFPFunc)

    def test_BinaryFCFP6(self):
        fp_func = BinaryFCFP6(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        self.assertEqual(fp_func.get_dimension(), 2048)
        self.assertTrue(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, BinaryFPFunc)

    def test_AtomPair(self):
        fp_func = AtomPair(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        self.assertEqual(fp_func.get_dimension(), 2048)
        self.assertFalse(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, DiscreteFPFunc)

    def test_BinaryAtomPair(self):
        fp_func = BinaryAtomPair(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        self.assertEqual(fp_func.get_dimension(), 2048)
        self.assertTrue(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, BinaryFPFunc)

    def test_Avalon(self):
        fp_func = Avalon(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        self.assertEqual(fp_func.get_dimension(), 2048)
        self.assertFalse(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, DiscreteFPFunc)

    def test_BinaryAvalon(self):
        fp_func = BinaryAvalon(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        self.assertEqual(fp_func.get_dimension(), 2048)
        self.assertTrue(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, BinaryFPFunc)

    def test_TopTor(self):
        fp_func = TopTor(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        self.assertEqual(fp_func.get_dimension(), 2048)
        self.assertFalse(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, DiscreteFPFunc)

    def test_BinaryTopTor(self):
        fp_func = BinaryTopTor(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        self.assertEqual(fp_func.get_dimension(), 2048)
        self.assertTrue(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, BinaryFPFunc)

    def test_RDK(self):
        fp_func = RDK(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 2048)

        self.assertEqual(fp_func.get_dimension(), 2048)
        self.assertTrue(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, BinaryFPFunc)

    def test_MACCS(self):
        fp_func = MACCS(use_tqdm=False)
        _fps = fp_func(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 167)

        _fps = fp_func.generate_fps(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 167)

        _fps, mask = fp_func.generate_fps(TEST_MOLS, return_mask=True)
        self.assertEqual(len(mask), len(TEST_MOLS))
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 167)
        self.assertFalse(np.any(np.isnan(_fps[mask])))
        self.assertTrue(np.any(np.isnan(_fps)))

        _fps = fp_func.transform(TEST_MOLS)
        self.assertEqual(len(_fps), len(TEST_MOLS))
        self.assertEqual(_fps.shape[1], 167)

        self.assertEqual(fp_func.get_dimension(), 167)
        self.assertTrue(fp_func.is_binary())

        self.assertFalse(fp_func.use_tqdm)
        fp_func.turn_on_tqdm()
        self.assertTrue(fp_func.use_tqdm)
        fp_func.turn_off_tqdm()
        self.assertFalse(fp_func.use_tqdm)

        self.assertIsInstance(fp_func, BinaryFPFunc)