import numpy as np
import unittest
from eggroll.roll_paillier_tensor import rpt_py_engine as CPUEngine
from eggroll.roll_paillier_tensor import rpt_gpu_engine as GPUEngine
from eggroll.roll_paillier_tensor.roll_paillier_tensor import PaillierTensor, NumpyTensor

class TestGpuCode(unittest.TestCase):
    def testEncrypt(self):
        pass

    def testDecrypt(self):
        pass

    def testScalaMul(self):
        pass

    def testVdot(self):
        pass

    def testMul(self):
        pass

    def testAdd(self):
        pass

    def testMatMul(self):
        pass


if __name__ == '__main__':
    unittest.main()