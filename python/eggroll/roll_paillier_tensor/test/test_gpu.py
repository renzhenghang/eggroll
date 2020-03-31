import numpy as np
import unittest
from eggroll.roll_paillier_tensor import rpt_py_engine as CPUEngine
from eggroll.roll_paillier_tensor import rpt_gpu_engine as GPUEngine
from eggroll.roll_paillier_tensor.roll_paillier_tensor import PaillierTensor, NumpyTensor
from eggroll.roll_paillier_tensor.paillier_gpu import init_gpu_keys
from federatedml.secureprotol.fate_paillier import PaillierKeypair, PaillierEncryptedNumber

class TestGpuCode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._pub_key, cls._priv_key = PaillierKeypair.generate_keypair()
        init_gpu_keys(cls._pub_key, cls._priv_key)

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