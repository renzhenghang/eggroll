import numpy as np
import unittest
from eggroll.roll_paillier_tensor import rpt_py_engine as CPUEngine
from eggroll.roll_paillier_tensor import rpt_gpu_engine as GPUEngine
from eggroll.roll_paillier_tensor.roll_paillier_tensor import PaillierTensor, NumpyTensor
from eggroll.roll_paillier_tensor.paillier_gpu import init_gpu_keys, encrypt, decrypt, add_impl, mul_impl, sum_impl
from federatedml.secureprotol.fate_paillier import PaillierKeypair, PaillierEncryptedNumber
from federatedml.secureprotol.fixedpoint import FixedPointNumber
import random

# random.seed()
def generate_fpn(length):
    return [FixedPointNumber.encode(random.random()) for _ in range(length)]

def dump_res(fpn_list):
    print('\nencoding\t\texponent')
    [print(hex(v.encoding), hex(v.exponent), sep='\t\t') for v in fpn_list]

class TestGpuCode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._pub_key, cls._priv_key = PaillierKeypair.generate_keypair()
        # print(cls._pub_key.n)
        init_gpu_keys(cls._pub_key, cls._priv_key)

    def testEncrypt(self):
        fpn_list = generate_fpn(1000)
        pen_list = encrypt(fpn_list, False)
        pen_list_2 = [self._pub_key.raw_encrypt(v.encoding, random_value=1) for v in fpn_list]

    def testDecrypt(self):
        fpn_list = generate_fpn(10)
        pen_list = encrypt(fpn_list, False)
        fpn_dec_list = decrypt(pen_list)
        print('dec res')
        dump_res(fpn_dec_list)
        # for i in range(10):
        #     print(fpn_list[i].encoding, fpn_dec_list[i].encoding)


    def testScalaMul(self):
        pass

    def testVdot(self):
        pass

    def testMul(self):
        fpn_list1 = generate_fpn(10)
        fpn_list2 = generate_fpn(10)

        pen_list2 = encrypt(fpn_list2, False)
        pen_res = mul_impl(pen_list2, fpn_list1)

        dec_res_gpu = decrypt(pen_res)
        
        std_res = [pen_list2[i] * fpn_list1[i].decode() for i in range(10)]
        
        std_dec = decrypt(std_res)

        dump_res(dec_res_gpu)
        dump_res(std_dec)


    def testAdd(self):
        fpn_list1 = generate_fpn(10)
        fpn_list2 = generate_fpn(10)
        pen_list1 = encrypt(fpn_list1, False)
        pen_list2 = encrypt(fpn_list2, False)

        add_res = add_impl(pen_list1, pen_list2)
        std_add_res = [pen_list1[i] + pen_list2[i] for i in range(10)]

        decry_list = decrypt(add_res)
        std_dec_res = decrypt(std_add_res)
        # print('before add')
        # print('a_encoding\t\ta_exponent\t\tb_encoding\t\tb_exponent')
        # [print(hex(fpn_list1[i].encoding), hex(fpn_list1[i].exponent),\
        #      hex(fpn_list2[i].encoding), hex(fpn_list2[i].exponent), sep='\t\t') for i in range(10)]
        # print('after add')
        # print('res_encoding\t\tres_exponent')
        # [print(hex(decry_list[i].encoding), hex(decry_list[i].exponent), sep='\t\t') for i in range(10)]
        # print('std res')
        # print('res_encoding\t\tres_exponent')
        # [print(hex(std_dec_res[i].encoding), hex(std_dec_res[i].exponent), sep='\t\t') for i in range(10)]

        # dec_1 = decrypt(pen_list1)
        # dec_2 = decrypt(pen_list2)

        # print('=============testAdd============')
        # print('before align')
        # print('a_encoding\t\ta_exponent\t\tb_encoding\t\tb_exponent')
        # [print(hex(fpn_list1[i].encoding), hex(fpn_list1[i].exponent),\
        #      hex(fpn_list2[i].encoding), hex(fpn_list2[i].exponent), sep='\t\t') for i in range(10)]
        # print('after align')
        # print('a_encoding\t\ta_exponent\t\tb_encoding\t\tb_exponent')
        # [print(hex(dec_1[i].encoding), hex(dec_1[i].exponent),\
        #      hex(dec_2[i].encoding), hex(dec_2[i].exponent), sep='\t\t') for i in range(10)]
        # pass

    def testMatMul(self):
        pass

    def testSum(self):
        fpn_list = generate_fpn(3)

        fpn_sum = sum(fpn_list)

        pen_list = encrypt(fpn_list)
        std_sum = sum(pen_list)
        print(len(pen_list))
        res = sum_impl(pen_list)
        dec = decrypt([res])
        std_dec = decrypt([std_sum])
        dump_res(dec)
        dump_res(std_dec)

        print(hex(fpn_sum.encoding))

        print(hex(fpn_list[0].encoding), fpn_list[0].exponent)
        print(hex(fpn_list[1].encoding), fpn_list[1].exponent)
        print(hex(fpn_list[2].encoding), fpn_list[2].exponent)


if __name__ == '__main__':
    unittest.main()