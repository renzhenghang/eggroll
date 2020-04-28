import numpy as np
import unittest
from eggroll.roll_paillier_tensor import rpt_py_engine as CPUEngine
from eggroll.roll_paillier_tensor import rpt_gpu_engine as GPUEngine
from eggroll.roll_paillier_tensor.roll_paillier_tensor import PaillierTensor, NumpyTensor
from eggroll.roll_paillier_tensor.paillier_gpu import init_gpu_keys, encrypt, decrypt, add_impl, mul_impl, sum_impl, matmul_impl, encrypt_async
from federatedml.secureprotol.fate_paillier import PaillierKeypair, PaillierEncryptedNumber
from federatedml.secureprotol.fixedpoint import FixedPointNumber
import random
import functools
import time
import asyncio

TEST_SIZE = 100000
TEST_SHAPE = (10, 100, 100)

# random.seed()
def generate_sample(length=TEST_SIZE, shape=TEST_SHAPE):
    return np.reshape([random.gauss(0, 10) for _ in range(length)], TEST_SHAPE)

def dump_res(fpn_list):
    print('\nencoding\t\texponent')
    [print((v.encoding), (v.exponent), sep='\t\t') for v in fpn_list]

def bench_mark(ins_num):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            throughput = ins_num/elapsed
            print('benchmark:', func.__name__)
            print('{0:.2f}s cost, {1:.2f} instance per second'.format(elapsed, throughput))
            return res
        return wrapper
    return decorator

async def enc_async_impl(fpn_list):
    job_list = [encrypt_async(fpn_list[i], i=i) for i in range(len(fpn_list))]
    res = await asyncio.gather(*job_list)
    return res

def enc_async_test(fpn_list):
    return asyncio.run(enc_async_impl(fpn_list))

# async def enc_multithreading(fpn_list):
    # thread 

class TestGpuCode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._pub_key, cls._priv_key = PaillierKeypair.generate_keypair()
        bench_mark(1)(init_gpu_keys)(cls._pub_key, cls._priv_key)

    def testEncrypt(self):
        fpn_list = generate_sample()
        pen_list = bench_mark(TEST_SIZE)(GPUEngine.encrypt_and_obfuscate)(fpn_list, self._pub_key, True)
        cpu_pen_list = bench_mark(TEST_SIZE)(CPUEngine.encrypt_and_obfuscate)(fpn_list, self._pub_key)

    def testDecrypt(self):
        fpn_list = generate_sample()
        pen_list = GPUEngine.encrypt_and_obfuscate(fpn_list, self._pub_key, True)
        fpn_dec_list = bench_mark(TEST_SIZE)(GPUEngine.decryptdecode)(pen_list, self._pub_key, self._priv_key)
        cpu_dec_list = bench_mark(TEST_SIZE)(CPUEngine.decryptdecode)(pen_list, self._pub_key, self._priv_key)
        print(fpn_list[0])
        print(fpn_dec_list[0])
        print(cpu_dec_list[0])

    def testScalaMul(self):
        pass

    def testVdot(self):
        pass

    def testMul(self):
        pass

    def testAdd(self):
        fpn_list1 = generate_sample()
        fpn_list2 = generate_sample()
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
        fpn_list1 = generate_fpn(4)
        decode_list1 = [t.decode() for t in fpn_list1]
        fpn_list2 = generate_fpn(4)
        decode_list2 = [t.decode() for t in fpn_list2]
    
        fpn_np1 = np.reshape(fpn_list1, (2,2))
        decode_np1 = np.reshape(decode_list1, (2,2))
        fpn_np2 = np.reshape(fpn_list2, (2,2))
        decode_np2 = np.reshape(decode_list2, (2,2))

        pen_np1 = GPUEngine.encrypt_and_obfuscate(fpn_np1, self._pub_key)

        res = GPUEngine.matmul(pen_np1, fpn_np2, self._pub_key)
        dec_res = GPUEngine.decryptdecode(res, self._pub_key, self._priv_key)

        print("fpn_list1")
        dump_res(fpn_list2)
        dump_res(fpn_list1)
        print("np1", decode_np1)
        print("np2", decode_np2)
        print("dec res", dec_res)
        print(decode_np1.dot(decode_np2))

        # pen_np2 = CPUEngine.encrypt_and_obfuscate(decode_np1, self._pub_key)

        # print(pen_np1)
        # print(pen_np1.shape)
        # for i in range(2):
            # for j in range(2):
                # print(pen_np1[i][j].ciphertext(False), pen_np1[i][j].exponent)
                # print(pen_np2[i][j].ciphertext(False), pen_np2[i][j].exponent)
                # print("")

        # dec_np1 = CPUEngine.decryptdecode(pen_np1, self._pub_key, self._priv_key)
        # dec_np2 = CPUEngine.decryptdecode(pen_np2, self._pub_key, self._priv_key)

    def testSum(self):
        fpn_list = generate_sample()

        fpn_sum = sum(fpn_list)

        pen_list = encrypt(fpn_list)
        std_sum = sum(pen_list)
        # print(len(pen_list))
        res = sum_impl(pen_list)
        dec = decrypt([res])
        std_dec = decrypt([std_sum])
        dump_res(dec)
        dump_res(std_dec)

        print(hex(fpn_sum.encoding))

        print(hex(fpn_list[0].encoding), fpn_list[0].exponent)
        print(hex(fpn_list[1].encoding), fpn_list[1].exponent)
        print(hex(fpn_list[2].encoding), fpn_list[2].exponent)
        # pass
    
    def testNegtive(self):
        num = -1.0
        encoded = FixedPointNumber.encode(num)
        encrypted = self._pub_key.encrypt(num)
        print(hex(encoded.encoding), encoded.encoding.bit_length())
        print(encoded.decode())
        print(hex(encrypted.ciphertext(False)))


if __name__ == '__main__':
    unittest.main()
