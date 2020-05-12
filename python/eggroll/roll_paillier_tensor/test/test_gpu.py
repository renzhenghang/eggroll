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
    return np.reshape([random.gauss(0, 10) for _ in range(length)], shape)

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
            throughput = round(ins_num/elapsed, 2)
            print('benchmark:', func.__name__)
            print('{0:.2f}s cost, {1:.2f} instance per second'.format(elapsed, throughput))
            return res, throughput
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
        pen_list, _ = bench_mark(TEST_SIZE)(GPUEngine.encrypt_and_obfuscate)(fpn_list, self._pub_key)
        fpn_dec_list, gpu_throughput = bench_mark(TEST_SIZE)(GPUEngine.decryptdecode)(pen_list, self._pub_key, self._priv_key)
        cpu_dec_list, cpu_throughput = bench_mark(TEST_SIZE)(CPUEngine.decryptdecode)(pen_list, self._pub_key, self._priv_key)
        print('accelerate: {}'.format(round(gpu_throughput/cpu_throughput, 2)))

    def testScalaMul(self):
        s = random.gauss(0, 10)
        plain = generate_sample()
        gpu_enc_list = GPUEngine.encrypt_and_obfuscate(plain, self._pub_key, True)
        std_res = s * plain
        gpu_smul, gpu_throughput = bench_mark(TEST_SIZE)(GPUEngine.scalar_mul)(gpu_enc_list, s, self._pub_key)
        gpu_dec = GPUEngine.decryptdecode(gpu_smul, self._pub_key, self._priv_key)
        print(std_res[0])
        print('=====')
        print(gpu_dec[0])


    def testVdot(self):
        sample1 = generate_sample()
        sample2 = generate_sample()
        gpu_enc_sample1 = GPUEngine.encrypt_and_obfuscate(sample1, self._pub_key, True)
        gpu_mul_res, throughput = bench_mark(TEST_SIZE)(GPUEngine.vdot)(sample2, gpu_enc_sample1, self._pub_key)
        gpu_dec_res = GPUEngine.decryptdecode(gpu_mul_res, self._pub_key, self._priv_key)
        std_res = np.vdot(sample1, sample2)
        print(std_res)
        print('=====')
        print(gpu_dec_res)

    def testMul(self):
        s = generate_sample()
        plain = generate_sample()
        gpu_enc_list = GPUEngine.encrypt_and_obfuscate(plain, self._pub_key, True)
        std_res = s * plain
        gpu_smul, gpu_throughput = bench_mark(TEST_SIZE)(GPUEngine.mul)(gpu_enc_list, s, self._pub_key)
        gpu_dec = GPUEngine.decryptdecode(gpu_smul, self._pub_key, self._priv_key)
        print(std_res[0])
        print('=====')
        print(gpu_dec[0])


    def testAdd(self):
        fpn_list1 = generate_sample()
        fpn_list2 = generate_sample()
        pen_list1 = encrypt(fpn_list1, False)
        pen_list2 = encrypt(fpn_list2, False)

        add_res = add_impl(pen_list1, pen_list2)
        std_add_res = [pen_list1[i] + pen_list2[i] for i in range(10)]

        decry_list = decrypt(add_res)
        std_dec_res = decrypt(std_add_res)

    def testMatMul(self):
        plain_1 = generate_sample(64, (8, 8))#np.array([[-0.45210151, -8.50218413], [-6.27697607, -4.32974792]])
        plain_2 = generate_sample(64, (8, 8))#np.array([[-8.71179594, -6.17352569], [14.58582564,  6.68687083]])
        # plain_1 = np.array([
        #     [  4.50448658,  17.12903836,  22.33682854,   0.24429959,  -5.12314784],
        #     [  9.66272756,   6.27501679,   3.57851277,   6.37858254,   0.26577805],
        #     [  2.60227914,  10.24448602,   5.88320728,   4.53571183,   2.14124761],
        #     [-11.29226341, -13.9175846,    9.19047705, -21.88322718,  15.05309201],
        #     [  9.681583,     7.37928047, -14.97224968,  23.23707535,  12.59830091]
        # ])
        # plain_2 = np.array([
        #     [  1.24948929,  -9.26879398,   8.70192574,  -9.20243046, -14.1412848 ],
        #     [  4.64989016, -13.42373887,   7.25417334,  13.47832899,  -9.41023313],
        #     [ -4.62543966,  -0.31392176,  17.89862106,   0.21662054,  -1.16100151],
        #     [ -5.2547534,   -6.96247477,   4.0229177,   -8.33523413,   1.72597071],
        #     [  5.80752999,  -5.42635357,  -8.72068844,  -2.94709472,   6.19900684]
        # ])
        pen_np1 = GPUEngine.encrypt_and_obfuscate(plain_1, self._pub_key)

        res, throughput = bench_mark(400)(GPUEngine.matmul)(pen_np1, plain_2, self._pub_key)
        dec_res = GPUEngine.decryptdecode(res, self._pub_key, self._priv_key)
        std_res = np.matmul(plain_1, plain_2)
        print(dec_res)
        print('=====')
        print(std_res)
        print('a')
        print(plain_1)
        print('b')
        print(plain_2)

    def testSum(self):
        plain_list = generate_sample()

        plain_sum = np.sum(plain_list)
        fpn_list = [FixedPointNumber.encode(v, self._pub_key.n, self._pub_key.max_int) for v in plain_list.flatten()]

        enc_list = GPUEngine.encrypt_and_obfuscate(plain_list, self._pub_key, True)
        res, throughput = bench_mark(TEST_SIZE)(sum_impl)(enc_list.flatten())
        res = self._priv_key.decrypt(res)
        print(res)
        print(plain_sum)
    
    def testNegtive(self):
        num = -1.0
        encoded = FixedPointNumber.encode(num)
        encrypted = self._pub_key.encrypt(num)
        print(hex(encoded.encoding), encoded.encoding.bit_length())
        print(encoded.decode())
        print(hex(encrypted.ciphertext(False)))


if __name__ == '__main__':
    unittest.main()
