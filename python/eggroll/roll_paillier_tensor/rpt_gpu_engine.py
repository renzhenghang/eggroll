
from .paillier_gpu import init_gpu_keys, init_err_report, \
     raw_encrypt_gpu, raw_encrypt_obfs_gpu, raw_add_gpu, raw_mul_gpu, raw_decrypt_gpu

from federatedml.secureprotol.fate_paillier import PaillierKeypair, PaillierPublicKey, PaillierEncryptedNumber
from federatedml.secureprotol.fixedpoint import FixedPointNumber
from federatedml.secureprotol import gmpy_math
import random
import numpy as np


def load(data):
    return data


def dump(data):
    return data


def load_pub_key(pub):
    return pub


def dump_pub_key(pub):
    return pub


def load_prv_key(priv):
    return priv


def dump_prv_key(priv):
    return priv


def num2Mng(data, pub):
    return data
    # return np.vectorize(FixedPointNumber.encode)(data)


def add(x, y, pub):
    # return x + y
    # if x and y are paillier tensor, add them together with add_gpu
    # if x or y is numpy tensor and the other one is paillier tensor?
    x_encoded = np.vectorize(FixedPointNumber.encode)(x)
    y_encoded = np.vectorize(FixedPointNumber.encode)(x)


def scalar_mul(x, s, pub):
    return x * s


def mul(x, s, pub):
    return x * s


def vdot(x, v, pub):
    return x * v


def matmul(x, y, _pub):
    return x @ y


def transe(data):
    return data.T


def mean(data, pub):
    return np.array([data.mean(axis=0)])


def hstack(x, y, pub):
    return np.hstack((x, y))


def decryptdecode(data, pub, priv):
    return np.vectorize(priv.decrypt)(data)


# def print(data, pub, priv):
#     pprint(decryptdecode(data, pub, priv))


def encrypt_and_obfuscate(data, pub, obfs=None):
    if obfs is None:
        return np.vectorize(pub.encrypt)(data)

    def func(value, obf):
        encoding = pub.encode(value)
        ciphertext = pub.raw_encrypt(encoding.encoding)
        ciphertext = pub.apply_obfuscator(ciphertext, obf=obf)
        return PaillierEncryptedNumber(pub, ciphertext, encoding.exponent)
    return np.vectorize(func)(data, obfs)


def keygen():
    pub, priv = PaillierKeypair().generate_keypair()
    init_gpu_keys(pub, priv)

    return pub, priv

