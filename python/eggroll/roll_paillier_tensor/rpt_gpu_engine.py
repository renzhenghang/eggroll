
from eggroll.roll_paillier_tensor import paillier_gpu

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
    shape = data.shape
    d_list = data.flatten()
    res = paillier_gpu.encrypt(d_list, obf=False) # Fast Encrypt
    return np.reshape(res, shape)


def add(x, y, pub):
    """
    if x and y are paillier tensor, align, add.
    if x or y is numpy tensor and the other one is paillier tensor,
      fast encrypt(num2Mng) and call recursively
    """
    # steps:
    #   flatten the numpy array
    #   align
    #   add together
    x_shape = x.shape
    # TODO: CHECK x_shape == y_shape
    x_list = x.flatten()
    y_list = y.flatten()
    res = paillier_gpu.add_impl(x_list, y_list)
    return np.reshape(res, x_shape)

def scalar_mul(x, s, pub):
    """
    scala multiplication of x(vector) and s(scala)
    x: numpy array of PaillierEncryptedNumber
    y: FixedPointNumber
    """
    return x * s


def mul(x, s, pub):
    """
    return the result of x * s (sematic of "*" explained in numpy)
    x: numpy array of PaillierEncryptedNumber
    s: numpy array of FixedPointNumber
    """
    return x * s


def vdot(x, v, pub):
    """
    return the dot product of two vectors
    x: numpy array of FixedPointNumber
    y: numpy array of PaillierEncryptedNumber
    """
    return np.vdot(x, v)


def matmul(x, y, _pub):
    """
    return the matrix multiplication of x and y
    x: numpy ndarray of PaillierEncryptedNumber
    y: numpy ndarray of FixedPointNumber
    """
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
    paillier_gpu.init_gpu_keys(pub, priv)

    return pub, priv

