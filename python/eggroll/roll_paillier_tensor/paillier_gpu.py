import ctypes
from ctypes import c_char_p, c_int32, c_int64, create_string_buffer
import ctypes.util
from functools import wraps
import random
from .paillier_exception import *

CPH_BITS = 2048
CPH_BYTES = CPH_BITS // 8
_key_init = False

def _load_cuda_lib():
    path = ctypes.util.find_library('paillier.so')
    if path == '':
        raise CudaLibLoadErr('Cuda Lib Not Found, please check LD_LIBRARY_PATH')
    lib = ctypes.CDLL(path)
    return lib

_cuda_lib = _load_cuda_lib()

def check_key(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if _key_init:
            return func(*args, **kwargs)
        # TODO: Raise Error
    return wrapper

def init_gpu_keys(pub_key, priv_key):
    global _key_init
    if _key_init:
        print('key initiated, return.')
    c_n = c_char_p(pub_key.n.to_bytes(CPH_BYTES, 'little'))
    c_g = c_char_p(pub_key.g.to_bytes(CPH_BYTES, 'little'))
    c_nsquare = c_char_p(pub_key.nsquare.to_bytes(CPH_BYTES, 'little'))
    c_max_int = c_char_p(pub_key.max_int.to_bytes(CPH_BYTES, 'little'))

    _cuda_lib.init_pub_key(c_n, c_g, c_nsquare, c_max_int)

    c_p = c_char_p(priv_key.p.to_bytes(CPH_BYTES, 'little'))
    c_q = c_char_p(priv_key.q.to_bytes(CPH_BYTES, 'little'))
    c_psquare = c_char_p(priv_key.psquare.to_bytes(CPH_BYTES, 'little'))
    c_qsquare = c_char_p(priv_key.qsquare.to_bytes(CPH_BYTES, 'little'))
    c_q_inverse = c_char_p(priv_key.q_inverse.to_bytes(CPH_BYTES, 'little'))
    c_hp = c_char_p(priv_key.hp.to_bytes(CPH_BYTES, 'little'))
    c_hq = c_char_p(priv_key.hq.to_bytes(CPH_BYTES, 'little'))

    _cuda_lib.init_priv_key(c_p, c_q, c_psquare, c_qsquare, c_q_inverse, c_hp, c_hq)

    _key_init = True

def init_err_report():
    _cuda_lib.init_err_report()

def get_bytes(int_array, length):
    res = b''
    for a in int_array:
        res += a.to_bytes(length, 'little')
    
    return res


def get_int(byte_array, count, length):
    res = []
    for i in range(count):
        res.append(int.from_bytes(byte_array[i * length: (i + 1) * length], 'little'))
    return res


@check_key
def raw_encrypt_gpu(values):
    global _cuda_lib
    res_p = create_string_buffer(len(values) * CPH_BYTES)
    c_count = c_int32(len(values))
    array_t = c_int32 * len(values)
    c_array = array_t(*values)
    _cuda_lib.call_raw_encrypt(c_array, c_count, res_p)
    res = get_int(res_p.raw, len(values), CPH_BYTES)

    return res

@check_key
def raw_encrypt_obfs_gpu(values, rand_vals):
    global _cuda_lib
    res_p = create_string_buffer(len(values) * CPH_BYTES)
    c_count = c_int32(len(values))
    array_t = c_int32 * len(values)
    c_input = array_t(*values)
    c_rand_vals = array_t(*rand_vals)
    _cuda_lib.call_raw_encrypt_obfs(c_input, c_count, res_p, c_rand_vals)
    res = get_int(res_p.raw, len(values), CPH_BYTES)

    return res

@check_key
def raw_add_gpu(ciphers_a, ciphers_b, res_p):
    global _cuda_lib
    ins_num = len(ciphers_a) # TODO: check len(ciphers_a) == len(ciphers_b)
    in_a = get_bytes(ciphers_a, CPH_BYTES)
    in_b = get_bytes(ciphers_b, CPH_BYTES)

    c_count = c_int32(ins_num)

    _cuda_lib.call_raw_add(in_a, in_b, res_p, c_count)

@check_key
def raw_mul_gpu(ciphers_a, plains_b, res_p):
    global _cuda_lib
    ins_num = len(ciphers_a) # TODO: check len(ciphers_a) == len(plains_b)
    in_a = get_bytes(ciphers_a, CPH_BYTES)
    in_b = get_bytes(plains_b, 4)

    c_count = c_int32(ins_num)

    _cuda_lib.call_raw_mul(in_a, in_b, res_p, c_count)

@check_key
def raw_decrypt_gpu(ciphers):
    global _cuda_lib
    res_p = create_string_buffer(len(ciphers) * 4)
    ins_num = len(ciphers)
    in_cipher = get_bytes(ciphers, CPH_BYTES)

    c_count = c_int32(ins_num)

    _cuda_lib.call_raw_decrypt(in_cipher, c_count, res_p)

    return get_int(res_p.raw, ins_num, 4)