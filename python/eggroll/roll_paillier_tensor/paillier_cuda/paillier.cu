#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include <sys/time.h>
#include "cgbn/cgbn.h"
#include "samples/utility/gpu_support.h"
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <cstdlib>

// #include <chrono>
#include <cassert>

#include "fixedpoint.h"

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 32
#define CPH_BITS 2048 // cipher bits
#define MAX_RAND_SEED 4294967295U
#define WINDOW_BITS 5

// helpful typedefs for the kernel
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, CPH_BITS> env_cph_t;
typedef cgbn_mem_t<CPH_BITS> gpu_cph;

void store2dev(void *address,  mpz_t z, unsigned int BITS) {
  size_t words;
  if(mpz_sizeinbase(z, 2)>BITS) {
    exit(1);
  }
  mpz_export((uint32_t *)address, &words, -1, sizeof(uint32_t), 0, 0, z);
  while(words<(BITS+31)/32)
    ((uint32_t *)address)[words++]=0;
}

void store2gmp(mpz_t z, void *address, unsigned int BITS ) {
  mpz_import(z, (BITS+31)/32, -1, sizeof(uint32_t), 0, 0, (uint32_t *)address);
}


void invert(mpz_t rop, mpz_t a, mpz_t b) {
  mpz_invert(rop, a, b);
}

inline void cudaMallocAndSet(void **addr, size_t size) {
  cudaMalloc(addr, size);
  cudaMemset(*addr, 0, size);
}

inline void dumpMem(char *addr, size_t size) {
  printf("dumping memory at 0x%x\n", addr);
  printf("0x");
  for (int64_t i = size - 1; i >= 0; i--) printf("%02x", addr[i] & 0xff);
  printf("\n");
}

enum MemcpyType {
HostToHost = 0,
HostToDevice,
DeviceToHost,
DeviceToDevice
};

class PaillierPublicKey {
 public:
  cgbn_mem_t<CPH_BITS> g;
  cgbn_mem_t<CPH_BITS> n;
  cgbn_mem_t<CPH_BITS> nsquare;
  cgbn_mem_t<CPH_BITS> max_int;
};


class PaillierPrivateKey {
 public:
  cgbn_mem_t<CPH_BITS> p;
  cgbn_mem_t<CPH_BITS> q;
  cgbn_mem_t<CPH_BITS> psquare;
  cgbn_mem_t<CPH_BITS> qsquare;
  cgbn_mem_t<CPH_BITS> q_inverse;
  cgbn_mem_t<CPH_BITS> hp;
  cgbn_mem_t<CPH_BITS> hq;
};

struct PaillierEncryptedNumber {
  char cipher[CPH_BITS/8]; // expected size: CPH_BITS/8 bytes
  uint32_t exponent;
  uint32_t base;
};

inline void extractPen(gpu_cph *dst, PaillierEncryptedNumber *src, uint32_t count, MemcpyType type) {
  for (int i = 0; i < count; i++) {
    if (type == HostToHost)
      memcpy(dst + i, src[i].cipher, sizeof(gpu_cph));
    else if (type == HostToDevice)
      cudaMemcpy(dst + i, src[i].cipher, sizeof(gpu_cph), cudaMemcpyHostToDevice);
    else if (type == DeviceToHost)
      cudaMemcpy(dst + i, src[i].cipher, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
    else if (type == DeviceToDevice)
      cudaMemcpy(dst + i, src[i].cipher, sizeof(gpu_cph), cudaMemcpyDeviceToDevice);
  }
}

inline void penFromBuffer(PaillierEncryptedNumber *dst, gpu_cph *src, uint32_t count, MemcpyType type) {
  for (int i = 0; i < count; i++) {
    if (type == HostToHost)
      memcpy((dst + i)->cipher, src + i, sizeof(gpu_cph));
    else if (type == HostToDevice)
      cudaMemcpy((dst + i)->cipher, src + i, sizeof(gpu_cph), cudaMemcpyHostToDevice);
    else if (type == DeviceToHost)
      cudaMemcpy((dst + i)->cipher, src + i, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
    else if (type == DeviceToDevice)
      cudaMemcpy((dst + i)->cipher, src + i, sizeof(gpu_cph), cudaMemcpyDeviceToDevice);
  }
}

__device__  __forceinline__ void l_func(env_cph_t &bn_env, env_cph_t::cgbn_t &out, 
		env_cph_t::cgbn_t &cipher_t, env_cph_t::cgbn_t &x_t, env_cph_t::cgbn_t &xsquare_t, 
		env_cph_t::cgbn_t &hx_t) {
/****************************************************************************************
* calculate L(cipher_t^(x_t - 1) mod xsquare_t) * hx_t. 
* input: cipher_t, x_t, xsquare-t, hx_t
* out:   out
*/
  env_cph_t::cgbn_t  tmp, tmp2, cipher_lt;
  env_cph_t::cgbn_wide_t  tmp_wide;
  cgbn_sub_ui32(bn_env, tmp2, x_t, 1);
  
  if(cgbn_compare(bn_env, cipher_t, xsquare_t) >= 0) {
    cgbn_rem(bn_env, cipher_lt, cipher_t, xsquare_t);
    mont_modular_power(bn_env,tmp,cipher_lt,tmp2,xsquare_t);
  } else {
    mont_modular_power(bn_env, tmp, cipher_t, tmp2, xsquare_t);
  }
 
  cgbn_sub_ui32(bn_env, tmp, tmp, 1);
  cgbn_div(bn_env, tmp, tmp, x_t);
  cgbn_mul_wide(bn_env, tmp_wide, tmp, hx_t);
  cgbn_rem_wide(bn_env, tmp, tmp_wide, x_t);
 
  cgbn_set(bn_env, out, tmp);
}


__global__ void setup_kernel(curandState *state){
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}

__global__ __noinline__ void apply_obfuscator(PaillierPublicKey *gpu_pub_key, cgbn_error_report_t *report, 
		gpu_cph *ciphers, gpu_cph *obfuscators, int count, curandState *state ) {
/******************************************************************************************
* obfuscate the encrypted text, obfuscator = cipher * r^n mod n^2
* in:
*   ciphers: encrypted text from simple raw encryption
*   state:   GPU random generator state.
* out:
*   obfuscators: obfused encryption text.
*/
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int tid= idx/TPI;
  if(tid>=count)
    return;

  context_t      bn_context(cgbn_report_monitor, report, tid);  
  env_cph_t     bn_env(bn_context.env<env_cph_t>());                   
  env_cph_t::cgbn_t  n, nsquare,cipher, r, tmp;
  env_cph_t::cgbn_wide_t tmp_wide;

  curandState localState = state[idx];
  unsigned int rand_r = curand_uniform(&localState) * MAX_RAND_SEED;                  
  state[idx] = localState;

  cgbn_set_ui32(bn_env, r, rand_r); // TODO: new rand or reuse
  cgbn_load(bn_env, n, &gpu_pub_key[0].n);
  cgbn_load(bn_env, nsquare, &gpu_pub_key[0].nsquare);
  cgbn_load(bn_env, cipher, &ciphers[tid]);
  mont_modular_power(bn_env, tmp, r, n, nsquare);
  cgbn_mul_wide(bn_env, tmp_wide, cipher, tmp);
  cgbn_rem_wide(bn_env, r, tmp_wide, nsquare);
  cgbn_store(bn_env, obfuscators + tid, r);   // store r into sum
}


__global__ void raw_encrypt(PaillierPublicKey *gpu_pub_key, cgbn_error_report_t *report, 
		gpu_cph *plains, gpu_cph *ciphers,int count) {
/*************************************************************************************
* simple encrption cipher = 1 + plain * n mod n^2
* in:
*   plains: plain text(2048 bits)
* out:
*   ciphers: encrypted result.
*/
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(tid>=count)
    return;
  context_t      bn_context(cgbn_report_monitor, report, tid);  
  env_cph_t      bn_env(bn_context.env<env_cph_t>());                   
  env_cph_t::cgbn_t  n, nsquare, plain, cipher;
  cgbn_load(bn_env, n, &gpu_pub_key[0].n);      
  cgbn_load(bn_env, plain, plains + tid);
  cgbn_load(bn_env, nsquare, &gpu_pub_key[0].nsquare);
  cgbn_load(bn_env, plain, plains + tid);
  cgbn_mul(bn_env, cipher, n, plain);
  cgbn_add_ui32(bn_env, cipher, cipher, 1);
  cgbn_rem(bn_env, cipher, cipher, nsquare);

  cgbn_store(bn_env, ciphers + tid, cipher);   // store r into sum
}

__global__ __noinline__ void raw_encrypt_with_obfs(PaillierPublicKey *gpu_pub_key, cgbn_error_report_t *report, 
		gpu_cph *plains, gpu_cph *ciphers, int count, uint32_t *rand_vals) {
/*******************************************************************************
* encryption and obfuscation in one function, with less memory copy.
* in:
*   plains: plain text.
*   state: random generator state.
* out:
*   ciphers: encrpted text.
*/
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int tid= idx/TPI;
  if(tid>=count)
    return;
  context_t     bn_context(cgbn_report_monitor, report, tid);  
  env_cph_t     bn_env(bn_context.env<env_cph_t>());                   
  env_cph_t::cgbn_t  n, nsquare, plain,  tmp, max_int, cipher; 
  env_cph_t::cgbn_wide_t tmp_wide;
  cgbn_load(bn_env, n, &gpu_pub_key[0].n);      
  cgbn_load(bn_env, plain, plains + tid);
  cgbn_load(bn_env, nsquare, &gpu_pub_key[0].nsquare);
  cgbn_load(bn_env, max_int, &gpu_pub_key[0].max_int);
  cgbn_load(bn_env, plain, plains + tid);
  cgbn_sub(bn_env, tmp, n, max_int); 
  cgbn_mul(bn_env, cipher, n, plain);
  cgbn_add_ui32(bn_env, cipher, cipher, 1);
  cgbn_rem(bn_env, cipher, cipher, nsquare);

  env_cph_t::cgbn_t r; 

  // curandState localState = state[idx];
  uint32_t rand_r = rand_vals[tid];
  // state[idx] = localState;

  cgbn_set_ui32(bn_env, r, rand_r); // TODO: new rand or reuse

  mont_modular_power(bn_env,tmp, r, n, nsquare);

  cgbn_mul_wide(bn_env, tmp_wide, cipher, tmp);
  cgbn_rem_wide(bn_env, r, tmp_wide, nsquare);
  cgbn_store(bn_env, ciphers + tid, r);   // store r into sum
}


__global__ __noinline__ void raw_add(PaillierPublicKey *gpu_pub_key, cgbn_error_report_t *report, gpu_cph *ciphers_r, 
		gpu_cph *ciphers_a, gpu_cph *ciphers_b,int count) {
/**************************************************************************************
* add under encrypted text.
* in: 
*   ciphers_a, ciphers_b: encrypted a and b.
* out:
*   ciphers_r: encrypted result.
*/
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(tid>=count)
    return;
  context_t      bn_context(cgbn_report_monitor, report, tid);  
  env_cph_t          bn_env(bn_context.env<env_cph_t>());                   
  env_cph_t::cgbn_t  nsquare, r, a, b;
  env_cph_t::cgbn_wide_t r_wide;
  cgbn_load(bn_env, nsquare, &gpu_pub_key[0].nsquare);      
  cgbn_load(bn_env, a, ciphers_a + tid);      
  cgbn_load(bn_env, b, ciphers_b + tid);
  cgbn_mul_wide(bn_env, r_wide, a, b);
  cgbn_rem_wide(bn_env, r, r_wide, nsquare);
  cgbn_store(bn_env, ciphers_r + tid, r);
}

__global__ void raw_mul(PaillierPublicKey *gpu_pub_key, cgbn_error_report_t *report, gpu_cph *ciphers_r, 
		gpu_cph *ciphers_a, gpu_cph *plains_b,int count) {
/****************************************************************************************
* multiplication under encrypted text. b * a.
* in:
*   ciphers_a, plains_b: encrypted a and b.
* out:
*   ciphers_r: encrypted result.
*/
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(tid>=count)
    return;
  context_t      bn_context(cgbn_report_monitor, report, tid);  
  env_cph_t      bn_env(bn_context.env<env_cph_t>());                   
  env_cph_t::cgbn_t  n,max_int, nsquare, r, cipher, plain, neg_c, neg_scalar,tmp;               

  cgbn_load(bn_env, n, &gpu_pub_key[0].n);      
  cgbn_load(bn_env, max_int, &gpu_pub_key[0].max_int);      
  cgbn_load(bn_env, nsquare, &gpu_pub_key[0].nsquare);      
  cgbn_load(bn_env, cipher, ciphers_a + tid);      
  cgbn_load(bn_env, plain, plains_b + tid);

  cgbn_sub(bn_env, tmp, n, max_int); 
  if(cgbn_compare(bn_env, plain, tmp) >= 0 ) {
    // Very large plaintext, take a sneaky shortcut using inverses
    cgbn_modular_inverse(bn_env,neg_c, cipher, nsquare);
    cgbn_sub(bn_env, neg_scalar, n, plain);
    mont_modular_power(bn_env, r, neg_c, neg_scalar, nsquare);
  } else {
    mont_modular_power(bn_env, r, cipher, plain, nsquare);
  }
  cgbn_store(bn_env, ciphers_r + tid, r);
}


__global__ void raw_matmul(PaillierPublicKey *pub_key, cgbn_error_report_t *report, gpu_cph *ciphers_a, \
gpu_cph *plains_b, gpu_cph *ciphers_res, const uint32_t P, const uint32_t Q, const uint32_t R) {
  // size: P * Q, Q * R
  int col = blockIdx.x * blockDim.x + threadIdx.y;
  int row = blockIdx.y * blockDim.y + threadIdx.z;

  int tid = (row * R) + col;

  if (row >= P || col >= R)
   return;
  context_t bn_context(cgbn_report_monitor, report, tid);
  env_cph_t bn_env(bn_context.env<env_cph_t>());
  env_cph_t::cgbn_t n, max_int, nsquare, r, cipher, plain, neg_c, neg_scalar, tmp, tmp_res;
  env_cph_t::cgbn_wide_t tmp_wide;

  cgbn_set_ui32(bn_env, tmp_res, 1);
  cgbn_load(bn_env, n, &pub_key[0].n);
  cgbn_load(bn_env, max_int, &pub_key[0].max_int);
  cgbn_load(bn_env, nsquare, &pub_key[0].nsquare);

  for (int i = 0; i < Q; i++) {
    cgbn_load(bn_env, cipher, ciphers_a + row * Q + i);
    cgbn_load(bn_env, plain, plains_b + col * R + i); // remember b is flattened vertically.

    cgbn_sub(bn_env, tmp, n, max_int);
    if (cgbn_compare(bn_env, plain, tmp) >= 0) {
      cgbn_modular_inverse(bn_env, neg_c, cipher, nsquare);
      cgbn_sub(bn_env, neg_scalar, n, plain);
      mont_modular_power(bn_env, r, neg_c, neg_scalar, nsquare);
    } else {
      mont_modular_power(bn_env, r, cipher, plain, nsquare);
    }
    cgbn_mul_wide(bn_env, tmp_wide, tmp_res, r);
    cgbn_rem_wide(bn_env, tmp_res, tmp_wide, nsquare);
  }
  cgbn_store(bn_env, ciphers_res + row * R + col, tmp_res);
}

__global__ __noinline__ void fpn_mul(plain_t *encoding, plain_t *b, const uint32_t count, plain_t *res) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t threadNum = gridDim.x * blockDim.x;
  uint32_t start = tid * count / threadNum;
  uint32_t end = start + count / threadNum > count ? count - 1 : start + count / threadNum;
  if (start >= count)
    return;
  
  for (int i = start; i <= end; i++)
    res[i] = encoding[i] * b[i];
}
  
__global__ void raw_decrypt(PaillierPrivateKey *gpu_priv_key, PaillierPublicKey *gpu_pub_key,
	   	cgbn_error_report_t *report, gpu_cph *plains, gpu_cph *ciphers, int count) {
/*************************************************************************************
* decryption
* in:
*   ciphers: encrypted text. 2048 bits.
* out:
*   plains: decrypted plain text.
*/
  int tid=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(tid>=count)
    return;
  context_t      bn_context(cgbn_report_monitor, report, tid);
  env_cph_t          bn_env(bn_context.env<env_cph_t>());
  env_cph_t::cgbn_t  mp, mq, tmp, q_inverse, n, p, q, hp, hq, psquare, qsquare, cipher;  
  cgbn_load(bn_env, cipher, ciphers + tid);
  cgbn_load(bn_env, q_inverse, &gpu_priv_key[0].q_inverse);
  cgbn_load(bn_env, n, &gpu_pub_key[0].n);
  cgbn_load(bn_env, p, &gpu_priv_key[0].p);
  cgbn_load(bn_env, q, &gpu_priv_key[0].q);
  cgbn_load(bn_env, hp, &gpu_priv_key[0].hp);
  cgbn_load(bn_env, hq, &gpu_priv_key[0].hq);
  cgbn_load(bn_env, psquare, &gpu_priv_key[0].psquare);
  cgbn_load(bn_env, qsquare, &gpu_priv_key[0].qsquare);
  
  l_func(bn_env, mp, cipher, p, psquare, hp); 
  l_func(bn_env, mq, cipher, q, qsquare, hq); 
  
  cgbn_sub(bn_env, tmp, mp, mq);
  cgbn_mul(bn_env, tmp, tmp, q_inverse); 
  cgbn_rem(bn_env, tmp, tmp, p);
  cgbn_mul(bn_env, tmp, tmp, q);
  cgbn_add(bn_env, tmp, mq, tmp);
  cgbn_rem(bn_env, tmp, tmp, n);
  
  cgbn_store(bn_env, plains + tid, tmp);
}

void print_buffer_in_hex(char *addr, int count) {
  printf("dumping memory in hex\n");
  for (int i = 0; i < count; i++)
    printf("%x", *(addr + i) & 0xff); // remove padding.
  printf("\n");
}

void print_num_hex(char *addr, int count) {
  printf("dumping memory in hex, little endine\n");
  bool leading = false;
  for (int i = count - 1; i >= 0; i--) {
    if (*(addr + i) >= 0 && leading == false) {
      leading = true;
      printf("%x", *(addr + i) & 0xff); // remove padding.
    } else if (leading == true) {
      printf("%x", *(addr + i) & 0xff); // remove padding.
    }
  }
  printf("\n");
}

extern "C" {
PaillierPublicKey* gpu_pub_key;
PaillierPrivateKey* gpu_priv_key;
cgbn_error_report_t* err_report;

void init_pub_key(void *n, void *g, void *nsquare, void *max_int) {
  cudaMalloc(&gpu_pub_key, sizeof(PaillierPublicKey));
  cudaMemcpy((void *)&gpu_pub_key->g, g, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_pub_key->n, n, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_pub_key->nsquare, nsquare, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_pub_key->max_int, max_int, CPH_BITS/8, cudaMemcpyHostToDevice);
}

void init_priv_key(void *p, void *q, void *psquare, void *qsquare, void *q_inverse,
                   void *hp, void *hq) {
  cudaMalloc(&gpu_priv_key, sizeof(PaillierPrivateKey));
  cudaMemcpy((void *)&gpu_priv_key->p, p, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->q, q, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->psquare, psquare, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->qsquare, qsquare, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->q_inverse, q_inverse, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->hp, hp, CPH_BITS/8, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)&gpu_priv_key->hq, hq, CPH_BITS/8, cudaMemcpyHostToDevice);
}

void init_err_report() {
  cgbn_error_report_alloc(&err_report);
}

void reset() {
  cgbn_error_report_free(err_report);
  cudaFree(gpu_pub_key);
  cudaFree(gpu_priv_key);
}

void call_raw_encrypt_obfs(gpu_cph *plains_on_gpu, const uint32_t count,  \
  gpu_cph *ciphers_on_gpu, uint32_t* rand_vals_gpu) {
  // all parameters on gpu

  int TPB = 128;
  int IPB = TPB/TPI;
  int block_size = (count + IPB - 1)/IPB;
  int thread_size = TPB;
  if (rand_vals_gpu != NULL) {
    raw_encrypt_with_obfs<<<block_size, thread_size>>>(gpu_pub_key, err_report, \
      plains_on_gpu, ciphers_on_gpu, count, rand_vals_gpu);
  }
  else
    raw_encrypt<<<block_size, thread_size>>>(gpu_pub_key, err_report, plains_on_gpu,\
       ciphers_on_gpu, count);

}

void call_raw_add(gpu_cph *cipher_a, gpu_cph *cipher_b, gpu_cph *cipher_res, const uint32_t count) {
  
  int TPB = 128;
  int IPB = TPB/TPI;

  int block_size = (count + IPB - 1) / IPB;
  int thread_size = TPB;

  raw_add<<<block_size, thread_size>>>(gpu_pub_key, err_report, cipher_res, cipher_a, cipher_b, count);

}

void call_raw_mul(gpu_cph *cipher_a, plain_t *plain_b, gpu_cph *cipher_res, const uint32_t count) {
  // a is cipher, b is plain
  gpu_cph *plain_b_ext;
  
  int TPB = 128;
  int IPB = TPB/TPI;

  cudaMallocAndSet((void **)&plain_b_ext, sizeof(gpu_cph) * count);

  for (int i = 0; i < count; i++)
    cudaMemcpy(plain_b_ext + i, plain_b + i, sizeof(plain_t), cudaMemcpyDeviceToDevice);
  
  int block_size = (count + IPB - 1) / IPB;
  int thread_size = TPB;

  raw_mul<<<block_size, thread_size>>>(gpu_pub_key, err_report, cipher_res, cipher_a, \
     plain_b_ext, count);

  cudaFree(plain_b_ext);
}

void call_raw_decrypt(gpu_cph *cipher_gpu, const uint32_t count, gpu_cph *res) {
  
  int TPB = 128;
  int IPB = TPB/TPI;
  int block_size = (count + IPB - 1) / IPB;
  int thread_size = TPB;

  raw_decrypt<<<block_size, thread_size>>>(gpu_priv_key, gpu_pub_key, err_report, res, \
  cipher_gpu, count);
}


void call_raw_matmul(gpu_cph *cipher_gpu, plain_t *plain_b, gpu_cph *cipher_res, const uint32_t P,\
   const uint32_t Q, const uint32_t R) {
  gpu_cph *plain_gpu;
  cudaMallocAndSet((void **)&plain_gpu, sizeof(gpu_cph) * Q * R);
  for (int i = 0; i < Q * R; i++)
    cudaMemcpy(plain_gpu + i, plain_b + i, sizeof(plain_t), cudaMemcpyDeviceToDevice);
  
  dim3 threadPerBlock(TPI, 4, 4); // TODO: remove hardcoded.
  uint32_t x_dim = ceil((double)P/(double)threadPerBlock.x);
  uint32_t y_dim = ceil((double)R/(double)threadPerBlock.y);

  dim3 blockPerGrid(x_dim, y_dim);

  raw_matmul<<<blockPerGrid, threadPerBlock>>>(gpu_pub_key, err_report, cipher_gpu, plain_gpu, \
    cipher_res, P, Q, R);
  
  cudaFree(plain_gpu);
}


void cipher_align(PaillierEncryptedNumber *a, PaillierEncryptedNumber *b, const uint32_t count) {
  // align exponent before executing "encrypted add" operation
  // parameters:
  //   a: PEN array on cpu, b: same as a
  // steps:
  //   1. figure out whose exponent is bigger
  //   2. update exponent
  //   3. perform raw mul
  //   4. copy back to PaillierEncryptedNumber
  int *map = (int *) malloc(sizeof(int) * count);
  plain_t *cof;
  cudaMallocManaged(&cof, sizeof(plain_t) * count);
  // 1
  for (int i = 0; i < count; i++) {
    map[i] = a[i].exponent < b[i].exponent ? 0 : 1;
    cof[i] = (plain_t) pow(a[i].base, abs((int)a[i].exponent- (int)b[i].exponent));
    if (a[i].exponent < b[i].exponent)
      a[i].exponent = b[i].exponent;
    else b[i].exponent = a[i].exponent;
  }
  // dumpMem(a[0].cipher, sizeof(gpu_cph));
  gpu_cph *encoding;
  gpu_cph *res;
  
  cudaMalloc(&encoding, sizeof(gpu_cph) * count);
  cudaMalloc(&res, sizeof(gpu_cph) * count);
  for (int i = 0; i < count; i++) {
    if (map[i] == 0)
      cudaMemcpy(encoding + i, a + i, sizeof(gpu_cph), cudaMemcpyHostToDevice);
    else
      cudaMemcpy(encoding + i, b + i, sizeof(gpu_cph), cudaMemcpyHostToDevice);
  }
  // 2
  call_raw_mul(encoding, cof, res, count);
  // 3
  for (int i = 0; i < count; i++) {
    if (map[i] == 0)
      cudaMemcpy((a + i)->cipher, res + i, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
    else
      cudaMemcpy((b + i)->cipher, res + i, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
  }
  
  //..
  cudaFree(encoding);
  cudaFree(res);
  free(map);
  cudaFree(cof);
 
}

void pen_increase_exponent_to(PaillierEncryptedNumber *a, const uint32_t exponent, \
   const uint32_t count) {
  printf("enter pen\n");
  printf("count: %d\n", count);
  plain_t *cof;
  gpu_cph *cipher_gpu = NULL;
  gpu_cph *cipher_res = NULL;
  printf("malloc managed\n");
  cudaMallocManaged(&cof, sizeof(plain_t) * count);
  printf("malloc cipher_gpu\n");
  cudaMallocAndSet((void **)&cipher_gpu, sizeof(gpu_cph) * count);
  printf("malloc cipher_res\n");
  cudaMallocAndSet((void **)&cipher_res, sizeof(gpu_cph) * count);
  uint32_t base = a[0].base;
  printf("calculating cof\n");
  for (int i = 0; i < count; i++) {
    uint32_t diff = exponent >= a[i].exponent ? exponent - a[i].exponent : 0;
    cof[i] = (uint32_t) pow(base, diff);
  }
  
  printf("extract Pen\n");
  extractPen(cipher_gpu, a, count, HostToDevice);

  printf("finish extract pen\n");
  
  call_raw_mul(cipher_gpu, cof, cipher_res, count);
  printf("finish call raw mul\n");
  for (int i = 0; i < count; i++) {
    cudaMemcpy((a + i)->cipher, cipher_res + i, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
    a[i].exponent = exponent;
  }

  printf("finish copy back\n");

  cudaFree(cipher_gpu);
  cudaFree(cipher_res);
  cudaFree(cof);
}

void fpn_increase_exponent_to(FixedPointNumber *a, const uint32_t exponent, const uint32_t count) {
  plain_t *fpn_gpu;
  plain_t *cof;
  plain_t *res;

  cudaMallocAndSet((void **)&fpn_gpu, sizeof(plain_t) * count);
  cudaMallocAndSet((void **)&res, sizeof(plain_t) * count);
  cudaMallocManaged((void **)&cof, sizeof(plain_t) * count);

  for (int i = 0; i < count; i++)
    cudaMemcpy(fpn_gpu + i, &a[i].encoding, sizeof(plain_t), cudaMemcpyHostToDevice);
  
  uint32_t base = a[0].base;
  for (int i = 0; i < count; i++)
    cof[i] = (plain_t) pow(base, exponent - a[i].exponent);
  uint32_t thread_size = 1024;
  uint32_t block_size = ceil((double)count/(double)thread_size);
  fpn_mul<<<block_size, thread_size>>>(fpn_gpu, cof, count, res);
  for (int i = 0; i < count; i++) {
    cudaMemcpy(&a[i].encoding, res + i, sizeof(plain_t), cudaMemcpyDeviceToHost);
    a[i].exponent = exponent;
  }

  cudaFree(fpn_gpu);
  cudaFree(cof);
  cudaFree(res);

}


void cipher_add_cipher(PaillierEncryptedNumber *a, PaillierEncryptedNumber *b, \
  PaillierEncryptedNumber *r, const uint32_t count) {
  // perform encrypted add on PEN
  // parameters:
  //   a, b: add numbers, on cpu. c: result on cpu
  // steps:
  //   1. align
  //   2. perform raw add
  //   3. copy to cpu
  cipher_align(a, b, count);
  gpu_cph *cipher_a;
  gpu_cph *cipher_b;
  gpu_cph *cipher_res;
  cudaMallocAndSet((void **)&cipher_a, sizeof(gpu_cph) * count);
  cudaMallocAndSet((void **)&cipher_b, sizeof(gpu_cph) * count);
  cudaMallocAndSet((void **)&cipher_res, sizeof(gpu_cph) * count);

  extractPen(cipher_a, a, count, HostToDevice);
  extractPen(cipher_b, b, count, HostToDevice);

  call_raw_add(cipher_a, cipher_b, cipher_res, count);
  penFromBuffer(r, cipher_res, count, DeviceToHost);

  for (int i = 0; i < count; i++) {
    r[i].exponent = a[i].exponent;
    r[i].base = a[i].base;
  }
  
}


void plain_mul_cipher(FixedPointNumber *b, PaillierEncryptedNumber *a, \
   PaillierEncryptedNumber *r, const int count) {
  // perform encrypted multiplication
  // parameters:
  //   b: coefficients, plain text on cpu
  //   a: encrypted num of arrays
  //   r: result, all on cpu
  // steps:
  //   1. perform raw mul
  //   2. add exponent together.
  //   3. copy to cpu
  plain_t *plain_gpu;
  gpu_cph *cipher_gpu;
  gpu_cph *cipher_res;
  cudaMallocAndSet((void **)&plain_gpu, sizeof(plain_t) * count);
  cudaMallocAndSet((void **)&cipher_gpu, sizeof(gpu_cph) * count);
  cudaMallocAndSet((void **)&cipher_res, sizeof(gpu_cph) * count);

  extractPen(cipher_gpu, a, count, HostToDevice);
  for (int i = 0; i < count; i++)
    cudaMemcpy(plain_gpu + i, &((b + i)->encoding), sizeof(plain_t), cudaMemcpyHostToDevice);
  
  call_raw_mul(cipher_gpu, plain_gpu, cipher_res, count);
  for (int i = 0; i < count; i++) {
    cudaMemcpy((r + i)->cipher, cipher_res + i, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
    (r + i)->base = (a + i)->base;
    (r + i)->exponent = (a + i)->exponent + (b + i)->exponent;
  }

  cudaFree(plain_gpu);
  cudaFree(cipher_gpu);
  cudaFree(cipher_res);
}


void encrypt(FixedPointNumber *plain, gpu_cph *r, const uint32_t count, const bool obf) {
  // encrypt function.
  // parameters:
  //   plain: in cpu
  //   r : in cpu
  // steps:
  //   1. copy encoding to gpu
  //   2. perform raw encrypt
  //   3. copy back to result(on cpu)
  gpu_cph *raw_plain_gpu;
  gpu_cph *raw_cipher_gpu;
  unsigned int *obfs = NULL;
  cudaMalloc(&raw_plain_gpu, sizeof(gpu_cph) * count);
  cudaMalloc(&raw_cipher_gpu, sizeof(gpu_cph) * count);
  memset(r, 0, sizeof(gpu_cph) * count);
  cudaMemset(raw_plain_gpu, 0, sizeof(gpu_cph) * count);
  
  for (int i = 0; i < count; i++) {
    cudaMemcpy(raw_plain_gpu + i, &plain[i].encoding, sizeof(plain_t), cudaMemcpyHostToDevice);
  }
  
  if (obf) {
    cudaMallocManaged(&obfs, sizeof(unsigned int) * count);
    for (int i = 0; i < count; i++) obfs[i] = rand();
  }
   
  call_raw_encrypt_obfs(raw_plain_gpu, count, raw_cipher_gpu, obfs);

  cudaMemcpy(r, raw_cipher_gpu, sizeof(gpu_cph) * count, cudaMemcpyDeviceToHost);

  cudaFree(raw_plain_gpu);
  cudaFree(raw_cipher_gpu);
  if (obf) cudaFree(obfs);
}

void encrypt_async(FixedPointNumber *plain, gpu_cph *r, const uint32_t count, const bool obf) {
  gpu_cph *raw_plain_gpu;
  gpu_cph *raw_cipher_gpu;
  unsigned int *obfs = NULL;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaMalloc(&raw_plain_gpu, sizeof(gpu_cph) * count);
  cudaMalloc(&raw_cipher_gpu, sizeof(gpu_cph) * count);
  memset(r, 0, sizeof(gpu_cph) * count);
  cudaMemset(raw_plain_gpu, 0, sizeof(gpu_cph) * count);
  
  for (int i = 0; i < count; i++) {
    cudaMemcpyAsync(raw_plain_gpu + i, &plain[i].encoding, sizeof(plain_t), cudaMemcpyHostToDevice, stream);
  }
  
  if (obf) {
    cudaMallocManaged(&obfs, sizeof(unsigned int) * count);
    for (int i = 0; i < count; i++) obfs[i] = rand();
  }
   
  // call_raw_encrypt_obfs(raw_plain_gpu, count, raw_cipher_gpu, obfs);
  int TPB = 128;
  int IPB = TPB/TPI;
  int block_size = (count + IPB - 1)/IPB;
  int thread_size = TPB;

  if (obf)
    raw_encrypt_with_obfs<<<block_size, thread_size, 0, stream>>>(gpu_pub_key, err_report, \
      raw_plain_gpu, raw_cipher_gpu, count, obfs);
  else
    raw_encrypt<<<block_size, thread_size, 0, stream>>>(gpu_pub_key, err_report, raw_plain_gpu,\
      raw_cipher_gpu, count);

  cudaMemcpyAsync(r, raw_cipher_gpu, sizeof(gpu_cph) * count, cudaMemcpyDeviceToHost, stream);

  cudaFree(raw_plain_gpu);
  cudaFree(raw_cipher_gpu);
  if (obf) cudaFree(obfs);
  cudaStreamDestroy(stream);
}


void decrypt(PaillierEncryptedNumber *cipher, gpu_cph *r, const uint32_t count) {
  // perform decrypt
  // parameters:
  //   cipher: in cpu
  //   r : in cpu
  // steps:
  //   1. copy to gpu
  //   2. perform raw decrypt
  //   3. copy back to cpu
  gpu_cph *raw_cipher_gpu;
  gpu_cph *res_gpu;
  cudaMalloc(&raw_cipher_gpu, sizeof(gpu_cph) * count);
  cudaMalloc(&res_gpu, sizeof(gpu_cph) * count);

  for (int i = 0; i < count; i++)
    cudaMemcpy(raw_cipher_gpu + i, cipher[i].cipher, sizeof(gpu_cph), cudaMemcpyHostToDevice);
  
  call_raw_decrypt(raw_cipher_gpu, count, res_gpu);
  cudaMemcpy(r, res_gpu, sizeof(gpu_cph) * count, cudaMemcpyDeviceToHost);

  cudaFree(raw_cipher_gpu);
  cudaFree(res_gpu);
}


void sum(PaillierEncryptedNumber *cipher, PaillierEncryptedNumber *res, const uint32_t count) {
  // sum
  // parameters:
  //  cipher: in cpu
  //  r: in cpu
  // steps:
  //  1. copy to gpu
  //  2. align
  //  3. perform raw add on half
  //  4. loop until only one left

  // if count is odd, add one
  printf("count: %d\n", count);
  int32_t num_elem = count % 2 == 1 ? count + 1 : count;
  gpu_cph *ciphers_buf[2];
  plain_t *inc;
  cudaMallocAndSet((void **)&ciphers_buf[0], sizeof(gpu_cph) * num_elem);
  cudaMallocAndSet((void **)&ciphers_buf[1], sizeof(gpu_cph) * num_elem);
  cudaMallocManaged((void **)&inc, sizeof(plain_t) * count);

  uint32_t max_exponent = 0;
  for (int i = 0; i < count; i++)
    max_exponent = max_exponent < cipher[i].exponent ? cipher[i].exponent : max_exponent;
  for (int i = 0; i < count; i++) {
    inc[i] = (int32_t) pow(cipher[i].base, max_exponent - cipher[i].exponent);
  }

  extractPen(ciphers_buf[0], cipher, count, HostToDevice);
  call_raw_mul(ciphers_buf[0], inc, ciphers_buf[1], count);
  
  if (count % 2 == 1)
    cudaMemset(ciphers_buf[1] + num_elem - 1, 1, 1);
    
  uint32_t dst_index = 0;
  gpu_cph *dst_buf;
  gpu_cph *src_buf;
  for (int i = num_elem / 2; i >= 1; i /= 2) {
    dst_buf = ciphers_buf[dst_index % 2];
    src_buf = ciphers_buf[(dst_index % 2 + 1) % 2];
    printf("check it %d\n", i);
    call_raw_add(src_buf, src_buf + i, dst_buf, i);
    dst_index += 1;
  }

  cudaMemcpy(res->cipher, dst_buf, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
  res->base = cipher[0].base;
  res->exponent = max_exponent;

  cudaFree(ciphers_buf[0]);
  cudaFree(ciphers_buf[1]);
  cudaFree(inc);
}

void matmul(PaillierEncryptedNumber *cipher_a, FixedPointNumber *plain_b, PaillierEncryptedNumber *r,\
   const uint32_t P, const uint32_t Q, const uint32_t R) {
  // perform matrix multiplication.
  // parameters:
  //  cipher_a: ciphers in cpu
  //  plain_b: plains in cpu
  //  r: result in cpu
  //  cipher_a has shape P * Q
  //  plain_b has shape Q * R, b is vertically flattened.
  // steps:
  //  1. copy cipher_a to GPU, plain_b to GPU
  //  2. align ciphers
  //  3. call_raw_matmul
  //  4. copy back to CPU with corresponding exponent
  gpu_cph *cipher_gpu = NULL;
  plain_t *plain_gpu = NULL;
  gpu_cph *cipher_res = NULL;
  
  // find the largest exponent
  uint32_t max_exponent = 0;
  for (int i = 0; i < P * Q; i++)
    max_exponent = max_exponent < cipher_a[i].exponent ? cipher_a[i].exponent : max_exponent;
  
  for (int i = 0; i < Q * R; i++)
    max_exponent = max_exponent < plain_b[i].exponent ? plain_b[i].exponent : max_exponent;
  
  // increase exponent
  pen_increase_exponent_to(cipher_a, max_exponent, P * Q);
  fpn_increase_exponent_to(plain_b, max_exponent, Q * R);

  cudaMallocAndSet((void **)&cipher_gpu, sizeof(gpu_cph) * P * Q);
  cudaMallocAndSet((void **)&plain_gpu, sizeof(plain_t) * Q * R);
  cudaMallocAndSet((void **)&cipher_res, sizeof(gpu_cph) * P * R);

  extractPen(cipher_gpu, cipher_a, P * Q, HostToDevice);
  for (int i = 0; i < Q * R; i++)
    cudaMemcpy(plain_gpu + i, &plain_b[i].encoding, sizeof(plain_t), cudaMemcpyHostToDevice);
  
  call_raw_matmul(cipher_gpu, plain_gpu, cipher_res, P, Q, R);

  for (int i = 0; i < P * R; i++) {
    cudaMemcpy(r[i].cipher, cipher_res + i, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
    r[i].exponent = 2 * max_exponent;
    r[i].base = cipher_a[0].base;
  }

}

void batch_matmul(PaillierEncryptedNumber *a, FixedPointNumber *b, PaillierEncryptedNumber *r, \
  uint32_t *size_a, uint32_t *size_b, const uint32_t dim) {
  // perform matmul
  // size_a: list of dimensions. i.e., 3,2,6,4
  // size_b: list of dimentions. i.e., 1,2,9,4
  // dim, dimentions of a and b, i.e. 4
  gpu_cph *cipher_a;
  gpu_cph *plain_b;
  gpu_cph *cipher_res;

  uint32_t max_exponent = 0;
  uint32_t count_a = 1;
  uint32_t count_b = 1;
  uint32_t count_res = 1;
  uint32_t P = size_a[dim - 2];
  uint32_t Q = size_a[dim - 1];
  uint32_t R = size_b[dim - 2];
  cudaStream_t streams[8];
  const uint32_t NUM_STREAMS = 8;
  for (int i = 0; i < NUM_STREAMS; i++) cudaStreamCreate(&streams[i]);

  for (int i = 0; i < dim; i++) count_a *= size_a[i];
  for (int i = 0; i < dim; i++) count_b *= size_b[i];
  for (int i = 0; i < dim - 2; i++) count_res *= size_a[i];
  count_res *= P * R;

  uint32_t stride_a = P * Q;
  uint32_t stride_b = Q * R;
  uint32_t stride_res = P * R;

  cudaMallocAndSet((void **)&cipher_a, sizeof(gpu_cph) * count_a);
  cudaMallocAndSet((void **)&cipher_res, sizeof(gpu_cph) * count_res);
  cudaMallocAndSet((void **)&plain_b, sizeof(plain_t) * count_b);
  
  for (int i = 0; i < count_a; i++)
    max_exponent = max_exponent < a[i].exponent ? a[i].exponent : max_exponent;
  for (int i = 0; i < count_b; i++)
    max_exponent = max_exponent < b[i].exponent ? b[i].exponent : max_exponent;
  
  // pen_increase_exponent_to()
  // fpn_increase_exponent_to()

  extractPen(cipher_a, a, count_a, HostToDevice);
  for (int i = 0; i < count_b; i++)
    cudaMemcpy(plain_b + i, &b[i].encoding, sizeof(plain_t), cudaMemcpyHostToDevice);

  uint32_t a_start = 0;
  uint32_t b_start = 0;
  uint32_t res_start = 0;

  dim3 threadPerBlock(TPI, 4, 4); // TODO: remove hardcoded.
  uint32_t x_dim = ceil((double)P/(double)threadPerBlock.x);
  uint32_t y_dim = ceil((double)R/(double)threadPerBlock.y);

  dim3 blockPerGrid(x_dim, y_dim);

  for (int i = 0; i < dim; i++) {
    uint32_t dim_a = size_a[i];
    uint32_t dim_b = size_b[i];
    uint32_t loop_dim = dim_a < dim_b ? dim_b : dim_a;
    bool brdcst_a = dim_a == 1, brdcst_b = dim_b == 1;
    
    for (int j = 0; j < loop_dim; j++) {
      // call raw matmul
      cudaStreamSynchronize(streams[j % NUM_STREAMS]);
      raw_matmul<<<blockPerGrid, threadPerBlock, 0, streams[j % NUM_STREAMS]>>>(gpu_pub_key, \
      err_report, cipher_a + a_start, plain_b + b_start, cipher_res + res_start, P, Q, R);
      if (!brdcst_a || j == loop_dim - 1) a_start += stride_a;
      if (!brdcst_b || j == loop_dim - 1) b_start += stride_b;
      res_start += stride_res;
    }
  }
  cudaDeviceSynchronize();

  for (int i = 0; i < count_res; i++) {
    cudaMemcpy(r[i].cipher, cipher_res + i, sizeof(gpu_cph), cudaMemcpyDeviceToHost);
    r[i].exponent = max_exponent * 2;
    r[i].base = a[0].base;
  }

  for (int i = 0; i < NUM_STREAMS; i++) cudaStreamDestroy(streams[i]);
  cudaFree(cipher_a);
  cudaFree(plain_b);
  cudaFree(cipher_res);
}

}// extern "C"
