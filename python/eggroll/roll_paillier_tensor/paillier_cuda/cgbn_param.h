#ifndef PAILLIER_CUDA_CGBN_PARAM
#define PAILLIER_CUDA_CGBN_PARAM
#include "cgbn/cgbn.h"

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define PAILLIER_TPI 32
#define CPH_BITS 2048 // cipher bits
#define MAX_RAND_SEED 4294967295U
#define WINDOW_BITS 5
// helpful typedefs for the kernel
typedef cgbn_context_t<PAILLIER_TPI>         context_t;
typedef cgbn_env_t<context_t, CPH_BITS> env_cph_t;
typedef cgbn_mem_t<CPH_BITS> gpu_cph;


#endif