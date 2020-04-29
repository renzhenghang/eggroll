#ifndef FIXEDPOINTNUMEBR_H_INCLUDED
#define FIXEDPOINTNUMEBR_H_INCLUDED
#include <stdint.h>
#include "cgbn/cgbn.h"
// #define PLAIN_INT64
#define PLAIN_BITS 1024

#ifdef PLAIN_INT32
  typedef uint32_t plain_t;
#else
  #ifdef PLAIN_INT64
  typedef uint64_t plain_t;
  #else
    #ifdef PLAIN_BITS
    typedef cgbn_context_t<4> plain_context;
    typedef cgbn_env_t<plain_context, PLAIN_BITS> env_pln_t;
    typedef cgbn_mem_t<PLAIN_BITS> plain_t;
    #endif
  #endif
#endif
struct FixedPointNumber {
  plain_t encoding;
  int32_t exponent;
  int32_t base;
};

#endif