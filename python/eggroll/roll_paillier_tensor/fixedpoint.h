#ifndef FIXEDPOINTNUMEBR_H_INCLUDED
#define FIXEDPOINTNUMEBR_H_INCLUDED
#include <stdint.h>
#define PLAIN_INT64 1

#ifdef PLAIN_INT32
  typedef uint32_t plain_t;
#else
  #ifdef PLAIN_INT64
  typedef uint64_t plain_t ;
  #endif
#endif
struct FixedPointNumber {
  plain_t encoding;
  int32_t exponent;
  int32_t base;
};

#endif