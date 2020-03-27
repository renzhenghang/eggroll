#ifndef FIXEDPOINTNUMEBR_H_INCLUDED
#define FIXEDPOINTNUMEBR_H_INCLUDED
#include <stdint.h>
#define PLAIN_INT64

#ifdef PLAIN_INT32
  typedef plain_t int32_t;
#else
  #ifdef PLAIN_INT64
  typedef plain_t int64_t;
  #endif
#endif
struct FixedPointNumber {
  plain_t encoding;
  int32_t exponent;
  int32_t base;
};

#endif