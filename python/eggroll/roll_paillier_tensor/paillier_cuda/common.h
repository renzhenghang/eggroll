#ifndef PAILLIER_CUDA_COMMON
#define PAILLIER_CUDA_COMMON
#include "cgbn/cgbn.h"
#include <stdint.h>
#include <cuda.h>

// template <uint32_t BIT, uint32_t TPI>
// class CGBNContext {
// public:
//   CGBNContext() {
//     _TPI = TPI;
//     _BIT = BIT;
//   }
// private:
//   uint32_t _TPI;
//   uint32_t _BIT;
// };

template<unsigned int _BITS, unsigned int _TPI>
__device__ __forceinline__ 
void mont_modular_power(cgbn_env_t<context_t, _BITS> &bn_env,   \
    typename cgbn_env_t<context_t, _BITS>::cgbn_t &result,      \
	const typename cgbn_env_t<context_t, _BITS>::cgbn_t &x,     \
    const typename cgbn_env_t<context_t, _BITS>::cgbn_t &power, \
	const typename cgbn_env_t<context_t, _BITS>::cgbn_t &modulus) {
/************************************************************************************
* calculate x^power mod modulus with montgomery multiplication.
* input: x, power, modulus.
* output: result
* requirement: x < modulus and modulus is an odd number.
*/

  typename cgbn_env_t<context_t, _BITS>::cgbn_t  t, starts;
  int32_t      index, position, leading;
  uint32_t     mont_inv;
  typename cgbn_env_t<context_t, _BITS>::cgbn_local_t odd_powers[1<<WINDOW_BITS-1];

  // find the leading one in the power
  leading=CPH_BITS-1-cgbn_clz(bn_env, power);
  if(leading>=0) {
    // convert x into Montgomery space, store in the odd powers table
    mont_inv=cgbn_bn2mont(bn_env, result, x, modulus);
    
    // compute t=x^2 mod modulus
    cgbn_mont_sqr(bn_env, t, result, modulus, mont_inv);
    
    // compute odd powers window table: x^1, x^3, x^5, ...
    cgbn_store(bn_env, odd_powers, result);
    #pragma nounroll
    for(index=1;index<(1<<WINDOW_BITS-1);index++) {
      cgbn_mont_mul(bn_env, result, result, t, modulus, mont_inv);
      cgbn_store(bn_env, odd_powers+index, result);
    }

    // starts contains an array of bits indicating the start of a window
    cgbn_set_ui32(bn_env, starts, 0);

    // organize p as a sequence of odd window indexes
    position=0;
    while(true) {
      if(cgbn_extract_bits_ui32(bn_env, power, position, 1)==0)
        position++;
      else {
        cgbn_insert_bits_ui32(bn_env, starts, starts, position, 1, 1);
        if(position+WINDOW_BITS>leading)
          break;
        position=position+WINDOW_BITS;
      }
    }

    // load first window.  Note, since the window index must be odd, we have to
    // divide it by two before indexing the window table.  Instead, we just don't
    // load the index LSB from power
    index=cgbn_extract_bits_ui32(bn_env, power, position+1, WINDOW_BITS-1);
    cgbn_load(bn_env, result, odd_powers+index);
    position--;
    
    // Process remaining windows 
    while(position>=0) {
      cgbn_mont_sqr(bn_env, result, result, modulus, mont_inv);
      if(cgbn_extract_bits_ui32(bn_env, starts, position, 1)==1) {
        // found a window, load the index
        index=cgbn_extract_bits_ui32(bn_env, power, position+1, WINDOW_BITS-1);
        cgbn_load(bn_env, t, odd_powers+index);
        cgbn_mont_mul(bn_env, result, result, t, modulus, mont_inv);
      }
      position--;
    }
    
    // convert result from Montgomery space
    cgbn_mont2bn(bn_env, result, result, modulus, mont_inv);
  }
  else {
    // p=0, thus x^p mod modulus=1
    cgbn_set_ui32(bn_env, result, 1);
  }
}

#endif