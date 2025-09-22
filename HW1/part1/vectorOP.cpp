#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  // Becasue, when N is not a multiple of VECTOR_WIDTH, the last iteration
  // will access out-of-bound memory locations.
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    int width = (N - i >= VECTOR_WIDTH) ? VECTOR_WIDTH : (N - i); // avoid out-of-bound access
    // All ones
    //maskAll = _pp_init_ones();
    maskAll = _pp_init_ones(width);

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  // assume exp >= 0
  
  __pp_vec_float vx;
  __pp_vec_int vy;

  __pp_vec_float res;
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int ones = _pp_vset_int(1);

  __pp_mask maskAll, maskIsZero, maskIsNotZero;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    int width = (N - i >= VECTOR_WIDTH) ? VECTOR_WIDTH : (N - i);
    // init mask
    maskAll = _pp_init_ones(width);
    maskIsZero = _pp_init_ones(0);
    
    // load
    _pp_vload_float(vx, values + i, maskAll);
    _pp_vload_int(vy, exponents + i, maskAll);
    _pp_vset_float(res, 1.f, maskAll); // default res = 1.f
    
    // check exp as 0 or not
    _pp_veq_int(maskIsZero, vy, zero, maskAll);

    // if exp != 0
    maskIsNotZero = _pp_mask_not(maskIsZero);
  
    while(_pp_cntbits(maskIsNotZero) > 0){
      _pp_vmult_float(res, res, vx, maskIsNotZero); // res *= x
      _pp_vsub_int(vy, vy, ones, maskIsNotZero); // exp -= 1
      _pp_veq_int(maskIsZero, vy, zero, maskIsNotZero);
      maskIsNotZero = _pp_mask_not(maskIsZero);
    }

    // clamp
    __pp_vec_float limit = _pp_vset_float(9.999999f);
    __pp_mask maskIsOverLimit = _pp_init_ones(0);
    _pp_vgt_float(maskIsOverLimit, res, limit, maskAll);
    _pp_vset_float(res, 9.999999f, maskIsOverLimit);
    
    // store
    _pp_vstore_float(output + i, res, maskAll);

  }

}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
// So, I don't need to worry about out-of-bound access
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  __pp_vec_float x;
  __pp_vec_float sum = _pp_vset_float(0.f);
  __pp_mask maskAll;

  float res = 0.0;


  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones();
    // load new vars
    _pp_vload_float(x, values + i, maskAll);
    // accum into sum
    _pp_vadd_float(sum,sum,x,maskAll);
  }

  // Tree reduction: O(log2(VECTOR_WIDTH))
  __pp_vec_float temp = sum;
  int remaining = VECTOR_WIDTH;

  while (remaining > 1) {
    _pp_hadd_float(temp, temp);
    remaining /= 2;
  }

  res = temp.value[0];

  return res;
}