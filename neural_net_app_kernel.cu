#include <stdio.h>
#include "neural_net_app_kernel.h"
#include "neural_net_constants.h"

__global__ void normalizeSunspotsKernel(REAL *sunspots, REAL min, REAL max, int size)
{
  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        sunspots[idx] = ((sunspots[idx] - min) / (max - min)) * (HI - LO) + LO;
    }
}
__device__ void atomicAddDouble(REAL* address, REAL val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
}

__global__ void ComputeOutputErrorKernel(REAL *d_Output, REAL *d_Target, REAL *d_Error, REAL gain, int units, REAL *d_NetError) {
    int i = threadIdx.x + 1;
    if (i <= units) {
        REAL Out = d_Output[i];
        REAL Err = d_Target[i-1] - Out;
        d_Error[i] = gain * Out * (1 - Out) * Err;
        atomicAddDouble(d_NetError, 0.5 * Err * Err);
    }
}



// Make sure to pass the device pointer as a parameter
void normalizeSunspotsLaunch(REAL *d_sunspots, REAL min, REAL max, int size)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    // Call the kernel with the device pointer
    normalizeSunspotsKernel<<<numBlocks, blockSize>>>(d_sunspots, min, max, size);
}



