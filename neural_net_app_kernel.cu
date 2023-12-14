#include <stdio.h>
#include "neural_net_app_kernel.h"

#include <curand_kernel.h>
#undef N
#include "neural_net_constants.h"

__global__ void normalizeSunspotsKernel(REAL *sunspots, REAL min, REAL max, int size)
{
  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        sunspots[idx] = ((sunspots[idx] - min) / (max - min)) * (HI - LO) + LO;
    }
}


__global__ void initRandomStates(curandState *state, unsigned long seed, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void setRandomWeights(curandState *state, REAL *weights, int totalWeights) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < totalWeights) {
        weights[id] = curand_uniform(&state[id]) * (HI - LO) + LO;
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



__global__ void CalculateError(REAL *d_Sunspots, REAL mean, REAL *d_TrainError, REAL *d_TestError, int M1, int trainLwb, int trainUpb, int testLwb, int testUpb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    REAL Out, Err;

    // Calculate Train Error
    for (int Year = trainLwb + idx; Year <= trainUpb; Year += stride) {
        for (int i = 0; i < M1; i++) {
            Out = d_Sunspots[Year + i];
            Err = mean - Out;
            atomicAddDouble(d_TrainError, 0.5 * sqr(Err));
        }
    }

    // Calculate Test Error
    for (int Year = testLwb + idx; Year <= testUpb; Year += stride) {
        for (int i = 0; i < M1; i++) {
            Out = d_Sunspots[Year + i];
            Err = mean - Out;
            atomicAddDouble(d_TestError, 0.5 * sqr(Err));
        }
    }
}

__global__ void PropagateLayerKernel(REAL* d_UpperWeights, REAL* d_LowerOutput, REAL* d_UpperOutput, int lowerUnits, int upperUnits, REAL gain) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i <= upperUnits) {
        REAL sum = 0;
        for (int j = 0; j <= lowerUnits; j++) {
            sum += d_UpperWeights[i * (lowerUnits + 1) + j] * d_LowerOutput[j];
        }
        d_UpperOutput[i] = 1 / (1 + exp(-gain * sum));
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




