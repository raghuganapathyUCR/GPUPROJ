#ifndef NEURAL_NET_KERNEL_H
#define NEURAL_NET_KERNEL_H
#include <curand_kernel.h> 
#include "neural_net_types.h" 
__global__ void normalizeSunspotsKernel(REAL *sunspots, REAL min, REAL max, int size);
__global__ void initRandomStates(curandState *state, unsigned long seed, int n);
__global__ void setRandomWeights(curandState *state, REAL *weights, int totalWeights);
__global__ void CalculateError(REAL *d_Sunspots, REAL mean, REAL *d_TrainError, REAL *d_TestError, int M1, int trainLwb, int trainUpb, int testLwb, int testUpb);
__global__ void PropagateLayerKernel(REAL* d_UpperWeights, REAL* d_LowerOutput, REAL* d_UpperOutput, int lowerUnits, int upperUnits, REAL gain);

__device__ void atomicAddDouble(REAL* address, REAL val);


void normalizeSunspotsLaunch(REAL *d_sunspots, REAL min, REAL max, int size);



#endif // NEURAL_NET_KERNEL_H
