#ifndef NEURAL_NET_KERNEL_H
#define NEURAL_NET_KERNEL_H
#include <curand_kernel.h> 
#include "neural_net_types.h" // This include is necessary for REAL type definition
 // This include is necessary for curandState type definition


// Declaration of the kernel function that will be implemented in .cu file
__global__ void normalizeSunspotsKernel(REAL *sunspots, REAL min, REAL max, int size);
__global__ void initRandomStates(curandState *state, unsigned long seed, int n);
__global__ void setRandomWeights(curandState *state, REAL *weights, int totalWeights);
__global__ void CalculateError(REAL *d_Sunspots, REAL mean, REAL *d_TrainError, REAL *d_TestError, int M1, int trainLwb, int trainUpb, int testLwb, int testUpb);
__global__ void PropagateLayerKernel(REAL* d_UpperWeights, REAL* d_LowerOutput, REAL* d_UpperOutput, int lowerUnits, int upperUnits, REAL gain);

__device__ void atomicAddDouble(REAL* address, REAL val);

// Declaration of the function that launches the kernel
void normalizeSunspotsLaunch(REAL *d_sunspots, REAL min, REAL max, int size);



#endif // NEURAL_NET_KERNEL_H
