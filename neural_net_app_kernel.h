#ifndef NEURAL_NET_KERNEL_H
#define NEURAL_NET_KERNEL_H

#include "neural_net_types.h" // This include is necessary for REAL type definition

// Declaration of the kernel function that will be implemented in .cu file
__global__ void normalizeSunspotsKernel(REAL *sunspots, REAL min, REAL max, int size);

__global__ void PropagateLayerKernel(REAL *lowerOutput, REAL *upperOutput, REAL *weight, int lowerUnits, int upperUnits, REAL gain);

// Declaration of the function that launches the kernel
void normalizeSunspotsLaunch(REAL *d_sunspots, REAL min, REAL max, int size);

void PropagateLayerLaunch(REAL *LowerOutput, REAL *UpperOutput, REAL *Weight, int LowerUnits, int UpperUnits, REAL Gain);

#endif // NEURAL_NET_KERNEL_H
