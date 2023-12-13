#ifndef NEURAL_NET_KERNEL_H
#define NEURAL_NET_KERNEL_H

#include "neural_net_types.h" // This include is necessary for REAL type definition

// Declaration of the kernel function that will be implemented in .cu file
__global__ void normalizeSunspotsKernel(REAL *sunspots, REAL min, REAL max, int size);


// Declaration of the function that launches the kernel
void normalizeSunspotsLaunch(REAL *d_sunspots, REAL min, REAL max, int size);


#endif // NEURAL_NET_KERNEL_H
