#include <stdio.h>
#include "neural_net_app_kernel.h"
#include "neural_net_constants.h"

__global__ void normalizeSunspotsKernel(REAL *sunspots, REAL min, REAL max, int size)
{
    // Kernel-level printf is supported in CUDA
    if (threadIdx.x == 0)
    { // Print only once, not for every thread
        printf("normalizeSunspotsKernel\n");
    }
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        sunspots[idx] = ((sunspots[idx] - min) / (max - min)) * (HI - LO) + LO;
    }
}

__global__ void PropagateLayerKernel(REAL *lowerOutput, REAL *upperOutput, REAL *weight, int lowerUnits, int upperUnits, REAL gain)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < upperUnits)
    {
        REAL Sum = 0;
        for (int j = 0; j < lowerUnits; j++)
        {
            Sum += weight[i * lowerUnits + j] * lowerOutput[j];
        }
        upperOutput[i] = 1 / (1 + exp(-gain * Sum));
    }
}

// Make sure to pass the device pointer as a parameter
void normalizeSunspotsLaunch(REAL *d_sunspots, REAL min, REAL max, int size)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    // Call the kernel with the device pointer
    normalizeSunspotsKernel<<<numBlocks, blockSize>>>(d_sunspots, min, max, size);
    // Always check for kernel launch error
}

void PropagateLayerLaunch(REAL *LowerOutput, REAL *UpperOutput, REAL *Weight, int LowerUnits, int UpperUnits, REAL Gain) {
    int blockSize = 256; // Example block size, adjust based on your needs
    int numBlocks = (UpperUnits + blockSize - 1) / blockSize;

    // Launch the kernel with the correct variables
    PropagateLayerKernel<<<numBlocks, blockSize>>>(LowerOutput, UpperOutput, Weight, LowerUnits, UpperUnits, Gain);
    cudaDeviceSynchronize();
}
