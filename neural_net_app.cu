#include <stdio.h>
extern REAL *d_sunspots; 

__global__ void normalizeSunspotsKernel(REAL *sunspots, REAL min, REAL max, int size) {
    // Kernel-level printf is supported in CUDA
    if (threadIdx.x == 0) { // Print only once, not for every thread
        printf("normalizeSunspotsKernel\n");
    }
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        sunspots[idx] = ((sunspots[idx] - min) / (max - min)) * (HI - LO) + LO;
    }
}

// Make sure to pass the device pointer as a parameter
void normalizeSunspots(REAL *d_sunspots, REAL min, REAL max, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    // Call the kernel with the device pointer
    normalizeSunspotsKernel<<<numBlocks, blockSize>>>(d_sunspots, min, max, size);
    // Always check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}
