#include <stdio.h>

__global__ void normalizeSunspotsKernel(REAL *sunspots, REAL min, REAL max, int size) {
    // put a print to check if this is being called
    printf("normalizeSunspotsKernel\n");
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        sunspots[idx] = ((sunspots[idx] - min) / (max - min)) * (HI - LO) + LO;
    }
}


void normalizeSunspots(REAL *sunspots, REAL min, REAL max, int size) {
    int blockSize = 256;  
    int numBlocks = (size + blockSize - 1) / blockSize;
    normalizeSunspotsKernel<<<numBlocks, blockSize>>>(d_sunspots, min, max, size);

}