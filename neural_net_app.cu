#ifndef NEURAL_NET_APP_H
#define NEURAL_NET_APP_H

#include "neural_net_types.h"
#include "neural_net_constants.h"
#include "neural_net_functions.h"
// #include "neural_net_app_kernel.cu"
#include "neural_net_app_kernel.h"
extern REAL *d_sunspots;



void NormalizeSunspots()
{
    REAL Min, Max;

    // Calculate Min and Max
    Min = MAX_REAL;
    Max = MIN_REAL;
    for (INT Year = 0; Year < NUM_YEARS; Year++) {
        Min = MIN(Min, Sunspots[Year]);
        Max = MAX(Max, Sunspots[Year]);
    }

    // Call the CUDA function for normalization
    normalizeSunspotsLaunch(d_sunspots, Min, Max, NUM_YEARS);

}


void InitializeApplication(NET* Net) {
    Net->Alpha = 0.5;
    Net->Eta   = 0.05;
    Net->Gain  = 1;

    REAL *d_Sunspots, *d_TrainError, *d_TestError;

    // Allocate memory and copy data to GPU
    cudaMalloc(&d_Sunspots, NUM_YEARS * sizeof(REAL));
    cudaMemcpy(d_Sunspots, Sunspots, NUM_YEARS * sizeof(REAL), cudaMemcpyHostToDevice);

    cudaMalloc(&d_TrainError, sizeof(REAL));
    cudaMalloc(&d_TestError, sizeof(REAL));
    cudaMemset(d_TrainError, 0, sizeof(REAL));
    cudaMemset(d_TestError, 0, sizeof(REAL));

    // Calculate the number of threads and blocks for training and testing
    int threadsPerBlock = 256; 
    int totalThreads = max((TRAIN_UPB - TRAIN_LWB + 1), (TEST_UPB - TEST_LWB + 1)) * M;
    int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    CalculateError<<<blocks, threadsPerBlock>>>(d_Sunspots, Mean, d_TrainError, d_TestError, M, TRAIN_LWB, TRAIN_UPB, TEST_LWB, TEST_UPB);
    cudaDeviceSynchronize();

    // Copy the results back
    cudaMemcpy(&TrainErrorPredictingMean, d_TrainError, sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TestErrorPredictingMean, d_TestError, sizeof(REAL), cudaMemcpyDeviceToHost);


    // Free GPU memory
    cudaFree(d_Sunspots);
    cudaFree(d_TrainError);
    cudaFree(d_TestError);

    // Output the results to a file
    f = fopen("BPN.txt", "w");
}



void FinalizeApplication(NET* Net)
{
  fclose(f);
}

#endif /* NEURAL_NET_APP_H */