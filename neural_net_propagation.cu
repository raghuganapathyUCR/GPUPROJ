#include "neural_net_types.h"
#include "neural_net_constants.h"
#include "neural_net_functions.h"
#include "neural_net_app_kernel.h"

#ifndef NEURAL_NET_PROPAGATION_H
#define NEURAL_NET_PROPAGATION_H

// void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper)
// {
//   INT  i,j;
//   REAL Sum;

//   for (i=1; i<=Upper->Units; i++) {
//     Sum = 0;
//     for (j=0; j<=Lower->Units; j++) {
//       Sum += Upper->Weight[i][j] * Lower->Output[j];
//     }
//     Upper->Output[i] = 1 / (1 + exp(-Net->Gain * Sum));
//   }
// }


void PropagateNet(NET* Net) {
    INT l;

    for (l = 0; l < NUM_LAYERS - 1; l++) {
        REAL *d_LowerOutput, *d_UpperOutput, *d_Weight;
        LAYER* Lower = Net->Layer[l];
        LAYER* Upper = Net->Layer[l+1];

        int lowerUnits = Lower->Units + 1; // Including bias unit
        int upperUnits = Upper->Units + 1; // Including bias unit

        // Ensure units are positive
        if (lowerUnits <= 0 || upperUnits <= 0) {
            fprintf(stderr, "Invalid number of units: lowerUnits=%d, upperUnits=%d\n", lowerUnits, upperUnits);
            continue;
        }

        // Allocate memory for Lower Output
        cudaError_t err = cudaMalloc(&d_LowerOutput, lowerUnits * sizeof(REAL));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error after cudaMalloc for Lower Output: %s\n", cudaGetErrorString(err));
            continue; // Skip to next iteration
        }
        cudaMemcpy(d_LowerOutput, Lower->Output, lowerUnits * sizeof(REAL), cudaMemcpyHostToDevice);

        // Allocate memory for Upper Weights (flattened 2D array)
        err = cudaMalloc(&d_Weight, lowerUnits * upperUnits * sizeof(REAL));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error after cudaMalloc for Weights: %s\n", cudaGetErrorString(err));
            cudaFree(d_LowerOutput);
            continue; // Skip to next iteration
        }
        cudaMemcpy(d_Weight, Upper->Weight[0], lowerUnits * upperUnits * sizeof(REAL), cudaMemcpyHostToDevice);

        // Allocate memory for Upper Output
        err = cudaMalloc(&d_UpperOutput, upperUnits * sizeof(REAL));
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error after cudaMalloc for Upper Output: %s\n", cudaGetErrorString(err));
            cudaFree(d_LowerOutput);
            cudaFree(d_Weight);
            continue; // Skip to next iteration
        }

        // Launch the kernel
        PropagateLayerLaunch(d_LowerOutput, d_UpperOutput, d_Weight, lowerUnits, upperUnits, Net->Gain);

        // Check for errors after kernel launch
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
            cudaFree(d_LowerOutput);
            cudaFree(d_UpperOutput);
            cudaFree(d_Weight);
            continue; // Skip to next iteration
        }

        // Synchronize to wait for kernel to finish
        cudaDeviceSynchronize();

        // Copy the results back to the host
        cudaMemcpy(Upper->Output, d_UpperOutput, upperUnits * sizeof(REAL), cudaMemcpyDeviceToHost);

        // Free GPU memory
        cudaFree(d_LowerOutput);
        cudaFree(d_UpperOutput);
        cudaFree(d_Weight);
    }
}





void ComputeOutputError(NET* Net, REAL* Target)
{
  INT  i;
  REAL Out, Err;
   
  Net->Error = 0;
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    Out = Net->OutputLayer->Output[i];
    Err = Target[i-1]-Out;
    Net->OutputLayer->Error[i] = Net->Gain * Out * (1-Out) * Err;
    Net->Error += 0.5 * sqr(Err);
  }
}


void BackpropagateLayer(NET* Net, LAYER* Upper, LAYER* Lower)
{
  INT  i,j;
  REAL Out, Err;
   
  for (i=1; i<=Lower->Units; i++) {
    Out = Lower->Output[i];
    Err = 0;
    for (j=1; j<=Upper->Units; j++) {
      Err += Upper->Weight[j][i] * Upper->Error[j];
    }
    Lower->Error[i] = Net->Gain * Out * (1-Out) * Err;
  }
}


void BackpropagateNet(NET* Net)
{
  INT l;
   
  for (l=NUM_LAYERS-1; l>1; l--) {
    BackpropagateLayer(Net, Net->Layer[l], Net->Layer[l-1]);
  }
}

#endif /* NEURAL_NET_PROPAGATION_H */