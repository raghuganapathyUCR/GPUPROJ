#include "neural_net_types.h"
#include "neural_net_constants.h"
#include "neural_net_functions.h"
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

        // Allocate memory for Lower Output
        cudaMalloc(&d_LowerOutput, lowerUnits * sizeof(REAL));
        cudaMemcpy(d_LowerOutput, Lower->Output, lowerUnits * sizeof(REAL), cudaMemcpyHostToDevice);

        // Allocate memory for Upper Weights (flattened 2D array)
        cudaMalloc(&d_Weight, lowerUnits * upperUnits * sizeof(REAL));
        // Assuming Weight is a contiguous block of memory
        cudaMemcpy(d_Weight, Upper->Weight[0], lowerUnits * upperUnits * sizeof(REAL), cudaMemcpyHostToDevice);

        // Allocate memory for Upper Output
        cudaMalloc(&d_UpperOutput, upperUnits * sizeof(REAL));

        // Launch the kernel
        PropagateLayerLaunch(d_LowerOutput, d_UpperOutput, d_Weight, lowerUnits, upperUnits, Net->Gain);

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