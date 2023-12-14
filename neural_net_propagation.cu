#include "neural_net_types.h"
#include "neural_net_constants.h"
#include "neural_net_functions.h"
#include "neural_net_app_kernel.h"

#ifndef NEURAL_NET_PROPAGATION_H
#define NEURAL_NET_PROPAGATION_H


void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper) {
    REAL *d_UpperWeights, *d_LowerOutput, *d_UpperOutput;

    cudaMalloc(&d_UpperWeights, sizeof(REAL) * (Upper->Units + 1) * (Lower->Units + 1));
    cudaMalloc(&d_LowerOutput, sizeof(REAL) * (Lower->Units + 1));
    cudaMalloc(&d_UpperOutput, sizeof(REAL) * (Upper->Units + 1));

    cudaMemcpy(d_UpperWeights, Upper->Weight, sizeof(REAL) * (Upper->Units + 1) * (Lower->Units + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_LowerOutput, Lower->Output, sizeof(REAL) * (Lower->Units + 1), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (Upper->Units + threadsPerBlock) / threadsPerBlock;

    PropagateLayerKernel<<<blocks, threadsPerBlock>>>(d_UpperWeights, d_LowerOutput, d_UpperOutput, Lower->Units, Upper->Units, Net->Gain);
    cudaDeviceSynchronize();

    cudaMemcpy(Upper->Output, d_UpperOutput, sizeof(REAL) * (Upper->Units + 1), cudaMemcpyDeviceToHost);

    cudaFree(d_UpperWeights);
    cudaFree(d_LowerOutput);
    cudaFree(d_UpperOutput);
}



void PropagateNet(NET* Net)
{
  INT l;
   
  for (l=0; l<NUM_LAYERS-1; l++) {
    PropagateLayer(Net, Net->Layer[l], Net->Layer[l+1]);
    cudaDeviceSynchronize();
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