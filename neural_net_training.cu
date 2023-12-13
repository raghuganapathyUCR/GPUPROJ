#include "neural_net_types.h"
#include "neural_net_constants.h"
#include "neural_net_functions.h"
#include "neural_net_app_kernel.h"

#include <stdlib.h>

#ifndef NEURAL_NET_TRAINING_H
#define NEURAL_NET_TRAINING_H
void AdjustWeights(NET* Net)
{
  INT  l,i,j;
  REAL Out, Err, dWeight;
   
  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Out = Net->Layer[l-1]->Output[j];
        Err = Net->Layer[l]->Error[i];
        dWeight = Net->Layer[l]->dWeight[i][j];
        Net->Layer[l]->Weight[i][j] += Net->Eta * Err * Out + Net->Alpha * dWeight;
        Net->Layer[l]->dWeight[i][j] = Net->Eta * Err * Out;
      }
    }
  }
}


void setUpDeviceForErrorCalc(NET* Net,REAL* Target) {
  REAL *d_Target, *d_Output, *d_Error;
  int units = Net->OutputLayer->Units;

  cudaMalloc(&d_Target, units * sizeof(REAL));
  cudaMalloc(&d_Output, units * sizeof(REAL));
  cudaMalloc(&d_Error, units * sizeof(REAL));

  cudaMemcpy(d_Target, Target, units * sizeof(REAL), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Output, Net->OutputLayer->Output, units * sizeof(REAL), cudaMemcpyHostToDevice);

  REAL *d_NetError;
  cudaMalloc(&d_NetError, sizeof(REAL));
  cudaMemset(d_NetError, 0, sizeof(REAL));

    // Launch here
    ComputeOutputErrorKernel<<<1, units>>>(d_Output, d_Target, d_Error, Net->Gain, units, d_NetError);

    // Copy the results back to host and free device memory
    cudaMemcpy(Net->OutputLayer->Error, d_Error, units * sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaMemcpy(&(Net->Error), d_NetError, sizeof(REAL), cudaMemcpyDeviceToHost);

    cudaFree(d_Target);
    cudaFree(d_Output);
    cudaFree(d_Error);
    cudaFree(d_NetError);

}


void SimulateNet(NET* Net, REAL* Input, REAL* Output, REAL* Target, BOOL Training)
{
  SetInput(Net, Input);
  PropagateNet(Net);
  GetOutput(Net, Output);
  setUpDeviceForErrorCalc(Net, Target);
  // ComputeOutputError(Net, Target);
  if (Training) {
    BackpropagateNet(Net);
    AdjustWeights(Net);
  }
}


void TrainNet(NET* Net, INT Epochs)
{
  INT  Year, n;
  REAL Output[M];

  for (n=0; n<Epochs*TRAIN_YEARS; n++) {
    Year = RandomEqualINT(TRAIN_LWB, TRAIN_UPB);
    SimulateNet(Net, &(Sunspots[Year-N]), Output, &(Sunspots[Year]), TRUE);
  }
}


void TestNet(NET* Net)
{
  INT  Year;
  REAL Output[M];

  TrainError = 0;
  for (Year=TRAIN_LWB; Year<=TRAIN_UPB; Year++) {
    SimulateNet(Net, &(Sunspots[Year-N]), Output, &(Sunspots[Year]), FALSE);
    TrainError += Net->Error;
  }
  TestError = 0;
  for (Year=TEST_LWB; Year<=TEST_UPB; Year++) {
    SimulateNet(Net, &(Sunspots[Year-N]), Output, &(Sunspots[Year]), FALSE);
    TestError += Net->Error;
  }
  fprintf(f, "\nNMSE is %0.3f on Training Set and %0.3f on Test Set",
             TrainError / TrainErrorPredictingMean,
             TestError / TestErrorPredictingMean);
}


void EvaluateNet(NET* Net)
{
  INT  Year;
  REAL Output [M];
  REAL Output_[M];

  fprintf(f, "\n\n\n");
  fprintf(f, "Year    Sunspots    Open-Loop Prediction    Closed-Loop Prediction\n");
  fprintf(f, "\n");
  for (Year=EVAL_LWB; Year<=EVAL_UPB; Year++) {
    SimulateNet(Net, &(Sunspots [Year-N]), Output,  &(Sunspots [Year]), FALSE);
    SimulateNet(Net, &(Sunspots_[Year-N]), Output_, &(Sunspots_[Year]), FALSE);
    Sunspots_[Year] = Output_[0];
    fprintf(f, "%d       %0.3f                   %0.3f                     %0.3f\n",
               FIRST_YEAR + Year,
               Sunspots[Year],
               Output [0],
               Output_[0]);
  }
}

#endif /* NEURAL_NET_TRAINING_H */

