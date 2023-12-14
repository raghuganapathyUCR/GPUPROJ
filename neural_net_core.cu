#ifndef NEURAL_NET_CORE_H
#define NEURAL_NET_CORE_H
#include <stdlib.h>


#include "neural_net_types.h"
#include "neural_net_constants.h"
#include "neural_net_functions.h"
#include "neural_net_app_kernel.h"



void GenerateNetwork(NET* Net)
{
  INT l,i;

  Net->Layer = (LAYER**) calloc(NUM_LAYERS, sizeof(LAYER*));
   
  for (l=0; l<NUM_LAYERS; l++) {
    Net->Layer[l] = (LAYER*) malloc(sizeof(LAYER));
      
    Net->Layer[l]->Units      = Units[l];
    Net->Layer[l]->Output     = (REAL*)  calloc(Units[l]+1, sizeof(REAL));
    Net->Layer[l]->Error      = (REAL*)  calloc(Units[l]+1, sizeof(REAL));
    Net->Layer[l]->Weight     = (REAL**) calloc(Units[l]+1, sizeof(REAL*));
    Net->Layer[l]->WeightSave = (REAL**) calloc(Units[l]+1, sizeof(REAL*));
    Net->Layer[l]->dWeight    = (REAL**) calloc(Units[l]+1, sizeof(REAL*));
    Net->Layer[l]->Output[0]  = BIAS;
      
    if (l != 0) {
      for (i=1; i<=Units[l]; i++) {
        Net->Layer[l]->Weight[i]     = (REAL*) calloc(Units[l-1]+1, sizeof(REAL));
        Net->Layer[l]->WeightSave[i] = (REAL*) calloc(Units[l-1]+1, sizeof(REAL));
        Net->Layer[l]->dWeight[i]    = (REAL*) calloc(Units[l-1]+1, sizeof(REAL));
      }
    }
  }
  Net->InputLayer  = Net->Layer[0];
  Net->OutputLayer = Net->Layer[NUM_LAYERS - 1];
  Net->Alpha       = 0.9;
  Net->Eta         = 0.25;
  Net->Gain        = 1;
}


// void RandomWeights(NET* Net)
// {
//   INT l,i,j;
   
//   for (l=1; l<NUM_LAYERS; l++) {
//     for (i=1; i<=Net->Layer[l]->Units; i++) {
//       for (j=0; j<=Net->Layer[l-1]->Units; j++) {
//         Net->Layer[l]->Weight[i][j] = RandomEqualREAL(-0.5, 0.5);
//       }
//     }
//   }
// }


void RandomWeights(NET *Net) {
    int totalWeights = 0;
    for (int l = 1; l < NUM_LAYERS; l++) {
        totalWeights += (Net->Layer[l-1]->Units + 1) * Net->Layer[l]->Units;
    }

    REAL *d_weights;
    curandState *d_states;
    
    cudaMalloc(&d_weights, totalWeights * sizeof(REAL));
    cudaMalloc(&d_states, totalWeights * sizeof(curandState));

    int blockSize = 256;
    int numBlocks = (totalWeights + blockSize - 1) / blockSize;
    
    // Initialize curand states
    initRandomStates<<<numBlocks, blockSize>>>(d_states, time(NULL), totalWeights);

    // Set random weights
    setRandomWeights<<<numBlocks, blockSize>>>(d_states, d_weights, totalWeights);

    // Copy the random weights back to the host
    int weightIdx = 0;
    for (int l = 1; l < NUM_LAYERS; l++) {
        for (int i = 1; i <= Net->Layer[l]->Units; i++) {
            cudaMemcpy(Net->Layer[l]->Weight[i], &d_weights[weightIdx],
                       (Net->Layer[l-1]->Units + 1) * sizeof(REAL), cudaMemcpyDeviceToHost);
            weightIdx += Net->Layer[l-1]->Units + 1;
        }
    }
}

void SaveWeights(NET* Net)
{
  INT l,i,j;

  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Net->Layer[l]->WeightSave[i][j] = Net->Layer[l]->Weight[i][j];
      }
    }
  }
}


void RestoreWeights(NET* Net)
{
  INT l,i,j;

  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Net->Layer[l]->Weight[i][j] = Net->Layer[l]->WeightSave[i][j];
      }
    }
  }
}

#endif /* NEURAL_NET_CORE_H */