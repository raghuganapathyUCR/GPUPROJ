#include "neural_net_types.h"
#include "neural_net_functions.h"

#ifndef NEURAL_NET_IO_H
#define NEURAL_NET_IO_H


void SetInput(NET* Net, REAL* Input)
{
  INT i;
   
  for (i=1; i<=Net->InputLayer->Units; i++) {
    Net->InputLayer->Output[i] = Input[i-1];
  }
}


void GetOutput(NET* Net, REAL* Output)
{
  INT i;
   
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    Output[i-1] = Net->OutputLayer->Output[i];
  }
}

#endif /* NEURAL_NET_IO_H */