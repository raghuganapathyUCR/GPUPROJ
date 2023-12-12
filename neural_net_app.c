#ifndef NEURAL_NET_APP_H
#define NEURAL_NET_APP_H

#include "neural_net_types.h"
#include "neural_net_constants.h"
#include "neural_net_functions.h"




void NormalizeSunspots()
{
  INT  Year;
  REAL Min, Max;
	
  Min = MAX_REAL;
  Max = MIN_REAL;
  for (Year=0; Year<NUM_YEARS; Year++) {
    Min = MIN(Min, Sunspots[Year]);
    Max = MAX(Max, Sunspots[Year]);
  }
  Mean = 0;
  for (Year=0; Year<NUM_YEARS; Year++) {
    Sunspots_[Year] = 
    Sunspots [Year] = ((Sunspots[Year]-Min) / (Max-Min)) * (HI-LO) + LO;
    Mean += Sunspots[Year] / NUM_YEARS;
  }
}


void InitializeApplication(NET* Net)
{
  INT  Year, i;
  REAL Out, Err;

  Net->Alpha = 0.5;
  Net->Eta   = 0.05;
  Net->Gain  = 1;

  NormalizeSunspots();
  TrainErrorPredictingMean = 0;
  for (Year=TRAIN_LWB; Year<=TRAIN_UPB; Year++) {
    for (i=0; i<M; i++) {
      Out = Sunspots[Year+i];
      Err = Mean - Out;
      TrainErrorPredictingMean += 0.5 * sqr(Err);
    }
  }
  TestErrorPredictingMean = 0;
  for (Year=TEST_LWB; Year<=TEST_UPB; Year++) {
    for (i=0; i<M; i++) {
      Out = Sunspots[Year+i];
      Err = Mean - Out;
      TestErrorPredictingMean += 0.5 * sqr(Err);
    }
  }
  f = fopen("BPN.txt", "w");
}


void FinalizeApplication(NET* Net)
{
  fclose(f);
}

#endif /* NEURAL_NET_APP_H */