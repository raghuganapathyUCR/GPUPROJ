#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "neural_net_types.h"
#include "neural_net_constants.h"
#include "neural_net_functions.h"


/******************************************************************************
                                    M A I N
 ******************************************************************************/


void main()
{
  NET  Net;
  BOOL Stop;
  REAL MinTestError;

  InitializeRandoms();
  GenerateNetwork(&Net);
  RandomWeights(&Net);
  InitializeApplication(&Net);

  Stop = FALSE;
  MinTestError = MAX_REAL;
  do {
    TrainNet(&Net, 10);
    TestNet(&Net);
    if (TestError < MinTestError) {
      fprintf(f, " - saving Weights ...");
      MinTestError = TestError;
      SaveWeights(&Net);
    }
    else if (TestError > 1.2 * MinTestError) {
      fprintf(f, " - stopping Training and restoring Weights ...");
      Stop = TRUE;
      RestoreWeights(&Net);
    }
  } while (NOT Stop);

  TestNet(&Net);
  EvaluateNet(&Net);
   
  FinalizeApplication(&Net);
}