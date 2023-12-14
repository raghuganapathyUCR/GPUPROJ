#include <math.h>
#include <stdio.h>
#ifndef NEURAL_NET_CONSTANTS_H
#define NEURAL_NET_CONSTANTS_H
#define FALSE         0
#define TRUE          1
#define NOT           !
#define AND           &&
#define OR            ||

#define MIN_REAL      -HUGE_VAL
#define MAX_REAL      +HUGE_VAL
#define MIN(x,y)      ((x)<(y) ? (x) : (y))
#define MAX(x,y)      ((x)>(y) ? (x) : (y))

#define LO            0.1
#define HI            0.9
#define BIAS          1

#define sqr(x)        ((x)*(x))


#define NUM_LAYERS    3
#define NN_YEARS      30
#define M             1
extern INT  Units[NUM_LAYERS];

#define FIRST_YEAR    1700
#define NUM_YEARS     280

#define TRAIN_LWB     (N)
#define TRAIN_UPB     (179)
#define TRAIN_YEARS   (TRAIN_UPB - TRAIN_LWB + 1)
#define TEST_LWB      (180)
#define TEST_UPB      (259)
#define TEST_YEARS    (TEST_UPB - TEST_LWB + 1)
#define EVAL_LWB      (260)
#define EVAL_UPB      (NUM_YEARS - 1)
#define EVAL_YEARS    (EVAL_UPB - EVAL_LWB + 1)





extern REAL                  Sunspots_[NUM_YEARS];
extern REAL                  Sunspots [NUM_YEARS];
extern REAL                  Mean;
extern REAL                  TrainError;
extern REAL                  TrainErrorPredictingMean;
extern REAL                  TestError;
extern REAL                  TestErrorPredictingMean;
extern FILE*                 f;
#endif /* NEURAL_NET_CONSTANTS_H */