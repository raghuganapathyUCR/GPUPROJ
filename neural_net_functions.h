#ifndef NEURAL_NET_FUNCTIONS_H
#define NEURAL_NET_FUNCTIONS_H

#include "neural_net_types.h"

/* Random Number Generation */
void InitializeRandoms();
INT RandomEqualINT(INT Low, INT High);
REAL RandomEqualREAL(REAL Low, REAL High);

/* Application-Specific Functions */
void NormalizeSunspots();
void InitializeApplication(NET* Net);
void FinalizeApplication(NET* Net);

/* Network Generation and Management */
void GenerateNetwork(NET* Net);
void RandomWeights(NET* Net);
void SaveWeights(NET* Net);
void RestoreWeights(NET* Net);

/* Input and Output Functions */
void SetInput(NET* Net, REAL* Input);
void GetOutput(NET* Net, REAL* Output);

/* Propagating Signals */
void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper);
void PropagateNet(NET* Net);

/* Backpropagating Errors */
void ComputeOutputError(NET* Net, REAL* Target);
void BackpropagateLayer(NET* Net, LAYER* Upper, LAYER* Lower);
void BackpropagateNet(NET* Net);

/* Adjusting Weights */
void AdjustWeights(NET* Net);

/* Simulating the Network */
void SimulateNet(NET* Net, REAL* Input, REAL* Output, REAL* Target, BOOL Training);

/* Training and Testing the Network */
void TrainNet(NET* Net, INT Epochs);
void TestNet(NET* Net);
void EvaluateNet(NET* Net);

#endif /* NEURAL_NET_FUNCTIONS_H */
