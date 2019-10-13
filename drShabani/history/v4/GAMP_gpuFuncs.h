#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "myStructs.h"
#define m(data,y,x,dim)		data[y*dim+x]
#include "gputimer.h"
#include "gpuerrors.h"
 

__device__ float cuMulReal(float* a, float* b, int row, int column, int di2, int di3);
__device__ void cuEstimX(cuComplex* out1, float* out2, float s1, float s2);
__global__ void kernelFunc(float* a1d, float* b1d, cuComplex* c1d, float* a2d, float* b2d, cuComplex* c2d, int di1, int di2, int di3);
__global__ void pCalc(cuComplex* hhat, cuComplex* xhat, float* vx, float* vh, cuComplex* shat, float* vp, cuComplex* phat, int N, int K, int data_len, int pilot_len);
void gpuKernelOld(float* a1, float* b1, cuComplex* c1, float* a2, float* b2, cuComplex* c2, int di1, int di2, int di3, double* gpu_kernel_time);