#ifndef cpuFuncs_h
#define cpuFuncs_h

#define m(data,y,x,dim)		data[y*dim+x]
#define MAX_THREADS		1024
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "myStructs.h"
#include "gputimer.h"
#include "gpuerrors.h"
 
void fill(float* data, int size);
double calc_mse (myComplex* data1, cuComplex* data2, int size);
double calc_mse (myComplex* data1, myComplex* data2, int size);
double calc_mse (float* data1, float* data2, int size);
void matAbs2(myComplex* a, float* b, int di1, int di2);
void matPlusCpx(myComplex* a, myComplex* b, myComplex* c, int di1, int di2);
void matPlus(float* a, float* b, float* c, int di1, int di2);
void matSubCpx(myComplex* a, myComplex* b, myComplex* c, int di1, int di2);
void matSub(float* a, float* b, float* c, int di1, int di2);
void matMulCpx_yx(myComplex* a, myComplex* b, myComplex* c, int di2, int di3, int y, int x);
void matMulCpx(myComplex* a, myComplex* b, myComplex* c, int di1, int di2, int di3);
void matDot_cpxReal(myComplex* a, float* b, myComplex* c, int di1, int di2);
void matMul_yx(float* a, float* b, float* c, int di2, int di3, int y, int x);
void matMul(float* a, float* b, float* c, int di1, int di2, int di3);
float maxi(float in1, float in2);
void estimX(myComplex v, myComplex wvar, myComplex* umean, myComplex* uvar);
void readFileReal(const char* fileName, double** out, int* s1,int* s2);
void readFileComplex(const char* fileName,myComplex** out,int* s1,int* s2);
void writeFileReal(const char* fileName, double* in,int s1,int s2);
void writeFileComplex(const char* fileName, myComplex* in,int s1,int s2);
float myErfcx(float in);
float myNormcdf(float in);
float mySign(float a);
float loglike(myComplex Y, myComplex Zhat, float Zvar);
void cpuKernel(myComplex* y, myComplex* hhat, myComplex* xhat, float* vx, float* vh, myComplex* shat, float* vp, myComplex* phat, int N, int K, int data_len, int pilot_len);
#endif