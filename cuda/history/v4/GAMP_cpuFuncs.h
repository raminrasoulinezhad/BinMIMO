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
float maxi(float in1, float in2);
void estimX(myComplex v, myComplex wvar, myComplex* umean, myComplex* uvar);
double calc_mse(float* data1, float* data2, int size);
double calc_mse(myComplex* data1, cuComplex* data2, int size);
void readFileReal(const char* fileName, double** out, int* s1,int* s2);
void readFileComplex(const char* fileName,myComplex** out,int* s1,int* s2) ;
void writeFileComplex(const char* fileName, myComplex* in,int s1,int s2);
void cpuMul_yx(float* a, float* b, myComplex* c, int di2, int di3, int y, int x);
void cpuMul(float* a, float* b, myComplex* c, int di1, int di2, int di3);
void cpuKernel(float* a1, float* b1, myComplex* c1, float* a2, float* b2, myComplex* c2, int di1, int di2, int di3);