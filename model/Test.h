#ifndef Test_h
#define Test_h

#include <stdio.h>
#include <complex.h>

typedef struct {
	float** mat;
	long s1;
	long s2;
} matrix;

typedef struct {
	float complex** mat;
	long s1;
	long s2;
} cmatrix;

//void cpuKernel2(myComplex* hhat, myComplex* xhat, float* vx, float* vh, myComplex* shat, float* vp, myComplex* phat, int N, int K, int data_len, int pilot_len);

#endif 