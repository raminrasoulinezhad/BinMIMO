#include <stdio.h>


typedef struct {
	double** mat;
	long s1;
	long s2;
} matrix;

typedef struct {
	double complex** mat;
	long s1;
	long s2;
} cmatrix;

void test(cmatrix* outcomp, matrix* outreal);