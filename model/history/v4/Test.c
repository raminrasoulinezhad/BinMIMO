#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>

#include "Test.h"

typedef struct {
	double stepMin;
	double stepMax;
	int nit;
	int nitMin;
	double stepInit;
//	double stepFilter;
	double stepIncr;
//	double stepDecr;
//	int stepWindow;
//	double tol;
//	double stepTol;
	double xvarMin;
	double AvarMin;
	double pvarMin;
	double zvarToPvarMax;
	double varThresh;
} opt;

void printReal(matrix* mat) {
	printf("size: %ld x %ld\n", mat->s1, mat->s2);
	long i,j;
	for ( i = 0; i < mat->s1; i++) {
		for ( j = 0; j < mat->s2; j++) {
			printf("%f  ", mat->mat[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void printComplex(cmatrix* mat) {
	printf("size: %ld x %ld\n", mat->s1, mat->s2);
	long i,j;
	for ( i = 0; i < mat->s1; i++) {
		for ( j = 0; j < mat->s2; j++) {
			printf("%f + %f*i  ", creal(mat->mat[i][j]), cimag(mat->mat[i][j]));
		}
		printf("\n");
	}
	printf("\n");
}

double min(double in1, double in2) {
	if (in1 < in2)
		return in1;
	else
		return in2;
}

double max(double in1, double in2) {
	if (in1 > in2)
		return in1;
	else
		return in2;
}

matrix* createMatReal(long s1, long s2) {

	matrix* out = (matrix *) malloc(sizeof(matrix));

	out->mat = malloc(s1 * sizeof *(out->mat));
	long i,j;
	for ( i = 0; i < s1; i++)
		out->mat[i] = malloc(s2 * sizeof *(out->mat[i]));

	for ( i = 0; i < s1; i++)
		for ( j = 0; j < s2; j++)
			out->mat[i][j] = 0;
	out->s1 = s1;
	out->s2 = s2;

	return out;
}

cmatrix* createMatComplex(long s1, long s2) {
	long i,j;	
	cmatrix* out = (cmatrix *) malloc(sizeof(cmatrix));

	out->mat = malloc(s1 * sizeof *(out->mat));
	for ( i = 0; i < s1; i++)
		out->mat[i] = malloc(s2 * sizeof *(out->mat[i]));

	for ( i = 0; i < s1; i++)
		for ( j = 0; j < s2; j++)
			out->mat[i][j] = 0;
	out->s1 = s1;
	out->s2 = s2;

	return out;
}

matrix* readFileReal(const char* fileName) {
	FILE* file = fopen(fileName, "r");
	long s1, s2;
	long i,j;
	
	fscanf(file, "%ld\n", &s1);
	fscanf(file, "%ld\n", &s2);

	matrix* out = createMatReal(s1, s2);

	for ( i = 0; i < s1; i++)
		for ( j = 0; j < s2; j++)
			fscanf(file, "%lf\n", &(out->mat[i][j]));
	return out;
}

cmatrix* readFileComplex(const char* fileName) {
	FILE* file = fopen(fileName, "r");
	long s1, s2;
	fscanf(file, "%ld\n", &s1);
	fscanf(file, "%ld\n", &s2);

	cmatrix* out = createMatComplex(s1, s2);
	long i,j;
	for ( i = 0; i < s1; i++)
		for ( j = 0; j < s2; j++) {
			double real, imag;
			fscanf(file, "%lf %lf\n", &real, &imag);
			out->mat[i][j] = real + I * imag;
		}

	return out;
}

void writeFileReal(const char* fileName, matrix* in) {
	FILE* file = fopen(fileName, "w");

	fprintf(file, "%ld\n", in->s1);
	fprintf(file, "%ld\n", in->s2);
	long i,j;
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++)
			fprintf(file, "%lf\n", in->mat[i][j]);
}

void writeFileComplex(const char* fileName, cmatrix* in) {
	FILE* file = fopen(fileName, "w");

	fprintf(file, "%ld\n", in->s1);
	fprintf(file, "%ld\n", in->s2);
	long i,j;
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++)
			fprintf(file, "%lf %lf\n", creal(in->mat[i][j]),
					cimag(in->mat[i][j]));
}

cmatrix* toComplex(matrix* in) {
	cmatrix* out = createMatComplex(in->s1, in->s2);
	long i,j;
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++)
			out->mat[i][j] = in->mat[i][j];
	return out;
}

matrix* matrixMaxScalar(matrix* in, double val) {

	matrix* out = createMatReal(in->s1, in->s2);
	int i,j;
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++) {
			if (in->mat[i][j] > val)
				out->mat[i][j] = val;
			else
				out->mat[i][j] = in->mat[i][j];
		}
	return out;
}

matrix* matrixMinScalar(matrix* in, double val) {

	matrix* out = createMatReal(in->s1, in->s2);
	long i,j;
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++) {
			if (in->mat[i][j] < val)
				out->mat[i][j] = val;
			else
				out->mat[i][j] = in->mat[i][j];
		}
	return out;
}

matrix* matrixMax(matrix* in, long dim) {
	matrix* out;
	long i,j;
	if (dim == 0) {
		out = createMatReal(1, in->s2);
		for ( j = 0; j < in->s2; j++) {
			double tmp = in->mat[0][j];
			for ( i = 1; i < in->s1; i++) {
				if (in->mat[i][j] > tmp)
					tmp = in->mat[i][j];
			}
			out->mat[0][j] = tmp;
		}
	}

	else if (dim == 1) {
		out = createMatReal(in->s1, 1);
		for ( i = 0; i < in->s1; i++) {
			double tmp = in->mat[i][0];
			for ( j = 1; j < in->s2; j++) {
				if (in->mat[i][j] > tmp)
					tmp = in->mat[i][j];
			}
			out->mat[i][0] = tmp;
		}
	}
	return out;
}

matrix* matrixSumReal(matrix* in, long dim) {
	matrix* out;
	long i,j;
	if (dim == 0) {
		out = createMatReal(1, in->s2);
		for ( j = 0; j < in->s2; j++) {
			double tmp = 0;
			for ( i = 0; i < in->s1; i++) {
				tmp += in->mat[i][j];
			}
			out->mat[0][j] = tmp;
		}
	}

	else if (dim == 1) {
		out = createMatReal(in->s1, 1);
		for ( i = 0; i < in->s1; i++) {
			double tmp = 0;
			for ( j = 0; j < in->s2; j++) {
				tmp += in->mat[i][j];
			}
			out->mat[i][0] = tmp;
		}
	}
	return out;
}

cmatrix* matrixSumComplex(cmatrix* in, long dim) {
	cmatrix* out;
	long i,j;
	if (dim == 0) {
		out = createMatComplex(1, in->s2);
		for ( j = 0; j < in->s2; j++) {
			double complex tmp = 0;
			for ( i = 0; i < in->s1; i++) {
				tmp += in->mat[i][j];
			}
			out->mat[0][j] = tmp;
		}
	}

	else if (dim == 1) {
		out = createMatComplex(in->s1, 1);
		for ( i = 0; i < in->s1; i++) {
			double complex tmp = 0;
			for ( j = 0; j < in->s2; j++) {
				tmp += in->mat[i][j];
			}
			out->mat[i][0] = tmp;
		}
	}
	return out;
}

matrix* matrixMultReal(matrix* in1, matrix* in2) {
	matrix* out = createMatReal(in1->s1, in2->s2);
	long i,j,k;
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in2->s2; j++) {
			out->mat[i][j] = 0;
			for ( k = 0; k < in1->s2; k++) {
				out->mat[i][j] += in1->mat[i][k] * in2->mat[k][j];
			}
		}
	return out;
}

cmatrix* matrixMultComplex(cmatrix* in1, cmatrix* in2) {
	cmatrix* out = createMatComplex(in1->s1, in2->s2);
	long i,j,k;
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in2->s2; j++) {
			out->mat[i][j] = 0;
			for ( k = 0; k < in1->s2; k++) {
				out->mat[i][j] += in1->mat[i][k] * in2->mat[k][j];
			}
		}
	return out;
}

matrix* matrixDot_RR(matrix* in1, matrix* in2) {
	matrix* out = createMatReal(in1->s1, in1->s2);
	long i,j;
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in2->s2; j++)
			out->mat[i][j] = in1->mat[i][j] * in2->mat[i][j];
	return out;
}

matrix* matrixDiv_RR(matrix* in1, matrix* in2) {
	matrix* out = createMatReal(in1->s1, in1->s2);
	long i,j;
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in2->s2; j++)
			out->mat[i][j] = in1->mat[i][j] / in2->mat[i][j];
	return out;
}

cmatrix* matrixDot_CC(cmatrix* in1, cmatrix* in2) {
	cmatrix* out = createMatComplex(in1->s1, in1->s2);
	long i,j;
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in2->s2; j++)
			out->mat[i][j] = in1->mat[i][j] * in2->mat[i][j];
	return out;
}

cmatrix* matrixDot_CR(cmatrix* in1, matrix* in2) {
	cmatrix* out = createMatComplex(in1->s1, in1->s2);
	long i,j;
	for ( i = 0; i < in1->s1; i++)
		for (j = 0; j < in2->s2; j++)
			out->mat[i][j] = in1->mat[i][j] * in2->mat[i][j];
	return out;
}

matrix* matrixAdd(matrix* in1, matrix* in2) {
	matrix* out = createMatReal(in1->s1, in1->s2);
	long i,j;
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in1->s2; j++)
			out->mat[i][j] = in1->mat[i][j] + in2->mat[i][j];
	return out;
}

cmatrix* matrixAddComplex(cmatrix* in1, cmatrix* in2) {
	cmatrix* out = createMatComplex(in1->s1, in1->s2);
	long i,j;
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in1->s2; j++)
			out->mat[i][j] = in1->mat[i][j] + in2->mat[i][j];
	return out;
}

matrix* matrixSub(matrix* in1, matrix* in2) {
	matrix* out = createMatReal(in1->s1, in1->s2);
	long i,j;
	for ( i = 0; i < in1->s1; i++)
		for (j = 0; j < in1->s2; j++)
			out->mat[i][j] = in1->mat[i][j] - in2->mat[i][j];
	return out;
}

cmatrix* matrixSubComplex(cmatrix* in1, cmatrix* in2) {
	cmatrix* out = createMatComplex(in1->s1, in1->s2);
	long i,j;
	for ( i = 0; i < in1->s1; i++)
		for (j = 0; j < in1->s2; j++)
			out->mat[i][j] = in1->mat[i][j] - in2->mat[i][j];
	return out;
}

matrix* matrixAddScalar(matrix* in1, double in2) {
	matrix* out = createMatReal(in1->s1, in1->s2);
	long i,j;
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in1->s2; j++)
			out->mat[i][j] = in1->mat[i][j] + in2;
	return out;
}

matrix* matrixMultScalarReal(matrix* in1, double in2) {
	matrix* out = createMatReal(in1->s1, in1->s2);
	long i,j;
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in1->s2; j++)
			out->mat[i][j] = in1->mat[i][j] * in2;
	return out;
}

cmatrix* matrixMultScalarComplex(cmatrix* in1, double in2) {
	cmatrix* out = createMatComplex(in1->s1, in1->s2);
	long i,j;	
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in1->s2; j++)
			out->mat[i][j] = in1->mat[i][j] * in2;
	return out;
}

matrix* matrixPowerReal(matrix* in, double power) {
	matrix* out = createMatReal(in->s1, in->s2);
	long i,j;	
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++)
			out->mat[i][j] = pow(in->mat[i][j], power);
	return out;
}

matrix* matrixPowerComplex(cmatrix* in, double power) {
	matrix* out = createMatReal(in->s1, in->s2);
		long i,j;
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++)
			out->mat[i][j] = pow(cabs(in->mat[i][j]), power);
	return out;
}

matrix* submatrixReal(matrix* in, long s11, long s12, long s21, long s22) {
	matrix* out = createMatReal(s12 - s11 + 1, s22 - s21 + 1);
	long i,j;
	for ( i = s11; i <= s12; i++)
		for ( j = s21; j <= s22; j++)
			out->mat[i - s11][j - s21] = in->mat[i][j];
	return out;
}

cmatrix* submatrixComplex(cmatrix* in, long s11, long s12, long s21, long s22) {
	cmatrix* out = createMatComplex(s12 - s11 + 1, s22 - s21 + 1);
	long i,j;
	for ( i = s11; i <= s12; i++)
		for ( j = s21; j <= s22; j++)
			out->mat[i - s11][j - s21] = in->mat[i][j];
	return out;
}

matrix* transposeReal(matrix* in) {
	matrix* out = createMatReal(in->s2, in->s1);
	long i,j;
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++)
			out->mat[j][i] = in->mat[i][j];

	return out;
}

cmatrix* transposeComplex(cmatrix* in) {
	cmatrix* out = createMatComplex(in->s2, in->s1);
	long i,j;
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++)
			out->mat[j][i] = creal(in->mat[i][j]) - I * cimag(in->mat[i][j]);

	return out;
}

matrix* sign(matrix* in) {
	matrix* out = createMatReal(in->s1, in->s2);
	long i,j;
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++) {
			if (in->mat[i][j] == 0)
				out->mat[i][j] = 0;
			else if (in->mat[i][j] > 0)
				out->mat[i][j] = 1;
			else
				out->mat[i][j] = -1;
		}
	return out;
}

matrix* realPart(cmatrix* in) {
	matrix* out = createMatReal(in->s1, in->s2);
	long i,j;	
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++)
			out->mat[i][j] = creal(in->mat[i][j]);
	return out;
}

matrix* imagPart(cmatrix* in) {
	matrix* out = createMatReal(in->s1, in->s2);
	long i,j;	
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++)
			out->mat[i][j] = cimag(in->mat[i][j]);
	return out;
}

double norm(cmatrix* in) {
	return 0;
}

matrix* normcdf(matrix* in) {
	matrix* out = createMatReal(in->s1, in->s2);
	long i,j;
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++)
			out->mat[i][j] = 0.5 * erfc(-1 * M_SQRT1_2 * in->mat[i][j]);
	return out;
}

matrix* erfcx(matrix* in) {
	matrix* out = createMatReal(in->s1, in->s2);
	long i,j;	
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++)
			out->mat[i][j] = exp(pow(in->mat[i][j], 2)) * erfc(in->mat[i][j]);
	return out;
}

matrix* Log(matrix* in) {
	matrix* out = createMatReal(in->s1, in->s2);
	long i,j;	
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++)
			out->mat[i][j] = log(in->mat[i][j]);
	return out;
}

matrix* Exp(matrix* in) {
	matrix* out = createMatReal(in->s1, in->s2);
		long i,j;
	for ( i = 0; i < in->s1; i++)
		for ( j = 0; j < in->s2; j++)
			out->mat[i][j] = exp(in->mat[i][j]);
	return out;
}

cmatrix* merge(matrix* real, matrix* imag) {
	cmatrix* out = createMatComplex(real->s1, real->s2);
	long i,j;	
	for ( i = 0; i < real->s1; i++)
		for ( j = 0; j < real->s2; j++)
			out->mat[i][j] = real->mat[i][j] + I * imag->mat[i][j];
	return out;
}

matrix* concat1Real(matrix* in1, matrix* in2) {
	long i,j;	
	matrix* out = createMatReal(in1->s1, in1->s2 + in2->s2);
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in1->s2; j++)
			out->mat[i][j] = in1->mat[i][j];
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in2->s2; j++)
			out->mat[i][j + in1->s2] = in2->mat[i][j];
	return out;
}

matrix* concat2Real(matrix* in1, matrix* in2) {
	matrix* out = createMatReal(in1->s1 + in2->s1, in1->s2);
	long i,j;	
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in1->s2; j++)
			out->mat[i][j] = in1->mat[i][j];
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in2->s2; j++)
			out->mat[i + in1->s1][j] = in2->mat[i][j];
	return out;
}

cmatrix* concat1Complex(cmatrix* in1, cmatrix* in2) {
	cmatrix* out = createMatComplex(in1->s1, in1->s2 + in2->s2);
	long i,j;	
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in1->s2; j++)
			out->mat[i][j] = in1->mat[i][j];
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in2->s2; j++)
			out->mat[i][j + in1->s2] = in2->mat[i][j];
	return out;
}

cmatrix* concat2Complex(cmatrix* in1, cmatrix* in2) {
	cmatrix* out = createMatComplex(in1->s1 + in2->s1, in1->s2);
	long i,j;	
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in1->s2; j++)
			out->mat[i][j] = in1->mat[i][j];
	for ( i = 0; i < in1->s1; i++)
		for ( j = 0; j < in2->s2; j++)
			out->mat[i + in1->s1][j] = in2->mat[i][j];
	return out;
}

matrix* reshapeReal(matrix* in, long s1, long s2) {
	matrix* out = createMatReal(s1, s2);

	long row = 0;
	long col = 0;
	long i,j;
	for ( j = 0; j < in->s2; j++) {
		for ( i = 0; i < in->s1; i++) {
			out->mat[row][col] = in->mat[i][j];
			if (row == s1 - 1) {
				row = 0;
				col++;
			} else
				row++;
		}
	}
	return out;
}

cmatrix* reshapeComplex(cmatrix* in, long s1, long s2) {
	cmatrix* out = createMatComplex(s1, s2);

	long row = 0;
	long col = 0;
	long i,j;
	for ( j = 0; j < in->s2; j++) {
		for ( i = 0; i < in->s1; i++) {
			out->mat[row][col] = in->mat[i][j];
			if (row == s1 - 1) {
				row = 0;
				col++;
			} else
				row++;
		}
	}
	return out;
}

matrix* expandReal(matrix* in, long size, long dim) {
	long i,j;		
	if (dim == 0) {
		matrix* out = createMatReal(size, in->s2);	
		for ( i = 0; i < size; i++)
			for ( j = 0; j < in->s2; j++)
				out->mat[i][j] = in->mat[0][j];
		return out;
	}

	else {
		matrix* out = createMatReal(in->s1, size);
		for ( i = 0; i < in->s1; i++)
			for ( j = 0; j < size; j++)
				out->mat[i][j] = in->mat[i][0];
		return out;
	}
}

cmatrix* expandComplex(cmatrix* in, long size, long dim) {
	long i,j;		
	if (dim == 0) {
		cmatrix* out = createMatComplex(size, in->s2);
		for ( i = 0; i < size; i++)
			for ( j = 0; j < in->s2; j++)
				out->mat[i][j] = in->mat[0][j];
		return out;
	}

	else {
		cmatrix* out = createMatComplex(in->s1, size);
		for ( i = 0; i < in->s1; i++)
			for ( j = 0; j < size; j++)
				out->mat[i][j] = in->mat[i][0];
		return out;
	}
}

matrix* bsxfunReal(long func, matrix* in1, matrix* in2) {
	matrix* out = createMatReal(max(in1->s1, in2->s1), max(in1->s2, in2->s2));

	matrix* in1_expanded = in1;
	matrix* in2_expanded = in2;
	long i,j;	
	if (in1->s1 > in2->s1)
		in2_expanded = expandReal(in2_expanded, in1->s1, 0);
	else if (in1->s1 < in2->s1)
		in1_expanded = expandReal(in1_expanded, in2->s1, 0);

	if (in1->s2 > in2->s2)
		in2_expanded = expandReal(in2_expanded, in1->s2, 1);
	else if (in1->s2 < in2->s2)
		in1_expanded = expandReal(in1_expanded, in2->s2, 1);

	for ( i = 0; i < in1_expanded->s1; i++) {
		for ( j = 0; j < in1_expanded->s2; j++) {
			switch (func) {
			case 0:
				out->mat[i][j] = in1_expanded->mat[i][j]
						- in2_expanded->mat[i][j];
				break;
			case 1:
				out->mat[i][j] = in1_expanded->mat[i][j]
						* in2_expanded->mat[i][j];
				break;
			case 2:
				out->mat[i][j] = in1_expanded->mat[i][j]
						/ in2_expanded->mat[i][j];
				break;
			default:
				break;
			}
		}
	}
	return out;
}

cmatrix* bsxfunComplex(long func, cmatrix* in1, cmatrix* in2) {
	cmatrix* out = createMatComplex(max(in1->s1, in2->s1),
			max(in1->s2, in2->s2));
	long i,j;	
	cmatrix* in1_expanded = in1;
	cmatrix* in2_expanded = in2;

	if (in1->s1 > in2->s1)
		in2_expanded = expandComplex(in2_expanded, in1->s1, 0);
	else if (in1->s1 < in2->s1)
		in1_expanded = expandComplex(in1_expanded, in2->s1, 0);

	if (in1->s2 > in2->s2)
		in2_expanded = expandComplex(in2_expanded, in1->s2, 1);
	else if (in1->s2 < in2->s2)
		in1_expanded = expandComplex(in1_expanded, in2->s2, 1);

	for ( i = 0; i < in1_expanded->s1; i++) {
		for ( j = 0; j < in1_expanded->s2; j++) {
			switch (func) {
			case 0:
				out->mat[i][j] = in1_expanded->mat[i][j]
						- in2_expanded->mat[i][j];
				break;
			case 1:
				out->mat[i][j] = in1_expanded->mat[i][j]
						* in2_expanded->mat[i][j];
				break;
			case 2:
				out->mat[i][j] = in1_expanded->mat[i][j]
						/ in2_expanded->mat[i][j];
				break;
			default:
				break;
			}
		}
	}
	return out;
}

matrix* loglike(cmatrix* Y, double mean, double var, bool maxSumVal,
		cmatrix* Zhat, matrix* Zvar) {
	long i,j;	
	matrix* PMonesY_R = sign(matrixAddScalar(realPart(Y), -0.1));
	matrix* PMonesY_I = sign(matrixAddScalar(imagPart(Y), -0.1));

	matrix* Zhat_R = realPart(Zhat);
	matrix* Zhat_I = imagPart(Zhat);

	double VarN = 0.5 * var;
	Zvar = matrixMultScalarReal(Zvar, 0.5);

	matrix* C_R;
	matrix* C_I;

	if (!maxSumVal) {
		C_R = matrixDot_RR(PMonesY_R,
				matrixDot_RR(matrixAddScalar(Zhat_R, -1 * mean),
						matrixPowerReal(matrixAddScalar(Zvar, VarN), -0.5)));
		C_I = matrixDot_RR(PMonesY_I,
				matrixDot_RR(matrixAddScalar(Zhat_I, -1 * mean),
						matrixPowerReal(matrixAddScalar(Zvar, VarN), -0.5)));
	}

	else {
		C_R = matrixDot_RR(PMonesY_R,
				matrixMultScalarReal(matrixAddScalar(Zhat_R, -1 * mean),
						sqrt(VarN)));
		C_I = matrixDot_RR(PMonesY_I,
				matrixMultScalarReal(matrixAddScalar(Zhat_I, -1 * mean),
						sqrt(VarN)));
	}

	matrix* ll_R = Log(normcdf(C_R));
	matrix* ll_I = Log(normcdf(C_I));

	for ( i = 0; i < ll_R->s1; i++)
		for ( j = 0; j < ll_R->s2; j++) {
			if (ll_R->mat[i][j] < -30)
				ll_R->mat[i][j] = -log(2) - 0.5 * pow(C_R->mat[i][j], 2)
						+ log(-C_R->mat[i][j] * M_SQRT1_2)
						+ pow(-C_R->mat[i][j] * M_SQRT1_2, 2);

			if (ll_I->mat[i][j] < -30)
				ll_I->mat[i][j] = -log(2) - 0.5 * pow(C_I->mat[i][j], 2)
						+ log(-C_I->mat[i][j] * M_SQRT1_2)
						+ pow(-C_I->mat[i][j] * M_SQRT1_2, 2);
		}
	matrix* ll = matrixAdd(ll_R, ll_I);
	return ll;
}

void estimZ(cmatrix* Y, double var, double mean, cmatrix* PHat, matrix* PVar,
		cmatrix** ZHat, matrix** ZVar) {
	long i,j;	
	matrix* PMonesY_R = sign(matrixAddScalar(realPart(Y), -0.1));
	matrix* PMonesY_I = sign(matrixAddScalar(imagPart(Y), -0.1));

	matrix* Phat_R = realPart(PHat);
	matrix* Phat_I = imagPart(PHat);

	double VarN = 0.5 * var;
	PVar = matrixMultScalarReal(PVar, 0.5);

	matrix* PvarAddVarN = matrixAddScalar(PVar, VarN);
	PvarAddVarN = matrixPowerReal(PvarAddVarN, -0.5);

	matrix* PHat_RSubMean = matrixAddScalar(Phat_R, -mean);
	matrix* PHat_ISubMean = matrixAddScalar(Phat_I, -mean);

	matrix* Dummy1 = matrixDot_RR(PHat_RSubMean, PvarAddVarN);
	matrix* Dummy2 = matrixDot_RR(PHat_ISubMean, PvarAddVarN);

	matrix* C_R = matrixDot_RR(PMonesY_R, Dummy1);
	matrix* C_I = matrixDot_RR(PMonesY_I, Dummy2);

	matrix* invCR = matrixMultScalarReal(C_R, -1);
	matrix* invCI = matrixMultScalarReal(C_I, -1);
	matrix* divCR = matrixMultScalarReal(invCR, M_SQRT1_2);
	matrix* divCI = matrixMultScalarReal(invCI, M_SQRT1_2);

	matrix* ratio_R = matrixMultScalarReal(matrixPowerReal(erfcx(divCR), -1),
	M_2_SQRTPI * M_SQRT1_2);
	matrix* ratio_I = matrixMultScalarReal(matrixPowerReal(erfcx(divCI), -1),
	M_2_SQRTPI * M_SQRT1_2);

	matrix* PVarN = matrixAddScalar(PVar, VarN);
	PVarN = matrixPowerReal(PVarN, -0.5);
	PVarN = matrixDot_RR(PVar, PVarN);
	matrix* PDummy_C = matrixDot_RR(PVarN, ratio_R);
	matrix* PDummy_I = matrixDot_RR(PVarN, ratio_I);

//////////////////  NOW WE CAN CALCULATE ZHat /////////////////////
	matrix* ZHat_R = matrixDot_RR(PMonesY_R, PDummy_C);
	matrix* ZHat_I = matrixDot_RR(PMonesY_I, PDummy_I);
	ZHat_R = matrixAdd(ZHat_R, Phat_R);
	ZHat_I = matrixAdd(ZHat_I, Phat_I);

	*ZHat = merge(ZHat_R, ZHat_I);

///////////////////////////////////////////////////////////////////

	PVarN = matrixAddScalar(PVar, VarN);
	PVarN = matrixPowerReal(PVarN, -1);

	matrix* PVarPow = matrixPowerReal(PVar, 2);
	PVarN = matrixDot_RR(PVarPow, PVarN);

	PVarN = matrixMultScalarReal(PVarN, -1);
	PDummy_C = matrixDot_RR(PVarN, ratio_R);
	PDummy_I = matrixDot_RR(PVarN, ratio_I);

	PDummy_C = matrixDot_RR(PDummy_C, matrixAdd(C_R, ratio_R));
	PDummy_I = matrixDot_RR(PDummy_I, matrixAdd(C_I, ratio_I));

////////////////// NOW WE CAN CALCULATE Zvar /////////////////////////
	matrix* ZVar_R = matrixAdd(PVar, PDummy_C);
	matrix* ZVar_I = matrixAdd(PVar, PDummy_I);

	*ZVar = matrixAdd(ZVar_R, ZVar_I);

}

void estimX(cmatrix* x0, cmatrix* v, matrix* wvar, cmatrix** umean,
		matrix** uvar, matrix** val) {
	long i,j;	
	long v_s1 = v->s1;
	long v_s2 = v->s2;

	v = reshapeComplex(v, (v_s1) * (v_s2), 1);
	wvar = reshapeReal(wvar, (v_s1) * (v_s2), 1);

	matrix* logpxr = bsxfunReal(1,
			matrixMultScalarReal(matrixPowerReal(wvar, -1), -1),
			matrixPowerComplex(bsxfunComplex(0, v, x0), 2));

	logpxr = bsxfunReal(0, logpxr, matrixMax(logpxr, 1));

	matrix* pxr = Exp(logpxr);
	pxr = bsxfunReal(2, pxr, matrixSumReal(pxr, 1));
	*umean = matrixSumComplex(bsxfunComplex(1, toComplex(pxr), x0), 1);
	*uvar = matrixSumReal(
			matrixDot_RR(pxr,
					matrixPowerComplex(bsxfunComplex(0, *umean, x0), 2)), 1);

	*val = matrixMultScalarReal(matrixSumReal(matrixDot_RR(pxr, Log(pxr)), 1),
			-1); //TODO: insert 1e-20

	*umean = reshapeComplex(*umean, v_s1, v_s2);

	*uvar = reshapeReal(*uvar, v_s1, v_s2);
	*val = reshapeReal(*val, v_s1, v_s2);
}

void estimH(cmatrix* mean0, matrix* var0, cmatrix* rhat, matrix* rvar,
		cmatrix** xhat, matrix** xvar, matrix** val) {

	matrix* gain = matrixDot_RR(var0,
			matrixPowerReal(matrixAdd(var0, rvar), -1));
	*xhat = matrixAddComplex(
			matrixDot_CR(
					matrixAddComplex(rhat, matrixMultScalarComplex(mean0, -1)),
					gain), mean0);
	*xvar = matrixDot_RR(gain, rvar);

	matrix* xvar_over_uvar0 = matrixDot_RR(rvar,
			matrixPowerReal(matrixAdd(var0, rvar), -1));
	matrix* arg0 = Log(xvar_over_uvar0);
	matrix* arg1 = matrixAddScalar(matrixMultScalarReal(xvar_over_uvar0, -1),
			1);
	matrix* arg2_0 = matrixPowerComplex(
			matrixAddComplex(*xhat, matrixMultScalarComplex(mean0, -1)), 2);
	matrix* arg2 = matrixMultScalarReal(
			matrixDot_RR(arg2_0, matrixPowerReal(var0, -1)), -1);

	*val = matrixAdd(matrixAdd(arg0, arg1), arg2);
}

void bigamp(opt* opt, cmatrix* A1hatInit, matrix* A1varInit,
		cmatrix* x12hatInit, matrix* x12varInit, cmatrix* pilot, cmatrix* Y,
		cmatrix* x0, cmatrix* mean0, matrix* var0, cmatrix** x12hatOpt,
		matrix** x12varOpt, cmatrix** A1hatOpt, matrix** A1varOpt, cmatrix** outcomp, matrix** outreal) {
	long i,j;	
	cmatrix* A1hat = A1hatInit;
	matrix* A1var = A1varInit;
	cmatrix* x12hat = x12hatInit;
	matrix* x12var = x12varInit;
	////////////////////////// problem variables
	const long M = 200;
	const long L1 = 50;
	const long L2 = 450;
	const long N1 = 50;

	int it = 0;
	double step = opt->stepInit;
	bool stop = false;
	double step1 = 1;
	double valIn = -INFINITY;
	double valOpt_0 = -INFINITY;
	double valOpt_1 = -INFINITY;

	cmatrix* x12hatBar = createMatComplex(N1, L2);
	cmatrix* A1hatBar = createMatComplex(M, N1);
	cmatrix* shat = createMatComplex(M, L1 + L2);
	matrix* svar = createMatReal(M, L1 + L2);
	matrix* pvarOpt = createMatReal(M, L1 + L2);
	cmatrix* zhatOpt = createMatComplex(M, L1 + L2);
	matrix* zvarOpt = createMatReal(M, L1 + L2);

	matrix* A1hat2;
	matrix* x12hat2;
	matrix* pilot2;

	cmatrix* zhat;
	matrix* zvar;

	cmatrix* phat;
	matrix* pvar;

	cmatrix* shatOpt;
	matrix* svarOpt;
	cmatrix* x12hatBarOpt;
	cmatrix* A1hatBarOpt;

	matrix* pvarInv;
	cmatrix* shatNew;
	matrix* svarNew;
	//------------------------------------------

	cmatrix* zhat0;
	matrix* zvar0;

	cmatrix* shat_submat;
	matrix* svar_submat;

	matrix* r12var;
	matrix* r12Gain;
	cmatrix* r12hat;

	matrix* q1var;
	matrix* q1Gain;
	cmatrix* q1hat;

	matrix* valInX12;
	matrix* valInA1;

	//while (~stop) {
	//	it++;

		if (it >= opt->nit)
			stop = true;

		A1hat2 = matrixPowerComplex(A1hat, 2);
		x12hat2 = matrixPowerComplex(x12hat, 2);
		pilot2 = matrixPowerComplex(pilot, 2);

		zvar = matrixAdd(matrixMultReal(A1var, concat1Real(pilot2, x12hat2)),
				concat1Real(createMatReal(M, L1),
						matrixMultReal(A1hat2, x12var)));

		pvar = matrixAdd(zvar,
				concat1Real(createMatReal(M, L1),
						matrixMultReal(A1var, x12var)));

		pvar = matrixAdd(matrixMultScalarReal(pvar, step1),
				matrixMultScalarReal(pvarOpt, 1 - step1));
		zvar = matrixAdd(matrixMultScalarReal(zvar, step1),
				matrixMultScalarReal(zvarOpt, 1 - step1));

		zhat = matrixMultComplex(A1hat, concat1Complex(pilot, x12hat));

		outcomp = zhat;
		outreal = zvar;
		//writeFileComplex("phat.txt", *zhat);
		//writeFileReal("pvar.txt", *zvar);
	
		/* double valOut =
				(matrixSumReal(
						matrixSumReal(loglike(Y, 0, 1, false, zhat, zvar), 1),
						0))->mat[0][0];
		double val = valOut + valIn;

		bool pass = true;

		if (it > 1)
			pass = val > min(valOpt_0, valOpt_1);

		if (pass) {
			step = opt->stepIncr * step;

			shatOpt = shat;
			svarOpt = svar;
			x12hatBarOpt = x12hatBar;
			A1hatBarOpt = A1hatBar;
			*x12hatOpt = x12hat;
			*A1hatOpt = A1hat;

			pvarOpt = pvar;
			zvarOpt = zvar;

			pvar = matrixMinScalar(pvar, opt->pvarMin);

			valOpt_0 = valOpt_1;
			valOpt_1 = val;

			phat = matrixSubComplex(zhat, matrixDot_CR(shat, zvar));

			estimZ(Y, 1, 0, phat, pvar, &zhat0, &zvar0);

			pvarInv = matrixPowerReal(pvar, -1);
			shatNew = matrixDot_CR(matrixSubComplex(zhat0, phat), pvarInv);
			svarNew = matrixDot_RR(
					matrixAddScalar(
							matrixMultScalarReal(
									matrixMaxScalar(matrixDiv_RR(zvar0, pvar),
											opt->zvarToPvarMax), -1), 1),
					pvarInv);

			if (step > opt->stepMax)
				step = opt->stepMax;
			if (step < opt->stepMin)
				step = opt->stepMin;

//			double testVal = norm(matrixSubComplex(zhat, zhatOpt)) / norm(zhat); //TODO: define norm
//
//			if (it > 1 && testVal < opt->tol)
//				stop = true;

			if (it >= 50)
				stop = true;

			*A1varOpt = A1var;
			*x12varOpt = x12var;
			zhatOpt = zhat;

		}

		if (it > 1)
			step1 = step;

		shat = matrixAddComplex(matrixMultScalarComplex(shatOpt, 1 - step1),
				matrixMultScalarComplex(shatNew, step1));
		svar = matrixAdd(matrixMultScalarReal(svarOpt, 1 - step1),
				matrixMultScalarReal(svarNew, step1));

		x12hatBar = matrixAddComplex(
				matrixMultScalarComplex(x12hatBarOpt, 1 - step1),
				matrixMultScalarComplex(*x12hatOpt, step1));
		A1hatBar = matrixAddComplex(
				matrixMultScalarComplex(A1hatBarOpt, 1 - step1),
				matrixMultScalarComplex(*A1hatOpt, step1));

		shat_submat = submatrixComplex(shat, 0, shat->s1 - 1, shat->s2 - L2,
				shat->s2 - 1);
		svar_submat = submatrixReal(svar, 0, svar->s1 - 1, svar->s2 - L2,
				svar->s2 - 1);

		r12var = matrixPowerReal(
				matrixMultReal(transposeReal(matrixPowerComplex(A1hatBar, 2)),
						svar_submat), -1);
		r12var = matrixMaxScalar(r12var, opt->varThresh);

		r12Gain = matrixAddScalar(
				matrixMultScalarReal(
						matrixDot_RR(r12var,
								matrixMultReal(transposeReal(A1var),
										svar_submat)), -1), 1);
		r12hat = matrixAddComplex(matrixDot_CR(x12hatBar, r12Gain),
				matrixDot_CR(
						matrixMultComplex(transposeComplex(A1hatBar),
								shat_submat), r12var));

		r12var = matrixMinScalar(r12var, opt->xvarMin);

		q1var = matrixPowerReal(
				matrixMultReal(svar,
						transposeReal(
								concat1Real(pilot2,
										matrixPowerComplex(x12hatBar, 2)))),
				-1);

		q1var = matrixMaxScalar(q1var, opt->varThresh);

		q1Gain = matrixAddScalar(
				matrixMultScalarReal(
						matrixDot_RR(q1var,
								matrixMultReal(svar,
										transposeReal(
												concat1Real(
														createMatReal(N1, L1),
														x12var)))), -1), 1);
		q1Gain = matrixMinScalar(matrixMaxScalar(q1Gain, 1), 0);

		q1hat = matrixAddComplex(matrixDot_CR(A1hatBar, q1Gain),
				matrixDot_CR(
						matrixMultComplex(shat,
								transposeComplex(
										concat1Complex(pilot, x12hatBar))),
						q1var));

		estimX(x0, r12hat, r12var, &x12hat, &x12var, &valInX12);
		estimH(mean0, var0, q1hat, q1var, &A1hat, &A1var, &valInA1);

		valIn = (matrixSumReal(matrixSumReal(valInA1, 1), 0))->mat[0][0]
				+ (matrixSumReal(matrixSumReal(valInX12, 1), 0))->mat[0][0];

		if (it < opt->nitMin)
			stop = false; */

	//	if(it == 15) {
	//		stop = true;
	//		break;
	//	}

		//}

	/*writeFileComplex("A1hatOpt_code.txt", *A1hatOpt);
	writeFileReal("A1varOpt_code.txt", *A1varOpt);
	writeFileComplex("x12hatOpt_code.txt", *x12hatOpt);
	writeFileReal("x12varOpt_code.txt", *x12varOpt);*/

}

void test(cmatrix* outcomp, matrix* outreal) {

	opt* setting = malloc(sizeof *setting);

	setting->stepInit = 0.05;
	setting->nit = 1;
	setting->nitMin = 0;
//	setting->nitMin = 0;
	setting->stepMin = 0.65;
	setting->stepMax = 0.75;
	setting->stepIncr = 1.1;
	setting->pvarMin = pow(10, -13);
	setting->xvarMin = 0;
	setting->AvarMin = 0;
	setting->zvarToPvarMax = 0.99;
	setting->varThresh = 1000000;

	cmatrix* A1hatInit = readFileComplex("A.txt");
	//matrix* A1varInit = readFileReal(".txt");
	cmatrix* x12hatInit = readFileComplex("Xdata.txt");
	//matrix* x12varInit = readFileReal("x12varInit.txt");

	cmatrix* pilot = readFileComplex("Xpilot.txt");
	cmatrix* Y = readFileComplex("y.txt");

//	cmatrix* mean0 = readFileComplex("mean0.txt");
//	matrix* var0 = readFileReal("var0.txt");
	cmatrix* x0 = readFileComplex("xdata.txt");
	var0 = createMatReal(200,50);
	mean0 = createMatReal(200,50);
	cmatrix* x12hatOpt;
	matrix* x12varOpt;
	cmatrix* A1hatOpt;
	matrix* A1varOpt;

	
	
	bigamp(setting, A1hatInit, A1varInit, x12hatInit, x12varInit, pilot, Y, x0,
			mean0, var0, &x12hatOpt, &x12varOpt, &A1hatOpt, &A1varOpt, &outcomp, &outreal);
	
	


}
