#include "GAMP_cpuFuncs.h"
#include "myStructs.h"


void fill(float* data, int size) {
    for (int i=0; i<size; ++i)
        data[i] = (float) (rand() % 33 - 16);
}

double calc_mse (myComplex* data1, cuComplex* data2, int size) {

	double mse = 0.0;
	int i; for (i=0; i<size; i++) {
		myComplex diff = (data1[i]/data2[i]);
		float e = diff.magnitude2();
		// printf("1r=%.4f, 1i=%.2f, 2r=%.4f, 2i=%.2f",data1[i].real(), data1[i].imag(), data2[i].real(), data2[i].imag());
		// printf("  diffR=%.2f, diffI=%.2f, e=%f\r\n",diff.real(), diff.imag(),e);
		mse += e;
		// printf("i=%d, mse=%f ",i,mse);
	}
	// i=1839;
	// printf("1r=%.4f, 1i=%.2f, 2r=%.4f, 2i=%.2f\r\n",data1[i].real(), data1[i].imag(), data2[i].real(), data2[i].imag());
	mse = mse / size;
	return mse;
} 

double calc_mse (float* data1, float* data2, int size) {

	double mse = 0.0;
	int i; for (i=0; i<size; i++) {
		double e = data1[i]-data2[i];
		e = e * e;
		mse += e;
	}
	mse = mse / size;
	return mse;
}

void cpuMul_yx(float* a, float* b, myComplex* c, int di2, int di3, int y, int x) { // one element
	m(c,y,x,di3)=0;
    for(int k=0; k<di2; k++) {
		m(c,y,x,di3) = m(c,y,x,di3) + myComplex(m(a,y,k,di2) * m(b,k,x,di3));
	}
}
void cpuMul(float* a, float* b, myComplex* c, int di1, int di2, int di3) { // entire matrix
    for(int y=0; y<di1; y++)
    for(int x=0; x<di3; x++)
	{
		cpuMul_yx(a,b,c,di2,di3,y,x);
	}
}


float maxi(float in1, float in2) {
	if (in1 > in2)
		return in1;
	else
		return in2;
}

void estimX(myComplex v, myComplex wvar, myComplex* umean, myComplex* uvar) {

	float wvar_inv;
	float logpxr1, logpxr2, logpxr3, logpxr4, max_log;
	float pxr1, pxr2, pxr3, pxr4, sum_pxr;
	float uvar1, uvar2, uvar3, uvar4;
	
	myComplex x0_1(-0.7071, 0.7071);
	myComplex x0_2(-0.7071, -0.7071);
	myComplex x0_3(0.7071, 0.7071);
	myComplex x0_4(0.7071, -0.7071);
	
	wvar_inv = 1;///(wvar.r);//imag=0 ast
	logpxr1 = wvar_inv*((v - x0_1).magnitude2());
	logpxr2 = wvar_inv*((v - x0_2).magnitude2());
	logpxr3 = wvar_inv*((v - x0_3).magnitude2());
	logpxr4 = wvar_inv*((v - x0_4).magnitude2());
	
	max_log = maxi(maxi(logpxr1,logpxr2),maxi(logpxr3,logpxr4));
	
	logpxr1 = logpxr1 - max_log;
	logpxr2 = logpxr2 - max_log;
	logpxr3 = logpxr3 - max_log;
	logpxr4 = logpxr4 - max_log;
	
	pxr1 = exp(logpxr1);
	pxr2 = exp(logpxr2);
	pxr3 = exp(logpxr3);
	pxr4 = exp(logpxr4);
	
	sum_pxr = pxr1 + pxr2 + pxr3 + pxr4;
	
	pxr1 = pxr1;///sum_pxr;
	pxr2 = pxr2;///sum_pxr;
	pxr3 = pxr3;///sum_pxr;
	pxr4 = pxr4;///sum_pxr;
	
	*umean = myComplex(pxr1,0)*x0_1 + myComplex(pxr2,0)*x0_2 + myComplex(pxr3,0)*x0_3 + myComplex(pxr4,0)*x0_4;
	// printf("umeanR=%.3f, I=%.3f\r\n", (*umean-x0_1).r, (*umean-x0_1).i);
	uvar1 = pxr1*((*umean-x0_1).magnitude2());
	uvar2 = pxr2*((*umean-x0_2).magnitude2());
	uvar3 = pxr3*((*umean-x0_3).magnitude2());
	uvar4 = pxr4*((*umean-x0_4).magnitude2());
	
	*uvar = myComplex(uvar1 + uvar2 + uvar3 + uvar4);
	
}

void readFileReal(const char* fileName, double** out, int* s1,int* s2) {
	FILE* file = fopen(fileName, "r");
	fscanf(file, "%d\n", s1);
	fscanf(file, "%d\n", s2);
	int dim = (*s2);
	*out  = (double*)malloc( (*s1)*(*s2) * sizeof(double));
	for (int i = 0; i < *s1; i++)
		for (int j = 0; j < *s2; j++)
			fscanf(file, "%lf\n", &m(*out,i,j,dim));
		
	// printf("%lf" ,m(out,0,0,dim));
	
}
void readFileComplex(const char* fileName,myComplex** out,int* s1,int* s2) {
	FILE* file = fopen(fileName, "r");
	fscanf(file, "%d\n", s1);
	fscanf(file, "%d\n", s2);
	int dim = (*s2);
	(*out)  = (myComplex*)malloc( (*s1)*(*s2) * sizeof(myComplex));
	for (int i = 0; i < *s1; i++)
		for (int j = 0; j < *s2; j++) {
			double real, imag;
			fscanf(file, "%lf %lf\n", &real, &imag);
			m((*out),i,j,dim).r = real;
			m((*out),i,j,dim).i = imag;
		}
	// printf("%lf" ,m((*out),0,1,dim).r);
}
void writeFileReal(const char* fileName, double* in,int s1,int s2) {
	FILE* file = fopen(fileName, "w");

	fprintf(file, "%d\n", s1);
	fprintf(file, "%d\n", s2);
	int dim = s2;
	for (int i = 0; i < s1; i++)
		for (int j = 0; j < s2; j++)
			fprintf(file, "%lf\n", m(in,i,j,dim));
}

void writeFileComplex(const char* fileName, myComplex* in,int s1,int s2) {
	FILE* file = fopen(fileName, "w");
	int dim = s2;
	fprintf(file, "%d\n", s1);
	fprintf(file, "%d\n", s2);

	for (int i = 0; i < s1; i++)
		for (int j = 0; j < s2; j++)
			fprintf(file, "%lf %lf\n", m(in,i,j,dim).r,
					m(in,i,j,dim).i);
}
void cpuKernel(float* a1, float* b1, myComplex* c1, float* a2, float* b2, myComplex* c2, int di1, int di2, int di3) { // entire matrix
	int i=0;
	// myComplex umean;
	// float uvar;
	
	cpuMul(a1,b1,c1,di1,di2,di3);
	cpuMul(a2,b2,c2,di1,di2,di3);
	for(i=0; i<di1*di3; i++)
		estimX(c1[i], c2[i], c1+i, c2+i);
	// printf("umean=%f, uvar=%f", creal(umean), creal(uvar));
}
