#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define m(data,y,x,dim)		data[y*dim+x]
#define MAX_THREADS		1024

struct cuComplex {
	float r;
	float i;
	__device__ cuComplex() : r(0), i(0) {}
	__device__ cuComplex( float a) : r(a), i(0) {}
	__device__ cuComplex( float a, float b ) : r(a), i(b) {}
	float real(void){return r;}
	float imag(void){return i;}
	__device__ float magnitude2( void ) {
	return r * r + i * i;
	}
	__device__ cuComplex operator*(const cuComplex& a) {
	return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a) {
	return cuComplex(r+a.r, i+a.i);
	}
	__device__ cuComplex operator-(const cuComplex& a) {
	return cuComplex(r-a.r, i-a.i);
	}
};
struct myComplex {
	float r;
	float i;
	myComplex() : r(0), i(0) {}
	myComplex( float a) : r(a), i(0) {}
	myComplex( float a, float b ) : r(a), i(b) {}
	float real(void){return r;}
	float imag(void){return i;}
	float magnitude2( void ) {
	return r * r + i * i;
	}
	myComplex operator*(const myComplex& a) {
	return myComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	myComplex operator+(const myComplex& a) {
	return myComplex(r+a.r, i+a.i);
	}
	myComplex operator-(const myComplex& a) {
	return myComplex(r-a.r, i-a.i);
	}
	myComplex operator/(const cuComplex& a) {
	return myComplex(r-a.r, i-a.i);
	}
};

// ===========================> Functions Prototype <===============================
void fill(float* data, int size);
float maxi(float in1, float in2);
void estimX(myComplex v, myComplex wvar, myComplex* umean, myComplex* uvar);
double calc_mse(float* data1, float* data2, int size);
double calc_mse(myComplex* data1, cuComplex* data2, int size);
void cpuMul_yx(float* a, float* b, myComplex* c, int di2, int di3, int y, int x);
void cpuMul(float* a, float* b, myComplex* c, int di1, int di2, int di3);
void cpuKernel(float* a1, float* b1, myComplex* c1, float* a2, float* b2, myComplex* c2, int di1, int di2, int di3);
__device__ float cuMulReal(float* a, float* b, int row, int column, int di2, int di3);
__device__ void cuEstimX(cuComplex* out1, float* out2, float s1, float s2);
__global__ void kernelFunc(float* a1d, float* b1d, cuComplex* c1d, float* a2d, float* b2d, cuComplex* c2d, int di1, int di2, int di3);
void gpuKernel(float* a1, float* b1, cuComplex* c1, float* a2, float* b2, cuComplex* c2, int di1, int di2, int di3);
// =================================================================================

int main(int argc, char** argv) {

    struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);
	printf("shared mem size: %d\n", p.regsPerBlock);
	
	// get parameter from command line to build Matrix dimension
	const int di1 = 200;
	const int di2 = 64;
	const int di3 = 450;
	
	// allocate memory in CPU for calculation
	float* a1;
	float* b1;
	myComplex* c1_serial;
	cuComplex* c1;
	float* a2;
	float* b2;
	myComplex* c2_serial;
	cuComplex* c2;
	a1        = (float*)malloc(di1*di2 * sizeof(float));
	b1        = (float*)malloc(di2*di3 * sizeof(float));
	c1_serial = (myComplex*)malloc(di1*di3 * sizeof(myComplex));
	c1        = (cuComplex*)malloc(di1*di3 * sizeof(cuComplex));
	a2        = (float*)malloc(di1*di2 * sizeof(float));
	b2        = (float*)malloc(di2*di3 * sizeof(float));
	c2_serial = (myComplex*)malloc(di1*di3 * sizeof(myComplex));
	c2        = (cuComplex*)malloc(di1*di3 * sizeof(cuComplex));
		
	// fill a, b matrices with random values between -16.0f and 16.0f
	srand(0);
	fill(a1, di1*di2);
	fill(b1, di2*di3);
	fill(a2, di1*di2);
	fill(b2, di2*di3);
	
	// time measurement for CPU calculation
	clock_t t0 = clock();
	cpuKernel (a1, b1, c1_serial, a2, b2, c2_serial, di1, di2, di3);
	clock_t t1 = clock();
		
	// time measurement for GPU calculation
	clock_t t2 = clock();
	gpuKernel (a1, b1, c1, a2, b2, c2, di1, di2, di3);
	clock_t t3 = clock();

	// check correctness of calculation
	float mse;
	mse = calc_mse( c1_serial, c1, di1*di3 ) + calc_mse( c2_serial, c2, di1*di3 );

	printf("dim1=%d dim2=%d dim3=%d\t CPU=%06ld ms GPU=%06ld ms mse=%f\n",di1, di2, di3, (t1-t0)/1000, (t3-t2)/1000, mse);
		
	// free allocated memory for later use
	free(a1);
	free(b1);
	free(c1_serial);
	free(c1);
	free(a2);
	free(b2);
	free(c2_serial);
	free(c2);
   
	return 0;
}

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
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

__device__ void cuEstimX(cuComplex* out1, float* out2, float s1, float s2){
	float s2_inv;
	cuComplex s1_cpx;
	
	float logpxr1, logpxr2, logpxr3, logpxr4, max_log;
	float pxr1, pxr2, pxr3, pxr4, sum_pxr;
	float uvar1, uvar2, uvar3, uvar4;
	
	cuComplex x0_1(-0.7071, 0.7071);
	cuComplex x0_2(-0.7071, -0.7071);
	cuComplex x0_3(0.7071 , 0.7071);
	cuComplex x0_4(0.7071 , -0.7071);
	
	s2_inv = 1;///s2;
	s1_cpx.r = s1;
	logpxr1 = s2_inv*((s1_cpx - x0_1).magnitude2());
	logpxr2 = s2_inv*((s1_cpx - x0_2).magnitude2());
	logpxr3 = s2_inv*((s1_cpx - x0_3).magnitude2());
	logpxr4 = s2_inv*((s1_cpx - x0_4).magnitude2());
	
	max_log = logpxr1;
	if(logpxr2>max_log)
		max_log=logpxr2;
	if(logpxr3>max_log)
		max_log=logpxr3;
	if(logpxr4>max_log)
		max_log=logpxr4;
	
	logpxr1 = logpxr1 - max_log;
	logpxr2 = logpxr2 - max_log;
	logpxr3 = logpxr3 - max_log;
	logpxr4 = logpxr4 - max_log;
	
	pxr1 = exp(logpxr1);
	pxr2 = exp(logpxr2);
	pxr3 = exp(logpxr3);
	pxr4 = exp(logpxr4);
	
	sum_pxr = pxr1 + pxr2 + pxr3 + pxr4;
	
	pxr1 = pxr1;//sum_pxr;
	pxr2 = pxr2;//sum_pxr;
	pxr3 = pxr3;//sum_pxr;
	pxr4 = pxr4;//sum_pxr;
	
	*out1 = (cuComplex(pxr1,0)*x0_1 + cuComplex(pxr2,0)*x0_2 + cuComplex(pxr3,0)*x0_3 + cuComplex(pxr4,0)*x0_4);
	uvar1 = pxr1*((*out1-x0_1).magnitude2());
	uvar2 = pxr2*((*out1-x0_2).magnitude2());
	uvar3 = pxr3*((*out1-x0_3).magnitude2());
	uvar4 = pxr4*((*out1-x0_4).magnitude2());
	
	*out2 = (uvar1 + uvar2 + uvar3 + uvar4);
}

__device__ float cuMulReal(float* a, float* b, int row, int column, int di2, int di3){
	int k;
	float s = 0.0f;
	
	for (k=0; k<di2; k++){
		s += m(a,row,k,di2) * m(b,k,column,di3);
	}
	return s;
}
__device__ cuComplex cuMulCpx(cuComplex* a, cuComplex* b, int row, int column, int di2, int di3){
	int k;
	cuComplex s(0.0f, 0.0f);
	
	for (k=0; k<di2; k++){
		s = s + (m(a,row,k,di2) * m(b,k,column,di3));
	}
	return s;
}
__global__ void kernelFunc(float* a1d, float* b1d, cuComplex* c1d, float* a2d, float* b2d, cuComplex* c2d, int di1, int di2, int di3) {

	int tx = threadIdx.x;
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int row, column;
	float s1 = 0.0f;
	float s2 = 0.0f;
	cuComplex tmp(0);
	cuComplex* out1=&tmp;
	float tmp2=0.0f;
	float* out2=&tmp2;
	
	row = bx;
	column = (by)*(blockDim.x)+tx;
	
	s1 = cuMulReal(a1d, b1d, row, column, di2, di3);
	
	s2 = cuMulReal(a2d, b2d, row, column, di2, di3);
	
	cuEstimX(out1, out2, s1, s2);
	
	m(c1d,row,column,di3) = cuComplex(*out1);
	m(c2d,row,column,di3) = cuComplex(*out2);
}
//-----------------------------------------------------------------------------
void gpuKernel(float* a1, float* b1, cuComplex* c1, float* a2, float* b2, cuComplex* c2, int di1, int di2, int di3) {
	// allocate memory on GPU
	// copy data to GPU
	// call kernelFunc
	// copy the results back to CPU
	// free GPU memory
	float *a1d, *b1d;
	float *a2d, *b2d;
	cuComplex *c1d, *c2d;
	
	cudaMalloc((void**)&a1d, di1*di2*sizeof(float));
	cudaMalloc((void**)&b1d, di2*di3*sizeof(float));
	cudaMalloc((void**)&c1d, di1*di3*sizeof(cuComplex));
	cudaMalloc((void**)&a2d, di1*di2*sizeof(float));
	cudaMalloc((void**)&b2d, di2*di3*sizeof(float));
	cudaMalloc((void**)&c2d, di1*di3*sizeof(cuComplex));
	
	cudaMemcpy(a1d, a1, di1*di2*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b1d, b1, di2*di3*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(a2d, a2, di1*di2*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b2d, b2, di2*di3*sizeof(float), cudaMemcpyHostToDevice);
	
	//kernelFunc <<<dim3(n/MAX_THREADS,n/MAX_THREADS,MAX_THREADS),MAX_THREADS>>> (ad,bd,cd,n);
	kernelFunc<<< dim3(di1,1,1), di3 >>>(a1d, b1d, c1d, a2d, b2d, c2d, di1, di2, di3);
	
	cudaMemcpy(c1, c1d, di1*di3*sizeof(cuComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(c2, c2d, di1*di3*sizeof(cuComplex), cudaMemcpyDeviceToHost);
	
	cudaFree(a1d);
	cudaFree(b1d);
	cudaFree(c1d);
	cudaFree(a2d);
	cudaFree(b2d);
	cudaFree(c2d);
}