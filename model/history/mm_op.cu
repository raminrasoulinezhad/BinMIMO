#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
// #include <cuComplex.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define m(data,y,x)		data[y*n+x]
#define MAX_THREADS		1024

struct cuComplex {
	float r;
	float i;
	__device__ cuComplex( float a, float b ) : r(a), i(b) {}
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

// ===========================> Functions Prototype <===============================
void fill(float* data, int size);
float maxi(float in1, float in2);
void estimX(float complex v, float wvar, float complex* umean, float* uvar);
double calc_mse(float* data1, float* data2, int size);
void cpuKernel_yx(float* a, float* b, float* c, int n, int y, int x);
void cpuKernel_y(float* a, float* b, float* c, int n, int y);
void cpuKernel(float* a, float* b, float* c, int n);
void cpuKernelTop(float* a1, float* b1, float* c1, float* a2, float* b2, float* c2, int n);
void cpuKernelTop_y(float* a1, float* b1, float* c1, float* a2, float* b2, float* c2, int n, int y);
__global__ void kernelFunc(float* a1d, float* b1d, float* c1d, float* a2d, float* b2d, float* c2d, int n);
void gpuKernel(float* a1, float* b1, float* c1, float* a2, float* b2, float* c2, int n);
// =================================================================================

int main(int argc, char** argv) {

    struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);
	
	// get parameter from command line to build Matrix dimension
	// check for 10<=m<=13, because m>=14 do not fit in the memory of our GPU, i.e., 1GB.
	int m = 5;
	int n = 1;
	
	if(argc > 1)
		m = atoi(argv[1]);
	for (int i=0;i<m;i++)
		n*=2; // n=2^m
	
	// allocate memory in CPU for calculation
	float* a1;
	float* b1;
	float* c1_serial;
	float* c1;
	float* a2;
	float* b2;
	float* c2_serial;
	float* c2;
	a1        = (float*)malloc(n*n * sizeof(float));
	b1        = (float*)malloc(n*n * sizeof(float));
	c1_serial = (float*)malloc(n*n * sizeof(float));
	c1        = (float*)malloc(n*n * sizeof(float));
	a2        = (float*)malloc(n*n * sizeof(float));
	b2        = (float*)malloc(n*n * sizeof(float));
	c2_serial = (float*)malloc(n*n * sizeof(float));
	c2        = (float*)malloc(n*n * sizeof(float));
	
	// fill a, b matrices with random values between -16.0f and 16.0f
	srand(0);
	fill(a1, n*n);
	fill(b1, n*n);
	fill(a2, n*n);
	fill(b2, n*n);
	
	// time measurement for CPU calculation
	clock_t t0 = clock(); 
	if (m<=10) {
		cpuKernelTop (a1, b1, c1_serial, a2, b2, c2_serial, n);
	} else {
		cpuKernelTop_y (a1, b1, c1_serial, a2, b2, c2_serial, n, 0); // 1st row
	}
	clock_t t1 = clock(); 
		
	// time measurement for GPU calculation
	clock_t t2 = clock();
	gpuKernel (a1, b1, c1, a2, b2, c2, n);
	clock_t t3 = clock();

	// check correctness of calculation
	float mse;
	if (m<=10) {
		mse = calc_mse( c1_serial, c1, n*n ) + calc_mse( c2_serial, c2, n*n );
	} else {
		mse = calc_mse( c1_serial, c1, n) + calc_mse( c2_serial, c2, n); // 1st row
	}

	printf("n=%d\t CPU=%06ld ms GPU=%06ld ms mse=%f\n",n, (t1-t0)/1000, (t3-t2)/1000, mse);
		
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
void cpuKernel_yx(float* a, float* b, float* c, int n, int y, int x) { // one element
	m(c,y,x)=0;
    for(int k=0; k<n; k++) {
		m(c,y,x) += m(a,y,k) * m(b,k,x);
	}
}
void cpuKernel_y(float* a, float* b, float* c, int n, int y) { // one row
    for(int x=0; x<n; x++)
	{
		cpuKernel_yx(a,b,c,n,y,x);
	}
}
void cpuKernel(float* a, float* b, float* c, int n) { // entire matrix
    for(int y=0; y<n; y++)
    for(int x=0; x<n; x++)
	{
		cpuKernel_yx(a,b,c,n,y,x);
	}
}
void cpuKernelTop(float* a1, float* b1, float* c1, float* a2, float* b2, float* c2, int n) { // entire matrix
	int i=0;
	float complex umean;
	float uvar;
	
	cpuKernel(a1,b1,c1,n);
	cpuKernel(a2,b2,c2,n);
	for(i=0; i<n*n; i++)
		estimX(c1[i], c2[i], &umean, &uvar);
	// printf("umean=%f, uvar=%f", creal(umean), creal(uvar));
}
void cpuKernelTop_y(float* a1, float* b1, float* c1, float* a2, float* b2, float* c2, int n, int y) { // one row
	int i=0;
	float complex umean;
	float uvar;
	
	cpuKernel_y(a1,b1,c1,n,y);
	cpuKernel_y(a2,b2,c2,n,y);
	for(i=0; i<n; i++)
		estimX(c1[i], c2[i], &umean, &uvar);
	// printf("umean=%f, uvar=%f", creal(umean), creal(uvar));
}

float maxi(float in1, float in2) {
	if (in1 > in2)
		return in1;
	else
		return in2;
}

void estimX(float complex v, float wvar, float complex* umean, float* uvar) {

	float wvar_inv;
	float logpxr1, logpxr2, logpxr3, logpxr4, max_log;
	float pxr1, pxr2, pxr3, pxr4, sum_pxr;
	float uvar1, uvar2, uvar3, uvar4;
	
	float complex x0_1 = -0.7071 + 0.7071*I;
	float complex x0_2 = -0.7071 - 0.7071*I;
	float complex x0_3 = 0.7071 + 0.7071*I;
	float complex x0_4 = 0.7071 - 0.7071*I;
	
	wvar_inv = 1/wvar;
	logpxr1 = wvar_inv*(pow(cabsf(v - x0_1),2));
	logpxr2 = wvar_inv*(pow(cabsf(v - x0_2),2));
	logpxr3 = wvar_inv*(pow(cabsf(v - x0_3),2));
	logpxr4 = wvar_inv*(pow(cabsf(v - x0_4),2));
	
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
	
	pxr1 = pxr1/sum_pxr;
	pxr2 = pxr2/sum_pxr;
	pxr3 = pxr3/sum_pxr;
	pxr4 = pxr4/sum_pxr;
	
	*umean = pxr1*x0_1 + pxr2*x0_2 + pxr3*x0_3 + pxr4*x0_4;
	uvar1 = pxr1*(pow(cabsf(*umean-x0_1),2));
	uvar2 = pxr2*(pow(cabsf(*umean-x0_2),2));
	uvar3 = pxr3*(pow(cabsf(*umean-x0_3),2));
	uvar4 = pxr4*(pow(cabsf(*umean-x0_4),2));
	
	*uvar = uvar1 + uvar2 + uvar3 + uvar4;
}





//-----------------------------------------------------------------------------
/*__global__ void kernelFunc(float* ad, float* bd, float* cd, int n) {
	// write your GPU kernel function here
	// note that maximum # of threads per block is 1024
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int bz=blockIdx.z;
	int tx=threadIdx.x;
	long i,j;
	float sum=0;
	i= bx+(by*MAX_THREADS*MAX_THREADS+bz*MAX_THREADS+tx)/n;
	j= (by*MAX_THREADS*MAX_THREADS+bz*MAX_THREADS+tx)%n;
	for(long k=0; k<n; k++) {
		sum += m(ad,i,k) * m(bd,k,j);
	}
	m(cd,i,j) = sum;

}*/
__global__ void kernelFunc(float* a1d, float* b1d, float* c1d, float* a2d, float* b2d, float* c2d, int n) {

	int tx = threadIdx.x;
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int row = bx;
	
	float s1_inv, out2;
	
	float logpxr1, logpxr2, logpxr3, logpxr4, max_log;
	float pxr1, pxr2, pxr3, pxr4, sum_pxr;
	float uvar1, uvar2, uvar3, uvar4;
    
	float s1 = 0.0f;
	float s2 = 0.0f;
	int column;
	column = (by)*(blockDim.x)+tx;
	int k;
	for (k=0; k<n; k++){
		s1 += m(a1d,row,k) * m(b1d,k,column);
	}
	for (k=0; k<n; k++){
		s2 += m(a2d,row,k) * m(b2d,k,column);
	}
		
	
	cuComplex x0_1(-0.7071, 0.7071);
	cuComplex x0_2(-0.7071, -0.7071);
	cuComplex x0_3(0.7071 , 0.7071);
	cuComplex x0_4(0.7071 , -0.7071);
	
	s1_inv = 1/s1;
	cuComplex s2_cpx(s2,0);
	logpxr1 = s1_inv*((s2_cpx - x0_1).magnitude2());
	logpxr2 = s1_inv*((s2_cpx - x0_2).magnitude2());
	logpxr3 = s1_inv*((s2_cpx - x0_3).magnitude2());
	logpxr4 = s1_inv*((s2_cpx - x0_4).magnitude2());
	
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
	
	pxr1 = pxr1/sum_pxr;
	pxr2 = pxr2/sum_pxr;
	pxr3 = pxr3/sum_pxr;
	pxr4 = pxr4/sum_pxr;
	
	cuComplex out1(0,0);
	out1 = (cuComplex(pxr1,0)*x0_1 + cuComplex(pxr2,0)*x0_2 + cuComplex(pxr3,0)*x0_3 + cuComplex(pxr4,0)*x0_4);
	uvar1 = pxr1*((out1-x0_1).magnitude2());
	uvar2 = pxr2*((out1-x0_2).magnitude2());
	uvar3 = pxr3*((out1-x0_3).magnitude2());
	uvar4 = pxr4*((out1-x0_4).magnitude2());
	
	out2 = (uvar1 + uvar2 + uvar3 + uvar4);
	
	
	m(c1d,row,column) = out1.magnitude2();
	m(c2d,row,column) = out2;
}
//-----------------------------------------------------------------------------
void gpuKernel(float* a1, float* b1, float* c1, float* a2, float* b2, float* c2, int n) {
	// allocate memory on GPU
	// copy data to GPU
	// call kernelFunc
	// copy the results back to CPU
	// free GPU memory
	float *a1d, *b1d, *c1d;
	float *a2d, *b2d, *c2d;
	
	cudaMalloc((void**)&a1d, n*n*sizeof(float));
	cudaMalloc((void**)&b1d, n*n*sizeof(float));
	cudaMalloc((void**)&c1d, n*n*sizeof(float));
	cudaMalloc((void**)&a2d, n*n*sizeof(float));
	cudaMalloc((void**)&b2d, n*n*sizeof(float));
	cudaMalloc((void**)&c2d, n*n*sizeof(float));
	
	cudaMemcpy(a1d, a1, n*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b1d, b1, n*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(a2d, a2, n*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b2d, b2, n*n*sizeof(float), cudaMemcpyHostToDevice);
	
	//kernelFunc <<<dim3(n/MAX_THREADS,n/MAX_THREADS,MAX_THREADS),MAX_THREADS>>> (ad,bd,cd,n);
	kernelFunc<<< dim3(n,1,1), n >>>(a1d, b1d, c1d, a2d, b2d, c2d, n);
	
	cudaMemcpy(c1, c1d, n*n*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(c2, c2d, n*n*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(a1d);
	cudaFree(b1d);
	cudaFree(c1d);
	cudaFree(a2d);
	cudaFree(b2d);
	cudaFree(c2d);
}