#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define m(data,y,x)		data[y*n+x]
#define MAX_THREADS		1024

// ===========================> Functions Prototype <===============================
void fill(float* data, int size);
double calc_mse(float* data1, float* data2, int size);
void cpuKernel_yx(float* a, float* b, float* c, int n, int y, int x);
void cpuKernel_y(float* a, float* b, float* c, int n, int y);
void cpuKernel(float* a, float* b, float* c, int n);
__global__ void kernelFunc(float* ad, float* bd, float* cd, int n);
void gpuKernel(float* a, float* b, float* c, int n);
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
	float* a;
	float* b;
	float* c_serial;
	float* c;
	a        = (float*)malloc(n*n * sizeof(float));
	b        = (float*)malloc(n*n * sizeof(float));
	c_serial = (float*)malloc(n*n * sizeof(float));
	c        = (float*)malloc(n*n * sizeof(float));
	
	// fill a, b matrices with random values between -16.0f and 16.0f
	srand(0);
	fill(a, n*n);
	fill(b, n*n);
	
	// time measurement for CPU calculation
	clock_t t0 = clock(); 
	if (m<=10) {
		cpuKernel (a, b, c_serial, n);
	} else {
		cpuKernel_y (a, b, c_serial, n, 0); // 1st row
	}
	clock_t t1 = clock(); 
		
	// time measurement for GPU calculation
	clock_t t2 = clock();
	gpuKernel (a, b, c, n);
	clock_t t3 = clock();

	// check correctness of calculation
	float mse;
	if (m<=10) {
		mse = calc_mse( c_serial, c, n*n );
	} else {
		mse = calc_mse( c_serial, c, n ); // 1st row
	}

	printf("n=%d\t CPU=%06ld ms GPU=%06ld ms mse=%f\n",n, (t1-t0)/1000, (t3-t2)/1000, mse);
		
	// free allocated memory for later use
	free(a);
	free(b);
	free(c_serial);
	free(c);
   
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
__global__ void kernelFunc(float* ad, float* bd, float* cd, int n) {

	int tx = threadIdx.x;
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int row = bx;
      
	float s = 0.0f;
	int column;
	column = (by)*(blockDim.x)+tx;
	int k; for (k=0; k<n; k++)
		s += m(ad,row,k) * m(bd,k,column);
	
	m(cd,row,column) = s;
}
//-----------------------------------------------------------------------------
void gpuKernel(float* a, float* b, float* c, int n) {
	// allocate memory on GPU
	// copy data to GPU
	// call kernelFunc
	// copy the results back to CPU
	// free GPU memory
	float *ad, *bd, *cd;
	
	cudaMalloc((void**)&ad, n*n*sizeof(float));
	cudaMalloc((void**)&bd, n*n*sizeof(float));
	cudaMalloc((void**)&cd, n*n*sizeof(float));
	
	cudaMemcpy(ad, a, n*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, n*n*sizeof(float), cudaMemcpyHostToDevice);
	
	//kernelFunc <<<dim3(n/MAX_THREADS,n/MAX_THREADS,MAX_THREADS),MAX_THREADS>>> (ad,bd,cd,n);
	kernelFunc<<< dim3(n,1,1), n >>>(ad, bd, cd, n);
	
	cudaMemcpy(c, cd, n*n*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(ad);
	cudaFree(bd);
	cudaFree(cd);
}