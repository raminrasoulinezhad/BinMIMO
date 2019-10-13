#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CpuFuncs.h"
#include "GPUFuncs.h" 
#include "mystructs.h"
 
#define m(data,y,x,dim)		data[y*dim+x]
#define MAX_THREADS		1024

 // ===========================> Functions Prototype <===============================
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
	const int N = 200;
	const int K = 64;
	const int pilot_len = 64;
	const int data_len = 450;
	
	const int di1 = N;
	const int di2 = K;
	const int di3 = data_len;
	
	// allocate memory in CPU for calculation
	myComplex* H = (myComplex*)malloc(N*K * sizeof(myComplex));
	myComplex* estimatedH_cp = (myComplex*)malloc(N*K * sizeof(myComplex));
	cuComplex* estimatedH_gp = (cuComplex*)malloc(N*K * sizeof(cuComplex));
	myComplex* Pilot_cp = (myComplex*)malloc(K * pilot_len * sizeof(myComplex));
	cuComplex* Pilot_gp = (cuComplex*)malloc(K * pilot_len * sizeof(cuComplex));
	myComplex* rec_sym_cp = (myComplex*)malloc(K * (data_len+pilot_len) * sizeof(myComplex));
	cuComplex* rec_sym_gp = (cuComplex*)malloc(K * (data_len+pilot_len) * sizeof(cuComplex));
	
	myComplex *c1_serial, *c2_serial;
	cuComplex *c1, *c2;
	float *a2, *a1;
	float *b2, *b1;
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
	
	//---reading from files
	int ChannelS2,ChannelS1,sentDataS1,sentDataS2,sentPilotS1,sentPilotS2,recSymS1,recSymS2;
	myComplex* Channel;//= (myComplex*)malloc(sizeof(myComplex));
	myComplex* sentData;//= (myComplex*)malloc(sizeof(myComplex));
	myComplex* sentPilot;// = (myComplex*)malloc(sizeof(myComplex));
	myComplex* recSym;//= (myComplex*)malloc(sizeof(myComplex));
	
    readFileComplex("data//A.txt",&Channel,&ChannelS1,&ChannelS2);
	readFileComplex("data/Xdata.txt",&sentData,&sentDataS1,&sentDataS2);
	readFileComplex("data/Xpilot.txt",&sentPilot,&sentPilotS1,&sentPilotS2);
	readFileComplex("data/y.txt",&recSym,&recSymS1,&recSymS2);

	 
	//writeFileComplex(".\data\estimated_data.txt", sentData,sentDataS1,sentDataS2); 
	int dim = (recSymS2);
	printf("%lf",m((recSym),0,0,dim).r);
	return 0;
}
 
