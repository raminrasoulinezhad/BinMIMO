#include "GPUFuncs.h"
#include "mystructs.h"


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