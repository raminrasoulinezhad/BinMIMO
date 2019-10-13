#include "GAMP_gpuFuncs.h" 
#include "myStructs.h"


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
	
	s2_inv = 1/s2;
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
	
	pxr1 = pxr1/sum_pxr;
	pxr2 = pxr2/sum_pxr;
	pxr3 = pxr3/sum_pxr;
	pxr4 = pxr4/sum_pxr;
	
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

__device__ float myCuErfcx(float in) {
	float out= exp(pow(in, 2)) * erfc(in);
	return out;
}
__device__ float myCuNormcdf(float in) {
	float out ;
	out = 0.5 * erfc(-1 * M_SQRT1_2 * in);
	return out;
}
__device__ float myCuSign(float a){
	if(a>0) return 1;
	else if(a==0) return 0;
	else return -1;
}
__device__ float culoglike(cuComplex Y, cuComplex Zhat, float Zvar) {
	
	float var = 1; //////////////////////////////////////
	float mean = 0;
	float PMonesY_R = myCuSign(Y.r - 0.1);
	float PMonesY_I = myCuSign(Y.i - 0.1);
	
	float VarN = 0.5*var ;
	Zvar = 0.5*Zvar;
		
	float C_R = PMonesY_R * ((Zhat.r - mean) / sqrt(Zvar + VarN));
	float C_I = PMonesY_I * ((Zhat.i - mean) / sqrt(Zvar + VarN));
	
	float CDF_R = myCuNormcdf(C_R);
	float CDF_I = myCuNormcdf(C_I);
	float ll_R = log(CDF_R);
	float ll_I = log(CDF_I);	
	
	bool I_R = (C_R < -30);
	bool I_I = (C_I < -30);
	
	float temp1 = 2.0f;
	if (I_R)
		ll_R = -log(temp1)-0.5* pow(C_R,2) +log(myCuErfcx(-C_R/sqrt(temp1))) ;
	if (I_I)
		ll_I = -log(temp1)-0.5* pow(C_I,2) +log(myCuErfcx(-C_I/sqrt(temp1))) ;
	
	
	float ll = ll_R + ll_I;
	return ll;
}


__global__ void pCalc(cuComplex* y, float* valout, cuComplex* hhat, cuComplex* xhat, float* vx, float* vh, cuComplex* shat, float* vp, cuComplex* phat, int N, int K, int data_len, int pilot_len){
	int tx = threadIdx.x;
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int row, column;
	float vpbar = 0.0f;
	float vptemp = 0.0f;
	cuComplex pbar(0,0);
	float vhi, vxi;
	cuComplex hhati, xhati;
	
	float vpThread;
	cuComplex phatThread;
	
	const int T = data_len + pilot_len;
	row = bx;
	column = (by)*(blockDim.x)+tx;
	
	for(int i=0; i<K; i++)
	{
		vhi = m(vh,row,i,K);
		vxi = m(vx,i,column,T);
		hhati = m(hhat,row,i,K);
		xhati = m(xhat,i,column,T);
		
		vpbar += (hhati.magnitude2())*vxi + vhi*(xhati.magnitude2());
		pbar = pbar + (hhati*xhati);
		vptemp += vhi*vxi; 
	}
	vpThread = vpbar+vptemp;
	phatThread = pbar-(m(shat,row,column,T)*(cuComplex(vpbar)));
	
	m(vp,row,column,T) = vpThread;
	m(phat,row,column,T) = phatThread;
	
	m(valout,row,column,T)=culoglike(m(y,row,column,T), m(phat,row,column,T), m(vp,row,column,T));
}
__global__ void cuMatSum(float* valoutdMat,float* valoutd,int N, int K, int data_len, int pilot_len){
	float temp = 0;
	int T = data_len + pilot_len;
	for (int i=0;i<N;i++)
		for (int j=0; j<T; j++)
			temp = temp + m(valoutdMat,i,j,T);
	*valoutd = temp;
}

void gpuKernel(myComplex* y, float* valout, myComplex* hhat, myComplex* xhat, float* vx, float* vh, myComplex* shat, float* vp, myComplex* phat, int N, int K, int data_len, int pilot_len, double* gpu_kernel_time) {
	float *vxd, *vhd, *vpd, *valoutdMat,*valoutd;
	cuComplex *hhatd, *xhatd, *shatd, *phatd, *yd;
	const int T = data_len + pilot_len;
	
	HANDLE_ERROR(cudaMalloc((void**)&vxd, K*T*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&vhd, N*K*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&vpd, N*T*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&shatd, N*T*sizeof(cuComplex)));
	HANDLE_ERROR(cudaMalloc((void**)&xhatd, K*T*sizeof(cuComplex)));
	HANDLE_ERROR(cudaMalloc((void**)&hhatd, N*K*sizeof(cuComplex)));
	HANDLE_ERROR(cudaMalloc((void**)&phatd, N*T*sizeof(cuComplex)));
	HANDLE_ERROR(cudaMalloc((void**)&valoutdMat, N*T*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&valoutd, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&yd, N*T*sizeof(cuComplex)));
	
	HANDLE_ERROR(cudaMemcpy(vxd, vx, K*T*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(vhd, vh, N*K*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(yd, y, N*T*sizeof(cuComplex), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(hhatd, hhat, N*K*sizeof(cuComplex), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(xhatd, xhat, K*T*sizeof(cuComplex), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(shatd, shat, N*T*sizeof(cuComplex), cudaMemcpyHostToDevice));
	
	GpuTimer timer;
    timer.Start();
	pCalc <<< dim3(N,1,1), T >>> (yd, valoutdMat, hhatd, xhatd, vxd, vhd, shatd, vpd, phatd, N, K, data_len, pilot_len);
	cuMatSum <<< dim3(1,1,1), 1 >>> (valoutdMat,valoutd, N, K, data_len, pilot_len);
	timer.Stop();
	*gpu_kernel_time = timer.Elapsed();
	
	HANDLE_ERROR(cudaMemcpy(phat, phatd, N*T*sizeof(cuComplex), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(vp, vpd, N*T*sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(valout, valoutd, sizeof(float), cudaMemcpyDeviceToHost));
	
	HANDLE_ERROR(cudaFree(vxd));
	HANDLE_ERROR(cudaFree(vhd));
	HANDLE_ERROR(cudaFree(vpd));
	HANDLE_ERROR(cudaFree(shatd));
	HANDLE_ERROR(cudaFree(xhatd));
	HANDLE_ERROR(cudaFree(hhatd));
	HANDLE_ERROR(cudaFree(phatd));
	HANDLE_ERROR(cudaFree(valoutdMat));
	HANDLE_ERROR(cudaFree(valoutd));
	HANDLE_ERROR(cudaFree(yd));
	

	
}