#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GAMP_cpuFuncs.h"
#include "GAMP_gpuFuncs.h" 
#include "myStructs.h"
#include "gputimer.h"
#include "gpuerrors.h"
#define m(data,y,x,dim)		data[y*dim+x]
#define MAX_THREADS		1024

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
	const int T = pilot_len + data_len;
	
	// allocate memory in CPU for calculation
	// myComplex* H = (myComplex*)malloc(N*K * sizeof(myComplex));
	// myComplex* estimatedH_cp = (myComplex*)malloc(N*K * sizeof(myComplex));
	// cuComplex* estimatedH_gp = (cuComplex*)malloc(N*K * sizeof(cuComplex));
	// myComplex* Pilot_cp = (myComplex*)malloc(K * pilot_len * sizeof(myComplex));
	// cuComplex* Pilot_gp = (cuComplex*)malloc(K * pilot_len * sizeof(cuComplex));
	// myComplex* rec_sym_cp = (myComplex*)malloc(K * (data_len+pilot_len) * sizeof(myComplex));
	// cuComplex* rec_sym_gp = (cuComplex*)malloc(K * (data_len+pilot_len) * sizeof(cuComplex));
	
	//---reading from files
	int ChannelS2,ChannelS1,sentDataS1,sentDataS2,sentPilotS1,sentPilotS2,recSymS1,recSymS2;
	myComplex* Channel;//= (myComplex*)malloc(sizeof(myComplex));
	myComplex* sentData;//= (myComplex*)malloc(sizeof(myComplex));
	myComplex* sentPilot;// = (myComplex*)malloc(sizeof(myComplex));
	myComplex* recSym;//= (myComplex*)malloc(sizeof(myComplex));
	
    readFileComplex("data//A.txt",&Channel,&ChannelS1,&ChannelS2);
    //readFileComplex("data/A.txt",&Channel,&ChannelS1,&ChannelS2);
	readFileComplex("data/Xdata.txt",&sentData,&sentDataS1,&sentDataS2);
	readFileComplex("data/Xpilot.txt",&sentPilot,&sentPilotS1,&sentPilotS2);
	readFileComplex("data/y.txt",&recSym,&recSymS1,&recSymS2);
	
	float *vp, *vp_serial;
	myComplex *phat_serial;
	float *valOut_serial;
	myComplex *phat;
	float *valOut;
	float *vx, *vh;
	myComplex *hhat, *xhat, *shat;
	
	// outputs
	phat_serial = (myComplex*)malloc(N*T * sizeof(myComplex));
	valOut_serial = (float*)malloc( sizeof(float));
	phat        = (myComplex*)malloc(N*T * sizeof(myComplex));
	valOut        = (float*)malloc(sizeof(float));
	vp          = (float*)malloc(N*T * sizeof(float));
	vp_serial   = (float*)malloc(N*T * sizeof(float));
	// inputs
	vx          = (float*)malloc(K*T * sizeof(float));
	vh          = (float*)malloc(N*K * sizeof(float));
	xhat        = (myComplex*)malloc(K*T * sizeof(myComplex));
	
	hhat = Channel;
	shat = recSym;
	for(int i=0; i<K*T; i++){
		vx[i]=1;
		xhat[i]=0;
	}
	for(int i=0; i<N*K; i++)
		vh[i]=1;
	 
	// time measurement for CPU calculation
	clock_t t0 = clock(); 
	cpuKernel(recSym, hhat, xhat, vx, vh, shat, vp_serial, phat_serial, N, K, data_len, pilot_len);
	clock_t t1 = clock(); 
		
	// time measurement for GPU calculation
	double gpu_kernel_time = 0.0;
	clock_t t2 = clock();
	gpuKernel(recSym, valOut, hhat, xhat, vx, vh, shat, vp, phat, N, K, data_len, pilot_len, &gpu_kernel_time);
	clock_t t3 = clock();

	// check correctness of calculation
	float mse;
	mse = calc_mse( valOut_serial, valOut, 1 );// + calc_mse( phat_serial, phat, N*T );

	printf("N=%d K=%d T=%d\t CPU=%g ms GPU=%g ms GPU-kernel=%g ms mse=%f\n", N, K, T, (t1-t0)/1000.0, (t3-t2)/1000.0, gpu_kernel_time, mse);
		
	// free allocated memory for later use
	free(phat_serial);
	free(phat);
	free(vp_serial);
	free(vp);
	free(vh); free(vx);
	free(hhat); free(shat); free(xhat);
	
	//free(Channel);
	free(sentData);
    free(sentPilot);
	//free(recSym);
	// //writeFileComplex(".\data\estimated_data.txt", sentData,sentDataS1,sentDataS2); 
	// int dim = (recSymS2);
	// printf("%lf",m((recSym),0,0,dim).r);
	return 0;
}
