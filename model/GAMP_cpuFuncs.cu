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
double calc_mse (myComplex* data1, myComplex* data2, int size) {

	double mse = 0.0;
	int i; for (i=0; i<size; i++) {
		myComplex diff = (data1[i]-data2[i]);
		float e = diff.magnitude2();
		mse += e;
	}
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

void matAbs2(myComplex* a, float* b, int di1, int di2){
	for(int i=0; i<di1; i++)
		for(int j=0; j<di2; j++)
			m(b,i,j,di2)=m(a,i,j,di2).magnitude2();
}
void matPlusCpx(myComplex* a, myComplex* b, myComplex* c, int di1, int di2){
	for(int i=0; i<di1; i++)
		for(int j=0; j<di2; j++)
			m(c,i,j,di2)=m(a,i,j,di2)+m(b,i,j,di2);
}
void matPlus(float* a, float* b, float* c, int di1, int di2){
	for(int i=0; i<di1; i++)
		for(int j=0; j<di2; j++)
			m(c,i,j,di2)=m(a,i,j,di2)+m(b,i,j,di2);
}
void matSubCpx(myComplex* a, myComplex* b, myComplex* c, int di1, int di2){
	for(int i=0; i<di1; i++)
		for(int j=0; j<di2; j++)
			m(c,i,j,di2)=m(a,i,j,di2)-m(b,i,j,di2);
}
void matSub(float* a, float* b, float* c, int di1, int di2){
	for(int i=0; i<di1; i++)
		for(int j=0; j<di2; j++)
			m(c,i,j,di2)=m(a,i,j,di2)-m(b,i,j,di2);
}
void matMulCpx_yx(myComplex* a, myComplex* b, myComplex* c, int di2, int di3, int y, int x) { // one element
	m(c,y,x,di3).r=0;
	m(c,y,x,di3).i=0;
    for(int k=0; k<di2; k++) {
		m(c,y,x,di3) = m(c,y,x,di3) + (m(a,y,k,di2) * m(b,k,x,di3));
	}
}
void matMulCpx(myComplex* a, myComplex* b, myComplex* c, int di1, int di2, int di3) { // entire matrix
    for(int y=0; y<di1; y++)
    for(int x=0; x<di3; x++)
	{
		matMulCpx_yx(a,b,c,di2,di3,y,x);
	}
}
void matDot_cpxReal(myComplex* a, float* b, myComplex* c, int di1, int di2){
	for(int i=0; i<di1; i++)
		for(int j=0; j<di2; j++)
			m(c,i,j,di2)=m(a,i,j,di2)*myComplex(m(b,i,j,di2));
}
void matMul_yx(float* a, float* b, float* c, int di2, int di3, int y, int x) { // one element
	m(c,y,x,di3)=0;
    for(int k=0; k<di2; k++) {
		m(c,y,x,di3) = m(c,y,x,di3) + m(a,y,k,di2) * m(b,k,x,di3);
	}
}
void matMul(float* a, float* b, float* c, int di1, int di2, int di3) { // entire matrix
    for(int y=0; y<di1; y++)
    for(int x=0; x<di3; x++)
	{
		matMul_yx(a,b,c,di2,di3,y,x);
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
	
	wvar_inv = 1/(wvar.r);//imag=0 ast
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
	
	pxr1 = pxr1/sum_pxr;
	pxr2 = pxr2/sum_pxr;
	pxr3 = pxr3/sum_pxr;
	pxr4 = pxr4/sum_pxr;
	
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

float myErfcx(float in) {
	float out= exp(pow(in, 2)) * erfc(in);
	return out;
}
float myNormcdf(float in) {
	float out ;
	out = 0.5 * erfc(-1 * M_SQRT1_2 * in);
	return out;
}

float mySign(float a){
	if(a>0) return 1;
	else if(a==0) return 0;
	else return -1;
}

float loglike(myComplex Y, myComplex Zhat, float Zvar) {
	
	float var = 1; //////////////////////////////////////
	float mean = 0;
	float PMonesY_R = mySign((Y.r - 0.1));
	float PMonesY_I = mySign((Y.i - 0.1));
	
	float VarN = 0.5*var ;
	Zvar = 0.5*Zvar;
		
	float C_R = PMonesY_R * ((Zhat.r - mean) / sqrt(Zvar + VarN));
	float C_I = PMonesY_I * ((Zhat.i - mean) / sqrt(Zvar + VarN));
	
	float CDF_R = myNormcdf(C_R);
	float CDF_I = myNormcdf(C_I);
	float ll_R = log(CDF_R);
	float ll_I = log(CDF_I);	
	
	bool I_R = (C_R < -30);
	bool I_I = (C_I < -30);
	
	if (I_R)
		ll_R = -log(2.0)-0.5* pow(C_R,2) +log(myErfcx(-C_R/sqrt(2.0))) ;
	if (I_I)
		ll_I = -log(2.0)-0.5* pow(C_I,2) +log(myErfcx(-C_I/sqrt(2.0))) ;
	
	
	float ll = ll_R + ll_I ;
	return ll;
	
}

void estimZ(myComplex Y, myComplex Phat, float Pvar, myComplex* Zhat, float* Zvar) {
		
	const float mean = 0.0;
	const float var = 1.0;
	const float pi = 3.1415;

	float PMonesY_R = mySign((Y.r) - 0.1);
	float PMonesY_I = mySign((Y.i) - 0.1);	
	
	float VarN = 0.5*var ;
	Pvar = 0.5*Pvar;
		
	//float C_R = PMonesY_R .* ((Phat.r - mean) ./ sqrt( Pvar + VarN ));
	//float C_I = PMonesY_I .* ((Phat.i - mean) ./ sqrt( Pvar + VarN ));
	float C_R = PMonesY_R * ((Phat.r - mean) / sqrt( Pvar + VarN ));
	float C_I = PMonesY_I * ((Phat.i - mean) / sqrt( Pvar + VarN ));


	float ratio_R = (2.0/sqrt(2.0*pi)) * (pow(erfcx(-C_R / sqrt(2.0)),(-1)));
	float ratio_I = (2.0/sqrt(2.0*pi)) * (pow(erfcx(-C_I / sqrt(2.0)),(-1)));
	
	
	//float Zhat_R = Phat.r + PMonesY_R * ((Pvar ./ sqrt(Pvar + VarN)) * ratio_R);
	//float Zhat_I = Phat.i + PMonesY_I * ((Pvar ./ sqrt(Pvar + VarN)) * ratio_I);
	float Zhat_R = Phat.r + PMonesY_R * ((Pvar / sqrt(Pvar + VarN)) * ratio_R);
	float Zhat_I = Phat.i + PMonesY_I * ((Pvar / sqrt(Pvar + VarN)) * ratio_I);
	*Zhat = myComplex(Zhat_R, Zhat_I); 
	
	float Zvar_R = Pvar - (( pow(Pvar,2) / (Pvar + VarN)) * ratio_R) * (C_R + ratio_R);
	float Zvar_I = Pvar - (( pow(Pvar,2) / (Pvar + VarN)) * ratio_I) * (C_I + ratio_I);
	*Zvar = (Zvar_R + Zvar_I);
}
void cpuKernel(myComplex* y, myComplex* hhat, myComplex* xhat, float* vx, float* vh, myComplex* shat, float* vp, myComplex* phat, int N, int K, int data_len, int pilot_len)
{
	const int T = data_len+pilot_len;
	float valIn = -10000000; 
	float val,valOut;
	//-------------lines 1-6
	float* hhatabs2 = (float*)malloc(N*K*sizeof(float));
	float* xhatabs2 = (float*)malloc(K*T*sizeof(float));
	float* vpbartemp1 = (float*)malloc(N*T*sizeof(float));
	float* vpbartemp2 = (float*)malloc(N*T*sizeof(float));
	float* vpbar = (float*)malloc(N*T*sizeof(float));
	float* vptemp = (float*)malloc(N*T*sizeof(float));
	myComplex* phattemp = (myComplex*)malloc(N*T*sizeof(myComplex));
	myComplex* pbar = (myComplex*)malloc(N*T*sizeof(myComplex));
	
	matAbs2(hhat,hhatabs2,N,K);
	matAbs2(xhat,xhatabs2,K,T);
	matMul(hhatabs2,vx,vpbartemp1,N,K,T);
	matMul(vh,xhatabs2,vpbartemp2,N,K,T);
	matPlus(vpbartemp1,vpbartemp2,vpbar,N,T);
	matMulCpx(hhat,xhat,pbar,N,K,T);
	matMul(vh,vx,vptemp,N,K,T);
	matPlus(vpbar,vptemp,vp,N,T);
	matDot_cpxReal(shat,vpbar,phattemp,N,T);
	matSubCpx(pbar,phattemp,phat,N,T);
	
	//-----------log like function
	valOut = 0;
	for (int i=0;i<N;i++)
		for (int j=0; j<T; j++){
			//m(valOutMat,i,j,T) = loglike(m(y,i,j,T),m(phat,i,j,T),m(vp,i,j,T));
			valOut = valOut + loglike(m(y,i,j,T),m(phat,i,j,T),m(vp,i,j,T));
	}
	val = valOut + valIn;
	
	free(hhatabs2);
	free(xhatabs2);
	free(vpbartemp1);
	free(vpbartemp2);
	free(vpbar);
	free(vptemp);
	free(phattemp);
	free(pbar);
}