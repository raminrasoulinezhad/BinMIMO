#ifndef mystruct_h
#define mystruct_h

//#include <stdio.h>
#include <time.h>
//#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef struct cuComplex cuComplex;
typedef struct myComplex myComplex;
 
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

#endif