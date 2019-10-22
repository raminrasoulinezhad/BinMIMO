//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "bmm.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

#define TILE 4

// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!

dim3 getDimGrid(const int m, const int n) {
	dim3 dimGrid(n/TILE,n/TILE);
	return dimGrid;
}
dim3 getDimBlock(const int m, const int n) {
	dim3 dimBlock(TILE,TILE);
	return dimBlock;
}
__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m, const int n) {

	const int Row = by * TILE + ty;
	const int Col = bx * TILE + tx;
	__shared__ float ad_shared[TILE][TILE];
	__shared__ float bd_shared[TILE][TILE];
	
	float s = 0.0f;
	
	//int Row_shared=ty;
	//int Col_shared=tx;
	
	int Row_Col_offset=0;
	int j,k;
	
	for (j=0; j<n/TILE; j++)
	{
		ad_shared[ty][tx] = mem2d(ad,m,Row,tx+Row_Col_offset);
		bd_shared[ty][tx] = mem2d(bd,m,ty+Row_Col_offset,Col);
		
		__syncthreads();
		
		for (k=0; k<TILE; k++) {
			s += ad_shared[ty][k] * bd_shared[k][tx];//(mem2d(ad_shared,0,Row_shared*TILE,k+Col_ad_offset) * mem2d(bd_shared,0,(k+Row_bd_offset)*TILE,Col_shared)) ;
		}
		Row_Col_offset+=TILE;
		__syncthreads();
	}
	mem2d(cd,m,Row,Col) = s;
}
