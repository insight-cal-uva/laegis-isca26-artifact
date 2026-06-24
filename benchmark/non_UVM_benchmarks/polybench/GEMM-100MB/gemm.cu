/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#include "../../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define NI 5120
#define NJ 5120
#define NK 5120

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412.0f
#define BETA 2123.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int i,j,k;
	
	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
    	{
			C[i*NJ + j] *= BETA;
	
			for (k = 0; k < NK; ++k)
			{
	  			C[i*NJ + j] += ALPHA * A[i*NK + k] * B[k*NJ + j];
			}
      	}
	}
}


void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int i, j;

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NK; j++)
		{
      		A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
		}
	}

  	for (i = 0; i < NK; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
      		B[i*NJ + j] = ((DATA_TYPE) i*j + 1) / NJ;
		}
	}

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
      		C[i*NJ + j] = ((DATA_TYPE) i*j + 2) / NJ;
		}
	}
}


void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
	int i, j, fail;
	fail = 0;
	
	// Compare C1 and C2
	for (i=0; i < NI; i++) 
	{
		for (j=0; j < NJ; j++) 
		{
			if (percentDiff(C[i*NJ + j], C_outputFromGpu[i*NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void gemm_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{	
		c[i * NJ + j] *= BETA;
		int k;
		for(k=0; k < NK; k++)
		{
			c[i * NJ + j] += ALPHA * a[i * NK + k] * b[k * NJ +j];
		}
	}
}


void gemmCuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	double t_malloc_s, t_malloc_e, t_memcpy_h2d, t_memcpy_d2h, t_lauch, t_free;

	t_malloc_s = rtclock();
	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
	t_malloc_e = rtclock();
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
	t_memcpy_h2d = rtclock();
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NI)/ ((float)block.x) )),(size_t)(ceil( ((float)NJ)/ ((float)block.y) )));

	t_start = rtclock();
	// for (int i = 0; i < 1024; i++){
		gemm_kernel<<< grid, block >>>(A_gpu, B_gpu, C_gpu);
	// }
	t_lauch = rtclock();
	cudaDeviceSynchronize();
	t_end = rtclock();
	// fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	// FILE * fp = fopen("/shared/uvm_bench/log/gemm-100m.txt", "a");
	// if (fp == NULL) {
	// 	fprintf(stderr, "Error opening file!\n");
	// 	exit(1);
	// }
	// fprintf(fp, "%0.6lf\n", t_end - t_start);
	cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);    
	t_memcpy_d2h = rtclock();
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	t_free = rtclock();
	// =================================================================
	double bt_malloc = t_malloc_e - t_malloc_s;
	double bt_memcpy_h2d = t_memcpy_h2d - t_malloc_e;
	double bt_lauch = (t_lauch - t_start);
	double bt_kernel = (t_end - t_lauch);
	double bt_memcpy_d2h = t_memcpy_d2h - t_end;
	double bt_free = t_free - t_memcpy_d2h;
	double bt_memset = 0.0;
	save_log(__FILE__, "nor-brk", "100mb", "%0.6lf,%0.6lf,%0.6lf,%0.6lf,%0.6lf,%0.6lf,%0.6lf\n", bt_malloc, bt_memcpy_h2d, bt_lauch, bt_kernel, bt_memcpy_d2h, bt_free,bt_memset);

}
	

int main(int argc, char *argv[])
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* C;  
	DATA_TYPE* C_outputFromGpu; 

	// A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE)); 
	// B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));   
	// C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 
	// C_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 


	cudaMallocHost((void **)&A,NI*NK*sizeof(DATA_TYPE)); 
	cudaMallocHost((void **)&B,NK*NJ*sizeof(DATA_TYPE));   
	cudaMallocHost((void **)&C,NI*NJ*sizeof(DATA_TYPE)); 
	cudaMallocHost((void **)&C_outputFromGpu,NI*NJ*sizeof(DATA_TYPE)); 


	init(A, B, C);
	
	GPU_argv_init();
	
	gemmCuda(A, B, C, C_outputFromGpu);

	// t_start = rtclock();	
	// gemm(A, B, C);
	// t_end = rtclock();
	fprintf(stdout, "C[0]: %0.6lf\n", C_outputFromGpu[0]);
	
	//compareResults(C, C_outputFromGpu);

	cudaFree(A);
	cudaFree(B);  
	cudaFree(C);  
	cudaFree(C_outputFromGpu); 

    	return 0;
}

