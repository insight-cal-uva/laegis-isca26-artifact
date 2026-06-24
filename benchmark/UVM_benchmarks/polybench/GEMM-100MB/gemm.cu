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

double t_kernel;
double t_lauch;

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


void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *A_gpu, DATA_TYPE *B_gpu, DATA_TYPE *C_gpu)
{
	int i, j;

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NK; j++)
		{
			A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
			A_gpu[i*NK + j] = ((DATA_TYPE) i*j) / NI;
		}
	}

  	for (i = 0; i < NK; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
			  B[i*NJ + j] = ((DATA_TYPE) i*j + 1) / NJ;
			  B_gpu[i*NJ + j] = ((DATA_TYPE) i*j + 1) / NJ;
		}
	}

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
			  C[i*NJ + j] = ((DATA_TYPE) i*j + 2) / NJ;
			  C_gpu[i*NJ + j] = ((DATA_TYPE) i*j + 2) / NJ;
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


void gemmCuda(DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, DATA_TYPE* C_gpu)
{
	double t_start, t_end;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NI)/ ((float)block.x) )),(size_t)(ceil( ((float)NJ)/ ((float)block.y) )));

	t_start = rtclock();
	gemm_kernel<<< grid, block >>>(A_gpu, B_gpu, C_gpu);
	double lauch_e = rtclock();
	cudaDeviceSynchronize();
	t_end = rtclock();

	t_lauch = lauch_e - t_start;
	t_kernel = t_end - lauch_e;


	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_lauch + t_kernel);   
	// FILE * fp = fopen("/shared/uvm_bench/log/u-gemm-100mb.txt", "a");
	// if (fp == NULL) {
	// 	fprintf(stderr, "Error opening file!\n");
	// 	exit(1);
	// }
	// fprintf(fp, "%0.6lf\n", t_end - t_start);   
}
	

int main(int argc, char *argv[])
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* C; 
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu; 

	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE)); 
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));   
	C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 

	double malloc_s = rtclock();
	cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMallocManaged(&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMallocManaged(&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
	double malloc_e = rtclock();



	init(A, B, C, A_gpu, B_gpu, C_gpu);
	
	GPU_argv_init();
	
	gemmCuda(A_gpu, B_gpu, C_gpu);

	// t_start = rtclock();	
	// gemm(A, B, C);
	// t_end = rtclock();
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	fprintf(stdout,"C[0]: %f\n",C_gpu[0]);


	// compareResults(C, C_gpu);

	free(A);
	free(B);  
	free(C);  

	double free_s = rtclock();
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	double free_e = rtclock();


	double bt_malloc = malloc_e - malloc_s;
	double bt_memcpy_h2d = 0.0;
	double bt_lauch =  t_lauch;
	double bt_kernel = t_kernel;
	double bt_memcpy_d2h = 0.0;
	double bt_free = free_e - free_s;
	double bt_memset = 0.0;
	save_log(__FILE__, "uvm-brk", "100mb", "%0.6lf,%0.6lf,%0.6lf,%0.6lf,%0.6lf,%0.6lf,%0.6lf\n", bt_malloc, bt_memcpy_h2d, bt_lauch, bt_kernel, bt_memcpy_d2h, bt_free,bt_memset);


    return 0;
}

