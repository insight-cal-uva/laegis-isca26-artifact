/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include "../../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU_DEVICE 0

/* Problem size */
#define tmax 500
#define NX 2048
#define NY 2048

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	int i, j;

  	for (i = 0; i < tmax; i++)
	{
		_fict_[i] = (DATA_TYPE) i;
	}
	
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			ex[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
			ey[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
			hz[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
		}
	}
}


void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	int t, i, j;
	
	for (t=0; t < tmax; t++)  
	{
		for (j=0; j < NY; j++)
		{
			ey[0*NY + j] = _fict_[t];
		}
	
		for (i = 1; i < NX; i++)
		{
       		for (j = 0; j < NY; j++)
			{
       			ey[i*NY + j] = ey[i*NY + j] - 0.5*(hz[i*NY + j] - hz[(i-1)*NY + j]);
        		}
		}

		for (i = 0; i < NX; i++)
		{
       		for (j = 1; j < NY; j++)
			{
				ex[i*(NY+1) + j] = ex[i*(NY+1) + j] - 0.5*(hz[i*NY + j] - hz[i*NY + (j-1)]);
			}
		}

		for (i = 0; i < NX; i++)
		{
			for (j = 0; j < NY; j++)
			{
				hz[i*NY + j] = hz[i*NY + j] - 0.7*(ex[i*(NY+1) + (j+1)] - ex[i*(NY+1) + j] + ey[(i+1)*NY + j] - ey[i*NY + j]);
			}
		}
	}
}


void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
{
	int i, j, fail;
	fail = 0;
	
	for (i=0; i < NX; i++) 
	{
		for (j=0; j < NY; j++) 
		{
			if (percentDiff(hz1[i*NY + j], hz2[i*NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
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



__global__ void fdtd_step1_kernel(DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NX) && (j < NY))
	{
		if (i == 0) 
		{
			ey[i * NY + j] = _fict_[t];
		}
		else
		{ 
			ey[i * NY + j] = ey[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[(i-1) * NY + j]);
		}
	}
}



__global__ void fdtd_step2_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < NX) && (j < NY) && (j > 0))
	{
		ex[i * (NY+1) + j] = ex[i * (NY+1) + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
	}
}


__global__ void fdtd_step3_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < NX) && (j < NY))
	{	
		hz[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * (NY+1) + (j+1)] - ex[i * (NY+1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
	}
}


void fdtdCuda(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, DATA_TYPE* hz_outputFromGpu)
{
	double t_start, t_end;

	DATA_TYPE *_fict_gpu;
	DATA_TYPE *ex_gpu;
	DATA_TYPE *ey_gpu;
	DATA_TYPE *hz_gpu;

	double t_malloc_s, t_malloc_e, t_memcpy_h2d, t_memcpy_d2h, t_kernel, t_lauch, t_free, t_memset;

	t_memset = 0.0;

	double s1 = rtclock();
	cudaMalloc((void **)&_fict_gpu, sizeof(DATA_TYPE) * tmax);
	cudaMalloc((void **)&ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1));
	cudaMalloc((void **)&ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY);
	cudaMalloc((void **)&hz_gpu, sizeof(DATA_TYPE) * NX * NY);
	double s2 = rtclock();
	cudaMemcpy(_fict_gpu, _fict_, sizeof(DATA_TYPE) * tmax, cudaMemcpyHostToDevice);
	cudaMemcpy(ex_gpu, ex, sizeof(DATA_TYPE) * NX * (NY + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(ey_gpu, ey, sizeof(DATA_TYPE) * (NX + 1) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
	double s3 = rtclock();


	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid( (size_t)ceil(((float)NY) / ((float)block.x)), (size_t)ceil(((float)NX) / ((float)block.y)));

	double t4, t5, t6, t7, t8, t9;
	for(int t = 0; t< tmax; t++)
	{
		t_start = rtclock();
		fdtd_step1_kernel<<<grid,block>>>(_fict_gpu, ex_gpu, ey_gpu, hz_gpu, t);
		t4 = rtclock();
		cudaDeviceSynchronize();
		t5 = rtclock();
		fdtd_step2_kernel<<<grid,block>>>(ex_gpu, ey_gpu, hz_gpu, t);
		t6 = rtclock();
		cudaDeviceSynchronize();
		t7 = rtclock();
		fdtd_step3_kernel<<<grid,block>>>(ex_gpu, ey_gpu, hz_gpu, t);
		t8 = rtclock();
		cudaDeviceSynchronize();
		t9 = rtclock();

		t_lauch += ((t4-t_start) + (t6-t5) + (t8-t7));
		t_kernel += ((t5-t4) + (t7-t6) + (t9-t8));
	}
    // fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	// save_log(__FILE__,"nor",NULL, "%0.6lf\n", t_end - t_start);
	double t = rtclock();
	cudaMemcpy(hz_outputFromGpu, hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyDeviceToHost);	
	double t10 = rtclock();
	cudaFree(_fict_gpu);
	cudaFree(ex_gpu);
	cudaFree(ey_gpu);
	cudaFree(hz_gpu);
	double t11 = rtclock();

	// =================================================================
	double bt_malloc = s2-s1;
	double bt_memcpy_h2d = s3-s2;
	double bt_lauch = t_lauch;
	double bt_kernel = t_kernel;
	double bt_memcpy_d2h = t10 - t;
	double bt_free = t11 - t10;
	double bt_memset = 0.0;
	save_log(__FILE__, "nor-brk", NULL, "%0.6lf,%0.6lf,%0.6lf,%0.6lf,%0.6lf,%0.6lf,%0.6lf\n", bt_malloc, bt_memcpy_h2d, bt_lauch, bt_kernel, bt_memcpy_d2h, bt_free, bt_memset);

}


int main()
{
	double t_start, t_end;

	DATA_TYPE* _fict_;
	DATA_TYPE* ex;
	DATA_TYPE* ey;
	DATA_TYPE* hz;
	DATA_TYPE* hz_outputFromGpu;

	// _fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
	// ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
	// ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
	// hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	// hz_outputFromGpu = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));


	cudaMallocHost((void **)&_fict_,tmax*sizeof(DATA_TYPE));
	cudaMallocHost((void **)&ex,NX*(NY+1)*sizeof(DATA_TYPE));
	cudaMallocHost((void **)&ey,(NX+1)*NY*sizeof(DATA_TYPE));
	cudaMallocHost((void **)&hz,NX*NY*sizeof(DATA_TYPE));
	cudaMallocHost((void **)&hz_outputFromGpu,NX*NY*sizeof(DATA_TYPE));


	init_arrays(_fict_, ex, ey, hz);

	GPU_argv_init();
	fdtdCuda(_fict_, ex, ey, hz, hz_outputFromGpu);

	// t_start = rtclock();
	// runFdtd(_fict_, ex, ey, hz);
	// t_end = rtclock();
	
	// fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	// compareResults(hz, hz_outputFromGpu);

	printf("hz[0] = %f\n", hz_outputFromGpu[0]);

	cudaFree(_fict_);
	cudaFree(ex);
	cudaFree(ey);
	cudaFree(hz);
	cudaFree(hz_outputFromGpu);

	return 0;
}

