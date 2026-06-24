#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    void *ptr = nullptr;
    size_t size = 4096;

    cudaError_t err = cudaMallocManaged(&ptr, size);
    if (err != cudaSuccess) {
        printf("cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaFree(ptr);
    return 0;
}
