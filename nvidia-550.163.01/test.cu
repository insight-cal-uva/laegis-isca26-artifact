// #include <stdio.h>
// #include <fcntl.h>
// #include <unistd.h>
// #include <sys/ioctl.h>
// #include <errno.h>


// typedef struct
// {
//     int rmStatus;                       // Out
// } UVM_TEST_SEC2_SANITY_PARAMS;

// typedef struct
// {
//     unsigned long long     flags;       // IN
//     int rmStatus;                       // OUT
// } UVM_INITIALIZE_PARAMS;

// typedef struct {
//     int uvmFd;
//     int rmStatus;
// } UVM_MM_INITIALIZE_PARAMS;


// #define UVM_INITIALIZE           0x30000001
// #define UVM_MM_INITIALIZE        75


// int main(int argc, const char* argv[]) {
//     int fd = open("/dev/nvidia-uvm", O_RDONLY);
//     if (fd < 0) {
//         perror("open");
//         return 1;
//     }
//     if (argc < 2){
// 	printf("need test cmd\n");
//         return -1;
//     }

//     int cmd = 295;
//     printf("test %d\n",cmd);


//     UVM_INITIALIZE_PARAMS init_params = {0};
//     if (ioctl(fd, UVM_INITIALIZE, &init_params) != 0 || init_params.rmStatus != 0) {
//         printf("UVM_INITIALIZE failed: ioctl=%d, rmStatus=%d, errno=%d\n",
//                ioctl(fd, UVM_INITIALIZE, &init_params), init_params.rmStatus, errno);
//         return 1;
//     }
//     printf("UVM_INITIALIZE success.\n");

//     UVM_MM_INITIALIZE_PARAMS mm_params = {.uvmFd = fd};
//     if (ioctl(fd, UVM_MM_INITIALIZE, &mm_params) != 0 || mm_params.rmStatus != 0) {
//         printf("UVM_MM_INITIALIZE failed: rmStatus=%d, errno=%d\n",
//                mm_params.rmStatus, errno);
//         return 1;
//     }
//     printf("UVM_MM_INITIALIZE success.\n");


//     UVM_TEST_SEC2_SANITY_PARAMS params = {0};
//     int status = ioctl(fd, cmd, &params);
//     if (status < 0) {
//         perror("ioctl");
//         close(fd);
//         return 1;
//     }

//     printf("Test returned: %d, rmStatus: %d\n", status, params.rmStatus);
//     close(fd);
//     return 0;
// }

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

    printf("cudaMallocManaged succeeded. ptr = %p\n", ptr);

    ((char*)ptr)[0] = 42;

    cudaFree(ptr);
    return 0;
}