#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <stdint.h>
//-----------------------------------------
#define NV_UUID_LEN 16
typedef struct {
    uint8_t uuid[NV_UUID_LEN];
} NvProcessorUuid;
typedef struct {
    NvProcessorUuid gpu_uuid;    // IN/OUT
    int32_t          numaEnabled; // OUT
    int32_t          numaNodeId;  // OUT
    int32_t          rmCtrlFd;    // IN
    uint32_t         hClient;     // IN
    uint32_t         hSmcPartRef; // IN
    int32_t          rmStatus;    // OUT
} UVM_REGISTER_GPU_PARAMS;

//-----------------------------------------
typedef struct
{
    int rmStatus;                            
} UVM_TEST_SEC2_SANITY_PARAMS;

// Tests the CSL/SEC2 encryption/decryption methods by doing a secure transfer
// of memory from CPU->GPU and a subsequent GPU->CPU transfer.
typedef struct
{
    int rmStatus;                       
    unsigned int my_test;
} UVM_TEST_SEC2_CPU_GPU_ROUNDTRIP_PARAMS;


typedef struct
{
    unsigned long long  flags;     
    int rmStatus;                    
} UVM_INITIALIZE_PARAMS;

typedef struct
{
    int uvmFd;   
    int rmStatus;
} UVM_MM_INITIALIZE_PARAMS;

typedef struct
{
    NvProcessorUuid gpu_uuid; // IN
    int       rmStatus; // OUT
} UVM_UNREGISTER_GPU_PARAMS;


#define UVM_IOCTL_BASE(i) i
#define UVM_TEST_IOCTL_BASE(i)              UVM_IOCTL_BASE(200 + i)
#define UVM_REGISTER_GPU                    UVM_IOCTL_BASE(37)
#define UVM_UNREGISTER_GPU                  UVM_IOCTL_BASE(38)
#define UVM_TEST_SEC2_SANITY                UVM_TEST_IOCTL_BASE(95)
#define UVM_TEST_SEC2_CPU_GPU_ROUNDTRIP     UVM_TEST_IOCTL_BASE(99)
#define UVM_INITIALIZE          0x30000001


int main(int argc, const char* argv[]) {

    int fd_uvm = open("/dev/nvidia-uvm", O_RDONLY);
    if (fd_uvm < 0) {
        perror("open");
        return 1;
    }
    int fd_rm = open("/dev/nvidiactl", O_RDONLY);
    if (fd_rm < 0) {
        perror("open");
        return 1;
    }


    UVM_INITIALIZE_PARAMS init_params = {0};
    int init_status = ioctl(fd_uvm, UVM_INITIALIZE, &init_params);
    if (init_status != 0 || init_params.rmStatus != 0) {
        printf("UVM_INITIALIZE failed: ioctl=%d, rmStatus=%d, errno=%d\n",init_status, init_params.rmStatus, errno);
        return -1;
    }
    printf("UVM_INITIALIZE success.\n");

    UVM_REGISTER_GPU_PARAMS params;
    memset(&params, 0, sizeof(params));
    sscanf("GPU-b6c540e1-c39b-dec8-9a68-080f9896391c",  // Replace with real UUID
           "GPU-%2hhx%2hhx%2hhx%2hhx-%2hhx%2hhx-%2hhx%2hhx-%2hhx%2hhx-%2hhx%2hhx%2hhx%2hhx%2hhx%2hhx",
           &params.gpu_uuid.uuid[0], &params.gpu_uuid.uuid[1], &params.gpu_uuid.uuid[2], &params.gpu_uuid.uuid[3],
           &params.gpu_uuid.uuid[4], &params.gpu_uuid.uuid[5],
           &params.gpu_uuid.uuid[6], &params.gpu_uuid.uuid[7],
           &params.gpu_uuid.uuid[8], &params.gpu_uuid.uuid[9],
           &params.gpu_uuid.uuid[10], &params.gpu_uuid.uuid[11], &params.gpu_uuid.uuid[12],
           &params.gpu_uuid.uuid[13], &params.gpu_uuid.uuid[14], &params.gpu_uuid.uuid[15]);
    params.rmCtrlFd = -1;
    // These handles would normally come from NVIDIA's RM client API
    // Here, we're just testing kernel routing; leave dummy
    params.hClient = 0x0;
    params.hSmcPartRef = 0x0;
    int ret = ioctl(fd_uvm, UVM_REGISTER_GPU, &params);
    if (ret < 0) {
        perror("ioctl UVM_REGISTER_GPU failed");
        return 1;
    }
    if (params.rmStatus != 0){
        printf("rmStatus = %x\n", params.rmStatus);
        return 1;
    }

    printf("GPU registered successfully\n");
    printf("NUMA: enabled=%d node=%d\n", params.numaEnabled, params.numaNodeId);


    // // Test SEC2 SANITY
    // UVM_TEST_SEC2_SANITY_PARAMS sec2_params = {0};
    // int sec2_status = ioctl(fd_uvm, UVM_TEST_SEC2_SANITY, &sec2_params);
    // if (sec2_status < 0) {
    //     perror("ioctl");
    //     close(fd_uvm);
    //     return 1;
    // }
    // printf("Test SEC2 returned: %d, rmStatus: %d\n", sec2_status, sec2_params.rmStatus);
    
    // for (int j = 0; j<10; j++){
        UVM_TEST_SEC2_CPU_GPU_ROUNDTRIP_PARAMS sec2_cg_params = {0, 0};
        int sec2_cg_status = ioctl(fd_uvm, UVM_TEST_SEC2_CPU_GPU_ROUNDTRIP, &sec2_cg_params);
        if (sec2_cg_status < 0){
            perror("ioctl");
            close(fd_uvm);
            return 1;        
        }
        printf("Test SEC2-CPU-GPU RT returned: %d, rmStatus: %d, my_test:%d\n", sec2_cg_status, sec2_cg_params.rmStatus, sec2_cg_params.my_test);
    // }
    close(fd_uvm);
    close(fd_rm);
    return 0;
}
