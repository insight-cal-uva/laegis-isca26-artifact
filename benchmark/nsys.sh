CUDA_VISIBLE_DEVICES=GPU-b6c540e1-c39b-dec8-9a68-080f9896391c 
nsys profile --trace=cuda --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true  --gpu-metrics-device=0 --gpu-metrics-frequency=20000 --force-overwrite true --output 


nsys stats --report osrt_sum --report cuda_api_sum --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum --report um_sum --report um_total_sum --format=csv --output . 


nsys stats --report osrt_sum --format=csv --output osrt_summary.csv report1.nsys-rep
nsys stats --report cuda_api_sum --format=csv --output cuda_api_summary.csv report1.nsys-rep
nsys stats --report cuda_gpu_kern_sum --format=csv --output cuda_gpu_kernel_summary.csv report1.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum --format=csv --output cuda_gpu_mem_time_summary.csv report1.nsys-rep
nsys stats --report um_sum --format=csv --output um_summary.csv report1.nsys-rep
nsys stats --report um_total_sum --format=csv --output um_total_summary.csv report1.nsys-rep


# do this under CC
find . -type f -exec sed -i 's|/usr/local/cuda/|/usr/local/cuda/|g' {} +



nsys profile --trace=cuda --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true  --gpu-metrics-device=0 --gpu-metrics-frequency=20000 --force-overwrite true --output /shared/uvm_bench/nsys_reports/