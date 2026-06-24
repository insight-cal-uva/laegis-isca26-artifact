import os
from benchmark.metric_list import metrics
# print(metrics[0])
import time
from time import localtime, strftime


# 33 apps in total
folders = ["BN", "bfs", "CNN", "kmeans", "knn", "logistic-regression", "SVM", "rodinia", "polybench"]
#folders = ["rodinia", "polybench"]
subfolders = ["backprop", "dwt2d", "gaussian", "hotspot", "hotspot3D", "nn", "nw", "particlefilter", "pathfinder", "srad", "streamcluster"]
subfolders2 = ["2DCONV", "2MM", "3DCONV", "3MM", "ATAX", "BICG", "CORR", "COVAR", "FDTD-2D", "GEMM", "GESUMMV", "GRAMSCHM", "MVT", "SYR2K", "SYRK"]

all_apps = ["BN", "bfs", "CNN", "kmeans", "knn", "logistic-regression", "SVM", "backprop", "dwt2d", "gaussian", "hotspot", "hotspot3D", "nn", "nw", "particlefilter", "pathfinder", "srad", "streamcluster","2DCONV", "2MM", "3DCONV", "3MM", "ATAX", "BICG", "CORR", "COVAR", "FDTD-2D", "GEMM", "GESUMMV", "GRAMSCHM", "MVT", "SYR2K", "SYRK"]

print_options = "--trace=cuda,osrt --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true  --gpu-metrics-device=0 --gpu-metrics-frequency=10000 --force-overwrite true "
print_file = " --output "

output_path = " ../../results/"
output_sub_path = " ../../../results/"

abs_path = "/shared/uvm_bench/"


cmd_osrt = "nsys stats --report osrt_sum --format=csv "
cmd_api_sum = "nsys stats --report cuda_api_sum --format=csv " 
cmd_kern_sum = "nsys stats --report cuda_gpu_kern_sum --format=csv " 
cmd_mem_time = "nsys stats --report cuda_gpu_mem_time_sum --format=csv " 
cmd_um = "nsys stats --report um_sum --format=csv "
cmd_um_tot = "nsys stats --report um_total_sum --format=csv " 
cmd_save = " --output "

# change this to real nsys output file name
csv_osrt = "_osrt_sum.csv"
csv_api_sum = "_cuda_api_sum.csv"
csv_kern_sum = "_cuda_gpu_kern_sum.csv"
csv_mem_time = "_cuda_gpu_mem_time_sum.csv"
csv_um = "_um_sum.csv"
csv_um_tot = "_um_total_sum.csv"
# nsys stats --report osrt_sum --format=csv --output osrt_summary.csv report1.nsys-rep
# nsys stats --report cuda_api_sum --format=csv --output cuda_api_summary.csv report1.nsys-rep
# nsys stats --report cuda_gpu_kern_sum --format=csv --output cuda_gpu_kernel_summary.csv report1.nsys-rep
# nsys stats --report cuda_gpu_mem_time_sum --format=csv --output cuda_gpu_mem_time_summary.csv report1.nsys-rep
# nsys stats --report um_sum --format=csv --output um_summary.csv report1.nsys-rep
# nsys stats --report um_total_sum --format=csv --output um_total_summary.csv report1.nsys-rep



def process_csv_app():
        # get file folder and file name
    for CC_Flag in [False, True]:
        base_path = "cc-results/"
        csv_path = "cc-csv/" 
        if not CC_Flag:
            base_path = "non-cc-results/"
            csv_path = "non-cc-csv/"
        
        for UVM_flag in [False,True]:
            base_path2 = base_path + "non-uvm/" if not UVM_flag else base_path + "uvm/"
            csv_path2 = csv_path + "non-uvm/" if not UVM_flag else csv_path + "uvm/"
            # print(base_path2)
            # print(csv_path2)

            for i, app in enumerate(all_apps):
                # /shared/uvm_bench/cc-results/uvm/nw_summary_UVM.nsys-rep
                result_file = abs_path + base_path2 + app + "_summary_" + ("UVM" if UVM_flag else "non_UVM")+".nsys-rep"
                # /shared/uvm_bench/cc-csv/uvm/nw_CC_UVM
                csv_file_base = abs_path + csv_path2 + app +("_CC_" if CC_Flag else "_non_CC_")+ ("UVM" if UVM_flag else "non_UVM")
                # print(result_file)
                # print(csv_file_base)
                # prepare commands
                csv_file_osrt = csv_file_base + csv_osrt
                csv_file_api_sum = csv_file_base + csv_api_sum
                csv_file_kern_sum = csv_file_base + csv_kern_sum
                csv_file_mem_time = csv_file_base + csv_mem_time
                csv_file_um = csv_file_base + csv_um
                csv_file_um_tot = csv_file_base + csv_um_tot

                print(csv_file_osrt)
                print(csv_file_api_sum)
                print(csv_file_kern_sum)
                print(csv_file_mem_time)
                print(csv_file_um)
                print(csv_file_um_tot)     



if __name__ == "__main__":
    #profile()
    process_csv_app()