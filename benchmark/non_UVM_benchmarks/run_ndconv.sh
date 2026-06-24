#!/bin/bash

# Define an array of executables
executables=(
# non uvm
# "/shared/uvm_bench/non_UVM_benchmarks/polybench/GEMM-1MB/gemm-1mb.exe" \
# "/shared/uvm_bench/non_UVM_benchmarks/polybench/GEMM-100MB/gemm-100mb.exe"
# "/shared/uvm_bench/non_UVM_benchmarks/polybench/GEMM-1024MB/gemm-1024mb.exe"\
# /shared/uvm_bench/non_UVM_benchmarks/polybench/GEMM-4MB/gemm-4mb.exe   \
# /shared/uvm_bench/non_UVM_benchmarks/polybench/GEMM-64MB/gemm-64mb.exe   \
# /shared/uvm_bench/non_UVM_benchmarks/polybench/GEMM-400MB/gemm-400mb.exe   \
# /shared/uvm_bench/non_UVM_benchmarks/polybench/GEMM-900MB/gemm-900mb.exe   \
# /shared/uvm_bench/non_UVM_benchmarks/polybench/GEMM-1600MB/gemm-1600mb.exe   \
"/shared/uvm_bench/non_UVM_benchmarks/polybench/2DCONV/2DConvolution.exe"
# "/shared/uvm_bench/non_UVM_benchmarks/polybench/3DCONV/3DConvolution.exe"
# "/shared/uvm_bench/non_UVM_benchmarks/polybench/ATAX/atax.exe"
# "/shared/uvm_bench/non_UVM_benchmarks/polybench/CORR/correlation.exe"
#  "/shared/uvm_bench/non_UVM_benchmarks/polybench/FDTD-2D/fdtd2d.exe"
# "/shared/uvm_bench/non_UVM_benchmarks/polybench/GRAMSCHM/gramschmidt.exe"
# "/shared/uvm_bench/non_UVM_benchmarks/polybench/SYR2K/syr2k.exe"
# "/shared/uvm_bench/non_UVM_benchmarks/polybench/2MM/2mm.exe"
# "/shared/uvm_bench/non_UVM_benchmarks/polybench/3MM/3mm.exe"
# "/shared/uvm_bench/non_UVM_benchmarks/polybench/BICG/bicg.exe"
# "/shared/uvm_bench/non_UVM_benchmarks/polybench/COVAR/covariance.exe"
# "/shared/uvm_bench/non_UVM_benchmarks/polybench/GESUMMV/gesummv.exe"
# "/shared/uvm_bench/non_UVM_benchmarks/polybench/MVT/mvt.exe"
# "/shared/uvm_bench/non_UVM_benchmarks/polybench/SYRK/syrk.exe"
# "/shared/uvm_bench/non_UVM_benchmarks/bfs/main 5 </shared/uvm_bench/data/bfs/inputGen/graph16M.txt"
# "/shared/uvm_bench/non_UVM_benchmarks/BN/ordergraph"
#"/shared/uvm_bench/non_UVM_benchmarks/kmeans/kmeans_cuda 2 /shared/uvm_bench/data/kmeans/1000000_points.txt 1000000"
#"/shared/uvm_bench/non_UVM_benchmarks/knn/knn"
#"/shared/uvm_bench/non_UVM_benchmarks/logistic-regression/gpu_exec"
# "/shared/uvm_bench/non_UVM_benchmarks/CNN/CNN"
# "/shared/uvm_bench/non_UVM_benchmarks/rodinia/backprop/backprop 65536"
)

# executables=(
#     ("/shared/uvm_bench/non_UVM_benchmarks/bfs/main" "5" "<" "/shared/uvm_bench/data/bfs/inputGen/graph16M.txt")
#     ("/shared/uvm_bench/non_UVM_benchmarks/BN/ordergraph")
#     ("/shared/uvm_bench/non_UVM_benchmarks/kmeans/kmeans_cuda" "2" "/shared/uvm_bench/data/kmeans/1000000_points.txt" "1000000")
#     ("/shared/uvm_bench/non_UVM_benchmarks/knn/knn")
#     ("/shared/uvm_bench/non_UVM_benchmarks/logistic-regression/gpu_exec")
#     ("/shared/uvm_bench/non_UVM_benchmarks/CNN/CNN")
# )

export CUDA_VISIBLE_DEVICES=0

# Loop through each executable and run it 15 times
for exe in "${executables[@]}"; do
    for i in {1..20}; do
        eval "$exe"
    done
done
