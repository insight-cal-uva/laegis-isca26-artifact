import os
import csv

all_apps = [
    "BN", "CNN", "knn", "logistic-regression",
    "backprop", "dwt2d", "hotspot", "hotspot3D", "nw", "particlefilter", "pathfinder", "streamcluster",
    "2DCONV", "2MM", "3DCONV", "3MM", "ATAX", "BICG", "CORR", "COVAR",
    "FDTD-2D", "GEMM-100MB", "GESUMMV", "GRAMSCHM", "MVT", "SYR2K", "SYRK"
]

columns = [
    "pid", "comm", "global_batch_count", "num_cached_faults", "num_coalesced_faults",
    "t", "SERVICE_TIME", "HIGH_SERVICE_TIME", "MAP_TIME", "BATCH_TRANSFER_TIME",
    "WHOLE_COPY_TIME", "POP_PAGE_TIME", "UVM_TSIZE", "FETCH_FB_TIME", "PREPROCESS_FB_TIME",
    "VAB_CRT_TIME", "CPU_ENCRYPT_TIME", "CPU_ENCRYPT_SIZE", "GPU_DECRYPT_TIME", "GPU_DECRYPT_SIZE",
    "GPU_DEC_CMD_TIME", "BB_ALLOC_TIME", "BB_FREE_TIME", "END_PUSH_TIME",
    "PUSH_C2G_TIME", "PUSH_C2G_TIME_WAIT", "t_VAB_CRT"
]

configs = ["p1b256", "p1b1024","p31b256", "p31b1024", "p51b256", "p51b1024","p71b256", "p71b1024", "p91b256", "p91b1024"]

def process_log_file(input_path, output_path):
    rows = []
    with open(input_path, 'r') as f:
        for line in f:
            if '] ' in line:
                line = line.split('] ', 1)[1].strip()
            values = [v.strip() for v in line.split(',')]
            if len(values) == len(columns):
                rows.append(values)
            else:
                print(f"Skipping malformed line in {input_path}: {line}")

    if not rows:
        print(f"No valid data in {input_path}")
        return

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"Saved: {output_path} ({len(rows)} rows)")

def process_all_logs(config):

    base_folder = f"./dmesg_results/{config}"
    output_folder = f"./dmesg_results/csv/{config}"
    os.makedirs(output_folder, exist_ok=True)

    for app in all_apps:
        input_file1 = os.path.join(base_folder, f"{app}-{config}.txt")
        input_file2 = os.path.join(base_folder, f"{app}-{config}.txt")
        input_file = input_file2
        
        output_file = os.path.join(output_folder, f"{app}-{config}.csv")
        if not os.path.exists(output_file):
            if os.path.exists(input_file):
                process_log_file(input_file, output_file)
            else:
                print(f"Missing file: {input_file}")

# Example usage
for config in configs:
    process_all_logs(config)
