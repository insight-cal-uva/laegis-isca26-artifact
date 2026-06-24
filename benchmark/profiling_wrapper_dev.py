import os
import argparse
import time
import pandas as pd
import subprocess
from time import localtime, strftime
from benchmark.uvmconfigs import all_supported_configs
from benchmark.configgen import *
from pathlib import Path


folders = ["BN", "knn", "logistic-regression", "rodinia", "polybench"]

subfolders = ["backprop", "dwt2d", "hotspot", "hotspot3D", "nw", "particlefilter", "pathfinder", "streamcluster"]

subfolders2 = ["2DCONV", "2MM", "3DCONV", "3MM", "ATAX", "BICG", "CORR", "COVAR", "FDTD-2D", 
               "GEMM-100MB", "GESUMMV", "GRAMSCHM", "MVT", "SYR2K", "SYRK"]

all_apps = ["knn", "logistic-regression", 
            "backprop", "dwt2d", "hotspot", "hotspot3D", "nw", "particlefilter", "pathfinder", "streamcluster", 
            "2DCONV", "2MM", "3DCONV", "3MM", "ATAX", "BICG", "CORR", "COVAR", "FDTD-2D", "GEMM-100MB", "GESUMMV", "GRAMSCHM", "MVT", "SYR2K", "SYRK"]


config_names = ['default-cc','default-uvm','prefetch', 'coalesce', 'batch_count', 'batch_per_serv', 'access_ctr', 'prefetch_ctr', 'prefetch_minf']

print_options = "--trace=cuda --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true --force-overwrite true "
print_options_cnn = "--trace=cuda --sample=none --cpuctxsw=none --cuda-um-cpu-page-faults=false --cuda-um-gpu-page-faults=false --force-overwrite true "
print_file = " --output "

cmd_kern_sum = "nsys stats --report cuda_gpu_kern_sum --format=csv " 
csv_kern_sum = "_cuda_gpu_kern_sum.csv"
# nsys stats --report cuda_gpu_kern_sum --format=csv --output cuda_gpu_kernel_summary.csv report1.nsys-rep


top_dir = "/asplos/uvm_bench/nsys_results"

# store the result at which level
def profile(config_name: str="default", postfix: str=None):

    # if cid!=-1:

    #     this_config = all_supported_configs[cid]
    #     config_name = this_config[1]
    #     postfix = this_config[2]
    # else:
    #     postfix = None
    
    if postfix is not None:
        config = config_name + '-' + postfix
    else:
        config = config_name

    result_path = top_dir
    output_path = result_path+"/"+config+"/"
    output_sub_path = result_path+"/"+config+"/"


    all_cmds = []

    try:
      os.mkdir(output_path)
      print(f"store to {output_path}")
      os.mkdir(output_path+"/uvm/")
      os.mkdir(output_path+"/nor/")
    except OSError:
      pass

    for UVM_flag in [True]:

        if UVM_flag:
            base_path = "UVM_benchmarks/"
        else:
            base_path = "non_UVM_benchmarks/"
        
        for i, rfolder in enumerate(folders):

            if rfolder != "rodinia" and rfolder != "polybench":
                
                if postfix is not None:
                    folder = rfolder +'-'+postfix
                else:
                    folder = rfolder
                
                path = base_path + rfolder + "/"
                
                if UVM_flag:   
                    command = "cd " + path + "; nsys profile " + print_options + print_file + output_path + "uvm" + "/"  + folder + " " + " ./run"
                else:
                    command = "cd " + path + "; nsys profile " + print_options + print_file + output_path + "nor" + "/"  + folder + " " + " ./run"                
                all_cmds.append(command)
                
            elif rfolder == "polybench":
                for j, rsubfolder in enumerate(subfolders2):
                    
                    if postfix is not None:
                        subfolder = rsubfolder +'-'+postfix
                    else:
                        subfolder = rsubfolder

                    path = base_path + rfolder + "/" + rsubfolder
                    if UVM_flag:
                        command = "cd " + path + "; nsys profile " + print_options + print_file + output_sub_path + "uvm" + "/"  + subfolder + " " + " ./run"
                    else:
                        command = "cd " + path + "; nsys profile " + print_options + print_file + output_sub_path + "nor" + "/"  + subfolder + " " + " ./run"                
                    all_cmds.append(command)
                    
            else:
                for j, rsubfolder in enumerate(subfolders):
                    
                    if postfix is not None:
                        subfolder = rsubfolder +'-'+postfix
                    else:
                        subfolder = rsubfolder

                    path = base_path + rfolder + "/" + rsubfolder
                    if UVM_flag:
                        command = "cd " + path + "; nsys profile " + print_options + print_file + output_sub_path + "uvm" + "/"  + subfolder + " " + " ./run"
                    else:
                        command = "cd " + path + "; nsys profile " + print_options + print_file + output_sub_path + "nor" + "/"  + subfolder + " " + " ./run"                
                    all_cmds.append(command)
    # print counters
    for idx, cmd in enumerate(all_cmds):
        warm_up_cmd = cmd.split(';')[0] +"; ./run"
        print("warm up once")
        os.system(warm_up_cmd)
        os.system(cmd)
        print(f"[{idx}] {cmd}")

# generate needed csv from nsys report
# the commands for nsys stats
# nsys stats --report osrt_sum --report cuda_api_sum --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum --report um_sum --report um_total_sum --format=csv --output . xxx
# all reports are stored results/
def process_csv(config_name: str="default", postfix: str=None):

    if postfix is not None:
        config = config_name + '-' + postfix
    else:
        config = config_name

    result_path = top_dir
    output_path = result_path+"/"+config+"/"


    csv_path =  output_path+"csv/"
    try:
        os.mkdir(csv_path)
    except OSError:
        pass
    all_cmd = []
    for UVM_flag in [True]:

  
        base_path2 = csv_path 
        report_path = output_path +"uvm/"
 
        
        try:
            os.mkdir(base_path2)
        except OSError:
            pass


        for i, app in enumerate(all_apps):
            # /shared/uvm_bench/cc-results/uvm/nw_summary_UVM.nsys-rep
            result_file = report_path + app +'-'+postfix+".nsys-rep"
            # /shared/uvm_bench/cc-csv/uvm/nw_CC_UVM
            csv_file_base = base_path2 + app+'-'+postfix
            stats_kern_sum_cmd = cmd_kern_sum + "--output " + csv_file_base + " " + result_file
            all_cmd.append(stats_kern_sum_cmd)
    
    for idx, cmd in enumerate(all_cmd):
        os.system(cmd)
        print(f"[{idx}] {cmd}")


def update_uvm_config(cid:int):
    dst_file = '/etc/modprobe.d/nvidia-uvm.conf'
    
    src_file = all_supported_configs[cid][0]
    postfix = all_supported_configs[cid][1]
    short = all_supported_configs[cid][2]

    with open(src_file,"r") as f:
        content = f.readlines()
    with open(dst_file, "w") as f:
        f.writelines(content)
    
    print("will use this config in next reboot")
    cmd_sh = 'cat '+ dst_file
    os.system(cmd_sh)
    print("reboot in 5s")
    time.sleep(5)
    # update the system
    cmd_cpy_persist='cp /asplos/nvidia-persistenced.service /usr/lib/systemd/system/nvidia-persistenced.service'
    cmd_reload='sudo systemctl daemon-reload'
    print(f"using config file {src_file}, folder {postfix}, short {short}")
    cmd_up = 'sudo update-initramfs -u'
    cmd_reb = 'sudo reboot'
    os.system(cmd_reload)
    os.system(cmd_cpy_persist)
    os.system(cmd_up)
    os.system(cmd_reb)

results = {}
# per config name used
def clean_csv(config_name: str):
    base_dir = Path("nsys_result_asplos")
    matching_folders = []

    # Find all folders matching the config_name
    for folder in base_dir.iterdir():
        if folder.is_dir() and config_name in folder.name:
            matching_folders.append(folder)       

    for folder in matching_folders:
            
        csv_dir = folder / "csv"
        
        if not csv_dir.exists():
            print(f"CSV folder does not exist in {folder}")
            continue
        
        if '-' in folder.name:
            postfix = '-'+folder.name.split('-')[-1]
        else:
            postfix = ''

        for app in all_apps:
            if config_name != 'default-uvm' and config_name!= 'default-cc':
                csv_file = csv_dir / f"{app}{postfix}_cuda_gpu_kern_sum.csv"
            else:
                csv_file = csv_dir / f"{app}_cuda_gpu_kern_sum.csv"

            npostfix=postfix[1:]
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    total_time = df['Total Time (ns)'].sum()
                    results.setdefault(app, {})[npostfix] = total_time
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
                    results.setdefault(app, {})[npostfix] = -1
            else:
                print(f"CSV file not found: {csv_file}")
                results.setdefault(app, {})[npostfix] = -1
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='UVM Parameter Tuning Script')
    
    parser.add_argument('--gen', '-g', help='update config', action='store_true',default=False)    
    parser.add_argument('--prof','-p', help='start profling', action='store_true', default=False)
    parser.add_argument('--csv','-c', help='generate csv files', action='store_true', default=False)
    parser.add_argument('--cid','-i', help='which config number to use', type=int, default=-1)
    parser.add_argument('--auto', '-a', help='automatic iterate over config id', action='store_true', default=False)
    parser.add_argument('--clean', '-d', help='generate cleaned data', action='store_true', default=False)
    args = parser.parse_args()

    # 1. python3 profiling_wrapper.py -g -i 0
    # 2. it will reboot
    # 3. then everytime after reboot: profiling_wrapper.py -g -a

    # set up config file only
    if args.gen:

        if args.cid == 0:
            with open("config-id", "w") as f:
                f.write('0')

        if args.auto == False and args.cid != -1:
            with open("config-id", "w") as f:
                f.write(str(args.cid))
            with open('profile-log',"a") as f:
                f.write(f"manual update to {args.cid}\n")
            update_uvm_config(args.cid)

        if args.auto:
            with open("config-id", "r") as f:
                status = f.read()
            
            if status != '-1':
                # current configuration id
                cid = int(status)
                dst_file = '/etc/modprobe.d/nvidia-uvm.conf'
                src_file = all_supported_configs[cid][0]
                postfix = all_supported_configs[cid][1]
                short = all_supported_configs[cid][2]
                with open('profile-log',"a") as f:
                    f.write(f"current config id {cid}, config name: {short}, file {src_file}, content below\n")
                with open(dst_file,"r") as f:
                    cur_config = f.readlines()
                with open("profile-log", "a") as f:
                    f.writelines(cur_config)

                
                profile(postfix,short)
                print("profile done")

                print(f"prepare next config")
                with open("config-id", "w") as f:
                    nid = cid + 1
                    f.write(str(nid))
                
                update_uvm_config(nid)

    else:
        if args.prof:
            profile()
        elif args.csv:
            with open('config-id',"r") as f:
                cid = int(f.read())
            
            for i in range (cid):
                cur_config = all_supported_configs[i]
                name = cur_config[1]
                postfix = cur_config[2]
                process_csv(name, postfix)
        elif args.clean:
            for cname in config_names:
                clean_csv(cname)
            
            summary_df = pd.DataFrame.from_dict(results, orient='index')
            summary_df.index.name = "App"
            summary_df.reset_index(inplace=True)

            # Sort all columns alphabetically, but keep 'App' as the first column
            cols = summary_df.columns.tolist()
            sorted_cols = ['App'] + sorted([col for col in cols if col != 'App'])

            summary_df = summary_df[sorted_cols]
            summary_df.to_csv("kernel_time_summary.csv", index=False)


            
    pass