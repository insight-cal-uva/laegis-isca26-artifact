# parameters we focus
# prefetch enable or not
# default 1
prefetch = ["0", "1"]
DEFAULT_PREFETCH = "1"

# for replay
# default 2
DEFAULT_POLICY = "2"
replay_policy = ["0", "1", "2", "3"]

# default 1
DEFAULT_COAL = "1"
fault_coalesce = ["0", "1"]

# Number of entries that are fetched from the GPU fault buffer and serviced in batch
# default 256
DEFAULT_FBC = 256
fault_batch_count = ["1", "2", "4", "6", "8", "16", "32", "64", "128", "256", "512", "1024"]

# default 20
DEFAULT_BAPS = "20"
fault_max_ba_per_service = ["1", "2", "4", "6", "8", "16", "20", "32", "64", "128", "256", "512", "1024"]

# Number of remote accesses on a region required to trigger a notification.
# after that much time faults
# default 256
DEFAULT_CTR_TH = 256
access_ctr_thresh = ["1", "2", "4", "6", "8", "16", "32", "64", "128", "256", "512", "1024"]

# for the TBNp
# default 51
DEFAULT_PREFETCH_TH = 51
prefetch_threshold = ["1", "11", "21", "31", "41", "50", "51", "61", "71", "81", "91", "99", "100"]

# for min-faults before trigger prefetch
# default 1
DEFAULT_MINF = 1
prefetch_min_faults = ["1", "2", "4", "6", "8", "16", "32", "64", "128", "256", "512", "1024"]

# customized config: prefetch_threshold and fault_batch_count and access_ctr_thresh
custmz_pth = ["1", "11", "21", "31", "41"]
custmz_fbc = ["512", "1024"]
custmz_access = ["1", "1024"]

# ===================
import os
import argparse
# .../asplos26/resources
current_folder = os.path.dirname(os.path.realpath(__file__))
# ../asplos26/configs
config_basefolder = os.path.abspath(os.path.join(current_folder, os.path.pardir, "configs"))
# resources
folder_basename = os.path.basename(current_folder)

config_num = 0
all_configs = []
# ===================
def gen_config_spec(
    prefetch: str=DEFAULT_PREFETCH,
    fault_batch: str=DEFAULT_FBC,
    fault_coalesce: str=DEFAULT_COAL,
    fault_max_baps: str=DEFAULT_BAPS,
    access_ctr_th: str=DEFAULT_CTR_TH,
    prefetch_th: str=DEFAULT_PREFETCH_TH,
    prefetch_min_f: str=DEFAULT_MINF,
    replay_policy: str=DEFAULT_POLICY):
    
    name = f"p_{prefetch}-fb_{fault_batch}-fc_{fault_coalesce}-fmbaps_{fault_max_baps}-acth_{access_ctr_th}-pth_{prefetch_th}-pmf_{prefetch_min_f}-rp_{replay_policy}.conf"
    prefix = "options nvidia_uvm"
    return [f"""{prefix} uvm_perf_prefetch_enable={prefetch}
{prefix} uvm_perf_fault_batch_count={fault_batch}
{prefix} uvm_perf_fault_coalesce={fault_coalesce}
{prefix} uvm_perf_fault_max_batches_per_service={fault_max_baps}
{prefix} uvm_perf_access_counter_threshold={access_ctr_th}
{prefix} uvm_perf_prefetch_threshold={prefetch_th}
{prefix} uvm_perf_prefetch_min_faults={prefetch_min_f}
{prefix} uvm_perf_fault_replay_policy={replay_policy}""", name]

def prefetch_sensitivity():
    folder = os.path.abspath(os.path.join(config_basefolder, os.path.curdir, "prefetch"))
    try:
      os.mkdir(folder)
    except OSError:
      pass

    for v in prefetch:
        configs, cname = gen_config_spec(prefetch=v)
        filename = folder + "/" + cname
        
        short_config = "p_"+v
        config = [filename,"prefetch", short_config]
        if v!=DEFAULT_PREFETCH:
          all_configs.append(config)
      
        with open(filename,"w") as f:
            f.write(configs)

def policy_sensitivity():
    folder = os.path.abspath(os.path.join(config_basefolder, os.path.curdir, "policy"))
    try:
      os.mkdir(folder)
    except OSError:
      pass

    for v in replay_policy:
        configs, name = gen_config_spec(replay_policy=v)
        filename = folder + "/" + name
        
        short_config = "rp_"+v
        config = [filename,"policy", short_config]
        with open(filename,"w") as f:
            f.write(configs)

def coalesce_sensitivity():
    folder = os.path.abspath(os.path.join(config_basefolder, os.path.curdir, "coalesce"))
    try:
      os.mkdir(folder)
    except OSError:
      pass

    for v in fault_coalesce:
        configs, name = gen_config_spec(fault_coalesce=v)
        filename = folder + "/" + name
        
        short_config = "fc_"+v
        config = [filename,"coalesce", short_config]
        all_configs.append(config)
        with open(filename,"w") as f:
            f.write(configs)

def fault_batch_cnt_sensitivity():
    folder = os.path.abspath(os.path.join(config_basefolder, os.path.curdir, "batch_count"))
    try:
      os.mkdir(folder)
    except OSError:
      pass

    for v in fault_batch_count:
        configs, cname = gen_config_spec(fault_batch=v)
        filename = folder + "/" + cname
        
        short_config = "fb_"+v
        config = [filename,"batch_count", short_config]
        all_configs.append(config)
        with open(filename,"w") as f:
            f.write(configs)

def fault_max_ba_per_service_sensitivity():
    folder = os.path.abspath(os.path.join(config_basefolder, os.path.curdir, "batch_per_serv"))
    try:
      os.mkdir(folder)
    except OSError:
      pass

    for v in fault_max_ba_per_service:
        configs, cname = gen_config_spec(fault_max_baps=v)
        filename = folder + "/" + cname

        short_config = "fmbaps_"+v
        config = [filename,"batch_per_serv", short_config]
        all_configs.append(config)
        with open(filename,"w") as f:
            f.write(configs)

def access_ctr_thresh_sensitivity():
    folder = os.path.abspath(os.path.join(config_basefolder, os.path.curdir, "access_ctr"))
    try:
      os.mkdir(folder)
    except OSError:
      pass

    for v in access_ctr_thresh:
        configs, cname = gen_config_spec(access_ctr_th=v)
        filename = folder + "/" + cname
        
        short_config = "acth_"+v
        config = [filename,"access_ctr", short_config]
        all_configs.append(config)

        with open(filename,"w") as f:
            f.write(configs)

def prefetch_threshold_sensitivity():
    folder = os.path.abspath(os.path.join(config_basefolder, os.path.curdir, "prefetch_ctr"))
    try:
      os.mkdir(folder)
    except OSError:
      pass

    for v in prefetch_threshold:
        configs, cname = gen_config_spec(prefetch_th=v)
        filename = folder + "/" + cname

        short_config = "pth_"+v
        config = [filename,"prefetch_ctr", short_config]
        all_configs.append(config)
        with open(filename,"w") as f:
            f.write(configs)

def prefetch_min_faults_sensitivity():
    folder = os.path.abspath(os.path.join(config_basefolder, os.path.curdir, "prefetch_minf"))
    try:
      os.mkdir(folder)
    except OSError:
      pass

    for v in prefetch_min_faults:
        configs, cname = gen_config_spec(prefetch_min_f=v)
        filename = folder + "/" + cname

        short_config = "pmf_"+v
        config = [filename,"prefetch_minf", short_config]
        all_configs.append(config)

        with open(filename,"w") as f:
            f.write(configs)

def custmz_config():
    folder = os.path.abspath(os.path.join(config_basefolder, os.path.curdir, "customize"))
    try:
       os.mkdir(folder)
    except OSError:
       pass
    
    idx = 0
    for v1 in custmz_pth:
       for v2 in custmz_fbc:
          for v3 in custmz_access:
            configs, cname = gen_config_spec(prefetch_th=v1,fault_batch=v2, access_ctr_th=v3)
            filename = folder + "/" + cname
            short_config = "cus_"+str(idx)
            config = [filename,"customize", short_config]
            all_configs.append(config)
            idx+=1
            with open(filename,"w") as f:
              f.write(configs)
             
       
   


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='UVM Parameter Tuning Script')
    
    parser.add_argument('--gen', '-g', help='generate config files', action='store_true',default=True)    
    parser.add_argument('--dry', '-d', help='generate config files lists', action='store_true',default=True)
    parser.add_argument('--cus', '-c', help="generate customized config files", action='store_true', default=True)    
    args = parser.parse_args()

    if args.gen:
      try:
          os.mkdir(config_basefolder)
      except OSError:
          pass
      prefetch_sensitivity()
      policy_sensitivity()
      coalesce_sensitivity()
      fault_batch_cnt_sensitivity()
      fault_max_ba_per_service_sensitivity()
      access_ctr_thresh_sensitivity()
      prefetch_threshold_sensitivity()
      prefetch_min_faults_sensitivity()
      print("configuration generated")
    
    if args.cus:
      try:
        os.mkdir(config_basefolder)
      except OSError:
        pass
      custmz_config()
      print("customized gen")
    
    print(all_configs)

    if args.dry:
      with open("uvmconfigs.py","w") as f:
          f.write("all_supported_configs = [ ")
          for idx, v in enumerate(all_configs):
            v.append(idx)
            f.write(f"[\'{v[0]}\', \'{v[1]}\', \'{v[2]}\', \'{v[3]}\'],\n")
          f.write("]")
