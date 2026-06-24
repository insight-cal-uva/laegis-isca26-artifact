from common import *
import pandas as pd
import matplotlib.pyplot as plt


def process_pf(app_name):
    result_pf = get_results_file(app_name,result_txt[3],commit_id)

    # Initialize empty lists to store data
    cycles = []
    n_pf = []
    pfns = []

    # Read the file line by line
    with open(result_pf[0], 'r') as file:
        lines = file.readlines()

    # Process the lines
    i = 2
    while i < len(lines):
        # Extract cycle and n_pf
        cycle_n_pf = lines[i].strip()
        cycle, n_pf_val = cycle_n_pf.split(',')
        cycles.append(cycle.strip())
        n_pf.append(int(n_pf_val.strip()))
        
        pages = lines[i+1].strip('').split(',')
        # Extract pfns
        pfn_list = []
        for pfn in pages:
            pfn_list.append(pfn.strip(''))
        pfns.append(pfn_list)
        
        i += 2

    agg_npf = []
    index = []
    blocks = len(n_pf) // 8

    for i in range(blocks + 1):
        low_b = i * 8
        high_b = min((i + 1) * 8, len(n_pf))  # Handle the case where the last block may have fewer than 8 elements
        new_num = sum(n_pf[low_b:high_b])
        agg_npf.append(new_num)
        index.append(i + 1)  # Start index from 1
    # Create DataFrame
    data = {'cycle-1000': index, 'n_pf': agg_npf}
    df = pd.DataFrame(data)
    #print(df)
    return df

def process_pe(app_name):
    result_pe = get_results_file(app_name,result_txt[2],commit_id)

    # Initialize empty lists to store data
    cycles = []
    n_pe = []
    pfns = []

    # Read the file line by line
    with open(result_pe[0], 'r') as file:
        lines = file.readlines()

    # Process the lines
    i = 2
    while i < len(lines):
        # Extract cycle and n_pf
        cycle_n_pe = lines[i].strip()
        cycle, n_pe_val = cycle_n_pe.split(',')
        cycles.append(cycle.strip())
        n_pe.append(int(n_pe_val.strip()))
        
        pages = lines[i+1].strip('').split(',')
        # Extract pfns
        pfn_list = []
        for pfn in pages:
            pfn_list.append(pfn.strip(''))
        pfns.append(pfn_list)
        
        i += 2

    agg_npe = []
    index = []
    blocks = len(n_pe) // 8

    for i in range(blocks + 1):
        low_b = i * 8
        high_b = min((i + 1) * 8, len(n_pe))  # Handle the case where the last block may have fewer than 8 elements
        new_num = sum(n_pe[low_b:high_b])
        agg_npe.append(new_num)
        index.append(i + 1)  # Start index from 1
    # Create DataFrame
    data = {'cycle-1000': index, 'n_pe': agg_npe}
    df = pd.DataFrame(data)
    #print(df)
    return df

def draw_fig3_per_app(app_name):
    
    
    df = process_pf(app_name)
    df2 = process_pe(app_name)

    plt.figure(figsize=(10, 7))
    plt.plot(df['cycle-1000'], df['n_pf'], marker='o', linestyle='-', color=color2[5], label='#PF')
    plt.plot(df2['cycle-1000'], df2['n_pe'], marker='x', linestyle='-', color=color2[10], label='#PE')
    plt.xlabel('Cycle*8000')
    plt.ylabel('Number of Events')
    plt.title('Number of Page Faults/Evictions During Execution')
    plt.legend()
    plt.tight_layout()
    figure_path = root_dir+"/saved_figures/"+"fig3-"+app_name+".png"
    plt.savefig(figure_path,dpi=1200)
    plt.close()

draw_fig3_per_app(uvm_dirs[0])
draw_fig3_per_app(uvm_dirs[1])