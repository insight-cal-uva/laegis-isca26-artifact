from common import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def draw_fig1_per_app(app_name):
    result = get_results_file(app_name,result_txt[1],commit_id)
    df = pd.read_csv(result[0])
    columns_of_interest = ["tot_access_count"]
    df_filtered = df[["pfn"] + columns_of_interest]

    df_grouped = df_filtered.groupby("tot_access_count").size().reset_index(name='num_pages')
    df_grouped['log2_tot_access_count'] = np.floor(np.log2(df_grouped['tot_access_count']))
    print(df_grouped)
    
    # Aggregate the number of pages for each log2_tot_access_count
    df_aggregated = df_grouped.groupby('log2_tot_access_count')['num_pages'].sum().reset_index()

    # Create a uniform range for x-axis from 1 to the maximum log2 value
    max_log2_value = df_aggregated['log2_tot_access_count'].max()
    if (int(max_log2_value)<=16):
        x_ticks = range(1, 17)
    else:
        x_ticks = range(1, int(max_log2_value)+1)
    df_aggregated = df_aggregated.set_index('log2_tot_access_count').reindex(x_ticks, fill_value=0).reset_index()

    colors = [color2[7] if x < 10 else color2[5] for x in df_aggregated['log2_tot_access_count']]


    plt.figure(figsize=(10, 6))
    plt.bar(df_aggregated['log2_tot_access_count'], df_aggregated['num_pages'], width=0.8, color=colors)
    plt.xlabel(f'LOG2(Access Count)')
    plt.ylabel('#Pages')
    plt.title(f'Hot and Cold Pages')

    blue_patch = mpatches.Patch(color=color2[7], label='Cold')
    red_patch = mpatches.Patch(color=color2[5], label='Hot')
    plt.legend(handles=[blue_patch, red_patch])

    # Show the plot
    plt.tight_layout()
    figure_path = root_dir+"/saved_figures/"+"fig6-"+app_name+".png"
    plt.savefig(figure_path,dpi=1200)
    plt.close()




draw_fig1_per_app(uvm_dirs[0])
draw_fig1_per_app(uvm_dirs[1])