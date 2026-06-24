from common import *
import pandas as pd
import matplotlib.pyplot as plt



def draw_fig2_per_app(app_name):
    result = get_results_file(app_name,result_txt[0],commit_id)
    df = pd.read_csv(result[0], names=['pfn', 'addr', 'size', 'cycle', 'RW', 'SM', 'Warp'])
    columns_of_interest = ["cycle", "RW"]
    df_filtered = df[["pfn"] + columns_of_interest]
    print(df_filtered)
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.figure(figsize=(10, 7))
    plt.plot(df['cycle'], df['pfn'], marker='o',linestyle='dashed', color=color2[10])
    plt.xlabel('Cycle')
    plt.ylabel('Accessed PFN')
    plt.title('Access Pattern During Execution')
    plt.legend('Access')
    plt.tight_layout()
    figure_path = root_dir+"/saved_figures/"+"fig2-"+app_name+".png"
    plt.savefig(figure_path,dpi=1200)
    plt.close()



draw_fig2_per_app(uvm_dirs[0])
draw_fig2_per_app(uvm_dirs[1])