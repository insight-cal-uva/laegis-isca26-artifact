from common import *
import pandas as pd
import matplotlib.pyplot as plt



def draw_fig1_per_app(app_name):
    result = get_results_file(app_name,result_txt[1],commit_id)
    df = pd.read_csv(result[0])
    columns_of_interest = ["tlb_hit", "tlb_miss_pt_hit", "tlb_miss_pt_miss_outstand", "tlb_miss_pt_miss_pending"]
    df_filtered = df[["pfn"] + columns_of_interest]

    large_pfn = 32
    # Calculate the number of chunks
    num_chunks = len(df_filtered) // large_pfn

    # Aggregate every large_pfn PFNs
    aggregated_data = []
    for i in range(num_chunks + 1):
        chunk = df_filtered.iloc[i * large_pfn:(i + 1) * large_pfn]
        if not chunk.empty:
            aggregated_chunk = chunk[columns_of_interest].sum()
            aggregated_chunk["pfn"] = chunk["pfn"].iloc[0]
            aggregated_data.append(aggregated_chunk)

    df_aggregated = pd.DataFrame(aggregated_data)

    # Normalize the aggregated values
    #df_aggregated[columns_of_interest] = df_aggregated[columns_of_interest].div(df_aggregated[columns_of_interest].sum(axis=1), axis=0)

    # Set 'pfn' as index
    df_aggregated.set_index("pfn", inplace=True)

    # Plot stacked bar chart
    ax = df_aggregated.plot(kind='bar', stacked=True, figsize=(10,7),color=color2)

    # Set labels and title
    plt.xlabel(f'{app_name}-PFN')
    plt.ylabel('Normalized Counts')
    plt.title(f'Page Access Events by Aggregated {large_pfn} 4KB Pages')
    ax.set_xticklabels([''] * len(df_aggregated.index), rotation=45, ha='right')
    plt.legend(ncols=1)

    # Show the plot
    plt.tight_layout()
    figure_path = root_dir+"/saved_figures/"+"fig1-ori-"+app_name+".png"
    plt.savefig(figure_path,dpi=1200)
    plt.close()



draw_fig1_per_app(uvm_dirs[0])
draw_fig1_per_app(uvm_dirs[1])