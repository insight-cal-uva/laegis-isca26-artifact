from common import *
import pandas as pd
import matplotlib.pyplot as plt



def draw_fig7():
    all_df =[]
    app_names =[]
    for app_name in uvm_dirs:
        result = get_results_file(app_name,result_txt[5],commit_id)
        df = pd.read_csv(result[0])
        all_df.append(df)
        app_names.append(app_name)

    combined_df = pd.concat(all_df, ignore_index=True)

    # Step 3: Plotting a stacked bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot stacked bars
    combined_df.plot(kind='bar', stacked=True, color=color2, ax=ax)

    # Set labels and title
    plt.xlabel(f'Applications')
    plt.ylabel('Counts')
    plt.title(f'Page Access Events')
    plt.legend(ncol=1)

    ax.set_xticks(range(len(app_names)))  # Set ticks for each application
    ax.set_xticklabels(app_names, rotation=45, ha='right')  # Set labels based on app_names with rotation for better readability

    # Show the plot
    plt.tight_layout()
    figure_path = root_dir+"/saved_figures/"+"fig7"+".png"
    plt.savefig(figure_path,dpi=1200)
    plt.close()
    #plt.show()

draw_fig7()