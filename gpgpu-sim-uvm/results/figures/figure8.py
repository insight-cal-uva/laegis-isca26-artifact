from common import *
import pandas as pd
import matplotlib.pyplot as plt



# Tot_kernel_exec_time: 2780119(cycle), 3085.592773(us)
# Tot_kernel_exec_time_and_fault_time: 28242379(cycle), 31345.593750(us)

def find_latest_numbers(fpath):
    with open(fpath, 'r') as file:
        lines = file.readlines()
    
    # Reverse the lines
    lines.reverse()
    
    # Initialize variables to store the cycle values
    tot_kernel_exec_time_cycle = None
    tot_kernel_exec_time_and_fault_time_cycle = None
    
    # Iterate through the lines in reverse order
    for i in range(len(lines) - 1):
        if 'Tot_kernel_exec_time:' in lines[i]:
            print(lines[i])
            # Extract the cycle value for Tot_kernel_exec_time
            tot_kernel_exec_time_cycle = int(lines[i].split(':')[1].split('(')[0].strip())
            # Check the next line for Tot_kernel_exec_time_and_fault_time
            if 'Tot_kernel_exec_time_and_fault_time:' in lines[i - 1]:
                print(lines[i-1])
                tot_kernel_exec_time_and_fault_time_cycle = int(lines[i - 1].split(':')[1].split('(')[0].strip())
                break
    
    return tot_kernel_exec_time_cycle, tot_kernel_exec_time_and_fault_time_cycle

def draw_fig8():
    exe_time =[]
    pf_time = []
    app_names =[]
    for app_name in uvm_dirs:
        result = get_results_file(app_name,result_txt[5],commit_id)
        tot_time, tot_time_pf = find_latest_numbers(result[1])
        print(tot_time,tot_time_pf)
        app_names.append(app_name)
        exe_time.append(tot_time)
        pf_time.append(tot_time_pf-tot_time)
    
    data = {
    'app_name': app_names,
    'exe_time': exe_time,
    'pf_time': pf_time
    }
    df = pd.DataFrame(data)

    df['total_time'] = df['exe_time'] + df['pf_time']
    df['local_compute'] = df['exe_time'] / df['total_time']
    df['page_fault_io'] = df['pf_time'] / df['total_time']

    columns_of_interest = ["local_compute", "page_fault_io"]
    df_filtered = df[["app_name"] + columns_of_interest]


    # # Plotting
    # fig, ax = plt.subplots()

    # # Plot the bars
    # bar_width = 0.5
    # bar1 = ax.bar(df['app_name'], df['local_compute'], bar_width, label='Execution Time')
    # bar2 = ax.bar(df['app_name'], df['page_fault_io'], bar_width, bottom=df['local_compute'], label='Page Fault Time')

        # Set 'pfn' as index
    df_filtered.set_index("app_name", inplace=True)

    # Plot stacked bar chart
    ax = df_filtered.plot(kind='bar', stacked=True, figsize=(10,7),color=color2)


    # Add labels and title
    plt.xlabel('Applications')
    plt.ylabel('Normalized Time')
    plt.title('Local Compute Time and Page Fault Time')
    plt.xticks(rotation=0)
    plt.legend()

    # Display the plot
    #plt.show()
    figure_path = root_dir+"/saved_figures/"+"fig8"+".png"
    plt.savefig(figure_path,dpi=1200)
    plt.close()

draw_fig8()