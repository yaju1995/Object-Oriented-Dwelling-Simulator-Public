import numpy as np
import matplotlib.pyplot as plt


def SingleData_Line(data, xlabel=None, ylabel=None, title=None):
    time = np.arange(data.shape[0])
    plt.figure()
    data = data.flatten()  # Flatten (24, 1) to (24,)
    plt.plot(data, marker='o', linestyle='-', color='b', label='Usage')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{title} ')
    plt.grid(True)
    plt.show(block=True)


def SingleData_Bar_Line(data, xlabel=None, ylabel=None, title=None):
    time = np.arange(data.shape[0])
    plt.figure()
    data = data.flatten()  # Flatten (24, 1) to (24,)
    # plt.plot(data, marker='o', linestyle='-', color='b', label='Usage')
    plt.bar(time, data, color='blue', alpha=0.8, label='Usage')  # Bar plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{title} ')
    plt.grid(True)
    plt.show(block=True)


def DualData_Bar_Line_Single_axis(data1, data2, xlabel=None, ylabel=None, title=None, d1_label=None, d2_label=None, save_path=None):
    time = np.arange(data1.shape[0])
    plt.figure()
    data1 = data1.flatten()
    data2 = data2.flatten()
    plt.plot(data1, marker='o', linestyle='-', color='b', label=d1_label)
    plt.bar(time, data2, color='skyblue', alpha=0.5, label=d2_label)  # Bar plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(f'{title} ')
    plt.grid(True)
    plt.show(block=True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")


def DualData_Bar_Line_Dual_axis(data1, data2, xlabel=None, ylabel1=None, ylabel2=None, title=None, syn_axis=False):
    # Ensure data is 1-dimensional
    data1 = data1.flatten()
    data2 = data2.flatten()
    x = np.arange(data1.shape[0])

    # Create the figure and axes
    fig, ax1 = plt.subplots()

    # Plot the bar chart on the primary axis
    ax1.bar(x, data1, color='r', alpha=0.5, label=ylabel1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color='black')

    # Create the secondary axis and plot the line chart
    ax2 = ax1.twinx()
    ax2.set_ylabel(ylabel2, color='black')
    ax2.plot(x, data2, marker='o', linestyle='-', color='b', label=ylabel2)

    # Synchronize the y-axes if requested
    if syn_axis:
        max_val = max(data1.max(), data2.max())
        min_val = min(data1.min(), data2.min())
        ax1.set_ylim(min_val, max_val)
        ax2.set_ylim(min_val, max_val)

    # Add title and grid
    plt.title(title)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid()

    # Show the plot
    plt.show(block=True)


def DualData_Bar_Line_Dual_axis_with_side_plot(data1, data2, data3, xlabel=None, ylabel1=None, ylabel2=None,
                                               ylabel3=None,
                                               title=None, syn_axis=False):
    # Ensure data is 1-dimensional
    data1 = data1.flatten()
    data2 = data2.flatten()
    data3 = data3.flatten()
    x = np.arange(data1.shape[0])

    # Create the figure and axes with equal-sized subplots
    fig, (ax_main, ax_side) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]}, figsize=(12, 6))

    # Main plot (dual axis: bar + line)
    ax1 = ax_main
    ax1.bar(x, data1, color='r', alpha=0.5, label=ylabel1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.set_ylabel(ylabel2, color='black')
    ax2.plot(x, data2, marker='o', linestyle='-', color='b', label=ylabel2)
    ax2.tick_params(axis='y', labelcolor='black')

    if syn_axis:
        max_val = max(data1.max(), data2.max())
        min_val = min(data1.min(), data2.min())
        ax1.set_ylim(min_val, max_val)
        ax2.set_ylim(min_val, max_val)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax_main.set_title(title)
    ax_main.grid()

    # Side plot (e.g., a line plot for data3)
    ax_side.plot(x, data3, marker='o', linestyle='-', color='g', label=ylabel3)
    ax_side.set_xlabel(xlabel)
    ax_side.set_ylabel(ylabel3, color='black')
    ax_side.set_title(f"Side Plot: {ylabel3}")
    ax_side.grid()
    ax_side.legend(loc='upper left')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show(block=True)


def DualData_Bar_Line_Top_Bottom_Plots(data1, data2, data3, xlabel=None, ylabel1=None, ylabel2=None, ylabel3=None,
                                       title=None, syn_axis=False):
    # Ensure data is 1-dimensional
    data1 = data1.flatten()
    data2 = data2.flatten()
    data3 = data3.flatten()
    x = np.arange(data1.shape[0])

    # Create the figure and axes with vertically stacked subplots
    fig, (ax_main, ax_bottom) = plt.subplots(2, 1, figsize=(8, 10))

    # Main plot (dual axis: bar + line)
    ax1 = ax_main
    ax1.bar(x, data1, color='r', alpha=0.5, label=ylabel1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.set_ylabel(ylabel2, color='black')
    ax2.plot(x, data2, marker='o', linestyle='-', color='b', label=ylabel2)
    ax2.tick_params(axis='y', labelcolor='black')

    if syn_axis:
        max_val = max(data1.max(), data2.max())
        min_val = min(data1.min(), data2.min())
        ax1.set_ylim(min_val, max_val)
        ax2.set_ylim(min_val, max_val)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax_main.set_title(title)
    ax_main.grid()

    # Bottom plot (e.g., a line plot for data3)
    ax_bottom.plot(x, data3, marker='o', linestyle='-', color='g', label=ylabel3)
    ax_bottom.set_xlabel(xlabel)
    ax_bottom.set_ylabel(ylabel3, color='black')
    ax_bottom.set_title(f"Bottom Plot: {ylabel3}")
    ax_bottom.grid()
    ax_bottom.legend(loc='upper left')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show(block=True)


def ThreeData_Bar_Line_Dual_axis(data1, data2, data3, xlabel=None, ylabel1=None, ylabel2=None, ylabel3=None,
                                 title=None, syn_axis=False):
    # Ensure the data is 1-dimensional
    data1 = data1.flatten()
    data2 = data2.flatten()
    data3 = data3.flatten()

    x = np.arange(data1.shape[0])
    fig, ax1 = plt.subplots()

    # Plot the bar and first line on primary axis
    ax1.bar(x, data1, color='r', alpha=0.5, label=ylabel1)
    ax1.plot(x, data2, marker='o', linestyle='-', color='b', label=ylabel2)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(f'{ylabel1} and {ylabel2}', color='black')

    # Plot the second line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, data3, marker='o', linestyle='-', color='r', label=ylabel3)
    ax2.set_ylabel(ylabel3, color='black')

    if syn_axis:
        # Synchronize y-axis limits
        max_val = max(data1.max(), data2.max(), data3.max())
        min_val = min(data1.min(), data2.min(), data3.min())
        ax1.set_ylim(min_val, max_val)
        ax2.set_ylim(min_val, max_val)

    # Add title and legend
    plt.title(title)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Add grid for better readability
    plt.grid()

    # Show the plot
    plt.show(block=True)


def Three_Stacked_Plots(data1, data2, data3, xlabel=None, ylabel1=None, ylabel2=None, ylabel3=None,
                        title1=None, title2=None, title3=None):
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure data is 1-dimensional
    data1 = data1.flatten()
    data2 = data2.flatten()
    data3 = data3.flatten()
    x = np.arange(data1.shape[0])

    # Create the figure and axes with 3 vertically stacked subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Top plot: Two datasets (data1 and data2)
    ax1.plot(x, data1, marker='o', linestyle='-', color='b', label=ylabel1)
    ax1.plot(x, data2, marker='s', linestyle='--', color='r', label=ylabel2)
    ax1.set_ylabel(f"{ylabel1} & {ylabel2}")
    ax1.set_title(title1)
    ax1.legend(loc='upper right')
    ax1.grid()

    # Middle plot: Single dataset (data3)
    ax2.plot(x, data3, marker='o', linestyle='-', color='g', label=ylabel3)
    ax2.set_ylabel(ylabel3)
    ax2.set_title(title2)
    ax2.legend(loc='upper right')
    ax2.grid()



    # Bottom plot: All three datasets (data1, data2, and data3)
    ax3.plot(x, data1, marker='o', linestyle='-', color='b', label=ylabel1)
    ax3.plot(x, data2, marker='s', linestyle='--', color='r', label=ylabel2)
    ax3.plot(x, data3, marker='^', linestyle='-.', color='g', label=ylabel3)
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(f"{ylabel1}, {ylabel2}, & {ylabel3}")
    ax3.set_title(title3)
    ax3.legend(loc='upper right')
    ax3.grid()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show(block=True)


def Three_Stacked_Plots_Six_Datasets(data1, data2, data3, data4, data5, data6,data7,
                                  xlabel=None, ylabel1=None, ylabel2=None, ylabel3=None, ylabel4=None,
                                  ylabel5=None, ylabel6=None,ylabel7= None, title1=None, title2=None, title3=None,
                                     save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure data is 1-dimensional
    data1 = data1.flatten()
    data2 = data2.flatten()
    data3 = data3.flatten()
    data4 = data4.flatten()
    data5 = data5.flatten()
    data6 = data6.flatten()
    data7 = data7.flatten()
    x = np.arange(data1.shape[0])
    # If no custom x-values are provided, use default indices
    x_values = np.arange(data1.shape[0])

    # Create the figure and axes with 3 vertically stacked subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 7), sharex=True)

    # Top plot: Two datasets (data1 and data2)
    ax1.plot(x, data1, marker='', linestyle='-', color='b', label=ylabel1)
    ax1.plot(x, data2, marker='', linestyle='-', color='r', label=ylabel2)
    ax1.set_ylabel(f"Wh")
    ax1.set_title(title1)
    ax1.legend(loc='upper right')
    ax1.grid()


    # Bottom plot: Three datasets (data4, data5, data6)
    ax2.plot(x, data4, marker='', linestyle='-.', color='m', label=ylabel4)
    ax2.plot(x, data5, marker='', linestyle='-', color='c', label=ylabel5)
    ax2.plot(x, data6, marker='', linestyle='--', color='r', label=ylabel6)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(f"Prices $")
    ax2.set_title(title3)
    ax2.legend(loc='upper right')
    ax2.set_ylim(bottom=0,top=0.45)
    ax2.grid()

    # Middle plot: Single dataset (data3)
    ax3.plot(x, data3, marker='o', linestyle='-', color='g', label=ylabel3)
    ax3.set_ylabel(ylabel3)
    ax3.set_title(title2)
    ax3.legend(loc='upper left')
    ax3.grid()

    ax3_bar = ax3.twinx()
    ax3_bar.bar(x_values, data7, color='red', alpha=0.5, label=ylabel7)
    ax3_bar.set_ylabel(ylabel7)
    # if ylim7:  # Apply y-axis limits for secondary axis if provided
    #     ax2_bar.set_ylim(ylim7)
    ax3_bar.legend(loc='upper right')


    # Adjust layout and show the plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    # plt.show(block=True)


def Three_Stacked_Plots_Nine_Datasets(data1, data2, data3, data4,
                                      data5, data6, data7,
                                      data8, data9,
                                      xlabel=None,
                                      ylabel1=None, ylabel2=None, ylabel3=None, ylabel4=None,
                                      ylabel5=None, ylabel6=None, ylabel7=None,
                                      ylabel8=None, ylabel9=None,
                                      title1=None, title2=None, title3=None,
                                      save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt

    # Flatten all data
    data1 = data1.flatten()
    data2 = data2.flatten()
    data3 = data3.flatten()
    data4 = data4.flatten()
    data5 = data5.flatten()
    data6 = data6.flatten()
    data7 = data7.flatten()
    data8 = data8.flatten()
    data9 = data9.flatten()

    x = np.arange(data1.shape[0])
    x_values = np.arange(data1.shape[0])

    # Create the figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    # --- Top Plot: Four Datasets ---
    ax1.plot(x, data1, linestyle='-', color='b', label=ylabel1)
    ax1.plot(x, data2, linestyle='-.', color='b', label=ylabel2)
    ax1.plot(x, data3, linestyle='-', color='g', label=ylabel3)
    ax1.plot(x, data4, linestyle='-.', color='g', label=ylabel4)
    ax1.set_ylabel("Wh")
    ax1.set_title(title1)
    ax1.legend(loc='upper right')
    ax1.grid()

    # --- Middle Plot: Three Datasets ---
    ax2.plot(x, data5, linestyle='-', color='blue', label=ylabel5)
    ax2.plot(x, data6, linestyle='--', color='green', label=ylabel6)
    ax2.plot(x, data7, linestyle=':', color='red', label=ylabel7)
    ax2.set_ylabel("Prices $")
    ax2.set_title(title2)
    ax2.legend(loc='upper right')
    ax2.set_ylim(bottom=0, top=0.45)
    ax2.grid()

    # --- Bottom Plot: Line and Bar ---
    ax3.plot(x, data8, linestyle='-', marker='o', color='green', label=ylabel8)
    ax3.set_ylabel(ylabel8)
    ax3.set_title(title3)
    ax3.legend(loc='upper left')
    ax3.grid()

    ax3_bar = ax3.twinx()
    ax3_bar.bar(x_values, data9, color='red', alpha=0.5, label=ylabel9)
    ax3_bar.set_ylabel(ylabel9)
    ax3_bar.legend(loc='upper right')

    # Final layout
    ax3.set_xlabel(xlabel)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")



import matplotlib.pyplot as plt

def plot_energy_data(df, save_path=None):
    """
    Plot energy system data with Battery Action as bars,
    Set Point as dashed line, and SOC on secondary axis.
    """

    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    # 1. Generation vs Demand (actual & forecast)
    axs[0].plot(df['Time'], df['Generation_Wh'], label="Generation (Wh)", color="green")
    axs[0].plot(df['Time'], df['Forecasted Gen Wh'], '--', label="Forecasted Gen (Wh)", color="lime")
    axs[0].plot(df['Time'], df['Demand_Wh'], label="Demand (Wh)", color="red")
    axs[0].plot(df['Time'], df['Forecasted Dem Wh'], '--', label="Forecasted Dem (Wh)", color="orange")
    axs[0].set_ylabel("Wh")
    axs[0].legend()
    axs[0].grid(True)

    # 2. Import, Average, Export Prices
    axs[1].plot(df['Time'], df['Import_$'], label="Import Price ($)", color="black")
    # axs[1].plot(df['Time'], df['Average_$'], label="Average Price ($)", color="gray")
    axs[1].plot(df['Time'], df['Export_$'], label="Export Price ($)", color="purple")
    axs[1].set_ylabel("Price ($)")
    axs[1].legend()
    axs[1].grid(True)

    # 3. Battery Action (bar) + Set Point (line) + SOC (secondary axis)
    ax3 = axs[2]
    ax3.bar(df['Time'], df['Set_Point'],
            width=0.8, label="Set Point (Commanded)",
            color="orange", alpha=0.4)

    # Battery Action (actual) - narrower bar overlay
    ax3.bar(df['Time'], df['Battery_action'],
            width=0.6, label="Battery Action (Taken)",
            color="brown", alpha=0.8)

    # Add SOC on secondary axis
    ax3b = ax3.twinx()
    ax3b.plot(df['Time'], df['SOC_%'], label="SOC (%)", color="blue")
    ax3b.set_ylabel("SOC (%)")
    ax3b.legend(loc="upper right")

    ax3.set_ylabel("Battery Set/Action")
    ax3.legend(loc="upper left")
    ax3.grid(True)

    ax3b = ax3.twinx()
    ax3b.plot(df['Time'], df['SOC_%'], label="SOC (%)", marker='o',color="blue")
    ax3b.set_ylabel("SOC (%)")
    ax3b.legend(loc="upper right")

    # 4. Grid Energy & Cost (dual axis)
    ax4 = axs[3]
    ax4.bar(df['Time'], df['Grid_Energy'], label="Grid Energy", color="cyan", alpha=0.6)
    ax4.set_ylabel("Grid Energy (Wh)")
    ax4.legend(loc="upper left")
    ax4.grid(True)

    ax4b = ax4.twinx()
    ax4b.plot(df['Time'], df['Grid_Energy_Cost'], label="Grid Cost", marker='o',color="darkcyan")
    ax4b.set_ylabel("Grid Cost ($)")
    ax4b.legend(loc="upper right")

    # 5. Rewards
    axs[4].plot(df['Time'], df['Rewards_t'], label="Reward", marker='o',color="magenta")
    axs[4].set_ylabel("Reward")
    axs[4].set_xlabel("Time step")
    axs[4].legend()
    axs[4].grid(True)

    # Title & Layout
    plt.suptitle(f"Energy System Data for Day {df['Day'].iloc[0]}", fontsize=16, y=1.02)
    plt.tight_layout()

    # Save option
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Plot saved as {save_path}")



