import argparse
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from utils import *

def plot_response_files(trace_response_data, measured_throughput_data):
    df = pd.DataFrame(trace_response_data)
    pd.set_option('display.max_rows', None)
    min_time = df['time'].min()
    max_time = df['time'].max()
    time_range = max_time - min_time
    width_factor = 0.1
    if time_range <= 120:
        fig_width = 16
    else:
        fig_width = time_range * width_factor

    base_height = 10
    num_subplots = 3
    total_height = base_height * num_subplots
    fig = plt.figure(figsize=(fig_width, total_height))

    gs = gridspec.GridSpec(num_subplots, 1, height_ratios=[15, 3, 3])
    ax1 = fig.add_subplot(gs[0])
    bin_edges = get_bin_edges(min_time, max_time, 10)
    label_numbers, rate_x_limit = plot_rate_vs_time(df, ax1)

    ax2 = fig.add_subplot(gs[1])
    line_positions, power_bins = plot_power_vs_time(df, ax2, bin_edges, label_numbers, rate_x_limit)

    ax3 = fig.add_subplot(gs[2])
    plot_throughput_vs_time(measured_throughput_data, ax3, bin_edges, line_positions, power_bins)
    fig.subplots_adjust(top=0.9, right=0.75)
    plt.tight_layout()
    plt.savefig('new_design_large_data1.png', bbox_inches='tight', pad_inches=0.1, dpi=700)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files in a directory.")
    parser.add_argument("directory", type=str, help="The directory path to process files from.")
    args = parser.parse_args()
    expected_throughput_files, measured_throughput_file, trace_response_files = categorize_files(args.directory)
    measured_throughput_data = process_measured_throughput_file(measured_throughput_file)
    trace_response_data = process_trace_response_files(trace_response_files)
    plot_response_files(trace_response_data, measured_throughput_data)
