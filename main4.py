import argparse
import matplotlib.pyplot as plt
import pandas as pd
from utils import *

def plot_response_files(trace_response_data,duration_labels, expected_throughput_data, measured_throughput_file):
    df = pd.DataFrame(trace_response_data)
    num_subplots = 4
    height_ratios = [6] + [1] * (num_subplots - 1)

    fig, axes = plt.subplots(num_subplots, 1, figsize=(16, 5 * num_subplots), gridspec_kw={'height_ratios': height_ratios})
    plot_rate_vs_time(df, axes[0])
    plot_power_vs_time(df, axes[1])
    plot_throughput_vs_time(measured_throughput_file, axes[2], 'Measured')
    plot_throughput_vs_time(expected_throughput_data, axes[3], 'Expected')

    # Adding custom legend for durations
    custom_legend = [plt.Line2D([0], [0], color="white", lw=0)] * len(duration_labels)
    fig.legend(custom_legend, duration_labels, title="Duration per Mode",bbox_to_anchor=(0.5, 1.05), ncol=1, loc='upper center')
    fig.subplots_adjust(top=0.9, right= 0.75)

    plt.tight_layout()
    plt.savefig('plots4.png', bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files in a directory.")
    parser.add_argument("directory", type=str, help="The directory path to process files from.")
    args = parser.parse_args()

    expected_throughput_files, measured_throughput_file, trace_response_files = categorize_files(args.directory)
    expected_throughput_data = process_expected_throughput_files(expected_throughput_files)
    measured_throughput_data = process_measured_throughput_file(measured_throughput_file)
    trace_response_data, duration_labels = process_trace_response_files(trace_response_files)
    plot_response_files(trace_response_data, duration_labels, expected_throughput_data, measured_throughput_data)
