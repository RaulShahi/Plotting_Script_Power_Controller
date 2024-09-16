import os
import pandas as pd
from datetime import datetime, timedelta
import csv
import seaborn as sns
import numpy as np

def hex_to_int(hex_str):
    try:
        return int(hex_str, 16)
    except ValueError as e:
        print(f"Error converting '{hex_str} to integer: {e}'")
        return None

def hex_to_time(hex_time):
    try:
        nanoseconds = int(hex_time, 16)
        seconds = nanoseconds / 1e9
        delta = timedelta(seconds=seconds)
        epoch = datetime(1970, 1, 1)
        actual_time = epoch + delta
        return actual_time

    except (ValueError, OSError) as e:
        print(f"Error converting hex time '{hex_time}': {e}")
        return None

def extract_expected_throughput_in_order(file_path):
    """Assuming filename format: "1724872312_lowest_mode_expected_throughput"""
    filename = os.path.basename(file_path)
    try:
        return filename.split('_')[0]
    except (IndexError, ValueError) as e:
        print(f"Error extracting order from filename '{filename}': {e}")
        return None

def extract_mode_from_filename(filename):
    # Assuming filename format: "1_ap_1-Lowest_Power_orca_trace.csv"
    base_filename = os.path.basename(filename)
    if '_orca_trace.csv' in base_filename:
        base_filename = base_filename.replace('_orca_trace.csv', '')
    parts = base_filename.split('_', 2)
    return parts[-1].split('-',1)[1]

def extract_experiment_order(file_path):
    """Assuming filename format: "1_ap_1-Lowest_Power_orca_trace.csv"""
    filename = os.path.basename(file_path)
    try:
        parts = filename.split('-')[0]
        order = int(parts.split('_')[2])
        return order
    except (IndexError, ValueError) as e:
        print(f"Error extracting order from filename '{filename}': {e}")
        return None

def categorize_files(directory):
    """Processing the different csv files obtained post experiment"""
    measured_throughput_file = None
    expected_throughput_files = []
    response_files = []

    try:
        if os.path.isdir(directory):
            print(f"Listing and processing files in directory: {directory}\n")
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)

                if os.path.isfile(file_path) and filename.endswith('.csv'):
                    if filename == "ap_orca_header.csv":
                        continue
                    if "expected_throughput" in filename.lower():
                        expected_throughput_files.append(file_path)
                    elif "throughput" in filename.lower():
                        measured_throughput_file = file_path
                    else:
                        response_files.append(file_path)

        else:
            print(f"The path '{directory}' is not a directory.")

    except Exception as e:
        print(f"An error occurred: {e}")

    return expected_throughput_files,measured_throughput_file, response_files

def read_csv_to_dict(file_path, delimiter):
    filtered_data = []
    first_time = None
    try:
        with open(file_path, "r") as file:
            csv_reader = csv.reader(file, delimiter=delimiter)
            for row in csv_reader:
                if len(row) < 11:
                    continue

                if row[2] == 'txs' and len(row[1]) == 16:
                    try:
                        actual_time = hex_to_time(row[1])

                        if first_time is None:
                            first_time = actual_time

                        relative_time = (actual_time - first_time).total_seconds()

                        for i in range(len(row) - 1, 6, -1):
                            split_values = row[i].split(',')
                            if split_values[-1].isdigit():
                                rate = split_values[0]
                                power = int(split_values[-1], 16)
                                filtered_data.append({
                                    'time': relative_time,
                                    'rate': rate,
                                    'power': power
                                })
                                break
                    except ValueError as e:
                        print(f"ValueError processing row {row}: {e}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return filtered_data

def process_expected_throughput_files(expected_throughput_files):
    """This function was used to process expected throughput file computed within the power controller.
    The obtained data was averaged per minute. However we have removed the usage of this function for now"""
    expected_throughput_files.sort(key=lambda x: extract_expected_throughput_in_order(x))
    combined_expected_throughput_data = []
    cumulative_time = 0
    current_time_offset = 0

    for file_path in expected_throughput_files:
        print(f"Processing expected throughput file: {file_path}")
        throughput_data = pd.read_csv(file_path, sep='\s+', header=None, skiprows=1, names=['time', 'throughput'])

        if throughput_data.empty:
            continue
        throughput_data['time'] = throughput_data['time'] / 1000.0
        throughput_data['time'] += current_time_offset

        start_time = cumulative_time
        end_time= cumulative_time + throughput_data['time'].iloc[-1]
        cumulative_time = end_time

        current_time_offset = throughput_data['time'].iloc[-1] + 1

        combined_expected_throughput_data.append(throughput_data)

    combined_df = pd.concat(combined_expected_throughput_data, ignore_index=True)
    combined_df['rounded_time'] = combined_df['time'].round()

    averaged_df = combined_df.groupby('rounded_time').agg({'throughput': 'mean'}).reset_index()
    averaged_df.rename(columns={'rounded_time': 'time'}, inplace=True)

    return averaged_df

def process_measured_throughput_file(file_path):

    print(f"Processing throughput file: {file_path}")
    measured_throughput_data = pd.read_csv(file_path, sep='\s+', header=None, skiprows=1, names=['time', 'throughput'])
    measured_throughput_data['file_name'] = os.path.basename(file_path)
    return measured_throughput_data

def process_trace_response_files(trace_response_files):
    """This function processes the response csv files of different parts from the experiment.
    The timing of different parts(csv files) is combined so that they appear to be linear
    """
    trace_response_files.sort(key=lambda x: extract_experiment_order(x))
    trace_data = []
    current_time_offset = 0
    cumulative_time = 0

    for file_path in trace_response_files:
        print(f"Processing response file: {file_path}")
        mode = extract_mode_from_filename(file_path)
        data = read_csv_to_dict(file_path, delimiter=';')
        if not data:
            continue
        start_time = cumulative_time
        end_time = cumulative_time + data[-1]['time']
        cumulative_time = end_time
        for entry in data:
            entry['time'] += current_time_offset
            entry['mode'] = mode
            trace_data.append(entry)

        current_time_offset = trace_data[-1]['time'] + 1
    return trace_data

def get_boxplot_properties():
    boxprops = dict(facecolor='none', edgecolor='black')
    medianprops = dict(color='black')
    whiskerprops = dict(color='black')
    capprops = dict(color='black')

    return boxprops, medianprops, whiskerprops, capprops

def bin_time(df, time_column='time', bin_size=10):
    min_time = df[time_column].min()
    max_time = df[time_column].max()

    bin_edges = np.arange(min_time, max_time + bin_size, bin_size)
    rounded_bin_edges = np.round(bin_edges / bin_size) * bin_size

    if rounded_bin_edges[-1] < max_time:
        rounded_bin_edges = np.append(rounded_bin_edges, np.ceil(max_time / bin_size) * bin_size)
    rounded_bin_edges = rounded_bin_edges[rounded_bin_edges <= max_time]

    df['binned_time'] = pd.cut(df[time_column], bins=rounded_bin_edges)
    return df

def get_bin_edges(min_time, max_time, bin_size):
    return np.arange(min_time, max_time + bin_size, bin_size)

def add_grid_lines_for_scatter_plot(ax, df):
    df_sorted = df.sort_values(by='time', ascending=True)
    prev_mode = None
    prev_time = None
    line_positions = []

    for idx, row in df_sorted.iterrows():
        current_mode = row['mode']
        current_time = row['time']

        if current_mode != prev_mode:
            if prev_mode is not None and prev_time is not None:
                line_positions.append(prev_time)
            prev_mode = current_mode
        prev_time = current_time

    first_time = df_sorted['time'].iloc[0]
    last_time = df_sorted['time'].iloc[-1]

    line_positions.insert(0, first_time)
    line_positions.append(last_time)
    for pos in sorted(set(line_positions)):
        ax.axvline(x=pos, color='black', linestyle='--', linewidth=1)

    return sorted(set(line_positions))

def scale_line_positions(line_positions, rate_x_range, power_x_range):
    scaling_factor = power_x_range / rate_x_range
    print('ranges',rate_x_range, power_x_range )
    return [pos * scaling_factor for pos in line_positions]

def plot_rate_vs_time(df, ax,):
    df['rate_int'] = df['rate'].apply(hex_to_int)
    df_sorted = df.sort_values(by='rate_int', ascending=True)
    num_modes = len(df_sorted['mode'].unique())
    color_palette = sns.color_palette('tab10', num_modes)
    sns.scatterplot(data=df_sorted, x='time', y='rate', hue='mode', palette=color_palette, alpha=0.7, ax=ax)
    line_positions = add_grid_lines_for_scatter_plot(ax, df)
    rounded_positions = [round(pos) for pos in line_positions]

    ax.set_xticks(rounded_positions)
    print("Rate Plot Line Positions:", rounded_positions)
    ax.set_xticklabels([f'{int(pos)}' for pos in rounded_positions])
    ax.legend(loc='lower left')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate')
    ax.set_title('Rate vs Time')
    ax.set_xlim(df['time'].min(), df['time'].max())
    ax.invert_yaxis()

    return rounded_positions, ax.get_xlim()

def plot_power_vs_time(df, ax, bin_edges, rate_line_positions, rate_x_limit):
    df = bin_time(df, time_column='time', bin_size=10)
    boxprops, medianprops, whiskerprops, capprops = get_boxplot_properties()
    bins = df['binned_time'].cat.categories
    for i in range(len(bin_edges) - 1):
        bin_data = df[(df['time'] >= bin_edges[i]) & (df['time'] < bin_edges[i + 1])]

        if not bin_data.empty:
            sns.boxplot(
                data=bin_data,
                x=[i] * len(bin_data),
                y='power',
                ax=ax,
                boxprops=boxprops,
                medianprops=medianprops,
                whiskerprops=whiskerprops,
                capprops=capprops
            )
            mean_value = bin_data['power'].mean()
            ax.text(i, mean_value, f'{mean_value:.2f}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='red',
                    fontsize=8)

        if rate_line_positions is not None:
            scaled_positions = scale_line_positions(rate_line_positions, rate_x_limit[1], len(bins))
            for pos in scaled_positions:
                ax.axvline(x=pos, color='black', linestyle='--', linewidth=1)

    print("Power Plot Line Positions:", scaled_positions, bins)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.set_xticklabels([])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power Index')
    ax.set_title('Power vs Time (Box Plot)')
    ax.set_xlim(0, len(bins))
    return scaled_positions, bins

def plot_throughput_vs_time(df, ax, bin_edges, line_positions, power_bins):
    df = bin_time(df, time_column='time', bin_size=10)
    boxprops, medianprops, whiskerprops, capprops = get_boxplot_properties()
    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        bin_data = df[(df['time'] >= bin_start) & (df['time'] < bin_end)]
        if not bin_data.empty:
            sns.boxplot(
                data=bin_data,
                x=[i] * len(bin_data),
                y='throughput',
                ax=ax,
                boxprops=boxprops,
                medianprops=medianprops,
                whiskerprops=whiskerprops,
                capprops=capprops
            )
            mean_value = bin_data['throughput'].mean()
            ax.text(i, mean_value, f'{mean_value:.2f}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='red',
                    fontsize=8)
    for pos in line_positions:
        ax.axvline(x=pos, color='black', linestyle='--', linewidth=1)
    print("Throughput Plot Line Positions:", line_positions,power_bins)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.set_xticklabels([])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Throughput')
    ax.set_title('Throughput vs Time (Box Plot)')
    ax.set_xlim(0, len(power_bins))
