import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import csv
import matplotlib.pyplot as plt
import seaborn as sns


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
    # Assuming filename format: "1724872312_lowest_mode_expected_throughput"
    filename = os.path.basename(file_path)
    try:
        return filename.split('_')[0]
    except (IndexError, ValueError) as e:
        print(f"Error extracting order from filename '{filename}': {e}")
        return None

def extract_experiment_order(file_path):
    # Assuming filename format: "1_ap_1-Lowest_Power_orca_trace.csv"
    filename = os.path.basename(file_path)
    try:
        parts = filename.split('-')[0]
        order = int(parts.split('_')[2])
        return order
    except (IndexError, ValueError) as e:
        print(f"Error extracting order from filename '{filename}': {e}")
        return None

def categorize_files(directory):
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
    trace_response_files.sort(key=lambda x: extract_experiment_order(x))

    trace_data = []
    current_time_offset = 0
    duration_labels = []
    cumulative_time = 0

    for file_path in trace_response_files:
        print(f"Processing response file: {file_path}")

        mode = os.path.basename(file_path).split('-')[1]

        data = read_csv_to_dict(file_path, delimiter=';')

        if not data:
            continue

        start_time = cumulative_time
        end_time = cumulative_time + data[-1]['time']
        cumulative_time = end_time

        duration_label = f"{mode} : {int(start_time)}s - {int(end_time)}s"
        duration_labels.append(duration_label)

        for entry in data:
            entry['time'] += current_time_offset
            entry['mode'] = mode
            trace_data.append(entry)

        current_time_offset = trace_data[-1]['time'] + 1
    return trace_data,duration_labels

def adjust_time_offsets(data, current_time_offset):
    for entry in data:
        entry['time'] += current_time_offset
    return data

def get_boxplot_properties():
    boxprops = dict(facecolor='none', edgecolor='black')
    medianprops = dict(color='black')
    whiskerprops = dict(color='black')
    capprops = dict(color='black')

    return boxprops, medianprops, whiskerprops, capprops

def plot_rate_vs_time(df, ax):
    df['rate_int'] = df['rate'].apply(hex_to_int)
    df_sorted = df.sort_values(by='rate_int', ascending=True)
    df_sorted = df_sorted.drop(columns='rate_int')

    num_modes = len(df_sorted['mode'].unique())
    color_palette = sns.color_palette('tab10', num_modes)

    sns.scatterplot(data=df_sorted, x='time', y='rate', hue='mode', palette=color_palette, alpha=0.7, ax=ax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate')
    ax.set_title('Rate vs Time')
    ax.set_xlim(left=0)
    ax.grid(True)

def plot_power_vs_time(df, ax):
    boxprops, medianprops, whiskerprops, capprops = get_boxplot_properties()
    df['binned_time'] = pd.cut(df['time'], bins=10)
    sns.boxplot(data=df, x='binned_time', y='power', ax=ax, boxprops=boxprops, medianprops=medianprops, whiskerprops =whiskerprops, capprops = capprops )
    bins = df['binned_time'].cat.categories
    xticks = range(len(bins))

    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{int(interval.left)} - {int(interval.right)}' for interval in df['binned_time'].cat.categories])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power Index')
    ax.set_title('Power vs Time (Box Plot)')
    ax.grid(True)


def plot_throughput_vs_time(measured_throughput_file, ax, ylabel):
    boxprops, medianprops, whiskerprops, capprops = get_boxplot_properties()

    measured_throughput_file['binned_time']= pd.cut(measured_throughput_file['time'], bins=10)
    sns.boxplot(data=measured_throughput_file, x='binned_time', y='throughput', ax=ax, boxprops=boxprops, medianprops=medianprops, whiskerprops =whiskerprops, capprops = capprops)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'{ylabel} Throughput')
    ax.set_title(f'{ylabel} Throughput vs Time')
    bins = pd.cut(measured_throughput_file['time'], bins=10).cat.categories
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f'{int(interval.left)} - {int(interval.right)}' for interval in bins])
    ax.grid(True)

