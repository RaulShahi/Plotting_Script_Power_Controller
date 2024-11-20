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
        return filename.split("_")[0]
    except (IndexError, ValueError) as e:
        print(f"Error extracting order from filename '{filename}': {e}")
        return None


def extract_mode_from_filename(filename):
    # Assuming filename format: "1_ap_1-Lowest_Power_orca_trace.csv"
    base_filename = os.path.basename(filename)
    if "_orca_trace.csv" in base_filename:
        base_filename = base_filename.replace("_orca_trace.csv", "")
    parts = base_filename.split("_", 2)
    return parts[-1].split("-", 1)[1]


def extract_experiment_order(file_path):
    """Assuming filename format: "1_ap_1-Lowest_Power_orca_trace.csv"""
    filename = os.path.basename(file_path)
    try:
        parts = filename.split("-")[0]
        order = int(parts.split("_")[2])
        return order
    except (IndexError, ValueError) as e:
        print(f"Error extracting order from filename '{filename}': {e}")
        return None


def categorize_files(directory):
    """Processing the different csv files obtained post-experiment."""
    measured_throughput_files = {}
    response_files = {}
    expected_throughput_files = {}

    try:
        for iteration_folder in os.listdir(directory):
            iteration_path = os.path.join(directory, iteration_folder)
            if os.path.isdir(iteration_path):
                print(
                    f"Listing and processing files in iteration folder: {iteration_path}"
                )
                measured_throughput_files[iteration_folder] = []
                response_files[iteration_folder] = []
                expected_throughput_files[iteration_folder] = []

                for filename in os.listdir(iteration_path):
                    file_path = os.path.join(iteration_path, filename)

                    if os.path.isfile(file_path) and filename.endswith(".csv"):
                        if filename == "ap_orca_header.csv":
                            continue
                        if "expected_throughput" in filename.lower():
                            expected_throughput_files[iteration_folder].append(
                                file_path
                            )
                        elif "throughput" in filename.lower():
                            measured_throughput_files[iteration_folder].append(
                                file_path
                            )
                        else:
                            response_files[iteration_folder].append(file_path)
        return measured_throughput_files, expected_throughput_files, response_files

    except Exception as e:
        print(f"An error occurred: {e}")


def read_csv_to_dict(file_path, delimiter):
    """This function reads the trace response csv files obtained after the experiment"""
    filtered_data = []
    first_time = None
    try:
        with open(file_path, "r") as file:
            csv_reader = csv.reader(file, delimiter=delimiter)
            for row in csv_reader:
                if len(row) < 11:
                    continue

                if row[2] == "txs" and len(row[1]) == 16:
                    try:
                        actual_time = hex_to_time(row[1])

                        if first_time is None:
                            first_time = actual_time

                        relative_time = (actual_time - first_time).total_seconds()

                        for i in range(len(row) - 1, 6, -1):
                            split_values = row[i].split(",")
                            if split_values[-1].isdigit():
                                rate = split_values[0]
                                power = int(split_values[-1], 16)
                                filtered_data.append(
                                    {
                                        "time": relative_time,
                                        "rate": rate,
                                        "power": power,
                                    }
                                )
                                break
                    except ValueError as e:
                        print(f"ValueError processing row {row}: {e}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return filtered_data


def process_expected_throughput_files(expected_throughput_files_dict):
    """This function was used to process expected throughput file computed within the power controller.
    The obtained data was averaged per minute."""

    combined_expected_throughput_data = {}
    for iteration, expected_throughput_files in expected_throughput_files_dict.items():
        expected_throughput_files.sort(
            key=lambda x: extract_expected_throughput_in_order(x)
        )
        cumulative_time = 0
        current_time_offset = 0
        iteration_data = []

        for file_path in expected_throughput_files:
            print(
                f"Processing expected throughput file: {file_path} for iteration: {iteration}"
            )
            throughput_data = pd.read_csv(
                file_path,
                sep="\s+",
                header=None,
                skiprows=1,
                names=["time", "throughput", "power_mode"],
            )
            if throughput_data.empty:
                continue

            throughput_data["time"] = throughput_data["time"] / 1000.0
            throughput_data["time"] += current_time_offset
            start_time = cumulative_time
            end_time = cumulative_time + throughput_data["time"].iloc[-1]
            cumulative_time = end_time
            current_time_offset = throughput_data["time"].iloc[-1] + 1
            iteration_data.append(throughput_data)
        if iteration_data:
            iteration_wise_df = pd.concat(iteration_data, ignore_index=True)

        combined_expected_throughput_data[iteration] = iteration_wise_df
    base_data = combined_expected_throughput_data.get("1", [])
    for iteration in combined_expected_throughput_data:
        if iteration == "1":
            continue
        base_data = pd.concat(
            [base_data, combined_expected_throughput_data[iteration]], ignore_index=True
        )
        base_data.sort_values(by="time", inplace=True)
        base_data.reset_index(drop=True, inplace=True)
    return base_data


def process_measured_throughput_files(iteration_files_dict):
    """Processes the measured throughput files for multiple iterations, averaging data based on the first iteration's timestamps."""

    combined_data = {}
    for iteration, throughput_files in iteration_files_dict.items():
        iteration_data = []

        for file_path in throughput_files:
            print(f"Processing throughput file: {file_path} for iteration: {iteration}")
            data = pd.read_csv(
                file_path,
                sep="\s+",
                header=None,
                skiprows=1,
                names=["time", "throughput"],
            )
            iteration_data.append(data)

        combined_data[iteration] = pd.concat(iteration_data, ignore_index=True)

    first_iteration_data = combined_data.get("1", pd.DataFrame())
    averaged_data = []

    for row_index in range(len(first_iteration_data)):
        base_entry = first_iteration_data.iloc[row_index]
        averaged_entry = {"time": base_entry["time"]}
        values_to_average = [base_entry["throughput"]]

        for iteration, data in combined_data.items():
            if iteration == "1":
                continue
            if row_index < len(data):
                values_to_average.append(data.iloc[row_index]["throughput"])

        if values_to_average:
            averaged_entry["throughput"] = sum(values_to_average) / len(
                values_to_average
            )

        averaged_data.append(averaged_entry)
        averaged_df = pd.DataFrame(averaged_data)

    return averaged_df


def map_modes_to_throughput(trace_df, throughput_df):
    trace_df["second"] = trace_df["time"].apply(lambda x: int(round(x)))
    mode_per_second = trace_df.groupby("second")["mode"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )
    throughput_df["second"] = throughput_df["time"].astype(int)
    throughput_df["mode"] = throughput_df["second"].map(mode_per_second)

    return throughput_df


def process_trace_response_files(iteration_files_dict):
    """Processes the response CSV files for multiple iterations, averaging data based on the first iteration's row indices."""

    combined_data = {}
    for iteration, trace_response_files in iteration_files_dict.items():
        trace_response_files.sort(key=lambda x: extract_experiment_order(x))

        iteration_data = []
        current_time_offset = 0

        for file_path in trace_response_files:
            print(f"Processing response file: {file_path} for iteration: {iteration}")
            mode = extract_mode_from_filename(file_path)
            data = read_csv_to_dict(file_path, delimiter=";")
            if not data:
                continue

            for entry in data:
                entry["time"] += current_time_offset
                entry["mode"] = mode
                iteration_data.append(entry)

            current_time_offset = iteration_data[-1]["time"] + 1

        combined_data[iteration] = iteration_data

    first_iteration_data = combined_data.get("1", [])
    averaged_data = []

    for row_index in range(len(first_iteration_data)):
        base_entry = first_iteration_data[row_index]
        averaged_entry = {"time": base_entry["time"], "mode": base_entry["mode"]}
        values_to_average = [base_entry]

        for iteration, data in combined_data.items():
            if iteration == "1":
                continue
            if row_index < len(data):
                current_entry = data[row_index]
                if current_entry["mode"] == base_entry["mode"]:
                    values_to_average.append(current_entry)

        hex_rates = [
            int(value["rate"], 16) for value in values_to_average if "rate" in value
        ]
        if hex_rates:
            average_rate = sum(hex_rates) // len(hex_rates)
            averaged_entry["rate"] = hex(average_rate)[2:]
        power_values = [
            value["power"] for value in values_to_average if "power" in value
        ]
        if power_values:
            averaged_entry["power"] = sum(power_values) // len(power_values)

        averaged_data.append(averaged_entry)

    return averaged_data


def get_boxplot_properties():
    boxprops = dict(facecolor="none", edgecolor="black")
    medianprops = dict(color="black")
    whiskerprops = dict(color="black")
    capprops = dict(color="black")

    return boxprops, medianprops, whiskerprops, capprops


def bin_time(df, time_column="time", bin_size=10):
    min_time = df[time_column].min()
    max_time = df[time_column].max()

    bin_edges = np.arange(min_time, max_time + bin_size, bin_size)

    rounded_bin_edges = np.round(bin_edges / bin_size) * bin_size

    if rounded_bin_edges[-1] < max_time:
        rounded_bin_edges = np.append(
            rounded_bin_edges, np.ceil(max_time / bin_size) * bin_size
        )
    rounded_bin_edges = rounded_bin_edges[rounded_bin_edges <= max_time]

    df["binned_time"] = pd.cut(
        df[time_column], bins=rounded_bin_edges, duplicates="drop"
    )
    return df


def get_bin_edges(min_time, max_time, bin_size):
    return np.arange(min_time, max_time + bin_size, bin_size)


def add_grid_lines_to_separate_modes(ax, df):
    df_sorted = df.sort_values(by="time", ascending=True)
    prev_mode = None
    prev_time = None
    line_positions = []
    modes_between_lines = []
    for idx, row in df_sorted.iterrows():
        current_mode = row["mode"]
        current_time = row["time"]

        if current_mode != prev_mode:
            if prev_mode is not None and prev_time is not None:
                line_positions.append(prev_time)
                modes_between_lines.append(prev_mode)
            prev_mode = current_mode
        prev_time = current_time

    first_time = df_sorted["time"].iloc[0]
    last_time = df_sorted["time"].iloc[-1]

    line_positions.insert(0, first_time)
    line_positions.append(last_time)
    modes_between_lines.append(df_sorted["mode"].iloc[-1])

    rounded_positions = [round(pos) for pos in line_positions]
    for pos in sorted(set(line_positions)):
        ax.axvline(x=pos, color="black", linestyle="--", linewidth=1)

    return sorted(set(rounded_positions)), modes_between_lines


def scale_line_positions(line_positions, rate_x_range, power_x_range):
    scaling_factor = power_x_range / rate_x_range
    return [pos * scaling_factor for pos in line_positions]


def plot_rate_vs_time(kwargs):
    df = kwargs["df"]
    ax = kwargs["ax"]
    rounded_positions = kwargs["rounded_position"]
    modes_between_lines = kwargs["modes_between_lines"]

    df["rate_int"] = df["rate"].apply(hex_to_int)
    df_sorted = df.sort_values(by="rate_int", ascending=True)
    num_modes = len(df_sorted["mode"].unique())
    color_palette = sns.color_palette("tab10", num_modes)
    sns.scatterplot(
        data=df_sorted,
        x="time",
        y="rate",
        hue="mode",
        palette=color_palette,
        alpha=0.7,
        ax=ax,
    )
    ax.set_xticks(rounded_positions)
    ax.set_xticklabels([f"{int(pos)}" for pos in rounded_positions])
    ax.legend(loc="lower left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate")
    ax.set_title("Rate vs Time")
    ax.set_xlim(df["time"].min(), df["time"].max())
    ax.invert_yaxis()

    return ax.get_xlim()


def calculate_mean_between_different_parts(mean_values, scaled_positions):
    interval_means = []
    covered_indices = [False] * len(mean_values)

    for i in range(len(scaled_positions) - 1):
        start_idx = round(scaled_positions[i])
        end_idx = round(scaled_positions[i + 1])
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(mean_values):
            end_idx = len(mean_values)

        bins_in_interval = [
            mean_values[j] for j in range(start_idx, end_idx) if not covered_indices[j]
        ]
        for j in range(start_idx, end_idx):
            covered_indices[j] = True

        if bins_in_interval:
            interval_mean = sum(bins_in_interval) / len(bins_in_interval)
            interval_means.append(interval_mean)
    return interval_means


def plot_power_vs_time(kwargs):
    df = kwargs["df"]
    ax = kwargs["ax"]
    bin_edges = kwargs["bin_edges"]
    rate_line_positions = kwargs["rounded_positions"]
    modes_between_lines = kwargs["modes_between_lines"]
    rate_x_limit = kwargs["rate_x_limit"]
    bin_size = kwargs["bin_size"]

    df = bin_time(df, time_column="time", bin_size=bin_size)
    boxprops, medianprops, whiskerprops, capprops = get_boxplot_properties()
    bins = df["binned_time"].cat.categories
    mean_values = []

    print("bin_edges", bin_edges, len(bin_edges))
    print("power_bins", bins, len(bins))

    for i in range(len(bin_edges) - 1):
        bin_data = df[(df["time"] >= bin_edges[i]) & (df["time"] < bin_edges[i + 1])]

        if not bin_data.empty:
            sns.boxplot(
                data=bin_data,
                x=[i] * len(bin_data),
                y="power",
                ax=ax,
                boxprops=boxprops,
                medianprops=medianprops,
                whiskerprops=whiskerprops,
                capprops=capprops,
            )
            mean_value = bin_data["power"].mean()
            mean_values.append(mean_value)

    scaled_positions = scale_line_positions(
        rate_line_positions, rate_x_limit[1], len(bins)
    )
    print("scaled_positions", scaled_positions)
    interval_means = calculate_mean_between_different_parts(
        mean_values, scaled_positions
    )

    for i, pos in enumerate(scaled_positions[:-1]):
        next_pos = scaled_positions[i + 1]
        mode = modes_between_lines[i]
        ax.axvline(x=pos, color="black", linestyle="--", linewidth=1)

    ax.axvline(x=scaled_positions[-1], color="black", linestyle="--", linewidth=1)

    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.set_xticklabels([])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power Index")
    ax.set_title("Power vs Time (Box Plot)", fontsize=16)
    ax.set_xlim(left=0)
    ax.set_xlim(0, len(bins))
    return scaled_positions, bins


def plot_throughput_vs_time(kwargs):
    df = kwargs["df"]
    ax = kwargs["ax"]
    bin_edges = kwargs["bin_edges"]
    line_positions = kwargs["line_positions"]
    modes_between_lines = kwargs["modes_between_lines"]
    power_bins = kwargs["power_bins"]
    bin_size = kwargs["bin_size"]

    # df = bin_time(df, time_column='time', bin_size=10)
    df = bin_time(df, time_column="time", bin_size=bin_size)
    boxprops, medianprops, whiskerprops, capprops = get_boxplot_properties()
    mean_values = []
    print("bin_edges", bin_edges, len(bin_edges))
    print("power_bins", power_bins, len(power_bins))
    print("line positions", line_positions)
    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        bin_data = df[(df["time"] >= bin_start) & (df["time"] < bin_end)]
        if not bin_data.empty:
            sns.boxplot(
                data=bin_data,
                x=[i] * len(bin_data),
                y="throughput",
                ax=ax,
                boxprops=boxprops,
                medianprops=medianprops,
                whiskerprops=whiskerprops,
                capprops=capprops,
            )
            mean_value = bin_data["throughput"].mean()
            mean_values.append(mean_value)

    for i, pos in enumerate(line_positions[:-1]):
        next_pos = line_positions[i + 1]
        ax.axvline(x=pos, color="black", linestyle="--", linewidth=1)

        for i, pos in enumerate(line_positions[:-1]):
            next_pos = line_positions[i + 1]
            mode = modes_between_lines[i]
            ax.axvline(x=pos, color="black", linestyle="--", linewidth=1)
    ax.axvline(x=line_positions[-1], color="black", linestyle="--", linewidth=1)
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.set_xticklabels([])
    ax.set_xlim(left=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Throughput")
    ax.set_title("Throughput vs Time (Box Plot)", fontsize=16)
    ax.set_xlim(0, len(power_bins))


def plot_expected_tp_vs_max_tp(kwargs):
    df = kwargs["df"]
    ax = kwargs["ax"]

    sns.scatterplot(
        data=df,
        x="time",
        y="throughput",
        hue="power_mode",
        style="power_mode",
        ax=ax,
        palette={"max_power": "red", "optimal_power": "blue"},
        markers={"max_power": "o", "optimal_power": "X"},
        s=100,
        alpha=0.7,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Expected Throughput vs Max Throughput")
