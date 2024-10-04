import os
import math
import toml
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
import scienceplots
import csv
import json
import argparse

# base_folder = "Path to Experimentor Data [ex: Desktop/ath9k-minstrel-ht-comp_20240102-114700]"
overhead_mcs = 108
overhead_legacy = 60

# plt.style.use("science")
# plt.rcParams["text.latex.preamble"] = r"\usepackage{sfmath} \boldmath"


rc_color = {
    "fixed-rate": "white",
    "py-minstrel-ht": "blue",
    "kernel-minstrel-ht": "red",
    "minstrel-ht-blues": "green",
}


def set_label(exp_seq, secs):
    for rc_alg, dur in exp_seq:
        if secs <= dur:
            return rc_alg

    return rc_alg


def get_part_color(rc_alg):
    if "blues" in rc_alg:
        return rc_color["minstrel-ht-blues"]
    elif "py-minstrel-ht" in rc_alg:
        return rc_color["py-minstrel-ht"]
    elif "kernel-minstrel-ht" in rc_alg:
        return rc_color["kernel-minstrel-ht"]
    elif "Fixed" in rc_alg:
        return rc_color["fixed-rate"]
    else:
        return "yellow"


def draw_rc_lines(ax, exp_seq, text=True, bins=1, x_offset=0):
    prev_dur = 0
    fontsize = 15
    y = max(ax.get_ylim())

    for rc_alg, dur in exp_seq:
        dur /= bins
        ax.axvline(dur + x_offset, color="gray", ls="dashed")
        if text:
            if rc_alg == "Fixed":
                x = prev_dur - 3
            else:
                x = prev_dur + ((dur - prev_dur) / 3.2)

            ax.text(x, y, r"\textbf{{{}}}".format(rc_alg), fontsize=fontsize)
        ax.axvspan(prev_dur, dur + x_offset, alpha=0.05, color=get_part_color(rc_alg))
        prev_dur = dur

    # ax.set_xticks(ax.get_xticks(), [fr'\textbf{{{int(v*bins)}}}' for v in ax.get_xticks()], fontsize=25)
    # ax.set_yticks(ax.get_yticks(), [fr'\textbf{{{int(v)}}}' if v >= 0 else r'' for v in ax.get_yticks()], fontsize=25)

def extract_full_rate_info(rates_info, rate_idx):
    if not rate_idx:
        raise ValueError(f"Invalid rate index: {rate_idx}")

    if len(rate_idx) == 1:
        rate_idx = "0" + rate_idx

    grp_idx = rate_idx[:-1]
    local_offset = int(rate_idx[-1], 16)
    try:
        MOD = [
            "BPSK,1/2",
            "QPSK,1/2",
            "QPSK,3/4",
            "16-QAM,1/2",
            "16-QAM,3/4",
            "64-QAM,2/3",
            "64-QAM,3/4",
            "64-QAM,5/6",
            "256-QAM,3/4",
            "256-QAM, 5/6",
        ]
        match rates_info[grp_idx]["bandwidth"]:
            case "0":
                BW = 20
            case "1":
                BW = 40
            case "2":
                BW = 80
        NSS = rates_info[grp_idx]["nss"]
        TYPE = rates_info[grp_idx]["type"]
        GI = int(rates_info[grp_idx]["guard_interval"])
        GI = "SG" if GI else "LG"

        return f"{BW} MHz. {NSS} Nss. {GI} - {MOD[local_offset]}"
    except KeyError:
        raise ValueError(f"Invalid rate index: {rate_idx}")


def compute_rate_weight(rates_info, rate_idx):
    if not rate_idx:
        raise ValueError(f"Invalid rate index: {rate_idx}")

    if len(rate_idx) == 1:
        rate_idx = "0" + rate_idx

    grp_idx = rate_idx[:-1]

    OFFSET = int(rate_idx[-1], 16) / 100
    try:
        match rates_info[grp_idx]["bandwidth"]:
            case "0":
                BW = 20
            case "1":
                BW = 40
            case "2":
                BW = 80
        NSS = int(rates_info[grp_idx]["nss"])
        GI = int(rates_info[grp_idx]["guard_interval"]) / 10
        weight = BW + NSS + OFFSET + GI

        return weight
    except KeyError:
        raise ValueError(f"Invalid rate index: {rate_idx}")


def plot_rate_choice(df, exp_seq, ax, full_rate_info=False):
    df = df[df.probe == 0]

    if full_rate_info:
        df = df.sort_values(by="succ_rate_weight", ascending=False)

    sns.scatterplot(
        data=df,
        x="secs",
        y="succ_rate_info",
        ax=ax,
        color="#e34a33",
        marker=".",
        linewidth=0,
        s=50,
        alpha=0.1,
        legend=True,
    )
    ax.minorticks_off()
    ax.grid()

    xticks = [dur for rc, dur in exp_seq]

    ax.set_xlim(0, exp_seq[-1][-1])
    ax.set(xticks=xticks)

    ax.set_xlabel(r"\bf{Time [seconds]}", fontsize=15)
    ax.set_ylabel(r"\bf{Rate Index}", fontsize=15)
    draw_rc_lines(ax, exp_seq)


def plot_tp_airtime(df, exp_seq, ax):
    dfg = df.copy()
    dfg = df[df.probe == 0]

    sns.scatterplot(
        data=dfg,
        x="secs",
        y="airtime_tp",
        ax=ax,
        color="#e34a33",
        marker=".",
        linewidth=0,
        s=70,
        alpha=1,
        legend=True,
    )

    xticks = [dur for rc, dur in exp_seq]

    ax.set_xlim(0, exp_seq[-1][-1])
    ax.set_ylim(bottom=0)
    ax.set(xticks=xticks)
    ax.set_xlabel(r"\bf{Time [seconds]}", fontsize=15)
    ax.set_ylabel(r"\bf{1/airtime [$ms^{-1}$]}", fontsize=15)
    draw_rc_lines(ax, exp_seq)

def plot_wasted_airtime(df, exp_seq, ax, size=1):
    df["bin"] = df["secs"] / size
    df["bin"] = df.bin.apply(lambda x: math.ceil(x))

    df = df[["bin", "Successful Airtime", "Failed Aggregation", "Wasted Airtime"]]
    df = df.groupby("bin").sum().reset_index()
    df = df.set_index("bin")

    df["Successful Airtime"] = (df["Successful Airtime"] / 10) / size
    df["Wasted Airtime"] = (df["Wasted Airtime"] / 10) / size
    df["Failed Aggregation"] = (df["Failed Aggregation"] / 10) / size
    df["Idle Time"] = 100 - (df["Wasted Airtime"] + df["Successful Airtime"])

    df.plot(kind="bar", stacked=True, color=["steelblue", "red", "yellow", "lightgrey"], ax=ax)

    xticks = [dur // size for rc, dur in exp_seq]

    ax.set_xlim(0.5, exp_seq[-1][-1] // size + 0.5)
    ax.set_ylim(0, 130)
    ax.set(yticks=[0, 20, 40, 60, 80, 100])
    ax.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,  # labels along the bottom edge are off
    )

    if not size:
        ax.set_xlabel(r"\bf{Time [seconds]}", fontsize=15)
    else:
        ax.set_xlabel(r"\textbf{{Time bin of {size} seconds}}".format(size=size), fontsize=15)
    ax.set_ylabel(r"\bf{Airtime Usage [\%]}", fontsize=15)

    ax.legend(
        [
            r"\textbf{Successsful}" "\n" r"\bf{Transmission}",
            r"\textbf{Frame}" "\n" r"\bf{Retries}",
            r"\textbf{Unacked Agg.}" "\n" r"\bf{Subframes}",
            r"\textbf{Idle}",
        ],
        loc="center left",
        bbox_to_anchor=(-0.09, 0.5),
        title=r"\textbf{Airtime}",
    )

    draw_rc_lines(ax, exp_seq, False, size, x_offset=0.5)


def plot_agg_len(df, exp_seq, ax, size=5):
    df["bin"] = df["secs"] / size
    df["bin"] = df.bin.apply(lambda x: math.floor(x) * size)
    last_bin = df.iloc[-1:]["bin"].values[0]

    df_agg = df[
        [
            "bin",
            "num_frames",
            "num_acked",
        ]
    ]
    df_agg = df_agg.groupby("bin").mean().reset_index()
    # df.loc[0, ['bin']] = [1]

    sns.barplot(
        data=df_agg,
        x="bin",
        y="num_acked",
        label=(r"\bf{No. of block-acked}" "\n" r"\bf{frames}"),
        color="black",
        ax=ax,
        width=0.3,
    )
    sns.barplot(
        data=df_agg,
        x="bin",
        y="num_frames",
        label=(r"\bf{No. of aggregated}" "\n" r"\bf{tx-frames}"),
        color="green",
        ax=ax,
    )

    for bar in ax.containers[0]:
        bar.set_alpha(1)
    for bar in ax.containers[1]:
        bar.set_alpha(0.5)

    prev_end_time = 0
    for rc, end_time in exp_seq:
        if not ("MMS" in rc or "Fixed" in rc or rc == "manual_mrr_setter"):
            rc_df = df_agg[(df_agg["bin"] >= prev_end_time) & (df_agg["bin"] < end_time)]
            agg_mean = rc_df["num_acked"].mean()
            agg_median = rc_df["num_acked"].median()

            ax.plot(
                (math.ceil(prev_end_time / size), math.ceil(end_time / size)),
                (agg_mean, agg_mean),
                linestyle="dashed",
                color="red",
                label=(r"\bf{Mean (block-acked}" "\n" r"\bf{frames)}"),
            )
            ax.plot(
                (math.ceil(prev_end_time / size), math.ceil(end_time / size)),
                (agg_median, agg_median),
                linestyle="dashed",
                color="blue",
                label=(r"\bf{Median (block-acked}" "\n" r"\bf{frames)}"),
            )

            fail_mean = (rc_df["num_frames"] - rc_df["num_acked"]).mean()

            ax.text(
                (prev_end_time + ((end_time - prev_end_time) / 2.5)) / size,
                max(ax.get_ylim()),
                r"\bf{{Mean Agg.: {fail}}}".format(fail=round(agg_mean, 2)),
                fontweight="bold",
                fontsize=10,
                color="black",
            )

        prev_end_time = end_time

    xticks = [dur // size for rc, dur in exp_seq]
    ax.set_xlim(xmin=0, xmax=exp_seq[-1][-1] // size)
    ax.set_ylim(ymin=0, ymax=max(ax.get_ylim()) + 1)
    ax.set(xticks=xticks)

    ax.set_xlabel(r"\textbf{{Time bin of {size} seconds}}".format(size=size), fontsize=15)
    ax.set_ylabel(r"\textbf{Aggregation Length}", fontsize=15)

    # Remove dupliate labels from the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(reversed(labels), reversed(handles)))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        fontsize=10,
        loc="center left",
        bbox_to_anchor=(-0.09, 0.5),
    )

    draw_rc_lines(ax, exp_seq, False, size)


def show_mean_median(df, row, col, exp_seq, ax, size):
    prev_end_time = 0
    for rc, end_time in exp_seq:
        if not ("MMS" in rc or "Fixed" in rc or rc == "manual_mrr_setter"):
            rc_df = df[(df[row] > prev_end_time) & (df[row] <= end_time)]
            mean = rc_df[col].mean()
            median = rc_df[col].median()

            ax.plot(
                (math.ceil(prev_end_time / size), math.ceil(end_time / size)),
                (mean, mean),
                linestyle="dashed",
                color="red",
            )
            ax.plot(
                (math.ceil(prev_end_time / size), math.ceil(end_time / size)),
                (median, median),
                linestyle="dashed",
                color="blue",
            )

            if col == "throughput":
                y_mean = 7
                y_median = 2
            else:
                max_yval = max(ax.get_ylim()) + 3.7
                y_mean = max_yval - 0.03 * max_yval
                y_median = max_yval - 0.08 * max_yval

            ax.text(
                (prev_end_time + ((end_time - prev_end_time) / 2.3)) / size,
                y_mean,
                r"\textbf{{Mean: {val}}}".format(val=round(mean, 2)),
                fontsize=15,
                color="red",
            )
            ax.text(
                (prev_end_time + ((end_time - prev_end_time) / 2.3)) / size,
                y_median,
                r"\textbf{{Median: {val}}}".format(val=round(median, 2)),
                fontsize=15,
                color="blue",
            )

        prev_end_time = end_time


def plot_succ_power(df, exp_seq, ax, size=5):
    bins = []
    bin = []

    prev_time_bin = None

    for index, row in df.iterrows():
        bin.append(row["succ_power"])
        time_bin = math.ceil(row["secs"] / size)

        if prev_time_bin != None and time_bin > prev_time_bin:
            bins.append(bin.copy())
            bin.clear()

        prev_time_bin = time_bin

    df = pd.DataFrame(
        {
            "Time [seconds]": [i * size for i in range(0, len(bins))],
            "Power Index": bins,
        }
    )
    df = df.explode("Power Index")

    sns.boxplot(
        data=df,
        x="Time [seconds]",
        y="Power Index",
        ax=ax,
        color="darkorange",
        width=0.5,
    )
    ax.grid()

    show_mean_median(df, "Time [seconds]", "Power Index", exp_seq, ax, size)

    xticks = [dur // size for rc, dur in exp_seq]
    ax.set_xlim(0, math.ceil(exp_seq[-1][-1] // size))
    ax.set(xticks=xticks)
    ax.set_xlabel(r"\textbf{{Time bin of {size} seconds}}".format(size=size), fontsize=15)
    ax.set_ylabel(r"\bf{Power Index}", fontsize=15)
    draw_rc_lines(ax, exp_seq, False, size)

    ax.set_ylim(bottom=-1)



def plot_power_choice(df, exp_seq, ax):
    dfg = df.copy()
    df = df[df.probe == 0]
    dfg = dfg[dfg.probe != 0]

    legend_handles = []

    sns.scatterplot(
        data=df,
        x="secs",
        y="power0",
        ax=ax,
        color="#e34a33",
        marker="o",
        linewidth=0,
        s=40,
        alpha=0.4,
        legend=True,
    )

    sns.scatterplot(
        data=dfg,
        x="secs",
        y="power1",
        ax=ax,
        color="black",
        marker=".",
        linewidth=0,
        s=35,
        alpha=0.4,
        legend=True,
    )

    sns.scatterplot(
        data=df,
        x="secs",
        y="power1",
        ax=ax,
        marker="x",
        color="cornflowerblue",
        linewidth=2,
        s=25,
        alpha=0.4,
        legend=True,
    )

    legend_handles.append(
        mlines.Line2D(
            [],
            [],
            color="#e34a33",
            marker="o",
            linewidth=0.01,
            markersize=5,
            label=r"\textbf{MRR[0]}" "\n" r"\bf{Power}",
        )
    )
    legend_handles.append(
        mlines.Line2D(
            [],
            [],
            color="cornflowerblue",
            marker="x",
            linestyle="None",
            markersize=5,
            label=r"\textbf{MRR[1]}" "\n" r"\bf{Power}",
        )
    )
    legend_handles.append(
        mlines.Line2D(
            [],
            [],
            color="black",
            marker=".",
            linestyle="None",
            markersize=5,
            label=r"\textbf{Probe}" "\n" r"\bf{Power}",
        )
    )

    xticks = [dur for rc, dur in exp_seq]

    ax.set_xlim(0, exp_seq[-1][-1])
    ax.set(xticks=xticks)
    ax.set_xlabel(r"\bf{Time [seconds]}", fontsize=18)
    ax.set_ylabel(r"\bf{Power Index}", fontsize=18)
    # ax.set_title(base_folder.split("/")[-1], fontsize=20)
    ax.legend(handles=legend_handles, fontsize=11, loc="center left", bbox_to_anchor=(-0.09, 0.5))
    draw_rc_lines(ax, exp_seq, False)


def plot_line_tp(df_tp, exp_seq, ax):
    sns.lineplot(data=df_tp, x="time", y="throughput", color="tab:blue", ax=ax)

    xticks = [dur for rc, dur in exp_seq]

    ax.set_xlim(0, exp_seq[-1][-1])
    ax.set(xticks=xticks)
    ax.set_xlabel(r"\textbf{Time [seconds]}", fontsize=18)
    ax.set_ylabel(r"\textbf{Throughput [Mbits/second]}", fontsize=18)

    draw_rc_lines(ax, exp_seq, False)


def plot_line_latency(df_tp, exp_seq, ax):
    sns.lineplot(data=df_tp, x="time", y="latency", color="tab:orange", ax=ax)

    xticks = [dur for rc, dur in exp_seq]

    ax.set_xlim(0, exp_seq[-1][-1])
    ax.set(xticks=xticks)

    ax.set_xlabel(r"\textbf{Time [seconds]}", fontsize=15)
    ax.set_ylabel(r"\textbf{Ping (ms) ICMP}", fontsize=15)
    ax.set_yscale("log")

    draw_rc_lines(ax, exp_seq, False)


def plot_box_tp(df_tp, exp_seq, ax, size=5):
    bins = []
    bin = []

    prev_time_bin = None

    for index, row in df_tp.iterrows():
        bin.append(row["throughput"])
        time_bin = math.ceil(row["time"] / size)

        if prev_time_bin != None and time_bin > prev_time_bin:
            bins.append(bin.copy())
            bin.clear()

        prev_time_bin = time_bin

    df = pd.DataFrame(
        {
            "Time [seconds]": [i * size for i in range(0, len(bins))],
            "Throughput [MBits/second]": bins,
        }
    )
    df = df.explode("Throughput [MBits/second]")

    sns.boxplot(
        data=df,
        x="Time [seconds]",
        y="Throughput [MBits/second]",
        ax=ax,
        color="darkorange",
        width=0.5,
    )
    ax.grid()

    show_mean_median(df_tp, "time", "throughput", exp_seq, ax, size)

    xticks = [dur // size for rc, dur in exp_seq]

    ax.set_xlim(0, math.ceil(exp_seq[-1][-1] // size))
    ax.set_ylim(bottom=0)
    ax.set(xticks=xticks)
    ax.set_xlabel(r"\textbf{{Time bin of {size} seconds}}".format(size=size), fontsize=15)
    ax.set_ylabel(r"\textbf{Throughput [MBits/second]}", fontsize=15)

    draw_rc_lines(ax, exp_seq, False, size)


def read_csv_data(rates_info, exp_seq, txstatus, tp, flent):
    columns = [
        "timestamp",
        "num_frames",
        "num_acked",
        "probe",
        "rate0",
        "count0",
        "power0",
        "rate1",
        "count1",
        "power1",
        "rate2",
        "count2",
        "power2",
        "rate3",
        "count3",
        "power3",
        "airtime_tp",
        "Wasted Airtime",
        "succ_rate",
        "succ_power",
        "Successful Airtime",
        "Failed Aggregation",
    ]
    rows = []

    with open(txstatus, "r") as trace:
        trace_reader = csv.reader(trace, delimiter=";")

        for fields in trace_reader:
            type = fields[2]
            if type == "txs" and len(fields[1]) == 16:
                timestamp = int(fields[1], 16)
                if len(str(timestamp)) > 19:
                    print(fields)
                    continue
                num_frames = int(fields[4], 16)
                num_acked = int(fields[5], 16)
                probe = int(fields[6])
                txs_mrr_chain = [tuple(s.split(",")) for s in fields[7:]]

                parsed_mrr_chain = []
                n_rates = 0
                for mrr in txs_mrr_chain:
                    if mrr[0] != "":
                        rate = mrr[0]
                        count = int(mrr[1], 16)
                        power = int(mrr[2], 16) if mrr[2] != "" else np.nan
                        parsed_mrr_chain.append([rate, count, power])
                        n_rates += 1

                wasted_airtime = 0

                if n_rates > 1:
                    for mrr in parsed_mrr_chain[:-1]:
                        rate, count = mrr[0], mrr[1]
                        rate_airtime = get_rate_airtime(rates_info, rate)
                        overhead = (
                            overhead_legacy if is_legacy_group(rates_info, rate) else overhead_mcs
                        )

                        wasted_airtime += count * (
                            (overhead / (1000 * num_frames)) + (rate_airtime * num_frames)
                        )

                succ_rate, count, succ_power = (
                    parsed_mrr_chain[n_rates - 1][0],
                    parsed_mrr_chain[n_rates - 1][1],
                    parsed_mrr_chain[n_rates - 1][2],
                )
                overhead = (
                    overhead_legacy if is_legacy_group(rates_info, succ_rate) else overhead_mcs
                )
                rate_airtime = get_rate_airtime(rates_info, succ_rate)
                succ_rate_airtime = (overhead / (1000 * num_frames)) + (rate_airtime * num_frames)
                airtime_tp = 1 / succ_rate_airtime

                # Check if the successful rate has failed attempts
                if count > 1:
                    failed_att = count - 1
                    wasted_airtime += failed_att * (
                        (overhead / (1000 * num_frames)) + (rate_airtime * num_frames)
                    )

                failed_agg_len = num_frames - num_acked
                failed_agg_airtime = 0

                if failed_agg_len > 0:
                    failed_agg_airtime = rate_airtime * failed_agg_len

                if succ_rate != int("110", 16):
                    row = [timestamp, num_frames, num_acked, probe]

                    empty_mrr = [np.nan, np.nan, np.nan] * (4 - n_rates)
                    parsed_mrr_chain_flattened = [item for mrr in parsed_mrr_chain for item in mrr]
                    parsed_mrr_chain_flattened.extend(empty_mrr)

                    row.extend(parsed_mrr_chain_flattened)
                    row.extend(
                        [
                            airtime_tp,
                            wasted_airtime,
                            succ_rate,
                            succ_power,
                            succ_rate_airtime,
                            failed_agg_airtime,
                        ]
                    )

                    rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ns")
    df["time"] -= df["time"].iloc[0]
    df["secs"] = df["time"].dt.total_seconds()

    df["rc"] = df.secs.apply(lambda x: set_label(exp_seq, x))
    df["succ_rate_info"] = df.succ_rate.apply(lambda x: extract_full_rate_info(rates_info, x))
    df["succ_rate_weight"] = df.succ_rate.apply(lambda x: compute_rate_weight(rates_info, x))

    if not flent:
        df_tp = pd.read_csv(
            tp, sep=" ", dtype={"time": float, "throughput": float}, engine="python"
        )
        df_tp["throughput"] *= 8
        df_tp["rc"] = df_tp.time.apply(lambda x: set_label(exp_seq, x))
    else:
        flent_file = open(flent, "r")
        flent_data = json.load(flent_file)

        df_cols = ["time", "throughput", "latency"]

        rows = []
        rows.append(flent_data["x_values"])
        try:
            rows.append(flent_data["results"]["TCP download"])
        except:
            rows.append(flent_data["results"]["TCP download sum"])
        rows.append(flent_data["results"]["Ping (ms) ICMP"])

        df_tp = pd.DataFrame(list(zip(*rows)), columns=df_cols)
        df_tp["rc"] = df_tp.time.apply(lambda x: set_label(exp_seq, x))

    return df, df_tp


def extract_rc_sequence(settings):
    settings_info = toml.load(settings)
    exp_seq = []
    time_elapsed = 0

    for part in settings_info["parts"]:
        name = part["name"]
        dur = part["duration"]

        end_time = int(dur) + time_elapsed
        time_elapsed = end_time

        exp_seq.append((name, end_time))

    return exp_seq


def parse_group_info(fields):
    fields = list(filter(None, fields))
    group_ind = fields[3]

    airtimes_hex = fields[9:]
    rate_offsets = [str(ii) for ii in range(len(airtimes_hex))]
    rate_inds = list(map(lambda jj: group_ind + jj, rate_offsets))
    airtimes_ns = [int(ii, 16) for ii in airtimes_hex]

    group_info = {
        "rate_inds": rate_inds,
        "airtimes_ns": airtimes_ns,
        "type": fields[5],
        "nss": fields[6],
        "bandwidth": fields[7],
        "guard_interval": fields[8],
    }

    return group_ind, group_info


def get_rate_airtime(rates_info, rate_idx):
    if not rate_idx:
        raise ValueError(f"Invalid rate index: {rate_idx}")

    if len(rate_idx) == 1:
        rate_idx = "0" + rate_idx

    grp_idx = rate_idx[:-1]
    local_offset = int(rate_idx[-1], 16)
    try:
        group_info = rates_info[grp_idx]
        return group_info["airtimes_ns"][local_offset] / pow(10, 6)
    except KeyError:
        raise ValueError(f"Invalid rate index: {rate_idx}")


def get_rate_type(rates_info, rate_idx):
    if not rate_idx:
        raise ValueError(f"Invalid rate index: {rate_idx}")

    if len(rate_idx) == 1:
        rate_idx = "0" + rate_idx

    grp_idx = rate_idx[:-1]
    try:
        group_info = rates_info[grp_idx]
        return group_info["type"]
    except KeyError:
        raise ValueError(f"Invalid rate index: {rate_idx}")


def is_legacy_group(rates_info, rate):
    """
    Checks if the provided rate index is from Legacy Group (CCK and OFDM).

    """
    if not rate:
        return False

    group_type = get_rate_type(rates_info, rate)
    return group_type == "cck" or group_type == "ofdm"


def load_rates_info():
    airtimes = {}
    with open("airtimes.csv") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            group_ind, group_info = parse_group_info(row)
            airtimes.update({group_ind: group_info})

    return airtimes


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('base_folder',type=str,help='Path to Experimentor Data')
    args=parser.parse_args()

    base_folder=args.base_folder

    settings = os.path.join(base_folder, "settings.toml")
    exp_seq = extract_rc_sequence(settings)

    rates_info = load_rates_info()
    iterations = [
        os.path.join(base_folder, iteration)
        for iteration in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, iteration))
    ]

    for iteration_dir in iterations:
        # Get the iteration number from the path
        iteration = os.path.basename(os.path.normpath(iteration_dir))

        # _apu-ath9k-ap_minstrel-rcd.csv
        iteration_files = []
        for f in os.listdir(iteration_dir):
            path = os.path.join(iteration_dir, f)
            if os.path.isfile(path):
                iteration_files.append(path)

        txstatus = [file for file in iteration_files if "minstrel-rcd.csv" in file][0]
        tp = [file for file in iteration_files if "throughput" in file]
        tp = tp[0] if tp else ""

        flent = [file for file in iteration_files if ".flent" in file]
        flent = flent[0] if flent else ""

        df, df_tp = read_csv_data(rates_info, exp_seq, txstatus, tp, flent)

        plt.figure(figsize=(30, 25))

        if not flent:
            ax1 = plt.subplot(6, 1, (1, 2))
            ax2 = plt.subplot(6, 1, 3)
            ax3 = plt.subplot(6, 1, 4)
            ax4 = plt.subplot(6, 1, 5)
            ax5 = plt.subplot(6, 1, 6)
        else:
            ax1 = plt.subplot(7, 1, (1, 2))
            ax2 = plt.subplot(7, 1, 3)
            ax3 = plt.subplot(7, 1, 4)
            ax4 = plt.subplot(7, 1, 5)
            ax5 = plt.subplot(7, 1, 6)
            ax6 = plt.subplot(7, 1, 7)

        # plot_tp_airtime(df, exp_seq, ax)

        plot_rate_choice(df, exp_seq, ax1, full_rate_info=True)
        # plot_wasted_airtime(df, exp_seq, ax2, size=10)
        # plot_power_choice(df, exp_seq, ax3)
        # plot_succ_power(df, exp_seq, ax3)
        # plot_agg_len(df, exp_seq, ax4)

        if not flent:
            plot_box_tp(df_tp, exp_seq, ax5)
        else:
            plot_line_tp(df_tp, exp_seq, ax5)
            plot_line_latency(df_tp, exp_seq, ax6)

        # plt.tight_layout()

        meta = "_flent" if flent else ""
        plt.savefig(
            os.path.join(base_folder, iteration, str(iteration) + meta + "_plot.jpeg"), dpi=300
        )
        