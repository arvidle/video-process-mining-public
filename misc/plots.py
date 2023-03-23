import pm4py
import numpy as np
import matplotlib.pyplot as plt
import datetime
from video_pm.activity_recognition import ActionDetectionResults
from functools import reduce
import pandas as pd

LOG_PATH = "./kmeans/clustered_log.xes"
SEGMENT_MINS = 1
ROLLING_WINDOW = 10
START_TIME = datetime.time(6, 0, 0)
START_TIMEDELTA = datetime.timedelta(hours=6)

plt.rcParams['figure.dpi'] = 300
# Load the data

days = ["20211113", "20211114", "20211115", "20211116", "20211117"]

res_days = []

for day in days:
    res = ActionDetectionResults.from_file(f"../data/action_detection/loaded/{day}.pkl")
    res_days.append(res)

# Combine action logs of the same segments for each day to get a log for each segment
days_combined = [reduce(lambda x, y: x.concat(y), res_days)]

log = pm4py.read_xes(LOG_PATH)
log_df = pm4py.convert_to_dataframe(log)

time_segments = [datetime.timedelta(minutes=m).total_seconds() for m in range(0, 12 * 60 + 1, SEGMENT_MINS)]
# Temporal distribution of the activities

# Temporal distribution of the clusters
clusters = log_df.groupby("case:cluster")
"""
start_times_per_cluster = dict()

for cluster_id, cluster in clusters:
    start_timestamps = list(cluster[cluster["concept:name"] == "start"]["Timestamp"])
    start_times = [(t - datetime.datetime.combine(t.date(), datetime.time(6, 0, 0))).total_seconds() for t in start_timestamps]
    start_times_per_cluster[cluster_id] = start_times

cluster_start_per_segment = {cluster_id: np.histogram(start_times, time_segments, density=False) for cluster_id, start_times in start_times_per_cluster.items()}
labels = [str((datetime.datetime.combine(datetime.datetime.min, START_TIME) +
               datetime.timedelta(seconds=segment)).time()) for segment in time_segments]

cluster_counts_list = [cluster_cnt[0] for cluster_cnt in cluster_start_per_segment.values()]

all_counts = reduce(lambda x, y: np.sum((x, y)), cluster_counts_list)

cluster_start_per_segment[-1] = (all_counts, cluster_start_per_segment[0][1])
"""
"""
for cluster_id, cluster_counts in cluster_start_per_segment.items():
    counts = cluster_counts[0]
    counts_series = pd.Series(counts)
    counts_series_mean = counts_series.rolling(window=ROLLING_WINDOW, min_periods=1).sum().rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    counts = counts_series_mean.to_numpy()

    coef = np.polyfit(np.arange(len(counts)), counts, 1)
    poly1d_fn = np.poly1d(coef)

    plt.clf()
    plt.plot(labels[:-1], counts)
    plt.plot(np.arange(len(counts)), poly1d_fn(np.arange(len(counts))), "--k")
    plt.ylabel(f"Cluster frequency")
    plt.xlabel(f"Time of day")
    plt.title(f"Cluster frequency binned by time for cluster {cluster_id}")
    plt.xticks(np.arange(0, len(counts)+1, len(counts)/24), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    #plt.savefig(f"cluster_{cluster_id}.png", dpi=200)
# Activity counts in clusters
"""

clusters = log_df.groupby("case:cluster")

for cluster_id, cluster in clusters:
    plt.clf()
    plt.cla()
    values = cluster["concept:name"].unique()
    keep = [name for name in values if name not in ["start", "end", "lying_start", "lying_end"]]
    ax = cluster[cluster["concept:name"].isin(keep)]["concept:name"].value_counts().sort_values().plot(kind="barh", figsize=(4, 2), color="lightgrey", edgecolor="black")
    fig = ax.get_figure()
    ax.set_ylabel(f"Activity class")
    ax.set_xlabel(f"Count of activity instances")
    plt.tight_layout(pad=0.2)
    fig.savefig(f"out/cluster_{cluster_id}.svg")
    #plt.show()

plt.clf()
plt.cla()
values = log_df["concept:name"].unique()
keep = [name for name in values if name not in ["start", "end", "lying_start", "lying_end"]]
ax = log_df[log_df["concept:name"].isin(keep)]["concept:name"].value_counts().sort_values().plot(kind="barh", figsize=(4, 2))
fig = ax.get_figure()
ax.set_ylabel(f"Activity class")
ax.set_xlabel(f"Count of activity instances")
plt.tight_layout()
fig.savefig(f"out/cluster_-1.svg")
#plt.show()

# Cluster sizes
plt.clf()
plt.cla()

ax = log_df["case:cluster"].value_counts().sort_values().plot(kind="barh")
fig = ax.get_figure()
fig.tight_layout()
ax.set_ylabel(f"Cluster ID")
ax.set_xlabel(f"Event count")
fig.savefig(f"cluster_event_counts.png", dpi=300)
#fig.show()

plt.clf()
plt.cla()
cluster_sizes = log_df[log_df["concept:name"] == "start"]["case:cluster"].value_counts().sort_values()
ax = cluster_sizes.plot(kind="barh")
fig = ax.get_figure()
ax.set_ylabel(f"Cluster ID")
ax.set_xlabel(f"Case count")
plt.tight_layout()
fig.savefig("cluster_case_counts.png", dpi=300)
#plt.show()

# Mean case duration per cluster (in seconds)
plt.clf()
plt.cla()
clusters = log_df.groupby("case:cluster")

cases = log_df.groupby("CaseID")

durations = log_df.groupby(["case:cluster", "case:concept:name"])["time:timestamp"].aggregate(lambda case: (case.iloc[-1] - case.iloc[0]).total_seconds())
mean_duration_per_cluster = durations.groupby(level=0).mean()
mean_duration_per_cluster["all"] = durations.mean()
mean_duration_per_cluster = mean_duration_per_cluster.sort_values()

ax = mean_duration_per_cluster.sort_values().plot(kind="barh")
fig = ax.get_figure()
ax.set_ylabel(f"Cluster ID")
ax.set_xlabel(f"Mean case duration in seconds")
fig.tight_layout()
fig.savefig(f"mean_case_duration_per_cluster.png", dpi=300)
#plt.show()

# Boxplot of durations per cluster
plt.cla()
plt.clf()

