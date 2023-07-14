from video_pm.activity_recognition import ActionDetectionResults
from video_pm.activity_recognition.event_activity_abstraction import BaselineAbstractor
from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['figure.dpi'] = 300

days = ["20211113", "20211114", "20211115", "20211116", "20211117"]

res_days = []

for day in days:
    res = ActionDetectionResults.from_file(f"../data/action_detection/loaded/{day}.pkl")
    res.action_detection_results["date"] = pd.Timestamp(day)
    res_days.append(res)

# Combine action logs of the same segments for each day to get a log for each segment
days_combined = reduce(lambda x, y: x.concat(y), res_days)

ba = BaselineAbstractor(10)

processed_action_log = ba.process_action_log(days_combined)
top1_events = processed_action_log.activity

print(top1_events.value_counts(normalize=True, ascending=True))
ax = top1_events.value_counts(normalize=True, ascending=True).plot(kind="barh", figsize=(5, 2.5), logx=True, color="lightgrey", edgecolor="black")
ax.set_xlabel("Relative duration")
ax.set_ylabel("Activity")
#ax.xaxis.tick_top()
#ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.savefig("activity_counts.svg")

# Start timestamp of all selected video recordings: 05:59:55

# Proportion of full hours to bin by. E.g., 0.25 would be 15 minute wide bins and 1 would be full hours
cut_proportion = 1/6
start_secs = pd.Timedelta(hours=6).total_seconds()
end_secs = pd.Timedelta(hours=18).total_seconds()
n_hours = int((end_secs - start_secs) / 3600)
fps = 30
processed_action_log["delta_secs"] = processed_action_log.frame / fps

# In our analysis, all videos start at 05:59:55, so we set this as a constant here. Replace if different!
processed_action_log["video_start_secs"] = pd.Timedelta(hours=5, minutes=59, seconds=55).total_seconds()
processed_action_log["timestamp_relative_to_day_start"] = processed_action_log.video_start_secs + processed_action_log.delta_secs
bins = np.linspace(start_secs, end_secs, int(n_hours/cut_proportion) + 1)
bins = [int(x) for x in bins]
bin_labels = [str(pd.Timedelta(seconds=b))[7:] for b in bins[:-1]]
bin_names = pd.cut(processed_action_log["timestamp_relative_to_day_start"], bins, include_lowest=True, labels=bin_labels)
processed_action_log["bin"] = bin_names

for date_timestamp in processed_action_log.date.unique():
    date = str(date_timestamp)[:10]
    log = processed_action_log[processed_action_log.date == date_timestamp]
    activity_frequency_per_bin = {}
    for current_bin in bin_labels:
        activity_frequency = log[log.bin == current_bin].activity.value_counts(normalize=True)
        activity_frequency_per_bin[current_bin] = activity_frequency
        frequency_df = pd.DataFrame(activity_frequency_per_bin).fillna(0).transpose()

    print(1)
    for activity_name in log.activity.unique():
        plt.cla()
        ax = frequency_df[activity_name].plot(kind="bar", figsize=(10, 5), edgecolor="black")
        ax.set_xlabel("Time of day")
        ax.set_ylabel("Relative frequency")
        # ax.xaxis.tick_top()
        # ax.xaxis.set_label_position("top")
        ax.set_title(f"{activity_name}")
        plt.tight_layout()
        plt.savefig(f"out/time_of_day/{date}_frequency_{activity_name}.png")
        #plt.show()
    frequency_df.to_excel(f"out/time_of_day/{date}_frequencies.xlsx")
