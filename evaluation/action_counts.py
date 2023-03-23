from video_pm.activity_recognition import ActionDetectionResults
from video_pm.activity_recognition.event_activity_abstraction import BaselineAbstractor
from functools import reduce
import matplotlib.pyplot as plt

days = ["20211113", "20211114", "20211115", "20211116", "20211117"]

res_days = []

for day in days:
    res = ActionDetectionResults.from_file(f"../data/action_detection/loaded/{day}.pkl")
    res_days.append(res)

# Combine action logs of the same segments for each day to get a log for each segment
days_combined = reduce(lambda x, y: x.concat(y), res_days)

ba = BaselineAbstractor(20)

top1_events = ba.process_action_log(days_combined).activity

print(top1_events.value_counts(normalize=True, ascending=True))
ax = top1_events.value_counts(normalize=True, ascending=True).plot(kind="barh", figsize=(5, 2.5), logx=True, color="lightgrey", edgecolor="black")
ax.set_xlabel("Relative duration")
ax.set_ylabel("Activity")
#ax.xaxis.tick_top()
#ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.savefig("activity_counts.svg")
