from video_pm.activity_recognition import ActionDetectionResults
from video_pm.activity_recognition.event_activity_abstraction import BaselineAbstractor
from functools import reduce
import matplotlib.pyplot as plt
import random

random.seed(42)

days = ["20211113", "20211114", "20211115", "20211116", "20211117"]

res_days = []

for day in days:
    res = ActionDetectionResults.from_file(f"../data/action_detection/loaded/{day}.pkl")
    res_days.append(res)

# Combine action logs of the same segments for each day to get a log for each segment
days_combined = reduce(lambda x, y: x.concat(y), res_days)

ba = BaselineAbstractor(20)

abstract_activities = ba.process_action_log(days_combined)
abstract_activities["mid_x"] = ((abstract_activities.x1 + abstract_activities.x2) / 2).astype(int)
abstract_activities["mid_y"] = ((abstract_activities.y1 + abstract_activities.y2) / 2).astype(int)

gps = abstract_activities.groupby("activity")
select_columns = ["mid_x", "mid_y"]
lying = gps.get_group("lying")[select_columns].sample(250)
feeding = gps.get_group("feeding")[select_columns].sample(100)
defecating = gps.get_group("defecating")[select_columns].sample(30)

marker_size = 10

im = plt.imread('video_screenshot.png')
#implot = plt.imshow(im)

fig, ax = plt.subplots()

ax.imshow(im)
ax.plot(lying.mid_x, lying.mid_y, 'kx', markersize=marker_size, label="lying")
ax.plot(feeding.mid_x, feeding.mid_y, 'ks', markersize=marker_size, label="feeding")
ax.plot(defecating.mid_x, defecating.mid_y, 'k^', markersize=marker_size, label="defecating")
plt.axis("off")
plt.legend(loc="lower right", fontsize="x-large")
plt.tight_layout(pad=0)
plt.savefig("activity_positions.svg", bbox_inches="tight")
