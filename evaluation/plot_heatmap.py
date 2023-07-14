from video_pm.activity_recognition import ActionDetectionResults
from video_pm.activity_recognition.event_activity_abstraction import BaselineAbstractor
from functools import reduce
import matplotlib.pyplot as plt
import random

random.seed(42)

days = ["20211113", "20211114", "20211115", "20211116", "20211117"]
days = ["20211113"]

res_days = []

for day in days:
    res = ActionDetectionResults.from_file(f"../data/action_detection/loaded/{day}.pkl")
    res_days.append(res)
# Combine action logs of the same segments for each day to get a log for each segment
days_combined = reduce(lambda x, y: x.concat(y), res_days)

ba = BaselineAbstractor(10)

#abstract_activities = ba.process_action_log(days_combined)
abstract_activities = ba.run(days_combined).activity_log
abstract_activities["mid_x"] = ((abstract_activities.x1 + abstract_activities.x2) / 2 * (1920/854)).astype(int)
abstract_activities["mid_y"] = ((abstract_activities.y1 + abstract_activities.y2) / 2 * (1080/480)).astype(int)

gps = abstract_activities.groupby("activity")
select_columns = ["mid_x", "mid_y"]
lying = gps.get_group("lying")[select_columns]
feeding = gps.get_group("feeding")[select_columns]
defecating = gps.get_group("defecating")[select_columns]

marker_size = 5

im = plt.imread('video_screenshot.png')
#implot = plt.imshow(im)

fig, ax = plt.subplots()

ax.imshow(im)

ax.scatter(lying.mid_x, lying.mid_y, s=marker_size, marker="X", c="indigo", label="lying")
ax.scatter(feeding.mid_x, feeding.mid_y, s=marker_size, marker="s", c="aqua", label="feeding")
ax.scatter(defecating.mid_x, defecating.mid_y, s=marker_size, marker="^", c="red", label="defecating")
#ax.plot(lying.mid_x, lying.mid_y, 'kx', markersize=marker_size, label="lying", color="red")
#ax.plot(feeding.mid_x, feeding.mid_y, 'ks', markersize=marker_size, label="feeding", color="green")
#ax.plot(defecating.mid_x, defecating.mid_y, 'k^', markersize=marker_size, label="defecating", color="blue")
plt.axis("off")
plt.legend(loc="lower right", fontsize="large")
plt.tight_layout(pad=0)
plt.savefig("activity_positions.png", bbox_inches="tight", dpi=600)
