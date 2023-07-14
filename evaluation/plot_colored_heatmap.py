from video_pm.activity_recognition import ActionDetectionResults
from video_pm.activity_recognition.event_activity_abstraction import BaselineAbstractor
from functools import reduce
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap

random.seed(42)

days = ["20211113", "20211114", "20211115", "20211116", "20211117"]

res_days = []

for day in days:
    res = ActionDetectionResults.from_file(f"../data/action_detection/loaded/{day}.pkl")
    res_days.append(res)

# Combine action logs of the same segments for each day to get a log for each segment
days_combined = reduce(lambda x, y: x.concat(y), res_days)

ba = BaselineAbstractor(1)

abstract_activities = ba.process_action_log(days_combined)
abstract_activities["mid_x"] = ((abstract_activities.x1 + abstract_activities.x2) / 2 * (1920/854)).astype(int)
abstract_activities["mid_y"] = ((abstract_activities.y1 + abstract_activities.y2) / 2 * (1080/480)).astype(int)

gps = abstract_activities.groupby("activity")
select_columns = ["mid_x", "mid_y"]
lying = gps.get_group("lying")[select_columns]
feeding = gps.get_group("feeding")[select_columns]
defecating = gps.get_group("defecating")[select_columns]

marker_size = 10

im = plt.imread('video_screenshot.png')

x_bins = np.linspace(0, 1920, num=21)
y_bins = np.linspace(0, 1080, num=21)

# Note that x and y are swapped here because of how numpy handles histograms
# (See "Notes" under https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html)
heatmap_lying, _, _ = np.histogram2d(lying.mid_y, lying.mid_x, bins=(y_bins, x_bins))
heatmap_feeding, _, _ = np.histogram2d(feeding.mid_y, feeding.mid_x, bins=(y_bins, x_bins))
heatmap_defecating, _, _ = np.histogram2d(defecating.mid_y, defecating.mid_x, bins=(y_bins, x_bins))

heatmap_shape = heatmap_lying.shape
scaler = MinMaxScaler()

heatmap_lying = scaler.fit_transform(heatmap_lying.flatten().reshape(-1, 1)).reshape(heatmap_shape)
heatmap_feeding = scaler.fit_transform(heatmap_feeding.flatten().reshape(-1, 1)).reshape(heatmap_shape)
heatmap_defecating = scaler.fit_transform(heatmap_defecating.flatten().reshape(-1, 1)).reshape(heatmap_shape)

extent = (0, 1920, 0, 1080)

# Lying heatmap
plt.clf()
plt.cla()
plt.imshow(im, extent=extent)
plt.imshow(heatmap_lying, extent=extent, cmap="plasma", alpha=0.8, interpolation="gaussian")
plt.axis("off")
plt.colorbar(shrink=0.6)
plt.tight_layout(pad=0)
#plt.show()
plt.savefig("out/heatmaps/heatmap_lying.png", bbox_inches="tight", dpi=600)

# Feeding heatmap
plt.clf()
plt.cla()
plt.imshow(im, extent=extent)
plt.imshow(heatmap_feeding, extent=extent, cmap="plasma", alpha=0.8, interpolation="gaussian")
plt.axis("off")
plt.colorbar(shrink=0.6)
plt.tight_layout(pad=0)
#plt.show()
plt.savefig("out/heatmaps/heatmap_feeding.png", bbox_inches="tight", dpi=600)


# Defecating heatmap
plt.clf()
plt.cla()
plt.imshow(im, extent=extent)
plt.imshow(heatmap_defecating, extent=extent, cmap="plasma", alpha=0.8, interpolation="gaussian")
plt.axis("off")
plt.colorbar(shrink=0.6)
plt.tight_layout(pad=0)
#plt.show()
plt.savefig("out/heatmaps/heatmap_defecating.png", bbox_inches="tight", dpi=600)


# All three
cmap_red_dict = {"red": [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                 "green": [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
                 "blue": [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]}
cmap_green_dict = {"red": [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
                   "green": [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                   "blue": [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]}
cmap_blue_dict = {"red": [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
                  "green": [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
                  "blue": [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]}

red_cmap = LinearSegmentedColormap("red_linear", segmentdata=cmap_red_dict, N=256)
green_cmap = LinearSegmentedColormap("green_linear", segmentdata=cmap_green_dict, N=256)
blue_cmap = LinearSegmentedColormap("blue_linear", segmentdata=cmap_blue_dict, N=256)

plt.clf()
plt.cla()
plt.imshow(heatmap_lying, extent=extent, cmap=red_cmap, interpolation="gaussian")
plt.axis("off")
plt.colorbar(shrink=0.6)
plt.tight_layout(pad=0)
#plt.show()
plt.savefig("out/heatmaps/heatmap_lying_red.png", bbox_inches="tight", dpi=600)

plt.clf()
plt.cla()
plt.imshow(heatmap_feeding, extent=extent, cmap=green_cmap, interpolation="gaussian")
plt.axis("off")
plt.colorbar(shrink=0.6)
plt.tight_layout(pad=0)
#plt.show()
plt.savefig("out/heatmaps/heatmap_feeding_green.png", bbox_inches="tight", dpi=600)

plt.clf()
plt.cla()
plt.imshow(heatmap_defecating, extent=extent, cmap=blue_cmap, interpolation="gaussian")
plt.axis("off")
plt.colorbar(shrink=0.6)
plt.tight_layout(pad=0)
#plt.show()
plt.savefig("out/heatmaps/heatmap_defecating_blue.png", bbox_inches="tight", dpi=600)

plt.clf()
plt.cla()
plt.imshow(im, extent=extent)
plt.axis("off")
plt.colorbar(shrink=0.6)
plt.tight_layout(pad=0)
#plt.show()
plt.savefig("out/heatmaps/heatmap_background.png", bbox_inches="tight", dpi=600)