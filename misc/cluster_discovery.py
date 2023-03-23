from video_pm.activity_recognition import ActionDetectionResults
from video_pm.visualization.activity_recognition import ACTIONS
import pandas as pd


days = ["20211113", "20211114", "20211115", "20211116", "20211117"]

WINDOW_SIZE = 10


def process_track(track):
    track_actions = track[ACTIONS]
    track_actions_s = track_actions.rolling(WINDOW_SIZE, center=True, min_periods=1, step=10).sum()
    #track_actions_m = track_actions_s.rolling(WINDOW_SIZE, center=True, min_periods=1).mean()

    return track_actions_s


data = ActionDetectionResults.from_file("../data/action_detection/loaded/20211113.pkl")
action_detection = data.action_detection_results

gps = action_detection.groupby("track_id")

aligned_actions = []
for track_id, track in gps:
    aligned_actions.append(process_track(track))

actions_df = pd.concat(aligned_actions)

actions = actions_df.idxmax(axis=1)

action_detection_processed = action_detection.copy()
action_detection_processed["action"] = actions

breakpoint()
