from video_pm.activity_recognition import ActionDetectionResults
from video_pm.tracking import TrackingResults
from video_pm import Video
from video_pm.visualization import ActionDetectionVideo
import cv2
import pandas as pd
from video_pm.visualization.activity_recognition import ACTIONS
import pm4py
import datetime

WINDOW_SIZE = 10


def process_track(track):
    track_actions = track[ACTIONS]
    track_actions_s = track_actions.rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
    track_actions_m = track_actions_s.rolling(WINDOW_SIZE, center=True, min_periods=1).mean()

    actions = track_actions_m.idxmax(axis=1)

    return actions


def collapse_processed_track(track):
    action_series = track["action"]
    changes = action_series.shift(1, fill_value=action_series.iloc[0]) != action_series
    changes.iloc[0] = True
    #changes.iloc[-1] = True

    track_changepoints = track[changes]
    start_marker = track.iloc[0].copy()
    start_marker["action"] = "start"
    end_marker = track.iloc[-1].copy()
    end_marker["action"] = "end"

    return pd.concat((start_marker.to_frame().T, track_changepoints, end_marker.to_frame().T))


def segment_day(n, start, end):
    day_length = end - start
    segment_size = day_length / n
    start_segments = [start + segment_size * x for x in range(n)]
    end_segments = [start + segment_size * x for x in range(1, n + 1)]
    return list(zip(start_segments, end_segments))


#video_filename = "/home/arvid/Documents/Arbeit_Data/videos/ch01_20211114/converted/ch01_20211114061218.mp4"
days = ["20211113", "20211114", "20211115", "20211116", "20211117"]
#video = Video(video_filename)

res_days = []

for day in days:
    tracking = TrackingResults.from_file(f"../data/tracking/processed/tracking_{day}.npz")
    res = ActionDetectionResults.from_mmaction_pkl(f"../data/action_detection/mmaction_out/{day}.pkl",
                                                   tracking_results=tracking,
                                                   start_time=(datetime.datetime.strptime(day, "%Y%m%d") +
                                                              datetime.timedelta(hours=6)))
    res.to_file(f"../data/action_detection/loaded/{day}.pkl")


"""
action_detection = res.action_detection_results

print("Loaded results successfully")

gps = action_detection.groupby("track_id")

aligned_actions = []
for track_id, track in gps:
    aligned_actions.append(process_track(track))


processed_action_detection = action_detection.merge(pd.concat(aligned_actions).rename("action"), left_index=True, right_index=True)
res_p = ActionDetectionResults(processed_action_detection)

processed_gps = processed_action_detection.groupby("track_id")

activity_logs = []

for track_id, track in processed_gps:
    changepoints = collapse_processed_track(track)
    activity_logs.append(changepoints[["track_id", "secs", "action"]])

y = pd.concat(activity_logs, axis=0)
y = y.sort_values(["secs"])
y["t"] = y["secs"].map(lambda x: datetime.datetime.strptime(f"{day}", "%Y%m%d") + datetime.timedelta(hours=6, seconds=int(x)))
y = y.rename(columns={'track_id': 'CaseID', 't': 'Timestamp', 'action': 'Activity'})

pre_actions = []
post_actions = []

for track_id, track in y.groupby("CaseID"):
    pre_actions_track = track["Activity"].shift(1, fill_value="None")
    post_actions_track = track["Activity"].shift(-1, fill_value="None")
    pre_actions.append(pre_actions_track)
    post_actions.append(post_actions_track)

#y = y.merge(pd.concat(pre_actions).rename("pre_activity"), left_index=True, right_index=True)
#y = y.merge(pd.concat(post_actions).rename("post_activity"), left_index=True, right_index=True)
#y["Modified_Activity"] = y["Activity"] #y["pre_activity"] + "_" + y["Activity"]# + "_" + y["post_activity"]

#segments = segment_day(12, y["Timestamp"].min(), y["Timestamp"].max())

#for segment in segments:
#    event_log_df = pm4py.format_dataframe(y[(segment[0] <= y["Timestamp"]) & (y["Timestamp"] < segment[1])], case_id="CaseID", activity_key="Activity", timestamp_key="Timestamp")
#    event_log = pm4py.convert_to_event_log(event_log_df)
#    event_log = pm4py.filter_case_performance(event_log, 600, 10000000)
    #event_log_export_df = pm4py.convert_to_dataframe(event_log)
    #event_log_export_df.to_csv(f'exported_{segment[0]}.csv')

    pm4py.write_xes(event_log, f"exported_{segment[0].strftime('%H')}.xes")

#dfg, start_activities, end_activities = pm4py.discover_dfg(event_log)
#pm4py.save_vis_dfg(dfg, start_activities, end_activities, "dfg.svg")

#net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(event_log, dependency_threshold=0.5, and_threshold=0.65, loop_two_threshold=0.5)
#pm4py.save_vis_petri_net(net, initial_marking, final_marking, "petri_net.svg")

#out_video = ActionDetectionVideo(res_p, video)

#for frame in out_video.frames():
#    image = cv2.resize(frame, (1700, 1000))
#    cv2.imshow("frame", frame)
#    cv2.waitKey(0)
"""