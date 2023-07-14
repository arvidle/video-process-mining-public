from video_pm.activity_recognition import ActionDetectionResults, ActivityLog
import pandas as pd


class EventActivityAbstractor:
    def __init__(self):
        pass

    def run(self, action_log: ActionDetectionResults) -> ActivityLog:
        pass


class BaselineAbstractor:
    def __init__(self, window_size: int = 10, rolling: bool = True):
        self.window_size = window_size
        self.rolling = rolling
        self.actions = ["lying", "sitting", "standing", "moving", "investigating", "feeding", "defecating", "playing", "other"]

    def process_trace(self, trace):
        trace_actions = trace[self.actions]
        if self.rolling:
            trace_actions_s = trace_actions.rolling(self.window_size, center=True, min_periods=1).sum()
            trace_actions_m = trace_actions_s.rolling(self.window_size, center=True, min_periods=1).mean()
        else:
            trace_actions_s = trace_actions.rolling(self.window_size, center=True, min_periods=1, step=self.window_size).sum()
            trace_actions_m = trace_actions_s.rolling(self.window_size, center=True, min_periods=1, step=self.window_size).mean()

        actions = trace_actions_m.idxmax(axis=1)

        return actions

    @staticmethod
    def collapse_processed_trace(trace):
        action_series = trace["activity"]
        changes = action_series.shift(1, fill_value=action_series.iloc[0]) != action_series
        # Always keep the first entry
        changes.iloc[0] = True
        # Keep all succeeding entries where the activity changes
        trace_changepoints = trace[changes]

        # Add start and end activities to all traces to conserve actual trace lengths
        start_marker = trace.iloc[0].copy()
        start_marker["activity"] = "start"
        end_marker = trace.iloc[-1].copy()
        end_marker["activity"] = "end"

        return pd.concat((start_marker.to_frame().T, trace_changepoints, end_marker.to_frame().T))

    def process_action_log(self, action_log: ActionDetectionResults) -> pd.DataFrame:
        gps = action_log.action_detection_results.groupby("case_id")
        aligned_activities = []

        for case_id, trace in gps:
            trace_actions = self.process_trace(trace)
            aligned_activities.append(trace_actions)

        processed_action_log = action_log.action_detection_results.merge(
            pd.concat(aligned_activities).rename("activity"),
            left_index=True, right_index=True)

        return processed_action_log

    def run(self, action_log: ActionDetectionResults) -> ActivityLog:
        processed_action_log = self.process_action_log(action_log)

        processed_gps = processed_action_log.groupby("case_id")
        activity_logs = []

        for case_id, trace in processed_gps:
            changepoints = self.collapse_processed_trace(trace)
            # Important: If we want to keep information to correlate the event log back to the source videos,
            # more columns need to be kept here.
            activity_logs.append(changepoints[["case_id", "t", "activity", "x1", "y1", "x2", "y2"]])

        event_log_df = pd.concat(activity_logs, axis=0)
        event_log_df["case_id"] = event_log_df["case_id"].astype(int)

        return ActivityLog(event_log_df)
