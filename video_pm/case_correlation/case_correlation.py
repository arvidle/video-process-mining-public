import pandas as pd

from video_pm.activity_recognition import ActivityLog
from video_pm.case_correlation import EventLog
from collections import deque
import datetime


class EventCaseCorrelator:
    def __init__(self):
        pass

    def run(self, action_log: ActivityLog) -> EventLog:
        pass


class TrackletCorrelator(EventCaseCorrelator):
    def __init__(self):
        pass

    def run(self, action_log: ActivityLog) -> EventLog:
        return EventLog(action_log.activity_log)


class StartEndCorrelator(EventCaseCorrelator):
    """Correlate activity logs grouped by tracklets to (shorter) cases

    Split tracklet-based traces into multiple cases by applying multiple heuristics.
    Splitting is done on a per-trace basis.
    A new case is opened when a starting activity is observed.
    A starting activity qualifies as a starting activity if and only if it has the correct activity type and reaches the
    minimal start duration.
    In general, splits are done at the end activity.
    An activity qualifies as an end activity if and only if it has the correct activity type and reaches the minimal
    end duration.
    The minimal end event duration can be used to avoid splits at points of short mis-classification (noise).
    If the start activity and the end activity are the same, the end activity is also used to start the next case.
    Otherwise, the next starting activity is searched to start a new case from there.

    If the minimal case duration is not reached even after observing the required number of end events,
    the case is continued until the minimal case duration is fulfilled and
    the next end event or maximum duration is reached.

    Maximal case duration is used as a cutoff.
    Cases are closed when the maximal duration is reached, even when the end event is not reached.

    A certain number of repeats of the end activity before closing the case
    can be allowed via the min_end_repeats parameter.
    """
    def __init__(self,
                 start_activity: str,
                 end_activity: str,
                 min_start_dur: float,
                 min_end_dur: float,
                 min_dur: float,
                 max_dur: float,
                 add_start_end_label_extensions: bool = True,
                 add_artificial_start_end: bool = True,
                 min_end_repeats: int = 1):
        """Construct a new StartEndCorrelator instance.

        :param start_activity: Starting activity
        :param end_activity: Ending activity
        :param min_start_dur: Minimum duration of the starting activity
        :param min_end_dur: Minimum duration of the ending activity
        :param min_dur: Minimal trace duration
        :param max_dur: Maximal trace duration
        :param add_start_end_label_extensions: Whether to add label extensions to case start and end activities
        :param add_artificial_start_end: Whether to add artificial case start and end markers
        :param min_end_repeats: Minimum repeat of end activity to end the case
        """
        self.start_activity: str = start_activity
        self.end_activity: str = end_activity
        self.min_end_repeats: int = min_end_repeats
        self.min_start_dur: datetime.timedelta = datetime.timedelta(seconds=min_start_dur)
        self.min_end_dur: datetime.timedelta = datetime.timedelta(seconds=min_end_dur)
        self.min_dur: datetime.timedelta = datetime.timedelta(seconds=min_dur)
        self.max_dur: datetime.timedelta = datetime.timedelta(seconds=max_dur)
        self.add_start_end_label_extensions: bool = add_start_end_label_extensions
        self.add_artificial_start_end: bool = add_artificial_start_end

        assert min_end_repeats >= 1
        assert self.min_dur <= self.max_dur

    def split_trace(self, trace):
        """Split a trace into multiple cases with the method specified in the class documentation.

                Assumes the trace is sorted by timestamp.

                :param trace: Trace to split
                :return: List of separated case logs
                """
        assert trace.iloc[0]["activity"] == "start"
        assert trace.iloc[-1]["activity"] == "end"

        durations = trace["t"].shift(-1) - trace["t"]
        durations = durations.fillna(datetime.timedelta(0))

        trace = trace.copy()
        trace["duration"] = durations
        # Remove artificial start/end activities
        # No information is lost because we calculated the durations above
        trace = trace[1:-1]

        trace_queue = deque([row for _, row in trace.iterrows()])

        case_open = False
        close_case = False

        all_cases = []
        dropped_event_cnt = 0

        while trace_queue:
            current_event = trace_queue.popleft()
            current_activity = current_event["activity"]
            current_timestamp = current_event["t"]
            current_duration = current_event["duration"]
            if not case_open:
                if (current_activity == self.start_activity) and (current_duration >= self.min_start_dur):
                    case_open = True
                    case_repeat_count = -1
                    case_start_timestamp = current_timestamp + current_duration
                    current_case = []
                else:
                    dropped_event_cnt += 1

            if case_open:
                if (current_activity == self.end_activity) and (current_duration >= self.min_end_dur):
                    case_repeat_count += 1

                case_duration_pre = current_timestamp - case_start_timestamp
                case_duration_post = case_duration_pre + current_duration

                min_duration_fulfilled = case_duration_post >= self.min_dur
                max_duration_exceeded = case_duration_post >= self.max_dur
                repeat_amount_fulfilled = case_repeat_count >= self.min_end_repeats

                if max_duration_exceeded:
                    # In case the case maximum duration is exceeded, limit the last event length,
                    # so we fulfill the maximum case duration
                    current_event_mod = current_event.copy()
                    current_event_mod["duration"] = self.max_dur - case_duration_pre
                    close_case = True
                elif (repeat_amount_fulfilled and min_duration_fulfilled) or (current_activity == "end"):
                    close_case = True
                else:
                    # If the case is not ending, just add the event to the case
                    # (Please note: if the current event ends the trace, it is added after being processed below)
                    current_case.append(current_event)

                if close_case:
                    case_open = False
                    close_case = False
                    if self.start_activity == self.end_activity \
                            and trace_queue \
                            and trace_queue[0]["activity"] != self.start_activity \
                            and current_activity == self.start_activity:
                        # If we and on a new start activity, duplicate that to start the next trace from
                        new_event = current_event.copy()
                        if max_duration_exceeded:
                            # If the maximum case duration was exceeded, split the event causing this
                            new_event["t"] = current_event_mod["t"] + current_event_mod["duration"]
                            new_event["duration"] = current_event["duration"] - current_event_mod["duration"]
                            current_case.append(current_event_mod)
                        else:
                            new_event["t"] = current_event["t"] + datetime.timedelta(microseconds=1)
                            current_case.append(current_event)
                        trace_queue.appendleft(new_event)
                    else:
                        current_case.append(current_event)

                    all_cases.append(current_case)

        return all_cases, dropped_event_cnt

    def run(self, activity_log: ActivityLog) -> EventLog:
        traces = activity_log.activity_log.groupby("case_id")

        all_cases = []
        dropped_event_cnt = 0

        for _, trace in traces:
            trace_cases = self.split_trace(trace)
            all_cases += trace_cases[0]
            dropped_event_cnt += trace_cases[1]

        case_dfs = []

        print("Total number of events: ", len(activity_log.activity_log))
        print("Number of dropped events: ", dropped_event_cnt)
        print("Percentage of events retained: ", 1 - float(dropped_event_cnt)/len(activity_log.activity_log))

        for case_id, case in enumerate(all_cases):
            case = case.copy()

            if self.add_artificial_start_end:
                start_event = case[0].copy()
                start_event["activity"] = "start"
                start_event["t"] = start_event["t"] - datetime.timedelta(microseconds=1)
                start_event["duration"] = datetime.timedelta(microseconds=1)

                end_event = case[-1].copy()
                end_event["t"] = end_event["t"] + end_event["duration"]
                end_event["duration"] = datetime.timedelta(microseconds=1)
                end_event["activity"] = "end"

            if self.add_start_end_label_extensions:
                annotated_start_event = case[0].copy()
                annotated_start_event["activity"] = annotated_start_event["activity"] + "_start"
                case[0] = annotated_start_event

                annotated_end_event = case[-1].copy()
                annotated_end_event["activity"] = annotated_end_event["activity"] + "_end"
                case[-1] = annotated_end_event

            case_trace = pd.DataFrame([start_event] + case + [end_event]).sort_values("t")
            case_trace["case_id"] = case_id

            case_dfs.append(case_trace)

        event_log_df = pd.concat(case_dfs)

        return EventLog(event_log_df.sort_values("t"))
