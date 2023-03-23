import pandas as pd
import pm4py
from functools import reduce
import numpy as np
from tqdm import tqdm
import pickle

ACTIONS = ["lying", "sitting", "standing", "moving", "investigating", "feeding", "defecating", "playing", "other"]


# Taken from https://folk.idi.ntnu.no/mlh/hetland_org/coding/python/levenshtein.py (CC0)
def levenshtein(a, b):
    """Calculates the Levenshtein distance between a and b."""
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


class EventLog:
    """
    Represents a process mining compatible event log.
    Provides an interface to convert to a pm4py compatible event log
    """
    def __init__(self, event_log: pd.DataFrame):
        self.event_log: pd.DataFrame = event_log

    def to_pm4py(self) -> pm4py.objects.log.obj.EventLog:
        """Convert the event log to a pm4py compatible EventLog

        :return:
        """
        event_log_renamed = self.event_log.rename(columns={'case_id': 'CaseID', 't': 'Timestamp', 'activity': 'Activity'})
        event_log_df = pm4py.format_dataframe(event_log_renamed, case_id="CaseID", activity_key="Activity", timestamp_key="Timestamp")

        return pm4py.convert_to_event_log(event_log_df)

    def to_file(self, filename):
        """Write the action log to a file using pickle

        :param filename: Filename to write the object to
        :return:
        """
        with open(filename, "wb") as file:
            pickle.dump(self.event_log, file)

    @classmethod
    def from_file(cls, filename):
        """Load an action detection log from a pickle file

        :param filename: Filename to load the object from
        :return: ActionDetectionResults instance loaded from the specified pickle file
        """
        with open(filename, "rb") as file:
            event_log = pickle.load(file)

        return cls(event_log)

    def get_case_features(self, case):
        def strip_annotations(label):
            # If the real start/end events are annotated, remove those annotations for feature calculation
            if label[-4:] == "_end":
                label = label[:-4]
            if label[-6:] == "_start":
                label = label[:-6]

            return label

        def series_to_feature_series(ser, prefix):
            feature_ser = ser.copy()
            feature_ser.index = prefix + feature_ser.index

            return feature_ser

        def value_to_feature_series(value, name):
            feature_series = pd.Series({name: value})

            return feature_series

        def get_eventually_follows_relationships(trace: pd.Series):
            all_rels = []
            for i in range(1, len(trace)):
                relationships = trace.shift(i) + "~>" + trace
                all_rels.append(relationships.dropna())

            return pd.concat(all_rels, ignore_index=True)

        case = case.copy()
        case["activity"] = case["activity"].apply(strip_annotations)

        directly_follows_relationships = case["activity"].shift(1) + "->" + case["activity"]
        directly_follows_relationships = directly_follows_relationships.iloc[1:]

        eventually_follows_relationships = get_eventually_follows_relationships(case["activity"])

        activity_counts = case["activity"].value_counts()
        dfr_counts = directly_follows_relationships.value_counts()
        efr_counts = eventually_follows_relationships.value_counts()
        activity_durations = case.groupby("activity")["duration"].sum().apply(lambda x: x.total_seconds())
        case_duration = (case.iloc[-1]["t"] - case.iloc[0]["t"]).total_seconds()
        activity_present = activity_counts.copy()
        activity_present[:] = 1
        dfr_present = dfr_counts.copy()
        dfr_present[:] = 1
        efr_present = efr_counts.copy()
        efr_present[:] = 1

        activity_counts_series = series_to_feature_series(activity_counts, "activity_count@")
        dfr_counts_series = series_to_feature_series(dfr_counts, "dfr_count@")
        efr_counts_series = series_to_feature_series(efr_counts, "efr_count@")
        activity_durations_series = series_to_feature_series(activity_durations, "activity_duration@")
        case_duration_series = value_to_feature_series(case_duration, "case_duration")
        activity_present_series = series_to_feature_series(activity_present, "activity_present@")
        dfr_present_series = series_to_feature_series(dfr_present, "dfr_present@")
        efr_present_series = series_to_feature_series(efr_present, "efr_present@")

        #features = [activity_counts_series, activity_durations_series, case_duration_series, dfr_present_series]#, efr_counts_series, efr_present_series]
        # features = [dfr_counts_series]
        features = [activity_present_series]

        return pd.concat(features)

    def get_features(self):
        """Calculate case-based features

        :return:
        """
        cases = self.event_log.groupby("case_id")
        case_features = dict()
        for case_id, case in cases:
            case_features[case_id] = self.get_case_features(case)

        all_cols = [case.index for case in case_features.values()]
        all_columns = reduce(lambda x, y: x.union(y), all_cols)

        feature_df = pd.DataFrame([], index=pd.Index(case_features.keys(), copy=True, name="case_id"), columns=all_columns)
        for case_id, features in case_features.items():
            for col, val in zip(features.index, features):
                feature_df.loc[case_id][col] = val

        return feature_df.fillna(0)

    def calc_edit_distance_matrix(self, norm=True):
        cases = self.event_log.groupby("case_id")
        case_ids = cases.groups.keys()
        # Map case ids to matrix indices
        index_case_id = dict(zip(range(len(case_ids)), case_ids))
        case_id_index = dict(zip(case_ids, range(len(case_ids))))

        distance_matrix = np.zeros((len(cases), len(cases)))

        traces = {i: cases.get_group(index_case_id[i])["activity"].to_list() for i in range(len(cases))}

        for i in tqdm(range(len(cases))):
            for j in range(i, len(cases)):
                distance = levenshtein(traces[i], traces[j])
                if norm:
                    distance = distance / max(len(traces[i]), len(traces[j]))
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        return distance_matrix, case_id_index

    def get_distance_matrix(self, method):
        if method == "edit_distance":
            return self.calc_edit_distance_matrix()
        else:
            raise ValueError("Method not supported")
