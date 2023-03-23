import pm4py
from video_pm.activity_recognition import ActionDetectionResults
from video_pm.activity_recognition.event_activity_abstraction import BaselineAbstractor
from video_pm.case_correlation import EventLog, TrackletCorrelator, StartEndCorrelator
from functools import reduce
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pm4py.algo.transformation.log_to_features import algorithm as log_to_features
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
import numpy as np
from scipy import stats

SEGMENT_N = 12
WINDOW_SIZE = 10
ROLLING = True
N_COMPONENTS = 6
N_CLUSTERS = 10
MIN_PERFORMANCE = 60 * 5
MAX_PERFORMANCE_CASE_ASSIGNMENT = 60 * 15
MAX_PERFORMANCE = 60 * 60 * 100

days = ["20211113", "20211114", "20211115", "20211116", "20211117"]

res_days = []

for day in days:
    res = ActionDetectionResults.from_file(f"../data/action_detection/loaded/{day}.pkl")
    res_days.append(res)


# Segment the action logs of all the days
segmented_days = [day.segment(SEGMENT_N) for day in res_days]

# Discard the segment timestamps for now
segments = [[x for _, _, x in day_segments] for day_segments in segmented_days]

# Combine action logs of the same segments for each day to get a log for each segment
segments_combined = [reduce(lambda x, y: x.concat(y), segment) for segment in zip(*segments)]

abstractor = BaselineAbstractor(window_size=WINDOW_SIZE, rolling=ROLLING)

segments_abstracted = [abstractor.run(action_log) for action_log in segments_combined]

correlator = TrackletCorrelator()
#correlator = StartEndCorrelator("lying", "lying", 20, MIN_PERFORMANCE, MAX_PERFORMANCE_CASE_ASSIGNMENT, 1)

segments_events_correlated = [correlator.run(segment) for segment in segments_abstracted]

segment_logs = [log.to_pm4py() for log in segments_events_correlated]

for segment_log in segment_logs:
    segment_log = pm4py.filter_case_performance(segment_log, MIN_PERFORMANCE, MAX_PERFORMANCE)

    pipeline = make_pipeline(PCA(N_COMPONENTS), StandardScaler(), KMeans(N_CLUSTERS))
    data, feature_names = log_to_features.apply(segment_log)
    clusters = pipeline.fit_predict(data)
    case_id_column = feature_names.index("event:CaseID")
    case_ids = np.array(data)[:, case_id_column]
    case_cluster_map = dict(zip(case_ids, clusters))

    segment_log_df = pm4py.convert_to_dataframe(segment_log)
    segment_log_df["cluster"] = segment_log_df["CaseID"].map(case_cluster_map)

    cluster_gps = segment_log_df.groupby("cluster")

    clustered_sublogs = [pm4py.convert_to_event_log(sublog_df) for cluster, sublog_df in cluster_gps]

    for sublog in clustered_sublogs:
       # net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(sublog, 0.2)
        net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(sublog, dependency_threshold=0.3, and_threshold=0.35, loop_two_threshold=0.4)

        fitness = pm4py.fitness_token_based_replay(sublog, net, initial_marking, final_marking)["log_fitness"]
        precision = pm4py.precision_token_based_replay(sublog, net, initial_marking, final_marking)
        f1 = stats.hmean((fitness, precision))

        simplicity = simplicity_evaluator.apply(net)
        generalization = generalization_evaluator.apply(sublog, net, initial_marking, final_marking)


        if f1 > 0.7:
            print("F1: ", f1)
            print("Simplicity: ", simplicity)
            print("Generalization: ", generalization)
            pm4py.save_vis_petri_net(net, initial_marking, final_marking, f"{id(net)}.svg")

print(1)

