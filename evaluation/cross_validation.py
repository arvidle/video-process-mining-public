import pandas as pd
import pm4py
from video_pm.activity_recognition import ActionDetectionResults
from video_pm.activity_recognition.event_activity_abstraction import BaselineAbstractor
from video_pm.case_correlation import StartEndCorrelator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, RepeatedKFold
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
import numpy as np
from scipy import stats
from functools import reduce

WINDOW_SIZE = 20
ROLLING = True
N_COMPONENTS = 0.99
N_CLUSTERS = 2
MIN_CLUSTER_SIZE = 10
REMOVE_SMALL_CLUSTERS = False

START_ACTIVITY = "lying"
END_ACTIVITY = "lying"
MIN_START_DUR = 5
MIN_END_DUR = 5
MIN_PERFORMANCE = 0
MAX_PERFORMANCE_CASE_ASSIGNMENT = 60 * 1000000
ADD_START_END_LABEL_EXTENSIONS = True
ADD_ARTIFICIAL_START_END = True
MIN_END_REPEATS = 1

MAX_PERFORMANCE = 60 * 60 * 100

days = ["20211113", "20211114", "20211115", "20211116", "20211117"]

res_days = []

for day in days:
    res = ActionDetectionResults.from_file(f"../data/action_detection/loaded/{day}.pkl")
    res_days.append(res)

# Combine action logs of the same segments for each day to get a log for each segment
action_log = reduce(lambda x, y: x.concat(y), res_days)


abstractor = BaselineAbstractor(window_size=WINDOW_SIZE, rolling=ROLLING)

abstracted_action_log = abstractor.run(action_log)

correlator = StartEndCorrelator(START_ACTIVITY,
                                END_ACTIVITY,
                                MIN_START_DUR,
                                MIN_END_DUR,
                                MIN_PERFORMANCE,
                                MAX_PERFORMANCE_CASE_ASSIGNMENT,
                                ADD_START_END_LABEL_EXTENSIONS,
                                ADD_ARTIFICIAL_START_END,
                                MIN_END_REPEATS)

events_correlated = correlator.run(abstracted_action_log)

case_features = events_correlated.get_features()

n_clusters_runs = dict()

for N_CLUSTERS in range(1, 21, 1):
    # Split the log for KFold cross-validation (but keep cases whole)
    splits = RepeatedKFold(n_splits=10, n_repeats=1, random_state=2).split(case_features)
    #splits = KFold(n_splits=5, shuffle=True, random_state=2).split(case_features)

    # We use KFold from sklearn and apply clustering, discovery and conformance evaluation to each split
    runs = []

    for train, test in splits:
        train = case_features.iloc[train]
        test = case_features.iloc[test]

        pipeline = make_pipeline(StandardScaler(), PCA(n_components=N_COMPONENTS), KMeans(N_CLUSTERS))
        clusters = pipeline.fit_predict(train)
        case_ids = train.index.tolist()
        case_cluster_map_train = dict(zip(case_ids, clusters))
        clusters_test = pipeline.predict(test)
        case_ids_test = test.index.tolist()
        case_cluster_map_test = dict(zip(case_ids_test, clusters_test))

        train_log_df = events_correlated.event_log[events_correlated.event_log["case_id"].isin(case_ids)].copy()
        test_log_df = events_correlated.event_log[events_correlated.event_log["case_id"].isin(case_ids_test)].copy()
        train_log = pm4py.format_dataframe(train_log_df, case_id="case_id", activity_key="activity", timestamp_key="t")
        test_log = pm4py.format_dataframe(test_log_df, case_id="case_id", activity_key="activity", timestamp_key="t")

        train_log["case:cluster"] = train_log["case_id"].map(case_cluster_map_train)
        test_log["case:cluster"] = test_log["case_id"].map(case_cluster_map_test)

        cluster_gps_train = train_log.groupby("case:cluster")
        cluster_gps_test = test_log.groupby("case:cluster")

        clustered_sublogs_train = [sublog_df.copy() for cluster, sublog_df in cluster_gps_train]
        clustered_sublogs_test = [sublog_df.copy() for cluster, sublog_df in cluster_gps_test]

        all_cluster_stats = []

        for sublog, test_log in zip(clustered_sublogs_train, clustered_sublogs_test):
            #net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(sublog, noise_threshold=0.15)
            #net, initial_marking, final_marking = pm4py.discover_petri_net_alpha(sublog)
            net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(sublog, dependency_threshold=0.8, and_threshold=0.8, loop_two_threshold=1)
            #net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(sublog)
            #pm4py.view_petri_net(net, initial_marking, final_marking)
            fitness = pm4py.fitness_token_based_replay(test_log, net, initial_marking, final_marking)["log_fitness"]
            precision = pm4py.precision_token_based_replay(test_log, net, initial_marking, final_marking)
            f1 = stats.hmean((fitness, precision))

            simplicity = simplicity_evaluator.apply(net)
            generalization = generalization_evaluator.apply(test_log, net, initial_marking, final_marking)

            cluster_stats = {
                "cluster": sublog.iloc[0]["case:cluster"],
                "fitness": fitness,
                "precision": precision,
                "f1": f1,
                "simplicity": simplicity,
                "generalization": generalization,
                "cluster_size": len(sublog),
                "cluster_trace_lengths": [len(trace) for trace in sublog]
            }

            all_cluster_stats.append(cluster_stats)

        cluster_stats_df = pd.DataFrame(all_cluster_stats)
        mean_f1 = stats.hmean(cluster_stats_df["f1"], weights=cluster_stats_df["cluster_size"])
        mean_precision = stats.hmean(cluster_stats_df["precision"], weights=cluster_stats_df["cluster_size"])
        mean_fitness = stats.hmean(cluster_stats_df["fitness"], weights=cluster_stats_df["cluster_size"])
        mean_simplicity = stats.hmean(cluster_stats_df["simplicity"], weights=cluster_stats_df["cluster_size"])
        mean_generalization = stats.hmean(cluster_stats_df["generalization"], weights=cluster_stats_df["cluster_size"])

        #pm4py.write_xes(clustered_log, "out/clustered_log.xes")

        run_stats = [mean_f1, mean_precision, mean_fitness, mean_simplicity, mean_generalization]
        runs.append(run_stats)
        print("Mean f1: ", mean_f1)
        print("Mean precision: ", mean_precision)
        print("Mean fitness: ", mean_fitness)
        print("Mean simplicity: ", mean_simplicity)
        print("Mean generalization: ", mean_generalization)
        print("Cluster sizes: ", cluster_stats_df["cluster_size"])

    print(runs)
    runs_df = pd.DataFrame(runs, columns=["f1", "precision", "fitness", "simplicity", "generalization"])
    n_clusters_runs[N_CLUSTERS] = runs_df
    n_clusters_runs[N_CLUSTERS]["n_clusters"] = N_CLUSTERS


clusters_runs = pd.concat(n_clusters_runs.values())
clusters_runs.to_csv("clusters_runs.csv", index=False)
print(1)