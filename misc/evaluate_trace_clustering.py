import pandas as pd
import pm4py
from video_pm.activity_recognition import ActionDetectionResults
from video_pm.activity_recognition.event_activity_abstraction import BaselineAbstractor
from video_pm.case_correlation import StartEndCorrelator
from functools import reduce
from sklearn.cluster import KMeans, DBSCAN, BisectingKMeans
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.visualization.petri_net import visualizer as pn_visualizer
import numpy as np
from scipy import stats

WINDOW_SIZE = 1
ROLLING = True
N_COMPONENTS = 0.99
N_CLUSTERS = 10
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
days_combined = [reduce(lambda x, y: x.concat(y), res_days)]


abstractor = BaselineAbstractor(window_size=WINDOW_SIZE, rolling=ROLLING)

segments_abstracted = [abstractor.run(action_log) for action_log in days_combined]

correlator = StartEndCorrelator(START_ACTIVITY,
                                END_ACTIVITY,
                                MIN_START_DUR,
                                MIN_END_DUR,
                                MIN_PERFORMANCE,
                                MAX_PERFORMANCE_CASE_ASSIGNMENT,
                                ADD_START_END_LABEL_EXTENSIONS,
                                ADD_ARTIFICIAL_START_END,
                                MIN_END_REPEATS)

segments_events_correlated = [correlator.run(segment) for segment in segments_abstracted]

segment_features = [log.get_features() for log in segments_events_correlated]

segment_logs = [log.to_pm4py() for log in segments_events_correlated]

runs = []

for i in range(10):
    for segment_log, case_features in zip(segment_logs, segment_features):
        segment_log = pm4py.filter_case_performance(segment_log, MIN_PERFORMANCE, MAX_PERFORMANCE)

        #pipeline = make_pipeline(PCA(N_COMPONENTS), StandardScaler(), KMeans(N_CLUSTERS))
        #data, feature_names = log_to_features.apply(segment_log, parameters={
        #    "enable_case_duration": True,
        #    "enable_times_from_first_occurrence": True,
        #    "enable_times_from_last_occurrence": True,
        #    "enable_direct_paths_times_last_occ": True,
        #    "enable_indirect_paths_times_last_occ": True
        #})
        pipeline = make_pipeline(StandardScaler(), PCA(n_components=N_COMPONENTS), KMeans(N_CLUSTERS))
        #pipeline = make_pipeline(MinMaxScaler(), PCA(n_components=N_COMPONENTS), KMeans(N_CLUSTERS))
        #pipeline = make_pipeline(MinMaxScaler(), TruncatedSVD(N_CLUSTERS), KMeans(N_CLUSTERS))
        #pipeline = make_pipeline(StandardScaler(), PCA(n_components=N_COMPONENTS), DBSCAN(eps=10, min_samples=5))
        #pipeline = make_pipeline(StandardScaler(), PCA(n_components=N_COMPONENTS), BisectingKMeans(n_clusters=N_CLUSTERS))
        """
        data2, feature_names2 = log_to_features.apply(segment_log, parameters={
            "str_ev_attr": ["concept:name"],
            "str_tr_attr": [],
            "num_ev_attr": [],
            "num_tr_attr": [],
            "str_evsucc_attr": ["concept:name"],
            "enable_case_duration": True,
            "enable_times_from_first_occurrence": True,
        })"""
        clusters = pipeline.fit_predict(case_features)
        #case_id_column = feature_names.index("event:CaseID")
        #case_ids = np.array(data)[:, case_id_column]
        case_ids = case_features.index.tolist()
        case_cluster_map = dict(zip(case_ids, clusters))

        if REMOVE_SMALL_CLUSTERS:
            # Remove very small clusters and reassign the corresponding traces to the other clusters
            cluster_ids, cluster_sizes = np.unique(clusters, return_counts=True)
            clusters_to_keep = [cluster_id for cluster_id, cluster_size in zip(cluster_ids, cluster_sizes) if cluster_size >= MIN_CLUSTER_SIZE]

            kmeans = KMeans(len(clusters_to_keep))
            pca = pipeline["pca"]
            kmeans.fit(np.zeros((len(clusters_to_keep), pca.n_components_)))
            kmeans.cluster_centers_ = pipeline["kmeans"].cluster_centers_[clusters_to_keep]

            new_pipeline = make_pipeline(pipeline["standardscaler"], pca, kmeans)

            clusters = new_pipeline.predict(case_features)
            case_cluster_map = dict(zip(case_ids, clusters))

        segment_log_df = pm4py.convert_to_dataframe(segment_log)
        segment_log_df["case:cluster"] = segment_log_df["CaseID"].map(case_cluster_map)

        cluster_gps = segment_log_df.groupby("case:cluster")

        clustered_log_df = pm4py.format_dataframe(segment_log_df)
        clustered_log = pm4py.convert_to_event_log(clustered_log_df)

        if i == 1:
            pm4py.write_xes(clustered_log, "clustered_log_test123.xes")

        clustered_sublogs = [pm4py.convert_to_event_log(sublog_df) for cluster, sublog_df in cluster_gps]

        all_cluster_stats = []

        for sublog in clustered_sublogs:
            #net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(sublog, noise_threshold=0.0)
            net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(sublog, dependency_threshold=0.6, and_threshold=1.0, loop_two_threshold=1.8)

            fitness = pm4py.fitness_token_based_replay(sublog, net, initial_marking, final_marking)["log_fitness"]
            precision = pm4py.precision_token_based_replay(sublog, net, initial_marking, final_marking)
            f1 = stats.hmean((fitness, precision))

            simplicity = simplicity_evaluator.apply(net)
            generalization = generalization_evaluator.apply(sublog, net, initial_marking, final_marking)

            cluster_stats = {
                "cluster": sublog[0].attributes["cluster"],
                "fitness": fitness,
                "precision": precision,
                "f1": f1,
                "simplicity": simplicity,
                "generalization": generalization,
                "cluster_size": len(sublog),
                "cluster_trace_lengths": [len(trace) for trace in sublog]
            }

            all_cluster_stats.append(cluster_stats)
            #if f1 > 0.0:
            #    print(id(net))
            #    print("F1: ", f1)
            #    print("Simplicity: ", simplicity)
            #    print("Generalization: ", generalization)
            #    parameters = {pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"}
            #    gviz = pn_visualizer.apply(net, initial_marking, final_marking, parameters=parameters,
            #                           variant=pn_visualizer.Variants.FREQUENCY, log=sublog)

            #    pn_visualizer.save(gviz, f"out/cluster_{cluster_stats['cluster']}.png")

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
print(runs_df.describe())