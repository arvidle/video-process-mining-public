import pandas as pd

def motMetricsEnhancedCalculator(gtSource, tSource):
    # import required packages
    import motmetrics as mm
    import numpy as np

    # load ground truth
    gt = np.loadtxt(gtSource, delimiter=',')

    # load tracking output
    t = np.loadtxt(tSource, delimiter=',')

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1

        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        gt_dets = gt[gt[:, 0] == frame, 1:6]  # select all detections in gt
        t_dets = t[t[:, 0] == frame, 1:6]  # select all detections in t

        C = mm.distances.iou_matrix(gt_dets[:, 1:], t_dets[:, 1:], \
                                    max_iou=0.5)  # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(gt_dets[:, 0].astype('int').tolist(), \
                   t_dets[:, 0].astype('int').tolist(), C)

    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                       'recall', 'precision', 'num_objects', \
                                       'mostly_tracked', 'partially_tracked', \
                                       'mostly_lost', 'num_false_positives', \
                                       'num_misses', 'num_switches', \
                                       'num_fragmentations', 'mota', 'motp' \
                                       ], \
                         name='acc')

    strsummary = mm.io.render_summary(
        summary,
        # formatters={'mota' : '{:.2%}'.format},
        namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
                 'precision': 'Prcn', 'num_objects': 'GT', \
                 'mostly_tracked': 'MT', 'partially_tracked': 'PT', \
                 'mostly_lost': 'ML', 'num_false_positives': 'FP', \
                 'num_misses': 'FN', 'num_switches': 'IDsw', \
                 'num_fragmentations': 'FM', 'mota': 'MOTA', 'motp': 'MOTP', \
                 }
    )
    print(strsummary)
    return summary


sequences = ["836-0107-0108",
             "836-0108-0109",
             "836-0109-0110",
             "836-0113-0114",
             "836-0114-0115",
             "836-0125-0126",
             "931-0015-0016",
             "931-0035-0036",
             "931-0036-0037",
             "946-0021-0022",
             "946-0022-0023",
             "946-0023-0024",
             "946-0024-0025",
             "954-0000-0001"]

gt_dir = "tracking_eval/ground_truth/"
inf_dir = "tracking_eval/inferred/"
ext = ".csv"
results = []
for tracking_sequence in sequences:
    gt = gt_dir + tracking_sequence + ext
    inf = inf_dir + tracking_sequence + ext
    print(tracking_sequence)
    metrics = motMetricsEnhancedCalculator(gt, inf)
    metrics["tracking_sequence"] = tracking_sequence
    results.append(metrics)

all_metrics = pd.concat(results)
all_metrics.to_excel("mot_metrics.xlsx")
