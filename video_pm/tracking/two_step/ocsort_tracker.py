import numpy as np

from .base import DetectionBasedTracker, DetectionResults
from .. import TrackingResults
from OC_SORT.trackers.ocsort_tracker.ocsort import OCSort
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials
from tqdm import tqdm
import pickle

from typing import Type
from typing_extensions import Self

CLASS_PER_ID = {1: "pig"}
ID_PER_CLASS = {"pig": 1}


class OCSortTracker(DetectionBasedTracker):
    def __init__(self,
                 det_thresh: float = 0.8,
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 delta_t: int = 3,
                 inertia: float = 0.2):
        self.tracker = OCSort(det_thresh=det_thresh,
                              max_age=max_age,
                              min_hits=min_hits,
                              iou_threshold=iou_threshold,
                              delta_t=delta_t,
                              inertia=inertia)

    def run(self, detections: DetectionResults) -> TrackingResults:
        tracking_results = []

        for frame, boxes in tqdm(detections.detections.groupby("frame"), leave=False, position=1):
            """
            Box format: XYXY_ABS
            boxes: 2d numpy array with 1 box per row
            cates: numpy array of category ids (len(cates) == number of boxes)
            scores: numpy array of box confidence scores
            Convert detections to the required format"""
            det_boxes = boxes[["x1", "y1", "x2", "y2"]].to_numpy()
            cates = boxes["det_class"].map(lambda x: ID_PER_CLASS[x]).to_numpy()
            #cates = np.ones_like(boxes["det_class"].to_numpy())
            scores = boxes["score"].to_numpy()
            tracks = self.tracker.update_public(det_boxes, cates, scores)

            # OCSort outputs tracks in the following format:
            # x1, x2, y1, y2, track_id, class_id, frame_offset
            # The frame offset is normally 0, but somtimes the offset is negative.
            # Negative offsets denote lag frames (?) and are discarded for now.

            for track in tracks:
                track_dict = {}
                if track[6] < 0:
                    continue
                track_dict["frame"] = frame + int(track[6])
                track_dict["track_id"] = int(track[4])
                track_dict["x1"], track_dict["y1"], track_dict["x2"], track_dict["y2"] = track[0:4]
                # The OCSort public interface does not visualization confidence scores, so we fix it to 1.0
                track_dict["score"] = 1.0
                track_dict["det_class"] = CLASS_PER_ID[track[5]]
                tracking_results.append(track_dict)

        tracking_df = pd.DataFrame(tracking_results)
        if len(tracking_df) > 0:
            sorted_tracking_df = tracking_df.sort_values(by=["frame", "track_id"], axis=0)
        else:
            sorted_tracking_df = pd.DataFrame([])

        return TrackingResults(sorted_tracking_df)

    @classmethod
    def optimize(cls: Type[Self], detections: DetectionResults, max_evals: int = 200) -> Self:
        """Optimize the parameters of the tracker on a set of detections.

        :param detections:
        :return:
        """
        # Optimized with YOLOv7 detections on ch01_20211114061218.mp4
        # Best tracker settings:  {'match_thresh': 0.75, 'track_buffer': 392, 'track_thresh': 0.83}
        def objective(params):
            print(params)
            tracker = cls(*params)
            results = tracker.run(detections)
            avg_trace_length = results.tracking["track_id"].value_counts().mean()
            return -avg_trace_length

        trials = Trials()
        best = fmin(
            fn=objective,
            space=[hp.uniform('det_thresh', 0.1, 1.0),
                   hp.uniform("iou_threshold", 0.1, 1.0)],
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )

        print(trials)
        with open("trials.pkl", "wb") as file:
            pickle.dump(trials, file)
        print("Best tracker settings: ", best)
        return cls(**best)
