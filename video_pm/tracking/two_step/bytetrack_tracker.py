from .base import DetectionBasedTracker, DetectionResults
from .. import TrackingResults
from bytetrack_realtime.byte_tracker import ByteTracker
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials
from tqdm import tqdm
import pickle
import sys

from typing import Type
from typing_extensions import Self


class ByteTrackTracker(DetectionBasedTracker):
    def __init__(self,
                 track_thresh: float = 0.8,
                 track_buffer: int = 507,
                 match_thresh: float = 0.86,
                 frame_rate: float = 18.75):
        self.tracker = ByteTracker(track_thresh=track_thresh,
                                   track_buffer=track_buffer,
                                   match_thresh=match_thresh,
                                   frame_rate=frame_rate)

    def run(self, detections: DetectionResults) -> TrackingResults:
        dets = detections.detections.copy()
        dets["w"] = dets["x2"] - dets["x1"]
        dets["h"] = dets["y2"] - dets["y1"]

        tracking_results = []

        reformat = lambda row: ([row["x1"], row["y1"], row["w"], row["h"]], row["score"], row["det_class"])

        for frame, boxes in tqdm(dets.groupby("frame"), leave=False, position=1):
            det_boxes = [reformat(box) for _, box in boxes[["x1", "y1", "w", "h", "score", "det_class"]].iterrows()]
            tracks = self.tracker.update(det_boxes)

            for track in tracks:
                track_dict = {}
                track_dict["frame"] = frame
                track_dict["track_id"] = track.track_id
                track_dict["x1"], track_dict["y1"], track_dict["x2"], track_dict["y2"] = track.ltrb.tolist()
                track_dict["score"] = track.score
                track_dict["det_class"] = track.det_class
                tracking_results.append(track_dict)

        return TrackingResults(pd.DataFrame(tracking_results))

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
            space=[hp.uniform('track_thresh', 0.2, 1.0), hp.uniform("track_buffer", 10, 1000),
                   hp.uniform("match_thresh", 0.2, 1.0)],
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )

        with open("trials.pkl", "wb") as file:
            pickle.dump(trials, file)
        print("Best tracker settings: ", best)
        return cls(**best)
