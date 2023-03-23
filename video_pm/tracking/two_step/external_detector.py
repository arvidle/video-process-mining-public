"""External detector

Implementation of a Detector used to load external detection results from JSON files.
"""

from .base import ObjectDetector, DetectionResults
from video_pm import Video
import json
import numpy as np
import pandas as pd
import cv2


class ExternalDetector(ObjectDetector):
    def __init__(self, filename: str):
        self.filename = filename

    def run(self, video: Video = None) -> DetectionResults:
        def process_row(row):
            return [([row["frame"], box[0], box[1], box[2], box[3]], box[4]) for box in row["boxes"]]

        with open(self.filename, "r") as file:
            data = json.load(file)
        # Convert the data to the required input format

        all_boxes = []
        all_confs = []
        for row in data:
            processed_row = process_row(row)

            if processed_row:
                boxes, confs = zip(*processed_row)
                np_boxes = np.array(boxes).astype(int)
                np_confs = np.array(confs)
                all_boxes.append(np_boxes)
                all_confs.append(np_confs)

        all_boxes_np = np.concatenate(all_boxes)
        all_confs_np = np.concatenate(all_confs)

        return DetectionResults.from_np(all_boxes_np, all_confs_np)


class ExternalYOLODetector(ObjectDetector):
    """Load detections saved in a YOLO-style format.

    Expected format is a single text file with whitespace separate lines containing:
        frame class_id x y w h score
    """
    def __init__(self, filename: str):
        self.filename = filename

    def run(self, video: Video) -> DetectionResults:
        cap = video.get_capture()
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()

        data = pd.read_csv(self.filename, delimiter=" ", names=["frame", "class_id", "x", "y", "w", "h", "score"])
        detections = pd.DataFrame()
        detections["frame"] = data["frame"]
        detections["x1"] = (data["x"] - data["w"] / 2) * width
        detections["y1"] = (data["y"] - data["h"] / 2) * height
        detections["x2"] = (data["x"] + data["w"] / 2) * width
        detections["y2"] = (data["y"] + data["h"] / 2) * height
        detections["score"] = data["score"]
        detections["det_class"] = "pig"  # TODO: Make parameter out of this

        return DetectionResults(detections)
