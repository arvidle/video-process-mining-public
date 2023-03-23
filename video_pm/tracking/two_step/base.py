from ..base import Tracker
from video_pm import Video
import numpy as np
import pandas as pd
from ..base import TrackingResults
import functools

from typing import Type, Tuple, List
from typing_extensions import Self
from tqdm import tqdm


def box_area(box: Tuple[int, int, int, int]) -> int:
    """Calculate the area of a bounding box.

    :param box: Bounding box in XYXY format
    :return: Area of the bounding box
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def box_intersection_area(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> int:
    """Calculate the area of the intersection of two bounding boxes.

    :param box1: First bounding box in XYXY format
    :param box2: Second bounding box in XYXY format
    :return: Area of the intersection of the two bounding boxes
    """
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0
    else:
        return box_area((x1, y1, x2, y2))


def box_union_area(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> int:
    """Calculate the area of the intersection of two bounding boxes.

    :param box1: First bounding box in XYXY format
    :param box2: Second bounding box in XYXY format
    :return: Area of the union of the two bounding boxes
    """
    return box_area(box1) + box_area(box2) - box_intersection_area(box1, box2)


def box_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate the IOU (intersection over union) of two bounding boxes.

    :param box1: First bounding box in XYXY format
    :param box2: Second bounding box in XYXY format
    :return: Area of the union of the two bounding boxes
    """
    iou = box_intersection_area(box1, box2) / box_union_area(box1, box2)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


class DetectionResults:
    def __init__(self, detections):
        self.detections: pd.DataFrame = detections

    def __str__(self):
        if len(self.detections) > 0:
            return str(self.detections)
        else:
            return "Empty detection results"

    def filter(self, threshold: float, intersect_threshold: float, area_threshold: float) -> None:
        def box(row):
            return row["x1"], row["y1"], row["x2"], row["y2"]

        # Filter boxes under a specified confidence threshold
        print(f"Filtering boxes below the specified score threshold ({threshold}).")
        self.detections = self.detections[self.detections["score"] >= threshold]
        print(f"Calculating box areas.")
        self.detections["area"] = self.detections.apply(lambda r: box_area(box(r)), axis=1)
        max_area = self.detections["area"].max()
        drop_rows = set()
        print("Filtering boxes per frame.")
        for _, boxes in tqdm(self.detections.groupby("frame")):
            drop_rows_frame = set()

            for index1, row1 in boxes.iterrows():
                box1 = box(row1)
                area1 = box_area(box1)

                if area1 / float(max_area) < area_threshold:
                    drop_rows_frame.add(index1)
                    #print(f"Drop box {box1} because of low box area ({area1 / float(max_area):.2f} of max area).")

                for index2, row2 in boxes.loc[index1 + 1:].iterrows():
                    box2 = box(row2)
                    assert box1 != box2
                    intersect = box_intersection_area(box1, box2)
                    area2 = box_area(box2)
                    ioa1 = intersect / area1
                    ioa2 = intersect / area2

                    if ioa1 >= intersect_threshold and ioa2 >= intersect_threshold:
                        #print("Both boxes are over the threshold, choosing the bigger one.")
                        if area1 > area2:
                            drop_rows_frame.add(index2)
                        else:
                            drop_rows_frame.add(index1)
                    elif ioa1 >= intersect_threshold:
                        #print(f"Remove box {box1} because its area mostly intersects with {box2} area (IOA: {ioa1}).")
                        drop_rows_frame.add(index1)
                    elif ioa2 >= intersect_threshold:
                        #print(f"Remove box {box2} because its area mostly intersects with {box1} area (IOA: {ioa2}).")
                        drop_rows_frame.add(index2)

            excess_box_count = len(boxes) - 11 - len(drop_rows_frame)
            if excess_box_count > 0:
                add_drop = set(boxes.drop(drop_rows_frame)["score"].nsmallest(excess_box_count).index)
                drop_rows_frame = drop_rows_frame.union(add_drop)

        drop_rows = drop_rows.union(drop_rows_frame)

        self.detections = self.detections.drop(drop_rows)

    def to_file(self, filename: str) -> None:
        """Serialize the DetectionResults to a compressed file.

        :param filename: Name of the file to save the DetectionResults to
        :return: None
        """
        boxes = self.detections[["frame", "x1", "y1", "x2", "y2"]].astype(int)
        confs = self.detections[["score"]]
        boxes_np = boxes.to_numpy()
        confs_np = confs.to_numpy()

        np.savez_compressed(filename, boxes=boxes_np, confs=confs_np)

    @classmethod
    def from_file(cls: Type[Self], filename: str) -> Self:
        """Load DetectionResults from a file (saved with to_file)

        :param filename: Name of the file with the serialized DetectionResults
        :return: A DetectionResults instance containing the loaded detection results
        """
        with np.load(filename) as data:
            boxes = data["boxes"]
            confs = data["confs"]

        return cls.from_np(boxes, confs)

    @classmethod
    def from_np(cls: Type[Self], boxes: np.ndarray, confs: np.ndarray) -> Self:
        """Create a DetectionResults instance from detection data stored in numpy arrays.

        :param boxes: Numpy array of shape (5, n) containing a frame number and an XYXY bounding box per row
        :param confs: Numpy array of shape (n, 1) containing the confidence for each bounding box in the boxes array
        :return: A DetectionResults instance containing the tracking information found in the arrays
        """
        columns = ["frame", "x1", "y1", "x2", "y2"]
        detections = pd.DataFrame(boxes, columns=columns)
        detections["score"] = confs
        detections["det_class"] = "pig"

        return cls(detections)


def concat_detection_results(detection_results: List[DetectionResults]) -> DetectionResults:
    def reduce_fn(fst: pd.DataFrame, snd: pd.DataFrame) -> pd.DataFrame:
        max_frame_fst = fst["frame"].max()
        min_frame_snd = snd["frame"].min()
        # Offset the frame numbers of the second dataframe
        snd_offset = snd.copy()
        snd_offset["frame"] += max_frame_fst - min_frame_snd + 1
        return pd.concat([fst, snd_offset], ignore_index=True)

    detection_dfs = [detections.detections for detections in detection_results]
    return DetectionResults(functools.reduce(reduce_fn, detection_dfs))


class ObjectDetector:
    """Base class for an object detector used by TwoStepTracker."""
    def __init__(self):
        pass

    def run(self, video: Video = None) -> DetectionResults:
        """Run the object detection.

        Do not use this function directly, but use an implementation provided by a subclass of this class.
        This only defines the interfaces used by concrete implementations.

        :param video: Video to run the detection on
        :return: Object detection results as an instance of DetectionResults
        """
        pass


class DetectionBasedTracker:
    def __init__(self):
        pass

    def run(self):
        pass


class TwoStepTracker(Tracker):
    def __init__(self, detector: ObjectDetector, tracker: DetectionBasedTracker) -> object:
        self.detector = detector
        self.tracker = tracker

    def run(self, video: Video = None) -> TrackingResults:
        detections = self.detector.run(video)
        tracking_results = self.tracker.run(detections)
        return tracking_results


