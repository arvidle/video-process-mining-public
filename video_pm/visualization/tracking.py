import pandas as pd

from video_pm.tracking import TrackingResults
from video_pm.tracking.two_step import DetectionResults
from video_pm import Video
from .base import VideoVisualizer
import cv2
from typing import Tuple
# Visualize tracking on video


def draw_box_on_image(image, bbox, label=None, color=(200, 100, 0), thickness=2):
    img = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness)
    if label:
        img = cv2.putText(img, label, (int(bbox[0]), int(bbox[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        #img = cv2.putText(img, label, (int(bbox[0]), int(bbox[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return img


def limit_box(width, height):
    limit_x = lambda x: 0 if x < 0 else (width if x >= width else x)
    limit_y = lambda y: 0 if y < 0 else (height if y >= height else y)

    return limit_x, limit_y


def limit_boxes(boxes, width, height):
    limit_x, limit_y = limit_box(width, height)
    limited_boxes: pd.DataFrame = boxes.copy()
    limited_boxes["x1"] = limited_boxes["x1"].apply(limit_x)
    limited_boxes["x2"] = limited_boxes["x2"].apply(limit_x)
    limited_boxes["y1"] = limited_boxes["y1"].apply(limit_y)
    limited_boxes["y2"] = limited_boxes["y2"].apply(limit_y)

    return limited_boxes


class TrackingVideo(VideoVisualizer):
    def __init__(self, tracking: TrackingResults, video: Video):
        """

        :param tracking:
        :param video:
        """
        self.tracking: TrackingResults = tracking
        self.video: Video = video
        self.tracking_by_frame = tracking.tracking.groupby("frame")

    def get_boxes(self, frame) -> Tuple[Tuple[int, int, int, int], str]:
        pass

    def frames(self, start_frame=None):
        video_capture = self.video.get_capture()
        if start_frame:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # Get video height and width
        width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        # For frame in video
        success, frame = video_capture.read()
        while success:
            frame_num = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            try:
                # Get boxes for frame
                boxes = self.tracking_by_frame.get_group(frame_num)
                # Limit boxes to video frame size
                limited_boxes = limit_boxes(boxes, width, height)
            except KeyError:
                limited_boxes = pd.DataFrame()
            # Draw boxes on frame (with ids)
            image = frame.copy()
            for _, box in limited_boxes.iterrows():
                image = draw_box_on_image(image, box[["x1", "y1", "x2", "y2"]], str(box["track_id"]))
            # Yield frame with boxes
            yield image
            success, frame = video_capture.read()


class DetectionVideo(VideoVisualizer):
    def __init__(self, detections: DetectionResults, video: Video):
        """

        :param detections:
        :param video:
        """
        self.detections: DetectionResults = detections
        self.video: Video = video
        self.detections_by_frame = detections.detections.groupby("frame")

    def frames(self, start_frame=None):
        video_capture = self.video.get_capture()
        if start_frame:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # Get video height and width
        width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        # For frame in video
        success, frame = video_capture.read()
        while success:
            frame_num = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            # Get boxes for frame
            try:
                boxes = self.detections_by_frame.get_group(frame_num)
                # Limit boxes to video frame size
                limited_boxes = limit_boxes(boxes, width, height)
                # Draw boxes on frame (with ids)
            except KeyError:
                limited_boxes = pd.DataFrame()
            image = frame.copy()
            for _, box in limited_boxes.iterrows():
                image = draw_box_on_image(image, box[["x1", "y1", "x2", "y2"]], f"{box['score']:.2f}")
            # Yield frame with boxes
            yield image
            success, frame = video_capture.read()
