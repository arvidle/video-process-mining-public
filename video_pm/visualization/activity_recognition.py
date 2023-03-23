# Visualize activities on video
# Visualize/log activity log

import cv2
import pandas as pd
from video_pm import Video
from video_pm.activity_recognition import ActionDetectionResults, ActivityLog
from typing import List, Tuple
from .base import VideoVisualizer

ACTIONS = ["lying", "sitting", "standing", "moving", "investigating", "feeding", "defecating", "playing", "other"]


def draw_box_on_image(image, bbox, labels: List[Tuple[str, float]] = None, extra_label: str = None, color=(200, 100, 0), thickness=2):
    img = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness)
    if labels:
        labels_filtered = labels[labels > 0.01]
        labels_n = labels_filtered.sort_values(ascending=False).to_dict()
        for i, label in enumerate(labels_n.items()):
            img = cv2.putText(img, f"{label[0]}: {label[1]:.3f}", (int(bbox[0]), int(bbox[1]) + 15 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if extra_label:
        img = cv2.putText(img, extra_label, (int(bbox[0]), int(bbox[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return img


class ActionDetectionVideo(VideoVisualizer):
    def __init__(self, action_detection: ActionDetectionResults, video: Video):
        """

        :param action_detection:
        :param video:
        """
        self.action_detection = action_detection.action_detection_results
        self.video: Video = video
        self.action_detection_by_frame = self.action_detection.groupby("frame")

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
                boxes = self.action_detection_by_frame.get_group(frame_num)
                image = frame.copy()
                for _, box in boxes.iterrows():
                    # Draw boxes on frame (with action labels)
                    image = draw_box_on_image(image, box[["x1", "y1", "x2", "y2"]], box[ACTIONS])
                # Yield frame with boxes
                yield image
            except KeyError:
                pass

            success, frame = video_capture.read()


class AbstractedActivitiesVideo(VideoVisualizer):
    def __init__(self, abstract_actions: ActivityLog, video: Video):
        """

        :param action_detection:
        :param video:
        """
        self.abstract_actions = abstract_actions.activity_log
        self.video: Video = video
        self.abstract_actions_by_frame = self.abstract_actions.groupby("frame")

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
                boxes = self.abstract_actions_by_frame.get_group(frame_num)
                image = frame.copy()
                for _, box in boxes.iterrows():
                    # Draw boxes on frame (with action labels)
                    image = draw_box_on_image(image, box[["x1", "y1", "x2", "y2"]], extra_label=str(box["activity"]))
                # Yield frame with boxes
                yield image
            except KeyError:
                pass

            success, frame = video_capture.read()
