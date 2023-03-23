from video_pm.tracking import TrackingResults, TrackingResultsPostProcessor
from video_pm.tracking.two_step import DetectionResults
from video_pm import Video
from video_pm.visualization import TrackingVideo, DetectionVideo
import os
import cv2
import numpy as np

FILENAME = "ch01_20211117"
#VIDEO_DIR = "../data/videos"
#VIDEO_EXT = ".mp4"
TRACKING_DIR = "../data/tracking/bytetrack"
TRACKING_EXT = ".npz"
OUT_DIR = "../data/tracking/processed"
#DETECTIONS_DIR = "../data/detection"
#DETECTIONS_EXT = ".npz"

TRACKING_FILENAME = os.path.join(TRACKING_DIR, FILENAME + TRACKING_EXT)
#DETECTIONS_FILENAME = os.path.join(DETECTIONS_DIR, FILENAME + DETECTIONS_EXT)
#VIDEO_FILENAME = os.path.join(VIDEO_DIR, FILENAME + VIDEO_EXT)

#video = Video(VIDEO_FILENAME)
tracking = TrackingResults.from_file(TRACKING_FILENAME)
print("Processing tracking")
processor = TrackingResultsPostProcessor(tracking, moving_average_window=1)
tracking_processed = processor.run()

#cap = video.get_capture()
#width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#cap.release()

tracking_processed.to_file(os.path.join(OUT_DIR, "tracking_20211117.npz"))
