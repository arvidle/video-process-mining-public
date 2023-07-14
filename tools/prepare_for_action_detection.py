from video_pm.tracking import TrackingResults, TrackingResultsPostProcessor
from video_pm.tracking.two_step import DetectionResults
from video_pm import Video
from video_pm.visualization import TrackingVideo, DetectionVideo
import os
import cv2
import numpy as np

FILENAMES = ["836-0107-0108",
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
         "954-0000-0001",]
#VIDEO_DIR = "../data/videos"
#VIDEO_EXT = ".mp4"
TRACKING_DIR = "../data/tracking/bytetrack"
TRACKING_EXT = ".npz"
OUT_DIR = "../data/tracking/processed"
#DETECTIONS_DIR = "../data/detection"
#DETECTIONS_EXT = ".npz"
for FILENAME in FILENAMES:
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

    tracking_processed.to_file(os.path.join(OUT_DIR, FILENAME + TRACKING_EXT))
