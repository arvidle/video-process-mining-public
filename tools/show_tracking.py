from video_pm.tracking import TrackingResults, TrackingResultsPostProcessor
from video_pm.tracking.two_step import DetectionResults
from video_pm import Video
from video_pm.visualization import TrackingVideo, DetectionVideo
import os
import cv2
import numpy as np

FILENAME = "ch01_20211114142540"
VIDEO_DIR = "/home/arvid/Documents/Arbeit_Data/videos/ch01_20211114/converted/"
VIDEO_EXT = ".mp4"
TRACKING_DIR = "../data/tracking/processed"
TRACKING_EXT = ".npz"
DETECTIONS_DIR = "../data/detection"
DETECTIONS_EXT = ".npz"

TRACKING_FILENAME = os.path.join(TRACKING_DIR, FILENAME + TRACKING_EXT)
DETECTIONS_FILENAME = os.path.join(DETECTIONS_DIR, FILENAME + DETECTIONS_EXT)
VIDEO_FILENAME = os.path.join(VIDEO_DIR, FILENAME + VIDEO_EXT)

TRACKING_FILENAME = "../data/tracking/processed/tracking_20211113.npz"
VIDEO_FILENAME = "/home/arvid/Documents/Arbeit_Data/videos/ch01_20211113/converted/ch01_20211113.mp4"

video = Video(VIDEO_FILENAME)
tracking = TrackingResults.from_file(TRACKING_FILENAME)
print("Number of track ids: ", len(tracking.tracking["track_id"].unique()))
print("Processing tracking")

tracking_processed = tracking

#detections = DetectionResults.from_file(DETECTIONS_FILENAME)
#print("Filtering detections")
#detections.filter(0.6, 0.8, 0.05)

tracking_video = TrackingVideo(tracking_processed, video)
#detections_video = DetectionVideo(detections, video)

fps = 30
width = 854 # * 2
height = 480
output_size = (width, height)
out = cv2.VideoWriter("tracking.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, output_size)

#for frame1, frame2 in zip(detections_video.frames(), tracking_video.frames()):
#    frame = np.hstack((frame1, frame2))

for i, frame in enumerate(tracking_video.frames(400000)):
    out.write(frame)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

#for cnt, frame in enumerate(tracking_video.frames()):
#    if cnt % 100 == 0:
#        out.write(frame)

#out.release()
